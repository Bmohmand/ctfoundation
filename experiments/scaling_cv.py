"""
Unified decoder evaluation on Google CT-FM embeddings (ICH + SinoCT).

For each dataset:
  1. Load + de-duplicate embeddings (cached to .npz after first parse).
  2. Hold out ONE fixed, stratified 20% test set (never used for training,
     tuning, early stopping or threshold selection).
  3. Order the remaining development pool once with a fixed seed, stratified so
     every prefix keeps the class balance. Training subset r% = first r% of that
     order, so subsets are NESTED (every scan in the 5% subset is also in 10%,
     20%, ...) and identical across models.
  4. For every subset size and every model:
       - 3-, 5- and 10-fold stratified CV inside the subset
       - refit on the whole subset, evaluate on the fixed test set
         (threshold = mean of the Youden-J thresholds from the 3/5/10-fold
          out-of-fold predictions, the same rule as the ASFNR slides)
       - bootstrap 95% CIs on the test metrics
  5. Append results to results/<dataset>/<model>.jsonl (resumable) and save
     test-set scores for ROC plots.

Usage:
  python experiments/scaling_cv.py --dataset ich  --models LR SVM RF XGB MLP5 MLP7
  python experiments/scaling_cv.py --dataset sino --models LR SVM RF XGB MLP3 MLP5
  (add --ratios 0.01 0.02 ... to restrict, --device cuda on a GPU box)
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
FOLDS = [3, 5, 10]
ICH_LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


# ----------------------------------------------------------------------------- data
def _parse(df):
    return np.stack([np.asarray(json.loads(s), dtype=np.float32) for s in df["embedding"]])


def load_dataset(name):
    cache = os.path.join(ROOT, "cache", f"{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return d["X"], d["Y"], list(d["labels"]), dict(json.loads(str(d["info"])))

    info = {}
    if name == "ich":
        df = pd.concat([pd.read_csv(os.path.join(ROOT, "Rsna_ICH_Hemm.csv")),
                        pd.read_csv(os.path.join(ROOT, "Rsna_ICH_Control.csv"))], ignore_index=True)
        info["rows_raw"] = len(df)
        # exact-duplicate embeddings: keep one copy if labels agree, drop all if they conflict
        df["h"] = df["embedding"].map(lambda s: hashlib.md5(s.encode()).hexdigest())
        conflict = df.groupby("h")[ICH_LABELS].nunique().max(axis=1) > 1
        info["dup_embedding_rows_dropped"] = int(df.duplicated("h").sum())
        info["conflicting_dup_groups_dropped"] = int(conflict.sum())
        df = df[~df["h"].isin(conflict[conflict].index)].drop_duplicates("h")
        labels = ICH_LABELS
        Y = df[labels].values.astype(np.float32)
    elif name == "sino":
        df = pd.read_csv(os.path.join(ROOT, "Sino_CT.csv"))
        info["rows_raw"] = len(df)
        info["dup_url_rows_dropped"] = int(df.duplicated("url").sum())
        df = df.drop_duplicates("url")
        labels = ["label"]
        # same mapping as NewSinoCTModels.py: '1,0' -> 1, '0,1' -> 0
        Y = (df["label"] == "1,0").values.astype(np.float32)[:, None]
    else:
        raise ValueError(name)

    X = _parse(df)
    info["rows_used"] = len(df)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".tmp.npz"  # write-then-rename so an interrupted first run can't leave a corrupt cache
    np.savez(tmp, X=X, Y=Y, labels=np.array(labels), info=json.dumps(info))
    os.replace(tmp, cache)
    return X, Y, labels, info


def nested_order(strat, seed=SEED):
    """Stratified permutation: each class is shuffled and the classes are interleaved
    by fractional rank, so any prefix has ~the pool's class balance."""
    rng = np.random.default_rng(seed)
    key = np.empty(len(strat))
    for c in np.unique(strat):
        idx = np.flatnonzero(strat == c)
        rng.shuffle(idx)
        key[idx] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    return np.argsort(key, kind="stable")


def make_splits(Y):
    strat = Y[:, 0].astype(int)  # 'any' for ICH, label for SinoCT
    idx = np.arange(len(Y))
    dev, test = train_test_split(idx, test_size=0.2, stratify=strat, random_state=SEED)
    dev = dev[nested_order(strat[dev])]
    return dev, test


# ----------------------------------------------------------------------------- models
class TorchMLP:
    """Binary/multilabel MLP with BCE (+pos_weight), early stopping on an inner 10%
    validation split taken from the TRAINING data only."""

    def __init__(self, hidden, dropout, lr, residual_first=False, pos_weight=True,
                 plateau=False, max_epochs=50, patience=5, batch_size=32, device="cpu"):
        self.hidden, self.dropout, self.lr = hidden, dropout, lr
        self.residual_first, self.pos_weight, self.plateau = residual_first, pos_weight, plateau
        self.max_epochs, self.patience, self.batch_size, self.device = max_epochs, patience, batch_size, device

    def _build(self, d_in, d_out):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                blocks, prev = [], d_in
                for h, p in zip(self.hidden, self.dropout):
                    blocks.append(nn.Sequential(nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)))
                    prev = h
                s.blocks = nn.ModuleList(blocks)
                s.res = nn.Linear(d_in, self.hidden[0]) if self.residual_first else None
                s.out = nn.Linear(prev, d_out)

            def forward(s, x):
                h = s.blocks[0](x)
                if s.res is not None:
                    h = h + s.res(x)
                for b in s.blocks[1:]:
                    h = b(h)
                return s.out(h)

        return Net()

    def fit(self, X, Y):
        import torch
        import torch.nn as nn
        torch.manual_seed(SEED)
        Y = Y.reshape(len(Y), -1)
        strat = Y[:, 0].astype(int)
        can_strat = np.bincount(strat).min() >= 2
        tr, va = train_test_split(np.arange(len(X)), test_size=0.1, random_state=SEED,
                                  stratify=strat if can_strat else None)
        self.scaler = StandardScaler().fit(X[tr])
        dev = self.device
        Xt = torch.tensor(self.scaler.transform(X[tr]), dtype=torch.float32, device=dev)
        Yt = torch.tensor(Y[tr], dtype=torch.float32, device=dev)
        Xv = torch.tensor(self.scaler.transform(X[va]), dtype=torch.float32, device=dev)
        Yv = torch.tensor(Y[va], dtype=torch.float32, device=dev)

        self.net = self._build(X.shape[1], Y.shape[1]).to(dev)
        pw = None
        if self.pos_weight:
            pos = Y[tr].sum(0)
            pw = torch.tensor((len(tr) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=dev)
        crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5) if self.plateau else None

        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(SEED)
        for _ in range(self.max_epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g)
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                if len(b) < 2:  # BatchNorm needs >1 sample
                    continue
                opt.zero_grad()
                loss = crit(self.net(Xt[b]), Yt[b])
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
            self.net.eval()
            with torch.no_grad():
                vl = crit(self.net(Xv), Yv).item()
            if sched:
                sched.step(vl)
            if vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            else:
                bad += 1
                if bad >= self.patience:
                    break
        self.net.load_state_dict(best_state)
        return self

    def score(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)
            return torch.sigmoid(self.net(x)).cpu().numpy()


class PerLabel:
    """One scaled sklearn estimator per label; returns continuous scores."""

    def __init__(self, make):
        self.make = make

    def fit(self, X, Y):
        Y = Y.reshape(len(Y), -1)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.models = []
        for j in range(Y.shape[1]):
            y = Y[:, j].astype(int)
            if y.min() == y.max():  # only one class present in this subset
                self.models.append(float(y[0]))
                continue
            self.models.append(self.make(y).fit(Xs, y))
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        out = []
        for m in self.models:
            if isinstance(m, float):
                out.append(np.full(len(X), m))
            elif hasattr(m, "predict_proba"):
                out.append(m.predict_proba(Xs)[:, 1])
            else:
                out.append(m.decision_function(Xs))
        return np.column_stack(out)


def model_factory(dataset, name, device):
    from xgboost import XGBClassifier

    if dataset == "ich":
        # tree/LR settings = tuned values in ich_results.txt (tuned on the training pool, 'any' label)
        zoo = {
            "LR": lambda: PerLabel(lambda y: LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000)),
            # C chosen by 3-fold CV on the 20% dev subset (C=0.001: 0.758, 0.01: 0.752, 0.1: 0.748 macro AUC);
            # the old C=1 barely converges at 1408 dims and is ~10x slower
            "SVM": lambda: PerLabel(lambda y: LinearSVC(C=0.001, class_weight="balanced", dual=False, max_iter=5000)),
            "RF": lambda: PerLabel(lambda y: RandomForestClassifier(
                n_estimators=200, max_depth=30, min_samples_split=5, class_weight="balanced_subsample",
                n_jobs=-1, random_state=SEED)),
            "XGB": lambda: PerLabel(lambda y: XGBClassifier(
                n_estimators=200, max_depth=10, learning_rate=0.1, subsample=0.8, scale_pos_weight=10,
                tree_method="hist", device=device, n_jobs=-1, random_state=SEED, eval_metric="logloss")),
            # "5-layer" MLP from MLP.ipynb: 1408 -> 128 -> 64 -> 6
            "MLP5": lambda: TorchMLP([128, 64], [0.5, 0.5], lr=1e-3, device=device),
            # "7-layer" Optuna architecture (model_hyperparameters.json)
            "MLP7": lambda: TorchMLP([807, 467, 258, 180, 126, 85, 43],
                                     [0.192, 0.252, 0.398, 0.441, 0.411, 0.411, 0.175],
                                     lr=1.8774581889e-4, device=device),
        }
    else:
        zoo = {
            "LR": lambda: PerLabel(lambda y: LogisticRegression(C=1.0, max_iter=2000)),
            # RBF SVC as in NewSinoCTModels.py; decision_function scores instead of Platt probabilities
            "SVM": lambda: PerLabel(lambda y: SVC(kernel="rbf", random_state=SEED)),
            "RF": lambda: PerLabel(lambda y: RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=SEED)),
            "XGB": lambda: PerLabel(lambda y: XGBClassifier(
                n_estimators=100, tree_method="hist", device=device, n_jobs=-1, random_state=SEED,
                eval_metric="logloss")),
            # "3-layer" MLP from KFoldMLP.ipynb: 1408 -> 1536 -> 896 -> 128 -> 1
            "MLP3": lambda: TorchMLP([1536, 896, 128], [0.4373, 0.3418, 0.2426], lr=1e-3,
                                     pos_weight=False, plateau=True, device=device),
            # "5-layer" SinoMLP from NewSinoCTModels.py: 1024(+residual) -> 512 -> 256 -> 128 -> 1
            "MLP5": lambda: TorchMLP([1024, 512, 256, 128], [0.1, 0.2, 0.3, 0.4], lr=1e-3,
                                     residual_first=True, pos_weight=False, device=device),
        }
    return zoo[name]


# ----------------------------------------------------------------------------- metrics
def youden(y, s):
    if y.min() == y.max():
        return np.nan
    fpr, tpr, thr = roc_curve(y, s)
    t = thr[np.argmax(tpr - fpr)]
    return float(min(t, np.max(s)))  # roc_curve's first threshold can be +inf


def binary_metrics(y, s, t):
    out = {}
    if y.min() != y.max():
        out["auc"] = roc_auc_score(y, s)
        out["ap"] = average_precision_score(y, s)
    if not np.isnan(t):
        p = (s >= t).astype(int)
        tp, fp = int(((p == 1) & (y == 1)).sum()), int(((p == 1) & (y == 0)).sum())
        tn, fn = int(((p == 0) & (y == 0)).sum()), int(((p == 0) & (y == 1)).sum())
        out["acc"] = (tp + tn) / len(y)
        out["sens"] = tp / (tp + fn) if tp + fn else np.nan
        out["spec"] = tn / (tn + fp) if tn + fp else np.nan
        out["prec"] = tp / (tp + fp) if tp + fp else np.nan
        out["f1"] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    return out


def all_metrics(Y, S, T, labels):
    per = {lab: binary_metrics(Y[:, j].astype(int), S[:, j], T[j]) for j, lab in enumerate(labels)}
    if len(labels) == 1:
        return {"overall": per[labels[0]], "per_label": per}
    keys = ["auc", "ap", "acc", "sens", "spec", "prec", "f1"]
    macro = {}
    for k in keys:
        v = [per[l][k] for l in labels if k in per[l] and not np.isnan(per[l][k])]
        if v:
            macro[k] = float(np.mean(v))
    macro["n_labels_auc"] = sum("auc" in per[l] for l in labels)
    return {"overall": macro, "per_label": per}


def bootstrap_ci(Y, S, T, labels, n=1000, seed=SEED):
    rng = np.random.default_rng(seed)
    strat = Y[:, 0].astype(int)
    pos, neg = np.flatnonzero(strat == 1), np.flatnonzero(strat == 0)
    draws = {}
    for _ in range(n):
        b = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        m = all_metrics(Y[b], S[b], T, labels)["overall"]
        for k, v in m.items():
            draws.setdefault(k, []).append(v)
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
            for k, v in draws.items() if k != "n_labels_auc"}


def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, np.integer):
        return int(o)
    return o


# ----------------------------------------------------------------------------- main loop
def run(dataset, models, ratios, folds, device, n_boot):
    X, Y, labels, info = load_dataset(dataset)
    dev, test = make_splits(Y)
    out_dir = os.path.join(ROOT, "results", dataset)
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump(clean({**info, "n_dev": len(dev), "n_test": len(test),
                         "test_pos_rate": Y[test].mean(0).tolist(), "labels": labels,
                         "dev_order_sha1": hashlib.sha1(dev.tobytes()).hexdigest()}), f, indent=1)
    np.save(os.path.join(out_dir, "test_idx.npy"), test)
    np.save(os.path.join(out_dir, "dev_order.npy"), dev)
    print(f"[{dataset}] {info} | dev={len(dev)} test={len(test)}", flush=True)

    for name in models:
        path = os.path.join(out_dir, f"{name}.jsonl")
        done = set()
        if os.path.exists(path):
            # A size counts as done only once its final 'test' record exists. Drop malformed lines and
            # records of sizes that were interrupted mid-run, so a rerun can't double-count their folds.
            recs_ok = []
            for line in open(path):
                try:
                    recs_ok.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            done = {round(r["ratio"], 4) for r in recs_ok if r.get("scheme") == "test"}
            keep = [r for r in recs_ok if round(r["ratio"], 4) in done]
            if len(keep) != len(recs_ok):
                print(f"[{dataset}] {name}: discarding {len(recs_ok) - len(keep)} records from an interrupted run",
                      flush=True)
                tmp = path + ".tmp"
                with open(tmp, "w") as fh:
                    fh.writelines(json.dumps(r) + "\n" for r in keep)
                os.replace(tmp, path)
        make = model_factory(dataset, name, device)
        for ratio in ratios:
            if round(ratio, 4) in done:
                continue
            n = max(int(round(ratio * len(dev))), 10)
            sub = dev[:n]
            Xs, Ys = X[sub], Y[sub]
            strat = Ys[:, 0].astype(int)
            t0 = time.time()
            recs, thresholds = [], []
            for k in folds:
                if np.bincount(strat, minlength=2).min() < k:
                    recs.append({"scheme": f"cv{k}", "fold": None, "skipped": "fewer positives than folds"})
                    continue
                oof = np.full(Ys.shape, np.nan)
                fold_ids = np.zeros(n, dtype=int)
                for f, (tr, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=SEED).split(Xs, strat)):
                    oof[va] = make().fit(Xs[tr], Ys[tr]).score(Xs[va])
                    fold_ids[va] = f
                T = np.array([youden(Ys[:, j], oof[:, j]) for j in range(Ys.shape[1])])
                thresholds.append(T)
                for f in range(k):
                    m = fold_ids == f
                    recs.append({"scheme": f"cv{k}", "fold": f, "threshold": T.tolist(),
                                 **all_metrics(Ys[m], oof[m], T, labels)})
                recs.append({"scheme": f"cv{k}", "fold": "pooled_oof", "threshold": T.tolist(),
                             **all_metrics(Ys, oof, T, labels)})
            model = make().fit(Xs, Ys)
            S = model.score(X[test])
            T = np.nanmean(np.vstack(thresholds), 0) if thresholds else np.full(len(labels), 0.5)
            T = np.where(np.isnan(T), 0.5, T)
            test_m = all_metrics(Y[test], S, T, labels)
            recs.append({"scheme": "test", "fold": None, "threshold": T.tolist(), **test_m,
                         "ci95": bootstrap_ci(Y[test], S, T, labels, n=n_boot)})
            np.savez_compressed(os.path.join(out_dir, "scores", f"{name}_r{int(round(ratio * 100)):03d}.npz"),
                                y=Y[test], s=S, t=T)
            block = "".join(json.dumps(clean({"dataset": dataset, "model": name, "ratio": ratio,
                                              "n_train": n, "n_pos_train": int(strat.sum()), **r})) + "\n"
                            for r in recs)
            with open(path, "a") as fh:  # one write per finished size, flushed to disk
                fh.write(block)
                fh.flush()
                os.fsync(fh.fileno())
            o = test_m["overall"]
            print(f"[{dataset}] {name:5s} {ratio*100:5.1f}% n={n:5d} test AUC={o.get('auc', float('nan')):.3f} "
                  f"AP={o.get('ap', float('nan')):.3f} F1={o.get('f1', float('nan')):.3f} "
                  f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ich", "sino"], required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    ap.add_argument("--folds", nargs="+", type=int, default=FOLDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        import torch
        torch.set_num_threads(a.threads)
    run(a.dataset, a.models, sorted(a.ratios), a.folds, a.device, a.n_boot)
