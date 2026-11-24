"""
Unified decoder evaluation on Google CT-FM embeddings (ICH + SinoCT).

Protocol (details in experiments/README.md):
  1. Load + de-duplicate embeddings (cached to cache/<dataset>.npz).
  2. ONE fixed, stratified 20% test set. It is never used for training, hyperparameter
     tuning, early stopping or threshold selection.
  3. The development pool is ordered once per seed (stratified), and training subset r% is
     the first r% of that order, so subsets are NESTED and shared by every model.
  4. For every (seed, size, model): 3-, 5- and 10-fold stratified CV inside the subset, then a
     refit on the whole subset scored on the fixed test set with bootstrap 95% CIs.
  5. Hyperparameters come from experiments/tune.py (tuned on the development pool only).

Usage:
  python experiments/tune.py       --dataset ich  --device cuda
  python experiments/scaling_cv.py --dataset ich  --device cuda
  python experiments/scaling_cv.py --dataset sino --device cuda --models LR SVM KNN MLP-focal
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, LinearSVC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42
RATIOS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 1.00]
FOLDS = [3, 5, 10]
ICH_LABELS = ["any", "epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]
DEFAULT_MODELS = {
    "ich": ["LR", "SVM", "KNN", "RF", "XGB", "MLP", "MLP-focal", "MLP5", "MLP7"],
    "sino": ["LR", "SVM", "KNN", "RF", "XGB", "MLP", "MLP-focal", "MLP3", "MLP5"],
}
DEFAULT_MODELS["ich_study"] = DEFAULT_MODELS["ich"]


# ============================================================================ data
def _parse(df):
    return np.stack([np.asarray(json.loads(s), dtype=np.float32) for s in df["embedding"]])


def load_dataset(name):
    cache = os.path.join(ROOT, "cache", f"{name}.npz")
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        info = dict(json.loads(str(d["info"])))
        if info.get("version") == 3:
            return d["X"], d["Y"], list(d["labels"]), info

    info = {"version": 3}
    if name in ("ich", "ich_study"):
        df = pd.concat([pd.read_csv(os.path.join(ROOT, "Rsna_ICH_Hemm.csv")),
                        pd.read_csv(os.path.join(ROOT, "Rsna_ICH_Control.csv"))], ignore_index=True)
        info["rows_raw"] = len(df)
        # exact-duplicate embeddings: keep one copy if labels agree, drop the group if they conflict
        df["h"] = df["embedding"].map(lambda s: hashlib.md5(s.encode()).hexdigest())
        conflict = df.groupby("h")[ICH_LABELS].nunique().max(axis=1) > 1
        info["dup_embedding_rows_dropped"] = int(df.duplicated("h").sum())
        info["conflicting_dup_groups_dropped"] = int(conflict.sum())
        df = df[~df["h"].isin(conflict[conflict].index)].drop_duplicates("h")
        labels = ICH_LABELS
        if name == "ich_study":
            # Sensitivity variant: each RSNA volume is ONE study of the patient, but the CSV labels are
            # patient-level (max over all of the patient's studies). Use the labels of the study that was
            # actually converted (data/rsna_ich_id_map.csv, built from data.ftr + nii_report), and drop
            # multi-study rows whose converted study can't be identified.
            m = pd.read_csv(os.path.join(ROOT, "data", "rsna_ich_id_map.csv"))
            m = m[m["study_any"].notna()].set_index("url")
            info["ich_study_rows_dropped"] = int((~df["url"].isin(m.index)).sum())
            df = df[df["url"].isin(m.index)].copy()
            for k in labels:
                df[k] = m.loc[df["url"], f"study_{k}"].values
            info["label_source"] = "study-level (converted study)"
        Y = df[labels].values.astype(np.float32)
    elif name == "sino":
        df = pd.read_csv(os.path.join(ROOT, "Sino_CT.csv"))
        info["rows_raw"] = len(df)
        info["dup_url_rows_dropped"] = int(df.duplicated("url").sum())
        df = df.drop_duplicates("url")
        labels = ["abnormal"]
        # Positive class = "1,0", the same mapping as NewSinoCTModels.py (confirmed by Bahram as the
        # abnormal class; consistent with SinoCT being ~55% abnormal and "1,0" being the larger class).
        # Note: an old comment in OIdSinoCTModels/sino_ct_model.ipynb called "1,0" normal.
        Y = (df["label"] == "1,0").values.astype(np.float32)[:, None]
        info["positive_class"] = "abnormal ('1,0')"
    else:
        raise ValueError(name)

    X = _parse(df)
    info["rows_used"] = len(df)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + ".tmp.npz"  # write-then-rename so an interrupted first run can't leave a corrupt cache
    np.savez(tmp, X=X, Y=Y, labels=np.array(labels), info=json.dumps(info))
    os.replace(tmp, cache)
    return X, Y, labels, info


def nested_order(strat, seed):
    """Stratified permutation: each class is shuffled and the classes are interleaved by
    fractional rank, so every prefix has ~the pool's class balance."""
    rng = np.random.default_rng(seed)
    key = np.empty(len(strat))
    for c in np.unique(strat):
        idx = np.flatnonzero(strat == c)
        rng.shuffle(idx)
        key[idx] = (np.arange(len(idx)) + rng.random(len(idx))) / len(idx)
    return np.argsort(key, kind="stable")


def make_splits(Y, order_seed=SEED):
    """Test split is always the same (SEED); only the dev-pool ordering depends on order_seed."""
    strat = Y[:, 0].astype(int)  # 'any' for ICH, abnormal for SinoCT
    idx = np.arange(len(Y))
    dev, test = train_test_split(idx, test_size=0.2, stratify=strat, random_state=SEED)
    dev = np.sort(dev)
    dev = dev[nested_order(strat[dev], order_seed)]
    return dev, test


# ============================================================================ models
class TorchMLP:
    """Binary/multilabel MLP. loss: 'bce' | 'wbce' (pos_weight) | 'focal' (gamma=2).
    Early stopping on a 10% split carved from the TRAINING data only.

    Training budget is size-aware: at least `min_steps` gradient updates (so tiny subsets, which
    have only a few batches per epoch, are not stopped after a few dozen updates), and early
    stopping cannot trigger during the first 20% of the budget. At full size this reduces to the
    original 50 epochs / patience 5."""

    def __init__(self, hidden, dropout, lr, loss="wbce", residual_first=False, plateau=False,
                 max_epochs=50, patience=5, batch_size=32, device="cpu", seed=SEED, gamma=2.0,
                 min_steps=3000, max_epochs_cap=500):
        self.hidden, self.dropout, self.lr, self.loss = list(hidden), list(dropout), lr, loss
        self.residual_first, self.plateau, self.max_epochs = residual_first, plateau, max_epochs
        # with an LR-on-plateau scheduler (patience 5), give it room to act before stopping
        self.patience = max(patience, 10) if plateau else patience
        self.batch_size, self.device, self.seed, self.gamma = batch_size, device, seed, gamma
        self.min_steps, self.max_epochs_cap = min_steps, max_epochs_cap

    def _build(self, d_in, d_out):
        import torch.nn as nn

        hidden, dropout, residual = self.hidden, self.dropout, self.residual_first

        class Net(nn.Module):
            def __init__(s):
                super().__init__()
                blocks, prev = [], d_in
                for h, p in zip(hidden, dropout):
                    blocks.append(nn.Sequential(nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)))
                    prev = h
                s.blocks = nn.ModuleList(blocks)
                s.res = nn.Linear(d_in, hidden[0]) if residual else None
                s.out = nn.Linear(prev, d_out)

            def forward(s, x):
                h = s.blocks[0](x)
                if s.res is not None:
                    h = h + s.res(x)
                for b in s.blocks[1:]:
                    h = b(h)
                return s.out(h)

        return Net()

    def _criterion(self, pos_weight):
        import torch
        import torch.nn.functional as F

        if self.loss == "bce":
            return lambda z, y: F.binary_cross_entropy_with_logits(z, y)
        if self.loss == "wbce":
            return lambda z, y: F.binary_cross_entropy_with_logits(z, y, pos_weight=pos_weight)
        if self.loss == "focal":
            g = self.gamma

            def focal(z, y):
                ce = F.binary_cross_entropy_with_logits(z, y, reduction="none")
                p_t = torch.exp(-ce)
                return ((1 - p_t) ** g * ce).mean()
            return focal
        raise ValueError(self.loss)

    def fit(self, X, Y):
        import torch
        import torch.nn as nn
        torch.manual_seed(self.seed)
        Y = Y.reshape(len(Y), -1)
        strat = Y[:, 0].astype(int)
        can_strat = np.bincount(strat, minlength=2).min() >= 2
        tr, va = train_test_split(np.arange(len(X)), test_size=0.1, random_state=self.seed,
                                  stratify=strat if can_strat else None)
        self.scaler = StandardScaler().fit(X[tr])
        dev = self.device
        Xt = torch.tensor(self.scaler.transform(X[tr]), dtype=torch.float32, device=dev)
        Yt = torch.tensor(Y[tr], dtype=torch.float32, device=dev)
        Xv = torch.tensor(self.scaler.transform(X[va]), dtype=torch.float32, device=dev)
        Yv = torch.tensor(Y[va], dtype=torch.float32, device=dev)

        self.net = self._build(X.shape[1], Y.shape[1]).to(dev)
        pos = Y[tr].sum(0)
        pw = torch.tensor((len(tr) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=dev)
        crit = self._criterion(pw)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=5) if self.plateau else None

        steps_per_epoch = max(1, int(np.ceil(len(tr) / self.batch_size)))
        max_epochs = min(self.max_epochs_cap, max(self.max_epochs, int(np.ceil(self.min_steps / steps_per_epoch))))
        patience = max(self.patience, int(round(self.patience * max_epochs / self.max_epochs)))
        warmup = int(0.2 * max_epochs) if max_epochs > self.max_epochs else 0
        best, best_state, bad = np.inf, None, 0
        g = torch.Generator(device="cpu").manual_seed(self.seed)
        for epoch in range(max_epochs):
            self.net.train()
            perm = torch.randperm(len(Xt), generator=g).to(dev)
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
            if best_state is None or vl < best - 1e-4:
                best, bad = vl, 0
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
            elif epoch >= warmup:
                bad += 1
                if bad >= patience:
                    break
        self.net.load_state_dict(best_state)
        self.best_val_loss, self.epochs_run = best, epoch + 1
        return self

    def score(self, X):
        import torch
        self.net.eval()
        with torch.no_grad():
            x = torch.tensor(self.scaler.transform(X), dtype=torch.float32, device=self.device)
            return torch.sigmoid(self.net(x)).cpu().numpy()


class PerLabel:
    """One standardized sklearn estimator per label; returns continuous scores."""

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


class InnerCVKNN:
    """Cosine k-NN on standardized embeddings. k is re-selected at EVERY fit by 3-fold CV on the
    training data (AUC on the first label), so it adapts to the training-set size."""
    K_GRID = (1, 3, 5, 10, 20, 50, 100)

    def __init__(self, seed=SEED):
        self.seed = seed

    def _knn(self, k):
        return KNeighborsClassifier(n_neighbors=k, metric="cosine", weights="distance", algorithm="brute")

    def fit(self, X, Y):
        Y = Y.reshape(len(Y), -1)
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        y0 = Y[:, 0].astype(int)
        n_inner = 3 if np.bincount(y0, minlength=2).min() >= 3 else 0
        max_k = int(len(X) * (2 / 3)) - 1
        grid = [k for k in self.K_GRID if k <= max_k] or [1]
        if n_inner and len(grid) > 1:
            scores = {k: [] for k in grid}
            for tr, va in StratifiedKFold(n_inner, shuffle=True, random_state=self.seed).split(Xs, y0):
                if y0[va].min() == y0[va].max():
                    continue
                for k in grid:
                    s = self._knn(k).fit(Xs[tr], y0[tr]).predict_proba(Xs[va])
                    scores[k].append(roc_auc_score(y0[va], s[:, 1] if s.shape[1] > 1 else s[:, 0]))
            self.k = max(grid, key=lambda k: np.mean(scores[k]) if scores[k] else -1)
        else:
            self.k = grid[-1]
        self.k = min(self.k, len(X) - 1)
        self.models = []
        for j in range(Y.shape[1]):
            y = Y[:, j].astype(int)
            self.models.append(float(y[0]) if y.min() == y.max() else self._knn(self.k).fit(Xs, y))
        return self

    def score(self, X):
        Xs = self.scaler.transform(X)
        return np.column_stack([np.full(len(X), m) if isinstance(m, float) else m.predict_proba(Xs)[:, 1]
                                for m in self.models])


def load_params(dataset):
    p = os.path.join(ROOT, "results", dataset, "tuned_params.json")
    return json.load(open(p))["best"] if os.path.exists(p) else None


C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100)
RBF_C_GRID = (0.1, 0.3, 1, 3, 10)
PER_SIZE_C = {"LR", "SVM"}  # regularization strength re-selected on each training subset


def select_C(dataset, name, device, seed, params, X, Y):
    """Pick C for LR/SVM by 3-fold CV on the TRAINING SUBSET only (AUC on the first label), using the
    one-standard-error rule: the most regularized C whose mean CV AUC is within 1 SE of the best.
    Small subsets therefore get stronger regularization than the full pool. Other tuned params are kept."""
    y0 = Y[:, 0].astype(int)
    if np.bincount(y0, minlength=2).min() < 3:
        return None
    grid = RBF_C_GRID if (dataset == "sino" and name.partition("-")[0] == "SVM") else C_GRID
    folds = list(StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y0))
    stats = []
    for C in grid:  # ascending C = decreasing regularization
        make = model_factory(dataset, name, device, seed, params, overrides={"C": C})
        aucs = []
        for tr, va in folds:
            if y0[va].min() == y0[va].max():
                continue
            aucs.append(roc_auc_score(y0[va], make().fit(X[tr], Y[tr]).score(X[va])[:, 0]))
        stats.append((C, np.mean(aucs), np.std(aucs, ddof=1) / np.sqrt(len(aucs)) if len(aucs) > 1 else 0.0))
    best = max(stats, key=lambda t: t[1])
    return next(C for C, mu, _ in stats if mu >= best[1] - best[2])


def model_factory(dataset, name, device, seed=SEED, params=None, overrides=None):
    """name = BASE or BASE-LOSS for MLPs (e.g. 'MLP-focal', 'MLP7-bce')."""
    from xgboost import XGBClassifier

    base, _, loss = name.partition("-")
    need_tuned = {"LR", "SVM", "RF", "XGB", "MLP"}
    if base in need_tuned and not (params and base in params):
        raise SystemExit(f"No tuned parameters for {base}. Run: python experiments/tune.py --dataset {dataset}")
    P = {**(params.get(base, {}) if params else {}), **(overrides or {})}
    is_ich = dataset.startswith("ich")
    default_loss = "wbce" if is_ich else "bce"
    loss = loss or default_loss

    if base == "LR":
        return lambda: PerLabel(lambda y: LogisticRegression(max_iter=3000, **P))
    if base == "SVM":
        if is_ich:  # linear SVM as in the deck (RBF is O(n^2) on 13k scans x 6 labels)
            return lambda: PerLabel(lambda y: LinearSVC(dual=False, max_iter=5000, **P))
        return lambda: PerLabel(lambda y: SVC(kernel="rbf", random_state=seed, **P))
    if base == "KNN":
        return lambda: InnerCVKNN(seed=seed)
    if base == "RF":
        return lambda: PerLabel(lambda y: RandomForestClassifier(n_jobs=-1, random_state=seed, **P))
    if base == "XGB":
        return lambda: PerLabel(lambda y: XGBClassifier(tree_method="hist", device=device, n_jobs=-1,
                                                        random_state=seed, eval_metric="logloss", **P))
    if base == "MLP":  # architecture/lr from the Optuna search in tune.py (dev pool only)
        return lambda: TorchMLP(P["hidden"], P["dropout"], P["lr"], loss=loss, device=device, seed=seed)

    # ---- legacy MLPs, exactly as in the deck (for continuity; see README caveats) ----
    if is_ich and base == "MLP5":  # MLP.ipynb: 1408 -> 128 -> 64 -> 6
        return lambda: TorchMLP([128, 64], [0.5, 0.5], 1e-3, loss=loss, device=device, seed=seed)
    if is_ich and base == "MLP7":  # Optuna architecture in model_hyperparameters.json
        return lambda: TorchMLP([807, 467, 258, 180, 126, 85, 43],
                                [0.192, 0.252, 0.398, 0.441, 0.411, 0.411, 0.175],
                                1.8774581889e-4, loss=loss, device=device, seed=seed)
    if dataset == "sino" and base == "MLP3":  # KFoldMLP.ipynb: 1408 -> 1536 -> 896 -> 128 -> 1
        return lambda: TorchMLP([1536, 896, 128], [0.4373, 0.3418, 0.2426], 1e-3, loss=loss,
                                plateau=True, device=device, seed=seed)
    if dataset == "sino" and base == "MLP5":  # NewSinoCTModels.py SinoMLP: 1024(+res) -> 512 -> 256 -> 128 -> 1
        return lambda: TorchMLP([1024, 512, 256, 128], [0.1, 0.2, 0.3, 0.4], 1e-3, loss=loss,
                                residual_first=True, device=device, seed=seed)
    raise SystemExit(f"Unknown model {name} for {dataset}")


# ============================================================================ metrics
def youden(y, s):
    if y.min() == y.max():
        return np.nan
    fpr, tpr, thr = roc_curve(y, s)
    return float(min(thr[np.argmax(tpr - fpr)], np.max(s)))  # first roc threshold can be +inf


def f1_opt(y, s):
    if y.min() == y.max():
        return np.nan
    order = np.argsort(-s)
    ys, ss = y[order], s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys); fn = ys.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    last = np.r_[ss[1:] != ss[:-1], True]  # only cut between distinct scores
    i = np.argmax(np.where(last, f1, -1))
    return float(ss[i])


def thresholds(Y, S, rule):
    fn = youden if rule == "youden" else f1_opt
    return np.array([fn(Y[:, j].astype(int), S[:, j]) for j in range(Y.shape[1])])


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
    macro = {}
    for k in ["auc", "ap", "acc", "sens", "spec", "prec", "f1"]:
        v = [per[l][k] for l in labels if k in per[l] and not np.isnan(per[l][k])]
        if v:
            macro[k] = float(np.mean(v))
    macro["n_labels_auc"] = sum("auc" in per[l] for l in labels)
    return {"overall": macro, "per_label": per}


def bootstrap_indices(Y, n, seed=SEED):
    """Stratified (on the first label) bootstrap resamples of the test set. Same indices for every
    model, so paired model comparisons can reuse them."""
    rng = np.random.default_rng(seed)
    strat = Y[:, 0].astype(int)
    pos, neg = np.flatnonzero(strat == 1), np.flatnonzero(strat == 0)
    return [np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]) for _ in range(n)]


def bootstrap_ci(Y, S, T, labels, boots):
    draws = {}
    for b in boots:
        for k, v in all_metrics(Y[b], S[b], T, labels)["overall"].items():
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


# ============================================================================ resume helpers
def load_done(path, dataset, name):
    """A (seed, size) counts as done only once its final 'test' record exists. Malformed lines and
    records of interrupted sizes are dropped so a rerun can't double-count their folds."""
    if not os.path.exists(path):
        return set()
    recs = []
    for line in open(path):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    key = lambda r: (r.get("seed", 0), round(r["ratio"], 4))
    done = {key(r) for r in recs if r.get("scheme") == "test"}
    keep = [r for r in recs if key(r) in done]
    if len(keep) != len(recs):
        print(f"[{dataset}] {name}: discarding {len(recs) - len(keep)} records from an interrupted run", flush=True)
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            fh.writelines(json.dumps(r) + "\n" for r in keep)
        os.replace(tmp, path)
    return done


# ============================================================================ main loop
def run_one(dataset, name, make, X, Y, labels, dev, test, seed, ratio, boots, out_dir, device, params):
    n = max(int(round(ratio * len(dev))), 10)
    sub = dev[:n]
    Xs, Ys = X[sub], Y[sub]
    strat = Ys[:, 0].astype(int)
    recs, th_y, th_f = [], [], []
    chosen_C = None
    if name.partition("-")[0] in PER_SIZE_C:
        chosen_C = select_C(dataset, name, device, SEED + seed, params, Xs, Ys)
        if chosen_C is not None:
            make = model_factory(dataset, name, device, SEED + seed, params, overrides={"C": chosen_C})
    for k in FOLDS:
        if np.bincount(strat, minlength=2).min() < k:
            recs.append({"scheme": f"cv{k}", "fold": None, "skipped": "fewer positives than folds"})
            continue
        oof = np.full(Ys.shape, np.nan)
        fold_ids = np.zeros(n, dtype=int)
        for f, (tr, va) in enumerate(StratifiedKFold(k, shuffle=True, random_state=SEED + seed).split(Xs, strat)):
            oof[va] = make().fit(Xs[tr], Ys[tr]).score(Xs[va])
            fold_ids[va] = f
        Ty, Tf = thresholds(Ys, oof, "youden"), thresholds(Ys, oof, "f1")
        th_y.append(Ty); th_f.append(Tf)
        for f in range(k):
            m = fold_ids == f
            # fold threshold from the OTHER folds' out-of-fold predictions (no peeking at fold f)
            Tyf, Tff = thresholds(Ys[~m], oof[~m], "youden"), thresholds(Ys[~m], oof[~m], "f1")
            recs.append({"scheme": f"cv{k}", "fold": f, "threshold": Tyf.tolist(),
                         **all_metrics(Ys[m], oof[m], Tyf, labels),
                         "overall_f1thr": all_metrics(Ys[m], oof[m], Tff, labels)["overall"]})
        recs.append({"scheme": f"cv{k}", "fold": "pooled_oof", "threshold": Ty.tolist(),
                     **all_metrics(Ys, oof, Ty, labels)})

    model = make().fit(Xs, Ys)
    S = model.score(X[test])
    mean_t = lambda L: np.where(np.isnan(m := (np.nanmean(np.vstack(L), 0) if L else np.full(len(labels), np.nan))),
                                0.5, m)
    Ty, Tf = mean_t(th_y), mean_t(th_f)
    test_m = all_metrics(Y[test], S, Ty, labels)
    rec = {"scheme": "test", "fold": None, "threshold": Ty.tolist(), "threshold_f1": Tf.tolist(), **test_m,
           "overall_f1thr": all_metrics(Y[test], S, Tf, labels)["overall"],
           "ci95": bootstrap_ci(Y[test], S, Ty, labels, boots),
           "ci95_f1thr": bootstrap_ci(Y[test], S, Tf, labels, boots)}
    if hasattr(model, "k"):
        rec["knn_k"] = int(model.k)
    if chosen_C is not None:
        rec["C"] = chosen_C
    recs.append(rec)
    np.savez_compressed(os.path.join(out_dir, "scores", f"{name}_s{seed}_r{int(round(ratio * 100)):03d}.npz"),
                        y=Y[test], s=S, t=Ty, t_f1=Tf)
    return n, int(strat.sum()), recs, test_m["overall"]


def run(dataset, models, ratios, seeds, repeat_max_ratio, device, n_boot):
    X, Y, labels, info = load_dataset(dataset)
    params = load_params(dataset)
    out_dir = os.path.join(ROOT, "results", dataset)
    os.makedirs(os.path.join(out_dir, "scores"), exist_ok=True)
    _, test = make_splits(Y)
    boots = bootstrap_indices(Y[test], n_boot)
    with open(os.path.join(out_dir, "split_info.json"), "w") as f:
        json.dump(clean({**info, "n_dev": len(Y) - len(test), "n_test": len(test), "labels": labels,
                         "test_pos_rate": Y[test].mean(0).tolist(), "seeds": seeds,
                         "repeat_max_ratio": repeat_max_ratio,
                         "test_idx_sha1": hashlib.sha1(np.sort(test).tobytes()).hexdigest()}), f, indent=1)
    np.save(os.path.join(out_dir, "test_idx.npy"), test)
    print(f"[{dataset}] {info} | test={len(test)} | tuned params: {'yes' if params else 'NO'}", flush=True)

    for seed in seeds:
        dev, _ = make_splits(Y, order_seed=SEED + seed)
        for name in models:
            path = os.path.join(out_dir, f"{name}.jsonl")
            done = load_done(path, dataset, name)
            make = model_factory(dataset, name, device, seed=SEED + seed, params=params)
            for ratio in ratios:
                if seed != seeds[0] and ratio > repeat_max_ratio + 1e-9:
                    continue  # extra seeds only where subset choice matters (small sizes)
                if (seed, round(ratio, 4)) in done:
                    continue
                t0 = time.time()
                n, npos, recs, o = run_one(dataset, name, make, X, Y, labels, dev, test, seed, ratio, boots, out_dir,
                                         device, params)
                block = "".join(json.dumps(clean({"dataset": dataset, "model": name, "seed": seed, "ratio": ratio,
                                                  "n_train": n, "n_pos_train": npos, **r})) + "\n" for r in recs)
                with open(path, "a") as fh:  # one flushed write per finished (seed, size)
                    fh.write(block); fh.flush(); os.fsync(fh.fileno())
                print(f"[{dataset}] seed{seed} {name:10s} {ratio*100:5.1f}% n={n:5d} test AUC={o.get('auc', np.nan):.3f} "
                      f"AP={o.get('ap', np.nan):.3f} F1={o.get('f1', np.nan):.3f} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ich", "ich_study", "sino"], required=True)
    ap.add_argument("--models", nargs="+", default=None, help="default: all models for the dataset")
    ap.add_argument("--ratios", nargs="+", type=float, default=RATIOS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4],
                    help="nested-subset orderings; seed 0 runs every size, others only up to --repeat-max-ratio")
    ap.add_argument("--repeat-max-ratio", type=float, default=0.10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args()
    run(a.dataset, a.models or DEFAULT_MODELS[a.dataset], sorted(a.ratios), a.seeds, a.repeat_max_ratio,
        a.device, a.n_boot)
