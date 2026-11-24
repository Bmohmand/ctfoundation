"""
Hyperparameter tuning on the DEVELOPMENT POOL ONLY (the fixed test set is never touched).

  python experiments/tune.py --dataset ich  --device cuda
  python experiments/tune.py --dataset sino --device cuda

Writes results/<dataset>/tuned_params.json, which scaling_cv.py reads.

- LR / SVM / RF / XGB: RandomizedSearchCV (3-fold, ROC AUC on the first label: ICH 'any',
  SinoCT 'abnormal'), with the grids from NewICHModels.py (plus a C grid for SVM).
- MLP: Optuna, same search space as MLP_Optimized.ipynb (3-7 layers, each layer between half and
  double the previous width, capped at 2048; dropout 0.1-0.5; lr 1e-5 to 1e-2), 50 trials. Each
  trial trains on 85% of the dev pool (early stopping on a slice of that 85%) and is scored by its
  loss on the remaining 15%. MLP_Optimized.ipynb scored trials on the test split; this does not.
- Hyperparameters are tuned once at 100% of the dev pool and reused at every training size
  (k-NN is the exception: k is re-selected inside every fit).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scaling_cv as S  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC, LinearSVC  # noqa: E402


def grids(dataset, device):
    from xgboost import XGBClassifier
    g = {
        "LR": (LogisticRegression(max_iter=3000),
               {"C": [0.001, 0.01, 0.1, 1, 10, 100], "class_weight": ["balanced", None]}),
        "RF": (RandomForestClassifier(n_jobs=-1, random_state=S.SEED),
               {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20, 30],
                "min_samples_split": [2, 5, 10], "class_weight": ["balanced", "balanced_subsample", None]}),
        "XGB": (XGBClassifier(tree_method="hist", device=device, n_jobs=-1, random_state=S.SEED,
                              eval_metric="logloss"),
                {"learning_rate": [0.01, 0.1, 0.2], "max_depth": [3, 6, 10], "n_estimators": [100, 200, 300],
                 "subsample": [0.8, 1.0], "scale_pos_weight": [1, 5, 10]}),
    }
    if dataset == "ich":
        g["SVM"] = (LinearSVC(dual=False, max_iter=5000),
                    {"C": [1e-4, 1e-3, 1e-2, 1e-1, 1], "class_weight": ["balanced", None]})
    else:
        g["SVM"] = (SVC(kernel="rbf", random_state=S.SEED),
                    {"C": [0.1, 0.3, 1, 3, 10], "gamma": ["scale"], "class_weight": ["balanced", None]})
    return g


def tune_sklearn(name, est, grid, X, y, n_iter):
    n_combos = int(np.prod([len(v) for v in grid.values()]))
    search = RandomizedSearchCV(make_pipeline(StandardScaler(), est),
                                {f"{est.__class__.__name__.lower()}__{k}": v for k, v in grid.items()},
                                n_iter=min(n_iter, n_combos), cv=StratifiedKFold(3, shuffle=True, random_state=S.SEED),
                                scoring="roc_auc", n_jobs=1, random_state=S.SEED, refit=False)
    t = time.time()
    search.fit(X, y)
    best = {k.split("__", 1)[1]: v for k, v in search.best_params_.items()}
    print(f"  {name}: {best}  CV AUC={search.best_score_:.4f}  ({time.time() - t:.0f}s)", flush=True)
    return best, float(search.best_score_)


def tune_mlp(dataset, X, Y, device, n_trials):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    strat = Y[:, 0].astype(int)
    tr, va = train_test_split(np.arange(len(X)), test_size=0.15, stratify=strat, random_state=S.SEED)
    loss = "wbce" if dataset == "ich" else "bce"

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 3, 7)
        hidden, prev = [], X.shape[1]
        for i in range(n_layers):
            hidden.append(trial.suggest_int(f"n_units_l{i}", int(prev / 2), min(2048, prev * 2)))
            prev = hidden[-1]
        dropout = [trial.suggest_float(f"dropout_l{i}", 0.1, 0.5) for i in range(n_layers)]
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        m = S.TorchMLP(hidden, dropout, lr, loss=loss, device=device).fit(X[tr], Y[tr])
        import torch
        with torch.no_grad():
            m.net.eval()
            xv = torch.tensor(m.scaler.transform(X[va]), dtype=torch.float32, device=device)
            yv = torch.tensor(Y[va], dtype=torch.float32, device=device)
            pos = Y[tr].sum(0)
            pw = torch.tensor((len(tr) - pos) / np.maximum(pos, 1), dtype=torch.float32, device=device)
            val = m._criterion(pw)(m.net(xv), yv).item()
        trial.set_user_attr("hidden", hidden); trial.set_user_attr("dropout", dropout)
        return val

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=S.SEED))
    t = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    b = study.best_trial
    best = {"hidden": b.user_attrs["hidden"], "dropout": [round(x, 4) for x in b.user_attrs["dropout"]],
            "lr": b.params["lr"]}
    print(f"  MLP: {len(best['hidden'])} hidden layers {best['hidden']} lr={best['lr']:.2e}  "
          f"val loss={b.value:.4f}  ({time.time() - t:.0f}s)", flush=True)
    return best, float(b.value)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ich", "sino"], required=True)
    ap.add_argument("--models", nargs="+", default=["LR", "SVM", "RF", "XGB", "MLP"])
    ap.add_argument("--n-iter", type=int, default=20, help="random-search draws per sklearn model")
    ap.add_argument("--n-trials", type=int, default=50, help="Optuna trials for the MLP")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true", help="retune models that already have saved params")
    a = ap.parse_args()

    X, Y, labels, info = S.load_dataset(a.dataset)
    dev, test = S.make_splits(Y)
    Xd, Yd = X[dev], Y[dev]
    print(f"[{a.dataset}] tuning on dev pool n={len(dev)} (test n={len(test)} untouched)", flush=True)

    out = os.path.join(S.ROOT, "results", a.dataset, "tuned_params.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    res = json.load(open(out)) if os.path.exists(out) else {"best": {}, "score": {}}
    g = grids(a.dataset, a.device)
    for m in a.models:
        if m in res["best"] and not a.force:
            print(f"  {m}: already tuned, skipping (use --force to redo)", flush=True)
            continue
        if m == "MLP":
            res["best"][m], res["score"][m] = tune_mlp(a.dataset, Xd, Yd, a.device, a.n_trials)
        else:
            est, grid = g[m]
            res["best"][m], res["score"][m] = tune_sklearn(m, est, grid, Xd, Yd[:, 0].astype(int), a.n_iter)
        res["tuned_on"] = {"dataset": a.dataset, "n_dev": len(dev), "target": labels[0],
                           "note": "dev pool only; test set never used"}
        tmp = out + ".tmp"
        json.dump(res, open(tmp, "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        os.replace(tmp, out)  # saved after every model, so an interrupted run keeps finished models
    print("saved", out)


if __name__ == "__main__":
    main()
