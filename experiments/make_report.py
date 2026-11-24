"""Aggregate results/<dataset>/*.jsonl into slide-ready tables and figures.

  python experiments/make_report.py --dataset ich
  python experiments/make_report.py --dataset sino

Writes results/<dataset>/report/:
  model_comparison.{csv,md}   test metrics at 100% of the dev pool, 95% bootstrap CI
  scaling_test.{csv,md}       test metrics vs training-set size, per model
  cv_by_size.{csv,md}         3/5/10-fold CV mean +- SD (across folds) at every size
  roc_<model>.png, roc_all_models.png, auc_vs_size.png, cv_auc_<model>.png
"""
import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# validated categorical palette (fixed order, never cycled)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
ORDER = ["LR", "SVM", "RF", "XGB", "MLP3", "MLP5", "MLP7"]
COLOR = dict(zip(ORDER, PALETTE))
NAMES = {"ich": {"LR": "Logistic regression", "SVM": "Linear SVM", "RF": "Random forest", "XGB": "XGBoost",
                 "MLP5": "MLP (5-layer)", "MLP7": "MLP (7-layer)"},
         "sino": {"LR": "Logistic regression", "SVM": "RBF SVM", "RF": "Random forest", "XGB": "XGBoost",
                  "MLP3": "MLP (3-layer)", "MLP5": "MLP (5-layer)"}}
METRICS = ["auc", "ap", "acc", "sens", "spec", "f1"]
LABEL = {"auc": "ROC AUC", "ap": "PR AUC (AP)", "acc": "Accuracy", "sens": "Sensitivity",
         "spec": "Specificity", "f1": "F1", "prec": "Precision"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a95", "axes.labelcolor": "#3d3d3a", "xtick.color": "#5f5f5a",
                     "ytick.color": "#5f5f5a", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
                     "legend.frameon": False})


def load(ds):
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", ds, "*.jsonl")):
        rows += [json.loads(l) for l in open(p)]
    return rows


def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out) + "\n"


def save(df, rep, name, note=""):
    df.to_csv(os.path.join(rep, f"{name}.csv"), index=False)
    with open(os.path.join(rep, f"{name}.md"), "w") as f:
        f.write(note + ("\n\n" if note else "") + to_md(df))


def main(ds):
    rows = load(ds)
    rep = os.path.join(ROOT, "results", ds, "report")
    os.makedirs(rep, exist_ok=True)
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    test = [r for r in rows if r["scheme"] == "test"]

    # --- model comparison at 100% ---
    tab = []
    for m in models:
        r = [x for x in test if x["model"] == m and abs(x["ratio"] - 1) < 1e-9]
        if not r:
            continue
        r = r[0]; o, ci = r["overall"], r["ci95"]
        row = {"Model": NAMES[ds][m], "n_train": r["n_train"]}
        for k in METRICS:
            row[LABEL[k]] = f"{fmt(o.get(k))} [{fmt(ci.get(k, [None])[0])}–{fmt(ci.get(k, [None, None])[1])}]" \
                if k in ci else fmt(o.get(k))
        tab.append(row)
    if tab:
        save(pd.DataFrame(tab), rep, "model_comparison",
             "Fixed held-out test set; models trained on 100% of the development pool. "
             "Brackets = 95% stratified bootstrap CI (1000 resamples). Threshold = mean Youden-J threshold from "
             "3/5/10-fold out-of-fold predictions." + (" Macro averages over the 6 ICH labels." if ds == "ich" else ""))

    # --- scaling (test) ---
    tab = []
    for r in sorted(test, key=lambda x: (ORDER.index(x["model"]), x["ratio"])):
        o = r["overall"]
        tab.append({"Model": NAMES[ds][r["model"]], "Train %": int(round(r["ratio"] * 100)), "n": r["n_train"],
                    "n_pos": r["n_pos_train"], **{LABEL[k]: fmt(o.get(k)) for k in METRICS},
                    "AUC 95% CI": f"{fmt(r['ci95'].get('auc', [None])[0])}–{fmt(r['ci95'].get('auc', [None, None])[1])}"})
    if tab:
        save(pd.DataFrame(tab), rep, "scaling_test",
             "Held-out test performance by training-set size (% of the development pool). Subsets are nested.")

    # --- CV by size ---
    tab = []
    for m in models:
        for ratio in sorted({r["ratio"] for r in rows if r["model"] == m}):
            for k in (3, 5, 10):
                fr = [r for r in rows if r["model"] == m and r["ratio"] == ratio and r["scheme"] == f"cv{k}"]
                if not fr or "skipped" in fr[0]:
                    tab.append({"Model": NAMES[ds][m], "Train %": int(round(ratio * 100)), "Folds": k,
                                **{LABEL[x]: "NA" for x in METRICS}})
                    continue
                folds = [r for r in fr if isinstance(r["fold"], int)]
                row = {"Model": NAMES[ds][m], "Train %": int(round(ratio * 100)), "Folds": k}
                for x in METRICS:
                    v = np.array([f["overall"].get(x) for f in folds], dtype=float)
                    v = v[~np.isnan(v)]
                    row[LABEL[x]] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else "NA"
                tab.append(row)
    if tab:
        save(pd.DataFrame(tab), rep, "cv_by_size",
             "Mean ± SD across folds. Threshold per CV scheme = Youden-J on that scheme's pooled out-of-fold "
             "predictions (slightly optimistic for threshold metrics; AUC/AP are threshold-free). "
             + ("ICH macro metrics exclude labels with no positives in a fold. " if ds == "ich" else ""))

    labels = None
    # --- ROC per model (100%) ---
    for m in models:
        p = os.path.join(ROOT, "results", ds, "scores", f"{m}_r100.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p); y, s = d["y"], d["s"]
        labels = json.load(open(os.path.join(ROOT, "results", ds, "split_info.json")))["labels"]
        fig, ax = plt.subplots(figsize=(4.6, 4.2))
        aucs = []
        for j, lab in enumerate(labels):
            fpr, tpr, _ = roc_curve(y[:, j], s[:, j]); a = roc_auc_score(y[:, j], s[:, j]); aucs.append(a)
            ax.plot(fpr, tpr, lw=2, color=PALETTE[j], label=f"{lab} ({a:.3f})" if len(labels) > 1 else f"AUC = {a:.3f}")
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#b5b4ad")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ttl = NAMES[ds][m] + (f" — macro AUC {np.mean(aucs):.3f}" if len(labels) > 1 else "")
        ax.set_title(ttl, fontsize=10.5, loc="left", color="#1f1f1d")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(rep, f"roc_{m}.png")); plt.close(fig)

    # --- ROC overlay, all models (binary target: SinoCT label / ICH 'any') ---
    fig, ax = plt.subplots(figsize=(5, 4.4)); drawn = False
    for m in models:
        p = os.path.join(ROOT, "results", ds, "scores", f"{m}_r100.npz")
        if not os.path.exists(p):
            continue
        d = np.load(p); y, s = d["y"][:, 0], d["s"][:, 0]
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2, color=COLOR[m], label=f"{NAMES[ds][m]} ({roc_auc_score(y, s):.3f})"); drawn = True
    if drawn:
        ax.plot([0, 1], [0, 1], ls="--", lw=1, color="#b5b4ad")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ax.set_title("ROC, all decoders" + (" ('any' hemorrhage)" if ds == "ich" else ""), loc="left", fontsize=10.5)
        ax.legend(loc="lower right", fontsize=8); fig.tight_layout()
        fig.savefig(os.path.join(rep, "roc_all_models.png"))
    plt.close(fig)

    # --- test AUC vs training size, all models, with CI band ---
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for m in models:
        r = sorted([x for x in test if x["model"] == m], key=lambda x: x["ratio"])
        if not r:
            continue
        xs = [x["ratio"] * 100 for x in r]
        ys = [x["overall"].get("auc", np.nan) for x in r]
        lo = [x["ci95"].get("auc", [np.nan, np.nan])[0] for x in r]
        hi = [x["ci95"].get("auc", [np.nan, np.nan])[1] for x in r]
        ax.fill_between(xs, lo, hi, color=COLOR[m], alpha=0.12, lw=0)
        ax.plot(xs, ys, lw=2, marker="o", ms=4, color=COLOR[m], label=NAMES[ds][m])
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set(xlabel="Training set (% of development pool, log scale)",
           ylabel="Test ROC AUC" + (" (macro, 6 labels)" if ds == "ich" else ""))
    ax.set_title("Held-out AUC vs training size (band = 95% bootstrap CI)", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, ncol=2, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(rep, "auc_vs_size.png")); plt.close(fig)

    # --- CV AUC by size per model (3/5/10 + test) ---
    for m in models:
        fig, ax = plt.subplots(figsize=(5.6, 3.8)); any_line = False
        for i, k in enumerate((3, 5, 10)):
            xs, mu, sd = [], [], []
            for ratio in sorted({r["ratio"] for r in rows if r["model"] == m}):
                f = [r["overall"].get("auc") for r in rows if r["model"] == m and r["ratio"] == ratio
                     and r["scheme"] == f"cv{k}" and isinstance(r.get("fold"), int)]
                f = np.array([v for v in f if v is not None], dtype=float)
                if len(f) > 1:
                    xs.append(ratio * 100); mu.append(f.mean()); sd.append(f.std(ddof=1))
            if xs:
                mu, sd = np.array(mu), np.array(sd)
                ax.fill_between(xs, mu - sd, mu + sd, color=PALETTE[i], alpha=0.12, lw=0)
                ax.plot(xs, mu, lw=2, marker="o", ms=4, color=PALETTE[i], label=f"{k}-fold CV (mean ± SD)")
                any_line = True
        r = sorted([x for x in test if x["model"] == m], key=lambda x: x["ratio"])
        if r:
            ax.plot([x["ratio"] * 100 for x in r], [x["overall"].get("auc") for x in r], lw=2, ls="--",
                    color="#3d3d3a", marker="s", ms=4, label="Held-out test")
        if any_line:
            ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
            ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
            ax.set(xlabel="Training set (% of development pool)", ylabel="ROC AUC")
            ax.set_title(f"{NAMES[ds][m]}: CV scheme × training size", loc="left", fontsize=10.5)
            ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
            fig.savefig(os.path.join(rep, f"cv_auc_{m}.png"))
        plt.close(fig)
    print("report written to", rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["ich", "sino"], required=True)
    main(ap.parse_args().dataset)
