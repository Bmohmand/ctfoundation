"""Aggregate results/<dataset>/*.jsonl into slide-ready tables and figures.

  python experiments/make_report.py --dataset ich
  python experiments/make_report.py --dataset sino

Writes results/<dataset>/report/:
  model_comparison.{csv,md}  test metrics at 100% (seed 0) with 95% bootstrap CIs, PR skill,
                             F1 at the F1-optimal threshold, and paired bootstrap ΔAUC vs LR (Holm)
  scaling_test.{csv,md}      test metrics vs training size; mean ± SD over seeds where repeated
  cv_by_size.{csv,md}        3/5/10-fold CV mean ± SD across folds at every size (seed 0)
  roc_<model>.png, roc_all_models.png, auc_vs_size.png, cv_auc_<model>.png
"""
import argparse
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling_cv import ROOT, bootstrap_indices  # noqa: E402

# validated categorical palette, fixed order (never cycled); legacy MLPs use neutral gray + dashes
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MAIN = ["LR", "SVM", "KNN", "RF", "XGB", "MLP", "MLP-focal", "MLP-bce"]
LEGACY = ["MLP3", "MLP5", "MLP7"]
GRAY = {"MLP3": "#5f5f5a", "MLP5": "#8a8984", "MLP7": "#3d3d3a"}
METRICS = ["auc", "ap", "acc", "sens", "spec", "f1"]
LABEL = {"auc": "ROC AUC", "ap": "PR AUC (AP)", "acc": "Accuracy", "sens": "Sensitivity",
         "spec": "Specificity", "f1": "F1", "prec": "Precision"}

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a95", "axes.labelcolor": "#3d3d3a", "xtick.color": "#5f5f5a",
                     "ytick.color": "#5f5f5a", "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
                     "legend.frameon": False})


def nice(ds, m):
    base, _, loss = m.partition("-")
    names = {"LR": "Logistic regression", "SVM": "Linear SVM" if ds == "ich" else "RBF SVM", "KNN": "k-NN (cosine)",
             "RF": "Random forest", "XGB": "XGBoost", "MLP": "MLP (tuned)",
             "MLP3": "MLP 3-layer (legacy)", "MLP5": "MLP 5-layer (legacy)", "MLP7": "MLP 7-layer (legacy)"}
    lossname = {"focal": "focal loss", "bce": "plain BCE", "wbce": "weighted BCE"}
    return names.get(base, base) + (f", {lossname.get(loss, loss)}" if loss else "")


def order(models):
    rank = {m: i for i, m in enumerate(MAIN + LEGACY)}
    return sorted(models, key=lambda m: (rank.get(m, rank.get(m.partition("-")[0], 99) + 0.5), m))


def style(m, i):
    base = m.partition("-")[0]
    if base in LEGACY:
        return {"color": GRAY.get(base, "#5f5f5a"), "ls": "--"}
    return {"color": PALETTE[i % len(PALETTE)], "ls": "-"}


def load(ds):
    rows = []
    for p in glob.glob(os.path.join(ROOT, "results", ds, "*.jsonl")):
        for line in open(p):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            r.setdefault("seed", 0)
            rows.append(r)
    return rows


def fmt(v, d=3):
    return "–" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def to_md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(out) + "\n"


def save(df, rep, name, note=""):
    df = df.astype(object).where(df.notna(), "–")
    df.to_csv(os.path.join(rep, f"{name}.csv"), index=False)
    with open(os.path.join(rep, f"{name}.md"), "w") as f:
        f.write(note + ("\n\n" if note else "") + to_md(df))


def score_file(ds, m, seed, pct):
    p = os.path.join(ROOT, "results", ds, "scores", f"{m}_s{seed}_r{pct:03d}.npz")
    return np.load(p) if os.path.exists(p) else None


def macro_auc(y, s):
    v = [roc_auc_score(y[:, j], s[:, j]) for j in range(y.shape[1]) if 0 < y[:, j].sum() < len(y)]
    return float(np.mean(v)) if v else np.nan


def macro_skill(y, s):
    """Prevalence-normalized PR AUC: (AP - prevalence) / (1 - prevalence); 0 = chance, 1 = perfect."""
    v = []
    for j in range(y.shape[1]):
        pi = y[:, j].mean()
        if 0 < pi < 1:
            v.append((average_precision_score(y[:, j], s[:, j]) - pi) / (1 - pi))
    return float(np.mean(v)) if v else np.nan


def holm(p):
    p = np.asarray(p, float); idx = np.argsort(p); adj = np.empty_like(p); run = 0
    for r, i in enumerate(idx):
        run = max(run, (len(p) - r) * p[i]); adj[i] = min(1, run)
    return adj


def paired_vs_ref(ds, models, ref="LR", n_boot=1000):
    ref_d = score_file(ds, ref, 0, 100)
    if ref_d is None:
        return {}
    y = ref_d["y"]; boots = bootstrap_indices(y, n_boot)
    ref_s = ref_d["s"]; out = {}
    for m in models:
        d = score_file(ds, m, 0, 100)
        if m == ref or d is None:
            continue
        delta = np.array([macro_auc(y[b], d["s"][b]) - macro_auc(y[b], ref_s[b]) for b in boots])
        p = min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
        out[m] = (macro_auc(y, d["s"]) - macro_auc(y, ref_s), np.percentile(delta, 2.5), np.percentile(delta, 97.5), p)
    if out:
        adj = holm([v[3] for v in out.values()])
        out = {m: (*v[:3], v[3], a) for (m, v), a in zip(out.items(), adj)}
    return out


def main(ds):
    rows = load(ds)
    rep = os.path.join(ROOT, "results", ds, "report")
    os.makedirs(rep, exist_ok=True)
    models = order({r["model"] for r in rows})
    test = [r for r in rows if r["scheme"] == "test"]
    test0 = [r for r in test if r["seed"] == 0]
    info = json.load(open(os.path.join(ROOT, "results", ds, "split_info.json")))
    labels = info["labels"]
    multi = len(labels) > 1

    # ---------------------------------------------------------------- model comparison at 100%
    paired = paired_vs_ref(ds, models)
    tab = []
    for m in models:
        r = [x for x in test0 if x["model"] == m and abs(x["ratio"] - 1) < 1e-9]
        if not r:
            continue
        r = r[0]; o, ci = r["overall"], r["ci95"]
        row = {"Model": nice(ds, m), "n_train": r["n_train"]}
        for k in METRICS:
            lo, hi = ci.get(k, [None, None])
            row[LABEL[k]] = f"{fmt(o.get(k))} [{fmt(lo)}–{fmt(hi)}]"
        d = score_file(ds, m, 0, 100)
        row["PR skill"] = fmt(macro_skill(d["y"], d["s"])) if d is not None else "–"
        row["F1 @ F1-opt thr"] = fmt(r.get("overall_f1thr", {}).get("f1"))
        if m in paired:
            dlt, lo, hi, p, padj = paired[m]
            row["ΔAUC vs LR [95% CI]"] = f"{dlt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
            row["p (Holm)"] = f"{padj:.3f}" if padj >= 0.001 else "<0.001"
        else:
            row["ΔAUC vs LR [95% CI]"] = "reference" if m == "LR" else "–"
            row["p (Holm)"] = "–"
        tab.append(row)
    if tab:
        save(pd.DataFrame(tab), rep, "model_comparison",
             "Fixed held-out test set; models trained on 100% of the development pool (seed 0). "
             "Brackets = 95% stratified bootstrap CI (1000 resamples). Threshold metrics use the mean Youden-J "
             "threshold from 3/5/10-fold out-of-fold predictions; 'F1 @ F1-opt thr' uses the F1-maximizing "
             "out-of-fold threshold instead. PR skill = (AP − prevalence)/(1 − prevalence). ΔAUC = paired "
             "bootstrap difference vs logistic regression on the same resamples, two-sided p, Holm-adjusted."
             + (" Macro averages over the 6 ICH labels." if multi else ""))

    # ---------------------------------------------------------------- scaling (test), seeds aggregated
    tab = []
    for m in models:
        for ratio in sorted({r["ratio"] for r in test if r["model"] == m}):
            rs = [r for r in test if r["model"] == m and r["ratio"] == ratio]
            r0 = [r for r in rs if r["seed"] == 0]
            base = r0[0] if r0 else rs[0]
            row = {"Model": nice(ds, m), "Train %": int(round(ratio * 100)), "n": base["n_train"],
                   "n_pos": base["n_pos_train"], "seeds": len(rs)}
            for k in METRICS:
                v = np.array([r["overall"].get(k) for r in rs], dtype=float)
                v = v[~np.isnan(v)]
                row[LABEL[k]] = (f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else fmt(v[0] if len(v) else None))
            lo, hi = base["ci95"].get("auc", [None, None])
            row["AUC 95% CI (seed 0)"] = f"{fmt(lo)}–{fmt(hi)}"
            if "C" in base:
                row["C"] = base["C"]
            if "knn_k" in base:
                row["k"] = base["knn_k"]
            tab.append(row)
    if tab:
        save(pd.DataFrame(tab), rep, "scaling_test",
             "Held-out test performance by training-set size (% of the development pool). Subsets are nested. "
             "Where seeds > 1, values are mean ± SD over independent nested-subset draws (same test set). "
             "C (LR/SVM) is re-selected on each training subset; k (k-NN) inside every fit.")

    # ---------------------------------------------------------------- CV by size (seed 0)
    tab = []
    for m in models:
        for ratio in sorted({r["ratio"] for r in rows if r["model"] == m and r["seed"] == 0}):
            for k in (3, 5, 10):
                fr = [r for r in rows if r["model"] == m and r["ratio"] == ratio and r["seed"] == 0
                      and r["scheme"] == f"cv{k}"]
                row = {"Model": nice(ds, m), "Train %": int(round(ratio * 100)), "Folds": k}
                folds = [r for r in fr if isinstance(r.get("fold"), int)]
                for x in METRICS:
                    v = np.array([f["overall"].get(x) for f in folds], dtype=float)
                    v = v[~np.isnan(v)]
                    row[LABEL[x]] = f"{v.mean():.3f} ± {v.std(ddof=1):.3f}" if len(v) > 1 else "NA"
                tab.append(row)
    if tab:
        save(pd.DataFrame(tab), rep, "cv_by_size",
             "Mean ± SD across folds (seed 0). Each fold's threshold comes from the OTHER folds' out-of-fold "
             "predictions (Youden J). NA = fewer positives than folds."
             + (" ICH macro metrics exclude labels with no positives in a fold." if multi else ""))

    # ---------------------------------------------------------------- ROC per model (100%)
    for m in models:
        d = score_file(ds, m, 0, 100)
        if d is None:
            continue
        y, s = d["y"], d["s"]
        fig, ax = plt.subplots(figsize=(4.6, 4.2)); aucs = []
        for j, lab in enumerate(labels):
            fpr, tpr, _ = roc_curve(y[:, j], s[:, j]); a = roc_auc_score(y[:, j], s[:, j]); aucs.append(a)
            ax.plot(fpr, tpr, lw=2, color=PALETTE[j], label=f"{lab} ({a:.3f})" if multi else f"AUC = {a:.3f}")
        ax.plot([0, 1], [0, 1], ls=":", lw=1, color="#b5b4ad")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ax.set_title(nice(ds, m) + (f" — macro AUC {np.mean(aucs):.3f}" if multi else ""), fontsize=10, loc="left")
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(rep, f"roc_{m}.png")); plt.close(fig)

    # ---------------------------------------------------------------- ROC overlay (first label)
    fig, ax = plt.subplots(figsize=(5.2, 4.6)); i = 0; drawn = False
    for m in models:
        d = score_file(ds, m, 0, 100)
        if d is None:
            continue
        y, s = d["y"][:, 0], d["s"][:, 0]; st = style(m, i); i += m.partition("-")[0] not in LEGACY
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, lw=2, **st, label=f"{nice(ds, m)} ({roc_auc_score(y, s):.3f})"); drawn = True
    if drawn:
        ax.plot([0, 1], [0, 1], ls=":", lw=1, color="#b5b4ad")
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1.01))
        ax.set_title("ROC, all decoders" + (" ('any' hemorrhage)" if multi else " (abnormal)"), loc="left", fontsize=10.5)
        ax.legend(loc="lower right", fontsize=7.5); fig.tight_layout()
        fig.savefig(os.path.join(rep, "roc_all_models.png"))
    plt.close(fig)

    # ---------------------------------------------------------------- test AUC vs size (mean over seeds)
    fig, ax = plt.subplots(figsize=(6.6, 4.2)); i = 0
    for m in models:
        xs, mu, lo, hi = [], [], [], []
        for ratio in sorted({r["ratio"] for r in test if r["model"] == m}):
            rs = [r for r in test if r["model"] == m and r["ratio"] == ratio]
            v = np.array([r["overall"].get("auc", np.nan) for r in rs], float)
            xs.append(ratio * 100); mu.append(np.nanmean(v))
            if len(v) > 1:  # spread across seeds
                lo.append(np.nanmin(v)); hi.append(np.nanmax(v))
            else:
                c = rs[0]["ci95"].get("auc", [np.nan, np.nan]); lo.append(c[0]); hi.append(c[1])
        if not xs:
            continue
        st = style(m, i); i += m.partition("-")[0] not in LEGACY
        ax.fill_between(xs, lo, hi, color=st["color"], alpha=0.10, lw=0)
        ax.plot(xs, mu, lw=2, marker="o", ms=4, **st, label=nice(ds, m))
    ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100]); ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
    ax.set(xlabel="Training set (% of development pool, log scale)",
           ylabel="Test ROC AUC" + (" (macro, 6 labels)" if multi else ""))
    ax.set_title("Held-out AUC vs training size (band: seed range where repeated, else 95% CI)", loc="left", fontsize=10)
    ax.legend(fontsize=7.5, ncol=2, loc="lower right"); fig.tight_layout()
    fig.savefig(os.path.join(rep, "auc_vs_size.png")); plt.close(fig)

    # ---------------------------------------------------------------- CV AUC by size per model
    for m in models:
        fig, ax = plt.subplots(figsize=(5.6, 3.8)); any_line = False
        for i, k in enumerate((3, 5, 10)):
            xs, mu, sd = [], [], []
            for ratio in sorted({r["ratio"] for r in rows if r["model"] == m and r["seed"] == 0}):
                f = [r["overall"].get("auc") for r in rows if r["model"] == m and r["ratio"] == ratio and r["seed"] == 0
                     and r["scheme"] == f"cv{k}" and isinstance(r.get("fold"), int)]
                f = np.array([v for v in f if v is not None], float)
                if len(f) > 1:
                    xs.append(ratio * 100); mu.append(f.mean()); sd.append(f.std(ddof=1))
            if xs:
                mu, sd = np.array(mu), np.array(sd)
                ax.fill_between(xs, mu - sd, mu + sd, color=PALETTE[i], alpha=0.12, lw=0)
                ax.plot(xs, mu, lw=2, marker="o", ms=4, color=PALETTE[i], label=f"{k}-fold CV (mean ± SD)")
                any_line = True
        r = sorted([x for x in test0 if x["model"] == m], key=lambda x: x["ratio"])
        if r:
            ax.plot([x["ratio"] * 100 for x in r], [x["overall"].get("auc") for x in r], lw=2, ls="--",
                    color="#3d3d3a", marker="s", ms=4, label="Held-out test")
        if any_line:
            ax.set_xscale("log"); ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
            ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
            ax.set(xlabel="Training set (% of development pool)", ylabel="ROC AUC")
            ax.set_title(f"{nice(ds, m)}: CV scheme × training size", loc="left", fontsize=10)
            ax.legend(fontsize=8, loc="lower right"); fig.tight_layout()
            fig.savefig(os.path.join(rep, f"cv_auc_{m}.png"))
        plt.close(fig)
    print("report written to", rep)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", choices=["ich", "sino"], required=True)
    main(ap.parse_args().dataset)
