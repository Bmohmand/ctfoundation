"""Aside: how much of the CT-FM embedding does a classifier actually need?

Compress the 1,408-dim embeddings to their top-k principal components and fit the same logistic
regression on each. If a few dozen components match the full vector, the embedding's usable signal
is low-dimensional and close to linearly separable, which is why every decoder lands at the same AUC.

  python experiments/pca_probe.py --dataset sino
  python experiments/pca_probe.py --dataset ich

Uses the same de-duplicated data and the same fixed 20% test set as scaling_cv.py. The scaler and PCA
are fit on the development pool only (never the test set). Classifier: logistic regression, C=0.1,
class_weight="balanced", one per label. Writes results/<dataset>/report/pca_probe.{csv,md,png}.
"""
import argparse
import os
import sys
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scaling_cv import ROOT, load_dataset, make_splits  # noqa: E402

warnings.filterwarnings("ignore")
KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
BLUE, ORANGE, GREEN, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#1f1f1d", "#5f5f5a"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#9a9a95", "axes.labelcolor": "#3d3d3a", "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.8,
                     "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.dpi": 200,
                     "legend.frameon": False})


def main(ds):
    X, Y, labels, _ = load_dataset(ds)
    dev, test = make_splits(Y)
    sc = StandardScaler().fit(X[dev])
    Xd, Xt = sc.transform(X[dev]), sc.transform(X[test])
    pca = PCA(random_state=0).fit(Xd)
    cum = np.cumsum(pca.explained_variance_ratio_)
    Zd, Zt = pca.transform(Xd), pca.transform(Xt)

    rows = []
    for k in KS + [X.shape[1]]:
        a, b = (Zd[:, :k], Zt[:, :k]) if k < X.shape[1] else (Xd, Xt)
        aucs = {}
        for j, lab in enumerate(labels):
            m = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000).fit(a, Y[dev, j])
            aucs[lab] = roc_auc_score(Y[test, j], m.predict_proba(b)[:, 1])
        rows.append({"components": k, "variance_explained": float(cum[min(k, len(cum)) - 1]),
                     "macro_auc": float(np.mean(list(aucs.values()))), **aucs})
        print(f"[{ds}] k={k:>4}  variance {rows[-1]['variance_explained']:.3f}  test AUC {rows[-1]['macro_auc']:.3f}",
              flush=True)
    df = pd.DataFrame(rows)

    rep = os.path.join(ROOT, "results", ds, "report")
    os.makedirs(rep, exist_ok=True)
    df.round(4).to_csv(os.path.join(rep, "pca_probe.csv"), index=False, encoding="utf-8-sig")
    full = df.iloc[-1]["macro_auc"]
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    k_match = int(df[df["macro_auc"] >= full - 0.005]["components"].min())
    with open(os.path.join(rep, "pca_probe.md"), "w", encoding="utf-8") as f:
        f.write(f"Logistic regression on the top-k principal components (fit on the development pool), scored on the "
                f"fixed test set. {k95} of {X.shape[1]} components explain 95% of the variance; {k_match} components "
                f"reach within 0.005 AUC of the full embedding ({full:.3f}).\n\n")
        cols = ["components", "variance_explained", "macro_auc"] + (labels if len(labels) > 1 else [])
        f.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for _, r in df.iterrows():
            f.write("| " + " | ".join(f"{int(r[c])}" if c == "components" else f"{r[c]:.3f}" for c in cols) + " |\n")

    # figure: two panels sharing the component axis (no dual y-axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.1))
    ks = df["components"].values
    if len(labels) > 1:
        ax1.plot(ks, df["any"], lw=2, marker="o", ms=4, color=BLUE, label="'any' hemorrhage")
        ax1.plot(ks, df["macro_auc"], lw=2, marker="o", ms=4, color=ORANGE, label="macro (6 labels)")
        ax1.plot(ks, df["epidural"], lw=2, marker="o", ms=4, color=GREEN, label="epidural (rarest)")
        ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8)
    else:
        ax1.plot(ks, df["macro_auc"], lw=2, marker="o", ms=4, color=BLUE)
    ax1.axvline(k95, color=MUTED, ls=":", lw=1)
    ax1.text(k95 * 1.08, ax1.get_ylim()[0] + 0.01 * np.ptp(ax1.get_ylim()), "95% variance", color=MUTED, fontsize=8)
    ax1.set(xscale="log", xlabel="Principal components used (log scale)", ylabel="Test ROC AUC")
    ax1.set_title("Logistic regression on top-k components", loc="left", fontsize=10, color=INK)
    ax2.plot(np.arange(1, len(cum) + 1), cum, lw=2, color=BLUE)
    ax2.axhline(0.95, color=MUTED, ls=":", lw=1)
    ax2.set(xscale="log", xlabel="Principal components (log scale)", ylabel="Cumulative variance explained",
            ylim=(0, 1.01))
    ax2.set_title(f"{k95} of {X.shape[1]} components hold 95% of the variance", loc="left", fontsize=10, color=INK)
    for ax in (ax1, ax2):
        ax.set_xticks([1, 4, 16, 64, 256, 1408]); ax.set_xticklabels(["1", "4", "16", "64", "256", "1408"])
    fig.suptitle(f"{'SinoCT' if ds == 'sino' else 'RSNA ICH'}: the embedding's usable signal is low-dimensional",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(rep, "pca_probe.png"))
    print(f"[{ds}] wrote {rep}/pca_probe.(csv|md|png)  | 95% variance: {k95} comps | within 0.005 of full: {k_match}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ich", "ich_study", "sino"], required=True)
    main(ap.parse_args().dataset)
