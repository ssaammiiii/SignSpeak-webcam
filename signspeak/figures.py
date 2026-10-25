"""
Generate report-ready figures from the collected datasets.

Re-runs the same stratified cross-validation as training (so the numbers in
the figures match the leaderboard you saw) and saves, per dataset:

  figures/<kind>_leaderboard.png   - CV accuracy per model, mean +/- std error bars
  figures/<kind>_confusion.png     - winner's confusion matrix (held-out predictions)
  figures/<kind>_per_class_f1.png  - winner's per-class F1, weakest classes highlighted

Run AFTER collecting data (models need not exist; CV is recomputed):

    uv run figures.py

Takes a few minutes for the static dataset (same cost as train.py's
comparison). Requires matplotlib (`uv add matplotlib`).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .training import _cv, candidates, feature_frame

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit("matplotlib is required for figures: uv add matplotlib") from e

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import cross_val_predict, cross_val_score

FIG_DIR = config.PROJECT_ROOT / "figures"

# Consistent, report-friendly styling.
BAR_COLOR = "#4878b0"
WIN_COLOR = "#d1495b"
plt.rcParams.update({"figure.dpi": 130, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def _benchmark(X, y, kind):
    """Same contest as training: every candidate + ensemble of the top 3.
    Returns {name: (mean, std)} and the winner's (name, unfitted model)."""
    cv = _cv(y)
    scores, order = {}, []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, model in candidates().items():
            r = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
            scores[name] = (r.mean(), r.std())
            order.append(name)
            print(f"[{kind}] {name:<12} {r.mean():.3f} ± {r.std():.3f}")
        top = sorted(scores, key=lambda n: scores[n][0], reverse=True)[:3]
        fresh = candidates()
        ens = VotingClassifier([(n, fresh[n]) for n in top], voting="soft")
        r = cross_val_score(ens, X, y, cv=cv, scoring="accuracy")
        ens_name = "ensemble\n(" + "+".join(top) + ")"
        scores[ens_name] = (r.mean(), r.std())
        order.append(ens_name)
        print(f"[{kind}] {ens_name.replace(chr(10), ' '):<12} {r.mean():.3f} ± {r.std():.3f}")

    best = max(scores, key=lambda n: scores[n][0])
    best_model = ens if best == ens_name else candidates()[best]
    return scores, order, best, best_model


def leaderboard_figure(scores, order, best, kind):
    names = order
    means = [scores[n][0] for n in names]
    stds = [scores[n][1] for n in names]
    colors = [WIN_COLOR if n == best else BAR_COLOR for n in names]

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bars = ax.bar(range(len(names)), means, yerr=stds, capsize=4,
                  color=colors, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=8)
    lo = max(0.0, min(means) - max(stds) - 0.03)
    ax.set_ylim(lo, 1.005)
    ax.set_ylabel("Cross-validated accuracy")
    ax.set_title(f"{kind.capitalize()} model comparison "
                 f"({_cv_desc(kind)}; error bars = ±1 std across folds)")
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 0.002, f"{m:.3f}",
                ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout()
    out = FIG_DIR / f"{kind}_leaderboard.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


_CV_DESCS = {}
def _cv_desc(kind):  # filled in main() once we know fold counts
    return _CV_DESCS.get(kind, "stratified k-fold CV")


def confusion_figure(model, X, y, kind):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred = cross_val_predict(model, X, y, cv=_cv(y))
    labels_sorted = sorted(set(y))
    cm = confusion_matrix(y, y_pred, labels=labels_sorted)

    n = len(labels_sorted)
    fig, ax = plt.subplots(figsize=(max(5, n * 0.32), max(4.4, n * 0.3)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels_sorted, fontsize=7, rotation=90)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels_sorted, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"{kind.capitalize()} winner - confusion matrix "
                 f"(every sample predicted while held out)")
    # annotate non-zero cells; off-diagonal errors in red
    for i in range(n):
        for j in range(n):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6.5,
                        color=("white" if cm[i, j] > cm.max() * 0.6
                               else (WIN_COLOR if i != j else "black")))
    fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    out = FIG_DIR / f"{kind}_confusion.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")
    return y_pred


def per_class_f1_figure(y, y_pred, kind):
    labels_sorted = sorted(set(y))
    f1 = f1_score(y, y_pred, labels=labels_sorted, average=None)
    order = np.argsort(f1)
    names = [labels_sorted[i] for i in order]
    vals = f1[order]
    colors = [WIN_COLOR if v < 0.98 else BAR_COLOR for v in vals]

    fig, ax = plt.subplots(figsize=(7.5, max(3.2, len(names) * 0.22)))
    ax.barh(range(len(names)), vals, color=colors, edgecolor="white")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlim(max(0.0, vals.min() - 0.05), 1.0)
    ax.set_xlabel("F1 score (cross-validated)")
    ax.set_title(f"{kind.capitalize()} per-class F1 - weakest at top "
                 f"(red: below 0.98)")
    for i, v in enumerate(vals):
        ax.text(v - 0.001, i, f"{v:.2f}", ha="right", va="center",
                fontsize=6.5, color="white")
    fig.tight_layout()
    out = FIG_DIR / f"{kind}_per_class_f1.png"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(f"saved {out}")


def run_for(csv_path, expected_features_prefix, kind):
    if not csv_path.exists():
        print(f"[{kind}] no dataset at {csv_path} - skipped.")
        return
    df = pd.read_csv(csv_path)
    X = feature_frame(df).values
    y = df["label"].values
    _CV_DESCS[kind] = f"{_cv(y).get_n_splits()}-fold stratified CV, n={len(df)}"
    print(f"\n[{kind}] {len(df)} samples, {df['label'].nunique()} classes - benchmarking...")
    scores, order, best, best_model = _benchmark(X, y, kind)
    leaderboard_figure(scores, order, best, kind)
    y_pred = confusion_figure(best_model, X, y, kind)
    per_class_f1_figure(y, y_pred, kind)


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Generate report figures.")
    p.add_argument("--only", choices=["static", "motion", "word", "onehand"],
                   help="make figures for just this dataset")
    args = p.parse_args(argv)

    FIG_DIR.mkdir(exist_ok=True)
    jobs = {
        "static": (config.STATIC_DATASET_CSV, "f"),
        "motion": (config.MOTION_DATASET_CSV, "m"),
        "word": (config.WORD_DATASET_CSV, "w"),
        "onehand": (config.ONEHAND_DATASET_CSV, "o"),
    }
    for kind, (csv_path, prefix) in jobs.items():
        if args.only in (None, kind):
            run_for(csv_path, prefix, kind)
    print(f"\nAll figures in {FIG_DIR}")


if __name__ == "__main__":
    main()