"""
Write report-style figure files (PNG + PDF) for PlasticDeg ESM + sklearn heads.

Figures mirror common thesis layouts: confusion (counts + row-normalized), ROC/PR,
per-class precision/recall/F1, score histograms by true class, RF feature-importance
views, and an RF **ensemble-growth** panel (accuracy + log-loss vs. number of trees
with out-of-bag score) — analogous to training curves for deep nets but technically
correct for this pipeline (no epoch logs from frozen ESM).

Requires trained v2 artifacts: embeddings .npz, split lists, tier CSV, metrics JSON,
and joblib models produced by ``plasticdeg.train.train_esm_baseline_v2``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.utils.class_weight import compute_sample_weight

from plasticdeg import paths
from plasticdeg.train.train_esm_baseline import label_from_split_id, load_split_ids
from plasticdeg.train.train_esm_baseline_v2 import load_tier_map, tier_for_split_id


def _build_xy(
    id_path: Path,
    id_to_row: dict[str, int],
    X_all: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    wanted = load_split_ids(id_path)
    rows: list[int] = []
    ylist: list[int] = []
    used: list[str] = []
    for sid in sorted(wanted):
        r = id_to_row.get(sid)
        if r is None:
            continue
        rows.append(r)
        ylist.append(label_from_split_id(sid))
        used.append(sid)
    if not rows:
        return np.empty((0, X_all.shape[1])), np.asarray([], dtype=int), []
    return X_all[rows], np.asarray(ylist, dtype=int), used


def _grid_factors(n: int) -> tuple[int, int]:
    """Height x width for reshaping 1D importances to a rectangle (row-major)."""
    s = int(math.isqrt(n))
    for h in range(s, 0, -1):
        if n % h == 0:
            return h, n // h
    return 1, n


def _save_fig(fig: Any, out_base: Path, formats: list[str]) -> None:
    import matplotlib.pyplot as plt

    out_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        p = out_base.with_suffix(f".{fmt}")
        fig.savefig(p, dpi=220, bbox_inches="tight", format=fmt)
        print(f"  Wrote {p}", flush=True)
    plt.close(fig)


def _plot_confusion_pair(
    cm: np.ndarray,
    class_names: tuple[str, str],
    title: str,
    out_base: Path,
    formats: list[str],
) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=axes[0],
        cbar_kws={"label": "Count"},
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title("Raw counts")

    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        ax=axes[1],
        cbar_kws={"label": "Recall (row norm)"},
        xticklabels=class_names,
        yticklabels=class_names,
    )
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")
    axes[1].set_title("Normalised (per true class)")
    fig.suptitle(title, fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_base, formats)


def _plot_roc_pr(
    y_test: np.ndarray,
    prob_rf: np.ndarray,
    prob_lr: np.ndarray,
    roc_base: Path,
    pr_base: Path,
    formats: list[str],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    fpr, tpr, _ = roc_curve(y_test, prob_rf)
    auc_rf = roc_auc_score(y_test, prob_rf)
    ax.plot(fpr, tpr, label=f"Random forest (AUC = {auc_rf:.4f})", linewidth=2)
    fpr2, tpr2, _ = roc_curve(y_test, prob_lr)
    auc_lr = roc_auc_score(y_test, prob_lr)
    ax.plot(fpr2, tpr2, label=f"Logistic regression (AUC = {auc_lr:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.45, label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC — test set (positive = polyester enzyme)")
    ax.set_xlim(-0.02, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _save_fig(fig, roc_base, formats)

    fig2, ax2 = plt.subplots(figsize=(6.2, 5.4))
    p1, r1, _ = precision_recall_curve(y_test, prob_rf)
    pr_rf = average_precision_score(y_test, prob_rf)
    ax2.plot(r1, p1, label=f"Random forest (PR-AUC = {pr_rf:.4f})", linewidth=2)
    p2, r2, _ = precision_recall_curve(y_test, prob_lr)
    pr_lr = average_precision_score(y_test, prob_lr)
    ax2.plot(r2, p2, label=f"Logistic regression (PR-AUC = {pr_lr:.4f})", linewidth=2)
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision–recall — test set")
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.35)
    ax2.legend(loc="upper right", fontsize=9)
    fig2.tight_layout()
    _save_fig(fig2, pr_base, formats)


def _plot_grouped_prf1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: tuple[str, str],
    title: str,
    out_base: Path,
    formats: list[str],
) -> None:
    import matplotlib.pyplot as plt

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    macro_f1 = float(np.mean(f1))
    x = np.arange(len(class_names))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(x - w, prec, width=w, label="Precision", color="#1f77b4")
    ax.bar(x, rec, width=w, label="Recall", color="#2ca02c")
    ax.bar(x + w, f1, width=w, label="F1-score", color="#d62728")
    for i, xi in enumerate(x):
        ax.text(xi - w, prec[i] + 0.02, f"{prec[i]:.2f}", ha="center", fontsize=8)
        ax.text(xi, rec[i] + 0.02, f"{rec[i]:.2f}", ha="center", fontsize=8)
        ax.text(xi + w, f1[i] + 0.02, f"{f1[i]:.2f}", ha="center", fontsize=8)
    ax.axhline(macro_f1, color="gray", linestyle="--", linewidth=1, label=f"Macro F1 = {macro_f1:.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.12)
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    _save_fig(fig, out_base, formats)


def _plot_confidence_grid(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    threshold: float,
    model_name: str,
    class_names: tuple[str, str],
    out_base: Path,
    formats: list[str],
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    bins = np.linspace(0, 1, 22)
    for ax, true_label, name in zip(
        axes,
        (0, 1),
        (f"True: {class_names[0]}", f"True: {class_names[1]}"),
    ):
        m = y_true == true_label
        correct = m & (y_pred == y_true)
        wrong = m & (y_pred != y_true)
        ax.hist(
            y_prob[correct],
            bins=bins,
            alpha=0.75,
            color="#2ca02c",
            label="Correct",
        )
        ax.hist(y_prob[wrong], bins=bins, alpha=0.75, color="#d62728", label="Wrong")
        ax.axvline(threshold, color="k", linestyle="--", alpha=0.55, label="Threshold")
        ax.set_xlabel("Predicted probability (positive)")
        ax.set_ylabel("Count")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(f"{model_name} — confidence by true class (test)", fontsize=12, y=1.02)
    fig.tight_layout()
    _save_fig(fig, out_base, formats)


def _plot_rf_importance_maps(
    importances: np.ndarray,
    out_heatmap: Path,
    out_profile: Path,
    formats: list[str],
) -> None:
    import matplotlib.pyplot as plt

    n = int(importances.size)
    h, w = _grid_factors(n)
    grid = importances.reshape(h, w)
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("Feature index (2-D layout of ESM dimensions)")
    ax.set_ylabel("Feature index (2-D layout)")
    ax.set_title("Random forest — normalised feature importances (embedding dims)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Importance")
    fig.tight_layout()
    _save_fig(fig, out_heatmap, formats)

    fig2, ax2 = plt.subplots(figsize=(8.5, 4.2))
    ax2.plot(np.arange(n), importances, color="#1f77b4", linewidth=0.9)
    ax2.set_xlabel("ESM embedding dimension index")
    ax2.set_ylabel("Gini importance")
    ax2.set_title("Random forest — importance profile across channels")
    ax2.grid(True, alpha=0.35)
    fig2.tight_layout()
    _save_fig(fig2, out_profile, formats)


def _plot_rf_ensemble_progress(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int,
    tree_steps: list[int],
    out_base: Path,
    formats: list[str],
) -> None:
    import matplotlib.pyplot as plt

    train_acc: list[float] = []
    oob_scores: list[float] = []
    train_loss: list[float] = []
    rf = RandomForestClassifier(
        n_estimators=1,
        max_depth=16,
        min_samples_leaf=2,
        warm_start=True,
        oob_score=True,
        bootstrap=True,
        random_state=seed,
        n_jobs=-1,
    )
    for n in tree_steps:
        rf.set_params(n_estimators=n)
        rf.fit(X_train, y_train, sample_weight=sample_weight)
        pred_tr = rf.predict(X_train)
        train_acc.append(float(accuracy_score(y_train, pred_tr)))
        oob = rf.oob_score_
        if oob is None:
            oob_scores.append(float("nan"))
        else:
            oob_scores.append(float(oob))
        proba = rf.predict_proba(X_train)
        train_loss.append(float(log_loss(y_train, proba, labels=[0, 1])))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.6))
    ax1.plot(tree_steps, train_acc, "o-", color="#1f77b4", label="Train accuracy", linewidth=1.5)
    ax1.plot(tree_steps, oob_scores, "s-", color="#d62728", label="OOB accuracy", linewidth=1.5)
    ax1.set_xlabel("Number of trees in ensemble")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("RF head — accuracy vs. ensemble size")
    ax1.grid(True, alpha=0.35)
    ax1.legend(loc="lower right")

    ax2.plot(tree_steps, train_loss, "o-", color="#ff7f0e", linewidth=1.5)
    ax2.set_xlabel("Number of trees in ensemble")
    ax2.set_ylabel("Log loss (train)")
    ax2.set_title("RF head — training log-loss vs. ensemble size")
    ax2.grid(True, alpha=0.35)

    fig.suptitle(
        "Ensemble growth (warm start; OOB ≈ generalisation on bagged out-of-bag samples). "
        "Not equivalent to neural-network epochs.",
        fontsize=10,
        y=1.05,
    )
    fig.tight_layout()
    _save_fig(fig, out_base, formats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export manuscript-style figure bundle (PNG/PDF)")
    parser.add_argument("--embeddings", type=Path, default=paths.embeddings_esm2_t33_mean_v2_npz())
    parser.add_argument("--train-ids", type=Path, default=paths.split_train_txt())
    parser.add_argument("--test-ids", type=Path, default=paths.split_test_txt())
    parser.add_argument("--tier-csv", type=Path, default=paths.positives_gt_expanded_csv())
    parser.add_argument("--metrics-json", type=Path, default=paths.metrics_esm_baseline_v2_json())
    parser.add_argument("--lr-joblib", type=Path, default=paths.model_lr_esm_baseline_v2_joblib())
    parser.add_argument("--rf-joblib", type=Path, default=paths.model_rf_esm_baseline_v2_joblib())
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: data/processed_v2/reports/figure_bundle)",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default="png,pdf",
        help="Comma-separated: png, pdf, svg",
    )
    parser.add_argument(
        "--skip-ensemble-curve",
        action="store_true",
        help="Skip RF warm-start re-fit (saves time if you only need confusion/ROC/etc.).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the optional RF ensemble-growth refit (match training).",
    )
    args = parser.parse_args(argv)

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    out_dir = args.out_dir or paths.report_figure_bundle_dir()

    required = [
        args.embeddings,
        args.train_ids,
        args.test_ids,
        args.metrics_json,
        args.lr_joblib,
        args.rf_joblib,
    ]
    for p in required:
        if not p.exists():
            print(f"ERROR: missing required file: {p}", file=sys.stderr)
            return 1
    if not args.tier_csv.exists():
        print(f"ERROR: missing tier CSV: {args.tier_csv}", file=sys.stderr)
        return 1

    metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))
    models_meta = metrics.get("models") or {}
    lr_thr = float(models_meta["logistic_regression"]["threshold_train_recall_ge_0.8"])
    rf_thr = float(models_meta["random_forest"]["threshold_train_recall_ge_0.8"])

    tier_map = load_tier_map(args.tier_csv)
    data = np.load(args.embeddings, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    X_all = np.asarray(data["embeddings"], dtype=np.float32)
    id_to_row = {i: r for r, i in enumerate(ids)}

    X_train, y_train, train_used = _build_xy(args.train_ids, id_to_row, X_all)
    X_test, y_test, _test_used = _build_xy(args.test_ids, id_to_row, X_all)
    if len(X_test) == 0 or len(X_train) == 0:
        print("ERROR: empty train or test after aligning with embeddings.", file=sys.stderr)
        return 1

    lr_pack = joblib.load(args.lr_joblib)
    scaler = lr_pack["scaler"]
    lr = lr_pack["model"]
    rf: RandomForestClassifier = joblib.load(args.rf_joblib)

    X_tr_s = scaler.transform(X_train)
    X_te_s = scaler.transform(X_test)

    lr_prob_te = lr.predict_proba(X_te_s)[:, 1]
    rf_prob_te = rf.predict_proba(X_test)[:, 1]
    lr_pred_te = (lr_prob_te >= lr_thr).astype(int)
    rf_pred_te = (rf_prob_te >= rf_thr).astype(int)

    tier1_w = float(metrics.get("tier1_weight", 1.0))
    tier2_w = float(metrics.get("tier2_weight", 0.5))
    sw = compute_sample_weight("balanced", y_train).astype(np.float64, copy=True)
    for i, sid in enumerate(train_used):
        if y_train[i] != 1:
            continue
        t = tier_for_split_id(sid, tier_map)
        sw[i] *= tier2_w if t == 2 else tier1_w

    class_names = ("Negative (non-hit)", "Positive (enzyme)")
    cm_rf = confusion_matrix(y_test, rf_pred_te, labels=[0, 1])
    cm_lr = confusion_matrix(y_test, lr_pred_te, labels=[0, 1])

    _plot_confusion_pair(
        cm_rf,
        class_names,
        "Confusion matrix — Random forest (test)",
        out_dir / "fig_confusion_matrix_rf",
        formats,
    )
    _plot_confusion_pair(
        cm_lr,
        class_names,
        "Confusion matrix — Logistic regression (test)",
        out_dir / "fig_confusion_matrix_lr",
        formats,
    )
    _plot_roc_pr(
        y_test,
        rf_prob_te,
        lr_prob_te,
        out_dir / "fig_roc_test",
        out_dir / "fig_pr_test",
        formats,
    )
    _plot_grouped_prf1(
        y_test,
        rf_pred_te,
        class_names,
        "Per-class precision, recall & F1 — Random forest (test)",
        out_dir / "fig_per_class_metrics_rf",
        formats,
    )
    _plot_grouped_prf1(
        y_test,
        lr_pred_te,
        class_names,
        "Per-class precision, recall & F1 — Logistic regression (test)",
        out_dir / "fig_per_class_metrics_lr",
        formats,
    )
    _plot_confidence_grid(
        y_test,
        rf_prob_te,
        rf_pred_te,
        rf_thr,
        "Random forest",
        class_names,
        out_dir / "fig_confidence_by_true_class_rf",
        formats,
    )
    _plot_confidence_grid(
        y_test,
        lr_prob_te,
        lr_pred_te,
        lr_thr,
        "Logistic regression",
        class_names,
        out_dir / "fig_confidence_by_true_class_lr",
        formats,
    )

    imp = np.asarray(rf.feature_importances_, dtype=float)
    imp = imp / (imp.sum() + 1e-12)
    _plot_rf_importance_maps(
        imp,
        out_dir / "fig_rf_feature_importance_heatmap",
        out_dir / "fig_rf_feature_importance_profile",
        formats,
    )

    if not args.skip_ensemble_curve:
        max_trees = int(getattr(rf, "n_estimators", 300) or 300)
        step = max(10, max_trees // 20)
        tree_steps = list(range(step, max_trees + 1, step))
        if tree_steps[-1] != max_trees:
            tree_steps.append(max_trees)
        _plot_rf_ensemble_progress(
            X_train,
            y_train,
            sw,
            seed=args.seed,
            tree_steps=tree_steps,
            out_base=out_dir / "fig_rf_ensemble_growth",
            formats=formats,
        )

    print(f"\nDone. Figures under: {out_dir.resolve()}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
