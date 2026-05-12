"""
Phase 2 — train sklearn heads on frozen ESM-2 embeddings (embed-once workflow).

Loads .npz from embed_sequences, aligns train/test split IDs and labels, fits
LogisticRegression + RandomForest, reports PR-AUC / ROC-AUC / MCC at a train-chosen
threshold (recall >= RECALL_TARGET on positives).

Optional: --augment-neg-npz appends extra negative rows (e.g. hard / adversarial
embeddings) to *training only* (capped) so heads learn a harder decision boundary
without changing the official test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler

from plasticdeg import paths
from plasticdeg.evaluation_spec import RECALL_TARGET_FOR_THRESHOLD


def load_split_ids(path: Path) -> set[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return {ln.strip() for ln in lines if ln.strip()}


def label_from_split_id(sid: str) -> int:
    return 0 if sid.startswith("NEG_") else 1


def threshold_for_min_recall_positives(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    min_recall: float,
) -> float:
    """Largest probability threshold t such that recall(positive, y_prob >= t) >= min_recall."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    pos = y_true == 1
    n_pos = int(pos.sum())
    if n_pos == 0:
        return 0.5
    uniq = np.sort(np.unique(y_prob))
    candidates: list[float] = []
    for t in uniq:
        pred = y_prob >= t
        rec = (pred & pos).sum() / n_pos
        if rec >= min_recall:
            candidates.append(float(t))
    if candidates:
        return max(candidates)
    return float(uniq.min())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train LR + RF on frozen ESM embeddings")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=paths.embeddings_esm2_t33_mean_v2_npz(),
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=paths.split_train_txt(),
    )
    parser.add_argument(
        "--test-ids",
        type=Path,
        default=paths.split_test_txt(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
        help="Artifact root (writes metrics/, models/, reports/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--augment-neg-npz",
        type=Path,
        default=None,
        help="Optional second .npz (e.g. embeddings_hard_negatives.npz); all rows are y=0, train-only, subsampled.",
    )
    parser.add_argument(
        "--augment-neg-max",
        type=int,
        default=400,
        help="Max extra negative rows to draw from --augment-neg-npz (uniform subsample if larger).",
    )
    parser.add_argument(
        "--artifact-suffix",
        type=str,
        default="",
        help="Suffix for outputs, e.g. _aug (default _aug when --augment-neg-npz is set).",
    )
    args = parser.parse_args(argv)

    suffix = args.artifact_suffix.strip()
    if args.augment_neg_npz is not None and not suffix:
        suffix = "_aug"
    if suffix and not suffix.startswith("_"):
        suffix = "_" + suffix

    if not args.embeddings.exists():
        print(f"ERROR: missing embeddings: {args.embeddings}", file=sys.stderr)
        return 1

    data = np.load(args.embeddings, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    X_all = np.asarray(data["embeddings"], dtype=np.float32)
    id_to_row = {i: r for r, i in enumerate(ids)}

    train_ids = load_split_ids(args.train_ids)
    test_ids = load_split_ids(args.test_ids)

    def build_xy(id_set: set[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
        rows: list[int] = []
        ylist: list[int] = []
        used: list[str] = []
        missing: list[str] = []
        for sid in sorted(id_set):
            r = id_to_row.get(sid)
            if r is None:
                missing.append(sid)
                continue
            rows.append(r)
            ylist.append(label_from_split_id(sid))
            used.append(sid)
        if missing:
            print(f"  WARNING: {len(missing)} split IDs not in embedding npz (first 5): {missing[:5]}", flush=True)
        return X_all[rows], np.asarray(ylist, dtype=int), used

    X_train, y_train, _ = build_xy(train_ids)
    X_test, y_test, _ = build_xy(test_ids)

    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: empty train or test after alignment.", file=sys.stderr)
        return 1

    rng = np.random.default_rng(args.seed)
    augment_note = ""
    if args.augment_neg_npz is not None:
        if not args.augment_neg_npz.exists():
            print(f"ERROR: --augment-neg-npz missing: {args.augment_neg_npz}", file=sys.stderr)
            return 1
        extra = np.load(args.augment_neg_npz, allow_pickle=True)
        X_extra = np.asarray(extra["embeddings"], dtype=np.float32)
        if X_extra.shape[1] != X_train.shape[1]:
            print(
                f"ERROR: augment embedding dim {X_extra.shape[1]} != train dim {X_train.shape[1]}",
                file=sys.stderr,
            )
            return 1
        n_extra = min(int(args.augment_neg_max), len(X_extra))
        if len(X_extra) > n_extra:
            pick = rng.choice(len(X_extra), size=n_extra, replace=False)
            X_extra = X_extra[pick]
        y_extra = np.zeros(len(X_extra), dtype=int)
        X_train = np.vstack([X_train, X_extra])
        y_train = np.concatenate([y_train, y_extra])
        augment_note = f"augment_neg_npz={args.augment_neg_npz} n_extra={len(X_extra)}"
        print(f"  Train augmented: +{len(X_extra)} negatives ({augment_note})", flush=True)

    print(f"  Train: {X_train.shape[0]} samples, dim={X_train.shape[1]}", flush=True)
    print(f"  Test:  {X_test.shape[0]} samples", flush=True)

    rng_sk = args.seed
    results: dict[str, dict] = {}

    # --- Model A: Logistic regression (scaled features) ---
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=4000,
        class_weight="balanced",
        random_state=rng_sk,
        solver="lbfgs",
    )
    lr.fit(X_tr_s, y_train)
    lr_prob_tr = lr.predict_proba(X_tr_s)[:, 1]
    lr_prob_te = lr.predict_proba(X_te_s)[:, 1]

    t_lr = threshold_for_min_recall_positives(
        y_train, lr_prob_tr, min_recall=RECALL_TARGET_FOR_THRESHOLD
    )
    lr_pred_te = (lr_prob_te >= t_lr).astype(int)

    results["logistic_regression"] = {
        "pr_auc": float(average_precision_score(y_test, lr_prob_te)),
        "roc_auc": float(roc_auc_score(y_test, lr_prob_te)),
        "threshold_train_recall_ge_0.8": t_lr,
        "mcc_at_threshold": float(matthews_corrcoef(y_test, lr_pred_te)),
        "precision_at_threshold": float(precision_score(y_test, lr_pred_te, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_test, lr_pred_te, pos_label=1, zero_division=0)),
        "confusion_matrix_test": confusion_matrix(y_test, lr_pred_te).tolist(),
    }

    # --- Model B: Random forest (raw features; tree models invariant to monotone rescale) ---
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=rng_sk,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_prob_tr = rf.predict_proba(X_train)[:, 1]
    rf_prob_te = rf.predict_proba(X_test)[:, 1]

    t_rf = threshold_for_min_recall_positives(
        y_train, rf_prob_tr, min_recall=RECALL_TARGET_FOR_THRESHOLD
    )
    rf_pred_te = (rf_prob_te >= t_rf).astype(int)

    results["random_forest"] = {
        "pr_auc": float(average_precision_score(y_test, rf_prob_te)),
        "roc_auc": float(roc_auc_score(y_test, rf_prob_te)),
        "threshold_train_recall_ge_0.8": t_rf,
        "mcc_at_threshold": float(matthews_corrcoef(y_test, rf_pred_te)),
        "precision_at_threshold": float(precision_score(y_test, rf_pred_te, zero_division=0)),
        "recall_at_threshold": float(recall_score(y_test, rf_pred_te, pos_label=1, zero_division=0)),
        "confusion_matrix_test": confusion_matrix(y_test, rf_pred_te).tolist(),
    }

    mdir = args.out_dir / "metrics"
    moddir = args.out_dir / "models"
    repdir = args.out_dir / "reports"
    for d in (mdir, moddir, repdir):
        d.mkdir(parents=True, exist_ok=True)
    metrics_path = mdir / f"metrics_esm_baseline{suffix}.json"
    payload = {
        "embeddings": str(args.embeddings),
        "train_ids": str(args.train_ids),
        "test_ids": str(args.test_ids),
        "recall_target_for_threshold": RECALL_TARGET_FOR_THRESHOLD,
        "artifact_suffix": suffix or None,
        "train_augmentation": augment_note or None,
        "models": results,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Wrote {metrics_path}", flush=True)

    lr_path = moddir / f"model_lr_esm_baseline{suffix}.joblib"
    rf_path = moddir / f"model_rf_esm_baseline{suffix}.joblib"
    joblib.dump({"scaler": scaler, "model": lr}, lr_path)
    joblib.dump(rf, rf_path)
    print(f"  Wrote {lr_path}", flush=True)
    print(f"  Wrote {rf_path}", flush=True)

    # PR curve for RF (primary tree baseline)
    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        prec, rec, _ = precision_recall_curve(y_test, rf_prob_te)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(rec, prec, label=f"RF PR-AUC={results['random_forest']['pr_auc']:.4f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("Test set: precision–recall (Random Forest)")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        pr_path = repdir / f"pr_curve_rf_esm_baseline{suffix}.png"
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Wrote {pr_path}", flush=True)

        fpr, tpr, _ = roc_curve(y_test, rf_prob_te)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.plot(fpr, tpr, label=f"RF ROC-AUC={results['random_forest']['roc_auc']:.4f}")
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.set_title("Test set: ROC (Random Forest)")
        ax2.legend()
        roc_path = repdir / f"roc_curve_rf_esm_baseline{suffix}.png"
        fig2.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Wrote {roc_path}", flush=True)
    except Exception as e:
        print(f"  WARNING: could not save plots: {e}", flush=True)

    print("\n  Summary (test set):", flush=True)
    for name, m in results.items():
        print(
            f"    {name}: PR-AUC={m['pr_auc']:.4f}  ROC-AUC={m['roc_auc']:.4f}  "
            f"MCC@thr={m['mcc_at_threshold']:.4f}  P@thr={m['precision_at_threshold']:.4f}  "
            f"R@thr={m['recall_at_threshold']:.4f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
