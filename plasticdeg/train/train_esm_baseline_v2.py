"""
Train sklearn heads on frozen ESM embeddings with **tier-aware sample_weight**
(Gold=1.0, Silver=0.5 on positives; negatives unchanged via balanced scaling).

Reports test metrics on:
  - **combined**: full official test split
  - **gold_positives_plus_all_negatives**: same negatives, positives restricted to Tier 1
    (curated PAZy-style Gold only) to see whether Silver mainly helps generalization.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from plasticdeg import paths
from plasticdeg.evaluation_spec import RECALL_TARGET_FOR_THRESHOLD
from plasticdeg.train.train_esm_baseline import (
    label_from_split_id,
    load_split_ids,
    threshold_for_min_recall_positives,
)


def _strip_uniprot_version(acc: str) -> str:
    acc = acc.strip()
    if "." in acc and acc.rsplit(".", 1)[-1].isdigit():
        return acc.rsplit(".", 1)[0]
    return acc


def load_tier_map(tier_csv: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    with tier_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            if not acc:
                continue
            try:
                tier = int(float((row.get("tier") or "1").strip()))
            except ValueError:
                tier = 1
            out[acc] = tier
            base = _strip_uniprot_version(acc)
            if base != acc:
                out.setdefault(base, tier)
    return out


def tier_for_split_id(sid: str, tier_map: dict[str, int]) -> int:
    if sid.startswith("NEG_"):
        return 0
    if sid in tier_map:
        return tier_map[sid]
    return tier_map.get(_strip_uniprot_version(sid), 1)


def _safe_pr_roc_mcc(y_true: np.ndarray, y_prob: np.ndarray, y_hat: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        out["pr_auc"] = None
        out["roc_auc"] = None
        out["mcc_at_threshold"] = None
        out["accuracy_at_threshold"] = float(accuracy_score(y_true, y_hat)) if len(y_true) else None
        out["precision_at_threshold"] = None
        out["recall_at_threshold"] = None
        out["confusion_matrix"] = None
        return out
    out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    out["mcc_at_threshold"] = float(matthews_corrcoef(y_true, y_hat))
    out["accuracy_at_threshold"] = float(accuracy_score(y_true, y_hat))
    out["precision_at_threshold"] = float(precision_score(y_true, y_hat, zero_division=0))
    out["recall_at_threshold"] = float(recall_score(y_true, y_hat, pos_label=1, zero_division=0))
    out["confusion_matrix"] = confusion_matrix(y_true, y_hat).tolist()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Tier-weighted LR+RF on frozen ESM embeddings (v2)")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=paths.embeddings_esm2_t33_mean_v2_npz(),
    )
    parser.add_argument("--train-ids", type=Path, default=paths.split_train_txt())
    parser.add_argument("--test-ids", type=Path, default=paths.split_test_txt())
    parser.add_argument(
        "--tier-csv",
        type=Path,
        default=paths.positives_gt_expanded_csv(),
        help="Must include accession + tier (1=Gold, 2=Silver).",
    )
    parser.add_argument(
        "--tier1-weight",
        type=float,
        default=1.0,
        help="Positive sample_weight multiplier for Tier 1 (after balanced scaling).",
    )
    parser.add_argument(
        "--tier2-weight",
        type=float,
        default=0.5,
        help="Positive sample_weight multiplier for Tier 2 (after balanced scaling).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
        help="Artifact root (metrics/, models/, reports/)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if not args.embeddings.exists():
        print(f"ERROR: missing embeddings: {args.embeddings}", file=sys.stderr)
        return 1
    if not args.tier_csv.exists():
        print(f"ERROR: missing tier CSV: {args.tier_csv}", file=sys.stderr)
        return 1

    tier_map = load_tier_map(args.tier_csv)
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
            print(f"  WARNING: {len(missing)} split IDs not in npz (first 5): {missing[:5]}", flush=True)
        return X_all[rows], np.asarray(ylist, dtype=int), used

    X_train, y_train, train_used = build_xy(train_ids)
    X_test, y_test, test_used = build_xy(test_ids)
    if len(X_train) == 0 or len(X_test) == 0:
        print("ERROR: empty train or test.", file=sys.stderr)
        return 1

    sw = compute_sample_weight("balanced", y_train).astype(np.float64, copy=True)
    for i, sid in enumerate(train_used):
        if y_train[i] != 1:
            continue
        t = tier_for_split_id(sid, tier_map)
        if t == 2:
            sw[i] *= float(args.tier2_weight)
        else:
            sw[i] *= float(args.tier1_weight)

    rng_sk = args.seed

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)

    lr = LogisticRegression(
        max_iter=4000,
        class_weight=None,
        random_state=rng_sk,
        solver="lbfgs",
    )
    lr.fit(X_tr_s, y_train, sample_weight=sw)
    lr_prob_tr = lr.predict_proba(X_tr_s)[:, 1]
    lr_prob_te = lr.predict_proba(X_te_s)[:, 1]
    t_lr = threshold_for_min_recall_positives(
        y_train, lr_prob_tr, min_recall=RECALL_TARGET_FOR_THRESHOLD
    )
    lr_pred_te = (lr_prob_te >= t_lr).astype(int)

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        class_weight=None,
        random_state=rng_sk,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train, sample_weight=sw)
    rf_prob_tr = rf.predict_proba(X_train)[:, 1]
    rf_prob_te = rf.predict_proba(X_test)[:, 1]
    t_rf = threshold_for_min_recall_positives(
        y_train, rf_prob_tr, min_recall=RECALL_TARGET_FOR_THRESHOLD
    )
    rf_pred_te = (rf_prob_te >= t_rf).astype(int)

    tiers_test = np.array([tier_for_split_id(s, tier_map) for s in test_used], dtype=int)
    mask_gold_eval = (y_test == 0) | ((y_test == 1) & (tiers_test == 1))

    def pack_model(name: str, prob_te: np.ndarray, pred_te: np.ndarray, thr: float) -> dict:
        base = {
            "name": name,
            "threshold_train_recall_ge_0.8": thr,
            "n_test_combined": int(len(y_test)),
            "test_combined": _safe_pr_roc_mcc(y_test, prob_te, pred_te),
            "n_test_gold_pos_plus_all_negs": int(mask_gold_eval.sum()),
            "test_gold_positives_plus_all_negatives": _safe_pr_roc_mcc(
                y_test[mask_gold_eval],
                prob_te[mask_gold_eval],
                pred_te[mask_gold_eval],
            ),
        }
        return base

    models_out = {
        "logistic_regression": pack_model("logistic_regression", lr_prob_te, lr_pred_te, t_lr),
        "random_forest": pack_model("random_forest", rf_prob_te, rf_pred_te, t_rf),
    }

    mdir = args.out_dir / "metrics"
    moddir = args.out_dir / "models"
    repdir = args.out_dir / "reports"
    for d in (mdir, moddir, repdir):
        d.mkdir(parents=True, exist_ok=True)

    metrics_path = mdir / "metrics_esm_baseline_v2.json"
    payload = {
        "embeddings": str(args.embeddings),
        "train_ids": str(args.train_ids),
        "test_ids": str(args.test_ids),
        "tier_csv": str(args.tier_csv),
        "tier1_weight": args.tier1_weight,
        "tier2_weight": args.tier2_weight,
        "recall_target_for_threshold": RECALL_TARGET_FOR_THRESHOLD,
        "note": "Train uses balanced sample_weight × tier multipliers on positives; "
        "threshold still chosen on train positives at recall target.",
        "models": models_out,
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Wrote {metrics_path}", flush=True)

    lr_path = moddir / "model_lr_esm_baseline_v2.joblib"
    rf_path = moddir / "model_rf_esm_baseline_v2.joblib"
    joblib.dump({"scaler": scaler, "model": lr}, lr_path)
    joblib.dump(rf, rf_path)
    print(f"  Wrote {lr_path}", flush=True)
    print(f"  Wrote {rf_path}", flush=True)

    try:
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve

        prec, rec, _ = precision_recall_curve(y_test, rf_prob_te)
        fig, ax = plt.subplots(figsize=(6, 5))
        m = models_out["random_forest"]["test_combined"]
        pr_auc = m.get("pr_auc")
        ax.plot(rec, prec, label=f"RF PR-AUC={pr_auc:.4f}" if pr_auc is not None else "RF")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title("v2 test combined: PR (Random Forest)")
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.05)
        pr_path = repdir / "pr_curve_rf_esm_baseline_v2.png"
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        fpr, tpr, _ = roc_curve(y_test, rf_prob_te)
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        roc_auc = m.get("roc_auc")
        ax2.plot(fpr, tpr, label=f"RF ROC-AUC={roc_auc:.4f}" if roc_auc is not None else "RF")
        ax2.plot([0, 1], [0, 1], "k--", alpha=0.3)
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.set_title("v2 test combined: ROC (Random Forest)")
        ax2.legend()
        roc_path = repdir / "roc_curve_rf_esm_baseline_v2.png"
        fig2.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"  Wrote {pr_path}", flush=True)
        print(f"  Wrote {roc_path}", flush=True)
    except Exception as e:
        print(f"  WARNING: could not save plots: {e}", flush=True)

    print("\n  Summary v2 (RF, combined test):", flush=True)
    m = models_out["random_forest"]["test_combined"]
    print(f"    PR-AUC={m.get('pr_auc')}  ROC-AUC={m.get('roc_auc')}  MCC@thr={m.get('mcc_at_threshold')}", flush=True)
    print("  Summary v2 (RF, Gold-only positives + all test negatives):", flush=True)
    m2 = models_out["random_forest"]["test_gold_positives_plus_all_negatives"]
    print(f"    PR-AUC={m2.get('pr_auc')}  ROC-AUC={m2.get('roc_auc')}  MCC@thr={m2.get('mcc_at_threshold')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
