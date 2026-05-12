"""
Generate a test-set evaluation report from saved ESM baselines (no retraining).

Reads frozen thresholds from metrics_esm_baseline.json, loads joblib heads,
scores the homology test split, writes JSON + human-readable TXT + error CSVs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from plasticdeg import paths
from plasticdeg.train.train_esm_baseline import label_from_split_id, load_split_ids


def _build_test_xy(
    embeddings_npz: Path,
    test_ids_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    data = np.load(embeddings_npz, allow_pickle=True)
    ids = [str(x) for x in data["ids"]]
    X_all = np.asarray(data["embeddings"], dtype=np.float32)
    id_to_row = {i: r for r, i in enumerate(ids)}
    test_ids = load_split_ids(test_ids_path)
    rows: list[int] = []
    y: list[int] = []
    used: list[str] = []
    for sid in sorted(test_ids):
        r = id_to_row.get(sid)
        if r is None:
            continue
        rows.append(r)
        y.append(label_from_split_id(sid))
        used.append(sid)
    return X_all[rows], np.asarray(y, dtype=int), used


def _load_pos_meta(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    out: dict[str, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            if acc:
                out[acc] = {k: (row.get(k) or "").strip() for k in ("substrate", "source_note", "description")}
    return out


def _split_accession(split_id: str) -> str:
    if split_id.startswith("NEG_"):
        return split_id[4:]
    return split_id


def _enrich_row(split_id: str, pos_meta: dict[str, dict[str, str]]) -> dict[str, str]:
    acc = _split_accession(split_id)
    base = acc.rsplit(".", 1)[0] if "." in acc and acc.rsplit(".", 1)[-1].isdigit() else acc
    meta = pos_meta.get(acc) or pos_meta.get(base) or {}
    return {
        "split_id": split_id,
        "accession": acc,
        "substrate": meta.get("substrate", ""),
        "source_note": meta.get("source_note", ""),
        "description": meta.get("description", ""),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluation report from saved ESM baseline models")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=paths.embeddings_esm2_t33_mean_v2_npz(),
    )
    parser.add_argument(
        "--test-ids",
        type=Path,
        default=paths.split_test_txt(),
    )
    parser.add_argument(
        "--positives-csv",
        type=Path,
        default=paths.positives_gt_csv(),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=paths.metrics_esm_baseline_json(),
    )
    parser.add_argument(
        "--lr-model",
        type=Path,
        default=paths.model_lr_esm_baseline_joblib(),
    )
    parser.add_argument(
        "--rf-model",
        type=Path,
        default=paths.model_rf_esm_baseline_joblib(),
    )
    parser.add_argument(
        "--hard-probe",
        type=Path,
        default=paths.hard_negative_probe_json(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
    )
    args = parser.parse_args(argv)

    for p, label in [
        (args.embeddings, "embeddings"),
        (args.test_ids, "test ids"),
        (args.metrics, "metrics"),
        (args.lr_model, "LR joblib"),
        (args.rf_model, "RF joblib"),
    ]:
        if not p.exists():
            print(f"ERROR: missing {label}: {p}", file=sys.stderr)
            return 1

    metrics_doc = json.loads(args.metrics.read_text(encoding="utf-8"))
    pos_meta = _load_pos_meta(args.positives_csv)
    X_test, y_test, test_ids_ordered = _build_test_xy(args.embeddings, args.test_ids)
    if len(y_test) == 0:
        print("ERROR: empty test set.", file=sys.stderr)
        return 1

    lr_bundle = joblib.load(args.lr_model)
    scaler = lr_bundle["scaler"]
    lr = lr_bundle["model"]
    rf = joblib.load(args.rf_model)

    X_te_s = scaler.transform(X_test)
    lr_prob = lr.predict_proba(X_te_s)[:, 1]
    rf_prob = rf.predict_proba(X_test)[:, 1]

    t_lr = float(metrics_doc["models"]["logistic_regression"]["threshold_train_recall_ge_0.8"])
    t_rf = float(metrics_doc["models"]["random_forest"]["threshold_train_recall_ge_0.8"])

    lr_pred = (lr_prob >= t_lr).astype(int)
    rf_pred = (rf_prob >= t_rf).astype(int)

    def pack_model_report(name: str, y_score: np.ndarray, y_hat: np.ndarray, thr: float) -> dict[str, Any]:
        return {
            "name": name,
            "threshold_frozen_from_train": thr,
            "n_test": int(len(y_test)),
            "pr_auc": float(average_precision_score(y_test, y_score)),
            "roc_auc": float(roc_auc_score(y_test, y_score)),
            "accuracy_at_threshold": float(accuracy_score(y_test, y_hat)),
            "mcc_at_threshold": float(matthews_corrcoef(y_test, y_hat)),
            "precision_at_threshold": float(precision_score(y_test, y_hat, zero_division=0)),
            "recall_at_threshold": float(recall_score(y_test, y_hat, pos_label=1, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_test, y_hat).tolist(),
            "classification_report": classification_report(
                y_test, y_hat, target_names=["negative", "positive"], output_dict=True, zero_division=0
            ),
        }

    report: dict[str, Any] = {
        "embeddings": str(args.embeddings),
        "test_ids_file": str(args.test_ids),
        "metrics_source": str(args.metrics),
        "note": "Thresholds are frozen from train (metrics JSON); no tuning on test or probe.",
        "models": {
            "logistic_regression": pack_model_report("logistic_regression", lr_prob, lr_pred, t_lr),
            "random_forest": pack_model_report("random_forest", rf_prob, rf_pred, t_rf),
        },
    }
    if args.hard_probe.exists():
        report["hard_negative_probe"] = json.loads(args.hard_probe.read_text(encoding="utf-8"))

    mdir = args.out_dir / "metrics"
    repdir = args.out_dir / "reports"
    tbldir = args.out_dir / "tables"
    for d in (mdir, repdir, tbldir):
        d.mkdir(parents=True, exist_ok=True)
    json_path = mdir / "evaluation_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {json_path}", flush=True)

    def write_errors_csv(
        path: Path,
        y_hat: np.ndarray,
        y_score: np.ndarray,
    ) -> tuple[int, int]:
        fn = fp = 0
        with path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = [
                "kind",
                "split_id",
                "accession",
                "y_true",
                "y_pred",
                "score",
                "substrate",
                "source_note",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for sid, yt, yp, sc in zip(test_ids_ordered, y_test, y_hat, y_score):
                if yt == 1 and yp == 0:
                    fn += 1
                    kind = "false_negative"
                elif yt == 0 and yp == 1:
                    fp += 1
                    kind = "false_positive"
                else:
                    continue
                ex = _enrich_row(sid, pos_meta)
                w.writerow(
                    {
                        "kind": kind,
                        "split_id": sid,
                        "accession": ex["accession"],
                        "y_true": int(yt),
                        "y_pred": int(yp),
                        "score": float(sc),
                        "substrate": ex["substrate"],
                        "source_note": ex["source_note"],
                    }
                )
        return fn, fp

    fn_lr, fp_lr = write_errors_csv(tbldir / "test_errors_lr.csv", lr_pred, lr_prob)
    fn_rf, fp_rf = write_errors_csv(tbldir / "test_errors_rf.csv", rf_pred, rf_prob)
    print(f"Wrote {tbldir / 'test_errors_lr.csv'} (FN={fn_lr}, FP={fp_lr})", flush=True)
    print(f"Wrote {tbldir / 'test_errors_rf.csv'} (FN={fn_rf}, FP={fp_rf})", flush=True)

    txt_lines = [
        "ESM baseline — test evaluation report (frozen thresholds)",
        "",
        f"Embeddings: {args.embeddings}",
        f"Test ID list: {args.test_ids}",
        f"Metrics / thresholds from: {args.metrics}",
        "",
    ]
    for key in ("random_forest", "logistic_regression"):
        m = report["models"][key]
        txt_lines.extend(
            [
                f"=== {key} ===",
                f"  PR-AUC: {m['pr_auc']:.6f}",
                f"  ROC-AUC: {m['roc_auc']:.6f}",
                f"  Threshold (train, recall>=0.8 rule): {m['threshold_frozen_from_train']:.6f}",
                f"  Accuracy @ threshold: {m['accuracy_at_threshold']:.6f}",
                f"  MCC @ threshold: {m['mcc_at_threshold']:.6f}",
                f"  Precision @ threshold (positive class): {m['precision_at_threshold']:.6f}",
                f"  Recall @ threshold (positive class): {m['recall_at_threshold']:.6f}",
                f"  Confusion matrix [[TN, FP],[FN, TP]]: {m['confusion_matrix']}",
                "",
            ]
        )
    if "hard_negative_probe" in report:
        hp = report["hard_negative_probe"]
        txt_lines.extend(
            [
                "=== Hard negative probe (RF threshold, no retuning) ===",
                json.dumps(hp, indent=2),
                "",
            ]
        )
    txt_path = repdir / "evaluation_report.txt"
    txt_path.write_text("\n".join(txt_lines), encoding="utf-8")
    print(f"Wrote {txt_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
