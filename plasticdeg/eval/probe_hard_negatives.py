"""
Stress-test the saved RF head on hard-negative embeddings using the *frozen* train
threshold from metrics_esm_baseline.json (no retuning on probe data).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

from plasticdeg import paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score hard negatives with frozen RF threshold")
    parser.add_argument(
        "--hard-embeddings",
        type=Path,
        default=paths.embeddings_hard_negatives_npz(),
    )
    parser.add_argument(
        "--main-embeddings",
        type=Path,
        default=paths.embeddings_esm2_t33_mean_v2_npz(),
    )
    parser.add_argument(
        "--test-ids",
        type=Path,
        default=paths.split_test_txt(),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=paths.metrics_esm_baseline_json(),
    )
    parser.add_argument(
        "--rf-model",
        type=Path,
        default=paths.model_rf_esm_baseline_joblib(),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=paths.hard_negative_probe_json(),
    )
    parser.add_argument(
        "--baseline-v2",
        action="store_true",
        help="Use tier-weighted v2 RF head + metrics (metrics_esm_baseline_v2.json); "
        "writes hard_negative_probe_v2.json by default.",
    )
    args = parser.parse_args(argv)

    if args.baseline_v2:
        args.metrics = paths.metrics_esm_baseline_v2_json()
        args.rf_model = paths.model_rf_esm_baseline_v2_joblib()
        if args.out_json == paths.hard_negative_probe_json():
            args.out_json = paths.hard_negative_probe_v2_json()

    if not args.hard_embeddings.exists():
        print(f"ERROR: missing {args.hard_embeddings}", file=sys.stderr)
        return 1
    if not args.rf_model.exists():
        print(f"ERROR: missing {args.rf_model}", file=sys.stderr)
        return 1
    if not args.metrics.exists():
        print(f"ERROR: missing {args.metrics}", file=sys.stderr)
        return 1

    metrics = json.loads(args.metrics.read_text(encoding="utf-8"))
    thr = float(metrics["models"]["random_forest"]["threshold_train_recall_ge_0.8"])

    rf = joblib.load(args.rf_model)
    hz = np.load(args.hard_embeddings, allow_pickle=True)
    Xh = np.asarray(hz["embeddings"], dtype=np.float32)
    hard_ids = [str(x) for x in hz["ids"]]
    prob_h = rf.predict_proba(Xh)[:, 1]
    pred_pos = (prob_h >= thr).astype(int)
    n_fp = int(pred_pos.sum())
    n = len(prob_h)

    easy_scores: list[float] = []
    if args.main_embeddings.exists() and args.test_ids.exists():
        test_neg = set()
        for ln in args.test_ids.read_text(encoding="utf-8", errors="replace").splitlines():
            s = ln.strip()
            if s.startswith("NEG_"):
                test_neg.add(s)
        if test_neg:
            data = np.load(args.main_embeddings, allow_pickle=True)
            ids = [str(x) for x in data["ids"]]
            X = np.asarray(data["embeddings"], dtype=np.float32)
            id_to_row = {i: r for r, i in enumerate(ids)}
            rows = [id_to_row[i] for i in sorted(test_neg) if i in id_to_row]
            if rows:
                prob_e = rf.predict_proba(X[rows])[:, 1]
                easy_scores = prob_e.tolist()

    payload = {
        "baseline": "v2" if args.baseline_v2 else "v1",
        "metrics_source": str(args.metrics),
        "rf_model_source": str(args.rf_model),
        "threshold_rf_train_recall_ge_0.8": thr,
        "hard_negative_count": n,
        "hard_negative_predicted_positive_count": n_fp,
        "hard_negative_false_positive_rate": n_fp / n if n else None,
        "hard_negative_score_mean": float(np.mean(prob_h)),
        "hard_negative_score_median": float(np.median(prob_h)),
        "hard_ids_sample": hard_ids[:5],
    }
    if easy_scores:
        arr = np.asarray(easy_scores, dtype=float)
        payload["easy_test_negative_score_mean"] = float(np.mean(arr))
        payload["easy_test_negative_score_median"] = float(np.median(arr))
        payload["easy_test_negative_count"] = len(easy_scores)

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"  Wrote {args.out_json}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
