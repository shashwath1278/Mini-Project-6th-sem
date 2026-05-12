"""
Merge probe JSON files (hard EC negatives, adversarial BLAST negatives, etc.)
into one table for the report.

Default: picks hard_negative_probe.json and adversarial_negative_probe.json
from data/processed_v2/probes/ if they exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plasticdeg import paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize tier probe JSON files")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Default: <out-dir>/probes/tier_probe_summary.json",
    )
    args = parser.parse_args(argv)

    pdir = args.out_dir / "probes"
    hard_probe = pdir / "hard_negative_probe_v2.json"
    if not hard_probe.is_file():
        hard_probe = pdir / "hard_negative_probe.json"
    default_pairs = [
        ("Tier2_EC_hydrolase", hard_probe),
        ("Tier3_BLAST_neighbor", pdir / "adversarial_negative_probe.json"),
    ]
    tiers: list[dict] = []
    for label, path in default_pairs:
        if not path.exists():
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        tiers.append(
            {
                "tier_label": label,
                "source_file": str(path),
                "n_probe_sequences": doc.get("hard_negative_count"),
                "predicted_positive_count": doc.get("hard_negative_predicted_positive_count"),
                "false_positive_rate": doc.get("hard_negative_false_positive_rate"),
                "score_mean": doc.get("hard_negative_score_mean"),
                "score_median": doc.get("hard_negative_score_median"),
                "threshold_rf_frozen": doc.get("threshold_rf_train_recall_ge_0.8"),
            }
        )

    out_path = args.out_json or (pdir / "tier_probe_summary.json")
    if not tiers:
        print(f"ERROR: no probe JSON files found under {args.out_dir}", file=sys.stderr)
        return 1

    payload = {"tiers": tiers, "note": "All probes use the same frozen RF threshold from train; no retuning."}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(f"Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
