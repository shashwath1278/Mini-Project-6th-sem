"""
Build a random CSV for manual dashboard / CLI checks.

Pools (all must exist under data/processed_v2/):
  - positives_gt_expanded.csv  (label 1)
  - negatives_gt.csv           (label 0, easy Swiss-Prot)
  - probes/hard_negatives_gt.csv (label 0, hydrolase stress negatives)

Example:

  python -m plasticdeg.eval.sample_manual_test_csv --seed 42 --out data/processed_v2/tables/manual_test_sample.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from plasticdeg import paths


def _load_split_ids(split_path: Path) -> set[str]:
    if not split_path.is_file():
        return set()
    return {ln.strip() for ln in split_path.read_text(encoding="utf-8").splitlines() if ln.strip()}


def _homology_split_hint(accession: str, label: int, train_ids: set[str], test_ids: set[str]) -> str:
    acc = accession.strip()
    if not acc:
        return ""
    if label == 1:
        if acc in test_ids:
            return "test"
        if acc in train_ids:
            return "train"
        for s in test_ids:
            if s == acc or s.endswith("|" + acc):
                return "test"
        for s in train_ids:
            if s == acc or s.endswith("|" + acc):
                return "train"
        return ""
    neg_key = f"NEG_{acc}"
    if neg_key in test_ids:
        return "test"
    if neg_key in train_ids:
        return "train"
    return ""


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Sample random rows with sequences for manual testing")
    p.add_argument("--positives-csv", type=Path, default=paths.positives_gt_expanded_csv())
    p.add_argument("--negatives-csv", type=Path, default=paths.negatives_gt_csv())
    p.add_argument("--hard-negatives-csv", type=Path, default=paths.hard_negatives_gt_csv())
    p.add_argument("--split-train", type=Path, default=paths.split_train_txt())
    p.add_argument("--split-test", type=Path, default=paths.split_test_txt())
    p.add_argument("--n-positive", type=int, default=25)
    p.add_argument("--n-negative-easy", type=int, default=25)
    p.add_argument("--n-negative-hard", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--out",
        type=Path,
        default=paths.tables_dir() / "manual_test_sample.csv",
        help="Output CSV path",
    )
    args = p.parse_args(argv)

    for path, name in (
        (args.positives_csv, "positives"),
        (args.negatives_csv, "negatives"),
        (args.hard_negatives_csv, "hard negatives"),
    ):
        if not path.is_file():
            print(f"ERROR: missing {name} file: {path}", file=sys.stderr)
            return 1

    train_ids = _load_split_ids(args.split_train)
    test_ids = _load_split_ids(args.split_test)

    pos_rows = _read_csv_rows(args.positives_csv)
    neg_easy = _read_csv_rows(args.negatives_csv)
    neg_hard = _read_csv_rows(args.hard_negatives_csv)

    rng = random.Random(args.seed)

    def pick(pool: list[dict[str, str]], k: int) -> list[dict[str, str]]:
        if k <= 0 or not pool:
            return []
        k = min(k, len(pool))
        return rng.sample(pool, k=k)

    chosen: list[dict[str, str]] = []
    for row in pick(pos_rows, args.n_positive):
        chosen.append({**row, "_pool": "positive_expanded"})
    for row in pick(neg_easy, args.n_negative_easy):
        chosen.append({**row, "_pool": "negative_easy_swissprot"})
    for row in pick(neg_hard, args.n_negative_hard):
        chosen.append({**row, "_pool": "negative_hard_hydrolase"})

    rng.shuffle(chosen)

    out_fields = [
        "manual_row",
        "pool",
        "accession",
        "sequence",
        "length",
        "label",
        "homology_split",
        "substrate",
        "tier",
        "source_note",
        "description",
    ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(chosen, start=1):
            acc = row.get("accession", "").strip()
            seq = row.get("sequence", "").strip()
            label_s = row.get("label", "")
            try:
                label = int(label_s)
            except (TypeError, ValueError):
                label = -1
            length = row.get("length", "") or (len(seq) if seq else "")
            w.writerow(
                {
                    "manual_row": i,
                    "pool": row.get("_pool", ""),
                    "accession": acc,
                    "sequence": seq,
                    "length": length,
                    "label": label,
                    "homology_split": _homology_split_hint(acc, label, train_ids, test_ids),
                    "substrate": row.get("substrate", "") if label == 1 else "",
                    "tier": row.get("tier", "") if label == 1 else "",
                    "source_note": row.get("source_note", ""),
                    "description": row.get("description", ""),
                }
            )

    print(f"Wrote {args.out} ({len(chosen)} rows, seed={args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
