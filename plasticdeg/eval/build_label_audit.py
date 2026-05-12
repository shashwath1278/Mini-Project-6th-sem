"""
Build a label-audit spreadsheet for all PAZy positives (Tier / manual review).

Adds boolean flags for test-set false negatives (LR / RF at frozen thresholds),
simple keyword hints (Trypsin, etc.), and empty columns tier_manual / review_note
for you to fill in Excel or a text editor.

Inputs:
  - data/processed_v2/tables/positives_gt.csv
  - data/processed_v2/tables/test_errors_rf.csv (from report_esm_baseline)
  - data/processed_v2/tables/test_errors_lr.csv

Output:
  - data/processed_v2/tables/label_audit_candidates.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from plasticdeg import paths


def _fn_accessions(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            if (row.get("kind") or "").strip() != "false_negative":
                continue
            acc = (row.get("accession") or "").strip()
            if acc:
                out.add(acc)
                if "." in acc and acc.rsplit(".", 1)[-1].isdigit():
                    out.add(acc.rsplit(".", 1)[0])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build label audit CSV for PAZy positives")
    parser.add_argument(
        "--positives-csv",
        type=Path,
        default=paths.positives_gt_csv(),
    )
    parser.add_argument(
        "--errors-rf",
        type=Path,
        default=paths.test_errors_rf_csv(),
    )
    parser.add_argument(
        "--errors-lr",
        type=Path,
        default=paths.test_errors_lr_csv(),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.label_audit_candidates_csv(),
    )
    args = parser.parse_args(argv)

    if not args.positives_csv.exists():
        print(f"ERROR: missing {args.positives_csv}", file=sys.stderr)
        return 1

    fn_rf = _fn_accessions(args.errors_rf)
    fn_lr = _fn_accessions(args.errors_lr)

    def match_fn(acc: str, fn_set: set[str]) -> bool:
        base = acc.rsplit(".", 1)[0] if "." in acc and acc.rsplit(".", 1)[-1].isdigit() else acc
        return acc in fn_set or base in fn_set

    rows_out: list[dict[str, str]] = []
    with args.positives_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        fieldnames_in = r.fieldnames or []
        for row in r:
            acc = (row.get("accession") or "").strip()
            desc = (row.get("description") or "").lower()
            note = (row.get("source_note") or "").lower()
            blob = f"{desc} {note}"
            tryp = "trypsin" in blob
            protease = any(
                k in blob
                for k in ("trypsin", "chymotrypsin", "peptidase", "protease", "subtilisin")
            )
            irf = match_fn(acc, fn_rf) if acc else False
            ilr = match_fn(acc, fn_lr) if acc else False
            base = {
                **{k: (row.get(k) or "").strip() for k in fieldnames_in},
                "is_test_fn_rf": "1" if irf else "0",
                "is_test_fn_lr": "1" if ilr else "0",
                "is_test_fn_both": "1" if (irf and ilr) else "0",
                "keyword_trypsin": "1" if tryp else "0",
                "keyword_protease_family": "1" if protease else "0",
                "tier_manual": "",
                "review_note": "",
            }
            rows_out.append(base)

    extra = [
        "is_test_fn_rf",
        "is_test_fn_lr",
        "is_test_fn_both",
        "keyword_trypsin",
        "keyword_protease_family",
        "tier_manual",
        "review_note",
    ]
    out_fields = list(fieldnames_in) + [c for c in extra if c not in fieldnames_in]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows_out)

    n_both = sum(1 for x in rows_out if x["is_test_fn_both"] == "1")
    n_rf = sum(1 for x in rows_out if x["is_test_fn_rf"] == "1")
    n_lr = sum(1 for x in rows_out if x["is_test_fn_lr"] == "1")
    n_tryp = sum(1 for x in rows_out if x["keyword_trypsin"] == "1")
    print(f"  Wrote {args.out} ({len(rows_out)} positives)", flush=True)
    print(f"  Flags: test FN (RF)={n_rf}, (LR)={n_lr}, both={n_both}, keyword_trypsin={n_tryp}", flush=True)
    print("  Fill tier_manual (e.g. 1=keep, 2=weak evidence, 3=exclude) and review_note.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
