"""
Remove UniProt accessions that have no FASTA at UniProt REST (inactive/deleted).

Reads a JSON array of {accession, ...}, drops rows whose accession is in
plasticdeg.support.dead_accessions.DEAD_UNIPROT_ACCESSIONS, writes the filtered list.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plasticdeg.support.dead_accessions import DEAD_UNIPROT_ACCESSIONS


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Remove dead UniProt accessions from ground-truth JSON lists"
    )
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            root / "data" / "pazy" / "ground_truth.json",
            root / "data" / "pazy" / "ground_truth_merged.json",
        ],
        help="One or more JSON files to filter in place",
    )
    args = parser.parse_args(argv)

    for path in args.inputs:
        if not path.exists():
            print(f"  SKIP (missing): {path}")
            continue
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)
        if not isinstance(rows, list):
            print(f"  ERROR: {path} is not a JSON array")
            return 1
        before = len(rows)
        kept: list[dict] = []
        removed = 0
        for row in rows:
            acc = str(row.get("accession", "")).strip()
            if acc in DEAD_UNIPROT_ACCESSIONS:
                removed += 1
                continue
            kept.append(row)
        with path.open("w", encoding="utf-8") as f:
            json.dump(kept, f, indent=2)
            f.write("\n")
        print(f"  {path.name}: {before} -> {len(kept)} rows (removed {removed} dead accessions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
