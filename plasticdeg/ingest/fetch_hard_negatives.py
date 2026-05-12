"""
Phase 4 lite — Swiss-Prot lipase / carboxylesterase hydrolases as hard negatives.

UniProt search: reviewed EC 3.1.1.* or 3.1.2.* (triglyceride lipases, carboxylesterases),
length window similar to positives. Strict subtraction of all PAZy / ground-truth
positive accessions so probes are not training labels.

Writes under data/processed_v2/probes/:
  - hard_negatives.fasta
  - hard_negatives_gt.csv  (same columns as negatives_gt.csv)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from io import StringIO
from pathlib import Path

import requests
from Bio import SeqIO

from plasticdeg import paths
from plasticdeg.ingest.fetch_sequences import fetch_fasta_serial, normalize_accession
from plasticdeg.ingest.sample_negatives import UNIPROT_SEARCH, _parse_next_search_url, _search_page_accs


def load_positive_accessions_union(
    positives_csv: Path,
    positives_json: Path,
) -> set[str]:
    accs: set[str] = set()
    if positives_csv.exists():
        with positives_csv.open(encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                a = (row.get("accession") or "").strip()
                if a:
                    accs.add(a)
                    if "." in a and a.rsplit(".", 1)[-1].isdigit():
                        accs.add(a.rsplit(".", 1)[0])
    if positives_json.exists():
        import json

        with positives_json.open(encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            a = normalize_accession(str(row.get("accession", "")))
            if a and a != "REPLACE_WITH_UNIPROT_ACCESSION":
                accs.add(a)
    return accs


def collect_hard_negative_accessions(
    session: requests.Session,
    exclude: set[str],
    target: int,
    *,
    min_len: int = 120,
    max_len: int = 700,
    page_size: int = 500,
    max_pages: int = 200,
    sleep_s: float = 0.55,
) -> list[str]:
    """
    EC 3.1.1.* = carboxylic-ester hydrolases (includes triacylglycerol lipases).
    EC 3.1.2.* = thioester hydrolases (subset of related ester chemistry).
    """
    q = (
        f"reviewed:true AND length:[{min_len} TO {max_len}] AND "
        "(ec:3.1.1.* OR ec:3.1.2.*)"
    )
    base_params = {
        "query": q,
        "format": "json",
        "fields": "accession",
        "size": page_size,
    }
    out: list[str] = []
    seen: set[str] = set()
    next_url: str | None = None
    for _page in range(max_pages):
        if next_url is None:
            r = session.get(UNIPROT_SEARCH, params=base_params, timeout=90)
        else:
            r = session.get(next_url, timeout=90)
        r.raise_for_status()
        batch = _search_page_accs(r)
        if not batch:
            break
        for acc in batch:
            base = acc.rsplit(".", 1)[0] if "." in acc and acc.rsplit(".", 1)[-1].isdigit() else acc
            if acc in exclude or base in exclude or acc in seen:
                continue
            seen.add(acc)
            out.append(acc)
            if len(out) >= target:
                return out[:target]
        next_url = _parse_next_search_url(r.headers.get("Link"))
        if not next_url:
            break
        time.sleep(sleep_s)
    return out[:target]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fetch Swiss-Prot lipase/esterase hard negatives")
    parser.add_argument(
        "--positives-csv",
        type=Path,
        default=paths.positives_gt_csv(),
    )
    parser.add_argument(
        "--positives-json",
        type=Path,
        default=paths.project_root() / "data" / "pazy" / "ground_truth_merged.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
    )
    parser.add_argument("--target", type=int, default=80, help="Number of hard negatives to fetch")
    parser.add_argument("--sleep-uniprot", type=float, default=0.45)
    parser.add_argument("--sleep-search", type=float, default=0.55)
    parser.add_argument("--max-pages", type=int, default=200)
    args = parser.parse_args(argv)

    exclude = load_positive_accessions_union(args.positives_csv, args.positives_json)
    print(f"  Excluding {len(exclude)} positive / alias accessions", flush=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "plasticdeg/1.0 (hard negative sampling)"})

    accs = collect_hard_negative_accessions(
        session,
        exclude,
        args.target,
        max_pages=args.max_pages,
        sleep_s=args.sleep_search,
    )
    if len(accs) < args.target:
        print(
            f"  WARNING: only {len(accs)} accessions (wanted {args.target}). "
            "Try raising --max-pages or widening length window in code.",
            flush=True,
        )
    if not accs:
        print("ERROR: no hard negatives collected.", file=sys.stderr)
        return 1

    pdir = args.out_dir / "probes"
    pdir.mkdir(parents=True, exist_ok=True)
    fasta_path = pdir / "hard_negatives.fasta"
    csv_path = pdir / "hard_negatives_gt.csv"

    combined, blocks = fetch_fasta_serial(
        accs,
        session=session,
        pause_s=args.sleep_uniprot,
        max_retries=6,
        progress_every=25,
    )
    fasta_path.write_text(combined, encoding="utf-8")
    print(f"  Wrote {fasta_path}", flush=True)

    note = (
        "UniProt Swiss-Prot EC 3.1.1.* / 3.1.2.* (lipase/esterase hard negative); "
        "Phase 4 probe"
    )
    records: list[dict] = []
    for requested_acc, block in blocks:
        if not block.strip():
            continue
        for record in SeqIO.parse(StringIO(block.strip() + "\n"), "fasta"):
            seq = str(record.seq)
            records.append(
                {
                    "accession": requested_acc,
                    "sequence": seq,
                    "length": len(seq),
                    "label": 0,
                    "source_note": note,
                }
            )
            break

    fieldnames = ["accession", "sequence", "length", "label", "source_note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"  Wrote {csv_path} ({len(records)} rows)", flush=True)
    print("  Next: conda run --no-capture-output -n plasticdeg python -u -m plasticdeg.embed.embed_hard_negatives", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
