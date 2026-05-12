"""
Phase C — sample negative UniProtKB (Swiss-Prot) sequences unrelated to positives.

1) Load positive accessions from ground-truth JSON (merged or PAZy-only).
2) Paginate UniProt search: reviewed + length window (default 50–1024 aa).
3) Strict set subtraction: never include a positive accession.
4) Fetch FASTA per negative (same requested-accession pairing as fetch_sequences).

Outputs (default artifact root `data/processed_v2/`):
  - sequences/negatives_from_uniprot.fasta
  - tables/negatives_gt.csv (accession, sequence, length, label=0, source_note)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from io import StringIO
from pathlib import Path

import requests
from Bio import SeqIO

from plasticdeg import paths
from plasticdeg.ingest.fetch_sequences import fetch_fasta_serial, normalize_accession

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"


def project_root() -> Path:
    return paths.project_root()


def load_positive_accessions(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    out: set[str] = set()
    for row in rows:
        acc = normalize_accession(str(row.get("accession", "")))
        if acc and acc != "REPLACE_WITH_UNIPROT_ACCESSION":
            out.add(acc)
    return out


def _parse_next_search_url(link_header: str | None) -> str | None:
    """UniProt search pagination: follow full URL in Link rel=\"next\" (includes cursor=)."""
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part or "rel='next'" in part:
            m = re.search(r"<([^>]+)>", part)
            return m.group(1).strip() if m else None
    return None


def _search_page_accs(response: requests.Response) -> list[str]:
    data = response.json()
    accs: list[str] = []
    for item in data.get("results", []) or []:
        a = item.get("primaryAccession")
        if a:
            accs.append(a)
    return accs


def collect_negative_accessions(
    session: requests.Session,
    positive_ids: set[str],
    target: int,
    *,
    page_size: int = 500,
    max_pages: int = 8000,
    sleep_s: float = 0.55,
    min_len: int = 50,
    max_len: int = 1024,
) -> list[str]:
    """
    Paginate UniProt search using Link rel=\"next\" (cursor-based). The `from` offset
    alone repeats the first page on current UniProt REST — do not use it for paging.
    """
    q = f"reviewed:true AND length:[{min_len} TO {max_len}]"
    base_params = {
        "query": q,
        "format": "json",
        "fields": "accession",
        "size": page_size,
    }
    out: list[str] = []
    seen: set[str] = set()
    next_url: str | None = None
    for page in range(max_pages):
        if next_url is None:
            r = session.get(UNIPROT_SEARCH, params=base_params, timeout=90)
        else:
            r = session.get(next_url, timeout=90)
        r.raise_for_status()
        batch = _search_page_accs(r)
        if not batch:
            break
        for acc in batch:
            if acc in positive_ids or acc in seen:
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
    root = project_root()
    parser = argparse.ArgumentParser(description="Sample Swiss-Prot negatives via UniProt search")
    parser.add_argument(
        "--positives-json",
        type=Path,
        default=root / "data" / "pazy" / "ground_truth_merged.json",
        help="Positive pool (accessions subtracted from negatives)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
        help="Artifact root (writes sequences/ and tables/)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=0,
        help="Number of negatives to collect (0 = 8 × number of positives in JSON)",
    )
    parser.add_argument(
        "--sleep-uniprot",
        type=float,
        default=0.45,
        help="Pause between FASTA GETs (rate limit friendly)",
    )
    parser.add_argument(
        "--sleep-search",
        type=float,
        default=0.55,
        help="Pause between UniProt search pages",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=8000,
        help="Safety cap on search pages (cursor pagination)",
    )
    args = parser.parse_args(argv)

    if not args.positives_json.exists():
        print(f"ERROR: {args.positives_json} not found")
        return 1

    pos = load_positive_accessions(args.positives_json)
    target = args.target if args.target > 0 else min(5000, len(pos) * 8)
    print(
        f"  Positives (accessions to subtract): {len(pos)} | target negatives: {target}",
        flush=True,
    )

    session = requests.Session()
    session.headers.update({"User-Agent": "plasticdeg/1.0 (negative sampling)"})

    neg_accs = collect_negative_accessions(
        session,
        pos,
        target,
        max_pages=args.max_pages,
        sleep_s=args.sleep_search,
    )
    if len(neg_accs) < target:
        print(
            f"  WARNING: collected only {len(neg_accs)} negatives (requested {target}). "
            "Increase --max-pages or relax query if needed.",
            flush=True,
        )

    # Reuse fetch_fasta_serial but with custom sleep by temporarily patching - simpler: copy loop with sleep
    # Use fetch_sequences module - it uses REQUEST_PAUSE_S; we pass via monkeypatch - instead inline thin wrapper
    seq_dir = args.out_dir / "sequences"
    tab_dir = args.out_dir / "tables"
    seq_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = seq_dir / "negatives_from_uniprot.fasta"
    csv_path = tab_dir / "negatives_gt.csv"

    combined, blocks = fetch_fasta_serial(
        neg_accs,
        session=session,
        pause_s=args.sleep_uniprot,
        max_retries=6,
        progress_every=100,
    )
    fasta_path.write_text(combined, encoding="utf-8")
    print(f"  Wrote {fasta_path}")

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
                    "source_note": "UniProtKB Swiss-Prot search (reviewed:true); Phase C easy negative",
                }
            )
            break

    fieldnames = ["accession", "sequence", "length", "label", "source_note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"  Wrote {csv_path} ({len(records)} rows)")
    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
