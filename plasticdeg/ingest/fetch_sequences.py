"""
Fetch UniProt sequences for accessions listed in data/pazy/ground_truth.json.

Outputs under artifact root (default data/processed_v2/):
  - sequences/positives_from_gt.fasta
  - tables/positives_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from io import StringIO
from pathlib import Path

import requests
from Bio import SeqIO

from plasticdeg import paths

UNIPROT_FASTA_ONE = "https://rest.uniprot.org/uniprotkb/{accession}.fasta"
REQUEST_PAUSE_S = 0.2


def project_root() -> Path:
    """Repository root (directory that contains `plasticdeg/` and `data/`)."""
    return paths.project_root()


def normalize_accession(raw: str) -> str:
    """Accept G9BY57 or sp|G9BY57|PETH_UNKP -> G9BY57."""
    s = raw.strip()
    if "|" in s and (s.startswith("sp|") or s.startswith("tr|")):
        parts = s.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    return s


def load_ground_truth(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("ground_truth.json must be a JSON array of objects")
    return data


def fetch_fasta_serial(
    accessions: list[str],
    timeout: int = 60,
    *,
    session: requests.Session | None = None,
    pause_s: float | None = None,
    max_retries: int = 5,
    progress_every: int = 0,
) -> tuple[str, list[tuple[str, str]]]:
    """
    One GET per accession.

    Returns (concatenated FASTA for file on disk, list of (requested_accession, block_text)).
    The requested accession is preserved so merged IDs (query AC ≠ FASTA header AC) still
    join to ground_truth.json metadata.
    """
    sess = session if session is not None else requests.Session()
    pause = REQUEST_PAUSE_S if pause_s is None else pause_s
    blocks: list[tuple[str, str]] = []
    file_chunks: list[str] = []
    empty_fasta: list[str] = []
    for i, acc in enumerate(accessions):
        if progress_every and (i + 1) % progress_every == 0:
            print(f"  FASTA progress {i + 1}/{len(accessions)}", flush=True)
        url = UNIPROT_FASTA_ONE.format(accession=acc)
        text = ""
        for attempt in range(max_retries):
            try:
                r = sess.get(url, timeout=timeout)
                if r.status_code == 429:
                    wait = int(r.headers.get("Retry-After", "30"))
                    time.sleep(min(120, max(5, wait)))
                    continue
                if r.status_code == 404:
                    print(f"  WARNING: 404 for {acc}")
                    text = ""
                    break
                r.raise_for_status()
                text = r.text.strip()
                break
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.Timeout,
            ):
                if attempt + 1 >= max_retries:
                    raise
                time.sleep(min(90.0, 1.8 ** (attempt + 1)))
        blocks.append((acc, text))
        if text:
            file_chunks.append(text)
        elif acc:
            empty_fasta.append(acc)
        if i + 1 < len(accessions):
            time.sleep(pause)
    combined = "\n".join(file_chunks) + ("\n" if file_chunks else "")
    if empty_fasta:
        print(
            f"  NOTE: {len(empty_fasta)} accessions returned empty FASTA "
            f"(inactive/deleted at UniProt REST), e.g. {empty_fasta[:5]}{'...' if len(empty_fasta) > 5 else ''}",
            flush=True,
        )
    return combined, blocks


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(description="Fetch UniProt FASTA for ground_truth.json")
    parser.add_argument(
        "--input",
        type=Path,
        default=root / "data" / "pazy" / "ground_truth.json",
        help="Path to ground_truth.json",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
        help="Artifact root (writes tables/ and sequences/ underneath)",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        print(f"ERROR: {args.input} not found. Copy ground_truth.template.json and fill accessions.")
        return 1

    rows_meta = load_ground_truth(args.input)
    by_acc: dict[str, dict] = {}
    accessions: list[str] = []
    for row in rows_meta:
        acc = normalize_accession(str(row.get("accession", "")))
        if not acc or acc == "REPLACE_WITH_UNIPROT_ACCESSION":
            continue
        if acc in by_acc:
            print(f"  WARNING: duplicate accession {acc}, keeping first metadata row")
            continue
        by_acc[acc] = row
        accessions.append(acc)

    if not accessions:
        print("ERROR: no valid accessions after normalization.")
        return 1

    seq_dir = args.out_dir / "sequences"
    tab_dir = args.out_dir / "tables"
    seq_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = seq_dir / "positives_from_gt.fasta"
    csv_path = tab_dir / "positives_gt.csv"

    print(f"  Fetching {len(accessions)} accessions from UniProt...")
    fasta_text, per_acc_blocks = fetch_fasta_serial(accessions)
    fasta_path.write_text(fasta_text, encoding="utf-8")
    print(f"  Wrote {fasta_path}")

    records: list[dict] = []
    missing = set(accessions)
    for requested_acc, block in per_acc_blocks:
        if not block.strip():
            continue
        meta = by_acc.get(requested_acc)
        if meta is None:
            continue
        for record in SeqIO.parse(StringIO(block.strip() + "\n"), "fasta"):
            seq = str(record.seq)
            records.append(
                {
                    "accession": requested_acc,
                    "substrate": meta.get("substrate", ""),
                    "evidence_tier": int(meta.get("evidence_tier", 1)),
                    "source_db": meta.get("source_db", ""),
                    "source_note": meta.get("source_note", ""),
                    "id": record.id,
                    "description": record.description,
                    "sequence": seq,
                    "length": len(seq),
                    "label": 1,
                }
            )
            missing.discard(requested_acc)
            break

    fieldnames = [
        "accession",
        "substrate",
        "evidence_tier",
        "source_db",
        "source_note",
        "id",
        "description",
        "sequence",
        "length",
        "label",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"  Wrote {csv_path} ({len(records)} rows)")

    if missing:
        print(f"  WARNING: no FASTA returned for: {sorted(missing)}")

    return 0 if records else 1


if __name__ == "__main__":
    sys.exit(main())
