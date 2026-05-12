"""
Build data/pazy/ground_truth.json from PAZy "sequences CSV" export.

The export at data/exported/pazy_proteins_sequences.csv has protein_id and
sequence but often NO UniProt column. This script, for each protein_id:

  GET https://pazy.eu/proteins/{id}
  Parse UniProt link: .../uniprotkb/ACCESSION
  Parse plastics:   ...>Plastics: </span><span>PEF, PET</span>...

Writes ground_truth.json with accession, substrate (primary), evidence_tier=1,
source_db=PAZy, source_note including PAZy protein_id and full plastics list.

Requires: requests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

from plasticdeg.support.uniprot_idmap import map_one

PAZY_PROTEIN_URL = "https://pazy.eu/proteins/{protein_id}"
UNIPROT_RE = re.compile(
    r"https?://(?:www\.)?uniprot\.org/uniprotkb/([A-Z0-9]{6,15})", re.I
)
PLASTICS_RE = re.compile(
    r'font-semibold">Plastics:\s*</span><span>([^<]+)</span>', re.I
)
# NCBI RefSeq / INSDC protein accessions linked from PAZy detail pages
REFSEQ_PROTEIN_RE = re.compile(
    r"ncbi\.nlm\.nih\.gov/protein/((?:WP|NP|XP|YP|AP)_\d+(?:\.\d+)?)", re.I
)
ENA_VIEW_RE = re.compile(
    r"ebi\.ac\.uk/ena/browser/view/([A-Z]{1,6}\d{5,12}(?:\.\d+)?)", re.I
)
BLAST_QUERY_RE = re.compile(
    r"blast\.ncbi\.nlm\.nih\.gov/[^\"']*QUERY=([A-Z0-9]+(?:\.\d+)?)", re.I
)

# Primary substrate column: first match in this priority (polyester-class pilot)
SUBSTRATE_PRIORITY = [
    "PET",
    "PEF",
    "PBT",
    "PLA",
    "PCL",
    "PBS",
    "PBAT",
    "PHA",
    "PHB",
]


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def pick_primary_substrate(plastics_raw: str) -> str:
    parts = [p.strip() for p in plastics_raw.split(",") if p.strip()]
    if not parts:
        return "UNKNOWN"
    upper = [p.upper() for p in parts]
    for pref in SUBSTRATE_PRIORITY:
        for i, u in enumerate(upper):
            if u == pref or u.startswith(pref) or pref in u:
                return parts[i]
    return parts[0]


def fetch_pazy_page(protein_id: int, session: requests.Session, timeout: int = 45) -> str:
    url = PAZY_PROTEIN_URL.format(protein_id=protein_id)
    r = session.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def parse_page(html: str) -> tuple[str | None, str | None]:
    um = UNIPROT_RE.search(html)
    acc = um.group(1) if um else None
    pm = PLASTICS_RE.search(html)
    plastics = pm.group(1).strip() if pm else None
    return acc, plastics


def _embl_candidates(raw: str) -> list[str]:
    """Try bare accession then .1 — UniProt INSDC mapping is version-sensitive."""
    raw = raw.strip()
    if not raw:
        return []
    if "." in raw:
        return [raw]
    return [raw, f"{raw}.1"]


def resolve_uniprot_via_idmap(session: requests.Session, html: str) -> tuple[str | None, str | None]:
    """
    When PAZy has no uniprot.org link, try RefSeq / EMBL-GenBank-DDBJ ID mapping.
    Returns (uniprot_accession, note_suffix) or (None, None).
    """
    tried: list[str] = []

    for rs in REFSEQ_PROTEIN_RE.findall(html):
        if rs in tried:
            continue
        tried.append(rs)
        acc = map_one(session, "RefSeq_Protein", rs)
        if acc:
            return acc, f"UniProt via ID map RefSeq_Protein:{rs}"

    ena_ids: list[str] = []
    for m in ENA_VIEW_RE.finditer(html):
        ena_ids.append(m.group(1))
    for m in BLAST_QUERY_RE.finditer(html):
        q = m.group(1)
        if q not in ena_ids:
            ena_ids.append(q)

    for raw in ena_ids:
        for cand in _embl_candidates(raw):
            for from_db in ("EMBL-GenBank-DDBJ", "EMBL-GenBank-DDBJ_CDS"):
                key = f"{from_db}:{cand}"
                if key in tried:
                    continue
                tried.append(key)
                acc = map_one(session, from_db, cand)
                if acc:
                    return acc, f"UniProt via ID map {from_db}:{cand}"
    return None, None


def main(argv: list[str] | None = None) -> int:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Resolve UniProt + plastics from PAZy pages; write ground_truth.json"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=root / "data" / "exported" / "pazy_proteins_sequences.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "data" / "pazy" / "ground_truth.json",
    )
    parser.add_argument("--min-len", type=int, default=50)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--sleep", type=float, default=0.35, help="Seconds between PAZy requests")
    parser.add_argument(
        "--sleep-uniprot",
        type=float,
        default=0.5,
        help="Seconds after each UniProt ID mapping call (when --idmap-fallback)",
    )
    parser.add_argument("--limit", type=int, default=0, help="Process only first N rows (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Print first 5 resolutions only")
    parser.add_argument(
        "--idmap-fallback",
        action="store_true",
        help="If PAZy has no UniProt link, map RefSeq / ENA INSDC IDs via UniProt REST",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"ERROR: CSV not found: {args.csv}")
        return 1

    df = pd.read_csv(args.csv)
    need = {"protein_id", "protein_name", "amino_acid_sequence"}
    missing = need - set(df.columns.str.strip())
    if missing:
        print(f"ERROR: CSV missing columns: {missing}")
        return 1

    session = requests.Session()
    session.headers.update({"User-Agent": "plasticdeg/1.0 (PAZy ground truth builder)"})

    rows_out: list[dict] = []
    errors: list[tuple[int, str]] = []

    n = len(df)
    if args.limit and args.limit > 0:
        n = min(n, args.limit)

    if not args.dry_run:
        est_s = n * (args.sleep + 0.5)
        print(
            f"PAZy resolver: {n} proteins — expect ~{est_s:.0f}s or more (network + polite delay). "
            "Progress every 25 rows.",
            flush=True,
        )

    for i in range(n):
        r = df.iloc[i]
        pid = int(r["protein_id"])
        pname = str(r["protein_name"])
        seq = str(r["amino_acid_sequence"])
        slen = len(seq)
        if slen < args.min_len or slen > args.max_len:
            errors.append((pid, f"length {slen} outside [{args.min_len},{args.max_len}]"))
            continue

        idmap_note: str | None = None
        html = ""
        try:
            html = fetch_pazy_page(pid, session)
            acc, plastics = parse_page(html)
        except Exception as e:
            errors.append((pid, str(e)))
            acc, plastics = None, None

        if not acc:
            if args.idmap_fallback and html:
                acc, idmap_note = resolve_uniprot_via_idmap(session, html)
                if acc:
                    time.sleep(args.sleep_uniprot)
            if not acc:
                errors.append((pid, "no UniProt accession in HTML"))
                if args.dry_run:
                    print(f"  id={pid} {pname}: FAIL no accession")
                if not args.dry_run and ((i + 1) % 25 == 0 or (i + 1) == n):
                    print(
                        f"  row {i + 1}/{n} | resolved UniProt rows: {len(rows_out)}",
                        flush=True,
                    )
                time.sleep(args.sleep)
                continue
        if not plastics:
            errors.append((pid, "no Plastics line in HTML"))
            plastics = ""

        primary = pick_primary_substrate(plastics) if plastics else "UNKNOWN"
        note = f"PAZy protein_id={pid} name={pname}; plastics={plastics}"
        if idmap_note:
            note = f"{note}; {idmap_note}"

        row = {
            "accession": acc,
            "substrate": primary,
            "evidence_tier": 1,
            "source_db": "PAZy",
            "source_note": note,
        }
        rows_out.append(row)

        if args.dry_run and i < 5:
            print(f"  id={pid} {pname} -> {acc} substrate={primary} ({plastics})")

        if not args.dry_run and ((i + 1) % 25 == 0 or (i + 1) == n):
            print(
                f"  row {i + 1}/{n} | resolved: {len(rows_out)} | last: {acc}",
                flush=True,
            )

        time.sleep(args.sleep)

    if args.dry_run:
        print(f"\nDry-run done. Would write {len(rows_out)} rows; errors={len(errors)}")
        return 0

    # Dedupe by accession (keep first occurrence)
    seen: set[str] = set()
    deduped: list[dict] = []
    for row in rows_out:
        a = row["accession"]
        if a in seen:
            continue
        seen.add(a)
        deduped.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(deduped, f, indent=2)
        f.write("\n")

    print(f"  Wrote {args.output} ({len(deduped)} accessions, deduped)")
    if errors:
        err_path = args.output.with_suffix(".build_errors.txt")
        err_path.write_text(
            "\n".join(f"protein_id={pid}: {msg}" for pid, msg in errors[:500]),
            encoding="utf-8",
        )
        print(f"  Logged {len(errors)} issues to {err_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
