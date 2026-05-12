"""
Import PlasticDB-style FASTA into UniProt-keyed rows **without** touching PAZy
`ground_truth.json` unless you explicitly pass `--merged-output`.

Reads headers like:
  >00001|PHB_depolymerase|Organism_name|PHB_PHA|BAA04986.1

Filters to polyester-class plastics (v1), maps GenBank/RefSeq IDs via UniProt
ID mapping, resolves UniProt-like tails with a direct FASTA probe, subtracts
accessions already present in PAZy ground truth, and writes reviewable outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests

from plasticdeg import paths
from plasticdeg.support.uniprot_idmap import map_one, run_id_mapping

FASTA_HEADER_RE = re.compile(
    r"^>(?P<pdb_id>\d+)\|(?P<enzyme>[^|]*)\|(?P<organism>[^|]*)\|(?P<plastics>[^|]*)\|(?P<tail>[^|]+)\s*$"
)
REFSEQ_RE = re.compile(r"^(?:WP|NP|XP|YP|AP)_\d+(?:\.\d+)?$", re.I)
# Primary UniProt accessions (excludes weird GenBank-only shapes)
UNIPROT_LIKE_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}|A0A[A-Z0-9]{6,9})(?:\.\d+)?$",
    re.I,
)

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

# v1 pilot: drop obvious out-of-scope chemistries (see data/pazy/README.md spirit)
EXCLUDE_PLASTIC_SUBSTRINGS = (
    "PU",  # polyurethane / PUR family
    "PVA",
    "PEG",
    "O-PVA",
    "PPU",
)

UNIPROT_FASTA = "https://rest.uniprot.org/uniprotkb/{acc}.fasta"


def plastics_field_in_scope(plastics_field: str) -> bool:
    u = plastics_field.upper()
    for bad in EXCLUDE_PLASTIC_SUBSTRINGS:
        if bad in u:
            return False
    for good in SUBSTRATE_PRIORITY:
        if good in u:
            return True
    return False


def pick_primary_substrate(plastics_field: str) -> str:
    parts = [p for p in plastics_field.replace(",", "_").split("_") if p.strip()]
    if not parts:
        return "UNKNOWN"
    upper = [p.upper() for p in parts]
    for pref in SUBSTRATE_PRIORITY:
        for i, tok in enumerate(upper):
            if pref == tok or tok.startswith(pref) or pref in tok:
                return parts[i]
    return parts[0]


def classify_tail(tail: str) -> str:
    t = tail.strip()
    if not t or t.upper() == "NA":
        return "na"
    if REFSEQ_RE.match(t):
        return "refseq"
    if UNIPROT_LIKE_RE.match(t):
        return "uniprot"
    return "embl"


def uniprot_try_variants(session: requests.Session, tail: str, timeout: int = 45) -> str | None:
    """Return first accession variant that yields non-empty FASTA (canonical header)."""
    candidates = [tail.strip()]
    if "." in tail and tail.upper() != "NA":
        base = tail.split(".", 1)[0]
        if base not in candidates:
            candidates.append(base)
    for cand in candidates:
        r = session.get(UNIPROT_FASTA.format(acc=cand), timeout=timeout)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        if r.text.strip():
            return cand
    return None


def load_pazy_accessions(path: Path) -> set[str]:
    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    out: set[str] = set()
    for row in rows:
        acc = str(row.get("accession", "")).strip()
        if acc and acc != "REPLACE_WITH_UNIPROT_ACCESSION":
            out.add(acc)
    return out


def parse_fasta(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith(">"):
                continue
            m = FASTA_HEADER_RE.match(line)
            if not m:
                rows.append(
                    {
                        "raw_header": line[1:],
                        "parse_ok": "0",
                        "enzyme": "",
                        "organism": "",
                        "plastics": "",
                        "tail": "",
                    }
                )
                continue
            rows.append(
                {
                    "raw_header": line[1:],
                    "parse_ok": "1",
                    "enzyme": m.group("enzyme"),
                    "organism": m.group("organism"),
                    "plastics": m.group("plastics"),
                    "tail": m.group("tail").strip(),
                }
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    root = paths.project_root()
    parser = argparse.ArgumentParser(
        description="PlasticDB FASTA → UniProt candidates (does not overwrite PAZy ground_truth.json)"
    )
    parser.add_argument(
        "--fasta",
        type=Path,
        default=root / "data" / "exported" / "PlasticDB (2).fasta",
        help="PlasticDB export FASTA path",
    )
    parser.add_argument(
        "--pazy-ground-truth",
        type=Path,
        default=root / "data" / "pazy" / "ground_truth.json",
        help="Used only to subtract existing UniProt accessions",
    )
    parser.add_argument(
        "--out-additions",
        type=Path,
        default=paths.plasticdb_additions_json(),
    )
    parser.add_argument(
        "--out-skips",
        type=Path,
        default=paths.plasticdb_import_skips_txt(),
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=None,
        help="Optional NEW file: PAZy ground_truth + PlasticDB additions (deduped by accession)",
    )
    parser.add_argument("--sleep-uniprot", type=float, default=0.45)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if not args.fasta.exists():
        print(f"ERROR: FASTA not found: {args.fasta}")
        return 1

    pazy_acc = load_pazy_accessions(args.pazy_ground_truth) if args.pazy_ground_truth.exists() else set()
    parsed = parse_fasta(args.fasta)
    session = requests.Session()
    session.headers.update({"User-Agent": "plasticdeg/1.0 (PlasticDB import)"})

    skips: list[str] = []
    additions: list[dict] = []
    seen_new: set[str] = set()

    # First pass: filter + classify
    work: list[tuple[dict, str, str]] = []  # (meta_row, kind, tail)
    for row in parsed:
        if row.get("parse_ok") != "1":
            skips.append(f"bad_header={row.get('raw_header', '')[:120]}")
            continue
        pl = row["plastics"]
        tail = row["tail"]
        if not plastics_field_in_scope(pl):
            skips.append(f"out_of_scope_plastics tail={tail} plastics={pl}")
            continue
        kind = classify_tail(tail)
        if kind == "na":
            skips.append(f"no_accession plastics={pl} enzyme={row['enzyme'][:40]}")
            continue
        work.append((row, kind, tail))

    if args.dry_run:
        n_rs = sum(1 for _, k, _ in work if k == "refseq")
        n_em = sum(1 for _, k, _ in work if k == "embl")
        n_up = sum(1 for _, k, _ in work if k == "uniprot")
        print(
            f"  dry-run: in_scope_rows={len(work)} "
            f"(uniprot_like={n_up}, refseq={n_rs}, genbank_embl={n_em}); "
            f"no UniProt calls made",
            flush=True,
        )
        return 0

    # Batch map RefSeq + EMBL
    refseq_ids = sorted({t for _, k, t in work if k == "refseq"})
    embl_ids = sorted({t for _, k, t in work if k == "embl"})

    refseq_map: dict[str, str] = {}
    if refseq_ids:
        for i in range(0, len(refseq_ids), 80):
            chunk = refseq_ids[i : i + 80]
            refseq_map.update(
                run_id_mapping(session, "RefSeq_Protein", "UniProtKB", chunk)
            )
            time.sleep(args.sleep_uniprot)

    embl_map: dict[str, str] = {}
    if embl_ids:
        for i in range(0, len(embl_ids), 80):
            chunk = embl_ids[i : i + 80]
            embl_map.update(
                run_id_mapping(session, "EMBL-GenBank-DDBJ", "UniProtKB", chunk)
            )
            time.sleep(args.sleep_uniprot)

    for row, kind, tail in work:
        primary = pick_primary_substrate(row["plastics"])
        note = (
            f"PlasticDB row={row.get('raw_header', '')[:200]}; "
            f"mapped_from={tail}"
        )
        uni: str | None = None
        if kind == "uniprot":
            uni = uniprot_try_variants(session, tail)
            if not uni:
                skips.append(f"uniprot_no_fasta tail={tail}")
                continue
        elif kind == "refseq":
            uni = refseq_map.get(tail) or map_one(session, "RefSeq_Protein", tail)
            if uni:
                time.sleep(args.sleep_uniprot)
            if not uni:
                skips.append(f"refseq_unmapped tail={tail}")
                continue
        else:
            uni = embl_map.get(tail)
            if not uni:
                cands = [tail]
                if "." not in tail:
                    cands.append(f"{tail}.1")
                else:
                    base = tail.rsplit(".", 1)[0]
                    if base != tail:
                        cands.append(base)
                for cand in cands:
                    u = map_one(session, "EMBL-GenBank-DDBJ", cand)
                    time.sleep(args.sleep_uniprot)
                    if u:
                        uni = u
                        break
            if not uni:
                skips.append(f"embl_unmapped tail={tail}")
                continue

        if uni in pazy_acc:
            skips.append(f"already_in_pazy acc={uni} tail={tail}")
            continue
        if uni in seen_new:
            skips.append(f"duplicate_new acc={uni} tail={tail}")
            continue
        seen_new.add(uni)

        additions.append(
            {
                "accession": uni,
                "substrate": primary,
                "evidence_tier": 2,
                "source_db": "PlasticDB",
                "source_note": note,
            }
        )

    print(
        f"PlasticDB import: parsed_headers={len(parsed)} in_scope_work={len(work)} "
        f"new_unique_additions={len(additions)} skips={len(skips)} pazy_subtract={len(pazy_acc)}",
        flush=True,
    )

    args.out_additions.parent.mkdir(parents=True, exist_ok=True)
    with args.out_additions.open("w", encoding="utf-8") as f:
        json.dump(additions, f, indent=2)
        f.write("\n")
    args.out_skips.write_text("\n".join(skips[:5000]) + ("\n" if skips else ""), encoding="utf-8")
    print(f"  Wrote {args.out_additions}")
    print(f"  Wrote {args.out_skips}")

    if args.merged_output:
        with args.pazy_ground_truth.open(encoding="utf-8") as f:
            merged = json.load(f)
        have = {str(r.get("accession", "")).strip() for r in merged}
        for row in additions:
            acc = row["accession"]
            if acc in have:
                continue
            merged.append(row)
            have.add(acc)
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        with args.merged_output.open("w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
            f.write("\n")
        print(f"  Wrote merged list ({len(merged)} rows) -> {args.merged_output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
