"""
Expand the gold positive pool via BLASTp against external plastic-enzyme FASTAs.

1. Optionally build BLAST databases (makeblastdb -dbtype prot) under a scratch dir (default %%TEMP%%/plasticdeg_expand_blast) so paths have no spaces (Windows BLAST quirk).
2. blastp gold queries vs each DB with evalue 1e-10 and custom outfmt (includes qcovs).
3. Filter: pident >= 90, qcovs >= 90; exclude accessions already in the gold set.
4. Emit Gold (tier 1) + Silver (tier 2) CSV/FASTA and a JSON log.

Requires: NCBI BLAST+ on PATH (blastp, makeblastdb), pandas, biopython.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
from Bio import SeqIO

from plasticdeg import paths


# UniProt-style accession in headers / BLAST subject fields
_UNIPROT_LIKE = re.compile(
    r"\b([OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2})(\.\d+)?\b",
    re.I,
)
_REFSEQ_WP = re.compile(r"\b(WP_[0-9]{9})(\.\d+)?\b", re.I)
_GENBANK_PROT = re.compile(r"\b([A-Z]{3}\d{5})(\.\d+)?\b")


def _strip_uniprot_version(acc: str) -> str:
    acc = acc.strip()
    if "." in acc and acc.rsplit(".", 1)[-1].isdigit():
        return acc.rsplit(".", 1)[0]
    return acc


def _find_blast_binaries(
    blastp_arg: Path | None,
    makeblastdb_arg: Path | None,
) -> tuple[str | None, str | None]:
    def _one(explicit: Path | None, names: tuple[str, ...]) -> str | None:
        if explicit is not None:
            p = Path(os.path.normpath(os.path.expandvars(str(explicit)))).expanduser()
            cand = [p, p.with_suffix(".exe")] if p.suffix.lower() != ".exe" else [p]
            for c in cand:
                if c.is_file():
                    return str(c.resolve())
        for name in names:
            w = shutil.which(name)
            if w:
                return w
        return None

    bp = _one(blastp_arg, ("blastp", "blastp.exe"))
    mb = _one(makeblastdb_arg, ("makeblastdb", "makeblastdb.exe"))
    return bp, mb


def _blast_db_files_complete(db_prefix: Path) -> bool:
    """NCBI protein DB: basename.pin alongside basename.phr, .psq."""
    pin = db_prefix.parent / f"{db_prefix.name}.pin"
    return pin.is_file()


def _ensure_blast_db(
    fasta: Path,
    db_out: Path,
    makeblastdb: str,
) -> Path:
    """
    Build BLAST DB with basename db_out (no extension).
    Returns path to pass to blastp -db (same as db_out, no extension).
    """
    db_out.parent.mkdir(parents=True, exist_ok=True)
    if _blast_db_files_complete(db_out):
        return db_out
    cmd = [
        makeblastdb,
        "-in",
        str(fasta.resolve()),
        "-dbtype",
        "prot",
        "-out",
        str(db_out.resolve()),
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        raise RuntimeError(f"makeblastdb failed for {fasta} (exit {r.returncode})")
    return db_out


def _run_blastp(
    blastp: str,
    query: Path,
    db_prefix: Path,
    evalue: float,
    out_tsv: Path,
) -> None:
    outfmt = (
        "6 qaccver saccver pident length mismatch gapopen qstart qend sstart send "
        "evalue bitscore qcovs"
    )
    cmd = [
        blastp,
        "-query",
        str(query.resolve()),
        "-db",
        str(db_prefix.resolve()),
        "-evalue",
        str(evalue),
        "-outfmt",
        outfmt,
        "-max_hsps",
        "1",
        "-max_target_seqs",
        "500",
        "-num_threads",
        "4",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
    if r.returncode != 0:
        print(r.stderr or r.stdout, file=sys.stderr)
        raise RuntimeError(f"blastp failed (exit {r.returncode})")
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    out_tsv.write_text(r.stdout, encoding="utf-8")


def _accession_from_gold_header(record_id: str, description: str) -> str | None:
    text = f"{record_id} {description}".strip()
    if "|" in text and (text.startswith("sp|") or text.startswith("tr|")):
        parts = text.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1].strip()
    m = _UNIPROT_LIKE.search(text)
    if m:
        return m.group(1).upper() + (m.group(3) or "")
    return None


def _gold_accessions_from_fasta(path: Path) -> set[str]:
    accs: set[str] = set()
    for rec in SeqIO.parse(path.open(encoding="utf-8", errors="replace"), "fasta"):
        acc = _accession_from_gold_header(rec.id, rec.description)
        if acc:
            accs.add(acc)
            accs.add(_strip_uniprot_version(acc))
    return accs


def _gold_unique_count(path: Path) -> int:
    bases: set[str] = set()
    for rec in SeqIO.parse(path.open(encoding="utf-8", errors="replace"), "fasta"):
        acc = _accession_from_gold_header(rec.id, rec.description)
        if acc:
            bases.add(_strip_uniprot_version(acc))
    return len(bases)


def _tokens_from_header(header: str) -> list[str]:
    """Split FASTA header into tokens for accession-like matching."""
    h = header.strip()
    if h.startswith(">"):
        h = h[1:].strip()
    parts = re.split(r"[\s|]+", h)
    return [p for p in parts if p]


def _build_subject_index(fasta: Path, source_label: str) -> dict[str, tuple[str, str]]:
    """
    Map various keys -> (sequence, source_label).
    Keys: full record.id, stripped versions, regex hits from header.
    """
    index: dict[str, tuple[str, str]] = {}
    for rec in SeqIO.parse(fasta.open(encoding="utf-8", errors="replace"), "fasta"):
        seq = str(rec.seq).replace("\n", "").replace(" ", "").upper()
        if not seq:
            continue
        blob = f"{rec.id} {rec.description}"
        keys: set[str] = {rec.id.strip()}
        for tok in _tokens_from_header(blob):
            keys.add(tok)
            keys.add(_strip_uniprot_version(tok))
        for m in _UNIPROT_LIKE.finditer(blob):
            keys.add(m.group(1).upper() + (m.group(3) or ""))
            keys.add(m.group(1).upper())
        for m in _REFSEQ_WP.finditer(blob):
            keys.add(m.group(1).upper() + (m.group(2) or ""))
            keys.add(m.group(1).upper())
        for m in _GENBANK_PROT.finditer(blob):
            keys.add(m.group(1).upper() + (m.group(2) or ""))
            keys.add(m.group(1).upper())
        for k in keys:
            if not k:
                continue
            index[k] = (seq, source_label)
    return index


def _lookup_sequence(
    saccver: str,
    indices: list[tuple[str, dict[str, tuple[str, str]]]],
) -> tuple[str, str] | None:
    """Return (sequence, source_db) or None."""
    candidates = [saccver.strip(), _strip_uniprot_version(saccver.strip())]
    for c in list(candidates):
        if "|" in c:
            for part in c.split("|"):
                candidates.append(part.strip())
    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        for source_db, idx in indices:
            if cand in idx:
                return idx[cand]
    return None


def _resolve_input_fasta(user_path: Path, fallbacks: list[Path]) -> Path:
    if user_path.exists():
        return user_path
    for fb in fallbacks:
        if fb.exists():
            print(f"  INFO: using fallback for {user_path.name}: {fb}", flush=True)
            return fb
    return user_path


def main(argv: list[str] | None = None) -> int:
    root = paths.project_root()
    ext = paths.external_dir()
    parser = argparse.ArgumentParser(description="BLAST-expand positives (Gold + Silver tiers)")
    parser.add_argument(
        "--query-fasta",
        type=Path,
        default=paths.positives_from_gt_fasta(),
    )
    parser.add_argument(
        "--plasticdb-fasta",
        type=Path,
        default=paths.plasticdb_fasta_default(),
    )
    parser.add_argument(
        "--pmbd-fasta",
        type=Path,
        default=ext / "PMBD.fasta",
    )
    parser.add_argument(
        "--plastenz-fasta",
        type=Path,
        default=ext / "PlasticEnz.fasta",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
    )
    parser.add_argument("--evalue", type=float, default=1e-10)
    parser.add_argument("--min-pident", type=float, default=90.0)
    parser.add_argument("--min-qcovs", type=float, default=90.0)
    parser.add_argument("--skip-missing-db", action="store_true", help="Skip FASTA files that do not exist")
    parser.add_argument(
        "--blastp",
        type=Path,
        default=None,
        help="Full path to blastp.exe if not on PATH",
    )
    parser.add_argument(
        "--makeblastdb",
        type=Path,
        default=None,
        help="Full path to makeblastdb.exe if not on PATH",
    )
    parser.add_argument(
        "--blast-work-dir",
        type=Path,
        default=None,
        help="Directory for BLAST DBs and temp FASTA copies (default: %%TEMP%%/plasticdeg_expand_blast; "
        "avoids spaces in project paths on Windows).",
    )
    args = parser.parse_args(argv)

    blastp, makeblastdb = _find_blast_binaries(args.blastp, args.makeblastdb)
    if not blastp or not makeblastdb:
        if args.blastp and not Path(args.blastp).expanduser().is_file():
            print(f"ERROR: --blastp path is not a file: {args.blastp}", file=sys.stderr)
        if args.makeblastdb and not Path(args.makeblastdb).expanduser().is_file():
            print(f"ERROR: --makeblastdb path is not a file: {args.makeblastdb}", file=sys.stderr)
        print(
            "ERROR: blastp and/or makeblastdb not found.\n"
            "  Install NCBI BLAST+ for Windows: https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/LATEST/\n"
            "  (download win64 .exe installer or .zip, extract, then use the real path under ...\\bin\\)\n"
            "  Or add that `bin` folder to your user PATH and open a new terminal.\n"
            "  Optional (Linux/macOS/WSL): conda install -c bioconda blast",
            file=sys.stderr,
        )
        return 1

    plasticdb = _resolve_input_fasta(
        args.plasticdb_fasta,
        [root / "data" / "FASTA" / "PlasticDB (3).fasta", root / "data" / "exported" / "PlasticDB (2).fasta"],
    )
    pmbd = args.pmbd_fasta
    plastenz = _resolve_input_fasta(
        args.plastenz_fasta,
        [root / "data" / "FASTA" / "PlastEnz_db_fetch.fasta"],
    )

    db_specs: list[tuple[str, Path]] = []
    for label, path in (
        ("PlasticDB", plasticdb),
        ("PMBD", pmbd),
        ("PlasticEnz", plastenz),
    ):
        if not path.exists():
            if args.skip_missing_db:
                print(f"  WARNING: skip missing FASTA: {path}", flush=True)
                continue
            print(
                f"ERROR: FASTA not found: {path}\n"
                f"  Create {ext} and copy/symlink files, or pass --plasticdb-fasta / --pmbd-fasta / --plastenz-fasta.\n"
                f"  (PlasticDB fallback tried data/FASTA; PlasticEnz fallback tried PlastEnz_db_fetch.fasta.)",
                file=sys.stderr,
            )
            return 1
        db_specs.append((label, path))

    if not db_specs:
        print("ERROR: no database FASTA files available (all paths missing?).", file=sys.stderr)
        return 1

    if not args.query_fasta.exists():
        print(f"ERROR: query FASTA missing: {args.query_fasta}", file=sys.stderr)
        return 1

    gold_acc = _gold_accessions_from_fasta(args.query_fasta)
    if not gold_acc:
        print("ERROR: no accessions parsed from gold FASTA.", file=sys.stderr)
        return 1

    if args.blast_work_dir is not None:
        blast_work = Path(args.blast_work_dir).expanduser().resolve()
    else:
        blast_work = Path(tempfile.gettempdir()).resolve() / "plasticdeg_expand_blast"
    blast_work.mkdir(parents=True, exist_ok=True)
    query_local = blast_work / "query_gold.fasta"
    shutil.copy2(args.query_fasta, query_local)
    print(f"  BLAST scratch dir (no spaces): {blast_work}", flush=True)

    indices: list[tuple[str, dict[str, tuple[str, str]]]] = []
    for label, fasta in db_specs:
        idx = _build_subject_index(fasta, label)
        indices.append((label, idx))
        print(f"  Indexed {label}: {len(idx)} keys from {fasta.name}", flush=True)

    log: dict[str, Any] = {
        "blast_work_dir": str(blast_work),
        "gold_unique_sequences": _gold_unique_count(args.query_fasta),
        "databases": {},
        "silver_before_dedup": 0,
        "silver_after_gold_exclude": 0,
        "silver_with_sequence": 0,
        "overlap_between_dbs": {},
    }

    all_hits: list[dict[str, Any]] = []
    for label, fasta in db_specs:
        subj_local = blast_work / f"subject_{label}.fasta"
        shutil.copy2(fasta, subj_local)
        db_stem = blast_work / f"{label}_blastdb"
        try:
            _ensure_blast_db(subj_local, db_stem, makeblastdb)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1
        out_tsv = blast_work / f"blast_raw_{label}.tsv"
        try:
            _run_blastp(blastp, query_local, db_stem, args.evalue, out_tsv)
        except RuntimeError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            return 1

        raw = out_tsv.read_text(encoding="utf-8", errors="replace").strip()
        cols = [
            "qaccver",
            "saccver",
            "pident",
            "length",
            "mismatch",
            "gapopen",
            "qstart",
            "qend",
            "sstart",
            "send",
            "evalue",
            "bitscore",
            "qcovs",
        ]
        n_lines = len([ln for ln in raw.splitlines() if ln.strip()]) if raw else 0
        if n_lines == 0:
            log["databases"][label] = {"raw_hits": 0, "after_filter": 0}
            continue

        df = pd.read_csv(StringIO(raw), sep="\t", names=cols, dtype=str)
        for c in ("pident", "qcovs", "bitscore", "evalue"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df_f = df[(df["pident"] >= args.min_pident) & (df["qcovs"] >= args.min_qcovs)].copy()
        log["databases"][label] = {"raw_hits": int(len(df)), "after_filter": int(len(df_f))}

        for _, row in df_f.iterrows():
            sacc = str(row["saccver"]).strip()
            base = _strip_uniprot_version(sacc.split()[0])
            all_hits.append(
                {
                    "saccver": sacc,
                    "accession_key": base,
                    "pident": float(row["pident"]),
                    "qcovs": float(row["qcovs"]),
                    "bitscore": float(row["bitscore"]),
                    "blast_db": label,
                }
            )

    log["silver_before_dedup"] = len(all_hits)

    # Exclude gold (match stripped accession)
    silver_rows: dict[str, dict[str, Any]] = {}
    for h in all_hits:
        key = h["accession_key"]
        variants = {key, _strip_uniprot_version(key), h["saccver"].split()[0]}
        if gold_acc.intersection(variants):
            continue
        prev = silver_rows.get(key)
        if prev is None or h["bitscore"] > prev["bitscore"]:
            silver_rows[key] = h

    log["silver_after_gold_exclude"] = len(silver_rows)

    gold_rows: dict[str, dict[str, Any]] = {}
    for rec in SeqIO.parse(args.query_fasta.open(encoding="utf-8", errors="replace"), "fasta"):
        acc = _accession_from_gold_header(rec.id, rec.description)
        if not acc:
            continue
        seq = str(rec.seq).replace("\n", "").replace(" ", "")
        base = _strip_uniprot_version(acc)
        if base not in gold_rows:
            gold_rows[base] = {
                "accession": base,
                "sequence": seq,
                "source_db": "PAZy_ground_truth",
                "tier": 1,
            }
    rows_out: list[dict[str, Any]] = sorted(gold_rows.values(), key=lambda r: r["accession"])

    accession_overlap: defaultdict[str, set[str]] = defaultdict(set)
    for h in all_hits:
        if not gold_acc.intersection({h["accession_key"], _strip_uniprot_version(h["saccver"])}):
            accession_overlap[h["accession_key"]].add(h["blast_db"])

    multi = sum(1 for _k, dbs in accession_overlap.items() if len(dbs) > 1)
    log["overlap_between_dbs"] = {"accessions_hit_by_multiple_dbs": int(multi)}

    silver_with_seq = 0
    for key, h in sorted(silver_rows.items(), key=lambda kv: -kv[1]["bitscore"]):
        looked = _lookup_sequence(h["saccver"], indices)
        if looked is None:
            continue
        seq, src = looked
        rows_out.append(
            {
                "accession": key,
                "sequence": seq,
                "source_db": f"BLAST_{src}",
                "tier": 2,
            }
        )
        silver_with_seq += 1
    log["silver_with_sequence"] = silver_with_seq

    if silver_with_seq == 0 and log["silver_after_gold_exclude"] > 0:
        print(
            "  WARNING: Silver hits passed filters but none resolved to a sequence in the FASTA indices. "
            "Check saccver vs FASTA headers.",
            flush=True,
        )

    exp_dir = args.out_dir / "expansion"
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "expanded_positives.csv"
    fasta_path = exp_dir / "expanded_positives.fasta"
    json_path = exp_dir / "blast_expansion_log.json"

    log["total_rows"] = len(rows_out)
    log["tier1_count"] = sum(1 for r in rows_out if r["tier"] == 1)
    log["tier2_count"] = sum(1 for r in rows_out if r["tier"] == 2)
    log["parameters"] = {
        "evalue": args.evalue,
        "min_pident": args.min_pident,
        "min_qcovs": args.min_qcovs,
    }

    pd.DataFrame(rows_out).to_csv(csv_path, index=False)
    with fasta_path.open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows_out):
            acc = r["accession"]
            tier = r["tier"]
            src = r["source_db"]
            fh.write(f">{acc}|tier={tier}|{src}\n")
            seq = r["sequence"]
            for j in range(0, len(seq), 60):
                fh.write(seq[j : j + 60] + "\n")

    json_path.write_text(json.dumps(log, indent=2), encoding="utf-8")
    print(f"  Wrote {csv_path} ({len(rows_out)} rows)", flush=True)
    print(f"  Wrote {fasta_path}", flush=True)
    print(f"  Wrote {json_path}", flush=True)
    print(
        f"  Summary: Gold tier={log['tier1_count']}, Silver tier={log['tier2_count']}, total={log['total_rows']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
