"""
Near-neighbor (BLAST) negatives for a stronger stress test than broad EC buckets.

Seeds: UniProt accessions that appear as *train-split positives* (non-NEG_ ids).
BLASTp against Swiss-Prot; keep hits with identity in [min_pident, max_pident],
exclude every PAZy / ground-truth positive accession, dedupe, fetch FASTA.

Modes:
  - **Local** (recommended): install NCBI BLAST+ and a Swiss-Prot BLAST DB, pass
    --blast-db path/to/swissprot. If `blastp` is not on PATH (common on Windows), pass
    --blastp "C:\\...\\blast-2.xx.x+\\bin\\blastp.exe".
  - **Remote** (slow, rate-limited): omit --blast-db; uses Biopython NCBIWWW.qblast
    for up to --remote-max-queries seed sequences (~12+ s between calls per NCBI).

Outputs under data/processed_v2/probes/:
  - adversarial_negatives.fasta
  - adversarial_negatives_gt.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import OrderedDict
from io import StringIO
from pathlib import Path

import requests
from Bio import SeqIO

from plasticdeg import paths
from plasticdeg.ingest.fetch_hard_negatives import load_positive_accessions_union
from plasticdeg.ingest.fetch_sequences import fetch_fasta_serial


def _strip_version(acc: str) -> str:
    if "." in acc and acc.rsplit(".", 1)[-1].isdigit():
        return acc.rsplit(".", 1)[0]
    return acc


def _parse_sseqid(raw: str) -> str:
    """Extract UniProt accession from BLAST subject id / NCBI alignment title."""
    s = raw.strip().split()[0]
    m = re.search(
        r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}",
        s,
        re.I,
    )
    if m:
        return m.group(0).upper()
    if "|" in s:
        parts = s.split("|")
        if len(parts) >= 2 and len(parts[1]) >= 4:
            return parts[1].upper()
    return s.upper()


def _train_positive_accessions(train_ids_path: Path, positives_csv: Path) -> list[tuple[str, str]]:
    train = {ln.strip() for ln in train_ids_path.read_text(encoding="utf-8").splitlines() if ln.strip()}
    pos_ids = sorted(s for s in train if not s.startswith("NEG_"))
    seq_by_acc: dict[str, str] = {}
    with positives_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if acc and seq:
                seq_by_acc[acc] = seq
                seq_by_acc.setdefault(_strip_version(acc), seq)
    out: list[tuple[str, str]] = []
    for acc in pos_ids:
        seq = seq_by_acc.get(acc) or seq_by_acc.get(_strip_version(acc))
        if seq:
            out.append((acc, seq))
    return out


def _resolve_blastp(explicit: Path | None) -> tuple[str | None, str]:
    """
    Resolve blastp executable. On Windows, --blastp may omit .exe or use env vars.
    Returns (path_or_none, error_hint).
    """
    if explicit is not None:
        raw = Path(os.path.normpath(os.path.expandvars(str(explicit)))).expanduser()
        candidates: list[Path] = [raw]
        if sys.platform == "win32":
            if raw.suffix.lower() != ".exe":
                candidates.append(raw.with_suffix(".exe"))
                candidates.append(raw.parent / "blastp.exe")
        seen: set[str] = set()
        for cand in candidates:
            key = str(cand)
            if key in seen:
                continue
            seen.add(key)
            try:
                if cand.is_file():
                    return str(cand.resolve()), ""
            except OSError:
                continue
        checked = ", ".join(str(c) for c in candidates)
        return (
            None,
            f"--blastp is not an existing file (checked: {checked}). "
            "Fix the path in Explorer (NCBI version folder name changes, e.g. blast-2.15.0+).",
        )

    for name in ("blastp", "blastp.exe"):
        p = shutil.which(name)
        if p:
            return p, ""
    return None, "blastp not on PATH."


def _blast_local_collect(
    seeds: list[tuple[str, str]],
    blast_db: Path,
    blastp_exe: str,
    *,
    evalue: float,
    min_pident: float,
    max_pident: float,
    max_hits_per_query: int,
    exclude: set[str],
) -> OrderedDict[str, float]:
    """Returns ordered accession -> best pident seen."""
    hits: OrderedDict[str, float] = OrderedDict()
    with tempfile.NamedTemporaryFile(mode="w", suffix=".fa", delete=False, encoding="utf-8") as tf:
        qpath = Path(tf.name)
        for acc, seq in seeds:
            tf.write(f">{acc}\n{seq}\n")
    try:
        cmd = [
            blastp_exe,
            "-query",
            str(qpath),
            "-db",
            str(blast_db),
            "-outfmt",
            "6 sseqid pident evalue",
            "-evalue",
            str(evalue),
            "-max_target_seqs",
            str(max_hits_per_query),
            "-num_threads",
            "4",
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            raise RuntimeError(f"blastp failed (exit {r.returncode})")
        for line in r.stdout.splitlines():
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            sid, pident_s, _ = parts[0], parts[1], parts[2]
            try:
                pident = float(pident_s)
            except ValueError:
                continue
            if not (min_pident <= pident <= max_pident):
                continue
            hit = _parse_sseqid(sid)
            hb = _strip_version(hit)
            if hit in exclude or hb in exclude:
                continue
            if hit not in hits or pident > hits[hit]:
                hits[hit] = pident
    finally:
        qpath.unlink(missing_ok=True)
    return hits


def _blast_remote_collect(
    seeds: list[tuple[str, str]],
    *,
    max_queries: int,
    sleep_s: float,
    evalue: float,
    min_pident: float,
    max_pident: float,
    max_hits_per_query: int,
    exclude: set[str],
) -> OrderedDict[str, float]:
    from Bio.Blast import NCBIWWW, NCBIXML

    hits: OrderedDict[str, float] = OrderedDict()
    for i, (_acc, seq) in enumerate(seeds[:max_queries]):
        print(f"  Remote BLAST query {i + 1}/{min(max_queries, len(seeds))} ...", flush=True)
        try:
            handle = NCBIWWW.qblast(
                "blastp",
                "swissprot",
                seq,
                hitlist_size=max_hits_per_query,
                expect=evalue,
            )
        except Exception as e:
            print(f"  WARNING: qblast failed for query {i + 1}: {e}", file=sys.stderr, flush=True)
            time.sleep(sleep_s)
            continue
        try:
            records = NCBIXML.parse(handle)
            rec = next(records, None)
            if rec is None:
                continue
            for alignment in rec.alignments:
                hit = _parse_sseqid(alignment.title)
                hb = _strip_version(hit)
                if hit in exclude or hb in exclude:
                    continue
                best_id = 0.0
                for hsp in alignment.hsps:
                    if hsp.align_length and hsp.identities is not None:
                        pid = 100.0 * hsp.identities / max(hsp.align_length, 1)
                        best_id = max(best_id, pid)
                if min_pident <= best_id <= max_pident:
                    if hit not in hits or best_id > hits[hit]:
                        hits[hit] = best_id
        finally:
            handle.close()
        time.sleep(sleep_s)
    return hits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BLAST-based near-neighbor negatives")
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=paths.split_train_txt(),
    )
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
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=12,
        help="How many train positives to use as BLAST queries (local uses all in one multi-FASTA)",
    )
    parser.add_argument(
        "--blast-db",
        type=Path,
        default=None,
        help="Swiss-Prot BLAST database path (no file extension), e.g. C:/blastdb/swissprot/swissprot",
    )
    parser.add_argument(
        "--blastp",
        type=Path,
        default=None,
        help="Full path to blastp.exe if not on PATH (Windows: .../blast-2.16.0+/bin/blastp.exe)",
    )
    parser.add_argument("--evalue", type=float, default=1e-5)
    parser.add_argument("--min-pident", type=float, default=35.0)
    parser.add_argument("--max-pident", type=float, default=92.0)
    parser.add_argument("--max-hits-per-query", type=int, default=150)
    parser.add_argument("--target", type=int, default=120, help="Max distinct hit accessions to keep")
    parser.add_argument(
        "--remote-max-queries",
        type=int,
        default=8,
        help="If no local BLAST DB, run NCBI qblast on this many seeds only (slow)",
    )
    parser.add_argument("--remote-sleep-s", type=float, default=15.0, help="Pause between NCBI calls")
    parser.add_argument("--sleep-uniprot", type=float, default=0.45)
    args = parser.parse_args(argv)

    if not args.train_ids.exists() or not args.positives_csv.exists():
        print("ERROR: need split_train_accessions.txt and positives_gt.csv", file=sys.stderr)
        return 1

    exclude = load_positive_accessions_union(args.positives_csv, args.positives_json)
    seeds_all = _train_positive_accessions(args.train_ids, args.positives_csv)
    if not seeds_all:
        print("ERROR: no train positive sequences resolved.", file=sys.stderr)
        return 1

    import random

    rng = random.Random(42)
    seeds = seeds_all if len(seeds_all) <= args.n_seeds else rng.sample(seeds_all, args.n_seeds)
    seeds.sort(key=lambda x: x[0])

    blastp_path, blastp_err = _resolve_blastp(args.blastp)
    hits: OrderedDict[str, float]
    if args.blast_db is not None:
        if not blastp_path:
            print(f"ERROR: {blastp_err}", file=sys.stderr)
            print(
                "  Install NCBI BLAST+, then either add its `bin` to PATH or pass the real path to blastp.exe, e.g.\n"
                '  --blastp "C:\\Program Files\\NCBI\\blast-2.15.0+\\bin\\blastp.exe"\n'
                "  Or omit --blast-db to use slow remote qblast (no local DB).",
                file=sys.stderr,
            )
            return 1
        print(f"  Local blastp ({blastp_path}) vs db {args.blast_db} ({len(seeds)} queries in one file)", flush=True)
        hits = _blast_local_collect(
            seeds,
            args.blast_db,
            blastp_path,
            evalue=args.evalue,
            min_pident=args.min_pident,
            max_pident=args.max_pident,
            max_hits_per_query=args.max_hits_per_query,
            exclude=exclude,
        )
    else:
        if not blastp_path:
            print("  No --blast-db: using remote NCBI qblast (slow). Install BLAST+ for local runs.", flush=True)
        else:
            print("  No --blast-db: using remote NCBI qblast (slow). Pass --blast-db for local BLAST.", flush=True)
        hits = _blast_remote_collect(
            seeds,
            max_queries=args.remote_max_queries,
            sleep_s=args.remote_sleep_s,
            evalue=args.evalue,
            min_pident=args.min_pident,
            max_pident=args.max_pident,
            max_hits_per_query=args.max_hits_per_query,
            exclude=exclude,
        )

    acc_list = list(hits.keys())[: args.target]
    if not acc_list:
        print("ERROR: no BLAST hits passed filters.", file=sys.stderr)
        return 1

    print(f"  Collected {len(acc_list)} hit accessions (cap {args.target})", flush=True)

    pdir = args.out_dir / "probes"
    pdir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "plasticdeg/1.0 (adversarial negatives)"})
    fasta_path = pdir / "adversarial_negatives.fasta"
    csv_path = pdir / "adversarial_negatives_gt.csv"

    combined, blocks = fetch_fasta_serial(
        acc_list,
        session=session,
        pause_s=args.sleep_uniprot,
        max_retries=6,
        progress_every=30,
    )
    fasta_path.write_text(combined, encoding="utf-8")
    print(f"  Wrote {fasta_path}", flush=True)

    note = "BLAST near-neighbor negative (train-seed vs Swiss-Prot); Phase 4 adversarial tier"
    records: list[dict] = []
    for requested_acc, block in blocks:
        if not block.strip():
            continue
        for record in SeqIO.parse(StringIO(block.strip() + "\n"), "fasta"):
            seq = str(record.seq)
            pid = hits.get(requested_acc, hits.get(_strip_version(requested_acc), 0.0))
            records.append(
                {
                    "accession": requested_acc,
                    "sequence": seq,
                    "length": len(seq),
                    "label": 0,
                    "source_note": f"{note}; blast_pident≈{pid:.1f}",
                }
            )
            break

    fieldnames = ["accession", "sequence", "length", "label", "source_note"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"  Wrote {csv_path} ({len(records)} rows)", flush=True)
    adv_npz = args.out_dir / "embeddings" / "embeddings_adversarial_negatives.npz"
    adv_json = pdir / "adversarial_negative_probe.json"
    print("  Next:", flush=True)
    print(
        "    conda run --no-capture-output -n plasticdeg python -u -m plasticdeg.embed.embed_hard_negatives "
        f"--csv {csv_path} --out {adv_npz} --id-prefix ADV",
        flush=True,
    )
    print(
        "    conda run --no-capture-output -n plasticdeg python -u -m plasticdeg.eval.probe_hard_negatives "
        f"--hard-embeddings {adv_npz} --out-json {adv_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
