"""
Merge BLAST expansion (Gold + Silver) with curated PAZy positives_gt.csv.

Reads:
  - data/processed_v2/expansion/expanded_positives.csv (tier 1 + 2)
  - data/processed_v2/tables/positives_gt.csv (rich PAZy metadata for Gold)

Writes (artifact layout via plasticdeg.paths):
  - tables/positives_gt_expanded.csv
  - sequences/positives_from_gt_expanded.fasta
  - tables/accession_sequence_alias_map.csv (canonical accession -> merged BLAST/DB aliases)

Deduplication: unique accession (UniProt version stripped for collision keys) and
unique sequence (second occurrence dropped; tier-1 preferred over tier-2).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

from plasticdeg import paths


def _write_accession_alias_map(
    expanded_rows: list[dict[str, str]],
    deduped_rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    """Map each final canonical accession to other raw accessions from expansion with the same sequence."""
    fp_to_canonical: dict[str, str] = {}
    for r in deduped_rows:
        fp = _seq_fingerprint(r["sequence"])
        fp_to_canonical[fp] = r["accession"].strip()

    alias_by_canonical: dict[str, set[str]] = {}
    for exp in expanded_rows:
        fp = _seq_fingerprint(exp["sequence"])
        can = fp_to_canonical.get(fp)
        if not can:
            continue
        raw = exp["accession"].strip()
        if not raw or raw == can:
            continue
        if _strip_uniprot_version(raw) == _strip_uniprot_version(can):
            continue
        alias_by_canonical.setdefault(can, set()).add(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "canonical_accession",
        "tier",
        "sequence_sha256",
        "n_aliases",
        "alias_accessions",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in sorted(deduped_rows, key=lambda x: _strip_uniprot_version(x["accession"])):
            acc = r["accession"].strip()
            fp = _seq_fingerprint(r["sequence"])
            al = sorted(alias_by_canonical.get(acc, ()))
            w.writerow(
                {
                    "canonical_accession": acc,
                    "tier": r.get("tier", ""),
                    "sequence_sha256": fp,
                    "n_aliases": len(al),
                    "alias_accessions": ";".join(al),
                }
            )


def _strip_uniprot_version(acc: str) -> str:
    acc = acc.strip()
    if "." in acc and acc.rsplit(".", 1)[-1].isdigit():
        return acc.rsplit(".", 1)[0]
    return acc


def _seq_fingerprint(seq: str) -> str:
    s = seq.strip().upper().replace("\n", "")
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _parse_int(x: str | None, default: int) -> int:
    if x is None or not str(x).strip():
        return default
    try:
        return int(float(str(x).strip()))
    except ValueError:
        return default


def _load_expanded(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if not acc or not seq:
                continue
            tier = _parse_int(row.get("tier"), 1)
            if tier not in (1, 2):
                tier = 1
            rows.append(
                {
                    "accession": acc,
                    "sequence": seq,
                    "source_db": (row.get("source_db") or "").strip(),
                    "tier": str(tier),
                }
            )
    return rows


def write_alias_map_from_files(
    expanded_csv: Path,
    deduped_positives_csv: Path,
    out_alias_csv: Path | None = None,
) -> Path:
    """
    Rebuild accession_sequence_alias_map.csv from BLAST expansion + merged positives
    without rewriting positives_gt_expanded.csv (safe to re-run before deadlines).
    """
    out = out_alias_csv or paths.accession_sequence_alias_map_csv()
    expanded = _load_expanded(expanded_csv)
    deduped: list[dict[str, str]] = []
    with deduped_positives_csv.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            deduped.append({k: (v or "").strip() for k, v in row.items()})
    _write_accession_alias_map(expanded, deduped, out)
    return out


def _load_positives_gt(path: Path) -> dict[str, dict[str, str]]:
    """Key: stripped accession -> full row dict."""
    out: dict[str, dict[str, str]] = {}
    if not path.exists():
        return out
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            if not acc:
                continue
            key = _strip_uniprot_version(acc)
            out[key] = {k: (row.get(k) or "").strip() if row.get(k) is not None else "" for k in row}
    return out


def _merge_row(
    exp: dict[str, str],
    gt_by_base: dict[str, dict[str, str]],
) -> dict[str, str]:
    acc = exp["accession"]
    base = _strip_uniprot_version(acc)
    tier = int(exp["tier"])
    seq = exp["sequence"]
    gt = gt_by_base.get(base)
    if tier == 1 and gt:
        out = dict(gt)
        out["accession"] = (gt.get("accession") or acc).strip().split()[0]
        out["sequence"] = seq
        out["length"] = str(len(seq))
        out["label"] = "1"
        out["tier"] = "1"
        return out
    if tier == 1:
        return {
            "accession": acc,
            "substrate": "",
            "evidence_tier": "1",
            "source_db": exp.get("source_db") or "PAZy_ground_truth",
            "source_note": "Gold (expansion query; no PAZy CSV row matched)",
            "id": f"sp|{base}|GOLD",
            "description": f"{base} gold from expansion",
            "sequence": seq,
            "length": str(len(seq)),
            "label": "1",
            "tier": "1",
        }
    note = f"Silver BLAST expansion; {exp.get('source_db', '')}".strip()
    return {
        "accession": acc,
        "substrate": "",
        "evidence_tier": "2",
        "source_db": exp.get("source_db") or "BLAST",
        "source_note": note,
        "id": f"tr|{base}|BLAST_SILVER",
        "description": f"{base} homology-expanded (tier 2)",
        "sequence": seq,
        "length": str(len(seq)),
        "label": "1",
        "tier": "2",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge expansion + PAZy positives into unified CSV/FASTA")
    parser.add_argument(
        "--expanded-csv",
        type=Path,
        default=paths.expanded_positives_csv(),
    )
    parser.add_argument(
        "--positives-gt",
        type=Path,
        default=paths.positives_gt_csv(),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=paths.positives_gt_expanded_csv(),
    )
    parser.add_argument(
        "--out-fasta",
        type=Path,
        default=paths.positives_from_gt_expanded_fasta(),
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=paths.processed_v2(),
        help="Used with --backup-prior-splits (writes under splits/)",
    )
    parser.add_argument(
        "--backup-prior-splits",
        action="store_true",
        help="Copy current split_test_accessions.txt to split_test_accessions_for_leakage_constraint.txt "
        "(for homology_split --enforce-silver-gold-test-constraint).",
    )
    parser.add_argument(
        "--out-alias-map",
        type=Path,
        default=paths.accession_sequence_alias_map_csv(),
        help="CSV: canonical accession + all expansion accessions merged to that sequence (reporting).",
    )
    parser.add_argument(
        "--no-alias-map",
        action="store_true",
        help="Skip writing accession_sequence_alias_map.csv",
    )
    args = parser.parse_args(argv)

    if not args.expanded_csv.exists():
        print(f"ERROR: missing {args.expanded_csv}", file=sys.stderr)
        return 1

    expanded = _load_expanded(args.expanded_csv)
    if not expanded:
        print("ERROR: no rows in expanded CSV.", file=sys.stderr)
        return 1

    gt_by_base = _load_positives_gt(args.positives_gt)

    by_acc_key: dict[str, dict[str, str]] = {}
    for exp in expanded:
        base = _strip_uniprot_version(exp["accession"])
        if base in by_acc_key:
            continue
        by_acc_key[base] = _merge_row(exp, gt_by_base)

    merged: list[dict[str, str]] = list(by_acc_key.values())
    merged.sort(key=lambda r: _strip_uniprot_version(r["accession"]))
    fp_to_idx: dict[str, int] = {}
    deduped: list[dict[str, str]] = []
    for row in merged:
        fp = _seq_fingerprint(row["sequence"])
        tier = int(row.get("tier", "1"))
        if fp not in fp_to_idx:
            fp_to_idx[fp] = len(deduped)
            deduped.append(row)
            continue
        j = fp_to_idx[fp]
        prev = deduped[j]
        prev_tier = int(prev.get("tier", "1"))
        if tier == 1 and prev_tier == 2:
            deduped[j] = row
        elif tier == 2 and prev_tier == 1:
            pass
        else:
            print(
                f"  WARNING: duplicate sequence; dropping accession {row['accession']} "
                f"(kept {prev['accession']})",
                flush=True,
            )

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
        "tier",
    ]
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_fasta.parent.mkdir(parents=True, exist_ok=True)

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in deduped:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    records = [
        SeqRecord(Seq(row["sequence"]), id=row["accession"], description="")
        for row in deduped
    ]
    with args.out_fasta.open("w", encoding="utf-8") as fh:
        SeqIO.write(records, fh, "fasta")

    if not args.no_alias_map:
        _write_accession_alias_map(expanded, deduped, args.out_alias_map)
        print(f"  Wrote {args.out_alias_map}", flush=True)

    if args.backup_prior_splits:
        splits = args.artifact_root / "splits"
        cur_test = splits / "split_test_accessions.txt"
        dst = splits / "split_test_accessions_for_leakage_constraint.txt"
        if cur_test.is_file():
            shutil.copyfile(cur_test, dst)
            print(f"  Backed up prior test split -> {dst}", flush=True)
        else:
            print(f"  WARNING: no {cur_test} to backup.", flush=True)

    n1 = sum(1 for r in deduped if r.get("tier") == "1")
    n2 = sum(1 for r in deduped if r.get("tier") == "2")
    manifest = {
        "n_rows": len(deduped),
        "tier1": n1,
        "tier2": n2,
        "out_csv": str(args.out_csv),
        "out_fasta": str(args.out_fasta),
        "accession_alias_map_csv": None if args.no_alias_map else str(args.out_alias_map),
    }
    man_path = args.out_csv.parent / "integrate_expansion_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    print(f"  Wrote {args.out_csv}", flush=True)
    print(f"  Wrote {args.out_fasta}", flush=True)
    print(f"  Wrote {man_path}", flush=True)
    print(
        "  Next (example):\n"
        f"    python -m plasticdeg.split.homology_split --positives-fasta {args.out_fasta} "
        f"--negatives-fasta {paths.negatives_from_uniprot_fasta()} "
        "--enforce-silver-gold-test-constraint "
        f"--prior-split-test {args.artifact_root / 'splits' / 'split_test_accessions_for_leakage_constraint.txt'} "
        f"--tier-csv {args.out_csv}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
