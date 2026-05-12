"""
Homology-aware train/test assignment via CD-HIT or MMseqs2 (optional).

Clusters combined positive+negative FASTA at a sequence identity threshold, then
assigns whole clusters to train vs test so no cluster appears in both splits.

Preference: CD-HIT on PATH (Linux/macOS); else MMseqs2 (`mmseqs` on PATH, or
`--mmseqs-binary` / env `MMSEQS_BINARY` — needed on Windows because Bioconda
does not ship mmseqs2/cd-hit for win-64).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from Bio import SeqIO


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_cd_hit() -> str | None:
    for name in ("cd-hit", "cd-hit.exe", "cd-hit-est", "cd-hit-est.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None


def find_mmseqs() -> str | None:
    for name in ("mmseqs", "mmseqs.exe"):
        p = shutil.which(name)
        if p:
            return p
    return None


def find_git_bash_exe() -> Path | None:
    roots: list[Path] = []
    for ev in ("ProgramFiles", "ProgramFiles(x86)", "LocalAppData"):
        b = os.environ.get(ev)
        if b:
            roots.append(Path(b))
    for root in roots:
        for rel in ("Git/bin/bash.exe", "Git/usr/bin/bash.exe"):
            p = root / rel
            if p.is_file():
                return p
    wh = shutil.which("bash.exe")
    return Path(wh) if wh else None


def find_cygwin_bash_exe() -> Path | None:
    """MMseqs Win builds use /cygdrive/... paths — Cygwin bash/sh understands those."""
    for base in ("C:\\cygwin64", "C:\\cygwin"):
        p = Path(base) / "bin" / "bash.exe"
        if p.is_file():
            return p
    return None


def resolve_windows_shell_for_mmseqs() -> tuple[Path, str]:
    cyg = find_cygwin_bash_exe()
    if cyg is not None:
        return cyg, "Cygwin bash (/cygdrive paths)"
    git_b = find_git_bash_exe()
    if git_b is not None:
        return git_b, "Git Bash"
    raise RuntimeError(
        "No Unix shell for MMseqs on Windows. Install Cygwin (https://cygwin.com/install.html) "
        "with bash, or Git for Windows. Better: install mmseqs2 inside WSL "
        "(Ubuntu: sudo apt update && sudo apt install -y mmseqs2; or conda/mamba). "
        "Then --mmseqs-backend wsl or auto."
    )


def windows_path_as_wsl(path: Path) -> str:
    """Map D:\\foo\\bar → /mnt/d/foo/bar for invoking Linux mmseqs from WSL."""
    resolved = path.resolve()
    s = str(resolved)
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}" if rest else f"/mnt/{drive}"
    raise ValueError(f"Expected a Windows drive path, got {path!r}")


def wsl_mmseqs_on_path() -> bool:
    try:
        r = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "command -v mmseqs >/dev/null"],
            capture_output=True,
            timeout=60,
            text=True,
        )
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def bash_single_quoted_posix_path(p: Path) -> str:
    """Safe single-quoted literal for Git Bash / MSYS."""
    s = str(p.resolve()).replace("\\", "/")
    return "'" + s.replace("'", "'\"'\"'") + "'"


def run_mmseqs_windows_pipeline(
    mmseqs: str,
    combined_fasta: Path,
    work_dir: Path,
    *,
    identity: float,
    threads: int,
) -> tuple[Path, str]:
    """
    Win MMseqs2 emits /cygdrive/... paths and shells out to .sh helpers (Cygwin).
    Git Bash (MSYS) often cannot execute those scripts → error 2.

    Prefer **Cygwin bash** when installed; otherwise Git Bash.
    """
    bash, bash_label = resolve_windows_shell_for_mmseqs()

    work_dir.mkdir(parents=True, exist_ok=True)
    db_pref = work_dir / "seq_db"
    tsv_out = work_dir / "cluster_mmseqs.tsv"
    mm_q = bash_single_quoted_posix_path(Path(mmseqs))
    comb_q = bash_single_quoted_posix_path(combined_fasta)
    db_q = bash_single_quoted_posix_path(db_pref)

    cluster_modes: list[tuple[str, Path, Path, bool]] = [
        ("cluster-single-step", work_dir / "seq_clu", work_dir / "tmp_cluster_ss", True),
        ("linclust", work_dir / "seq_lin", work_dir / "tmp_linclust", False),
    ]
    last_err = ""
    child_env = os.environ.copy()
    child_env.setdefault("MSYS_NO_PATHCONV", "1")

    for label, clu_base, tmpd, use_single_step in cluster_modes:
        shutil.rmtree(tmpd, ignore_errors=True)
        tmpd.mkdir(parents=True, exist_ok=True)
        for pat in sorted(work_dir.glob(clu_base.name + "*"), reverse=True):
            if pat.is_dir():
                shutil.rmtree(pat, ignore_errors=True)
            else:
                pat.unlink(missing_ok=True)

        if tsv_out.exists():
            tsv_out.unlink(missing_ok=True)

        clu_q = bash_single_quoted_posix_path(clu_base)
        tmp_q = bash_single_quoted_posix_path(tmpd)
        out_q = bash_single_quoted_posix_path(tsv_out)
        subcmd = "cluster" if use_single_step else "linclust"
        tail = (
            f"--min-seq-id {identity} -c 0.8 --cov-mode 1 --threads {threads}"
        )
        single = "--single-step-clustering " if use_single_step else ""
        script_body = "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"{mm_q} createdb {comb_q} {db_q}",
                f"{mm_q} {subcmd} {db_q} {clu_q} {tmp_q} {single}{tail}",
                f"{mm_q} createtsv {db_q} {db_q} {clu_q} {out_q}",
            ]
        )
        sh_path = work_dir / f"run_mmseqs_{label.replace('-', '_')}.sh"
        sh_path.write_text(script_body.replace("\r\n", "\n"), encoding="utf-8", newline="\n")

        bash_cmd = [str(bash)]
        if bash_label.startswith("Cygwin"):
            bash_cmd.append("--login")
        bash_cmd.append(str(sh_path))
        r = subprocess.run(
            bash_cmd,
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            env=child_env,
        )
        if r.returncode != 0:
            last_err = (r.stderr or r.stdout or "").strip()
            continue
        if not tsv_out.exists():
            last_err = "createtsv produced no cluster_mmseqs.tsv"
            continue
        return tsv_out, label

    raise RuntimeError(
        "MMseqs clustering failed on native Windows (cluster single-step + linclust).\n"
        f"Shell used: {bash_label} ({bash})\n"
        "Portable MMseqs2 on Windows expects Cygwin (/cygdrive paths). Install Cygwin, "
        "or in Ubuntu/WSL run: sudo apt update && sudo apt install -y mmseqs2 — verify "
        "with `wsl -e bash -lc \"command -v mmseqs\"` — then homology_split --mmseqs-backend auto "
        "(default). Last resort: --split-method stratified.\n"
        "Last error:\n"
        + last_err[:2000]
    )


def run_mmseqs_wsl_pipeline(
    combined_fasta: Path,
    work_dir: Path,
    *,
    identity: float,
    threads: int,
) -> tuple[Path, str]:
    """Linux mmseqs inside WSL; FASTA/work dirs on NTFS via /mnt/… ."""
    work_dir.mkdir(parents=True, exist_ok=True)
    db_pref = work_dir / "seq_db"
    tsv_out = work_dir / "cluster_mmseqs.tsv"
    comb_w = windows_path_as_wsl(combined_fasta)
    db_w = windows_path_as_wsl(db_pref)
    tsv_w = windows_path_as_wsl(tsv_out)

    cluster_modes: list[tuple[str, Path, Path, bool]] = [
        ("cluster-single-step", work_dir / "seq_clu", work_dir / "tmp_cluster_ss", True),
        ("linclust", work_dir / "seq_lin", work_dir / "tmp_linclust", False),
    ]
    last_err = ""
    for label, clu_base, tmpd, use_single_step in cluster_modes:
        shutil.rmtree(tmpd, ignore_errors=True)
        tmpd.mkdir(parents=True, exist_ok=True)
        for pat in sorted(work_dir.glob(clu_base.name + "*"), reverse=True):
            if pat.is_dir():
                shutil.rmtree(pat, ignore_errors=True)
            else:
                pat.unlink(missing_ok=True)
        tsv_out.unlink(missing_ok=True)

        clu_w = windows_path_as_wsl(clu_base)
        tmp_w = windows_path_as_wsl(tmpd)
        subcmd = "cluster" if use_single_step else "linclust"
        cluster_parts = [
            "mmseqs",
            subcmd,
            shlex.quote(db_w),
            shlex.quote(clu_w),
            shlex.quote(tmp_w),
        ]
        if use_single_step:
            cluster_parts.append("--single-step-clustering")
        cluster_parts.extend(
            [
                "--min-seq-id",
                str(identity),
                "-c",
                "0.8",
                "--cov-mode",
                "1",
                "--threads",
                str(threads),
            ]
        )
        inner = "; ".join(
            [
                "set -euo pipefail",
                f"rm -rf {shlex.quote(tmp_w)}",
                f"mkdir -p {shlex.quote(tmp_w)}",
                "mmseqs createdb " + " ".join(map(shlex.quote, [comb_w, db_w])),
                " ".join(cluster_parts),
                "mmseqs createtsv "
                + " ".join(map(shlex.quote, [db_w, db_w, clu_w, tsv_w])),
            ]
        )
        r = subprocess.run(
            ["wsl", "-e", "bash", "-lc", inner],
            capture_output=True,
            text=True,
            timeout=86400,
        )
        if r.returncode != 0:
            last_err = (r.stderr or r.stdout or "").strip()
            continue
        if not tsv_out.exists():
            last_err = "createtsv produced no cluster_mmseqs.tsv"
            continue
        return tsv_out, label

    raise RuntimeError(
        "MMseqs (inside WSL) failed.\n"
        "Ubuntu/WSL: sudo apt update && sudo apt install -y mmseqs2\n"
        "Or: conda install -c conda-forge -c bioconda mmseqs2\n"
        "Last error:\n" + last_err[:2000]
    )


def stratified_accession_split(
    positives_fasta: Path,
    negatives_fasta: Path,
    *,
    test_fraction: float,
    seed: int,
) -> tuple[set[str], set[str]]:
    """Random stratified split by label (NOT homology-safe — use only if MMseqs fails)."""
    from sklearn.model_selection import train_test_split

    pos_ids: list[str] = []
    for rec in SeqIO.parse(positives_fasta, "fasta"):
        pos_ids.append(accession_from_biopython_id(rec.id))
    neg_ids: list[str] = []
    for rec in SeqIO.parse(negatives_fasta, "fasta"):
        neg_ids.append(f"NEG_{accession_from_biopython_id(rec.id)}")

    xs = pos_ids + neg_ids
    ys = [1] * len(pos_ids) + [0] * len(neg_ids)
    tr, te, _, _ = train_test_split(
        xs,
        ys,
        test_size=test_fraction,
        stratify=ys,
        random_state=seed,
        shuffle=True,
    )
    return set(tr), set(te)


def resolve_mmseqs_executable(mmseqs_binary: Path | None) -> str | None:
    """Explicit path wins; then MMSEQS_BINARY; then PATH."""
    if mmseqs_binary is not None:
        p = mmseqs_binary.expanduser().resolve()
        if p.is_file():
            return str(p)
    env = os.environ.get("MMSEQS_BINARY") or os.environ.get("MMSEQS2")
    if env:
        ep = Path(env).expanduser().resolve()
        if ep.is_file():
            return str(ep)
    return find_mmseqs()


def parse_mmseqs_cluster_tsv(path: Path) -> dict[str, int]:
    """MMseqs easy-cluster TSV: representative<TAB>member → member -> cluster index."""
    rep_to_members: dict[str, list[str]] = {}
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0].strip(), parts[1].strip()
            rep_to_members.setdefault(rep, []).append(member)
    reps_sorted = sorted(rep_to_members.keys())
    rep_to_idx = {r: i for i, r in enumerate(reps_sorted)}
    seq_to_cluster: dict[str, int] = {}
    for rep, members in rep_to_members.items():
        cid = rep_to_idx[rep]
        for m in members:
            seq_to_cluster[m] = cid
    return seq_to_cluster


def parse_cdhit_clstr(path: Path) -> dict[str, int]:
    """Map sequence id (FASTA id before first space) -> cluster index."""
    seq_to_cluster: dict[str, int] = {}
    cluster_idx = -1
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith(">Cluster "):
            cluster_idx = int(line.split()[1])
            continue
        m = re.search(r">([^\s.]+)", line)
        if m:
            seq_id = m.group(1)
            seq_to_cluster[seq_id] = cluster_idx
    return seq_to_cluster


def cluster_train_test(
    seq_to_cluster: dict[str, int],
    test_fraction: float = 0.2,
) -> tuple[set[str], set[str]]:
    """Assign each cluster entirely to train or test (deterministic hash by cluster id)."""
    clusters: dict[int, list[str]] = {}
    for sid, cid in seq_to_cluster.items():
        clusters.setdefault(cid, []).append(sid)

    train: set[str] = set()
    test: set[str] = set()
    for cid, members in sorted(clusters.items()):
        h = int(hashlib.sha256(str(cid).encode()).hexdigest(), 16)
        to_test = (h % 10000) / 10000.0 < test_fraction
        bucket = test if to_test else train
        for m in members:
            bucket.add(m)
    return train, test


def _strip_uniprot_version(acc: str) -> str:
    acc = acc.strip()
    if "." in acc and acc.rsplit(".", 1)[-1].isdigit():
        return acc.rsplit(".", 1)[0]
    return acc


def _clusters_from_mapping(seq_to_cluster: dict[str, int]) -> dict[int, set[str]]:
    clusters: dict[int, set[str]] = defaultdict(set)
    for sid, cid in seq_to_cluster.items():
        clusters[cid].add(sid)
    return clusters


def load_tier_map(tier_csv: Path) -> dict[str, int]:
    """accession (and stripped accession) -> tier int (1=Gold, 2=Silver)."""
    out: dict[str, int] = {}
    with tier_csv.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            if not acc:
                continue
            try:
                tier = int(float((row.get("tier") or "1").strip()))
            except ValueError:
                tier = 1
            out[acc] = tier
            base = _strip_uniprot_version(acc)
            if base != acc:
                out.setdefault(base, tier)
    return out


def enforce_silver_prior_gold_test_leakage(
    train_ids: set[str],
    test_ids: set[str],
    seq_to_cluster: dict[str, int],
    *,
    prior_split_test: Path,
    tier_csv: Path,
) -> tuple[set[str], set[str]]:
    """
    If a cluster contains both (a) any Tier-2 (Silver) positive and (b) any positive
    accession that appeared in the *prior* test split, force the entire cluster to test.

    This prevents train/test leakage when adding Silver relatives of old-test Gold.
    """
    prior_lines = {
        ln.strip()
        for ln in prior_split_test.read_text(encoding="utf-8", errors="replace").splitlines()
        if ln.strip()
    }
    prior_test_gold = {s for s in prior_lines if not s.startswith("NEG_")}
    tier_map = load_tier_map(tier_csv)

    def tier_of(sid: str) -> int:
        if sid.startswith("NEG_"):
            return 0
        if sid in tier_map:
            return tier_map[sid]
        return tier_map.get(_strip_uniprot_version(sid), 1)

    clusters = _clusters_from_mapping(seq_to_cluster)
    train = set(train_ids)
    test = set(test_ids)
    n_forced = 0
    for _cid, members in clusters.items():
        pos_members = [m for m in members if not m.startswith("NEG_")]
        has_silver = any(tier_of(m) == 2 for m in pos_members)
        has_prior_test_gold = any(m in prior_test_gold for m in pos_members)
        if has_silver and has_prior_test_gold:
            if members & train:
                train -= members
                test |= members
                n_forced += 1
    if n_forced:
        print(
            f"  Silver/Gold leakage rule: forced {n_forced} cluster(s) entirely to test "
            f"(Silver ∩ prior-test-positive in same MMseqs/CD-HIT cluster).",
            flush=True,
        )
    return train, test


def accession_from_biopython_id(seq_id: str) -> str:
    if "|" in seq_id and (seq_id.startswith("sp|") or seq_id.startswith("tr|")):
        parts = seq_id.split("|")
        if len(parts) >= 2:
            return parts[1]
    return seq_id.split()[0]


def main(argv: list[str] | None = None) -> int:
    from plasticdeg import paths

    root = project_root()
    parser = argparse.ArgumentParser(description="CD-HIT–based cluster split manifest")
    parser.add_argument(
        "--positives-fasta",
        type=Path,
        default=paths.positives_from_gt_fasta(),
    )
    parser.add_argument(
        "--negatives-fasta",
        type=Path,
        default=paths.negatives_from_uniprot_fasta(),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=paths.processed_v2(),
        help="Artifact root (writes splits/ underneath)",
    )
    parser.add_argument(
        "--identity",
        type=float,
        default=0.7,
        help="Cluster identity (CD-HIT -c, or MMseqs --min-seq-id)",
    )
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument(
        "--mmseqs-binary",
        type=Path,
        default=None,
        help="Full path to mmseqs executable (Windows native: portable MMseqs2 release)",
    )
    parser.add_argument(
        "--mmseqs-backend",
        choices=("auto", "native", "wsl"),
        default="auto",
        help="Windows only: auto uses WSL mmseqs when available, else native exe; "
        "native = portable mmseqs.exe (+ Cygwin or Git Bash); wsl = Linux mmseqs in WSL only.",
    )
    parser.add_argument(
        "--split-method",
        choices=("homology", "stratified"),
        default="homology",
        help="homology: CD-HIT/MMseqs clustering (recommended); stratified: label-balanced random split "
        "(allows sequence leakage — mini-project fallback only)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Used with --split-method stratified")
    parser.add_argument(
        "--enforce-silver-gold-test-constraint",
        action="store_true",
        help="After cluster split: any cluster with both Tier-2 (Silver) and a prior-test "
        "positive accession is forced entirely to test (requires --prior-split-test + --tier-csv).",
    )
    parser.add_argument(
        "--prior-split-test",
        type=Path,
        default=None,
        help="Prior split_test_accessions.txt (e.g. from integrate_expansion --backup-prior-splits).",
    )
    parser.add_argument(
        "--tier-csv",
        type=Path,
        default=None,
        help="CSV with accession + tier columns (e.g. positives_gt_expanded.csv).",
    )
    args = parser.parse_args(argv)

    splits_out = args.out_dir / "splits"
    splits_out.mkdir(parents=True, exist_ok=True)

    if not args.positives_fasta.exists() or not args.negatives_fasta.exists():
        print(
            "ERROR: Missing FASTA inputs. Run:\n"
            "  python -m plasticdeg.ingest.fetch_sequences\n"
            "  python -m plasticdeg.ingest.sample_negatives\n"
            f"  Expected: {args.positives_fasta} and {args.negatives_fasta}",
            file=sys.stderr,
            flush=True,
        )
        return 1

    train_path = splits_out / "split_train_accessions.txt"
    test_path = splits_out / "split_test_accessions.txt"
    if args.split_method == "stratified":
        train_ids, test_ids = stratified_accession_split(
            args.positives_fasta,
            args.negatives_fasta,
            test_fraction=args.test_fraction,
            seed=args.seed,
        )
        train_path.write_text("\n".join(sorted(train_ids)) + "\n", encoding="utf-8")
        test_path.write_text("\n".join(sorted(test_ids)) + "\n", encoding="utf-8")
        print(
            "  WARNING: stratified random split — NOT homology-aware (possible train/test leakage).",
            flush=True,
        )
        print(f"  Wrote {train_path} ({len(train_ids)} ids)")
        print(f"  Wrote {test_path} ({len(test_ids)} ids)")
        return 0

    cdhit = find_cd_hit()
    mmseqs = resolve_mmseqs_executable(args.mmseqs_binary)
    # Windows: mmseqs may exist only inside WSL (not on Windows PATH); don't bail out early.
    mmseqs_via_wsl_ok = (
        sys.platform == "win32"
        and args.mmseqs_backend in ("auto", "wsl")
        and wsl_mmseqs_on_path()
    )
    if not cdhit and not mmseqs and not mmseqs_via_wsl_ok:
        print(
            "ERROR: Homology clustering tool not found (no CD-HIT, no mmseqs on Windows PATH, "
            "and WSL has no mmseqs).\n\n"
            "Windows (win-64): Bioconda does NOT ship mmseqs2 or cd-hit for win-64.\n"
            "  • Recommended: in WSL Ubuntu — sudo apt update && sudo apt install -y mmseqs2\n"
            "    Then from Windows: python -m plasticdeg.split.homology_split --mmseqs-backend auto\n"
            "  • Portable MMseqs2 (.exe) + Cygwin bash: https://github.com/soedinglab/MMseqs2/releases\n"
            "    python -m plasticdeg.split.homology_split --mmseqs-binary \"D:\\\\path\\\\mmseqs.exe\" "
            "--mmseqs-backend native\n"
            "  • Fallback (not homology-safe): --split-method stratified\n\n"
            "Linux/macOS: conda install mmseqs2 or cd-hit (bioconda) so the tool is on PATH.",
            file=sys.stderr,
            flush=True,
        )
        return 0

    # WSL mmseqs under /mnt/c/... can leave symlinks (e.g. latest/) Windows cannot unlink cleanly.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
        tdir = Path(td)
        combined = tdir / "combined.fasta"
        with combined.open("w", encoding="utf-8") as out:
            for rec in SeqIO.parse(args.positives_fasta, "fasta"):
                acc = accession_from_biopython_id(rec.id)
                out.write(f">{acc}\n{rec.seq}\n")
            for rec in SeqIO.parse(args.negatives_fasta, "fasta"):
                acc = accession_from_biopython_id(rec.id)
                nid = f"NEG_{acc}"
                out.write(f">{nid}\n{rec.seq}\n")

        if cdhit:
            out_pre = tdir / "combined_cdhit"
            cmd = [
                cdhit,
                "-i",
                str(combined),
                "-o",
                str(out_pre),
                "-c",
                str(args.identity),
                "-n",
                "5",
                "-M",
                "16000",
                "-d",
                "0",
                "-T",
                str(max(1, (os.cpu_count() or 2) - 1)),
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            clstr_path = Path(str(out_pre) + ".clstr")
            if not clstr_path.exists():
                print("ERROR: CD-HIT did not write .clstr")
                return 1
            mapping = parse_cdhit_clstr(clstr_path)
            print(f"  Clustering with CD-HIT ({cdhit})", flush=True)
        else:
            threads = max(1, (os.cpu_count() or 2) - 1)
            use_native = sys.platform == "win32"
            if use_native:
                mm_work = tdir / "mmseq_native"
                ran_wsl = False
                want_wsl = args.mmseqs_backend == "wsl" or (
                    args.mmseqs_backend == "auto" and wsl_mmseqs_on_path()
                )
                if want_wsl:
                    if not wsl_mmseqs_on_path():
                        print(
                            "ERROR: --mmseqs-backend wsl but `mmseqs` not found in WSL.\n"
                            "  Ubuntu/WSL: sudo apt update && sudo apt install -y mmseqs2\n"
                            "  Or: conda install -c conda-forge -c bioconda mmseqs2",
                            flush=True,
                        )
                        return 1
                    try:
                        tsv_path, mm_label = run_mmseqs_wsl_pipeline(
                            combined,
                            mm_work,
                            identity=args.identity,
                            threads=threads,
                        )
                        mapping = parse_mmseqs_cluster_tsv(tsv_path)
                        print(f"  Clustering with MMseqs2 via WSL ({mm_label})", flush=True)
                        ran_wsl = True
                    except RuntimeError as e:
                        if args.mmseqs_backend == "wsl":
                            print(f"ERROR: {e}", flush=True)
                            return 1
                        print(f"  WSL MMseqs failed ({e}); falling back to native Windows exe…", flush=True)

                if not ran_wsl:
                    if mmseqs is None:
                        print(
                            "ERROR: No mmseqs.exe on Windows PATH (and no --mmseqs-binary).\n"
                            "  Fix WSL mmseqs or install portable MMseqs2 + Cygwin; "
                            "or pass --mmseqs-binary path to mmseqs.exe.",
                            flush=True,
                        )
                        return 1
                    try:
                        tsv_path, mm_label = run_mmseqs_windows_pipeline(
                            mmseqs,
                            combined,
                            mm_work,
                            identity=args.identity,
                            threads=threads,
                        )
                    except RuntimeError as e:
                        print(f"ERROR: {e}", flush=True)
                        print(
                            "  Tip: Ubuntu/WSL → sudo apt install mmseqs2; or Cygwin for portable .exe; "
                            "or --split-method stratified (weaker evaluation).",
                            flush=True,
                        )
                        return 1
                    mapping = parse_mmseqs_cluster_tsv(tsv_path)
                    print(f"  Clustering with MMseqs2 ({mm_label}; {mmseqs})", flush=True)
            else:
                if mmseqs is None:
                    print(
                        "ERROR: mmseqs not on PATH. Linux/macOS: conda install mmseqs2 or add to PATH.",
                        flush=True,
                    )
                    return 1
                mm_prefix = tdir / "mmcls"
                mm_tmp = tdir / "mmseqs_tmp"
                mm_tmp.mkdir(parents=True, exist_ok=True)
                cmd = [
                    mmseqs,
                    "easy-cluster",
                    str(combined),
                    str(mm_prefix),
                    str(mm_tmp),
                    "--min-seq-id",
                    str(args.identity),
                    "-c",
                    "0.8",
                    "--cov-mode",
                    "1",
                    "--threads",
                    str(threads),
                ]
                r = subprocess.run(cmd, capture_output=True, text=True)
                if r.returncode != 0:
                    print(r.stderr or r.stdout or "mmseqs failed")
                    r.check_returncode()
                tsv_path = Path(str(mm_prefix) + "_cluster.tsv")
                if not tsv_path.exists():
                    print(f"ERROR: MMseqs2 did not write {tsv_path}")
                    return 1
                mapping = parse_mmseqs_cluster_tsv(tsv_path)
                print(f"  Clustering with MMseqs2 easy-cluster ({mmseqs})", flush=True)

        train_ids, test_ids = cluster_train_test(mapping, args.test_fraction)

        if args.enforce_silver_gold_test_constraint:
            if args.prior_split_test is None or args.tier_csv is None:
                print(
                    "ERROR: --enforce-silver-gold-test-constraint requires "
                    "--prior-split-test and --tier-csv.",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
            if not args.prior_split_test.is_file():
                print(f"ERROR: missing --prior-split-test file: {args.prior_split_test}", file=sys.stderr)
                return 1
            if not args.tier_csv.is_file():
                print(f"ERROR: missing --tier-csv: {args.tier_csv}", file=sys.stderr)
                return 1
            train_ids, test_ids = enforce_silver_prior_gold_test_leakage(
                train_ids,
                test_ids,
                mapping,
                prior_split_test=args.prior_split_test,
                tier_csv=args.tier_csv,
            )

        train_path.write_text("\n".join(sorted(train_ids)) + "\n", encoding="utf-8")
        test_path.write_text("\n".join(sorted(test_ids)) + "\n", encoding="utf-8")
        print(f"  Wrote {train_path} ({len(train_ids)} ids)")
        print(f"  Wrote {test_path} ({len(test_ids)} ids)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
