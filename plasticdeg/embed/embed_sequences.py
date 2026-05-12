"""
Phase 1 — frozen ESM-2 (t33 650M) mean-pooled embeddings, embed-once workflow.

Loads sequences for IDs listed in homology split files (union of train ∪ test),
writes a compressed .npz with aligned `ids` and `embeddings` (float32, N×1280).

Requires: pip install fair-esm torch  (see requirements.txt)
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import numpy as np

from plasticdeg import paths
from plasticdeg.split.homology_split import accession_from_biopython_id


def _strip_uniprot_version(acc: str) -> str:
    if "." in acc and acc.rsplit(".", 1)[-1].isdigit():
        return acc.rsplit(".", 1)[0]
    return acc


def _torch_import_help(exc: BaseException) -> str:
    return (
        f"PyTorch failed to load ({type(exc).__name__}: {exc}).\n\n"
        "On Windows this often happens with Python 3.13 + pip torch (e.g. shm.dll / WinError 127).\n"
        "Fix: use the env's Python 3.12 (not Miniconda **base** 3.13). In **PowerShell**, `where` is **not** the "
        "PATH finder — run `where.exe python` or `Get-Command python`. The first path must be "
        "`...\\envs\\plasticdeg\\python.exe`. If `conda activate` does not stick (common in pasted blocks), use:\n"
        "  conda run -n plasticdeg python -m pip install -r requirements.txt\n"
        "  conda run --no-capture-output -n plasticdeg python -u -m plasticdeg.embed.embed_sequences --batch-size 4\n"
        "  (-u = unbuffered; --no-capture-output = live logs under conda run)\n"
        "or call the interpreter explicitly, e.g. "
        "C:\\Users\\<you>\\miniconda3\\envs\\plasticdeg\\python.exe -m plasticdeg.embed.embed_sequences\n\n"
        "If `conda create` fails with CondaToSNonInteractiveError, either accept defaults channel ToS:\n"
        "  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main\n"
        "  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r\n"
        "  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2\n"
        "or create from conda-forge only:\n"
        "  conda create -n plasticdeg python=3.12 -c conda-forge --override-channels -y\n"
        "Then:\n"
        "  conda activate plasticdeg\n"
        "  python -m pip install --upgrade pip\n"
        "  pip install torch fair-esm numpy\n\n"
        "Install latest \"Microsoft Visual C++ Redistributable\" (x64) if DLL errors persist.\n"
        "If you see OpenMP duplicate-runtime warnings from numpy/sklearn + torch, set env:\n"
        "  set KMP_DUPLICATE_LIB_OK=TRUE\n"
        "Alternative: run embedding in WSL Ubuntu (pip install torch fair-esm; python -m plasticdeg.embed.embed_sequences).\n"
    )


def _import_torch():
    try:
        import torch

        return torch
    except OSError as e:
        print("ERROR: " + _torch_import_help(e), file=sys.stderr)
        raise SystemExit(1) from e


def load_split_ids(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def load_positive_rows(csv_path: Path) -> dict[str, str]:
    """
    Map accession -> sequence. Adds aliases so split IDs match homology_split FASTA ids:
    - UniProt version: Q96US9.1 -> also key Q96US9
    - UniProt id column (sp|Q4W8C9|...) when accession is secondary (e.g. Q0KBZ6 row).
    """
    out: dict[str, str] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if not acc or not seq:
                continue
            out[acc] = seq
            base = _strip_uniprot_version(acc)
            if base != acc:
                out.setdefault(base, seq)
            uid = (row.get("id") or "").strip()
            if uid:
                alt = accession_from_biopython_id(uid)
                if alt and alt != acc:
                    out.setdefault(alt, seq)
    return out


def load_negative_rows(csv_path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if not acc or not seq:
                continue
            out[acc] = seq
            base = _strip_uniprot_version(acc)
            if base != acc:
                out.setdefault(base, seq)
    return out


def resolve_sequence(
    split_id: str,
    positives: dict[str, str],
    negatives: dict[str, str],
) -> tuple[str, str] | None:
    """
    Map split file ID to (canonical_id, sequence).
    Positives: UniProt accession. Negatives: NEG_<accession> in split files.
    """
    if split_id.startswith("NEG_"):
        acc = split_id[4:]
        seq = negatives.get(acc)
        if seq is None:
            return None
        return split_id, seq
    seq = positives.get(split_id)
    if seq is None:
        return None
    return split_id, seq


def mean_pool_esm2_batch(
    model: Any,
    alphabet: Any,
    batch_converter: Any,
    pairs: list[tuple[str, str]],
    device: Any,
) -> tuple[list[str], np.ndarray]:
    """One forward pass: pairs is a minibatch. Returns (ids, embeddings float32 B×1280)."""
    import torch

    model.eval()
    repr_layer = 33
    max_len = 1022  # safe for t33; avoids OOM on outliers

    trimmed = [(lid, seq[:max_len]) for lid, seq in pairs]
    _labels, _strs, batch_tokens = batch_converter(trimmed)
    batch_tokens = batch_tokens.to(device)
    with torch.no_grad():
        out = model(batch_tokens, repr_layers=[repr_layer], return_contacts=False)
    tok = out["representations"][repr_layer]
    mask = (batch_tokens != alphabet.padding_idx).unsqueeze(-1).to(tok.dtype)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    pooled = (tok * mask).sum(dim=1) / denom
    emb = pooled.float().cpu().numpy().astype(np.float32, copy=False)
    ids_out = [lid for lid, _ in trimmed]
    return ids_out, emb


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ESM-2 mean-pooled embeddings for split IDs")
    parser.add_argument(
        "--positives-csv",
        type=Path,
        default=paths.positives_gt_csv(),
    )
    parser.add_argument(
        "--negatives-csv",
        type=Path,
        default=paths.negatives_gt_csv(),
    )
    parser.add_argument(
        "--train-ids",
        type=Path,
        default=paths.split_train_txt(),
    )
    parser.add_argument(
        "--test-ids",
        type=Path,
        default=paths.split_test_txt(),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.embeddings_esm2_t33_mean_v2_npz(),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="esm2_t33_650M_UR50D",
        help="ESM-2 checkpoint name (fair-esm). Full-quality default is t33 650M (~1280-d, very slow on CPU). "
        "Faster CPU option: esm2_t12_35M_UR50D (~480-d, much faster). If you change --model, use a distinct --out path.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="auto | cuda | cpu",
    )
    args = parser.parse_args(argv)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Line-buffer stdio when possible (helps under `conda run`, which captures subprocess output).
    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    print("embed_sequences: starting (loading torch ...)", file=sys.stderr, flush=True)

    torch = _import_torch()

    try:
        import esm
    except ImportError:
        print("ERROR: fair-esm not installed. Run: pip install fair-esm torch", file=sys.stderr)
        return 1

    train_ids = load_split_ids(args.train_ids)
    test_ids = load_split_ids(args.test_ids)
    want: set[str] = set(train_ids) | set(test_ids)

    positives = load_positive_rows(args.positives_csv)
    negatives = load_negative_rows(args.negatives_csv)

    pairs: list[tuple[str, str]] = []
    missing: list[str] = []
    for sid in sorted(want):
        r = resolve_sequence(sid, positives, negatives)
        if r is None:
            missing.append(sid)
        else:
            pairs.append(r)

    if missing:
        print(f"  WARNING: {len(missing)} split IDs missing from CSVs (first 10): {missing[:10]}", flush=True)

    if not pairs:
        print("ERROR: no sequences to embed.", file=sys.stderr)
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(
        f"  Loading {args.model} on {device} ... (first run may download ~1-2 GB; be patient)",
        file=sys.stderr,
        flush=True,
    )
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    print("  Weights loaded; embedding batches ...", file=sys.stderr, flush=True)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    from tqdm import tqdm

    all_ids: list[str] = []
    all_emb: list[np.ndarray] = []
    bs = max(1, args.batch_size)
    n = len(pairs)
    n_batches = (n + bs - 1) // bs
    # tqdm → stderr by default; updates each batch so `conda run` does not look hung.
    for start in tqdm(
        range(0, n, bs),
        total=n_batches,
        desc="ESM-2 embed",
        unit="batch",
        file=sys.stderr,
        mininterval=0.3,
    ):
        chunk = pairs[start : start + bs]
        ids_chunk, emb_chunk = mean_pool_esm2_batch(model, alphabet, batch_converter, chunk, device)
        all_ids.extend(ids_chunk)
        all_emb.append(emb_chunk)

    embeddings = np.vstack(all_emb).astype(np.float32, copy=False)
    ids_arr = np.array(all_ids, dtype=object)

    meta = np.array(
        [
            args.model,
            str(args.positives_csv),
            str(args.negatives_csv),
            str(args.train_ids),
            str(args.test_ids),
        ],
        dtype=object,
    )
    np.savez_compressed(
        args.out,
        embeddings=embeddings,
        ids=ids_arr,
        meta=meta,
    )
    print(f"  Wrote {args.out} shape={embeddings.shape}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
