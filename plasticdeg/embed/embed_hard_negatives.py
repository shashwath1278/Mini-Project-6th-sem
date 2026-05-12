"""
Embed hard-negative sequences (CSV) with the same frozen ESM-2 mean-pool as split data.

Default CSV: data/processed_v2/probes/hard_negatives_gt.csv
Default out:   data/processed_v2/embeddings/embeddings_hard_negatives.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from plasticdeg import paths
from plasticdeg.embed.embed_sequences import _import_torch, mean_pool_esm2_batch


def load_probe_csv(path: Path, *, id_prefix: str) -> list[tuple[str, str]]:
    prefix = id_prefix if id_prefix.endswith("_") else f"{id_prefix}_"
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            acc = (row.get("accession") or "").strip()
            seq = (row.get("sequence") or "").strip()
            if acc and seq:
                pairs.append((f"{prefix}{acc}", seq))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ESM-2 embeddings for hard-negative CSV")
    parser.add_argument(
        "--csv",
        type=Path,
        default=paths.hard_negatives_gt_csv(),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=paths.embeddings_hard_negatives_npz(),
    )
    parser.add_argument(
        "--id-prefix",
        type=str,
        default="HARD",
        help="Embedding id prefix (e.g. HARD, ADV). Stored as PREFIX_accession.",
    )
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(
            f"ERROR: missing CSV: {args.csv}\n"
            "  Produce it with e.g. plasticdeg.ingest.fetch_hard_negatives or plasticdeg.ingest.fetch_adversarial_negatives, "
            "then re-run this command.",
            file=sys.stderr,
        )
        return 1

    try:
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass

    pairs = load_probe_csv(args.csv, id_prefix=args.id_prefix)
    if not pairs:
        print("ERROR: no rows in CSV.", file=sys.stderr)
        return 1

    print(f"embed_hard_negatives: {len(pairs)} sequences", file=sys.stderr, flush=True)
    torch = _import_torch()
    try:
        import esm
    except ImportError:
        print("ERROR: pip install fair-esm torch", file=sys.stderr)
        return 1

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"  Loading {args.model} on {device} ...", file=sys.stderr, flush=True)
    model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    model = model.to(device)
    batch_converter = alphabet.get_batch_converter()

    from tqdm import tqdm

    all_ids: list[str] = []
    all_emb: list[np.ndarray] = []
    bs = max(1, args.batch_size)
    n = len(pairs)
    n_batches = (n + bs - 1) // bs
    for start in tqdm(
        range(0, n, bs),
        total=n_batches,
        desc="ESM-2 probe",
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    meta = np.array([args.model, str(args.csv)], dtype=object)
    np.savez_compressed(args.out, embeddings=embeddings, ids=ids_arr, meta=meta)
    print(f"  Wrote {args.out} shape={embeddings.shape}", file=sys.stderr, flush=True)
    print("  Next: conda run --no-capture-output -n plasticdeg python -u -m plasticdeg.eval.probe_hard_negatives", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
