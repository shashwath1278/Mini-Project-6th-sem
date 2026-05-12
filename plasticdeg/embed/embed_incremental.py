"""
Append ESM-2 embeddings for *new* positive accessions to an existing .npz.

Keeps existing vectors for IDs already present in the base .npz (typically the
original Gold + all negatives), embeds only missing IDs (typically Silver),
then writes a full matrix ordered like `sorted(train_ids ∪ test_ids)` so it
matches fresh split files from homology_split.

Requires: fair-esm, torch (same as embed_sequences).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from plasticdeg import paths
from plasticdeg.embed.embed_sequences import (
    _import_torch,
    load_negative_rows,
    load_positive_rows,
    load_split_ids,
    mean_pool_esm2_batch,
    resolve_sequence,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Incremental ESM-2 embed merge into existing .npz")
    parser.add_argument(
        "--base-npz",
        type=Path,
        default=paths.embeddings_esm2_t33_mean_v2_npz(),
        help="Existing embeddings (same model dim as --model).",
    )
    parser.add_argument(
        "--positives-csv",
        type=Path,
        default=paths.positives_gt_expanded_csv(),
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
        help="Output .npz (can equal --base-npz to overwrite after merge).",
    )
    parser.add_argument("--model", type=str, default="esm2_t33_650M_UR50D")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args(argv)

    if not args.base_npz.exists():
        print(f"ERROR: missing base npz: {args.base_npz}", file=sys.stderr)
        return 1

    train_ids = set(load_split_ids(args.train_ids))
    test_ids = set(load_split_ids(args.test_ids))
    want_order = sorted(train_ids | test_ids)

    positives = load_positive_rows(args.positives_csv)
    negatives = load_negative_rows(args.negatives_csv)

    base = np.load(args.base_npz, allow_pickle=True)
    old_ids = [str(x) for x in base["ids"]]
    old_emb = np.asarray(base["embeddings"], dtype=np.float32)
    old_map: dict[str, np.ndarray] = {old_ids[i]: old_emb[i].copy() for i in range(len(old_ids))}
    dim = old_emb.shape[1]

    meta_model = str(base["meta"][0]) if base["meta"].shape[0] > 0 else args.model
    if meta_model != args.model:
        print(
            f"  WARNING: base npz meta model={meta_model!r} != --model {args.model!r}; "
            f"using --model for new rows.",
            flush=True,
        )

    missing: list[tuple[str, str]] = []
    for sid in want_order:
        if sid in old_map:
            continue
        pair = resolve_sequence(sid, positives, negatives)
        if pair is None:
            print(f"  ERROR: no sequence for split id {sid!r} (check CSVs).", file=sys.stderr)
            return 1
        missing.append(pair)

    if missing:
        print(f"  Embedding {len(missing)} new sequences (incremental) ...", flush=True)
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
        model, alphabet = esm.pretrained.load_model_and_alphabet(args.model)
        model = model.to(device)
        batch_converter = alphabet.get_batch_converter()
        from tqdm import tqdm

        bs = max(1, args.batch_size)
        new_stack: list[np.ndarray] = []
        new_ids: list[str] = []
        for start in tqdm(
            range(0, len(missing), bs),
            total=(len(missing) + bs - 1) // bs,
            desc="ESM-2 incremental",
            unit="batch",
            file=sys.stderr,
            mininterval=0.3,
        ):
            chunk = missing[start : start + bs]
            ids_chunk, emb_chunk = mean_pool_esm2_batch(model, alphabet, batch_converter, chunk, device)
            if emb_chunk.shape[1] != dim:
                print(
                    f"ERROR: new embedding dim {emb_chunk.shape[1]} != base dim {dim}. "
                    "Use the same --model as the base .npz.",
                    file=sys.stderr,
                )
                return 1
            new_ids.extend(ids_chunk)
            new_stack.append(emb_chunk)
        extra = np.vstack(new_stack).astype(np.float32, copy=False)
        for i, lid in enumerate(new_ids):
            old_map[lid] = extra[i]
        print(f"  Added {len(new_ids)} vectors.", flush=True)
    else:
        print("  No missing IDs vs base npz; only reordering / checking coverage.", flush=True)

    out_rows = []
    for sid in want_order:
        if sid not in old_map:
            print(f"ERROR: still missing embedding for {sid!r}", file=sys.stderr)
            return 1
        out_rows.append(old_map[sid])
    embeddings = np.stack(out_rows, axis=0).astype(np.float32, copy=False)
    ids_arr = np.array(want_order, dtype=object)
    meta = np.array(
        [
            args.model,
            str(args.positives_csv),
            str(args.negatives_csv),
            str(args.train_ids),
            str(args.test_ids),
            f"incremental_from={args.base_npz}",
        ],
        dtype=object,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, embeddings=embeddings, ids=ids_arr, meta=meta)
    print(f"  Wrote {args.out} shape={embeddings.shape}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
