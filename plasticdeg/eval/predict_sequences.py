"""
Score arbitrary protein sequences with the saved v2 ESM baseline heads (CLI).

Loads ESM-2 + sklearn heads per invocation. For repeated UI testing without reloading weights,
run the HTTP service instead (model stays in memory):

  uvicorn plasticdeg.serve.predict_app:app --host 127.0.0.1 --port 8765

Then set NEXT_PUBLIC_PREDICT_SERVICE_URL=http://127.0.0.1:8765 in frontend/.env.local

  python -m plasticdeg.eval.predict_sequences --in request.json --out response.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from plasticdeg import paths
from plasticdeg.eval.predict_runtime import LoadedPredictor, parse_sequences_payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ESM-2 embed + v2 RF/LR predict on arbitrary sequences")
    p.add_argument("--in", dest="in_path", type=Path, required=True, help="Input JSON path")
    p.add_argument("--out", dest="out_path", type=Path, required=True, help="Output JSON path")
    p.add_argument("--metrics", type=Path, default=paths.metrics_esm_baseline_v2_json())
    p.add_argument("--rf-model", type=Path, default=paths.model_rf_esm_baseline_v2_joblib())
    p.add_argument("--lr-model", type=Path, default=paths.model_lr_esm_baseline_v2_joblib())
    p.add_argument(
        "--embeddings-ref",
        type=Path,
        default=paths.embeddings_esm2_t33_mean_v2_npz(),
        help="npz used only to read ESM checkpoint name from meta[0]",
    )
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args(argv)
    try:
        sys.stderr.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    except Exception:
        pass
    print(
        "predict_sequences: started (CLI loads ESM each run; use predict_app for a warm server)",
        flush=True,
        file=sys.stderr,
    )

    def fail(msg: str, code: int = 1) -> int:
        args.out_path.write_text(json.dumps({"ok": False, "error": msg}, indent=2), encoding="utf-8")
        print(msg, file=sys.stderr)
        return code

    try:
        payload = json.loads(args.in_path.read_text(encoding="utf-8"))
    except Exception as e:
        return fail(f"Invalid input JSON: {e}")

    pairs, err = parse_sequences_payload(payload.get("sequences"))
    if err:
        return fail(err)
    assert pairs is not None

    try:
        pred = LoadedPredictor.load(
            args.metrics,
            args.rf_model,
            args.lr_model,
            args.embeddings_ref,
            device=args.device,
            batch_size=args.batch_size,
        )
        out = pred.predict_pairs(pairs)
    except Exception as e:
        return fail(str(e))

    args.out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"predict_sequences: wrote {args.out_path}", file=sys.stderr, flush=True)
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
