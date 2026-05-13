"""
FastAPI app that loads ESM-2 + v2 RF/LR **once** at startup; `/predict` only runs embedding + sklearn.

Run from repository root:

  uvicorn plasticdeg.serve.predict_app:app --host 127.0.0.1 --port 8765

Frontend: set `NEXT_PUBLIC_PREDICT_SERVICE_URL=http://127.0.0.1:8765` in `frontend/.env.local`.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from plasticdeg import paths
from plasticdeg.eval.predict_runtime import LoadedPredictor, parse_sequences_payload

_predictor: LoadedPredictor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _predictor
    print("predict_app: loading metrics, sklearn heads, and ESM-2 (one-time) …", flush=True)
    _predictor = LoadedPredictor.load(
        paths.metrics_esm_baseline_v2_json(),
        paths.model_rf_esm_baseline_v2_joblib(),
        paths.model_lr_esm_baseline_v2_joblib(),
        paths.embeddings_esm2_t33_mean_v2_npz(),
        device="auto",
        batch_size=4,
    )
    print("predict_app: ready.", flush=True)
    yield
    _predictor = None


app = FastAPI(title="PlasticDeg predict", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://127.0.0.1:3001",
        "http://localhost:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SeqIn(BaseModel):
    id: str | None = None
    sequence: str = Field(..., min_length=1)


class PredictBody(BaseModel):
    sequences: list[SeqIn] = Field(..., min_length=1)


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "model_loaded": _predictor is not None}


@app.post("/predict")
def predict(body: PredictBody) -> dict[str, Any]:
    if _predictor is None:
        raise HTTPException(503, "Predictor not loaded yet")
    raw = [s.model_dump(exclude_none=True) for s in body.sequences]
    pairs, err = parse_sequences_payload(raw)
    if err or pairs is None:
        raise HTTPException(400, err or "invalid sequences")
    out = _predictor.predict_pairs(pairs)
    if not out.get("ok"):
        raise HTTPException(422, out.get("error", "prediction failed"))
    return out
