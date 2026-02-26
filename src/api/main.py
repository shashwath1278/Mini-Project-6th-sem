"""
FastAPI backend for plastic-degrading enzyme prediction.

Endpoints:
  POST /predict        — Predict from a raw protein sequence
  POST /predict/fasta  — Predict from an uploaded FASTA file
  GET  /health         — Health check
  GET  /models         — List available models and their metrics
"""

import json
import numpy as np
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib

from src.features.kmer_features import extract_kmer_features

app = FastAPI(
    title="PlasticDeg Predictor",
    description="AI prediction of plastic-degrading potential in microbial enzymes",
    version="1.0.0",
)

# Allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load models at startup ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "src" / "models" / "saved"

loaded_models = {}


@app.on_event("startup")
def load_models():
    """Load saved models on startup."""
    model_files = {
        "rf_kmer": MODELS_DIR / "rf_kmer.joblib",
        "mlp_esm": MODELS_DIR / "mlp_esm.joblib",
    }
    for name, path in model_files.items():
        if path.exists():
            loaded_models[name] = joblib.load(path)
            print(f"  Loaded model: {name}")
        else:
            print(f"  Model not found (skipping): {name}")


# --- Request/Response schemas ---

class SequenceRequest(BaseModel):
    sequence: str
    model_name: str = "rf_kmer"


class PredictionResponse(BaseModel):
    sequence_length: int
    model_used: str
    prediction: str  # "Plastic-Degrading" or "Non-Degrading"
    confidence: float
    probability_positive: float
    probability_negative: float


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": list(loaded_models.keys())}


@app.get("/models")
def list_models():
    """List available models and their evaluation metrics."""
    metrics_path = MODELS_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)

    return {
        "available_models": list(loaded_models.keys()),
        "metrics": metrics,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_sequence(req: SequenceRequest):
    """Predict plastic-degrading potential for a protein sequence."""
    # Validate
    seq = req.sequence.strip().upper()
    seq = "".join(c for c in seq if c.isalpha())

    if len(seq) < 20:
        raise HTTPException(status_code=400, detail="Sequence too short (minimum 20 amino acids)")
    if len(seq) > 5000:
        raise HTTPException(status_code=400, detail="Sequence too long (maximum 5000 amino acids)")

    model_name = req.model_name
    if model_name not in loaded_models:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Choose from: {list(loaded_models.keys())}",
        )

    bundle = loaded_models[model_name]
    model = bundle["model"]
    scaler = bundle["scaler"]

    # Extract features based on model type
    if "kmer" in model_name:
        features = extract_kmer_features([seq], k_values=[1, 2])
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' requires ESM-2 embeddings (not supported in real-time API yet)",
        )

    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    label = "Plastic-Degrading" if prediction == 1 else "Non-Degrading"
    confidence = float(max(probabilities))

    return PredictionResponse(
        sequence_length=len(seq),
        model_used=model_name,
        prediction=label,
        confidence=round(confidence, 4),
        probability_positive=round(float(probabilities[1]), 4),
        probability_negative=round(float(probabilities[0]), 4),
    )


@app.post("/predict/fasta")
async def predict_fasta(file: UploadFile = File(...)):
    """Predict for all sequences in an uploaded FASTA file."""
    from Bio import SeqIO
    from io import StringIO

    content = await file.read()
    text = content.decode("utf-8")

    sequences = []
    for record in SeqIO.parse(StringIO(text), "fasta"):
        sequences.append({"id": record.id, "sequence": str(record.seq)})

    if not sequences:
        raise HTTPException(status_code=400, detail="No valid sequences found in FASTA file")
    if len(sequences) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 sequences per request")

    # Use RF k-mer model by default
    if "rf_kmer" not in loaded_models:
        raise HTTPException(status_code=500, detail="No model loaded")

    bundle = loaded_models["rf_kmer"]
    model = bundle["model"]
    scaler = bundle["scaler"]

    seqs = [s["sequence"] for s in sequences]
    features = extract_kmer_features(seqs, k_values=[1, 2])
    features_scaled = scaler.transform(features)

    predictions = model.predict(features_scaled)
    probabilities = model.predict_proba(features_scaled)

    results = []
    for i, s in enumerate(sequences):
        results.append({
            "id": s["id"],
            "sequence_length": len(s["sequence"]),
            "prediction": "Plastic-Degrading" if predictions[i] == 1 else "Non-Degrading",
            "confidence": round(float(max(probabilities[i])), 4),
            "probability_positive": round(float(probabilities[i][1]), 4),
        })

    return {"results": results, "total": len(results)}
