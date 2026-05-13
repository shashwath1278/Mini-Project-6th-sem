"""
Shared prediction runtime: parse request payloads, load RF/LR + ESM once, embed + score.

Used by `plasticdeg.eval.predict_sequences` (CLI) and `plasticdeg.serve.predict_app` (HTTP, model stays loaded).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from plasticdeg.embed.embed_sequences import _import_torch, mean_pool_esm2_batch

AA_OK = set("ACDEFGHIKLMNPQRSTVWY")
MAX_SEQS = 48
MAX_LEN = 1022


def clean_sequence(raw: str) -> str:
    s = "".join(raw.split()).upper()
    s = re.sub(r"[^A-Z]", "", s)
    return "".join(c for c in s if c in AA_OK)


def default_esm_name() -> str:
    return "esm2_t33_650M_UR50D"


def esm_model_from_training(metrics_path: Path, emb_path: Path) -> str:
    if emb_path.is_file():
        try:
            data = np.load(emb_path, allow_pickle=True)
            meta = data.get("meta")
            if meta is not None and len(meta):
                return str(meta[0])
        except Exception:
            pass
    if metrics_path.is_file():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            ep = m.get("embeddings")
            if isinstance(ep, str) and Path(ep).is_file():
                data = np.load(ep, allow_pickle=True)
                meta = data.get("meta")
                if meta is not None and len(meta):
                    return str(meta[0])
        except Exception:
            pass
    return default_esm_name()


def parse_sequences_payload(seqs_in: object) -> tuple[list[tuple[str, str]] | None, str | None]:
    """Returns (pairs, error_message). pairs is None on error."""
    if not isinstance(seqs_in, list) or not seqs_in:
        return None, 'Body must include non-empty "sequences" array'
    if len(seqs_in) > MAX_SEQS:
        return None, f"At most {MAX_SEQS} sequences per request"
    pairs: list[tuple[str, str]] = []
    for i, item in enumerate(seqs_in):
        if not isinstance(item, dict):
            return None, f"sequences[{i}] must be an object"
        sid = (item.get("id") or f"query_{i + 1}").strip() or f"query_{i + 1}"
        raw = item.get("sequence")
        if not isinstance(raw, str) or not raw.strip():
            return None, f"sequences[{i}].sequence must be a non-empty string"
        seq = clean_sequence(raw)
        if not seq:
            return None, f"sequences[{i}]: no valid amino acids after cleaning"
        if len(seq) > MAX_LEN:
            return None, f"sequences[{i}]: length {len(seq)} exceeds max {MAX_LEN}"
        pairs.append((sid, seq))
    return pairs, None


@dataclass
class LoadedPredictor:
    """Holds RF, LR+scaler, thresholds, and ESM model in memory for repeated predictions."""

    rf: Any
    lr: Any
    scaler: Any
    t_rf: float
    t_lr: float
    esm_name: str
    torch: Any
    esm_model: Any
    alphabet: Any
    device: Any
    batch_converter: Any
    batch_size: int
    expect_dim: int | None
    metrics_path: Path

    @classmethod
    def load(
        cls,
        metrics_path: Path,
        rf_path: Path,
        lr_path: Path,
        embeddings_ref: Path,
        *,
        device: str = "auto",
        batch_size: int = 4,
    ) -> LoadedPredictor:
        if not metrics_path.is_file():
            raise FileNotFoundError(f"Missing metrics JSON: {metrics_path}")
        if not rf_path.is_file():
            raise FileNotFoundError(f"Missing RF model: {rf_path}")
        if not lr_path.is_file():
            raise FileNotFoundError(f"Missing LR bundle: {lr_path}")

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        models = metrics.get("models") or {}
        rf_meta = models.get("random_forest") or {}
        lr_meta = models.get("logistic_regression") or {}
        t_rf = rf_meta.get("threshold_train_recall_ge_0.8")
        t_lr = lr_meta.get("threshold_train_recall_ge_0.8")
        if t_rf is None or t_lr is None:
            raise ValueError("metrics JSON missing threshold_train_recall_ge_0.8 for RF or LR")

        rf = joblib.load(rf_path)
        lr_bundle = joblib.load(lr_path)
        scaler = lr_bundle["scaler"]
        lr = lr_bundle["model"]
        expect_dim = int(rf.n_features_in_) if getattr(rf, "n_features_in_", None) is not None else None

        esm_name = esm_model_from_training(metrics_path, embeddings_ref)
        torch = _import_torch()
        try:
            import esm
        except ImportError as e:
            raise ImportError("fair-esm not installed (pip install fair-esm torch)") from e

        if device == "auto":
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            dev = torch.device(device)

        model, alphabet = esm.pretrained.load_model_and_alphabet(esm_name)
        model = model.to(dev)
        model.eval()
        batch_converter = alphabet.get_batch_converter()

        return cls(
            rf=rf,
            lr=lr,
            scaler=scaler,
            t_rf=float(t_rf),
            t_lr=float(t_lr),
            esm_name=esm_name,
            torch=torch,
            esm_model=model,
            alphabet=alphabet,
            device=dev,
            batch_converter=batch_converter,
            batch_size=max(1, int(batch_size)),
            expect_dim=expect_dim,
            metrics_path=metrics_path,
        )

    def predict_pairs(self, pairs: list[tuple[str, str]]) -> dict[str, Any]:
        all_emb: list[np.ndarray] = []
        bs = self.batch_size
        for start in range(0, len(pairs), bs):
            chunk = pairs[start : start + bs]
            _, emb_chunk = mean_pool_esm2_batch(
                self.esm_model, self.alphabet, self.batch_converter, chunk, self.device
            )
            all_emb.append(emb_chunk)
        X = np.vstack(all_emb).astype(np.float32, copy=False)

        if self.expect_dim is not None and X.shape[1] != self.expect_dim:
            return {
                "ok": False,
                "error": (
                    f"Embedding dim {X.shape[1]} does not match RF n_features_in_={self.expect_dim} "
                    "(wrong ESM checkpoint vs training?)"
                ),
            }

        rf_prob = self.rf.predict_proba(X)[:, 1]
        lr_prob = self.lr.predict_proba(self.scaler.transform(X))[:, 1]
        rf_hat = (rf_prob >= self.t_rf).astype(int)
        lr_hat = (lr_prob >= self.t_lr).astype(int)

        results = []
        for i, (sid, seq) in enumerate(pairs):
            results.append(
                {
                    "id": sid,
                    "sequence_length": len(seq),
                    "rf_probability": float(rf_prob[i]),
                    "rf_predicted_positive": bool(rf_hat[i]),
                    "lr_probability": float(lr_prob[i]),
                    "lr_predicted_positive": bool(lr_hat[i]),
                }
            )

        return {
            "ok": True,
            "esm_model": self.esm_name,
            "thresholds": {"rf": self.t_rf, "lr": self.t_lr},
            "metrics_path": str(self.metrics_path),
            "results": results,
        }
