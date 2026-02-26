"""
Protein sequence embeddings using ESM-2 (Evolutionary Scale Modeling).

ESM-2 is Meta's protein language model that produces rich vector
representations capturing structural and functional properties.
We use the smallest model (esm2_t6_8M) for efficiency on standard hardware.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from tqdm import tqdm

# Model choices (smallest to largest):
#   esm2_t6_8M_UR50D    — 8M params, 320-dim embeddings (fast, good baseline)
#   esm2_t12_35M_UR50D  — 35M params, 480-dim
#   esm2_t30_150M_UR50D — 150M params, 640-dim
#   esm2_t33_650M_UR50D — 650M params, 1280-dim (best accuracy, needs GPU)
MODEL_NAME = "esm2_t6_8M_UR50D"
MAX_SEQ_LEN = 1022  # ESM-2 max input length


def load_esm_model(model_name: str = MODEL_NAME):
    """Load ESM-2 model and alphabet."""
    import esm
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"  Loaded {model_name} on {device}")
    return model, alphabet, device


def get_embedding(model, alphabet, device, sequence: str) -> np.ndarray:
    """Get mean-pooled embedding for a single protein sequence."""
    # Truncate if too long
    seq = sequence[:MAX_SEQ_LEN]

    batch_converter = alphabet.get_batch_converter()
    data = [("protein", seq)]
    _, _, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[model.num_layers])

    # Mean pool over sequence length (excluding BOS/EOS tokens)
    token_embeddings = results["representations"][model.num_layers]
    seq_embedding = token_embeddings[0, 1:len(seq)+1, :].mean(dim=0)

    return seq_embedding.cpu().numpy()


def extract_esm_embeddings(
    sequences: list[str],
    model_name: str = MODEL_NAME,
) -> np.ndarray:
    """Extract ESM-2 embeddings for a list of protein sequences."""
    model, alphabet, device = load_esm_model(model_name)

    embeddings = []
    for seq in tqdm(sequences, desc="  Computing ESM-2 embeddings"):
        emb = get_embedding(model, alphabet, device, seq)
        embeddings.append(emb)

    return np.stack(embeddings)


def main():
    """Extract ESM-2 embeddings from the processed dataset."""
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = project_root / "data" / "processed" / "dataset.csv"
    output_path = project_root / "data" / "embeddings" / "esm2_embeddings.npz"

    print("=" * 60)
    print("  ESM-2 Embedding Extraction")
    print("=" * 60)

    df = pd.read_csv(dataset_path)
    print(f"\n  Loaded {len(df)} sequences")

    sequences = df["sequence"].tolist()
    labels = df["label"].values

    embeddings = extract_esm_embeddings(sequences)

    print(f"\n  Embedding matrix shape: {embeddings.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        labels=labels,
        ids=df["id"].values,
    )
    print(f"  Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
