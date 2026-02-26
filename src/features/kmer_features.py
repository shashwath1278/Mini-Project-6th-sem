"""
K-mer frequency feature extraction from protein sequences.

Computes amino acid composition (1-mer), dipeptide composition (2-mer),
and optionally tripeptide composition (3-mer) as numerical feature vectors.
"""

import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def get_kmer_vocab(k: int) -> list[str]:
    """Generate all possible k-mers from the 20 standard amino acids."""
    return ["".join(p) for p in product(AMINO_ACIDS, repeat=k)]


def compute_kmer_frequencies(sequence: str, k: int) -> dict[str, float]:
    """Compute normalized k-mer frequencies for a single sequence."""
    vocab = get_kmer_vocab(k)
    counts = {kmer: 0 for kmer in vocab}

    # Clean sequence — keep only standard amino acids
    seq = "".join(c for c in sequence.upper() if c in AMINO_ACIDS)

    total = len(seq) - k + 1
    if total <= 0:
        return counts

    for i in range(total):
        kmer = seq[i:i+k]
        if kmer in counts:
            counts[kmer] += 1

    # Normalize to frequencies
    for kmer in counts:
        counts[kmer] /= total

    return counts


def extract_kmer_features(sequences: list[str], k_values: list[int] = None) -> np.ndarray:
    """
    Extract k-mer feature vectors for a list of sequences.

    Args:
        sequences: List of protein sequences
        k_values: List of k values (default: [1, 2] for amino acid + dipeptide)

    Returns:
        Feature matrix of shape (n_sequences, n_features)
    """
    if k_values is None:
        k_values = [1, 2]

    all_features = []

    for seq in sequences:
        feature_vector = []
        for k in k_values:
            freqs = compute_kmer_frequencies(seq, k)
            feature_vector.extend(freqs.values())
        all_features.append(feature_vector)

    return np.array(all_features, dtype=np.float32)


def get_feature_names(k_values: list[int] = None) -> list[str]:
    """Get feature names corresponding to extract_kmer_features output."""
    if k_values is None:
        k_values = [1, 2]
    names = []
    for k in k_values:
        names.extend([f"kmer_{k}_{kmer}" for kmer in get_kmer_vocab(k)])
    return names


def main():
    """Extract k-mer features from the processed dataset."""
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_path = project_root / "data" / "processed" / "dataset.csv"
    output_path = project_root / "data" / "processed" / "features_kmer.npz"

    print("=" * 60)
    print("  K-mer Feature Extraction")
    print("=" * 60)

    df = pd.read_csv(dataset_path)
    print(f"\n  Loaded {len(df)} sequences")

    sequences = df["sequence"].tolist()
    labels = df["label"].values

    k_values = [1, 2]  # Amino acid composition + dipeptide composition
    print(f"  Computing k-mer features for k = {k_values}...")

    features = extract_kmer_features(sequences, k_values)
    feature_names = get_feature_names(k_values)

    print(f"  Feature matrix shape: {features.shape}")
    print(f"  Feature names: {len(feature_names)} (first 5: {feature_names[:5]})")

    # Save
    np.savez(
        output_path,
        features=features,
        labels=labels,
        feature_names=np.array(feature_names),
        ids=df["id"].values,
    )
    print(f"\n  Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
