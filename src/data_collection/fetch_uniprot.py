"""
Fetch known plastic-degrading enzyme sequences from UniProt.

Queries UniProt REST API for enzymes known to degrade plastics:
  - PETase, MHETase, cutinase, laccase, manganese peroxidase, lipase

Saves results as FASTA files in data/raw/positive/
"""

import os
import time
import requests
from pathlib import Path

BASE_URL = "https://rest.uniprot.org/uniprotkb/search"

# Queries targeting known plastic-degrading enzyme families
# Uses correct UniProt REST API field names: protein_name, gene, ec, organism_id
# Taxonomy IDs: 2 = Bacteria, 4751 = Fungi
PLASTIC_ENZYME_QUERIES = {
    "petase": "(protein_name:PETase OR gene:petase) AND reviewed:true",
    "mhetase": "(protein_name:MHETase OR gene:mhetase) AND reviewed:true",
    "cutinase": "ec:3.1.1.74 AND reviewed:true",                          # EC for cutinase
    "laccase": "ec:1.10.3.2 AND reviewed:true",                           # EC for laccase
    "lipase": "ec:3.1.1.3 AND reviewed:true AND (taxonomy_id:2 OR taxonomy_id:4751)",  # microbial lipases
    "peroxidase_mn": "ec:1.11.1.13 AND reviewed:true",                    # Mn peroxidase
    "peroxidase_lignin": "ec:1.11.1.14 AND reviewed:true",                # lignin peroxidase
    "esterase": "protein_name:esterase AND protein_name:polyester AND reviewed:true",
}

# Broad query for negative samples — general metabolic enzymes unlikely to degrade plastic
NEGATIVE_QUERIES = {
    "dehydrogenase": "ec:1.1.1.1 AND reviewed:true AND (taxonomy_id:2 OR taxonomy_id:4751)",   # alcohol dehydrogenase
    "hexokinase": "ec:2.7.1.1 AND reviewed:true AND (taxonomy_id:2 OR taxonomy_id:4751)",      # hexokinase
    "dna_polymerase": "protein_name:\"DNA polymerase\" AND reviewed:true AND taxonomy_id:2",
    "rna_polymerase": "protein_name:\"RNA polymerase\" AND reviewed:true AND taxonomy_id:2",
}


def fetch_fasta_from_uniprot(query: str, max_results: int = 500) -> str:
    """Fetch FASTA sequences from UniProt REST API."""
    params = {
        "query": query,
        "format": "fasta",
        "size": max_results,
    }
    print(f"  Querying UniProt: {query[:80]}...")
    resp = requests.get(BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp.text


def count_sequences(fasta_text: str) -> int:
    """Count number of sequences in FASTA text."""
    return fasta_text.count(">")


def save_fasta(fasta_text: str, filepath: Path):
    """Save FASTA text to file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        f.write(fasta_text)


def collect_positive_samples(output_dir: Path, max_per_query: int = 500):
    """Download known plastic-degrading enzyme sequences."""
    print("\n=== Collecting POSITIVE samples (plastic-degrading enzymes) ===\n")
    all_fasta = ""

    for name, query in PLASTIC_ENZYME_QUERIES.items():
        try:
            fasta = fetch_fasta_from_uniprot(query, max_results=max_per_query)
            n = count_sequences(fasta)
            print(f"  {name}: {n} sequences fetched")
            if n > 0:
                save_fasta(fasta, output_dir / f"positive_{name}.fasta")
                all_fasta += fasta
            time.sleep(1)  # Rate limiting
        except Exception as e:
            print(f"  WARNING: Failed to fetch {name}: {e}")

    # Save combined file
    total = count_sequences(all_fasta)
    if total > 0:
        save_fasta(all_fasta, output_dir / "positive_all.fasta")
    print(f"\n  Total positive sequences: {total}")
    return total


def collect_negative_samples(output_dir: Path, max_per_query: int = 250):
    """Download non-plastic-degrading enzyme sequences as negative controls."""
    print("\n=== Collecting NEGATIVE samples (non-plastic-degrading enzymes) ===\n")
    all_fasta = ""

    for name, query in NEGATIVE_QUERIES.items():
        try:
            fasta = fetch_fasta_from_uniprot(query, max_results=max_per_query)
            n = count_sequences(fasta)
            print(f"  {name}: {n} sequences fetched")
            if n > 0:
                save_fasta(fasta, output_dir / f"negative_{name}.fasta")
                all_fasta += fasta
            time.sleep(1)
        except Exception as e:
            print(f"  WARNING: Failed to fetch {name}: {e}")

    total = count_sequences(all_fasta)
    if total > 0:
        save_fasta(all_fasta, output_dir / "negative_all.fasta")
    print(f"\n  Total negative sequences: {total}")
    return total


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"

    print("=" * 60)
    print("  UniProt Data Collection for Plastic-Degrading Enzymes")
    print("=" * 60)

    pos = collect_positive_samples(raw_dir, max_per_query=500)
    neg = collect_negative_samples(raw_dir, max_per_query=250)

    print("\n" + "=" * 60)
    print(f"  Done! Positive: {pos} | Negative: {neg}")
    print(f"  Files saved to: {raw_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
