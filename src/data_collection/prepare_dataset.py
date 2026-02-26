"""
Prepare a clean, labeled dataset from raw FASTA files.

Reads positive and negative FASTA files, assigns labels, removes
duplicates, and saves a processed CSV + combined FASTA for downstream use.

Output columns: id, sequence, length, label (1=plastic-degrading, 0=non-degrading)
"""

import csv
from pathlib import Path
from Bio import SeqIO


def parse_fasta_dir(raw_dir: Path):
    """Parse all positive and negative FASTA files into labeled records."""
    records = []
    seen_ids = set()

    # Positive samples
    for fasta_file in sorted(raw_dir.glob("positive_*.fasta")):
        if fasta_file.name == "positive_all.fasta":
            continue  # Skip combined file to avoid duplicates
        for record in SeqIO.parse(fasta_file, "fasta"):
            uid = record.id
            if uid not in seen_ids:
                seen_ids.add(uid)
                records.append({
                    "id": uid,
                    "description": record.description,
                    "sequence": str(record.seq),
                    "length": len(record.seq),
                    "label": 1,
                    "source": fasta_file.stem,
                })

    # Negative samples
    for fasta_file in sorted(raw_dir.glob("negative_*.fasta")):
        for record in SeqIO.parse(fasta_file, "fasta"):
            uid = record.id
            if uid not in seen_ids:
                seen_ids.add(uid)
                records.append({
                    "id": uid,
                    "description": record.description,
                    "sequence": str(record.seq),
                    "length": len(record.seq),
                    "label": 0,
                    "source": fasta_file.stem,
                })

    return records


def filter_sequences(records, min_length=50, max_length=5000):
    """Remove sequences that are too short or too long."""
    filtered = [r for r in records if min_length <= r["length"] <= max_length]
    removed = len(records) - len(filtered)
    if removed > 0:
        print(f"  Filtered out {removed} sequences outside length range [{min_length}, {max_length}]")
    return filtered


def save_csv(records, filepath: Path):
    """Save records as a CSV file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "description", "sequence", "length", "label", "source"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def save_labeled_fasta(records, filepath: Path):
    """Save combined labeled FASTA (label in header)."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        for r in records:
            label_str = "POSITIVE" if r["label"] == 1 else "NEGATIVE"
            f.write(f">{r['id']} | {label_str} | {r['description']}\n")
            # Write sequence in 80-char lines
            seq = r["sequence"]
            for i in range(0, len(seq), 80):
                f.write(seq[i:i+80] + "\n")


def main():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"

    print("=" * 60)
    print("  Dataset Preparation")
    print("=" * 60)

    # Parse
    records = parse_fasta_dir(raw_dir)
    pos = sum(1 for r in records if r["label"] == 1)
    neg = sum(1 for r in records if r["label"] == 0)
    print(f"\n  Raw sequences: {len(records)} (positive: {pos}, negative: {neg})")

    # Filter
    records = filter_sequences(records)
    pos = sum(1 for r in records if r["label"] == 1)
    neg = sum(1 for r in records if r["label"] == 0)
    print(f"  After filtering: {len(records)} (positive: {pos}, negative: {neg})")

    # Save
    save_csv(records, processed_dir / "dataset.csv")
    save_labeled_fasta(records, processed_dir / "dataset_labeled.fasta")

    print(f"\n  Saved to: {processed_dir}")
    print(f"    - dataset.csv")
    print(f"    - dataset_labeled.fasta")
    print("=" * 60)


if __name__ == "__main__":
    main()
