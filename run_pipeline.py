"""
Main pipeline runner for PlasticDeg Predictor.

Executes the full pipeline:
  1. Data collection from UniProt
  2. Dataset preparation
  3. K-mer feature extraction
  4. ESM-2 embedding extraction (optional, requires torch + esm)
  5. Model training & evaluation

Usage:
  python run_pipeline.py              # Run full pipeline
  python run_pipeline.py --skip-esm   # Skip ESM-2 embeddings (faster)
  python run_pipeline.py --step 3     # Run from step 3 onward
"""

import sys
import argparse


def run_step(step_num: int, name: str, func):
    """Run a pipeline step with formatting."""
    print(f"\n{'#' * 60}")
    print(f"  STEP {step_num}: {name}")
    print(f"{'#' * 60}")
    func()


def main():
    parser = argparse.ArgumentParser(description="PlasticDeg ML Pipeline")
    parser.add_argument("--skip-esm", action="store_true", help="Skip ESM-2 embedding extraction")
    parser.add_argument("--step", type=int, default=1, help="Start from this step (1-5)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  PlasticDeg Predictor — Full ML Pipeline")
    print("=" * 60)

    # Step 1: Data collection
    if args.step <= 1:
        from src.data_collection.fetch_uniprot import main as fetch_main
        run_step(1, "Data Collection (UniProt)", fetch_main)

    # Step 2: Dataset preparation
    if args.step <= 2:
        from src.data_collection.prepare_dataset import main as prep_main
        run_step(2, "Dataset Preparation", prep_main)

    # Step 3: K-mer feature extraction
    if args.step <= 3:
        from src.features.kmer_features import main as kmer_main
        run_step(3, "K-mer Feature Extraction", kmer_main)

    # Step 4: ESM-2 embeddings (optional)
    if args.step <= 4 and not args.skip_esm:
        try:
            from src.features.esm_embeddings import main as esm_main
            run_step(4, "ESM-2 Embedding Extraction", esm_main)
        except ImportError:
            print("\n  SKIPPING Step 4: ESM-2 not installed (pip install fair-esm torch)")
    elif args.skip_esm:
        print("\n  SKIPPING Step 4: ESM-2 embeddings (--skip-esm flag)")

    # Step 5: Model training
    if args.step <= 5:
        from src.models.train import main as train_main
        run_step(5, "Model Training & Evaluation", train_main)

    print("\n" + "=" * 60)
    print("  Pipeline Complete!")
    print("  ")
    print("  Next steps:")
    print("    1. Start the API:  uvicorn src.api.main:app --reload")
    print("    2. Start frontend: cd frontend && npm install && npm run dev")
    print("    3. Open: http://localhost:3000")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
