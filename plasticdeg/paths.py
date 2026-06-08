"""
Single layout for pipeline artifacts under data/processed_v2/.

  splits/       train/test ID lists
  tables/       CSV tables (positives, negatives, errors, PlasticDB sidecars)
  sequences/    FASTA (gold positives, UniProt negatives)
  embeddings/   .npz (gitignored)
  models/       .joblib (gitignored)
  metrics/      JSON metrics, evaluation_report.json
  reports/      .txt + .png figures
  probes/       hard negatives fetch, probe JSON, tier summaries, adversarial CSV/FASTA
  expansion/    BLAST-expanded positives + log
"""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def processed_v2() -> Path:
    return project_root() / "data" / "processed_v2"


def splits_dir() -> Path:
    return processed_v2() / "splits"


def tables_dir() -> Path:
    return processed_v2() / "tables"


def sequences_dir() -> Path:
    return processed_v2() / "sequences"


def embeddings_dir() -> Path:
    return processed_v2() / "embeddings"


def models_dir() -> Path:
    return processed_v2() / "models"


def metrics_dir() -> Path:
    return processed_v2() / "metrics"


def reports_dir() -> Path:
    return processed_v2() / "reports"


def manuscript_bundle_dir() -> Path:
    """Report-ready copies + evaluation snippets (see plasticdeg.bundle.manuscript_bundle)."""
    return reports_dir() / "manuscript_bundle"


def report_figure_bundle_dir() -> Path:
    """High-resolution PNG/PDF figures for manuscripts (see plasticdeg.eval.report_figure_bundle)."""
    return reports_dir() / "figure_bundle"


def probes_dir() -> Path:
    return processed_v2() / "probes"


def expansion_dir() -> Path:
    return processed_v2() / "expansion"


def external_dir() -> Path:
    return project_root() / "data" / "external"


def ensure_artifact_dirs() -> None:
    for d in (
        splits_dir(),
        tables_dir(),
        sequences_dir(),
        embeddings_dir(),
        models_dir(),
        metrics_dir(),
        reports_dir(),
        probes_dir(),
        expansion_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)


# --- Common files (defaults for argparse) ---

def split_train_txt() -> Path:
    return splits_dir() / "split_train_accessions.txt"


def split_test_txt() -> Path:
    return splits_dir() / "split_test_accessions.txt"


def positives_gt_csv() -> Path:
    return tables_dir() / "positives_gt.csv"


def negatives_gt_csv() -> Path:
    return tables_dir() / "negatives_gt.csv"


def positives_from_gt_fasta() -> Path:
    return sequences_dir() / "positives_from_gt.fasta"


def positives_gt_expanded_csv() -> Path:
    return tables_dir() / "positives_gt_expanded.csv"


def positives_from_gt_expanded_fasta() -> Path:
    return sequences_dir() / "positives_from_gt_expanded.fasta"


def accession_sequence_alias_map_csv() -> Path:
    return tables_dir() / "accession_sequence_alias_map.csv"


def negatives_from_uniprot_fasta() -> Path:
    return sequences_dir() / "negatives_from_uniprot.fasta"


def embeddings_esm2_t33_mean_v2_npz() -> Path:
    return embeddings_dir() / "embeddings_esm2_t33_mean_v2.npz"


def metrics_esm_baseline_json() -> Path:
    return metrics_dir() / "metrics_esm_baseline.json"


def metrics_esm_baseline_v2_json() -> Path:
    return metrics_dir() / "metrics_esm_baseline_v2.json"


def model_lr_esm_baseline_joblib() -> Path:
    return models_dir() / "model_lr_esm_baseline.joblib"


def model_rf_esm_baseline_joblib() -> Path:
    return models_dir() / "model_rf_esm_baseline.joblib"


def model_lr_esm_baseline_v2_joblib() -> Path:
    return models_dir() / "model_lr_esm_baseline_v2.joblib"


def model_rf_esm_baseline_v2_joblib() -> Path:
    return models_dir() / "model_rf_esm_baseline_v2.joblib"


def hard_negatives_gt_csv() -> Path:
    return probes_dir() / "hard_negatives_gt.csv"


def hard_negatives_fasta() -> Path:
    return probes_dir() / "hard_negatives.fasta"


def embeddings_hard_negatives_npz() -> Path:
    return embeddings_dir() / "embeddings_hard_negatives.npz"


def hard_negative_probe_json() -> Path:
    return probes_dir() / "hard_negative_probe.json"


def hard_negative_probe_v2_json() -> Path:
    return probes_dir() / "hard_negative_probe_v2.json"


def adversarial_negatives_gt_csv() -> Path:
    return probes_dir() / "adversarial_negatives_gt.csv"


def adversarial_negatives_fasta() -> Path:
    return probes_dir() / "adversarial_negatives.fasta"


def embeddings_adversarial_npz() -> Path:
    return embeddings_dir() / "embeddings_adversarial_negatives.npz"


def adversarial_probe_json() -> Path:
    return probes_dir() / "adversarial_negative_probe.json"


def tier_probe_summary_json() -> Path:
    return probes_dir() / "tier_probe_summary.json"


def expanded_positives_csv() -> Path:
    return expansion_dir() / "expanded_positives.csv"


def expanded_positives_fasta() -> Path:
    return expansion_dir() / "expanded_positives.fasta"


def blast_expansion_log_json() -> Path:
    return expansion_dir() / "blast_expansion_log.json"


def evaluation_report_json() -> Path:
    return metrics_dir() / "evaluation_report.json"


def evaluation_report_txt() -> Path:
    return reports_dir() / "evaluation_report.txt"


def test_errors_rf_csv() -> Path:
    return tables_dir() / "test_errors_rf.csv"


def test_errors_lr_csv() -> Path:
    return tables_dir() / "test_errors_lr.csv"


def label_audit_candidates_csv() -> Path:
    return tables_dir() / "label_audit_candidates.csv"


def pr_curve_rf_png(suffix: str = "") -> Path:
    return reports_dir() / f"pr_curve_rf_esm_baseline{suffix}.png"


def roc_curve_rf_png(suffix: str = "") -> Path:
    return reports_dir() / f"roc_curve_rf_esm_baseline{suffix}.png"


def plasticdb_additions_json() -> Path:
    return tables_dir() / "plasticdb_additions.json"


def plasticdb_import_skips_txt() -> Path:
    return tables_dir() / "plasticdb_import_skips.txt"


def plasticdb_fasta_default() -> Path:
    return external_dir() / "PlasticDB.fasta"
