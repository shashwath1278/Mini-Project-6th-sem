/**
 * Types for `plasticdeg` pipeline artifacts served from `/api/pipeline/dashboard`.
 * Shapes follow metrics JSON from `train_esm_baseline` / v2 and evaluation_report.
 */

export interface ArtifactFileInfo {
  path: string;
  exists: boolean;
  lines?: number | null;
}

export interface PipelineArtifactsManifest {
  splits: {
    train: ArtifactFileInfo;
    test: ArtifactFileInfo;
  };
  tables: {
    positives: ArtifactFileInfo;
    negatives: ArtifactFileInfo;
    positives_expanded: ArtifactFileInfo;
  };
  sequences: {
    positives_fasta: ArtifactFileInfo;
    negatives_fasta: ArtifactFileInfo;
    positives_expanded_fasta: ArtifactFileInfo;
  };
  embeddings: {
    esm_v2_npz: ArtifactFileInfo;
    hard_npz: ArtifactFileInfo;
  };
  models: {
    rf_v2: ArtifactFileInfo;
    lr_v2: ArtifactFileInfo;
  };
  metrics: {
    baseline_v1: ArtifactFileInfo;
    baseline_v2: ArtifactFileInfo;
    evaluation_report: ArtifactFileInfo;
  };
  probes: {
    hard_negative_v2: ArtifactFileInfo;
    tier_summary: ArtifactFileInfo;
  };
  reports: {
    evaluation_txt: ArtifactFileInfo;
  };
  charts: {
    pr_v1: ArtifactFileInfo;
    pr_v2: ArtifactFileInfo;
    roc_v1: ArtifactFileInfo;
    roc_v2: ArtifactFileInfo;
  };
}

export interface ModelEvalBlock {
  pr_auc?: number;
  roc_auc?: number;
  mcc_at_threshold?: number;
  accuracy_at_threshold?: number;
  precision_at_threshold?: number;
  recall_at_threshold?: number;
  confusion_matrix?: number[][];
}

export interface ModelMetricsEntry {
  name: string;
  /** sklearn-style train threshold at recall target (JSON key contains a dot). */
  "threshold_train_recall_ge_0.8"?: number;
  threshold_frozen_from_train?: number;
  n_test_combined?: number;
  n_test?: number;
  test_combined?: ModelEvalBlock;
  test_gold_positives_plus_all_negatives?: ModelEvalBlock;
  pr_auc?: number;
  roc_auc?: number;
  mcc_at_threshold?: number;
}

export interface MetricsEsmBaselineJson {
  recall_target_for_threshold?: number;
  tier1_weight?: number;
  tier2_weight?: number;
  note?: string;
  models?: Record<string, ModelMetricsEntry>;
}

export interface HardNegativeProbePayload {
  baseline?: string;
  "threshold_rf_train_recall_ge_0.8"?: number;
  hard_negative_count?: number;
  hard_negative_predicted_positive_count?: number;
  hard_negative_false_positive_rate?: number;
  hard_negative_score_mean?: number;
  hard_negative_score_median?: number;
  easy_test_negative_count?: number;
  easy_test_negative_score_mean?: number;
}

export interface TierRow {
  tier_label: string;
  n_probe_sequences?: number;
  predicted_positive_count?: number;
  false_positive_rate?: number;
  score_mean?: number;
  score_median?: number;
  threshold_rf_frozen?: number;
}

export interface TierProbeSummary {
  tiers?: TierRow[];
  note?: string;
}

export interface PipelineDashboardResponse {
  repoRoot: string;
  artifacts: PipelineArtifactsManifest;
  metrics_esm_baseline_v2: MetricsEsmBaselineJson | null;
  metrics_esm_baseline: MetricsEsmBaselineJson | null;
  evaluation_report: Record<string, unknown> | null;
  tier_probe_summary: TierProbeSummary | null;
  hard_negative_probe_v2: HardNegativeProbePayload | null;
}

export interface PredictResultRow {
  id: string;
  sequence_length: number;
  rf_probability: number;
  rf_predicted_positive: boolean;
  lr_probability: number;
  lr_predicted_positive: boolean;
}

export interface PredictResponse {
  ok: boolean;
  error?: string;
  esm_model?: string;
  thresholds?: { rf: number; lr: number };
  metrics_path?: string;
  results?: PredictResultRow[];
}

export type PredictStatusResponse =
  | { status: "running" }
  | { status: "done"; result: PredictResponse }
  | { status: "error"; error: string; exit_code?: number };
