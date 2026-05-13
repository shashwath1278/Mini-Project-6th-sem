import { NextResponse } from "next/server";
import {
  artifactPath,
  fileExists,
  getRepoRoot,
  lineCount,
  readJsonFile,
} from "@/lib/artifacts";

export const dynamic = "force-dynamic";

export async function GET() {
  const splits = {
    train: artifactPath("splits", "split_train_accessions.txt"),
    test: artifactPath("splits", "split_test_accessions.txt"),
  };
  const tables = {
    positives: artifactPath("tables", "positives_gt.csv"),
    negatives: artifactPath("tables", "negatives_gt.csv"),
    positives_expanded: artifactPath("tables", "positives_gt_expanded.csv"),
  };
  const sequences = {
    positives_fasta: artifactPath("sequences", "positives_from_gt.fasta"),
    negatives_fasta: artifactPath("sequences", "negatives_from_uniprot.fasta"),
    positives_expanded_fasta: artifactPath(
      "sequences",
      "positives_from_gt_expanded.fasta"
    ),
  };
  const embeddings = {
    v2: artifactPath("embeddings", "embeddings_esm2_t33_mean_v2.npz"),
    hard: artifactPath("embeddings", "embeddings_hard_negatives.npz"),
  };
  const models = {
    rf_v2: artifactPath("models", "model_rf_esm_baseline_v2.joblib"),
    lr_v2: artifactPath("models", "model_lr_esm_baseline_v2.joblib"),
  };
  const metrics = {
    v1: artifactPath("metrics", "metrics_esm_baseline.json"),
    v2: artifactPath("metrics", "metrics_esm_baseline_v2.json"),
    evaluation: artifactPath("metrics", "evaluation_report.json"),
  };
  const probes = {
    hard_v2: artifactPath("probes", "hard_negative_probe_v2.json"),
    tiers: artifactPath("probes", "tier_probe_summary.json"),
  };
  const reports = {
    evaluation_txt: artifactPath("reports", "evaluation_report.txt"),
  };

  const exists = (p: string) => fileExists(p);

  const metricsV2 = readJsonFile<Record<string, unknown>>(metrics.v2);
  const metricsV1 = readJsonFile<Record<string, unknown>>(metrics.v1);
  const evaluationReport = readJsonFile<Record<string, unknown>>(
    metrics.evaluation
  );
  const tierSummary = readJsonFile<Record<string, unknown>>(probes.tiers);
  const hardProbeV2 = readJsonFile<Record<string, unknown>>(probes.hard_v2);

  const charts = {
    pr_v1: artifactPath("reports", "pr_curve_rf_esm_baseline.png"),
    pr_v2: artifactPath("reports", "pr_curve_rf_esm_baseline_v2.png"),
    roc_v1: artifactPath("reports", "roc_curve_rf_esm_baseline.png"),
    roc_v2: artifactPath("reports", "roc_curve_rf_esm_baseline_v2.png"),
  };

  return NextResponse.json({
    repoRoot: getRepoRoot(),
    artifacts: {
      splits: {
        train: { path: splits.train, exists: exists(splits.train), lines: lineCount(splits.train) },
        test: { path: splits.test, exists: exists(splits.test), lines: lineCount(splits.test) },
      },
      tables: {
        positives: { path: tables.positives, exists: exists(tables.positives) },
        negatives: { path: tables.negatives, exists: exists(tables.negatives) },
        positives_expanded: {
          path: tables.positives_expanded,
          exists: exists(tables.positives_expanded),
        },
      },
      sequences: {
        positives_fasta: {
          path: sequences.positives_fasta,
          exists: exists(sequences.positives_fasta),
        },
        negatives_fasta: {
          path: sequences.negatives_fasta,
          exists: exists(sequences.negatives_fasta),
        },
        positives_expanded_fasta: {
          path: sequences.positives_expanded_fasta,
          exists: exists(sequences.positives_expanded_fasta),
        },
      },
      embeddings: {
        esm_v2_npz: { path: embeddings.v2, exists: exists(embeddings.v2) },
        hard_npz: { path: embeddings.hard, exists: exists(embeddings.hard) },
      },
      models: {
        rf_v2: { path: models.rf_v2, exists: exists(models.rf_v2) },
        lr_v2: { path: models.lr_v2, exists: exists(models.lr_v2) },
      },
      metrics: {
        baseline_v1: { path: metrics.v1, exists: exists(metrics.v1) },
        baseline_v2: { path: metrics.v2, exists: exists(metrics.v2) },
        evaluation_report: {
          path: metrics.evaluation,
          exists: exists(metrics.evaluation),
        },
      },
      probes: {
        hard_negative_v2: { path: probes.hard_v2, exists: exists(probes.hard_v2) },
        tier_summary: { path: probes.tiers, exists: exists(probes.tiers) },
      },
      reports: {
        evaluation_txt: {
          path: reports.evaluation_txt,
          exists: exists(reports.evaluation_txt),
        },
      },
      charts: {
        pr_v1: { path: charts.pr_v1, exists: exists(charts.pr_v1) },
        pr_v2: { path: charts.pr_v2, exists: exists(charts.pr_v2) },
        roc_v1: { path: charts.roc_v1, exists: exists(charts.roc_v1) },
        roc_v2: { path: charts.roc_v2, exists: exists(charts.roc_v2) },
      },
    },
    metrics_esm_baseline_v2: metricsV2,
    metrics_esm_baseline: metricsV1,
    evaluation_report: evaluationReport,
    tier_probe_summary: tierSummary,
    hard_negative_probe_v2: hardProbeV2,
  });
}
