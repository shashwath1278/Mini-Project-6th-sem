"use client";

import { Check } from "lucide-react";
import type { PipelineArtifactsManifest } from "@/types";

interface Step {
  id: string;
  label: string;
  description: string;
  done: boolean;
}

function buildSteps(a: PipelineArtifactsManifest): Step[] {
  const curvesOk =
    a.charts.pr_v2.exists &&
    a.charts.roc_v2.exists;
  return [
    {
      id: "tables",
      label: "Tables & ground truth",
      description: "positives_gt.csv, negatives_gt.csv",
      done: a.tables.positives.exists && a.tables.negatives.exists,
    },
    {
      id: "splits",
      label: "Homology splits",
      description: "split_train / split_test accession lists",
      done: a.splits.train.exists && a.splits.test.exists,
    },
    {
      id: "embed",
      label: "ESM-2 embeddings (v2)",
      description: "embeddings_esm2_t33_mean_v2.npz",
      done: a.embeddings.esm_v2_npz.exists,
    },
    {
      id: "train",
      label: "Trained heads",
      description: "RF/LR joblib under models/",
      done: a.models.rf_v2.exists && a.models.lr_v2.exists,
    },
    {
      id: "metrics",
      label: "Metrics JSON",
      description: "metrics_esm_baseline_v2.json",
      done: a.metrics.baseline_v2.exists,
    },
    {
      id: "probes",
      label: "Stress probes",
      description: "hard_negative_probe_v2, tier summary",
      done: a.probes.hard_negative_v2.exists && a.probes.tier_summary.exists,
    },
    {
      id: "curves",
      label: "PR / ROC figures",
      description: "reports/pr_curve_rf_esm_baseline_v2.png (+ ROC)",
      done: curvesOk,
    },
  ];
}

interface Props {
  manifest: PipelineArtifactsManifest;
}

export default function PipelineStageRail({ manifest }: Props) {
  const steps = buildSteps(manifest);

  return (
    <div className="rounded-xl border border-border bg-muted/30 p-5">
      <h3 className="section-title mb-4">Pipeline alignment (artifact presence)</h3>
      <p className="text-xs text-muted-foreground mb-5">
        Mirrors <code className="text-[10px]">plasticdeg.paths</code> layout under{" "}
        <code className="text-[10px]">data/processed_v2/</code>.
      </p>
      <div>
        {steps.map((s) => (
          <div key={s.id} className="audit-step">
            <span
              className={`audit-step__dot flex items-center justify-center ${
                s.done ? "bg-success" : "bg-muted-foreground/30"
              }`}
            >
              {s.done ? <Check size={10} className="text-white" strokeWidth={3} /> : null}
            </span>
            <div>
              <p className="text-sm font-medium text-foreground">{s.label}</p>
              <p className="text-xs text-muted-foreground mt-0.5">{s.description}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
