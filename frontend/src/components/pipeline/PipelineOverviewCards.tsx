"use client";

import {
  Activity,
  Binary,
  FlaskConical,
  Layers,
  Target,
} from "lucide-react";
import SpotlightCard from "@/components/ui/SpotlightCard";
import MetricHint from "@/components/ui/MetricHint";
import { METRIC_TIPS } from "@/lib/metricLabels";
import type { MetricsEsmBaselineJson } from "@/types";

function pickRfBlock(m: MetricsEsmBaselineJson | null) {
  const rf = m?.models?.random_forest;
  if (!rf) return null;
  const block = rf.test_combined ?? {
    pr_auc: rf.pr_auc,
    roc_auc: rf.roc_auc,
    mcc_at_threshold: rf.mcc_at_threshold,
    accuracy_at_threshold: undefined,
    precision_at_threshold: undefined,
    recall_at_threshold: undefined,
  };
  return { rf, block };
}

interface Props {
  metricsV2: MetricsEsmBaselineJson | null;
  recallTarget: number;
}

export default function PipelineOverviewCards({ metricsV2, recallTarget }: Props) {
  const picked = pickRfBlock(metricsV2);
  const rf = picked?.rf;
  const block = picked?.block;
  const thr =
    rf?.["threshold_train_recall_ge_0.8"] ?? rf?.threshold_frozen_from_train;

  const cards = [
    {
      key: "rf-cutoff",
      label: (
        <MetricHint title={METRIC_TIPS.trainThrRf}>RF cutoff (training)</MetricHint>
      ),
      value: thr != null ? thr.toFixed(4) : "—",
      sub: `Chosen so about ≥${(recallTarget * 100).toFixed(0)}% of training positives score above it; then frozen for test.`,
      icon: <Target size={16} className="text-muted-foreground" />,
    },
    {
      key: "pr-auc",
      label: (
        <MetricHint title={METRIC_TIPS.prAuc}>PR-AUC (test, RF)</MetricHint>
      ),
      value: block?.pr_auc != null ? block.pr_auc.toFixed(4) : "—",
      sub: "Precision–recall area · held-out homology test",
      icon: <Activity size={16} className="text-muted-foreground" />,
    },
    {
      key: "roc-auc",
      label: (
        <MetricHint title={METRIC_TIPS.rocAuc}>ROC-AUC (test, RF)</MetricHint>
      ),
      value: block?.roc_auc != null ? block.roc_auc.toFixed(4) : "—",
      sub: "Receiver-operating-characteristic area · same test split",
      icon: <Layers size={16} className="text-muted-foreground" />,
    },
    {
      key: "mcc",
      label: (
        <MetricHint title={METRIC_TIPS.mcc}>MCC @ RF cutoff (test)</MetricHint>
      ),
      value: block?.mcc_at_threshold != null ? block.mcc_at_threshold.toFixed(3) : "—",
      sub: "How well RF matches labels on held-out test at the training cutoff",
      icon: <Binary size={16} className="text-muted-foreground" />,
    },
  ];

  const extras: { label: string; pass: boolean; tip: string }[] = [
    {
      label: "Metrics v2 present",
      pass: !!metricsV2?.models?.random_forest,
      tip: "v2 metrics JSON is loaded (tier-weighted training snapshot).",
    },
    {
      label: "Silver down-weight",
      pass:
        metricsV2?.tier1_weight != null &&
        metricsV2?.tier2_weight != null &&
        metricsV2.tier2_weight < metricsV2.tier1_weight,
      tip: METRIC_TIPS.silverDown,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className="inline-flex items-center gap-1.5 rounded-full border border-border px-3 py-1 text-xs font-medium text-muted-foreground cursor-help"
          title={`${METRIC_TIPS.esm} ${METRIC_TIPS.rf} ${METRIC_TIPS.lr}`}
        >
          <FlaskConical size={12} />
          ESM-2 + RF / LR baseline
        </span>
        {extras.map((e) => (
          <div
            key={e.label}
            title={e.tip}
            className={`metric-pill cursor-help ${e.pass ? "metric-pill--pass" : "metric-pill--fail"}`}
          >
            {e.label}
          </div>
        ))}
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 stagger">
        {cards.map((c) => (
          <SpotlightCard
            key={c.key}
            spotlightColor="rgba(34, 197, 94, 0.12)"
            className="p-4 animate-fade-in-up"
          >
            <div className="mb-2">{c.icon}</div>
            <p className="text-xs text-muted-foreground leading-snug">{c.label}</p>
            <p className="text-lg font-semibold mt-0.5 tracking-tight">{c.value}</p>
            <p className="text-[11px] text-muted-foreground mt-1 leading-snug">{c.sub}</p>
          </SpotlightCard>
        ))}
      </div>
    </div>
  );
}
