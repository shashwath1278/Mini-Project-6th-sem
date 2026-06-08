"use client";

import type { ReactNode } from "react";
import { Activity, Binary, Layers, Target } from "lucide-react";
import SpotlightCard from "@/components/ui/SpotlightCard";
import MetricHint from "@/components/ui/MetricHint";
import { METRIC_TIPS } from "@/lib/metricLabels";
import type { MetricsEsmBaselineJson, ModelEvalBlock, ModelMetricsEntry } from "@/types";

type HeadKey = "logistic_regression" | "random_forest";

function pickHeadBlock(m: MetricsEsmBaselineJson | null, key: HeadKey) {
  const head = m?.models?.[key] as ModelMetricsEntry | undefined;
  if (!head) return null;
  const block: ModelEvalBlock =
    head.test_combined ??
    ({
      pr_auc: head.pr_auc,
      roc_auc: head.roc_auc,
      mcc_at_threshold: head.mcc_at_threshold,
    } as ModelEvalBlock);
  return { head, block };
}

function thresholdOf(head: ModelMetricsEntry) {
  return head["threshold_train_recall_ge_0.8"] ?? head.threshold_frozen_from_train;
}

interface Props {
  metricsV2: MetricsEsmBaselineJson | null;
}

function HeadColumn({
  title,
  accentClass,
  spotlight,
  cards,
}: {
  title: string;
  accentClass: string;
  spotlight: string;
  cards: {
    key: string;
    label: ReactNode;
    value: string;
    sub: string;
    icon: ReactNode;
  }[];
}) {
  return (
    <div className={`rounded-2xl border border-border/80 bg-muted/20 p-4 dark:bg-white/[0.03] ${accentClass}`}>
      <p className="mb-3 text-center text-xs font-semibold uppercase tracking-widest text-muted-foreground">
        {title}
      </p>
      <div className="grid grid-cols-2 gap-3">
        {cards.map((c) => (
          <SpotlightCard
            key={c.key}
            spotlightColor={spotlight}
            className="p-3 animate-fade-in-up"
          >
            <div className="mb-1.5">{c.icon}</div>
            <p className="text-[11px] leading-snug text-muted-foreground">{c.label}</p>
            <p className="mt-0.5 text-base font-semibold tabular-nums tracking-tight">{c.value}</p>
            {c.sub ? (
              <p className="mt-1 text-[10px] leading-snug text-muted-foreground">{c.sub}</p>
            ) : null}
          </SpotlightCard>
        ))}
      </div>
    </div>
  );
}

export default function PipelineOverviewCards({ metricsV2 }: Props) {
  const lrP = pickHeadBlock(metricsV2, "logistic_regression");
  const rfP = pickHeadBlock(metricsV2, "random_forest");
  const lr = lrP?.head;
  const lrB = lrP?.block;
  const rf = rfP?.head;
  const rfB = rfP?.block;

  const tLr = lr ? thresholdOf(lr) : null;
  const tRf = rf ? thresholdOf(rf) : null;

  const lrCards = [
    {
      key: "lr-cutoff",
      label: (
        <MetricHint title={METRIC_TIPS.trainThrLr}>LR cutoff (training, τ_LR)</MetricHint>
      ),
      value: tLr != null ? Number(tLr).toFixed(4) : "—",
      sub: "",
      icon: <Target size={14} className="text-info" />,
    },
    {
      key: "lr-pr",
      label: <MetricHint title={METRIC_TIPS.prAuc}>PR-AUC (test)</MetricHint>,
      value: lrB?.pr_auc != null ? lrB.pr_auc.toFixed(4) : "—",
      sub: "",
      icon: <Activity size={14} className="text-info" />,
    },
    {
      key: "lr-roc",
      label: <MetricHint title={METRIC_TIPS.rocAuc}>ROC-AUC (test)</MetricHint>,
      value: lrB?.roc_auc != null ? lrB.roc_auc.toFixed(4) : "—",
      sub: "",
      icon: <Layers size={14} className="text-info" />,
    },
    {
      key: "lr-mcc",
      label: <MetricHint title={METRIC_TIPS.mcc}>MCC @ LR cutoff (test)</MetricHint>,
      value: lrB?.mcc_at_threshold != null ? lrB.mcc_at_threshold.toFixed(3) : "—",
      sub: "",
      icon: <Binary size={14} className="text-info" />,
    },
  ];

  const rfCards = [
    {
      key: "rf-cutoff",
      label: (
        <MetricHint title={METRIC_TIPS.trainThrRf}>RF cutoff (training, τ_RF)</MetricHint>
      ),
      value: tRf != null ? Number(tRf).toFixed(4) : "—",
      sub: "",
      icon: <Target size={14} className="text-success" />,
    },
    {
      key: "rf-pr",
      label: <MetricHint title={METRIC_TIPS.prAuc}>PR-AUC (test)</MetricHint>,
      value: rfB?.pr_auc != null ? rfB.pr_auc.toFixed(4) : "—",
      sub: "",
      icon: <Activity size={14} className="text-success" />,
    },
    {
      key: "rf-roc",
      label: <MetricHint title={METRIC_TIPS.rocAuc}>ROC-AUC (test)</MetricHint>,
      value: rfB?.roc_auc != null ? rfB.roc_auc.toFixed(4) : "—",
      sub: "",
      icon: <Layers size={14} className="text-success" />,
    },
    {
      key: "rf-mcc",
      label: <MetricHint title={METRIC_TIPS.mcc}>MCC @ RF cutoff (test)</MetricHint>,
      value: rfB?.mcc_at_threshold != null ? rfB.mcc_at_threshold.toFixed(3) : "—",
      sub: "",
      icon: <Binary size={14} className="text-success" />,
    },
  ];

  return (
    <div className="space-y-4">
      <div className="grid gap-5 lg:grid-cols-2">
        <HeadColumn
          title="Logistic regression (LR)"
          accentClass="ring-1 ring-info/15"
          spotlight="rgba(59, 130, 246, 0.14)"
          cards={lrCards}
        />
        <HeadColumn
          title="Random forest (RF)"
          accentClass="ring-1 ring-success/15"
          spotlight="rgba(34, 197, 94, 0.12)"
          cards={rfCards}
        />
      </div>
    </div>
  );
}
