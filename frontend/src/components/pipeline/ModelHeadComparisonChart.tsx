"use client";

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import SpotlightCard from "@/components/ui/SpotlightCard";
import MetricHint from "@/components/ui/MetricHint";
import { METRIC_TIPS } from "@/lib/metricLabels";
import type { MetricsEsmBaselineJson } from "@/types";

interface Props {
  metricsV2: MetricsEsmBaselineJson | null;
}

export default function ModelHeadComparisonChart({ metricsV2 }: Props) {
  const lr = metricsV2?.models?.logistic_regression;
  const rf = metricsV2?.models?.random_forest;
  const lrB = lr?.test_combined;
  const rfB = rf?.test_combined;

  const data = [
    {
      name: "PR-AUC",
      LR: lrB?.pr_auc ?? lr?.pr_auc ?? null,
      RF: rfB?.pr_auc ?? rf?.pr_auc ?? null,
    },
    {
      name: "ROC-AUC",
      LR: lrB?.roc_auc ?? lr?.roc_auc ?? null,
      RF: rfB?.roc_auc ?? rf?.roc_auc ?? null,
    },
  ].filter((row) => row.LR != null || row.RF != null);

  if (data.length === 0) {
    return (
      <SpotlightCard className="p-6 text-sm text-muted-foreground">
        No <code className="text-xs">metrics_esm_baseline_v2.json</code> loaded — run{" "}
        <code className="text-xs">plasticdeg.train.train_esm_baseline_v2</code> first.
      </SpotlightCard>
    );
  }

  return (
    <SpotlightCard spotlightColor="rgba(59, 130, 246, 0.12)" className="p-4">
      <h3 className="section-title mb-2">
        <MetricHint title={METRIC_TIPS.testCombined}>Head comparison (test_combined)</MetricHint>
      </h3>
      <p className="text-xs text-muted-foreground mb-3 leading-relaxed">
        <strong className="text-foreground font-medium">PR-AUC</strong> — precision–recall curve area (ranking
        positives). <strong className="text-foreground font-medium">ROC-AUC</strong> — overall separation of
        classes. Both are 0–1. <MetricHint title={METRIC_TIPS.mcc}>MCC</MetricHint> below uses a different scale.
      </p>
      <div className="h-[260px] w-full min-w-0 shrink-0">
        <ResponsiveContainer width="100%" height={260} minWidth={0}>
          <BarChart data={data} margin={{ top: 8, right: 8, left: -8, bottom: 0 }}>
            <CartesianGrid strokeDasharray="3 3" vertical={false} />
            <XAxis dataKey="name" tickLine={false} axisLine={false} />
            <YAxis domain={[0, 1.05]} tickLine={false} axisLine={false} width={36} />
            <Tooltip
              contentStyle={{
                background: "var(--color-card)",
                border: "1px solid var(--color-border)",
                borderRadius: 8,
              }}
            />
            <Legend />
            <Bar dataKey="LR" name="Logistic regression" fill="var(--color-info)" radius={[4, 4, 0, 0]} />
            <Bar dataKey="RF" name="Random forest" fill="var(--color-success)" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
      <div className="mt-4 grid grid-cols-2 gap-3 text-xs border-t border-border pt-3">
        <div>
          <p className="text-muted-foreground mb-1">
            <MetricHint title={METRIC_TIPS.lr}>Logistic regression</MetricHint> ·{" "}
            <MetricHint title={METRIC_TIPS.mcc}>MCC</MetricHint>
            <span className="text-muted-foreground/80"> @ training cutoff</span>
          </p>
          <p className="font-mono tabular-nums font-medium">
            {lrB?.mcc_at_threshold != null ? lrB.mcc_at_threshold.toFixed(4) : "—"}
          </p>
        </div>
        <div>
          <p className="text-muted-foreground mb-1">
            <MetricHint title={METRIC_TIPS.rf}>Random forest</MetricHint> ·{" "}
            <MetricHint title={METRIC_TIPS.mcc}>MCC</MetricHint>
            <span className="text-muted-foreground/80"> @ training cutoff</span>
          </p>
          <p className="font-mono tabular-nums font-medium">
            {rfB?.mcc_at_threshold != null ? rfB.mcc_at_threshold.toFixed(4) : "—"}
          </p>
        </div>
      </div>
    </SpotlightCard>
  );
}
