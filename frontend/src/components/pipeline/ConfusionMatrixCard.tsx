"use client";

import SpotlightCard from "@/components/ui/SpotlightCard";
import { METRIC_TIPS } from "@/lib/metricLabels";

interface Props {
  matrix: number[][] | undefined;
  title: string;
  /** Optional tint for LR (info) vs RF (success) storytelling */
  spotlightColor?: string;
}

/** Expects sklearn-style [[TN, FP], [FN, TP]] */
export default function ConfusionMatrixCard({
  matrix,
  title,
  spotlightColor = "rgba(168, 85, 247, 0.1)",
}: Props) {
  if (!matrix || matrix.length !== 2 || matrix[0]?.length !== 2) {
    return (
      <SpotlightCard className="p-6 text-sm text-muted-foreground">
        No confusion matrix for this view.
      </SpotlightCard>
    );
  }
  const [[tn, fp], [fn, tp]] = matrix;
  const cells = [
    { v: tn, lab: "TN", tip: METRIC_TIPS.tn },
    { v: fp, lab: "FP", tip: METRIC_TIPS.fp },
    { v: fn, lab: "FN", tip: METRIC_TIPS.fn },
    { v: tp, lab: "TP", tip: METRIC_TIPS.tp },
  ];
  const max = Math.max(tn, fp, fn, tp, 1);

  return (
    <SpotlightCard spotlightColor={spotlightColor} className="p-4">
      <h3 className="section-title mb-3">{title}</h3>
      <div className="grid grid-cols-2 gap-2 max-w-[220px]">
        {cells.map((c) => (
          <div
            key={c.lab}
            className="rounded-lg border border-border p-3 text-center"
            style={{
              background: `color-mix(in srgb, var(--color-accent) ${Math.round((c.v / max) * 55)}%, transparent)`,
            }}
          >
            <p className="text-[10px] uppercase tracking-wider text-muted-foreground cursor-help" title={c.tip}>
              {c.lab}
            </p>
            <p className="text-xl font-semibold tabular-nums">{c.v}</p>
          </div>
        ))}
      </div>
    </SpotlightCard>
  );
}
