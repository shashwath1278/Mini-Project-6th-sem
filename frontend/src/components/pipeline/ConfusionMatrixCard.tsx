"use client";

import SpotlightCard from "@/components/ui/SpotlightCard";
import { METRIC_TIPS } from "@/lib/metricLabels";

interface Props {
  matrix: number[][] | undefined;
  title: string;
}

/** Expects sklearn-style [[TN, FP], [FN, TP]] */
export default function ConfusionMatrixCard({ matrix, title }: Props) {
  if (!matrix || matrix.length !== 2 || matrix[0]?.length !== 2) {
    return (
      <SpotlightCard className="p-6 text-sm text-muted-foreground">
        No confusion matrix in metrics JSON for {title}.
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
    <SpotlightCard spotlightColor="rgba(168, 85, 247, 0.1)" className="p-4">
      <h3 className="section-title mb-3">{title}</h3>
      <p className="text-xs text-muted-foreground mb-3 leading-relaxed">
        Rows: <span title="Label from training data (0 = not in positive set, 1 = polyester-class positive)">true class</span>{" "}
        (negative, positive). Columns: <span title="Model prediction at frozen cutoff">predicted class</span>{" "}
        (negative, positive). Cell codes: TN / FP / FN / TP — hover each cell label for full words.
      </p>
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
