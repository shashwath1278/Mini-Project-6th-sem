"use client";

import { ShieldAlert, TrendingDown } from "lucide-react";
import SpotlightCard from "@/components/ui/SpotlightCard";
import { METRIC_TIPS } from "@/lib/metricLabels";
import type { HardNegativeProbePayload, TierProbeSummary } from "@/types";

interface Props {
  probe: HardNegativeProbePayload | null;
  tiers: TierProbeSummary | null;
}

export default function ProbeStressSection({ probe, tiers }: Props) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
      <SpotlightCard spotlightColor="rgba(239, 68, 68, 0.08)" className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <ShieldAlert size={18} className="text-warning" />
          <h3 className="section-title !mb-0">
            <span title={`${METRIC_TIPS.rf} ${METRIC_TIPS.tauRf}`} className="cursor-help border-b border-dotted border-muted-foreground/40">
              Hard-negative probe (RF)
            </span>
          </h3>
        </div>
        {!probe ? (
          <p className="text-sm text-muted-foreground">No hard-negative probe data.</p>
        ) : (
          <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
            <dt className="text-muted-foreground cursor-help" title={METRIC_TIPS.baseline}>
              Baseline
            </dt>
            <dd className="font-medium">{probe.baseline ?? "—"}</dd>
            <dt className="text-muted-foreground cursor-help" title={METRIC_TIPS.hardN}>
              Hard N
            </dt>
            <dd>{probe.hard_negative_count ?? "—"}</dd>
            <dt className="text-muted-foreground cursor-help" title={METRIC_TIPS.predPosHard}>
              Pred + on hard
            </dt>
            <dd>{probe.hard_negative_predicted_positive_count ?? "—"}</dd>
            <dt className="text-muted-foreground cursor-help" title={METRIC_TIPS.fprHard}>
              FPR on hard
            </dt>
            <dd className="tabular-nums">
              {probe.hard_negative_false_positive_rate != null
                ? probe.hard_negative_false_positive_rate.toFixed(4)
                : "—"}
            </dd>
            <dt className="text-muted-foreground cursor-help" title="Mean random-forest probability on hard-negative sequences (not necessarily above cutoff).">
              Mean score (hard)
            </dt>
            <dd className="tabular-nums">
              {probe.hard_negative_score_mean != null
                ? probe.hard_negative_score_mean.toFixed(4)
                : "—"}
            </dd>
            <dt className="text-muted-foreground cursor-help" title={METRIC_TIPS.easyNegN}>
              Easy neg N (test)
            </dt>
            <dd>{probe.easy_test_negative_count ?? "—"}</dd>
          </dl>
        )}
      </SpotlightCard>

      <SpotlightCard spotlightColor="rgba(234, 179, 8, 0.1)" className="p-4">
        <div className="flex items-center gap-2 mb-3">
          <TrendingDown size={18} className="text-muted-foreground" />
          <h3 className="section-title !mb-0">
            <span title={METRIC_TIPS.tierProbe} className="cursor-help border-b border-dotted border-muted-foreground/40">
              Tier probe summary
            </span>
          </h3>
        </div>
        {!tiers?.tiers?.length ? (
          <p className="text-sm text-muted-foreground">No tier probe summary.</p>
        ) : (
          <ul className="space-y-3">
            {tiers.tiers.map((t) => (
              <li
                key={t.tier_label}
                className="rounded-lg border border-border p-3 text-sm"
              >
                <p className="font-medium text-foreground">{t.tier_label.replace(/_/g, " ")}</p>
                <p className="text-xs text-muted-foreground mt-1">
                  <span title="Number of sequences in this tier’s probe set">n</span>={t.n_probe_sequences} ·{" "}
                  <span title={METRIC_TIPS.fprHard}>FPR</span>={t.false_positive_rate?.toFixed(4) ?? "—"} ·{" "}
                  <span title="Mean model score on that tier’s probe sequences">mean score</span>=
                  {t.score_mean?.toFixed(4) ?? "—"}
                </p>
              </li>
            ))}
          </ul>
        )}
        {tiers?.note && (
          <p className="text-[11px] text-muted-foreground mt-3 border-t border-border pt-2">
            {tiers.note}
          </p>
        )}
      </SpotlightCard>
    </div>
  );
}
