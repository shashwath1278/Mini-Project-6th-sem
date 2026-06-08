"use client";

import SpotlightCard from "@/components/ui/SpotlightCard";
import type { PipelineArtifactsManifest } from "@/types";

const ITEMS: {
  kind: keyof PipelineArtifactsManifest["figure_bundle"];
  label: string;
}[] = [
  { kind: "roc_test", label: "ROC — test (LR & RF)" },
  { kind: "pr_test", label: "PR — test (LR & RF)" },
  { kind: "confusion_lr", label: "Confusion — LR" },
  { kind: "confusion_rf", label: "Confusion — RF" },
];

interface Props {
  figureBundle: PipelineArtifactsManifest["figure_bundle"];
}

export default function CurveGallery({ figureBundle }: Props) {
  return (
    <SpotlightCard className="p-4">
      <h3 className="section-title mb-4">Key figures</h3>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {ITEMS.map(({ kind, label }) => {
          const info = figureBundle[kind];
          if (!info?.exists) {
            return (
              <div
                key={kind}
                className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground"
              >
                {label}
                <br />
                <span className="text-xs">reports/figure_bundle</span>
              </div>
            );
          }
          const src = `/api/pipeline/figure?kind=${kind}`;
          return (
            <div key={kind} className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground">{label}</p>
              <div className="relative flex aspect-[4/3] w-full items-center justify-center overflow-hidden rounded-lg border border-border bg-muted/20 p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={src}
                  alt={label}
                  className="max-h-full max-w-full object-contain"
                  loading="lazy"
                  decoding="async"
                />
              </div>
            </div>
          );
        })}
      </div>
    </SpotlightCard>
  );
}
