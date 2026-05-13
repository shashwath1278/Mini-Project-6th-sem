"use client";

import SpotlightCard from "@/components/ui/SpotlightCard";
import type { PipelineArtifactsManifest } from "@/types";

const ITEMS: { kind: keyof PipelineArtifactsManifest["charts"]; label: string }[] = [
  { kind: "pr_v2", label: "PR curve (RF, v2)" },
  { kind: "roc_v2", label: "ROC curve (RF, v2)" },
  { kind: "pr_v1", label: "PR curve (RF, v1)" },
  { kind: "roc_v1", label: "ROC curve (RF, v1)" },
];

interface Props {
  charts: PipelineArtifactsManifest["charts"];
}

export default function CurveGallery({ charts }: Props) {
  return (
    <SpotlightCard className="p-4">
      <h3 className="section-title mb-4">Curves (from backend reports)</h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {ITEMS.map(({ kind, label }) => {
          const info = charts[kind];
          if (!info?.exists) {
            return (
              <div
                key={kind}
                className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground"
              >
                {label}
                <br />
                <span className="text-xs">File missing — run training / reporting.</span>
              </div>
            );
          }
          const src = `/api/pipeline/chart?kind=${kind}`;
          return (
            <div key={kind} className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground">{label}</p>
              <div className="relative w-full aspect-[4/3] rounded-lg border border-border overflow-hidden bg-muted/20 flex items-center justify-center p-2">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={src}
                  alt={label}
                  className="max-w-full max-h-full object-contain"
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
