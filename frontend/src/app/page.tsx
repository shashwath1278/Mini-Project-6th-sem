"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ArrowDown,
  Dna,
  Loader2,
  Moon,
  RefreshCw,
  Sun,
} from "lucide-react";
import { useTheme } from "@/components/ThemeProvider";
import { api } from "@/lib/api";
import type { PipelineDashboardResponse } from "@/types";
import PipelineOverviewCards from "@/components/pipeline/PipelineOverviewCards";
import ModelHeadComparisonChart from "@/components/pipeline/ModelHeadComparisonChart";
import ConfusionMatrixCard from "@/components/pipeline/ConfusionMatrixCard";
import ProbeStressSection from "@/components/pipeline/ProbeStressSection";
import PipelineStageRail from "@/components/pipeline/PipelineStageRail";
import CurveGallery from "@/components/pipeline/CurveGallery";
import SequencePredictPanel from "@/components/pipeline/SequencePredictPanel";
import HeroBackdrop from "@/components/HeroBackdrop";
import SpotlightCard from "@/components/ui/SpotlightCard";
import { METRIC_TIPS } from "@/lib/metricLabels";

export default function Home() {
  const { theme, toggleTheme } = useTheme();
  const [data, setData] = useState<PipelineDashboardResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const d = await api.pipelineDashboard();
      setData(d);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const recallTarget =
    (data?.metrics_esm_baseline_v2?.recall_target_for_threshold as number | undefined) ??
    (data?.metrics_esm_baseline?.recall_target_for_threshold as number | undefined) ??
    0.8;

  const rfCm =
    data?.metrics_esm_baseline_v2?.models?.random_forest?.test_combined
      ?.confusion_matrix;

  const evalRf = data?.evaluation_report?.models as
    | { random_forest?: { confusion_matrix?: number[][] } }
    | undefined;

  return (
    <main className="min-h-screen">
      <nav className="sticky top-0 z-50 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="max-w-6xl mx-auto flex items-center justify-between px-6 h-14">
          <span className="font-semibold text-sm tracking-tight flex items-center gap-2">
            <Dna size={16} className="text-success" />
            PlasticDeg · PAZy pipeline
          </span>
          <div className="flex items-center gap-3">
            <button
              type="button"
              onClick={() => void load()}
              className="btn-secondary text-xs"
              disabled={loading}
            >
              {loading ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <RefreshCw size={14} />
              )}
              Refresh
            </button>
            <a href="#predict" className="text-sm text-muted-foreground hover:text-foreground">
              Predict
            </a>
            <a href="#dashboard" className="text-sm text-muted-foreground hover:text-foreground">
              Dashboard
            </a>
            <button
              type="button"
              onClick={toggleTheme}
              className="p-2 rounded-lg border border-border hover:bg-accent transition-colors"
              aria-label="Toggle theme"
            >
              {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
            </button>
          </div>
        </div>
      </nav>

      <section className="relative flex flex-col items-center justify-center py-20 sm:py-28 overflow-hidden">
        <HeroBackdrop />
        <div className="relative z-10 text-center px-6 max-w-2xl">
          <a
            href="#dashboard"
            className="inline-flex items-center gap-1.5 rounded-full border border-border px-4 py-1.5 text-xs text-muted-foreground hover:text-foreground transition-colors mb-6"
          >
            Live artifacts · data/processed_v2
          </a>
          <h1 className="text-4xl sm:text-5xl font-bold tracking-tight text-foreground mb-4 leading-[1.1]">
            Polyester enzyme
            <br />
            discovery dashboard
          </h1>
          <p className="text-base text-muted-foreground max-w-lg mx-auto leading-relaxed mb-8">
            Visualizes the same outputs as <code className="text-xs">plasticdeg</code>: ESM-2
            embeddings, homology splits, RF/LR heads, frozen-threshold probes, and PR/ROC
            figures under <code className="text-xs">data/processed_v2</code>.
          </p>
          <div className="flex items-center justify-center gap-3 flex-wrap">
            <a href="#predict" className="btn-secondary">
              Test sequences
            </a>
            <a href="#dashboard" className="btn-primary">
              View metrics <ArrowDown size={14} />
            </a>
          </div>
        </div>
      </section>

      <SequencePredictPanel />

      <section id="dashboard" className="border-t border-border scroll-mt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 py-12 space-y-10">
          {error && (
            <div className="p-4 rounded-lg bg-destructive/5 border border-destructive/20 text-sm text-destructive">
              {error}
              <p className="mt-2 text-xs text-muted-foreground">
                Run the dev server from <code className="text-[10px]">frontend/</code> so the API
                resolves the repo root, or set <code className="text-[10px]">PIPELINE_ARTIFACTS_ROOT</code>{" "}
                to your project directory.
              </p>
            </div>
          )}

          {loading && !data && (
            <div className="flex justify-center py-16">
              <div className="loading-ring" />
            </div>
          )}

          {data && (
            <>
              <div>
                <h2 className="section-title mb-2">Model metrics (v2)</h2>
                <p className="text-xs text-muted-foreground max-w-3xl leading-relaxed mb-3">
                  <span className="text-foreground font-medium">Quick glossary</span> — hover dotted terms
                  elsewhere, or read here:{" "}
                  <span title={METRIC_TIPS.esm} className="cursor-help border-b border-dotted border-muted-foreground/50">
                    ESM-2
                  </span>{" "}
                  embeds the sequence;{" "}
                  <span title={METRIC_TIPS.rf} className="cursor-help border-b border-dotted border-muted-foreground/50">
                    RF
                  </span>{" "}
                  and{" "}
                  <span title={METRIC_TIPS.lr} className="cursor-help border-b border-dotted border-muted-foreground/50">
                    LR
                  </span>{" "}
                  are two classifiers on that vector;{" "}
                  <span title={METRIC_TIPS.prAuc} className="cursor-help border-b border-dotted border-muted-foreground/50">
                    PR-AUC
                  </span>
                  ,{" "}
                  <span title={METRIC_TIPS.rocAuc} className="cursor-help border-b border-dotted border-muted-foreground/50">
                    ROC-AUC
                  </span>{" "}
                  summarize test ranking (0–1);{" "}
                  <span title={METRIC_TIPS.mcc} className="cursor-help border-b border-dotted border-muted-foreground/50">
                    MCC
                  </span>{" "}
                  summarizes errors at the fixed cutoff (different scale than AUC).
                </p>
                <p className="text-xs text-muted-foreground mb-4 font-mono truncate" title={data.repoRoot}>
                  Repo: {data.repoRoot}
                </p>
                <PipelineOverviewCards
                  metricsV2={data.metrics_esm_baseline_v2}
                  recallTarget={recallTarget}
                />
              </div>

              <ModelHeadComparisonChart metricsV2={data.metrics_esm_baseline_v2} />

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
                <ConfusionMatrixCard
                  matrix={rfCm}
                  title="RF confusion (test_combined, v2)"
                />
                <ConfusionMatrixCard
                  matrix={evalRf?.random_forest?.confusion_matrix}
                  title="RF confusion (evaluation_report.json)"
                />
              </div>

              <ProbeStressSection
                probe={data.hard_negative_probe_v2}
                tiers={data.tier_probe_summary}
              />

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                <div className="lg:col-span-5">
                  <PipelineStageRail manifest={data.artifacts} />
                </div>
                <div className="lg:col-span-7 space-y-4">
                  <SpotlightCard className="p-4">
                    <h3 className="section-title mb-2">Split counts</h3>
                    <dl className="grid grid-cols-2 gap-2 text-sm">
                      <dt className="text-muted-foreground">Train IDs</dt>
                      <dd className="font-mono tabular-nums">
                        {data.artifacts.splits.train.lines ?? "—"}
                      </dd>
                      <dt className="text-muted-foreground">Test IDs</dt>
                      <dd className="font-mono tabular-nums">
                        {data.artifacts.splits.test.lines ?? "—"}
                      </dd>
                    </dl>
                  </SpotlightCard>
                  {data.evaluation_report && (
                    <SpotlightCard className="p-4 overflow-x-auto">
                      <details className="group">
                        <summary className="section-title cursor-pointer list-none [&::-webkit-details-marker]:hidden">
                          evaluation_report.json{" "}
                          <span className="text-xs font-normal text-muted-foreground">
                            (click to expand — large JSON)
                          </span>
                        </summary>
                        <pre className="mt-3 text-[10px] leading-relaxed text-muted-foreground whitespace-pre-wrap break-all max-h-64 overflow-y-auto">
                          {JSON.stringify(data.evaluation_report, null, 2)}
                        </pre>
                      </details>
                    </SpotlightCard>
                  )}
                </div>
              </div>

              <CurveGallery charts={data.artifacts.charts} />
            </>
          )}
        </div>
      </section>

      <footer className="border-t border-border py-8 text-center text-xs text-muted-foreground">
        PlasticDeg · Mini project — frontend reads pipeline artifacts only (no mediation API).
      </footer>
    </main>
  );
}
