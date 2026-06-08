"use client";

import { useCallback, useEffect, useState } from "react";
import {
  ArrowDown,
  Cpu,
  Layers,
  Loader2,
  Moon,
  RefreshCw,
  Sparkles,
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
import HeroParticles from "@/components/HeroParticles";
import SpotlightCard from "@/components/ui/SpotlightCard";

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

  const rfCm =
    data?.metrics_esm_baseline_v2?.models?.random_forest?.test_combined
      ?.confusion_matrix;

  const lrCm =
    data?.metrics_esm_baseline_v2?.models?.logistic_regression?.test_combined
      ?.confusion_matrix;

  const evalModels = data?.evaluation_report?.models as
    | {
        logistic_regression?: { confusion_matrix?: number[][] };
        random_forest?: { confusion_matrix?: number[][] };
      }
    | undefined;

  return (
    <main className="min-h-screen bg-background">
      <nav className="sticky top-0 z-50 border-b border-border/60 bg-[var(--nav-blur)] backdrop-blur-xl backdrop-saturate-150">
        <div className="mx-auto flex h-14 max-w-6xl items-center justify-between gap-3 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <span
              className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-gradient-to-br from-emerald-500/35 via-teal-500/25 to-sky-500/30 text-white shadow-lg shadow-emerald-900/20 ring-1 ring-white/20 dark:from-emerald-500/25 dark:via-teal-500/15 dark:to-sky-500/25 dark:text-emerald-50 dark:shadow-emerald-950/40 dark:ring-white/10"
              aria-hidden
            >
              <Sparkles size={18} strokeWidth={2} />
            </span>
            <div className="min-w-0 leading-tight">
              <span className="block truncate text-sm font-semibold tracking-tight text-foreground">
                Bioplastic AI
              </span>
              <span className="block truncate text-[10px] font-medium uppercase tracking-[0.14em] text-muted-foreground">
                PAZy · screening
              </span>
            </div>
          </div>
          <div className="flex shrink-0 items-center gap-1.5 sm:gap-2">
            <button
              type="button"
              onClick={() => void load()}
              className="btn-secondary px-3 py-2 text-xs"
              disabled={loading}
            >
              {loading ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <RefreshCw size={14} />
              )}
              Refresh
            </button>
            <div className="hidden items-center rounded-full border border-border/70 bg-muted/50 p-0.5 sm:flex dark:bg-muted/30">
              <a
                href="#predict"
                className="rounded-full px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
              >
                Predict
              </a>
              <a
                href="#dashboard"
                className="rounded-full px-3 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:bg-card hover:text-foreground"
              >
                Dashboard
              </a>
            </div>
            <a
              href="#predict"
              className="rounded-full px-2.5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground sm:hidden"
            >
              Predict
            </a>
            <a
              href="#dashboard"
              className="rounded-full px-2.5 py-1.5 text-xs font-medium text-muted-foreground transition-colors hover:text-foreground sm:hidden"
            >
              Dash
            </a>
            <button
              type="button"
              onClick={toggleTheme}
              className="flex size-9 items-center justify-center rounded-full border border-border/80 bg-card text-foreground transition-colors hover:border-border hover:bg-accent"
              aria-label="Toggle theme"
            >
              {theme === "light" ? <Moon size={16} /> : <Sun size={16} />}
            </button>
          </div>
        </div>
      </nav>

      <section className="relative flex min-h-[calc(100dvh-3.5rem)] flex-col items-center justify-center overflow-hidden px-4 py-16 sm:px-6">
        <HeroBackdrop />
        <div
          className="pointer-events-none absolute -left-24 top-[18%] size-64 rounded-full bg-emerald-500/25 blur-[88px] dark:bg-emerald-500/12 sm:-left-28 sm:size-80"
          aria-hidden
        />
        <div
          className="pointer-events-none absolute -right-16 bottom-[14%] size-72 rounded-full bg-sky-500/22 blur-[96px] dark:bg-sky-500/12 sm:-right-24 sm:size-96"
          aria-hidden
        />
        <div className="pointer-events-none absolute left-1/2 top-[8%] size-40 -translate-x-1/2 rounded-full bg-violet-500/15 blur-[72px] dark:bg-violet-500/10" aria-hidden />
        <div className="hero-grid-overlay" aria-hidden />
        <HeroParticles />

        <div className="relative z-10 mx-auto w-full max-w-lg sm:max-w-2xl px-1 sm:px-2">
          <div className="hero-landing-card px-6 py-9 sm:px-10 sm:py-11">
            <div className="hero-landing-card__inner flex flex-col items-center text-center">
              <div className="mb-5 flex flex-wrap items-center justify-center gap-2">
                <span className="inline-flex items-center gap-1.5 rounded-full border border-border/80 bg-muted/50 px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground backdrop-blur-sm dark:border-white/10 dark:bg-white/[0.05]">
                  <Layers size={12} className="text-success" aria-hidden />
                  ESM-2
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-full border border-border/80 bg-muted/50 px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-muted-foreground backdrop-blur-sm dark:border-white/10 dark:bg-white/[0.05]">
                  <Cpu size={12} className="text-info" aria-hidden />
                  RF / LR
                </span>
                <span className="inline-flex items-center gap-1.5 rounded-full border border-emerald-500/25 bg-emerald-500/[0.08] px-3 py-1 text-[11px] font-medium uppercase tracking-wider text-emerald-800 dark:border-emerald-400/20 dark:bg-emerald-500/10 dark:text-emerald-200/90">
                  Homology splits
                </span>
              </div>

              <div className="relative">
                <div
                  className="pointer-events-none absolute -inset-x-6 -top-3 h-px bg-gradient-to-r from-transparent via-white/30 to-transparent dark:via-white/15"
                  aria-hidden
                />
                <h1 className="text-balance bg-gradient-to-br from-foreground via-foreground to-muted-foreground bg-clip-text text-3xl font-bold leading-[1.1] tracking-tight text-transparent sm:text-5xl sm:leading-[1.06] sm:tracking-tighter">
                  Polyester enzyme discovery
                </h1>
                <p className="mt-2 text-lg font-medium text-muted-foreground sm:text-xl">
                  Live screening console
                </p>
              </div>

              <div className="mt-8 flex w-full max-w-md flex-col items-stretch gap-3 border-t border-border/50 pt-8 dark:border-white/[0.07] sm:flex-row sm:items-center sm:justify-center">
                <a href="#predict" className="btn-secondary min-h-[44px] min-w-0 flex-1 justify-center sm:min-w-[9.5rem] sm:flex-initial">
                  Test sequences
                </a>
                <a href="#dashboard" className="btn-primary min-h-[44px] min-w-0 flex-1 justify-center sm:min-w-[9.5rem] sm:flex-initial">
                  View metrics <ArrowDown size={14} />
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      <SequencePredictPanel />

      <section id="dashboard" className="scroll-mt-14 border-t border-border/60 bg-muted/20 dark:bg-muted/10">
        <div className="mx-auto max-w-6xl space-y-12 px-4 py-14 sm:px-6">
          {error && (
            <div className="rounded-2xl border border-destructive/25 bg-destructive/5 p-5 text-sm text-destructive shadow-sm">
              {error}
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
                <h2 className="section-title mb-6">Model metrics (v2)</h2>
                <PipelineOverviewCards metricsV2={data.metrics_esm_baseline_v2} />
              </div>

              <ModelHeadComparisonChart metricsV2={data.metrics_esm_baseline_v2} />

              <div>
                <h3 className="mb-4 text-sm font-semibold tracking-tight text-foreground">
                  Confusion matrices
                </h3>
                <div className="grid grid-cols-1 gap-5 sm:grid-cols-2">
                  <ConfusionMatrixCard
                    matrix={lrCm}
                    title="LR — metrics v2"
                    spotlightColor="rgba(59, 130, 246, 0.12)"
                  />
                  <ConfusionMatrixCard
                    matrix={rfCm}
                    title="RF — metrics v2"
                    spotlightColor="rgba(34, 197, 94, 0.12)"
                  />
                  <ConfusionMatrixCard
                    matrix={evalModels?.logistic_regression?.confusion_matrix}
                    title="LR — evaluation report"
                    spotlightColor="rgba(59, 130, 246, 0.12)"
                  />
                  <ConfusionMatrixCard
                    matrix={evalModels?.random_forest?.confusion_matrix}
                    title="RF — evaluation report"
                    spotlightColor="rgba(34, 197, 94, 0.12)"
                  />
                </div>
              </div>

              <ProbeStressSection
                probe={data.hard_negative_probe_v2}
                tiers={data.tier_probe_summary}
              />

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
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

              <CurveGallery figureBundle={data.artifacts.figure_bundle} />
            </>
          )}
        </div>
      </section>

      <footer className="border-t border-border/60 bg-muted/15 py-10 text-center">
        <p className="mx-auto max-w-md text-[11px] leading-relaxed tracking-wide text-muted-foreground">
          Bioplastic AI
        </p>
      </footer>
    </main>
  );
}
