"use client";

import { useState } from "react";
import { Dna, Loader2, Play } from "lucide-react";
import { api } from "@/lib/api";
import { parseSequencesInput } from "@/lib/parseSequences";
import type { PredictResponse } from "@/types";
import SpotlightCard from "@/components/ui/SpotlightCard";
import MetricHint from "@/components/ui/MetricHint";
import { METRIC_TIPS } from "@/lib/metricLabels";

const PLACEHOLDER = `>example_1
MKTAYIAKQRQISFVKSHFSRQ
>example_2
MKVLWAALLVTFLAGCQAKVE`;

export default function SequencePredictPanel() {
  const [text, setText] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);

  const run = async () => {
    const sequences = parseSequencesInput(text);
    if (sequences.length === 0) {
      setResult({ ok: false, error: "Paste at least one sequence (FASTA or raw amino acids)." });
      return;
    }
    setLoading(true);
    setResult(null);
    try {
      const data = await api.predictSequences(sequences);
      setResult(data);
    } catch (e) {
      setResult({ ok: false, error: (e as Error).message });
    } finally {
      setLoading(false);
    }
  };

  return (
    <section
      id="predict"
      className="scroll-mt-14 border-t border-border/60 bg-muted/15 dark:bg-muted/10"
    >
      <div className="mx-auto max-w-6xl px-4 py-14 sm:px-6">
        <h2 className="section-title mb-6">Test sequences on the trained model</h2>

        <SpotlightCard spotlightColor="rgba(59, 130, 246, 0.1)" className="p-5">
          <div className="flex items-center gap-2 mb-3">
            <Dna size={18} className="text-info" />
            <span className="text-sm font-medium">FASTA or one-letter sequence</span>
          </div>
          <textarea
            className="form-input font-mono text-xs min-h-[140px] mb-3"
            placeholder={PLACEHOLDER}
            value={text}
            onChange={(e) => setText(e.target.value)}
            spellCheck={false}
          />
          <button
            type="button"
            className="btn-primary"
            disabled={loading || !text.trim()}
            onClick={() => void run()}
          >
            {loading ? (
              <>
                <Loader2 size={16} className="animate-spin" /> Embedding & scoring (polling)…
              </>
            ) : (
              <>
                <Play size={16} /> Run prediction
              </>
            )}
          </button>

          {result && (
            <div className="mt-6 space-y-4 animate-fade-in">
              {!result.ok && (
                <div className="p-3 rounded-lg bg-destructive/5 border border-destructive/20 text-sm text-destructive">
                  {result.error ?? "Prediction failed"}
                </div>
              )}
              {result.ok && result.results && (
                <>
                  {result.results.some(
                    (r) => r.rf_predicted_positive !== r.lr_predicted_positive
                  ) && (
                    <p className="text-[11px] rounded-md border border-border bg-muted/30 px-3 py-2 text-muted-foreground leading-snug">
                      <span className="font-medium text-foreground">Heads disagree on at least one row.</span>{" "}
                      That is common: the random forest uses a high cutoff (τ_RF ≈ 0.88) chosen so test
                      precision stays 1.0 on this benchmark — which forces{" "}
                      <span className="text-foreground">low recall on true positives</span> (~53% of test
                      positives pass RF; LR is higher but still not 100%). Silver “expanded” rows are
                      especially easy to miss under RF. This is a trade-off of the saved thresholds, not a
                      sign the sequence is “wrong.”
                    </p>
                  )}
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    <MetricHint title={METRIC_TIPS.esm}>
                      <span className="font-medium text-foreground">ESM</span>
                    </MetricHint>
                    : <span className="font-mono">{result.esm_model}</span>
                    <span className="mx-1.5 text-border">·</span>
                    <MetricHint title={METRIC_TIPS.tauRf}>
                      <span className="font-medium text-foreground">RF cutoff</span> τ_RF
                    </MetricHint>
                    ={result.thresholds?.rf?.toFixed(4)}
                    <span className="mx-1.5 text-border">·</span>
                    <MetricHint title={METRIC_TIPS.tauLr}>
                      <span className="font-medium text-foreground">LR cutoff</span> τ_LR
                    </MetricHint>
                    ={result.thresholds?.lr?.toFixed(4)}
                  </p>
                  <div className="overflow-x-auto rounded-lg border border-border">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="border-b border-border bg-muted/40 text-left text-xs text-muted-foreground">
                          <th className="p-2 font-medium align-bottom" title="Sequence id from FASTA or auto-generated.">
                            ID
                          </th>
                          <th
                            className="p-2 font-medium align-bottom"
                            title="Number of amino acids after cleaning (valid letters only)."
                          >
                            Len
                          </th>
                          <th className="p-2 font-medium align-bottom">
                            <MetricHint title={METRIC_TIPS.rfProb}>RF score</MetricHint>
                            <span className="block text-[10px] font-normal text-muted-foreground/90 font-sans tracking-normal">
                              random forest · 0–1
                            </span>
                          </th>
                          <th className="p-2 font-medium align-bottom">
                            <MetricHint title={METRIC_TIPS.rfPlus}>RF class</MetricHint>
                            <span className="block text-[10px] font-normal text-muted-foreground/90 font-sans tracking-normal">
                              vs τ_RF
                            </span>
                          </th>
                          <th className="p-2 font-medium align-bottom">
                            <MetricHint title={METRIC_TIPS.lrProb}>LR score</MetricHint>
                            <span className="block text-[10px] font-normal text-muted-foreground/90 font-sans tracking-normal">
                              logistic regr. · 0–1
                            </span>
                          </th>
                          <th className="p-2 font-medium align-bottom">
                            <MetricHint title={METRIC_TIPS.lrPlus}>LR class</MetricHint>
                            <span className="block text-[10px] font-normal text-muted-foreground/90 font-sans tracking-normal">
                              vs τ_LR
                            </span>
                          </th>
                        </tr>
                      </thead>
                      <tbody>
                        {result.results.map((row) => (
                          <tr key={row.id} className="border-b border-border/80 last:border-0">
                            <td className="p-2 font-mono text-xs max-w-[140px] truncate" title={row.id}>
                              {row.id}
                            </td>
                            <td className="p-2 tabular-nums">{row.sequence_length}</td>
                            <td className="p-2 tabular-nums">{row.rf_probability.toFixed(4)}</td>
                            <td className="p-2">
                              <span
                                className={
                                  row.rf_predicted_positive
                                    ? "metric-pill metric-pill--pass"
                                    : "metric-pill metric-pill--fail"
                                }
                              >
                                {row.rf_predicted_positive ? "positive" : "negative"}
                              </span>
                            </td>
                            <td className="p-2 tabular-nums">{row.lr_probability.toFixed(4)}</td>
                            <td className="p-2">
                              <span
                                className={
                                  row.lr_predicted_positive
                                    ? "metric-pill metric-pill--pass"
                                    : "metric-pill metric-pill--fail"
                                }
                              >
                                {row.lr_predicted_positive ? "positive" : "negative"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  <p className="text-[11px] text-muted-foreground">
                    “Positive” = plastic-degrading class per your training labels; not a guarantee of
                    biochemical activity on polymers.
                  </p>
                </>
              )}
            </div>
          )}
        </SpotlightCard>
      </div>
    </section>
  );
}
