import fs from "fs";
import path from "path";

/** Repo root containing `plasticdeg/` and `data/processed_v2/`. */
export function getRepoRoot(): string {
  const env = process.env.PIPELINE_ARTIFACTS_ROOT;
  if (env) return path.resolve(env);
  const fromCwd = process.cwd();
  const parent = path.resolve(fromCwd, "..");
  if (fs.existsSync(path.join(parent, "plasticdeg", "paths.py"))) return parent;
  if (fs.existsSync(path.join(fromCwd, "plasticdeg", "paths.py"))) return fromCwd;
  return parent;
}

export const PROCESSED_V2 = "data/processed_v2";

export function artifactPath(...segments: string[]): string {
  return path.join(getRepoRoot(), PROCESSED_V2, ...segments);
}

export function readJsonFile<T>(absPath: string): T | null {
  try {
    if (!fs.existsSync(absPath)) return null;
    const raw = fs.readFileSync(absPath, "utf-8");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export function fileExists(absPath: string): boolean {
  try {
    return fs.statSync(absPath).isFile();
  } catch {
    return false;
  }
}

export function lineCount(absPath: string): number | null {
  try {
    if (!fs.existsSync(absPath)) return null;
    const buf = fs.readFileSync(absPath, "utf-8");
    return buf.split(/\r?\n/).filter((l) => l.trim().length > 0).length;
  } catch {
    return null;
  }
}

/** Allowed PNG basenames under `data/processed_v2/reports/`. */
export const CHART_FILES = {
  pr_v1: "pr_curve_rf_esm_baseline.png",
  pr_v2: "pr_curve_rf_esm_baseline_v2.png",
  roc_v1: "roc_curve_rf_esm_baseline.png",
  roc_v2: "roc_curve_rf_esm_baseline_v2.png",
} as const;

export type ChartKind = keyof typeof CHART_FILES;

export function chartAbsolutePath(kind: ChartKind): string {
  return artifactPath("reports", CHART_FILES[kind]);
}

/** Curated PNGs under `data/processed_v2/reports/figure_bundle/`. */
export const FIGURE_BUNDLE_FILES = {
  roc_test: "fig_roc_test.png",
  pr_test: "fig_pr_test.png",
  confusion_lr: "fig_confusion_matrix_lr.png",
  confusion_rf: "fig_confusion_matrix_rf.png",
} as const;

export type FigureBundleKind = keyof typeof FIGURE_BUNDLE_FILES;

export function figureBundleAbsolutePath(kind: FigureBundleKind): string {
  return artifactPath("reports", "figure_bundle", FIGURE_BUNDLE_FILES[kind]);
}
