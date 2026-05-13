import type { PipelineDashboardResponse, PredictResponse, PredictStatusResponse } from "@/types";

async function jsonFetch<T>(path: string): Promise<T> {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }));
    throw new Error(
      typeof err === "object" && err && "error" in err
        ? String((err as { error: string }).error)
        : `Request failed ${res.status}`
    );
  }
  return res.json() as Promise<T>;
}

export const api = {
  pipelineDashboard: () =>
    jsonFetch<PipelineDashboardResponse>("/api/pipeline/dashboard"),

  /**
   * Starts a background Python job (202), then polls until scores are ready.
   * Avoids Next.js ~10 minute limit on a single blocking API call.
   */
  predictSequences: async (sequences: { id?: string; sequence: string }[]): Promise<PredictResponse> => {
    const warmBase = (process.env.NEXT_PUBLIC_PREDICT_SERVICE_URL ?? "").trim().replace(/\/$/, "");
    if (warmBase) {
      const res = await fetch(`${warmBase}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sequences }),
      });
      const raw = (await res.json()) as Record<string, unknown>;
      if (!res.ok) {
        const detail =
          typeof raw.detail === "string"
            ? raw.detail
            : typeof raw.error === "string"
              ? raw.error
              : res.statusText;
        throw new Error(detail);
      }
      return raw as PredictResponse;
    }

    const res = await fetch("/api/pipeline/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ sequences }),
    });
    const raw = (await res.json()) as Record<string, unknown>;

    if (res.status === 400) {
      throw new Error(typeof raw.error === "string" ? raw.error : "Bad request");
    }
    if (res.status !== 202 || typeof raw.job_id !== "string") {
      throw new Error(typeof raw.error === "string" ? raw.error : `Unexpected response ${res.status}`);
    }

    const jobId = raw.job_id;
    const maxPolls = 7200;

    for (let i = 0; i < maxPolls; i++) {
      if (i > 0) {
        await new Promise((r) => setTimeout(r, 2000));
      }
      const pr = await fetch(`/api/pipeline/predict/status?job=${encodeURIComponent(jobId)}`, {
        cache: "no-store",
      });
      if (!pr.ok) {
        throw new Error(`Status poll failed (HTTP ${pr.status})`);
      }
      const body = (await pr.json()) as PredictStatusResponse;

      if (body.status === "done") {
        return body.result;
      }
      if (body.status === "error") {
        return { ok: false, error: body.error };
      }
    }

    return {
      ok: false,
      error:
        "Stopped waiting after ~4 hours. For very slow CPU runs, use: python -m plasticdeg.eval.predict_sequences --in in.json --out out.json",
    };
  },
};
