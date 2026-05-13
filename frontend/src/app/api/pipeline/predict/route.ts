import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { NextResponse } from "next/server";
import { getRepoRoot } from "@/lib/artifacts";

export const runtime = "nodejs";
/** POST returns immediately; long work runs in a detached child from Next’s request timeout. */
export const maxDuration = 60;

const MAX_SEQS = 48;
const MAX_LEN = 1022;

type SeqItem = { id?: string; sequence: string };

function normalizeBody(body: unknown): { sequences: SeqItem[] } | { error: string } {
  if (!body || typeof body !== "object") return { error: "JSON body required" };
  const o = body as Record<string, unknown>;
  const sequences = o.sequences;
  if (!Array.isArray(sequences) || sequences.length === 0) {
    return { error: 'Provide a non-empty "sequences" array' };
  }
  if (sequences.length > MAX_SEQS) {
    return { error: `At most ${MAX_SEQS} sequences per request` };
  }
  const out: SeqItem[] = [];
  for (let i = 0; i < sequences.length; i++) {
    const row = sequences[i];
    if (!row || typeof row !== "object") return { error: `sequences[${i}] must be an object` };
    const seq = (row as SeqItem).sequence;
    if (typeof seq !== "string" || !seq.trim()) {
      return { error: `sequences[${i}].sequence must be a non-empty string` };
    }
    if (seq.length > MAX_LEN + 200) {
      return { error: `sequences[${i}] is too long (>${MAX_LEN} AA after cleanup)` };
    }
    const id = (row as SeqItem).id;
    if (id !== undefined && typeof id !== "string") return { error: `sequences[${i}].id must be a string` };
    out.push({ id: id?.trim() || undefined, sequence: seq });
  }
  return { sequences: out };
}

export async function POST(req: Request) {
  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ ok: false, error: "Invalid JSON" }, { status: 400 });
  }

  const norm = normalizeBody(body);
  if ("error" in norm) {
    return NextResponse.json({ ok: false, error: norm.error }, { status: 400 });
  }

  const repo = getRepoRoot();
  const py =
    process.env.PYTHON_BIN ||
    process.env.PYTHON_EXECUTABLE ||
    (process.platform === "win32" ? "python" : "python3");

  const jobId = randomBytes(16).toString("hex");
  const jobDir = join(tmpdir(), "plasticdeg-predict", jobId);
  mkdirSync(jobDir, { recursive: true });
  const inPath = join(jobDir, "in.json");
  const outPath = join(jobDir, "out.json");
  const donePath = join(jobDir, "done.json");

  writeFileSync(inPath, JSON.stringify({ sequences: norm.sequences }), "utf-8");

  if (process.env.NODE_ENV === "development") {
    console.info(
      `[predict] job ${jobId}: spawning ${py} -u -m plasticdeg.eval.predict_sequences (cwd=${repo}). Logs stream below.`
    );
  }

  const child = spawn(
    py,
    ["-u", "-m", "plasticdeg.eval.predict_sequences", "--in", inPath, "--out", outPath],
    {
      cwd: repo,
      stdio: ["ignore", "inherit", "inherit"],
      env: {
        ...process.env,
        PYTHONUTF8: "1",
        PYTHONIOENCODING: "utf-8",
        PYTHONUNBUFFERED: "1",
      },
    }
  );

  let finished = false;
  const finishOnce = (exitCode: number | null) => {
    if (finished) return;
    finished = true;
    try {
      if (!existsSync(outPath)) {
        writeFileSync(
          outPath,
          JSON.stringify({
            ok: false,
            error: `Python exited with code ${exitCode ?? "?"} before writing output.json. Check this terminal for torch/fair-esm errors.`,
          }),
          "utf-8"
        );
      }
    } catch {
      /* ignore */
    }
    try {
      writeFileSync(donePath, JSON.stringify({ exitCode: exitCode ?? -1 }), "utf-8");
    } catch {
      /* ignore */
    }
    if (process.env.NODE_ENV === "development") {
      console.info(`[predict] job ${jobId}: subprocess finished (exit ${exitCode ?? "?"})`);
    }
  };

  child.on("error", (err) => {
    try {
      writeFileSync(
        outPath,
        JSON.stringify({
          ok: false,
          error: `Could not spawn Python (${py}): ${err.message}. Set PYTHON_BIN to your conda env python.exe.`,
        }),
        "utf-8"
      );
    } catch {
      /* ignore */
    }
    finishOnce(-1);
    console.error(`[predict] job ${jobId}: spawn error`, err);
  });

  child.on("exit", (code) => finishOnce(code));

  return NextResponse.json(
    {
      accepted: true,
      job_id: jobId,
      message:
        "Python is running in the background. Poll GET /api/pipeline/predict/status?job=… every few seconds until status is done (avoids Next.js ~10 min HTTP limit on the start request).",
    },
    { status: 202 }
  );
}
