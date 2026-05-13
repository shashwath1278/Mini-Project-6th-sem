import { existsSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const JOB_RE = /^[a-f0-9]{32}$/;

export async function GET(req: NextRequest) {
  const job = req.nextUrl.searchParams.get("job");
  if (!job || !JOB_RE.test(job)) {
    return NextResponse.json({
      status: "error" as const,
      error: "Missing or invalid job id",
    });
  }

  const jobDir = join(tmpdir(), "plasticdeg-predict", job);
  const donePath = join(jobDir, "done.json");
  const outPath = join(jobDir, "out.json");

  if (!existsSync(jobDir)) {
    return NextResponse.json({
      status: "error" as const,
      error: "Job not found (already retrieved, expired, or invalid id).",
    });
  }

  if (!existsSync(donePath)) {
    return NextResponse.json({ status: "running" as const });
  }

  let result: unknown;
  try {
    if (!existsSync(outPath)) {
      return NextResponse.json({
        status: "error" as const,
        error: "Job finished but output file is missing.",
      });
    }
    const raw = readFileSync(outPath, "utf-8");
    result = JSON.parse(raw);
  } catch (e) {
    return NextResponse.json({
      status: "error" as const,
      error: `Could not read result: ${(e as Error).message}`,
    });
  }

  try {
    rmSync(jobDir, { recursive: true, force: true });
  } catch {
    /* ignore */
  }

  return NextResponse.json({ status: "done" as const, result });
}
