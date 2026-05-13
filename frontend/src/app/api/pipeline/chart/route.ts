import fs from "fs";
import { NextRequest, NextResponse } from "next/server";
import { CHART_FILES, type ChartKind, chartAbsolutePath } from "@/lib/artifacts";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const kind = req.nextUrl.searchParams.get("kind") as ChartKind | null;
  if (!kind || !(kind in CHART_FILES)) {
    return NextResponse.json(
      { error: "Invalid kind. Use pr_v1|pr_v2|roc_v1|roc_v2" },
      { status: 400 }
    );
  }
  const abs = chartAbsolutePath(kind);
  try {
    if (!fs.existsSync(abs)) {
      return NextResponse.json({ error: "Chart file not found" }, { status: 404 });
    }
    const buf = fs.readFileSync(abs);
    return new NextResponse(buf, {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "public, max-age=60",
      },
    });
  } catch {
    return NextResponse.json({ error: "Read failed" }, { status: 500 });
  }
}
