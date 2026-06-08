import fs from "fs";
import { NextRequest, NextResponse } from "next/server";
import {
  FIGURE_BUNDLE_FILES,
  type FigureBundleKind,
  figureBundleAbsolutePath,
} from "@/lib/artifacts";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const kind = req.nextUrl.searchParams.get("kind") as FigureBundleKind | null;
  if (!kind || !(kind in FIGURE_BUNDLE_FILES)) {
    return NextResponse.json(
      { error: "Invalid kind for figure bundle." },
      { status: 400 }
    );
  }
  const abs = figureBundleAbsolutePath(kind);
  try {
    if (!fs.existsSync(abs)) {
      return NextResponse.json({ error: "Figure not found" }, { status: 404 });
    }
    const buf = fs.readFileSync(abs);
    return new NextResponse(buf, {
      headers: {
        "Content-Type": "image/png",
        "Cache-Control": "public, max-age=120",
      },
    });
  } catch {
    return NextResponse.json({ error: "Read failed" }, { status: 500 });
  }
}
