"use client";

import type { ReactNode } from "react";

type Props = {
  /** Short visible label (may include abbreviations). */
  children: ReactNode;
  /** Shown on hover / long-press as browser tooltip. */
  title: string;
  className?: string;
};

/** Dotted underline + cursor-help; uses native `title` for zero-dependency tooltips. */
export default function MetricHint({ children, title, className = "" }: Props) {
  return (
    <span
      title={title}
      className={`cursor-help border-b border-dotted border-muted-foreground/45 decoration-muted-foreground/45 ${className}`}
    >
      {children}
    </span>
  );
}
