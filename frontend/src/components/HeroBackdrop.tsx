"use client";

/**
 * Lightweight hero background (CSS only). Replaces WebGL Prism for dev machines
 * with limited RAM — no requestAnimationFrame / GPU shader loops.
 */
export default function HeroBackdrop() {
  return (
    <div
      className="hero-backdrop pointer-events-none absolute inset-0"
      aria-hidden
    />
  );
}
