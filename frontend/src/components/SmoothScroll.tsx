"use client";

/**
 * Lenis ran a perpetual requestAnimationFrame loop (~60fps) which spikes CPU/RAM on
 * low-end machines. Native scroll + CSS `scroll-behavior: smooth` on `html` is enough.
 */
export default function SmoothScroll({ children }: { children: React.ReactNode }) {
  return <>{children}</>;
}
