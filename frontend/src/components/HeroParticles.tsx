"use client";

/**
 * Hero background: linked particles + faint triangle mesh + continuous drift (no pointer interaction).
 */

import Particles, { initParticlesEngine } from "@tsparticles/react";
import type { ISourceOptions } from "@tsparticles/engine";
import { loadSlim } from "@tsparticles/slim";
import { useTheme } from "@/components/ThemeProvider";
import { useEffect, useMemo, useRef, useState } from "react";

export default function HeroParticles() {
  const { theme } = useTheme();
  const [engineReady, setEngineReady] = useState(false);
  const [reducedMotion] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches
  );
  const engineStarted = useRef(false);

  useEffect(() => {
    if (reducedMotion || engineStarted.current) return;
    engineStarted.current = true;
    void initParticlesEngine(async (engine) => {
      await loadSlim(engine);
    })
      .then(() => setEngineReady(true))
      .catch((err) => {
        console.warn("[HeroParticles] init failed", err);
      });
  }, [reducedMotion]);

  const linkColor = theme === "dark" ? "#94a3b8" : "#64748b";
  const triColor = theme === "dark" ? "#64748b" : "#94a3b8";

  const options = useMemo<ISourceOptions>(
    () => ({
      fullScreen: { enable: false },
      background: { color: { value: "transparent" } },
      fpsLimit: 60,
      detectRetina: true,
      smooth: true,
      interactivity: {
        events: {
          onHover: { enable: false },
          onClick: { enable: false },
        },
      },
      particles: {
        number: {
          value: 56,
          density: { enable: true, width: 960, height: 720 },
        },
        color: { value: linkColor },
        opacity: {
          value: { min: 0.2, max: 0.45 },
          animation: {
            enable: true,
            speed: 0.35,
            sync: false,
            minimumValue: 0.12,
          },
        },
        size: { value: { min: 1.2, max: 2.4 } },
        rotate: {
          value: { min: 0, max: 360 },
          direction: "random",
          animation: {
            enable: true,
            speed: 8,
            sync: false,
          },
        },
        links: {
          enable: true,
          distance: 118,
          color: linkColor,
          opacity: 0.22,
          width: 0.65,
          blink: { enable: false },
          triangles: {
            enable: true,
            color: triColor,
            opacity: 0.06,
          },
        },
        move: {
          enable: true,
          speed: { min: 0.12, max: 0.32 },
          direction: "none",
          random: true,
          straight: false,
          outModes: { default: "bounce" },
          attract: { enable: false },
        },
      },
    }),
    [linkColor, triColor]
  );

  if (reducedMotion || !engineReady) {
    return null;
  }

  return (
    <div
      className="pointer-events-none absolute inset-0 z-[1] h-full min-h-full w-full opacity-[0.78] dark:opacity-[0.72]"
      aria-hidden
    >
      <Particles
        id="hero-particles"
        className="size-full min-h-full"
        style={{ width: "100%", height: "100%" }}
        options={options}
      />
    </div>
  );
}
