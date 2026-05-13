"use client";
import { useRef, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  className?: string;
  glowColor?: string;
  borderRadius?: number;
}

export default function GlowBorder({
  children,
  className = "",
  glowColor = "124, 58, 237",
  borderRadius = 16,
}: Props) {
  const ref = useRef<HTMLDivElement>(null);

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * 100;
    const y = ((e.clientY - rect.top) / rect.height) * 100;
    ref.current.style.setProperty("--glow-x", `${x}%`);
    ref.current.style.setProperty("--glow-y", `${y}%`);
  };

  return (
    <div
      ref={ref}
      onMouseMove={handleMouseMove}
      className={`glow-border ${className}`}
      style={
        {
          "--glow-rgb": glowColor,
          borderRadius: `${borderRadius}px`,
        } as React.CSSProperties
      }
    >
      {children}
    </div>
  );
}
