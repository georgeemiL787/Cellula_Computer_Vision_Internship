import type { MaskColorScheme } from "../types";

const MASK_COLORS: Record<MaskColorScheme, string> = {
  blues: "#1e3a5f",
  viridis: "#440154",
  teal: "#0d9488",
  cyan: "#0891b2",
  green: "#059669",
};

interface OverlayViewProps {
  baseImageUrl: string;
  maskImageUrl: string;
  maskOpacity: number;
  colorScheme: MaskColorScheme;
  className?: string;
}

export function OverlayView({
  baseImageUrl,
  maskImageUrl,
  maskOpacity,
  colorScheme,
  className = "",
}: OverlayViewProps) {
  const color = MASK_COLORS[colorScheme];

  return (
    <div className={`relative overflow-hidden rounded-lg bg-surface-900 aspect-square max-h-[70vh] ${className}`}>
      <img
        src={baseImageUrl}
        alt="Base"
        className="absolute inset-0 w-full h-full object-contain"
      />
      <div
        className="absolute inset-0 mix-blend-multiply"
        style={{
          background: color,
          maskImage: `url(${maskImageUrl})`,
          WebkitMaskImage: `url(${maskImageUrl})`,
          maskSize: "cover",
          WebkitMaskSize: "cover",
          maskPosition: "center",
          WebkitMaskPosition: "center",
          opacity: maskOpacity,
        }}
      />
    </div>
  );
}
