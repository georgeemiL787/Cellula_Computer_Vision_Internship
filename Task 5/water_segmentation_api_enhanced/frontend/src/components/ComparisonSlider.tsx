import { useRef, useState, useCallback } from "react";

interface ComparisonSliderProps {
  leftLabel: string;
  rightLabel: string;
  leftImage: string;
  rightImage: string;
  className?: string;
}

export function ComparisonSlider({
  leftLabel,
  rightLabel,
  leftImage,
  rightImage,
  className = "",
}: ComparisonSliderProps) {
  const [position, setPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleMove = useCallback(
    (clientX: number) => {
      const el = containerRef.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      const x = Math.max(0, Math.min(100, ((clientX - rect.left) / rect.width) * 100));
      setPosition(x);
    },
    []
  );

  const onPointerDown = useCallback(
    (e: React.PointerEvent) => {
      (e.target as HTMLElement).setPointerCapture(e.pointerId);
      handleMove(e.clientX);
    },
    [handleMove]
  );

  const onPointerMove = useCallback(
    (e: React.PointerEvent) => {
      if (e.buttons !== 1) return;
      handleMove(e.clientX);
    },
    [handleMove]
  );

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-lg bg-surface-900 ${className}`}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={(e) => (e.target as HTMLElement).releasePointerCapture(e.pointerId)}
      onPointerLeave={() => {}}
    >
      <div className="relative aspect-square w-full max-h-[70vh]">
        <img
          src={leftImage}
          alt={leftLabel}
          className="absolute inset-0 w-full h-full object-contain"
        />
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - position}% 0 0)` }}
        >
          <img
            src={rightImage}
            alt={rightLabel}
            className="absolute inset-0 w-full h-full object-contain"
          />
        </div>
        <div
          className="absolute top-0 bottom-0 w-1 bg-brand-400 cursor-ew-resize flex items-center justify-center"
          style={{ left: `${position}%`, transform: "translateX(-50%)" }}
        >
          <div className="w-2 h-10 rounded-full bg-brand-400 shadow-lg" />
        </div>
      </div>
      <div className="absolute bottom-2 left-2 right-2 flex justify-between text-xs text-surface-300">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  );
}
