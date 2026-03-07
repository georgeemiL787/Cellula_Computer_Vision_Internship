import { useRef, useState, useCallback } from "react";

interface ZoomPanViewProps {
  children: React.ReactNode;
  className?: string;
}

export function ZoomPanView({ children, className = "" }: ZoomPanViewProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [scale, setScale] = useState(1);
  const [translate, setTranslate] = useState({ x: 0, y: 0 });
  const lastPoint = useRef({ x: 0, y: 0 });
  const lastTranslate = useRef({ x: 0, y: 0 });

  const onWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? -0.1 : 0.1;
    setScale((s) => Math.min(5, Math.max(0.2, s + delta)));
  }, []);

  const onPointerDown = useCallback((e: React.PointerEvent) => {
    lastPoint.current = { x: e.clientX, y: e.clientY };
    lastTranslate.current = { ...translate };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  }, [translate]);

  const onPointerMove = useCallback((e: React.PointerEvent) => {
    if (e.buttons !== 1) return;
    const dx = e.clientX - lastPoint.current.x;
    const dy = e.clientY - lastPoint.current.y;
    const next = {
      x: lastTranslate.current.x + dx,
      y: lastTranslate.current.y + dy,
    };
    lastPoint.current = { x: e.clientX, y: e.clientY };
    lastTranslate.current = next;
    setTranslate(next);
  }, []);

  const reset = useCallback(() => {
    setScale(1);
    setTranslate({ x: 0, y: 0 });
  }, []);

  return (
    <div
      ref={containerRef}
      className={`relative overflow-hidden rounded-lg bg-surface-900 ${className}`}
      onWheel={onWheel}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={(e) => (e.target as HTMLElement).releasePointerCapture(e.pointerId)}
      style={{ touchAction: "none" }}
    >
      <div
        className="origin-center inline-block min-w-full min-h-full"
        style={{
          transform: `translate(${translate.x}px, ${translate.y}px) scale(${scale})`,
          cursor: "grab",
        }}
      >
        {children}
      </div>
      <div className="absolute bottom-2 right-2 flex gap-2">
        <button
          type="button"
          onClick={() => setScale((s) => Math.min(5, s + 0.25))}
          className="px-2 py-1 rounded bg-surface-800 text-surface-200 text-sm hover:bg-surface-700"
        >
          +
        </button>
        <button
          type="button"
          onClick={() => setScale((s) => Math.max(0.2, s - 0.25))}
          className="px-2 py-1 rounded bg-surface-800 text-surface-200 text-sm hover:bg-surface-700"
        >
          −
        </button>
        <button
          type="button"
          onClick={reset}
          className="px-2 py-1 rounded bg-surface-800 text-surface-200 text-sm hover:bg-surface-700"
        >
          Reset
        </button>
      </div>
    </div>
  );
}
