import { useEffect, useState, useCallback } from "react";
import { useParams } from "react-router-dom";
import { fetchHealth, downloadGeoTiff } from "../api";
import type { ProcessedResult } from "../types";
import type { MaskColorScheme } from "../types";
import { ComparisonSlider } from "./ComparisonSlider";
import { MetadataPanel } from "./MetadataPanel";
import { OverlayView } from "./OverlayView";
import { ZoomPanView } from "./ZoomPanView";

type ViewMode = "comparison" | "overlay" | "panels";

interface DetailViewProps {
  getResult: (id: string) => ProcessedResult | undefined;
  onRemove: (id: string) => void;
  onBack: () => void;
}

export function DetailView({ getResult, onRemove, onBack }: DetailViewProps) {
  const { id } = useParams<{ id: string }>();
  const result = id ? getResult(id) : undefined;
  const [viewMode, setViewMode] = useState<ViewMode>("comparison");
  const [overlayOpacity, setOverlayOpacity] = useState(0.6);
  const [colorScheme, setColorScheme] = useState<MaskColorScheme>("blues");
  const [health, setHealth] = useState<{ device?: string; threshold?: number }>({});
  const [geoTiffLoading, setGeoTiffLoading] = useState(false);

  useEffect(() => {
    fetchHealth()
      .then((h) => setHealth({ device: h.device, threshold: h.threshold }))
      .catch(() => {});
  }, []);

  const revokeAndRemove = useCallback(() => {
    if (!result) return;
    [result.pseudoRgbUrl, result.maskUrl, result.probabilityUrl, result.visualizationUrl].forEach(
      URL.revokeObjectURL
    );
    onRemove(result.id);
    onBack();
  }, [result, onRemove, onBack]);

  if (!result) {
    return (
      <div className="rounded-xl border border-surface-700 bg-surface-900/50 p-8 text-center">
        <p className="text-surface-400">Result not found.</p>
        <button
          type="button"
          onClick={onBack}
          className="mt-4 text-brand-400 hover:underline"
        >
          Back to gallery
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <button
            type="button"
            onClick={onBack}
            className="text-surface-400 hover:text-surface-100 transition-colors"
          >
            ← Back
          </button>
          <h1 className="text-xl font-semibold text-surface-100 truncate max-w-xs sm:max-w-md" title={result.filename}>
            {result.filename}
          </h1>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-surface-500">View:</span>
          {(["comparison", "overlay", "panels"] as const).map((mode) => (
            <button
              key={mode}
              type="button"
              onClick={() => setViewMode(mode)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium capitalize transition-colors ${
                viewMode === mode
                  ? "bg-brand-600 text-surface-50"
                  : "bg-surface-800 text-surface-300 hover:bg-surface-700"
              }`}
            >
              {mode}
            </button>
          ))}
          <button
            type="button"
            onClick={revokeAndRemove}
            className="px-3 py-1.5 rounded-lg text-sm text-red-400 hover:bg-surface-800"
          >
            Delete
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          {(viewMode === "comparison" || viewMode === "overlay") && (
            <div className="space-y-2">
              {viewMode === "overlay" && (
                <div className="flex flex-wrap items-center gap-4 p-3 rounded-lg bg-surface-800/50">
                  <label className="flex items-center gap-2 text-sm text-surface-300">
                    Overlay opacity
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.05}
                      value={overlayOpacity}
                      onChange={(e) => setOverlayOpacity(Number(e.target.value))}
                      className="w-24"
                    />
                    <span className="font-mono w-8">{(overlayOpacity * 100).toFixed(0)}%</span>
                  </label>
                  <label className="flex items-center gap-2 text-sm text-surface-300">
                    Color
                    <select
                      value={colorScheme}
                      onChange={(e) => setColorScheme(e.target.value as MaskColorScheme)}
                      className="rounded border border-surface-600 bg-surface-800 px-2 py-1 text-surface-100"
                    >
                      <option value="blues">Blues</option>
                      <option value="teal">Teal</option>
                      <option value="cyan">Cyan</option>
                      <option value="green">Green</option>
                      <option value="viridis">Viridis</option>
                    </select>
                  </label>
                </div>
              )}
              <ZoomPanView>
                {viewMode === "comparison" ? (
                  <ComparisonSlider
                    leftLabel="Pseudo RGB"
                    rightLabel="Mask"
                    leftImage={result.pseudoRgbUrl}
                    rightImage={result.maskUrl}
                    className="min-w-[400px] min-h-[400px]"
                  />
                ) : (
                  <OverlayView
                    baseImageUrl={result.pseudoRgbUrl}
                    maskImageUrl={result.maskUrl}
                    maskOpacity={overlayOpacity}
                    colorScheme={colorScheme}
                    className="min-w-[400px] min-h-[400px]"
                  />
                )}
              </ZoomPanView>
            </div>
          )}
          {viewMode === "panels" && (
            <ZoomPanView>
              <img
                src={result.visualizationUrl}
                alt="Panels"
                className="max-w-full h-auto block"
              />
            </ZoomPanView>
          )}
        </div>

        <div className="space-y-4">
          <MetadataPanel
            stats={result.stats}
            processingTimeMs={result.processingTimeMs}
            device={health.device}
            modelThreshold={health.threshold}
          />
          <button
            type="button"
            disabled={geoTiffLoading}
            onClick={async () => {
              setGeoTiffLoading(true);
              try {
                await downloadGeoTiff(result.file, result.stats.threshold);
              } finally {
                setGeoTiffLoading(false);
              }
            }}
            className="w-full py-2 rounded-lg bg-surface-700 text-surface-200 text-sm font-medium hover:bg-surface-600 disabled:opacity-50"
          >
            {geoTiffLoading ? "Downloading…" : "Download GeoTIFF"}
          </button>
          <div className="rounded-xl border border-surface-700 bg-surface-900/50 p-4">
            <h3 className="text-sm font-semibold text-surface-200 mb-2">Probability map</h3>
            <img
              src={result.probabilityUrl}
              alt="Probability"
              className="w-full rounded-lg border border-surface-600"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
