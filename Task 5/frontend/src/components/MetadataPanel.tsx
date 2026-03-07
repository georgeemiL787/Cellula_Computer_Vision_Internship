import type { PredictStats } from "../types";

interface MetadataPanelProps {
  stats: PredictStats;
  processingTimeMs: number;
  device?: string;
  modelThreshold?: number;
}

export function MetadataPanel({
  stats,
  processingTimeMs,
  device,
  modelThreshold,
}: MetadataPanelProps) {
  return (
    <div className="rounded-xl border border-surface-700 bg-surface-900/50 p-4 space-y-3">
      <h3 className="text-sm font-semibold text-surface-200">Result metadata</h3>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
        <dt className="text-surface-500">Filename</dt>
        <dd className="text-surface-100 font-mono truncate" title={stats.filename}>{stats.filename}</dd>

        <dt className="text-surface-500">Input shape</dt>
        <dd className="text-surface-100 font-mono">{stats.input_shape.join(" × ")}</dd>

        <dt className="text-surface-500">Threshold</dt>
        <dd className="text-surface-100 font-mono">{stats.threshold.toFixed(2)}</dd>

        <dt className="text-surface-500">Water pixels</dt>
        <dd className="text-surface-100 font-mono">{stats.water_pixels.toLocaleString()}</dd>

        <dt className="text-surface-500">Total pixels</dt>
        <dd className="text-surface-100 font-mono">{stats.total_pixels.toLocaleString()}</dd>

        <dt className="text-surface-500">Water ratio</dt>
        <dd className="text-surface-100 font-mono">{(stats.water_ratio * 100).toFixed(2)}%</dd>

        <dt className="text-surface-500">Processing time</dt>
        <dd className="text-surface-100 font-mono">{processingTimeMs} ms</dd>

        {device != null && (
          <>
            <dt className="text-surface-500">Device</dt>
            <dd className="text-surface-100 font-mono">{device}</dd>
          </>
        )}
        {modelThreshold != null && (
          <>
            <dt className="text-surface-500">Model default threshold</dt>
            <dd className="text-surface-100 font-mono">{modelThreshold.toFixed(2)}</dd>
          </>
        )}
      </dl>
      {stats.selected_channels?.length > 0 && (
        <div>
          <dt className="text-surface-500 text-sm mb-1">Channels</dt>
          <dd className="text-surface-300 text-xs font-mono break-all">
            {stats.selected_channels.join(", ")}
          </dd>
        </div>
      )}
    </div>
  );
}
