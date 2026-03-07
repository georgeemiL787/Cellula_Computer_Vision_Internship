import { useCallback, useState } from "react";
import {
  fetchPredict,
  fetchMask,
  fetchProbability,
  fetchVisualization,
  fetchPseudoRgb,
} from "../api";
import type { ProcessedResult } from "../types";

interface UploadFlowProps {
  onComplete: (result: ProcessedResult) => void;
}

export function UploadFlow({ onComplete }: UploadFlowProps) {
  const [file, setFile] = useState<File | null>(null);
  const [thresholdInput, setThresholdInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState("");

  const runPipeline = useCallback(async () => {
    if (!file) return;
    setError(null);
    setLoading(true);
    const th = thresholdInput === "" ? undefined : parseFloat(thresholdInput);
    if (thresholdInput !== "" && Number.isNaN(th)) {
      setError("Invalid threshold");
      setLoading(false);
      return;
    }
    const start = performance.now();
    const urls: string[] = [];

    try {
      setProgress("Running prediction…");
      const [stats, maskBlob, probBlob, vizBlob, rgbBlob] = await Promise.all([
        fetchPredict(file, th),
        fetchMask(file, th),
        fetchProbability(file, th),
        fetchVisualization(file, th),
        fetchPseudoRgb(file),
      ]);

      const pseudoRgbUrl = URL.createObjectURL(rgbBlob);
      const maskUrl = URL.createObjectURL(maskBlob);
      const probabilityUrl = URL.createObjectURL(probBlob);
      const visualizationUrl = URL.createObjectURL(vizBlob);
      urls.push(pseudoRgbUrl, maskUrl, probabilityUrl, visualizationUrl);

      const processingTimeMs = Math.round(performance.now() - start);
      const result: ProcessedResult = {
        id: crypto.randomUUID(),
        filename: file.name,
        file,
        stats,
        pseudoRgbUrl,
        maskUrl,
        probabilityUrl,
        visualizationUrl,
        processingTimeMs,
        createdAt: Date.now(),
      };
      onComplete(result);
      setFile(null);
      setThresholdInput("");
    } catch (e) {
      urls.forEach(URL.revokeObjectURL);
      setError(e instanceof Error ? e.message : "Processing failed");
    } finally {
      setLoading(false);
      setProgress("");
    }
  }, [file, thresholdInput, onComplete]);

  const onFileChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    setFile(f ?? null);
    setError(null);
  }, []);

  const validFile = file && (file.name.toLowerCase().endsWith(".tif") || file.name.toLowerCase().endsWith(".tiff"));

  return (
    <section className="mb-8">
      <h2 className="text-lg font-semibold text-surface-100 mb-3">Upload & process</h2>
      <div className="rounded-xl border border-surface-700 bg-surface-900/50 p-6">
        <div className="flex flex-col sm:flex-row gap-4 items-start">
          <label className="flex-1 w-full cursor-pointer">
            <span className="sr-only">Choose .tif / .tiff file</span>
            <input
              type="file"
              accept=".tif,.tiff"
              onChange={onFileChange}
              disabled={loading}
              className="block w-full text-sm text-surface-300 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:bg-brand-600 file:text-surface-50 file:font-medium hover:file:bg-brand-500 file:transition-colors"
            />
          </label>
          <div className="flex items-center gap-2 shrink-0">
            <label className="text-sm text-surface-400">Threshold</label>
            <input
              type="number"
              min={0}
              max={1}
              step={0.01}
              placeholder="auto"
              value={thresholdInput}
              onChange={(e) => setThresholdInput(e.target.value)}
              disabled={loading}
              className="w-20 rounded-lg border border-surface-600 bg-surface-800 px-2 py-1.5 text-sm text-surface-100 font-mono"
            />
          </div>
          <button
            type="button"
            onClick={runPipeline}
            disabled={!validFile || loading}
            className="shrink-0 px-4 py-2 rounded-lg bg-brand-600 text-surface-50 font-medium hover:bg-brand-500 disabled:opacity-50 disabled:pointer-events-none transition-colors"
          >
            {loading ? "Processing…" : "Process"}
          </button>
        </div>
        {progress && (
          <p className="mt-3 text-sm text-brand-400">{progress}</p>
        )}
        {error && (
          <p className="mt-3 text-sm text-red-400" role="alert">{error}</p>
        )}
      </div>
    </section>
  );
}
