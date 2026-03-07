import { Link } from "react-router-dom";
import type { ProcessedResult } from "../types";

interface GalleryProps {
  results: ProcessedResult[];
  onRemove: (id: string) => void;
}

export function Gallery({ results, onRemove }: GalleryProps) {
  if (results.length === 0) {
    return (
      <section>
        <h2 className="text-lg font-semibold text-surface-100 mb-3">Processed images</h2>
        <div className="rounded-xl border border-dashed border-surface-600 bg-surface-900/30 p-12 text-center text-surface-500">
          <p>No results yet. Upload a .tif or .tiff image above to run water segmentation.</p>
        </div>
      </section>
    );
  }

  return (
    <section>
      <h2 className="text-lg font-semibold text-surface-100 mb-3">Processed images</h2>
      <ul className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-4">
        {results.map((r) => (
          <li key={r.id} className="group">
            <Link
              to={`/detail/${r.id}`}
              className="block rounded-xl border border-surface-700 bg-surface-900/50 overflow-hidden hover:border-brand-500/50 hover:ring-1 hover:ring-brand-500/30 transition-all focus:outline-none focus-visible:ring-2 focus-visible:ring-brand-500"
            >
              <div className="aspect-square relative bg-surface-800">
                <img
                  src={r.visualizationUrl}
                  alt={r.filename}
                  className="w-full h-full object-cover object-center"
                />
              </div>
              <div className="p-3">
                <p className="text-sm font-medium text-surface-200 truncate" title={r.filename}>
                  {r.filename}
                </p>
                <p className="text-xs text-surface-500 mt-0.5">
                  Water: {(r.stats.water_ratio * 100).toFixed(1)}% · {r.processingTimeMs}ms
                </p>
              </div>
            </Link>
            <button
              type="button"
              onClick={(e) => {
                e.preventDefault();
                onRemove(r.id);
              }}
              className="mt-1 text-xs text-surface-500 hover:text-red-400 transition-colors"
            >
              Remove
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
