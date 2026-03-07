const API_BASE = typeof import.meta.env.VITE_API_URL === "string" && import.meta.env.VITE_API_URL
  ? import.meta.env.VITE_API_URL
  : "/api";

function formBody(file: File, threshold?: number): FormData {
  const form = new FormData();
  form.append("file", file);
  if (threshold != null && !Number.isNaN(threshold)) {
    form.append("threshold", String(threshold));
  }
  return form;
}

export async function fetchHealth(): Promise<import("./types").HealthResponse> {
  const r = await fetch(`${API_BASE}/health`);
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
  return r.json();
}

export async function fetchPredict(file: File, threshold?: number): Promise<import("./types").PredictStats> {
  const r = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    body: formBody(file, threshold),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Predict failed: ${r.status}`);
  }
  return r.json();
}

export async function fetchMask(file: File, threshold?: number): Promise<Blob> {
  const r = await fetch(`${API_BASE}/predict/mask`, {
    method: "POST",
    body: formBody(file, threshold),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Mask failed: ${r.status}`);
  }
  return r.blob();
}

export async function fetchProbability(file: File, threshold?: number): Promise<Blob> {
  const r = await fetch(`${API_BASE}/predict/probability`, {
    method: "POST",
    body: formBody(file, threshold),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Probability failed: ${r.status}`);
  }
  return r.blob();
}

export async function fetchVisualization(file: File, threshold?: number): Promise<Blob> {
  const r = await fetch(`${API_BASE}/predict/visualization`, {
    method: "POST",
    body: formBody(file, threshold),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Visualization failed: ${r.status}`);
  }
  return r.blob();
}

export async function fetchPseudoRgb(file: File): Promise<Blob> {
  const r = await fetch(`${API_BASE}/predict/pseudo_rgb`, {
    method: "POST",
    body: formBody(file),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `Pseudo RGB failed: ${r.status}`);
  }
  return r.blob();
}

export async function downloadGeoTiff(file: File, threshold?: number): Promise<void> {
  const r = await fetch(`${API_BASE}/predict/geotiff`, {
    method: "POST",
    body: formBody(file, threshold),
  });
  if (!r.ok) {
    const err = await r.json().catch(() => ({}));
    throw new Error(err.detail || `GeoTIFF failed: ${r.status}`);
  }
  const blob = await r.blob();
  const name = file.name.replace(/\.[^.]+$/i, "") + "_mask.tif";
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  URL.revokeObjectURL(a.href);
}
