export interface HealthResponse {
  status: string;
  device: string;
  threshold: number;
  selected_channels: string[];
  raw_band_order: string[];
}

export interface PredictStats {
  filename: string;
  input_shape: number[];
  selected_channels: string[];
  threshold: number;
  water_pixels: number;
  total_pixels: number;
  water_ratio: number;
}

export interface ProcessedResult {
  id: string;
  filename: string;
  file: File;
  stats: PredictStats;
  pseudoRgbUrl: string;
  maskUrl: string;
  probabilityUrl: string;
  visualizationUrl: string;
  processingTimeMs: number;
  createdAt: number;
}

export type MaskColorScheme = "blues" | "viridis" | "teal" | "cyan" | "green";
