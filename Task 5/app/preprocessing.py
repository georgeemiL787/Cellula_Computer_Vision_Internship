from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Any

import numpy as np
import rasterio
from rasterio.io import MemoryFile

from config import get_config

EPS = 1e-8


@dataclass
class RasterInput:
    image: np.ndarray
    profile: dict[str, Any]


def read_raster_bytes(data: bytes) -> RasterInput:
    with MemoryFile(data) as memfile:
        with memfile.open() as src:
            arr = src.read().astype(np.float32)
            profile = src.profile.copy()
    return RasterInput(image=arr, profile=profile)


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return a / (b + EPS)


def compute_water_indices(raw: np.ndarray) -> np.ndarray:
    g = raw[1]
    r = raw[2]
    nir = raw[6]
    swir1 = raw[8]
    swir2 = raw[9]
    blue = raw[0]

    ndwi = safe_divide(g - nir, g + nir)
    mndwi = safe_divide(g - swir1, g + swir1)
    awei_nsh = 4.0 * (g - swir1) - (0.25 * nir + 2.75 * swir2)
    ndvi = safe_divide(nir - r, nir + r)
    wri = safe_divide(g + r, nir + swir1)
    ndwi2 = safe_divide(g - swir2, g + swir2)
    ewi = safe_divide(g - swir1 - nir, g + swir1 + nir)
    tcw = 0.1511 * blue + 0.1973 * g + 0.3283 * r + 0.3407 * nir - 0.7117 * swir1 - 0.4559 * swir2

    indices = np.stack([ndwi, mndwi, awei_nsh, ndvi, wri, ndwi2, ewi, tcw], axis=0)
    return np.clip(indices, -1.0, 1.0).astype(np.float32)


def build_full_feature_stack(raw: np.ndarray) -> np.ndarray:
    if raw.ndim != 3:
        raise ValueError(f"Expected (C, H, W), got {raw.shape}")
    if raw.shape[0] < 12:
        raise ValueError(f"Expected 12 raw bands, got {raw.shape[0]}")
    raw12 = raw[:12].astype(np.float32)
    return np.concatenate([raw12, compute_water_indices(raw12)], axis=0)


def build_model_input(raw: np.ndarray) -> np.ndarray:
    cfg = get_config()
    full = build_full_feature_stack(raw)
    selected_idx = np.array(cfg["selected_channel_indices"], dtype=np.int64)
    mean = np.array(cfg["normalization_mean"], dtype=np.float32)[:, None, None]
    std = np.array(cfg["normalization_std"], dtype=np.float32)[:, None, None]
    selected = full[selected_idx]
    normalized = (selected - mean) / (std + EPS)
    return normalized.astype(np.float32)


def pseudo_rgb(raw: np.ndarray, r_idx: int = 2, g_idx: int = 1, b_idx: int = 0) -> np.ndarray:
    rgb = np.stack([raw[r_idx], raw[g_idx], raw[b_idx]], axis=-1).astype(np.float32)
    for c in range(3):
        p2, p98 = np.percentile(rgb[..., c], [2, 98])
        rgb[..., c] = np.clip((rgb[..., c] - p2) / (p98 - p2 + EPS), 0.0, 1.0)
    return rgb
