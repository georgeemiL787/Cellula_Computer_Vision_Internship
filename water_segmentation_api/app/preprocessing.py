"""
Preprocessing for water segmentation: read TIF, compute water indices, normalize.
Band indices follow Sentinel-2 order: B2,B3,B4,B5,B6,B7,B8,B8A,B11,B12,B1,B9.
"""
import io
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import rasterio

# Band indices (0-based) for 12-band Sentinel-2 order
GREEN_IDX = 1   # B3
RED_IDX = 2     # B4
NIR_IDX = 6     # B8
SWIR1_IDX = 8   # B11
SWIR2_IDX = 9   # B12


def compute_water_indices(image: np.ndarray) -> np.ndarray:
    """
    Compute 8 water spectral indices.
    Input: (C, H, W) float32, C >= 10
    Output: (8, H, W) float32, clipped to [-1, 1]
    """
    g = image[GREEN_IDX]
    r = image[RED_IDX]
    ni = image[NIR_IDX]
    s1 = image[SWIR1_IDX]
    s2 = image[SWIR2_IDX]
    b = image[0]  # B2 (Blue)
    eps = 1e-8

    ndwi = (g - ni) / (g + ni + eps)
    mndwi = (g - s1) / (g + s1 + eps)
    awei = 4 * (g - s1) - (0.25 * ni + 2.75 * s1)
    awei = np.clip(awei / (np.abs(awei).max() + eps), -1, 1)
    ndvi = (ni - r) / (ni + r + eps)
    wri = (g + r) / (ni + s1 + eps)
    wri = np.clip((wri - 1) / 2, -1, 1)
    ndwi2 = (g - s2) / (g + s2 + eps)
    ewi = (g - s1 - ni) / (g + s1 + ni + eps)
    tcw = (
        0.1511 * b + 0.1973 * g + 0.3283 * r + 0.3407 * ni
        - 0.7117 * s1 - 0.4559 * s2
    )
    tcw = np.clip(tcw / (np.abs(tcw).max() + eps), -1, 1)

    indices = np.stack([ndwi, mndwi, awei, ndvi, wri, ndwi2, ewi, tcw], axis=0)
    return np.clip(indices, -1, 1).astype(np.float32)


def build_full_image(raw: np.ndarray, selected_channels: List[int]) -> np.ndarray:
    """
    Stack raw bands + 8 water indices -> (20, H, W), then select channels.
    raw: (12, H, W) float32
    """
    indices = compute_water_indices(raw)
    full = np.concatenate([raw, indices], axis=0)
    return full[selected_channels].astype(np.float32)


def normalize(
    image: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """Per-channel z-score normalization. image: (C, H, W) float32."""
    return (image - mean[:, None, None]) / (std[:, None, None] + eps)


def read_image_from_bytes(data: bytes) -> np.ndarray:
    """Read multi-band TIF from bytes. Returns float32 (C, H, W)."""
    with rasterio.open(io.BytesIO(data)) as src:
        return src.read().astype(np.float32)


def read_image_from_path(path: Union[str, Path]) -> np.ndarray:
    """Read multi-band TIF from path. Returns float32 (C, H, W)."""
    with rasterio.open(path) as src:
        return src.read().astype(np.float32)


def preprocess_for_inference(
    raw: np.ndarray,
    selected_channels: List[int],
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Full pipeline: build 20ch -> select channels -> normalize.
    Optionally resize to target_size (H, W). Returns (C, H, W) float32.
    """
    full = build_full_image(raw, selected_channels)
    x = normalize(full, norm_mean, norm_std)
    if target_size is not None:
        # Resize each channel (C, H, W)
        resized = np.zeros((x.shape[0], target_size[0], target_size[1]), dtype=np.float32)
        for c in range(x.shape[0]):
            resized[c] = cv2.resize(
                x[c], (target_size[1], target_size[0]),
                interpolation=cv2.INTER_LINEAR,
            )
        x = resized
    return x
