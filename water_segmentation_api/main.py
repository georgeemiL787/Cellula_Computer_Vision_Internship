"""
Water segmentation FastAPI application.
Run: uvicorn main:app --host 0.0.0.0 --port 8000
"""
import base64
import io
import logging
import os
import time
import uuid

import cv2
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.config import load_config
from app.model import WaterSegModel
from app.preprocessing import (
    preprocess_for_inference,
    read_image_from_bytes,
)

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
logger = logging.getLogger("water_segmentation_api")

# -----------------------------------------------------------------------------
# Config and model state
# -----------------------------------------------------------------------------
CONFIG = load_config()
DEVICE = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
MODEL: WaterSegModel | None = None

# Limits
MAX_UPLOAD_BYTES = 100 * 1024 * 1024  # 100 MB
EXPECTED_BANDS = 12
IMG_SIZE = CONFIG["img_size"]


def get_model() -> WaterSegModel:
    if MODEL is None:
        raise RuntimeError("Model not loaded")
    return MODEL


def _make_overlay(raw: np.ndarray, mask: np.ndarray) -> str | None:
    """Build an RGB image from raw bands (R,G,B = indices 2,1,0), resize to match mask, highlight water in cyan. Returns base64 PNG."""
    try:
        # Sentinel-2 order: 0=B2(blue), 1=B3(green), 2=B4(red)
        r = np.nan_to_num(raw[2], nan=0.0)
        g = np.nan_to_num(raw[1], nan=0.0)
        b = np.nan_to_num(raw[0], nan=0.0)
        rgb = np.stack([r, g, b], axis=-1)
        p2, p98 = np.percentile(rgb, (2, 98))
        rgb = np.clip((rgb - p2) / max(p98 - p2, 1e-8) * 255, 0, 255).astype(np.uint8)
        h, w = mask.shape
        rgb_small = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        # Overlay: where mask==1, blend in cyan (RGB)
        cyan = np.array([0, 255, 255], dtype=np.uint8)
        alpha = 0.5
        water = (mask > 0)[:, :, np.newaxis]
        rgb_small = np.where(water, (alpha * cyan + (1 - alpha) * rgb_small).astype(np.uint8), rgb_small)
        _, png_bytes = cv2.imencode(".png", cv2.cvtColor(rgb_small, cv2.COLOR_RGB2BGR))
        return base64.b64encode(png_bytes.tobytes()).decode("ascii")
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    logger.info("Loading config and model...")
    model_path = Path(CONFIG["model_path"])
    if not model_path.is_absolute():
        model_path = (Path(__file__).resolve().parent / model_path).resolve()
    if not model_path.exists():
        logger.error("Model file not found: %s", model_path)
        raise FileNotFoundError(f"Model file not found: {model_path}")
    MODEL = WaterSegModel(
        in_channels=CONFIG["in_channels"],
        encoder_name="efficientnet-b4",
        pretrained=False,
    )
    state = torch.load(model_path, map_location=DEVICE, weights_only=True)
    MODEL.load_state_dict(state, strict=True)
    MODEL.to(DEVICE)
    MODEL.eval()
    logger.info("Model loaded on device=%s", DEVICE)
    yield
    MODEL = None
    logger.info("Shutdown: model unloaded")


app = FastAPI(
    title="Water Segmentation API",
    description="Segment water bodies from multi-band satellite (TIF) imagery.",
    version="1.0.0",
    lifespan=lifespan,
)

# Frontend (serve at root)
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
async def index():
    """Serve the web interface."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(index_path, media_type="text/html")


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# -----------------------------------------------------------------------------
# Schemas
# -----------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str = "cpu"


class PredictResponse(BaseModel):
    mask_shape: list[int]
    threshold: float
    water_ratio: float
    mask_base64_png: str | None = None
    overlay_base64_png: str | None = None  # Your image with water regions highlighted


class PredictRawBody(BaseModel):
    """Base64-encoded numpy array (C, H, W) float32, C=10."""

    data_base64: str = Field(..., description="Base64-encoded numpy array (.npy bytes)")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Binary threshold for mask")


# -----------------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness: service is running."""
    return HealthResponse(
        status="ok",
        device=str(DEVICE),
    )


@app.get("/ready")
async def ready():
    """Readiness: model is loaded and ready for inference."""
    try:
        get_model()
        return {"status": "ready"}
    except RuntimeError:
        raise HTTPException(status_code=503, detail="Model not loaded")


@app.post("/predict", response_model=PredictResponse)
async def predict(
    file: UploadFile = File(..., description="Multi-band GeoTIFF (12 bands)"),
    threshold: float = 0.5,
):
    """
    Run water segmentation on an uploaded TIF image.
    Image is resized to the model input size (default 128x128) and normalized
    using configured mean/std. Returns mask shape, water ratio, and optional PNG.
    """
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be in [0, 1]")
    request_id = str(uuid.uuid4())[:8]
    logger.info("request_id=%s predict file=%s size=%s", request_id, file.filename, file.size)
    t0 = time.perf_counter()
    try:
        raw_bytes = await file.read()
    except Exception as e:
        logger.exception("request_id=%s read file failed: %s", request_id, e)
        raise HTTPException(status_code=400, detail="Failed to read uploaded file")
    if len(raw_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)} MB)",
        )
    try:
        raw = read_image_from_bytes(raw_bytes)
    except Exception as e:
        logger.warning("request_id=%s rasterio read failed: %s", request_id, e)
        raise HTTPException(
            status_code=400,
            detail="Invalid or unsupported TIF; expected multi-band raster",
        )
    if raw.ndim != 3:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 3D array (C, H, W), got ndim={raw.ndim}",
        )
    if raw.shape[0] < 10:
        raise HTTPException(
            status_code=400,
            detail=f"Expected at least 10 bands (Sentinel-2 style), got {raw.shape[0]}",
        )
    selected = CONFIG["selected_channel_indices"]
    norm_mean = CONFIG["norm_mean"]
    norm_std = CONFIG["norm_std"]
    try:
        x = preprocess_for_inference(
            raw,
            selected_channels=selected,
            norm_mean=norm_mean,
            norm_std=norm_std,
            target_size=(IMG_SIZE, IMG_SIZE),
        )
    except Exception as e:
        logger.exception("request_id=%s preprocessing failed: %s", request_id, e)
        raise HTTPException(status_code=500, detail="Preprocessing failed")
    # (C, H, W) -> (1, C, H, W)
    x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
    model = get_model()
    try:
        with torch.no_grad():
            logits = model(x_t)
    except Exception as e:
        logger.exception("request_id=%s inference failed: %s", request_id, e)
        raise HTTPException(status_code=500, detail="Inference failed")
    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    mask = (probs >= threshold).astype(np.uint8)
    water_ratio = float(mask.sum()) / max(mask.size, 1)
    # Encode mask as PNG
    _, png_bytes = cv2.imencode(".png", mask * 255)
    mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")
    # Build overlay: your image with water regions highlighted (RGB from bands R,G,B = 2,1,0)
    overlay_b64 = _make_overlay(raw, mask)
    elapsed = time.perf_counter() - t0
    logger.info("request_id=%s predict done shape=%s water_ratio=%.4f elapsed=%.3fs", request_id, list(mask.shape), water_ratio, elapsed)
    return PredictResponse(
        mask_shape=list(mask.shape),
        threshold=threshold,
        water_ratio=round(water_ratio, 6),
        mask_base64_png=mask_b64,
        overlay_base64_png=overlay_b64,
    )


@app.post("/predict/mask.png")
async def predict_mask_png(
    file: UploadFile = File(..., description="Multi-band GeoTIFF (12 bands)"),
    threshold: float = 0.5,
):
    """Same as /predict but returns only the mask as PNG (no JSON)."""
    resp = await predict(file=file, threshold=threshold)
    # Decode the base64 we just produced (we could avoid round-trip by computing mask in a shared helper)
    mask_b64 = resp.mask_base64_png
    if not mask_b64:
        raise HTTPException(status_code=500, detail="No mask generated")
    raw_png = base64.b64decode(mask_b64)
    return Response(content=raw_png, media_type="image/png")


@app.post("/predict/raw", response_model=PredictResponse)
async def predict_raw(body: PredictRawBody):
    """
    Run segmentation on a preprocessed 10-channel tensor provided as base64-encoded
    numpy array. Useful for testing or when you already have normalized (C, H, W) data.
    """
    request_id = str(uuid.uuid4())[:8]
    t0 = time.perf_counter()
    try:
        arr_bytes = base64.b64decode(body.data_base64)
        buf = io.BytesIO(arr_bytes)
        x = np.load(buf).astype(np.float32)
    except Exception as e:
        logger.warning("request_id=%s raw decode/load failed: %s", request_id, e)
        raise HTTPException(
            status_code=400,
            detail="Invalid data_base64: must be base64-encoded .npy array (C, H, W) float32, C=10",
        )
    if x.ndim != 3 or x.shape[0] != CONFIG["in_channels"]:
        raise HTTPException(
            status_code=400,
            detail=f"Expected shape (10, H, W), got {x.shape}",
        )
    x_t = torch.from_numpy(x).unsqueeze(0).to(DEVICE)
    model = get_model()
    with torch.no_grad():
        logits = model(x_t)
    probs = torch.sigmoid(logits).squeeze(0).squeeze(0).cpu().numpy()
    mask = (probs >= body.threshold).astype(np.uint8)
    water_ratio = float(mask.sum()) / max(mask.size, 1)
    _, png_bytes = cv2.imencode(".png", mask * 255)
    mask_b64 = base64.b64encode(png_bytes.tobytes()).decode("ascii")
    elapsed = time.perf_counter() - t0
    logger.info("request_id=%s predict/raw done shape=%s elapsed=%.3fs", request_id, list(mask.shape), elapsed)
    return PredictResponse(
        mask_shape=list(mask.shape),
        threshold=body.threshold,
        water_ratio=round(water_ratio, 6),
        mask_base64_png=mask_b64,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
