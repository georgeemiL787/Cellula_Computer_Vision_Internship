from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from config import get_config
from inference import predict_from_raw
from model import load_model
from preprocessing import read_raster_bytes, pseudo_rgb
from visualization import prediction_visualization_png, probability_to_grayscale_png

STATE: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = os.getenv("MODEL_PATH")
    device = os.getenv("DEVICE")
    model, torch_device = load_model(model_path=model_path, device=device)
    STATE["model"] = model
    STATE["device"] = torch_device
    yield
    STATE.clear()


app = FastAPI(
    title="Water Segmentation API",
    version="2.0.0",
    description="FastAPI service for water-body segmentation with visualization and GeoTIFF export.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = APIRouter(prefix="/api")


def validate_tiff(filename: str) -> None:
    if not filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Upload a .tif or .tiff file.")


@api.get("/health")
def health() -> dict[str, Any]:
    cfg = get_config()
    return {
        "status": "ok",
        "device": str(STATE.get("device", "unknown")),
        "threshold": cfg["best_threshold"],
        "selected_channels": cfg["selected_channel_names"],
        "raw_band_order": cfg["raw_band_order"],
    }


@api.post("/predict")
async def predict(file: UploadFile = File(...), threshold: float | None = Form(default=None)) -> JSONResponse:
    validate_tiff(file.filename)
    raw = await file.read()
    try:
        raster = read_raster_bytes(raw)
        result = predict_from_raw(STATE["model"], STATE["device"], raster.image, threshold=threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    response = {
        "filename": file.filename,
        "input_shape": result["input_shape"],
        "selected_channels": result["selected_channels"],
        "threshold": result["threshold"],
        "water_pixels": result["water_pixels"],
        "total_pixels": result["total_pixels"],
        "water_ratio": result["water_ratio"],
    }
    return JSONResponse(response)


@api.post("/predict/mask")
async def predict_mask(file: UploadFile = File(...), threshold: float | None = Form(default=None)) -> StreamingResponse:
    validate_tiff(file.filename)
    raw = await file.read()
    try:
        raster = read_raster_bytes(raw)
        result = predict_from_raw(STATE["model"], STATE["device"], raster.image, threshold=threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    mask = Image.fromarray((result["binary_mask"] * 255).astype(np.uint8), mode="L")
    buffer = io.BytesIO()
    mask.save(buffer, format="PNG")
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/png")


@api.post("/predict/probability")
async def predict_probability(file: UploadFile = File(...), threshold: float | None = Form(default=None)) -> Response:
    validate_tiff(file.filename)
    raw = await file.read()
    try:
        raster = read_raster_bytes(raw)
        result = predict_from_raw(STATE["model"], STATE["device"], raster.image, threshold=threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    png_bytes = probability_to_grayscale_png(result["probability_map"])
    return Response(content=png_bytes, media_type="image/png")


@api.post("/predict/visualization")
async def predict_visualization(
    file: UploadFile = File(...),
    mask: UploadFile | None = File(default=None),
    threshold: float | None = Form(default=None),
) -> Response:
    validate_tiff(file.filename)
    raw = await file.read()
    try:
        raster = read_raster_bytes(raw)
        result = predict_from_raw(STATE["model"], STATE["device"], raster.image, threshold=threshold)
        gt = None
        if mask is not None:
            validate_tiff(mask.filename)
            mask_raster = read_raster_bytes(await mask.read())
            gt = mask_raster.image[0]
            gt = (gt > 0).astype(np.uint8)
        png_bytes = prediction_visualization_png(
            raw_chw=raster.image,
            probability_map=result["probability_map"],
            binary_mask=result["binary_mask"],
            ground_truth=gt,
            threshold=result["threshold"],
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Visualization failed: {exc}") from exc

    return Response(content=png_bytes, media_type="image/png")


@api.post("/predict/pseudo_rgb")
async def predict_pseudo_rgb(file: UploadFile = File(...)) -> Response:
    """Return pseudo-RGB preview of the input raster as PNG (for frontend comparison)."""
    validate_tiff(file.filename)
    raw = await file.read()
    try:
        raster = read_raster_bytes(raw)
        rgb = pseudo_rgb(raster.image)
        img = Image.fromarray((np.clip(rgb * 255, 0, 255).astype(np.uint8)), mode="RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        return Response(content=buffer.getvalue(), media_type="image/png")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Pseudo RGB failed: {exc}") from exc


@api.post("/predict/geotiff")
async def predict_geotiff(file: UploadFile = File(...), threshold: float | None = Form(default=None)) -> StreamingResponse:
    validate_tiff(file.filename)
    raw = await file.read()
    try:
        raster = read_raster_bytes(raw)
        result = predict_from_raw(STATE["model"], STATE["device"], raster.image, threshold=threshold)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"GeoTIFF prediction failed: {exc}") from exc

    profile = raster.profile.copy()
    profile.update(dtype=rasterio.uint8, count=1, compress="lzw")
    buffer = io.BytesIO()
    with rasterio.io.MemoryFile() as memfile:
        with memfile.open(**profile) as dst:
            dst.write(result["binary_mask"].astype(np.uint8), 1)
        buffer.write(memfile.read())
    buffer.seek(0)
    headers = {"Content-Disposition": f'attachment; filename="{os.path.splitext(file.filename)[0]}_mask.tif"'}
    return StreamingResponse(buffer, media_type="image/tiff", headers=headers)


app.include_router(api)

# Optional: serve frontend static files when present (e.g. after `cd frontend && npm run build`)
_frontend_dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if _frontend_dist.exists():
    app.mount("/assets", StaticFiles(directory=_frontend_dist / "assets"), name="assets")

    @app.get("/{full_path:path}")
    def serve_spa(full_path: str):
        if full_path.startswith("api/") or full_path == "api":
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        path = _frontend_dist / full_path
        if path.is_file():
            return FileResponse(path)
        return FileResponse(_frontend_dist / "index.html")
