# Water Segmentation API

FastAPI service for water body segmentation from multi-band satellite (GeoTIFF) imagery, using the PyTorch model from Task 4/5 (EfficientNet-B4 + U-Net++ with channel adapter).

## Features

- **Endpoints**: `GET /health`, `GET /ready`, `POST /predict` (TIF upload), `POST /predict/mask.png`, `POST /predict/raw` (base64 numpy)
- **Input validation**: file size limit (100 MB), band count, shape checks
- **Error handling**: 400/413/500 with clear messages; structured logging with request IDs
- **Config**: `config.json` + env vars (`MODEL_PATH`, `DEVICE`, `LOG_LEVEL`); optional norm stats for best accuracy

## Local run (no Docker)

1. **Model file**  
   Copy the trained weights into this directory or set `MODEL_PATH`:
   ```bash
   copy "..\Task 5\best_model.pth" best_model.pth
   # or
   set MODEL_PATH=..\Task 5\best_model.pth
   ```

2. **Install and run**
   ```bash
   cd water_segmentation_api
   pip install -r requirements.txt
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

3. **Try**
   - Open http://localhost:8000/docs for Swagger UI.
   - `GET /health` and `GET /ready` for liveness/readiness.

## Docker (build from repo root)

Build and run:

```bash
# From repository root (Cellula_Internship)
docker build -f water_segmentation_api/Dockerfile -t water-segmentation-api .
docker run -p 8000:8000 water-segmentation-api
```

Or with docker-compose:

```bash
docker compose -f water_segmentation_api/docker-compose.yml up -d
```

The Dockerfile expects:
- Build context = **repository root** (so it can copy `Task 5/best_model.pth` and `water_segmentation_api/`).
- Multi-stage build for smaller image; non-root user; healthcheck on `/health`.

## Configuration

- **config.json**: `in_channels` (10), `selected_channel_indices`, `norm_mean`, `norm_std`, `img_size` (128), `model_path`, `device`, `log_level`.  
  For best accuracy, replace `norm_mean` and `norm_std` with values exported from training (e.g. from `compute_norm_stats` in the Task 4 notebook) and set `selected_channel_indices` to the channels used when training the saved model.
- **Environment**: `MODEL_PATH`, `DEVICE` (e.g. `cuda`), `LOG_LEVEL` override config.

## Production notes

- **Scalability**: Run multiple containers behind a load balancer; model is loaded per process.
- **Security**: Non-root user in Docker; limit upload size; validate inputs; avoid exposing internals in errors.
- **Performance**: Use `DEVICE=cuda` and a GPU-capable image if needed; consider batching for high throughput.
