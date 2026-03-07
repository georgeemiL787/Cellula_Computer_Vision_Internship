# Water Segmentation API - production image
# Multi-stage for smaller image; non-root user; minimal deps

# -----------------------------------------------------------------------------
# Builder: install dependencies and copy app
# -----------------------------------------------------------------------------
FROM python:3.11-slim as builder

WORKDIR /build

# System deps for rasterio (GDAL)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    build-essential \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Build from repo root: docker build -f water_segmentation_api/Dockerfile .
COPY water_segmentation_api/requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# -----------------------------------------------------------------------------
# Runtime: minimal image
# -----------------------------------------------------------------------------
FROM python:3.11-slim as runtime

WORKDIR /app

# Runtime deps only (rasterio needs libgdal)
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    libgdal32 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Application code (build from repo root: docker build -f water_segmentation_api/Dockerfile .)
COPY water_segmentation_api/app/ ./app/
COPY water_segmentation_api/main.py water_segmentation_api/config.json ./

# Model weights from Task 5 (build context = repo root)
COPY "Task 5/best_model.pth" ./best_model.pth

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

EXPOSE 8000

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/best_model.pth

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
