"""
Load deployment config from JSON and environment variables.
"""
import json
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

# Defaults (model expects 10 channels; replace norm stats with training values in production)
DEFAULT_IN_CHANNELS = 10
DEFAULT_SELECTED_CHANNELS = list(range(10))
DEFAULT_IMG_SIZE = 128


def load_config(config_path: Optional[Path] = None) -> dict:
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.json"
    if not config_path.exists():
        return _default_config()
    with open(config_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return _normalize_config(data)


def _default_config() -> dict:
    return {
        "in_channels": DEFAULT_IN_CHANNELS,
        "selected_channel_indices": DEFAULT_SELECTED_CHANNELS,
        "norm_mean": [0.0] * DEFAULT_IN_CHANNELS,
        "norm_std": [1.0] * DEFAULT_IN_CHANNELS,
        "img_size": DEFAULT_IMG_SIZE,
        "model_path": os.environ.get("MODEL_PATH", "best_model.pth"),
        "device": os.environ.get("DEVICE", "cpu"),
        "log_level": os.environ.get("LOG_LEVEL", "INFO"),
    }


def _normalize_config(data: dict) -> dict:
    in_channels = int(data.get("in_channels", DEFAULT_IN_CHANNELS))
    selected = data.get("selected_channel_indices")
    if selected is None:
        selected = list(range(in_channels))
    norm_mean = data.get("norm_mean")
    norm_std = data.get("norm_std")
    if norm_mean is None:
        norm_mean = [0.0] * in_channels
    if norm_std is None:
        norm_std = [1.0] * in_channels
    return {
        "in_channels": in_channels,
        "selected_channel_indices": selected,
        "norm_mean": np.array(norm_mean, dtype=np.float32),
        "norm_std": np.array(norm_std, dtype=np.float32),
        "img_size": int(data.get("img_size", DEFAULT_IMG_SIZE)),
        "model_path": os.environ.get("MODEL_PATH", data.get("model_path", "best_model.pth")),
        "device": os.environ.get("DEVICE", data.get("device", "cpu")),
        "log_level": os.environ.get("LOG_LEVEL", data.get("log_level", "INFO")),
    }
