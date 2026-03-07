from __future__ import annotations

from typing import Any

import numpy as np
import torch

from config import get_config
from preprocessing import build_model_input


@torch.inference_mode()
def predict_from_raw(model: torch.nn.Module, device: torch.device, raw_chw: np.ndarray, threshold: float | None = None) -> dict[str, Any]:
    cfg = get_config()
    th = float(cfg["best_threshold"] if threshold is None else threshold)
    x = torch.from_numpy(build_model_input(raw_chw)).unsqueeze(0).to(device)
    logits = model(x)
    probs = torch.sigmoid(logits).squeeze().detach().cpu().numpy().astype(np.float32)
    pred = (probs > th).astype(np.uint8)

    return {
        "probability_map": probs,
        "binary_mask": pred,
        "threshold": th,
        "input_shape": list(raw_chw.shape),
        "selected_channels": cfg["selected_channel_names"],
        "water_pixels": int(pred.sum()),
        "total_pixels": int(pred.size),
        "water_ratio": float(pred.sum() / max(pred.size, 1)),
    }


def compute_mask_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = (pred > 0).astype(np.uint8)
    target = (target > 0).astype(np.uint8)
    tp = float((pred * target).sum())
    fp = float((pred * (1 - target)).sum())
    fn = float(((1 - pred) * target).sum())
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    iou = tp / (tp + fp + fn + 1e-8)
    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
