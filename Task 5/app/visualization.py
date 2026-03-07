from __future__ import annotations

import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from inference import compute_mask_metrics
from preprocessing import pseudo_rgb


def render_png_figure(fig) -> bytes:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    return buffer.getvalue()


def probability_to_grayscale_png(probability_map: np.ndarray) -> bytes:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(probability_map, cmap="gray", vmin=0, vmax=1)
    ax.set_title("Probability Map")
    ax.axis("off")
    return render_png_figure(fig)


def prediction_visualization_png(raw_chw: np.ndarray, probability_map: np.ndarray, binary_mask: np.ndarray, ground_truth: np.ndarray | None = None, threshold: float | None = None) -> bytes:
    has_gt = ground_truth is not None
    ncols = 4 if has_gt else 3
    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    axes[0].imshow(pseudo_rgb(raw_chw))
    axes[0].set_title("Pseudo RGB")
    axes[0].axis("off")

    next_col = 1
    if has_gt:
        axes[1].imshow(ground_truth, cmap="Blues", vmin=0, vmax=1)
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        next_col = 2

    axes[next_col].imshow(probability_map, cmap="RdBu_r", vmin=0, vmax=1)
    axes[next_col].set_title("Probability Map")
    axes[next_col].axis("off")

    title = "Prediction"
    if threshold is not None:
        title = f"Prediction (t={threshold:.2f})"
    if has_gt:
        metrics = compute_mask_metrics(binary_mask, ground_truth)
        title += f"\nIoU={metrics['iou']:.3f} F1={metrics['f1']:.3f}"

    axes[next_col + 1].imshow(binary_mask, cmap="Blues", vmin=0, vmax=1)
    axes[next_col + 1].set_title(title)
    axes[next_col + 1].axis("off")

    fig.suptitle("Water Segmentation Prediction", fontsize=12)
    fig.tight_layout()
    return render_png_figure(fig)
