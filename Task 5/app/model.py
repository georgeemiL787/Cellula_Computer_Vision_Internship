from __future__ import annotations

from pathlib import Path
from typing import Tuple

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from config import WEIGHTS_PATH, get_config


class DeepChannelAdapter(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.main(x) + self.shortcut(x))


class WaterSegmentationModel(nn.Module):
    def __init__(self, in_channels: int, encoder_name: str = "efficientnet-b4") -> None:
        super().__init__()
        self.adapter = DeepChannelAdapter(in_channels)
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type="scse",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(self.adapter(x))


def load_model(model_path: str | Path | None = None, device: str | None = None) -> Tuple[WaterSegmentationModel, torch.device]:
    cfg = get_config()
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    selected_channels = cfg["selected_channel_indices"]
    arch = cfg["architecture"]
    model = WaterSegmentationModel(
        in_channels=len(selected_channels),
        encoder_name=arch["encoder_name"],
    )
    state = torch.load(str(model_path or WEIGHTS_PATH), map_location=resolved_device)
    model.load_state_dict(state)
    model.to(resolved_device)
    model.eval()
    return model, resolved_device
