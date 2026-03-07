"""
Water segmentation model: DeepChannelAdapter + UnetPlusPlus (EfficientNet-B4).
Must match the architecture used in Task 4 notebook and Task 5 training.
"""
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class DeepChannelAdapter(nn.Module):
    """
    Projects multi-channel input to 3 channels for ImageNet-pretrained encoders.
    Uses a 3-layer conv stack with a residual 1x1 shortcut.
    """

    def __init__(self, in_channels: int):
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


class WaterSegModel(nn.Module):
    def __init__(
        self,
        in_channels: int,
        encoder_name: str = "efficientnet-b4",
        pretrained: bool = True,
    ):
        super().__init__()
        self.adapter = DeepChannelAdapter(in_channels)
        weights = "imagenet" if pretrained else None
        self.unet = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=3,
            classes=1,
            activation=None,
            decoder_attention_type="scse",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unet(self.adapter(x))
