import os
import tempfile
from pathlib import Path
import base64

import cv2
import numpy as np
import torch
import torch.nn as nn
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from torchvision.models import video as video_models


class VideoSwinClassifier(nn.Module):
    """
    Video Swin Transformer (Swin-T) pretrained on Kinetics-400,
    adapted for binary video classification.

    Input shape : (B, 3, T, H, W)
    Output shape: (B,) — raw logits (apply sigmoid for probability)
    """

    def __init__(self, dropout: float = 0.5, freeze_backbone: bool = True):
        super().__init__()
        from torchvision.models.video import Swin3D_T_Weights

        weights = Swin3D_T_Weights.KINETICS400_V1
        self.backbone = video_models.swin3d_t(weights=weights)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        in_features = self.backbone.head.in_features
        self.backbone.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def sample_frames_uniform(video_path, num_frames, size):
    """
    Uniformly sample `num_frames` from a video file.
    Returns np.ndarray (num_frames, H, W, 3), float32, [0, 1]
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return np.zeros((num_frames, size[1], size[0], 3), dtype=np.float32)

    if total_frames >= num_frames:
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        indices = np.array([i % total_frames for i in range(num_frames)])

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            frame = (
                frames[-1]
                if frames
                else np.zeros((size[1], size[0], 3), dtype=np.uint8)
            )
        else:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, size)
        frames.append(frame)

    cap.release()
    return np.stack(frames, axis=0).astype(np.float32) / 255.0


_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_FRAMES = 32
FRAME_SIZE = (112, 112)


def get_model():
    global _model
    if _model is None:
        ckpt_path = Path(settings.MODEL_CHECKPOINT_PATH)
        model = VideoSwinClassifier(dropout=0.5, freeze_backbone=True)
        state = torch.load(ckpt_path, map_location=_device)
        if any(k.startswith("module.") for k in state.keys()):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
        model.load_state_dict(state)
        model.to(_device)
        model.eval()
        _model = model
    return _model


def index(request):
    return render(request, "index.html")


@csrf_exempt
def preview_frames_view(request):
    if request.method != "POST" or "video" not in request.FILES:
        return JsonResponse({"error": "POST a file as 'video'."}, status=400)

    video_file = request.FILES["video"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        for chunk in video_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        # Sample frames at the same resolution as the model input
        frames = sample_frames_uniform(tmp_path, NUM_FRAMES, FRAME_SIZE)
        # Pick a small number of evenly spaced frames for preview
        num_preview = 6
        indices = np.linspace(0, NUM_FRAMES - 1, num_preview, dtype=int)
        thumbs = []
        for i in indices:
            frame = frames[i]  # (H, W, 3), float32 [0,1]
            frame_uint8 = (np.clip(frame, 0.0, 1.0) * 255).astype(np.uint8)
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(frame_uint8, cv2.COLOR_RGB2BGR))
            if not ok:
                continue
            b64 = base64.b64encode(buf.tobytes()).decode("ascii")
            thumbs.append(f"data:image/jpeg;base64,{b64}")

        return JsonResponse({"thumbnails": thumbs})
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


@csrf_exempt
def predict_view(request):
    if request.method != "POST" or "video" not in request.FILES:
        return JsonResponse({"error": "POST a file as 'video'."}, status=400)

    video_file = request.FILES["video"]

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        for chunk in video_file.chunks():
            tmp.write(chunk)
        tmp_path = tmp.name

    try:
        frames = sample_frames_uniform(tmp_path, NUM_FRAMES, FRAME_SIZE)
        frames_t = torch.from_numpy(frames)
        tensor = frames_t.permute(3, 0, 1, 2).float()
        tensor = tensor.unsqueeze(0).to(_device)

        model = get_model()
        with torch.no_grad():
            logits = model(tensor)
            prob = torch.sigmoid(logits)[0].item()

        threshold = 0.5
        is_shoplifting = prob >= threshold

        return JsonResponse(
            {
                "probability": prob,
                "is_shoplifting": bool(is_shoplifting),
                "threshold": threshold,
            }
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

