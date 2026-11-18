from __future__ import annotations

import numpy as np
import torch
from PIL import Image

from utils.inference import to_tensor

_COLOR_STOPS = np.array(
    [
        [30, 58, 138],
        [6, 182, 212],
        [132, 204, 22],
        [250, 204, 21],
        [239, 68, 68],
    ],
    dtype=np.float32,
) / 255.0


def _colorize(cam: np.ndarray) -> Image.Image:
    positions = np.linspace(0.0, 1.0, len(_COLOR_STOPS))
    channels = [
        np.interp(cam, positions, _COLOR_STOPS[:, channel]) for channel in range(3)
    ]
    rgb = (np.stack(channels, axis=-1) * 255).astype(np.uint8)
    return Image.fromarray(rgb, mode="RGB")


def generate_gradcam(
    model,
    image: Image.Image,
    class_index: int,
    max_display_size: int = 1200,
) -> tuple[Image.Image, Image.Image] | None:
    tensor = to_tensor(image).requires_grad_(True)
    activations: dict[str, object] = {}

    def capture(_module, _inputs, output):
        output.retain_grad()
        activations["features"] = output

    handle = model.features.register_forward_hook(capture)
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            logits = model(tensor)
            score = logits[0, class_index]
            score.backward()
        features = activations.get("features")
        if features is None or features.grad is None:
            return None
        gradients = features.grad
        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * features).sum(dim=1)[0]
        cam = cam.relu()
        cam = cam - cam.min()
        maximum = cam.max()
        if float(maximum) > 0:
            cam = cam / maximum
        cam_array = cam.detach().cpu().numpy()
    finally:
        handle.remove()

    base = image.copy()
    base.thumbnail((max_display_size, max_display_size), Image.LANCZOS)
    cam_image = Image.fromarray((cam_array * 255).astype(np.uint8), mode="L")
    cam_image = cam_image.resize(base.size, Image.BILINEAR)
    cam_scaled = np.asarray(cam_image, dtype=np.float32) / 255.0
    heatmap = _colorize(cam_scaled)
    overlay = Image.blend(base.convert("RGB"), heatmap, alpha=0.48)
    return overlay, heatmap
