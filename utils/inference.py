from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from utils.constants import (
    CLASS_DISPLAY,
    CLASS_KIND,
    CLASS_NAMES,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
)

MAX_PIXELS = 40_000_000
MIN_SIDE = 48


def prepare_image(image: Image.Image) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image.")
    image = ImageOps.exif_transpose(image)
    if image.width * image.height > MAX_PIXELS:
        raise ValueError("Image is too large. Please upload an image below 40 megapixels.")
    if min(image.width, image.height) < MIN_SIDE:
        raise ValueError("Image is too small. Please use an image at least 48 x 48 pixels.")
    if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


def load_uploaded(file) -> tuple[Image.Image, str]:
    payload = file.getvalue() if hasattr(file, "getvalue") else bytes(file)
    digest = hashlib.sha1(payload).hexdigest()
    image = Image.open(BytesIO(payload))
    image.load()
    return prepare_image(image), digest


_transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
)


def to_tensor(image: Image.Image) -> torch.Tensor:
    return _transform(image).unsqueeze(0)


def predict(image: Image.Image, model) -> dict:
    tensor = to_tensor(image)
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()
    order = np.argsort(probabilities)[::-1]
    top3 = [
        {
            "class_index": int(index),
            "class_name": CLASS_NAMES[int(index)],
            "display_name": CLASS_DISPLAY[CLASS_NAMES[int(index)]],
            "probability": float(probabilities[int(index)]),
            "kind": CLASS_KIND[CLASS_NAMES[int(index)]],
        }
        for index in order[:3]
    ]
    best = top3[0]
    return {
        "class_index": best["class_index"],
        "class_name": best["class_name"],
        "display_name": best["display_name"],
        "confidence": best["probability"],
        "kind": best["kind"],
        "top3": top3,
        "probabilities": [float(value) for value in probabilities],
    }
