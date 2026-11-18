"""Shared constants for the tomato leaf disease classifier."""

from __future__ import annotations

DATASET_NAME = "Project-AgML/tomato_leaf_disease"
SEED = 42
IMAGE_SIZE = 224
NUM_CLASSES = 10

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CLASS_NAMES = [
    "Bacterial Spot",
    "Early Blight",
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites Two-spotted Spider Mite",
    "Target Spot",
    "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
]

CLASS_DISPLAY = {
    "Bacterial Spot": "Bacterial Spot",
    "Early Blight": "Early Blight",
    "Healthy": "Healthy",
    "Late Blight": "Late Blight",
    "Leaf Mold": "Leaf Mold",
    "Septoria Leaf Spot": "Septoria Leaf Spot",
    "Spider Mites Two-spotted Spider Mite": "Spider Mites (Two-spotted Spider Mite)",
    "Target Spot": "Target Spot",
    "Tomato Mosaic Virus": "Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus": "Tomato Yellow Leaf Curl Virus",
}

CLASS_SLUGS = {
    "Bacterial Spot": "bacterial_spot",
    "Early Blight": "early_blight",
    "Healthy": "healthy",
    "Late Blight": "late_blight",
    "Leaf Mold": "leaf_mold",
    "Septoria Leaf Spot": "septoria_leaf_spot",
    "Spider Mites Two-spotted Spider Mite": "spider_mites",
    "Target Spot": "target_spot",
    "Tomato Mosaic Virus": "tomato_mosaic_virus",
    "Tomato Yellow Leaf Curl Virus": "tomato_yellow_leaf_curl_virus",
}

CLASS_KIND = {
    "Bacterial Spot": "disease",
    "Early Blight": "disease",
    "Healthy": "healthy",
    "Late Blight": "disease",
    "Leaf Mold": "disease",
    "Septoria Leaf Spot": "disease",
    "Spider Mites Two-spotted Spider Mite": "pest",
    "Target Spot": "disease",
    "Tomato Mosaic Virus": "disease",
    "Tomato Yellow Leaf Curl Virus": "disease",
}

DISPLAY_TO_CLASS = {v: k for k, v in CLASS_DISPLAY.items()}
