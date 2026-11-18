from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = ROOT / "assets"
ARTIFACTS_DIR = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"

CSS_PATH = ASSETS_DIR / "styles.css"
LOGO_PATH = ASSETS_DIR / "logo.png"
FAVICON_PATH = ASSETS_DIR / "favicon.png"
LEAF_MARK_PATH = ASSETS_DIR / "leaf_mark.png"
MODEL_PATH = MODEL_DIR / "convnext_tiny_best.pth"

PROJECT_NAME = "TomatoGuard AI"
PROJECT_TAGLINE = "AI-assisted tomato leaf diagnostics"
MODEL_NAME = "ConvNeXt-Tiny"
SOURCE_NOTEBOOK = "notebook/IA2_Tomato_HuggingFace_Simple_Colab.ipynb"
PAPER_DOI = "https://doi.org/10.3389/fpls.2024.1420584"
DATASET_URL = "https://huggingface.co/datasets/Project-AgML/tomato_leaf_disease"
COURSE_NAME = "Advanced Neural Network Architectures (2307411T)"
