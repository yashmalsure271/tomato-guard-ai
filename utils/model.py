from __future__ import annotations

import os

import streamlit as st
import torch
from torch import nn
from torchvision import models

from utils.config import MODEL_PATH
from utils.constants import CLASS_NAMES


@st.cache_resource(show_spinner=False)
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "The deployment checkpoint is missing. Run the training script first."
        )
    torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(CLASS_NAMES))
    state = torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
