"""Rebuild dataset statistics, sample images and original notebook evidence."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from utils.constants import CLASS_NAMES, CLASS_SLUGS, DATASET_NAME, SEED

ARTIFACTS = ROOT / "artifacts"
ASSETS = ROOT / "assets"
SAMPLES = ASSETS / "samples"
ORIGINAL = ASSETS / "original_run"
NOTEBOOK = ROOT / "notebook" / "IA2_Tomato_HuggingFace_Simple_Colab.ipynb"
MODEL_ORDER = ["LDAMNet", "ResNet50", "EfficientNetV2-S", "ConvNeXt-Tiny"]


def split_indices(labels: np.ndarray):
    all_indices = np.arange(len(labels))
    train_idx, temp_idx = train_test_split(
        all_indices, test_size=0.30, random_state=SEED, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, random_state=SEED, stratify=labels[temp_idx]
    )
    return train_idx, val_idx, test_idx


def index_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def extract_notebook_assets() -> dict:
    if not NOTEBOOK.exists():
        return {}
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cells = notebook["cells"]
    ORIGINAL.mkdir(parents=True, exist_ok=True)

    image_targets = {
        5: "class_distribution.png",
        10: "sample_batch.png",
        25: "training_curves.png",
        27: "confusion_matrices.png",
    }
    for cell_index, prefix in image_targets.items():
        images = []
        for output in cells[cell_index].get("outputs", []):
            png = output.get("data", {}).get("image/png")
            if png:
                raw = png if isinstance(png, str) else "".join(png)
                images.append(base64.b64decode(raw))
        for position, payload in enumerate(images):
            if prefix == "training_curves.png" and len(images) == 4:
                name = f"training_curves_{MODEL_ORDER[position].replace('-', '_')}.png"
            elif prefix == "confusion_matrices.png" and len(images) == 4:
                name = f"confusion_matrix_{MODEL_ORDER[position].replace('-', '_')}.png"
            else:
                name = prefix
            (ORIGINAL / name).write_bytes(payload)

    comparison = []
    for output in cells[26].get("outputs", []):
        html = output.get("data", {}).get("text/html")
        if html:
            raw = html if isinstance(html, str) else "".join(html)
            table = pd.read_html(StringIO(raw))[0]
            if "Unnamed: 0" in table.columns:
                table = table.drop(columns=["Unnamed: 0"])
            comparison = table.to_dict(orient="records")
            break

    stream_text = ""
    for output in cells[24].get("outputs", []):
        if output.get("output_type") == "stream":
            stream_text += "".join(output.get("text", []))

    history: dict[str, dict[str, list[float]]] = {}
    test_metrics: dict[str, dict[str, float]] = {}
    current_model = None
    for line in stream_text.splitlines():
        line = line.strip()
        epoch_match = re.match(
            r"^(LDAMNet|ResNet50|EfficientNetV2-S|ConvNeXt-Tiny) \| Epoch (\d+)/10 \| "
            r"Train Acc: ([0-9.]+) \| Val Acc: ([0-9.]+)$",
            line,
        )
        if epoch_match:
            current_model = epoch_match.group(1)
            history.setdefault(current_model, {"train_acc": [], "val_acc": []})
            history[current_model]["train_acc"].append(float(epoch_match.group(3)))
            history[current_model]["val_acc"].append(float(epoch_match.group(4)))
            continue
        metric_match = re.match(
            r"^(Accuracy|Precision|Recall|F1|AUC): ([0-9.]+)$", line
        )
        if metric_match and current_model:
            test_metrics.setdefault(current_model, {})[metric_match.group(1)] = float(
                metric_match.group(2)
            )

    reports_text = ""
    for output in cells[28].get("outputs", []):
        if output.get("output_type") == "stream":
            reports_text += "".join(output.get("text", []))
    reports: dict[str, str] = {}
    for block in re.split(r"\n={80}\n", reports_text):
        block = block.strip()
        if not block:
            continue
        report_match = re.match(r"^(LDAMNet|ResNet50|EfficientNetV2-S|ConvNeXt-Tiny)\n", block)
        if report_match:
            reports[report_match.group(1)] = block

    payload = {
        "source": "Saved outputs embedded in the original Google Colab notebook",
        "source_notebook": str(NOTEBOOK.relative_to(ROOT)),
        "model_order": MODEL_ORDER,
        "comparison": comparison,
        "training_history": history,
        "test_metrics": test_metrics,
        "classification_reports": reports,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACTS / "original_run.json").write_text(
        json.dumps(payload, indent=2, default=lambda value: value.item()), encoding="utf-8"
    )
    return payload


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SAMPLES.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(DATASET_NAME, split="train")
    labels = np.asarray(dataset["label"])
    class_names = dataset.features["label"].names
    if class_names != CLASS_NAMES:
        raise SystemExit(f"Unexpected class order: {class_names}")

    counts = np.bincount(labels, minlength=len(CLASS_NAMES))
    train_idx, val_idx, test_idx = split_indices(labels)

    sizes: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    chosen: dict[str, int] = {}
    for position in range(len(dataset)):
        image = dataset[position]["image"]
        sizes[image.size] += 1
        modes[image.mode] += 1
        label = int(dataset[position]["label"])
        name = CLASS_NAMES[label]
        if name not in chosen:
            chosen[name] = position
            image.save(SAMPLES / f"{CLASS_SLUGS[name]}.png")
        if position % 2000 == 0:
            print(f"scanned {position}/{len(dataset)} images", flush=True)

    stats = {
        "dataset_name": DATASET_NAME,
        "source_url": "https://huggingface.co/datasets/Project-AgML/tomato_leaf_disease",
        "total_images": len(dataset),
        "num_classes": len(CLASS_NAMES),
        "class_names": CLASS_NAMES,
        "class_counts": {
            name: int(count) for name, count in zip(CLASS_NAMES, counts, strict=True)
        },
        "splits": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
            "strategy": "Stratified 70/15/15 split of the single Hugging Face train split",
            "seed": SEED,
        },
        "index_sha256": {
            "train": index_hash(train_idx),
            "validation": index_hash(val_idx),
            "test": index_hash(test_idx),
        },
        "image_sizes": {
            f"{width}x{height}": count for (width, height), count in sizes.most_common()
        },
        "image_modes": dict(modes),
        "features": ["image", "label"],
        "label_type": "10-class single-label classification",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    (ARTIFACTS / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    extract_notebook_assets()
    print("Prepared dataset stats, sample images and notebook evidence.")


if __name__ == "__main__":
    main()
