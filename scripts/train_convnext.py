"""Reproduce the notebook's ConvNeXt-Tiny training run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from utils.constants import (
    CLASS_NAMES,
    CLASS_SLUGS,
    DATASET_NAME,
    IMAGE_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEED,
)

ARTIFACTS = ROOT / "artifacts"
MODEL_DIR = ROOT / "model"
CHECKPOINTS = ROOT / "checkpoints"
MISTAKES = ROOT / "assets" / "test_mistakes"
MISTAKES.mkdir(parents=True, exist_ok=True)

class HFDataset(Dataset):
    def __init__(self, hf_dataset, transform):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[int(index)]
        image = item["image"].convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, int(item["label"])


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_model() -> nn.Module:
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features, len(CLASS_NAMES)
    )
    return model


def build_untrained_model() -> nn.Module:
    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features, len(CLASS_NAMES)
    )
    return model


def cpu_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


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


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    y_prob: list[np.ndarray] = []
    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(outputs.argmax(1).cpu().numpy().tolist())
            y_prob.extend(probabilities.cpu().numpy())
    y_true_array = np.asarray(y_true)
    y_pred_array = np.asarray(y_pred)
    y_prob_array = np.asarray(y_prob)
    metrics = {
        "Accuracy": float(accuracy_score(y_true_array, y_pred_array)),
        "Precision": float(
            precision_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
        "Recall": float(
            recall_score(y_true_array, y_pred_array, average="weighted", zero_division=0)
        ),
        "F1": float(f1_score(y_true_array, y_pred_array, average="weighted", zero_division=0)),
    }
    try:
        metrics["AUC"] = float(
            roc_auc_score(
                y_true_array, y_prob_array, multi_class="ovr", average="weighted"
            )
        )
    except ValueError:
        metrics["AUC"] = float("nan")
    return metrics, y_true_array, y_pred_array, y_prob_array


def sample_curve(fpr: np.ndarray, tpr: np.ndarray, points: int = 101):
    grid = np.linspace(0.0, 1.0, points)
    unique_fpr, unique_indices = np.unique(fpr, return_index=True)
    unique_tpr = tpr[unique_indices]
    return grid.tolist(), np.interp(grid, unique_fpr, unique_tpr).tolist()


def build_roc_payload(y_true: np.ndarray, y_prob: np.ndarray) -> dict:
    payload = {}
    for index, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true == index, y_prob[:, index])
        grid, interpolated = sample_curve(fpr, tpr)
        payload[name] = {
            "fpr": grid,
            "tpr": interpolated,
            "auc": float(auc(fpr, tpr)),
        }
    return payload


def top_confusions(matrix: np.ndarray, limit: int = 10):
    pairs = []
    for actual in range(len(CLASS_NAMES)):
        for predicted in range(len(CLASS_NAMES)):
            if actual == predicted:
                continue
            count = int(matrix[actual, predicted])
            if count:
                pairs.append(
                    {
                        "actual": CLASS_NAMES[actual],
                        "predicted": CLASS_NAMES[predicted],
                        "count": count,
                    }
                )
    pairs.sort(key=lambda item: item["count"], reverse=True)
    return pairs[:limit]


def save_mistakes(test_hf, y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray):
    for existing in MISTAKES.glob("mistake_*.png"):
        existing.unlink()
    payload = []
    for index, (actual, predicted) in enumerate(zip(y_true, y_pred, strict=True)):
        if actual == predicted or len(payload) >= 12:
            continue
        image = test_hf[index]["image"].convert("RGB")
        actual_name = CLASS_NAMES[int(actual)]
        predicted_name = CLASS_NAMES[int(predicted)]
        filename = (
            f"mistake_{len(payload) + 1:02d}_true-{CLASS_SLUGS[actual_name]}"
            f"_pred-{CLASS_SLUGS[predicted_name]}.png"
        )
        image.save(MISTAKES / filename)
        payload.append(
            {
                "file": f"assets/test_mistakes/{filename}",
                "actual": actual_name,
                "predicted": predicted_name,
                "confidence": float(y_prob[index, int(predicted)]),
            }
        )
    (ARTIFACTS / "misclassifications.json").write_text(
        json.dumps(
            {"test_size": len(y_true), "count": len(payload), "items": payload},
            indent=2,
        ),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    seed_everything(SEED)

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    print(f"device={device}", flush=True)

    raw_train = load_dataset(DATASET_NAME, split="train")
    labels = np.asarray(raw_train["label"])
    if raw_train.features["label"].names != CLASS_NAMES:
        raise SystemExit("Dataset class order differs from utils.constants")
    train_idx, val_idx, test_idx = split_indices(labels)
    print(
        f"split sizes train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}",
        flush=True,
    )

    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_hf = raw_train.select(train_idx.tolist())
    val_hf = raw_train.select(val_idx.tolist())
    test_hf = raw_train.select(test_idx.tolist())

    train_loader = DataLoader(
        HFDataset(train_hf, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
        generator=torch.Generator().manual_seed(SEED),
    )
    val_loader = DataLoader(
        HFDataset(val_hf, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=args.workers > 0,
    )
    test_loader = DataLoader(
        HFDataset(test_hf, eval_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
    )

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0
    best_state = None
    start_epoch = 0
    total_seconds = 0.0

    last_checkpoint = CHECKPOINTS / "convnext_tiny_last.pt"
    best_checkpoint = CHECKPOINTS / "convnext_tiny_best_fp32.pth"

    if args.resume and last_checkpoint.exists() and not args.fresh:
        checkpoint = torch.load(last_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_val_acc = checkpoint["best_val_acc"]
        history = checkpoint["history"]
        total_seconds = checkpoint["total_seconds"]
        if best_checkpoint.exists():
            best_state = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
        print(f"resumed from epoch {start_epoch}", flush=True)

    model.to(device)
    run_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for batch_index, (images, targets) in enumerate(train_loader):
            if args.limit_train_batches and batch_index >= args.limit_train_batches:
                break
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == targets).sum().item()
            total += targets.size(0)
            if batch_index % 25 == 0:
                print(
                    f"epoch {epoch + 1}/{args.epochs} batch {batch_index}/"
                    f"{len(train_loader)} loss={loss.item():.4f}",
                    flush=True,
                )
        train_loss = total_loss / max(total, 1)
        train_acc = correct / max(total, 1)

        model.eval()
        val_loss_total = 0.0
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for batch_index, (images, targets) in enumerate(val_loader):
                if args.limit_val_batches and batch_index >= args.limit_val_batches:
                    break
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss_total += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == targets).sum().item()
                val_total += targets.size(0)
        val_loss = val_loss_total / max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        history["train_loss"].append(float(train_loss))
        history["train_acc"].append(float(train_acc))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_state = cpu_state(model)
            torch.save(best_state, best_checkpoint)

        epoch_seconds = time.time() - epoch_start
        total_seconds += epoch_seconds
        print(
            f"epoch {epoch + 1}/{args.epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} best={best_val_acc:.4f} "
            f"time={epoch_seconds:.1f}s",
            flush=True,
        )
        torch.save(
            {
                "model": cpu_state(model),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "best_val_acc": best_val_acc,
                "history": history,
                "total_seconds": total_seconds,
            },
            last_checkpoint,
        )

    if best_state is None:
        best_state = cpu_state(model)

    trained_at = datetime.now(timezone.utc).isoformat()
    model.load_state_dict(best_state)
    fp32_metrics, fp32_true, fp32_pred, fp32_prob = evaluate(model, test_loader, device)

    fp16_state = {
        key: (
            value.detach().cpu().half()
            if value.is_floating_point()
            else value.detach().cpu()
        )
        for key, value in best_state.items()
    }
    deploy_path = MODEL_DIR / "convnext_tiny_best.pth"
    torch.save(fp16_state, deploy_path)
    size_bytes = deploy_path.stat().st_size
    print(f"saved {deploy_path} ({size_bytes / 1_000_000:.1f} MB)", flush=True)

    deployed = build_untrained_model()
    deployed.load_state_dict(torch.load(deploy_path, map_location="cpu", weights_only=True))
    deployed = deployed.to(device)
    metrics, y_true, y_pred, y_prob = evaluate(deployed, test_loader, device)

    identical = bool(np.array_equal(fp32_pred, y_pred))
    max_probability_difference = float(np.max(np.abs(fp32_prob - y_prob)))
    equivalence = {
        "identical_predictions": identical,
        "changed_predictions": int(np.sum(fp32_pred != y_pred)),
        "test_size": int(len(y_true)),
        "max_absolute_probability_difference": max_probability_difference,
        "fp32_accuracy": fp32_metrics["Accuracy"],
        "deployed_checkpoint_accuracy": metrics["Accuracy"],
        "checkpoint_storage": "float16 state_dict loaded into a float32 model",
    }
    (ARTIFACTS / "fp16_equivalence.json").write_text(
        json.dumps(equivalence, indent=2), encoding="utf-8"
    )

    matrix = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    )
    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    roc_payload = build_roc_payload(y_true, y_prob)
    top_confusion_pairs = top_confusions(matrix)
    mistakes = save_mistakes(test_hf, y_true, y_pred, y_prob)

    np.savez(
        ARTIFACTS / "test_predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        test_indices=test_idx,
    )

    metrics_payload = {
        "model": "ConvNeXt-Tiny",
        "checkpoint": str(deploy_path.relative_to(ROOT)),
        "checkpoint_size_bytes": int(size_bytes),
        "test_size": int(len(y_true)),
        "split_seed": SEED,
        "metrics": metrics,
        "per_class": {
            name: {
                key: float(report[name][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for name in CLASS_NAMES
        },
        "macro_average": {
            key: float(report["macro avg"][key])
            for key in ("precision", "recall", "f1-score", "support")
        },
        "weighted_average": {
            key: float(report["weighted avg"][key])
            for key in ("precision", "recall", "f1-score", "support")
        },
        "confusion_matrix": {
            "labels": CLASS_NAMES,
            "matrix": matrix.astype(int).tolist(),
        },
        "roc_curves": roc_payload,
        "top_confusions": top_confusion_pairs,
        "misclassifications": mistakes,
        "storage_precision": "float16 checkpoint loaded into float32 model",
        "evaluated_at": trained_at,
    }
    (ARTIFACTS / "test_metrics.json").write_text(
        json.dumps(metrics_payload, indent=2), encoding="utf-8"
    )

    history_payload = {
        "model": "ConvNeXt-Tiny",
        "config": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": "Adam",
            "loss": "CrossEntropyLoss",
            "image_size": IMAGE_SIZE,
            "seed": SEED,
            "device": str(device),
            "torch_version": torch.__version__,
            "train_transform": [
                "Resize(224, 224)",
                "RandomHorizontalFlip()",
                "RandomRotation(15)",
                "ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15)",
                "ToTensor()",
                "Normalize(ImageNet mean, ImageNet std)",
            ],
            "eval_transform": [
                "Resize(224, 224)",
                "ToTensor()",
                "Normalize(ImageNet mean, ImageNet std)",
            ],
        },
        "best_val_accuracy": float(best_val_acc),
        "training_seconds": float(total_seconds),
        "history": history,
        "split_sizes": {
            "train": int(len(train_idx)),
            "validation": int(len(val_idx)),
            "test": int(len(test_idx)),
        },
        "index_sha256": {
            "train": index_hash(train_idx),
            "validation": index_hash(val_idx),
            "test": index_hash(test_idx),
        },
        "completed_at": trained_at,
    }
    (ARTIFACTS / "training_history.json").write_text(
        json.dumps(history_payload, indent=2), encoding="utf-8"
    )

    model_card = {
        "name": "TomatoGuard ConvNeXt-Tiny",
        "architecture": "torchvision.models.convnext_tiny",
        "task": "10-class tomato leaf image classification",
        "class_names": CLASS_NAMES,
        "input": {
            "type": "RGB image",
            "resize": [IMAGE_SIZE, IMAGE_SIZE],
            "normalization_mean": list(IMAGENET_MEAN),
            "normalization_std": list(IMAGENET_STD),
            "interpolation": "torchvision Resize default (bilinear)",
        },
        "checkpoint": str(deploy_path.relative_to(ROOT)),
        "checkpoint_storage_precision": "float16",
        "checkpoint_size_bytes": int(size_bytes),
        "parameters": int(count_parameters(deployed)),
        "training": {
            "dataset": DATASET_NAME,
            "split": "stratified 70/15/15, seed 42",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "optimizer": "Adam",
            "learning_rate": args.learning_rate,
            "pretrained_initialization": "ImageNet-1K ConvNeXt-Tiny weights",
            "best_val_accuracy": float(best_val_acc),
            "training_seconds": float(total_seconds),
            "device": str(device),
            "torch_version": torch.__version__,
        },
        "test_metrics": metrics,
        "test_size": int(len(y_true)),
        "trained_at": trained_at,
        "source_notebook": "notebook/IA2_Tomato_HuggingFace_Simple_Colab.ipynb",
        "provenance": "Retrained from the original notebook pipeline because the original .pth weights were not preserved.",
    }
    (MODEL_DIR / "model_card.json").write_text(
        json.dumps(model_card, indent=2), encoding="utf-8"
    )

    print("test metrics:", json.dumps(metrics, indent=2), flush=True)
    print(f"fp16 equivalence: {equivalence}", flush=True)
    print(f"session seconds: {time.time() - run_start:.1f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
