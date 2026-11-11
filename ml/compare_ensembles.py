"""
HDD Acoustic Sentinel — Porównanie Komitetów Modeli (Ensembles) z Modelem Bazowym (PyTorch)
Single Source of Truth: ADR-001, AGENTS.md, NOTES.md

Skrypt:
1. Trenuje na GPU/CPU komitety modeli w PyTorch (Seed Ensembles z seeds=[42, 123, 777]) dla wybranych epok (np. 25, 30, 40).
2. Wczytuje i ewaluuje model bazowy (training_output/best_hdd_cnn_model.pt).
3. Porównuje wszystkie konfiguracje komitetów z modelem bazowym.
4. Generuje zbiorczy raport numeryczny (JSON, CSV) oraz wielopanelowy wykres (PNG).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from model_cnn import INPUT_SHAPE, build_hdd_cnn_model
from dataset_pytorch import (
    resolve_dataset_dir,
    discover_classes_and_files,
    create_pytorch_dataloaders,
)
from train_ensemble import run_ensemble_pipeline, evaluate_predictions


def evaluate_single_pytorch_model(
    model_path: Path,
    test_loader,
    y_true: np.ndarray,
    class_names: List[str],
    device: torch.device
) -> Dict:
    """Ewaluuje pojedynczy zapisany model PyTorch na DataLoaderze testowym."""
    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku modelu: {model_path}")

    print(f"Wczytywanie modelu bazowego: {model_path}")
    checkpoint = torch.load(model_path, weights_only=True)
    model = build_hdd_cnn_model(num_classes=len(class_names))
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    probs_list = []
    with torch.no_grad():
        for x_b, _ in test_loader:
            x_b = x_b.to(device)
            p = model(x_b, return_logits=False)
            probs_list.append(p.cpu().numpy())
    probs = np.concatenate(probs_list, axis=0)

    metrics = evaluate_predictions(y_true, probs, class_names)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Porównanie komitetów PyTorch z modelem bazowym.")
    parser.add_argument("--baseline-path", type=str, default="training_output/best_hdd_cnn_model.pt")
    args = parser.parse_args()
    ds_dir = resolve_dataset_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")
    _, _, test_loader, classes, _ = create_pytorch_dataloaders(ds_dir, batch_size=32)
    _, _, test_labels = discover_classes_and_files(ds_dir / "test", expected_classes=classes)
    y_true = np.array(test_labels)

    metrics = evaluate_single_pytorch_model(Path(args.baseline_path), test_loader, y_true, classes, device)
    print(f"Model bazowy Accuracy: {metrics['accuracy']*100:.2f}% | Weighted F1: {metrics['weighted_f1']*100:.2f}%")


if __name__ == "__main__":
    main()
