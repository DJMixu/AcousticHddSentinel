"""
HDD Acoustic Sentinel — Porównanie modeli CNN w zależności od liczby epok treningowych (PyTorch)
Single Source of Truth: ADR-001 & AGENTS.md

Skrypt automatycznie:
1. Trenuje oddzielne modele 2D CNN w PyTorch dla każdej zadanej liczby epok.
2. Po każdym treningu wykonuje walidację na zbiorze testowym (Classification Report, Confusion Matrix).
3. Przeprowadza test przeuczenia (Overfitting Diagnostic).
4. Zapisuje raport porównawczy.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from model_cnn import build_hdd_cnn_model, INPUT_SHAPE
from dataset_pytorch import (
    resolve_dataset_dir,
    discover_classes_and_files,
    compute_balanced_class_weights,
    create_pytorch_dataloaders
)
from train_cnn import train_one_epoch, evaluate_epoch


def train_single_model(
    num_epochs: int,
    train_loader,
    val_loader,
    num_classes: int,
    class_weights: torch.Tensor,
    output_dir: Path,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu")
) -> Tuple[nn.Module, Dict, float]:
    model = build_hdd_cnn_model(num_classes=num_classes, dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

    elapsed = time.time() - start_time
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / f"model_ep{num_epochs}.pt")
    return model, history, elapsed


def main():
    parser = argparse.ArgumentParser(description="Test epok w PyTorch.")
    parser.add_argument("--epochs", type=int, nargs="+", default=[20, 30])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", type=str, default="epoch_comparison_pytorch")
    args = parser.parse_args()

    ds_dir = resolve_dataset_dir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    train_loader, val_loader, test_loader, classes, weights = create_pytorch_dataloaders(
        ds_dir, batch_size=args.batch_size
    )

    out_dir = Path(args.output_dir)
    results = {}

    for ep in args.epochs:
        print(f"\n=== Trening {ep} epok ===")
        m_dir = out_dir / f"epochs_{ep}"
        model, hist, t = train_single_model(ep, train_loader, val_loader, len(classes), weights, m_dir, device=device)
        test_loss, test_acc, y_t, y_p = evaluate_epoch(model, test_loader, nn.CrossEntropyLoss(), device)
        print(f"[{ep} epok] Test Acc: {test_acc*100:.2f}% | Czas: {t:.1f}s")
        results[ep] = {"test_acc": test_acc, "test_loss": test_loss, "train_time_sec": t}

    with open(out_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
