#!/usr/bin/env python3
"""
train_no_normal.py
Pipeline treningowy modelu 2D CNN trenowanego WYŁĄCZNIE na uszkodzeniach mechanicznych (Fault-Only).
Projekt: Acoustic HDD Sentinel (Eksperyment PrototypBezNormal).

Akceleracja: AMD ROCm HIP / CUDA lub CPU.
Optymalizator: AdamW + Label Smoothing + ReduceLROnPlateau.
Kalibracja: Po zakończeniu treningu automatycznie wylicza prototypy 128D i progi odrzucenia.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt

# Dodanie ścieżek
PROTOTYPE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOTYPE_DIR.parent
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_no_normal import AcousticFaultOnlyCNN
from dataset_no_normal import create_dataloaders, FAULT_CLASSES_3


def get_device() -> torch.device:
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
        print(f"[Hardware] Wykryto akcelerator GPU: {dev_name} (ROCm / CUDA)")
        return torch.device("cuda:0")
    print("[Hardware] Brak akceleracji GPU – używam CPU.")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return total_loss / total, correct / total


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            logits, _ = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)
            preds = torch.argmax(logits, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    total = len(all_targets)
    acc = sum(p == t for p, t in zip(all_preds, all_targets)) / total
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return total_loss / total, acc, macro_f1, all_preds, all_targets


def calibrate_model(model, train_loader, device, output_dir: Path):
    """Wyciąga embeddingi i kalibruje prototypy oraz progi odrzucenia."""
    print("\n[Kalibracja] Wyznaczanie prototypów metrycznych i progów detekcji 'normal'...")
    model.eval()
    all_embeds = []
    all_labels = []

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(device)
            _, norm_embed = model.extract_features(x)
            all_embeds.append(norm_embed)
            all_labels.append(y.to(device))

    all_embeds = torch.cat(all_embeds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    model.calibrate_prototypes(all_embeds, all_labels, percentile=95.0)

    glob_thresh = model.global_threshold.item()
    print(f"  -> Kalibracja zakończona pomyślnie!")
    print(f"  -> Wyznaczony globalny próg odległości (Global Threshold): {glob_thresh:.4f}")
    for i, c in enumerate(FAULT_CLASSES_3):
        print(f"     * Klasa {c:12s}: promień 95% = {model.thresholds[i].item():.4f}")

    calib_info = {
        "global_threshold": glob_thresh,
        "class_thresholds": {c: float(model.thresholds[i].item()) for i, c in enumerate(FAULT_CLASSES_3)},
        "fault_classes": FAULT_CLASSES_3,
        "embedding_dim": model.embedding_dim,
        "calibrated_samples": len(all_labels)
    }
    with open(output_dir / "calibration_config.json", "w", encoding="utf-8") as f:
        json.dump(calib_info, f, indent=2)


def run_training(
    epochs: int = 35,
    batch_size: int = 32,
    lr: float = 1e-3,
    dataset_encoded_dir: str = "dataset_encoded",
    output_dir: str = "PrototypBezNormal/output"
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = get_device()

    print("\n[Dane] Przygotowanie DataLoaderów dla Fault-Only (BEZ KLASY NORMAL W TRENINGU)...")
    train_loader, val_loader, test_loader, fault_weights = create_dataloaders(
        dataset_encoded_dir=dataset_encoded_dir,
        batch_size=batch_size
    )

    print(f"  * Próbki treningowe (tylko awarie):   {len(train_loader.dataset)}")
    print(f"  * Próbki walidacyjne (tylko awarie):  {len(val_loader.dataset)}")
    print(f"  * Próbki testowe (Z KLASĄ NORMAL!):   {len(test_loader.dataset)}")
    print(f"  * Wagi klas awaryjnych:               {fault_weights.tolist()}")

    model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(device)
    criterion = nn.CrossEntropyLoss(weight=fault_weights.to(device), label_smoothing=0.03)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_f1": []
    }

    best_val_f1 = -1.0
    best_weights_path = out_path / "best_model_no_normal.pt"

    print("\n" + "="*70)
    print(f"START TRENINGU MODELU FAULT-ONLY (EPOKI: {epochs}, BATCH: {batch_size})")
    print("="*70)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc, val_f1, _, _ = validate_epoch(model, val_loader, criterion, device)
        ep_time = time.time() - t0

        scheduler.step(val_f1)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_f1": val_f1,
                "fault_classes": FAULT_CLASSES_3
            }, best_weights_path)
            mark = "🔥 [NAJLEPSZY MODEL]"
        else:
            mark = ""

        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoka [{epoch:02d}/{epochs:02d}] ({ep_time:.1f}s) | "
              f"TrLoss: {tr_loss:.4f} TrAcc: {tr_acc*100:5.2f}% | "
              f"ValLoss: {val_loss:.4f} ValAcc: {val_acc*100:5.2f}% ValF1: {val_f1*100:5.2f}% | "
              f"LR: {curr_lr:.1e} {mark}")

    total_time = time.time() - start_time
    print("="*70)
    print(f"TRENING ZAKOŃCZONY w {total_time:.1f} s. Najlepszy Val Macro F1: {best_val_f1*100:.2f}%")

    # Wczytanie najlepszych wag do kalibracji
    print(f"\nWczytywanie najlepszych wag z {best_weights_path}...")
    checkpoint = torch.load(best_weights_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])

    # Kalibracja prototypów metrycznych na zbiorze train
    calibrate_model(model, train_loader, device, out_path)

    # Zapisanie ostatecznego modelu z zapisanymi buforami kalibracji
    torch.save({
        "epoch": checkpoint["epoch"],
        "state_dict": model.state_dict(),
        "val_f1": checkpoint["val_f1"],
        "fault_classes": FAULT_CLASSES_3,
        "eval_classes": ["beeping", "clicking", "normal", "scratching"],
        "global_threshold": model.global_threshold.item(),
        "is_calibrated": True
    }, best_weights_path)
    print(f"Zapisano w pełni wykalibrowany model do: {best_weights_path}")

    # Wykres krzywych treningowych
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Funkcja straty (Fault-Only)")
    plt.xlabel("Epoka")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot([a * 100 for a in history["train_acc"]], label="Train Acc")
    plt.plot([a * 100 for a in history["val_acc"]], label="Val Acc")
    plt.plot([f * 100 for f in history["val_f1"]], label="Val Macro F1", linestyle="--")
    plt.title("Dokładność i F1 (Fault-Only)")
    plt.xlabel("Epoka")
    plt.ylabel("Procent (%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = out_path / "training_curves_no_normal.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"Zapisano wykresy zbieżności do: {chart_path}")


if __name__ == "__main__":
    run_training()
