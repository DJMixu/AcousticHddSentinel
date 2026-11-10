"""
HDD Acoustic Fault Detection - CNN Training Pipeline (Pure PyTorch)
Single Source of Truth: ADR-001, AGENTS.md

Główny potok treningowy sieci 2D CNN w PyTorch z:
1. Dataset & DataLoader wczytującym tensory .npy z dataset_encoded/{train,validate,test}.
2. Augmentacją SpecAugment (Time & Frequency Masking) w locie.
3. Ważeniem klas (balanced class weights) w funkcji straty CrossEntropyLoss.
4. Mechanizmem EarlyStopping oraz ReduceLROnPlateau.
5. Pełną ewaluacją na zbiorze testowym, generowaniem macierzy pomyłek i raportów metryk.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from model_cnn import build_hdd_cnn_model, AcousticHddSentinelCNN, INPUT_SHAPE
from dataset_pytorch import resolve_dataset_dir, create_pytorch_dataloaders


def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path
):
    """Generuje i zapisuje wykres macierzy pomyłek."""
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="PyTorch CNN Confusion Matrix (Test Set)",
        ylabel="True Label (Rzeczywista klasa)",
        xlabel="Predicted Label (Przewidziana klasa)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            text_str = f"{val}\n({pct:.1f}%)" if val > 0 else "0"
            ax.text(
                j, i, text_str,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontsize=9
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Zapisano macierz pomyłek: {output_path}")


def plot_and_save_training_curves(history: Dict[str, List[float]], output_path: Path):
    """Zapisuje wykres krzywych funkcji straty i dokładności (Train vs Val)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs, history["train_loss"], "b-o", label="Train Loss")
    ax1.plot(epochs, history["val_loss"], "r--s", label="Val Loss")
    ax1.set_title("Krzywa funkcji straty (Loss)")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_acc"], "b-o", label="Train Accuracy")
    ax2.plot(epochs, history["val_acc"], "r--s", label="Val Accuracy")
    ax2.set_title("Krzywa dokładności (Accuracy)")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"      Zapisano wykresy treningowe: {output_path}")


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x_batch, return_logits=True)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)

    return running_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        logits = model(x_batch, return_logits=True)
        loss = criterion(logits, y_batch)

        running_loss += loss.item() * x_batch.size(0)
        preds = torch.argmax(logits, dim=-1)
        correct += (preds == y_batch).sum().item()
        total += x_batch.size(0)

        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(y_batch.cpu().numpy().tolist())

    return running_loss / max(total, 1), correct / max(total, 1), np.array(all_targets), np.array(all_preds)


def run_training(
    dataset_dir: Path,
    output_dir: Path,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device_name: Optional[str] = None
):
    print("=" * 70)
    print("TRENING MODELU 2D CNN W PYTORCH (ACOUSTIC HDD SENTINEL)")
    print("=" * 70)

    if device_name:
        device = torch.device(device_name)
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(device)
        gpu_props = torch.cuda.get_device_properties(device)
        vram_gb = gpu_props.total_memory / (1024 ** 3)
        print(f"[1/6] Urządzenie obliczeniowe: {device} ({gpu_name}, {vram_gb:.2f} GB VRAM)")
    else:
        print(f"[1/6] Urządzenie obliczeniowe: {device}")

    print(f"[2/6] Przygotowywanie DataLoaderów z: {dataset_dir}")
    train_loader, val_loader, test_loader, classes, class_weights = create_pytorch_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=2,
        augment_train=True
    )
    num_classes = len(classes)
    print(f"      Wykryto {num_classes} klas: {classes}")
    print(f"      Zbalansowane wagi klas: {class_weights.tolist()}")

    print(f"[3/6] Inicjalizacja architektury AcousticHddSentinelCNN...")
    model = build_hdd_cnn_model(num_classes=num_classes, dropout_rate=0.3)
    model = model.to(device)

    weights_tensor = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_path = output_dir / "best_hdd_cnn_model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[4/6] Rozpoczęcie pętli treningowej ({epochs} epok)...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        curr_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(curr_lr)

        elapsed = time.time() - t0
        print(
            f"Epoka {epoch:2d}/{epochs:2d} [{elapsed:.1f}s] - "
            f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
            f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - lr: {curr_lr:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_loss": val_loss,
                "val_acc": val_acc,
                "classes": classes,
                "num_classes": num_classes
            }
            torch.save(checkpoint, best_model_path)
            print(f"      -> Zapisano nowy najlepszy checkpoint: {best_model_path} (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"      [EarlyStopping] Brak poprawy przez {patience} epok. Zatrzymanie treningu.")
                break

    total_time = time.time() - start_time
    print(f"\nTrening zakończony w {total_time:.1f} s.")

    print(f"[5/6] Wczytywanie najlepszego modelu do ewaluacji testowej...")
    best_checkpoint = torch.load(best_model_path, weights_only=True)
    model.load_state_dict(best_checkpoint["state_dict"])

    print(f"[6/6] Finalna ewaluacja na zbiorze testowym...")
    test_loss, test_acc, y_true, y_pred = evaluate_epoch(model, test_loader, criterion, device)

    print(f"\nWyniki na zbiorze testowym:")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=classes, digits=4, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0)
    print("\n" + report_text)

    with open(output_dir / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    summary = {
        "test_metrics": {
            "loss": float(test_loss),
            "accuracy": float(test_acc)
        },
        "detailed_metrics": report,
        "classes": classes,
        "total_test_samples": int(len(y_true))
    }
    with open(output_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Zapis mapowania klas i historii epok
    classes_info = {
        "classes": classes,
        "input_shape": [128, 259, 3]
    }
    with open(output_dir / "classes.json", "w", encoding="utf-8") as f:
        json.dump(classes_info, f, indent=2)

    import pandas as pd
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    plot_and_save_confusion_matrix(y_true, y_pred, classes, output_dir / "confusion_matrix.png")
    plot_and_save_training_curves(history, output_dir / "training_curves.png")

    print(f"\n[OK] Pipeline treningowy PyTorch zakończony pomyślnie! Wyniki zapisano w: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Trening modelu 2D CNN w PyTorch dla Acoustic HDD Sentinel.")
    parser.add_argument("--epochs", type=int, default=30, help="Maksymalna liczba epok")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--lr", type=float, default=1e-3, help="Współczynnik uczenia")
    parser.add_argument("--dataset-dir", type=str, default="dataset_encoded", help="Ścieżka do dataset_encoded")
    parser.add_argument("--output-dir", type=str, default="training_output", help="Katalog wyjściowy")
    parser.add_argument("--patience", type=int, default=8, help="Cierpliwość EarlyStopping")
    args = parser.parse_args()

    ds_dir = resolve_dataset_dir(args.dataset_dir)
    out_dir = Path(args.output_dir)

    run_training(
        dataset_dir=ds_dir,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience
    )


if __name__ == "__main__":
    main()
