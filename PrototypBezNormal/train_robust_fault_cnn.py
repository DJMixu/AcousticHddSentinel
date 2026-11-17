#!/usr/bin/env python3
"""
train_robust_fault_cnn.py
Trening bazowej sieci 2D CNN wyłącznie na 3 klasach uszkodzeń (beeping, clicking, scratching)
z wykorzystaniem zaktualizowanego zbioru danych (3 421 tensorów awarii).

Zero Data Leakage: Brak klasy 'normal' w treningu i walidacji.
Klasa 'normal' pojawi się wyłącznie podczas testowania i ewaluacji OOD.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.model_no_normal import AcousticFaultOnlyCNN
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset, compute_fault_class_weights


def train_epoch(model, loader, optimizer, class_weights, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()

        logits, _ = model(x)
        loss = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.03)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(y)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)

            total_loss += loss.item() * len(y)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
            total += len(y)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    acc = float(np.mean(all_preds == all_targets))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return total_loss / total, acc, macro_f1


def main():
    parser = argparse.ArgumentParser(description="Trening 2D CNN wyłącznie na awariach")
    parser.add_argument("--epochs", type=int, default=35, help="Liczba epok")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="PrototypBezNormal/output_v2", help="Katalog wyjściowy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie treningowe: {device}")
    if device.type == "cuda":
        print(f"Karta GPU: {torch.cuda.get_device_name(0)}")

    data_dir = PROJECT_ROOT / "dataset_encoded"
    train_ds = FaultOnlyNpyDataset(data_dir / "train", include_normal=False, augment=True)
    val_ds = FaultOnlyNpyDataset(data_dir / "validate", include_normal=False, augment=False)

    print(f"Liczba próbek treningowych (Awarie): {len(train_ds)}")
    print(f"Liczba próbek walidacyjnych (Awarie): {len(val_ds)}")

    class_weights = compute_fault_class_weights(train_ds).to(device)
    print(f"Wagi klas (beeping, clicking, scratching): {class_weights.cpu().numpy()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=4)

    best_val_f1 = 0.0
    best_state = None

    print("\nRozpoczynanie treningu...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, class_weights, device)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)
        scheduler.step(val_f1)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoka [{epoch:02d}/{args.epochs:02d}] | "
              f"TrLoss: {tr_loss:.4f} TrAcc: {tr_acc*100:.2f}% | "
              f"ValLoss: {val_loss:.4f} ValAcc: {val_acc*100:.2f}% ValF1: {val_f1*100:.2f}% | LR: {current_lr:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "fault_classes": ["beeping", "clicking", "scratching"],
                "eval_classes": ["beeping", "clicking", "normal", "scratching"]
            }

    print(f"\nNajlepszy wynik walidacyjny F1: {best_val_f1*100:.2f}%")
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])

    # Zapis modelu
    save_path = output_dir / "best_fault_cnn_v2.pt"
    torch.save(best_state, save_path)
    print(f"Zapisano wagi do: {save_path}")


if __name__ == "__main__":
    main()
