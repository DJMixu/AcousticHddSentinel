#!/usr/bin/env python3
"""
train_smpc.py
Pipeline treningowy modelu AcousticSMPCModel (Fault-Only Sub-Center ArcFace).

Single Source of Truth: ADR-001, AGENTS.md, PROJECT.md, SCOPE.md
Niezmiennik Zero Data Leakage:
- Zbiór train (3 421 próbek): wyłącznie klasy beeping, clicking, scratching. Zero próbek normal.
- Zbiór validate (131 próbek): wyłącznie klasy beeping, clicking, scratching. Zero próbek normal.
- Model uczy się zwartej reprezentacji awarii z M=2 sub-centrami per klasa i regularyzacją ortogonalności.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Zapewnienie dostępu do głównego katalogu repozytorium w sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PrototypBezNormal.dataset_no_normal import (
    FaultOnlyNpyDataset,
    FAULT_CLASSES_3,
    compute_fault_class_weights
)
from PrototypBezNormal.model_smpc import AcousticSMPCModel
from PrototypBezNormal.loss_smpc import SubCenterArcMarginLoss, compute_orthogonality_loss


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser(description="Trening modelu AcousticSMPCModel (Fault-Only)")
    parser.add_argument("--dataset-dir", type=str, default="dataset_encoded", help="Ścieżka do dataset_encoded")
    parser.add_argument("--output-dir", type=str, default="PrototypBezNormal/output_smpc", help="Katalog wyjściowy")
    parser.add_argument("--epochs", type=int, default=35, help="Liczba epok treningowych")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--lr", type=float, default=1e-3, help="Współczynnik uczenia AdamW")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Współczynnik weight decay")
    parser.add_argument("--scale", type=float, default=16.0, help="Skala s w ArcFace")
    parser.add_argument("--margin", type=float, default=0.20, help="Margines kątowy m w ArcFace")
    parser.add_argument("--ortho-lambda", type=float, default=0.10, help="Waga straty ortogonalności sub-centrów")
    parser.add_argument("--ortho-threshold", type=float, default=0.30, help="Próg iloczynu skalarnego sub-centrów")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Wymiar embeddingu sferycznego")
    parser.add_argument("--sub-centers", type=int, default=2, help="Liczba sub-centrów per klasa awarii")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowości")
    return parser.parse_args()


def train():
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"===========================================================")
    print(f" Trening SMPC-OOD (Sub-Center Prototype Constrained Margin)")
    print(f" Urządzenie: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f" Epoki: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f" ArcFace scale={args.scale}, margin={args.margin}, ortho_lambda={args.ortho_lambda}")
    print(f"===========================================================")

    # Wczytanie zbiorów danych z bezwzględnym wykluczeniem klasy normal
    data_root = Path(args.dataset_dir)
    train_ds = FaultOnlyNpyDataset(
        data_root / "train",
        include_normal=False,
        augment=True,
        preload_to_ram=True
    )
    val_ds = FaultOnlyNpyDataset(
        data_root / "validate",
        include_normal=False,
        augment=False,
        preload_to_ram=True
    )

    # Niezmiennik braku wycieku danych (Zero Data Leakage Invariant)
    assert all(s[2] != "normal" for s in train_ds.samples), "BŁĄD KRYTYCZNY: Wykryto próbkę 'normal' w train_ds!"
    assert all(s[2] != "normal" for s in val_ds.samples), "BŁĄD KRYTYCZNY: Wykryto próbkę 'normal' w val_ds!"
    assert all(s[2] in FAULT_CLASSES_3 for s in train_ds.samples)
    assert all(s[2] in FAULT_CLASSES_3 for s in val_ds.samples)

    print(f"Załadowano dane treningowe awarii: {len(train_ds)} tensorów (0 normal)")
    print(f"Załadowano dane walidacyjne awarii: {len(val_ds)} tensorów (0 normal)")

    fault_weights = compute_fault_class_weights(train_ds).to(device)
    print(f"Wagi klas awarii (Inverse Frequency): {fault_weights.cpu().numpy()}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda")
    )

    # Inicjalizacja modelu
    model = AcousticSMPCModel(
        num_fault_classes=len(FAULT_CLASSES_3),
        embedding_dim=args.embedding_dim,
        sub_centers=args.sub_centers,
        scale=args.scale
    ).to(device)

    # Funkcja straty ArcFace z wagami klas i karą ortogonalności
    criterion = SubCenterArcMarginLoss(
        scale=args.scale,
        margin=args.margin,
        easy_margin=False,
        class_weights=fault_weights,
        label_smoothing=0.02
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4
    )

    best_val_f1 = -1.0
    best_epoch = 0
    best_model_path = output_dir / "best_smpc_model.pt"

    history = {
        "train_loss": [],
        "train_margin_loss": [],
        "train_ortho_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_macro_f1": [],
        "per_class_f1": []
    }

    start_total_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_margin_loss = 0.0
        total_ortho_loss = 0.0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad()
            _, norm_embeddings = model(batch_x)
            cos_sims = model.compute_cosine_similarities(norm_embeddings)

            margin_loss = criterion(cos_sims, batch_y)
            ortho_loss = compute_orthogonality_loss(model.prototypes, threshold=args.ortho_threshold)
            loss = margin_loss + args.ortho_lambda * ortho_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            total_margin_loss += margin_loss.item()
            total_ortho_loss += ortho_loss.item()
            num_batches += 1

        avg_train_loss = total_loss / max(1, num_batches)
        avg_margin_loss = total_margin_loss / max(1, num_batches)
        avg_ortho_loss = total_ortho_loss / max(1, num_batches)

        # Walidacja na czystych 131 próbkach awarii
        model.eval()
        val_targets = []
        val_preds = []
        val_loss_total = 0.0
        val_batches = 0

        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x = val_x.to(device, non_blocking=True)
                val_y = val_y.to(device, non_blocking=True)

                logits, norm_embed = model(val_x)
                cos_sims = model.compute_cosine_similarities(norm_embed)
                v_margin_loss = criterion(cos_sims, val_y)
                v_ortho_loss = compute_orthogonality_loss(model.prototypes, threshold=args.ortho_threshold)
                v_loss = v_margin_loss + args.ortho_lambda * v_ortho_loss

                val_loss_total += v_loss.item()
                val_batches += 1

                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy().tolist())
                val_targets.extend(val_y.cpu().numpy().tolist())

        avg_val_loss = val_loss_total / max(1, val_batches)
        val_acc = accuracy_score(val_targets, val_preds)
        val_macro_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        class_f1s = f1_score(val_targets, val_preds, average=None, zero_division=0)

        scheduler.step(val_macro_f1)

        history["train_loss"].append(avg_train_loss)
        history["train_margin_loss"].append(avg_margin_loss)
        history["train_ortho_loss"].append(avg_ortho_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_accuracy"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["per_class_f1"].append([float(f) for f in class_f1s])

        epoch_duration = time.time() - epoch_start
        print(
            f"Epoka [{epoch:02d}/{args.epochs:02d}] ({epoch_duration:.1f}s) | "
            f"Train Loss: {avg_train_loss:.4f} (Margin: {avg_margin_loss:.4f}, Ortho: {avg_ortho_loss:.4f}) | "
            f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro F1: {val_macro_f1*100:.2f}% | "
            f"F1s: [beep:{class_f1s[0]*100:.1f}%, click:{class_f1s[1]*100:.1f}%, scratch:{class_f1s[2]*100:.1f}%]"
        )

        # Zapis najlepszego modelu na podstawie Validation Macro F1
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": float(val_macro_f1),
                    "val_accuracy": float(val_acc),
                    "num_classes": len(FAULT_CLASSES_3),
                    "sub_centers": args.sub_centers,
                    "embedding_dim": args.embedding_dim,
                    "classes": FAULT_CLASSES_3,
                    "scale": args.scale,
                    "margin": args.margin,
                    "history": history
                },
                best_model_path
            )
            print(f"  >>> Zapisano nowy najlepszy model do: {best_model_path} (Val Macro F1 = {best_val_f1*100:.2f}%)")

    total_time = time.time() - start_total_time
    print(f"\nTrening zakończony w {total_time:.1f}s! Najlepsza epoka: {best_epoch} z Val Macro F1: {best_val_f1*100:.2f}%")

    # Zapis historii do JSON
    history_path = output_dir / "training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Zapisano historię treningu do: {history_path}")

    # Wygenerowanie wykresów treningowych
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(range(1, args.epochs + 1), history["train_loss"], label="Train Total Loss", color="#1f77b4")
    plt.plot(range(1, args.epochs + 1), history["train_margin_loss"], label="Train Margin Loss", linestyle="--", color="#aec7e8")
    plt.plot(range(1, args.epochs + 1), history["val_loss"], label="Val Loss", color="#ff7f0e")
    plt.xlabel("Epoka")
    plt.ylabel("Funkcja straty")
    plt.title("Krzywe funkcji straty (ArcFace + Ortho)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(range(1, args.epochs + 1), [v * 100 for v in history["val_accuracy"]], label="Val Accuracy (%)", color="#2ca02c")
    plt.plot(range(1, args.epochs + 1), [v * 100 for v in history["val_macro_f1"]], label="Val Macro F1 (%)", color="#d62728")
    plt.xlabel("Epoka")
    plt.ylabel("Skuteczność (%)")
    plt.title(f"Dokładność i Macro F1 na walidacji (Best: {best_val_f1*100:.2f}%)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    curves_path = output_dir / "smpc_training_curves.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Zapisano wykresy treningowe do: {curves_path}")


if __name__ == "__main__":
    train()
