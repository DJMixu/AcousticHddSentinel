#!/usr/bin/env python3
"""
Pipeline treningowy 2D CNN z uczeniem metrycznym (ArcFace + Center Loss)
oraz kalibracją wielowymiarowej detekcji Out-Of-Distribution (OOD).

Single Source of Truth: ADR-001, ArcFace (Deng et al., 2019), Center Loss (Wen et al., 2016).
ZAKAZ: Klasa 'normal' NIE MOŻE pojawić się w procesie optymalizacji ani kalibracji progów!
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

from PrototypBezNormal.model_metric_cnn import MetricFaultOnlyCNN
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset, compute_fault_class_weights
from PrototypBezNormal.loss_functions import CenterLoss


def train_one_epoch(
    model: nn.Module,
    center_loss: nn.Module,
    loader: DataLoader,
    optimizer_model: torch.optim.Optimizer,
    optimizer_centers: torch.optim.Optimizer,
    class_weights: torch.Tensor,
    device: torch.device,
    lambda_center: float = 0.05
) -> tuple[float, float, float]:
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_center_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer_model.zero_grad()
        optimizer_centers.zero_grad()

        # Forward
        logits, norm_embed = model(x, labels=y)
        loss_ce = F.cross_entropy(logits, y, weight=class_weights, label_smoothing=0.03)
        loss_cen = center_loss(norm_embed, y)
        loss = loss_ce + lambda_center * loss_cen

        # Backward
        loss.backward()
        optimizer_model.step()

        # Zgodnie z Center Loss: aktualizacja centroidów z oddzielnym współczynnikiem uczenia
        for param in center_loss.parameters():
            if param.grad is not None:
                param.grad.data.mul_(1.0 / (lambda_center + 1e-6))
        optimizer_centers.step()

        total_loss += loss.item() * len(y)
        total_ce_loss += loss_ce.item() * len(y)
        total_center_loss += loss_cen.item() * len(y)

        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += len(y)

    return total_loss / total, correct / total, total_center_loss / total


def evaluate_faults(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_preds = []
    all_targets = []
    all_embs = []
    all_logits = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, norm_embed = model(x, labels=None)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_targets.extend(y.cpu().numpy().tolist())
            all_embs.append(norm_embed.cpu().numpy())
            all_logits.append(logits.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_embs = np.concatenate(all_embs, axis=0)
    all_logits = np.concatenate(all_logits, axis=0)

    acc = float(np.mean(all_preds == all_targets))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    return acc, macro_f1, all_embs, all_targets, all_logits


def main():
    parser = argparse.ArgumentParser(description="Trening 2D CNN z ArcFace i Center Loss (Fault-Only)")
    parser.add_argument("--epochs", type=int, default=40, help="Liczba epok treningowych")
    parser.add_argument("--batch-size", type=int, default=32, help="Rozmiar batcha")
    parser.add_argument("--lr", type=float, default=1e-3, help="Współczynnik uczenia dla modelu")
    parser.add_argument("--lr-centers", type=float, default=0.5, help="Współczynnik uczenia dla CenterLoss")
    parser.add_argument("--lambda-center", type=float, default=0.05, help="Waga CenterLoss")
    parser.add_argument("--margin", type=float, default=0.35, help="Marża kątowa ArcFace")
    parser.add_argument("--scale", type=float, default=16.0, help="Skalowanie promienia sfery ArcFace")
    parser.add_argument("--output-dir", type=str, default="PrototypBezNormal/output_metric", help="Katalog wyjściowy")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Używane urządzenie: {device}")
    if device.type == "cuda":
        print(f"Karta GPU: {torch.cuda.get_device_name(0)}")

    # 1. Zbiory danych (BEZ klasy normal w treningu i walidacji!)
    data_dir = PROJECT_ROOT / "dataset_encoded"
    train_ds = FaultOnlyNpyDataset(data_dir / "train", include_normal=False, augment=True)
    val_ds = FaultOnlyNpyDataset(data_dir / "validate", include_normal=False, augment=False)

    print(f"Liczba próbek treningowych (Awarie): {len(train_ds)}")
    print(f"Liczba próbek walidacyjnych (Awarie): {len(val_ds)}")

    class_weights = compute_fault_class_weights(train_ds).to(device)
    print(f"Wagi klas (beeping, clicking, scratching): {class_weights.cpu().numpy()}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # 2. Inicjalizacja modelu i funkcji kosztu
    model = MetricFaultOnlyCNN(
        num_fault_classes=3,
        embedding_dim=128,
        scale=args.scale,
        margin=args.margin,
        dropout_rate=0.25
    ).to(device)

    center_loss = CenterLoss(num_classes=3, feat_dim=128, device=device).to(device)

    # 3. Optymalizatory i Scheduler
    optimizer_model = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    optimizer_centers = torch.optim.SGD(center_loss.parameters(), lr=args.lr_centers)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_model, mode="max", factor=0.5, patience=5)

    best_val_f1 = 0.0
    best_checkpoint = None

    print("\nRozpoczynanie treningu metrycznego na awariach HDD...")
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_cen = train_one_epoch(
            model, center_loss, train_loader, optimizer_model, optimizer_centers,
            class_weights, device, lambda_center=args.lambda_center
        )
        val_acc, val_f1, _, _, _ = evaluate_faults(model, val_loader, device)
        scheduler.step(val_f1)

        current_lr = optimizer_model.param_groups[0]["lr"]
        print(f"Epoka [{epoch:02d}/{args.epochs:02d}] | "
              f"TrLoss: {tr_loss:.4f} (Cen: {tr_cen:.4f}) TrAcc: {tr_acc*100:.2f}% | "
              f"ValAcc: {val_acc*100:.2f}% ValF1: {val_f1*100:.2f}% | LR: {current_lr:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_checkpoint = {
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_f1": val_f1,
                "val_acc": val_acc,
                "fault_classes": ["beeping", "clicking", "scratching"],
                "scale": args.scale,
                "margin": args.margin
            }

    print(f"\nNajlepszy wynik walidacyjny F1 awarii: {best_val_f1*100:.2f}%")
    if best_checkpoint is not None:
        model.load_state_dict(best_checkpoint["state_dict"])

    # 4. Kalibracja parametrów OOD WYŁĄCZNIE na danych awarii
    print("\nKalibracja progów OOD (ArcFace Cosine, Mahalanobis, Free Energy) na zbiorze treningowym/walidacyjnym awarii...")
    # Do solidnej statystyki kowariancji i progów używamy zbioru treningowego awarii
    calib_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False)
    _, _, calib_embs, calib_labels, calib_logits = evaluate_faults(model, calib_loader, device)

    calib_summary = model.calibrate_ood_parameters(
        torch.from_numpy(calib_embs).to(device),
        torch.from_numpy(calib_labels).to(device),
        torch.from_numpy(calib_logits).to(device),
        percentile=95.0
    )

    print("Podsumowanie kalibracji OOD:")
    for k, v in calib_summary.items():
        print(f"  {k}: {v}")

    # 5. Zapis finalnego modelu i konfiguracji
    best_checkpoint["state_dict"] = model.state_dict()
    best_checkpoint["calib_summary"] = calib_summary
    model_save_path = output_dir / "best_metric_fault_model.pt"
    torch.save(best_checkpoint, model_save_path)
    print(f"\nZapisano model produkcyjny do: {model_save_path}")

    config_save_path = output_dir / "ood_calibration_config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(calib_summary, f, indent=2)
    print(f"Zapisano konfigurację progów OOD do: {config_save_path}")


if __name__ == "__main__":
    main()
