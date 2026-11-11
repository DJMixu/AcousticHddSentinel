"""
HDD Acoustic Sentinel — Trening i Ewaluacja Komitetu Modeli (Seed Ensembling w PyTorch)
Single Source of Truth: ADR-001, AGENTS.md, NOTES.md

Moduł implementuje:
1. Trening N modeli 2D CNN o identycznej architekturze, ale z różnymi ziarnami losowości (seed).
2. Różną inicjalizację wag, różną kolejność próbek (shuffle) oraz różne maskowania SpecAugment/Dropout.
3. Inferencję komitetową (Soft Voting): uśrednianie wektorów prawdopodobieństw Softmax.
4. Ewaluację na zbiorze testowym z generowaniem macierzy pomyłek i raportu klasyfikacyjnego.
5. Zestawienie zysku: pojedyncze modele składowe vs Komitet (Ensemble).
6. Fuzję modeli do jednego modułu PyTorch dla jednolitej inferencji.
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from model_cnn import build_hdd_cnn_model, AcousticHddSentinelCNN, INPUT_SHAPE
from dataset_pytorch import (
    resolve_dataset_dir,
    discover_classes_and_files,
    compute_balanced_class_weights,
    HddNpyDataset,
)
from train_cnn import plot_and_save_confusion_matrix, train_one_epoch, evaluate_epoch
from torch.utils.data import DataLoader


class FusedEnsemblePyTorch(nn.Module):
    """Fuzja N modeli w jeden moduł PyTorch uśredniający prawdopodobieństwa Softmax."""
    def __init__(self, models: List[nn.Module]):
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
        probs = [m(x, return_logits=False) for m in self.models]
        avg_prob = torch.stack(probs, dim=0).mean(dim=0)
        if return_logits:
            return torch.log(avg_prob + 1e-9)
        return avg_prob


def evaluate_predictions(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str]
) -> Dict:
    y_pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    per_class_f1 = {}
    for c_name in class_names:
        per_class_f1[c_name] = float(report[c_name]["f1-score"])

    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "report": report,
        "predictions": y_pred
    }


def train_single_ensemble_member(
    seed: int,
    epochs: int,
    train_files: List[Path],
    train_labels: List[int],
    val_files: List[Path],
    val_labels: List[int],
    class_names: List[str],
    class_weights: torch.Tensor,
    output_model_path: Path,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device = torch.device("cpu")
) -> nn.Module:
    print(f"\n--- Trening modelu komitetu: Seed = {seed} ---")
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_ds = HddNpyDataset(train_files, train_labels, is_training=True, augment=True)
    val_ds = HddNpyDataset(val_files, val_labels, is_training=False, augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    model = build_hdd_cnn_model(num_classes=len(class_names), dropout_rate=0.3).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    output_model_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_model_path)

    # Załaduj najlepsze wagi
    model.load_state_dict(torch.load(output_model_path, weights_only=True))
    model.eval()
    print(f"  -> Zapisano model: {output_model_path} (min val_loss: {best_val_loss:.4f})")
    return model


def run_ensemble_pipeline(
    dataset_dir: Path,
    output_dir: Path,
    seeds: List[int] = [42, 123, 777],
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3
):
    print("=" * 70)
    print("TRENING KOMITETU MODELI 2D CNN (PYTORCH SEED ENSEMBLE)")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "validate"
    test_dir = dataset_dir / "test"

    classes, train_files, train_labels = discover_classes_and_files(train_dir)
    _, val_files, val_labels = discover_classes_and_files(val_dir, expected_classes=classes)
    _, test_files, test_labels = discover_classes_and_files(test_dir, expected_classes=classes)

    class_weights = compute_balanced_class_weights(train_labels, num_classes=len(classes))

    models_dir = output_dir / "models"
    trained_models = []
    y_test = np.array(test_labels)

    # Wczytanie próbek testowych do pamięci
    X_test_np = np.stack([np.load(f).astype(np.float32) for f in test_files], axis=0)
    X_test_t = torch.from_numpy(X_test_np).to(device)

    individual_metrics = {}
    model_probabilities = []

    for seed in seeds:
        m_path = models_dir / f"ensemble_model_seed_{seed}.pt"
        model = train_single_ensemble_member(
            seed=seed,
            epochs=epochs,
            train_files=train_files,
            train_labels=train_labels,
            val_files=val_files,
            val_labels=val_labels,
            class_names=classes,
            class_weights=class_weights,
            output_model_path=m_path,
            batch_size=batch_size,
            lr=lr,
            device=device
        )
        trained_models.append(model)

        with torch.no_grad():
            probs = model(X_test_t, return_logits=False).cpu().numpy()
        model_probabilities.append(probs)

        metrics = evaluate_predictions(y_test, probs, classes)
        individual_metrics[f"Seed {seed}"] = metrics
        print(f"  [Seed {seed}] Test Accuracy: {metrics['accuracy']:.4f} | Weighted F1: {metrics['weighted_f1']:.4f}")

    # Soft Voting
    ensemble_probs = np.mean(model_probabilities, axis=0)
    ensemble_metrics = evaluate_predictions(y_test, ensemble_probs, classes)

    # Zapis zfuzowanego modelu
    fused_model = FusedEnsemblePyTorch(trained_models)
    fused_path = models_dir / "fused_ensemble_model.pt"
    torch.save(fused_model.state_dict(), fused_path)
    print(f"\n[OK] Zfuzowany model komitetu zapisany do: {fused_path}")

    # Podsumowanie
    print("\n" + "=" * 70)
    print("WYNIKI KOMITETU VS MODELE INDYWIDUALNE")
    print("=" * 70)
    print(f"{'Model':<15} | {'Accuracy':<10} | {'Weighted F1':<12} | {'Macro F1':<10}")
    print("-" * 55)
    for name, m in individual_metrics.items():
        print(f"{name:<15} | {m['accuracy']*100:6.2f}%    | {m['weighted_f1']*100:8.2f}%   | {m['macro_f1']*100:6.2f}%")
    print("-" * 55)
    print(f"{'KOMITET':<15} | {ensemble_metrics['accuracy']*100:6.2f}%    | {ensemble_metrics['weighted_f1']*100:8.2f}%   | {ensemble_metrics['macro_f1']*100:6.2f}%")
    print("=" * 70)

    plot_and_save_confusion_matrix(
        y_test, ensemble_metrics["predictions"], classes, output_dir / "ensemble_confusion_matrix.png"
    )

    summary = {
        "seeds": seeds,
        "classes": classes,
        "individual_metrics": {k: {m: v for m, v in val.items() if m != "predictions"} for k, val in individual_metrics.items()},
        "ensemble_metrics": {m: v for m, v in ensemble_metrics.items() if m != "predictions"}
    }
    with open(output_dir / "ensemble_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Trening komitetu modeli w PyTorch.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--output-dir", type=str, default="ensemble_output_pytorch")
    args = parser.parse_args()

    ds_dir = resolve_dataset_dir()
    run_ensemble_pipeline(
        dataset_dir=ds_dir,
        output_dir=Path(args.output_dir),
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )


if __name__ == "__main__":
    main()
