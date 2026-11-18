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
import math
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from model_cnn import build_hdd_cnn_model, AcousticHddSentinelCNN, INPUT_SHAPE
from dataset_pytorch import resolve_dataset_dir, create_pytorch_dataloaders


class SubCenterArcMarginLoss(nn.Module):
    """
    Sub-Center Additive Angular Margin Loss (ArcFace).
    cos(theta_{y_i} + m) = cos(theta) * cos(m) - sin(theta) * sin(m)
    """

    def __init__(
        self,
        scale: float = 16.0,
        margin: float = 0.25,
        easy_margin: bool = False,
        class_weights: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.register_buffer("class_weights", class_weights if class_weights is not None else None)
        self.label_smoothing = label_smoothing

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, cosine_similarities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if cosine_similarities.max() > 1.05 or cosine_similarities.min() < -1.05:
            cosine = cosine_similarities / self.scale
        else:
            cosine = cosine_similarities

        eps = 1e-7
        cosine_clamped = torch.clamp(cosine, -1.0 + eps, 1.0 - eps)
        sine = torch.sqrt(1.0 - torch.pow(cosine_clamped, 2))

        phi = cosine_clamped * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine_clamped > 0, phi, cosine_clamped)
        else:
            phi = torch.where(cosine_clamped > self.th, phi, cosine_clamped - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, targets.view(-1, 1).long(), 1.0)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        weights = self.class_weights if self.class_weights is not None else None
        loss = F.cross_entropy(output, targets, weight=weights, label_smoothing=self.label_smoothing)
        return loss


def compute_orthogonality_loss(prototypes_weight: torch.Tensor, threshold: float = 0.30) -> torch.Tensor:
    """
    Wylicza karę ortogonalności sub-centrów:
        L_ortho = sum_{k=0}^{K-1} sum_{m1 < m2} max(0, W_{k,m1} · W_{k,m2} - threshold)
    na znormalizowanych sferycznie prototypach (K, M, D).
    """
    w_norm = F.normalize(prototypes_weight, p=2, dim=-1)  # (K, M, D)
    K, M, D = w_norm.shape

    if M < 2:
        return torch.tensor(0.0, device=prototypes_weight.device, dtype=prototypes_weight.dtype)

    if M == 2:
        w1 = w_norm[:, 0, :]
        w2 = w_norm[:, 1, :]
        dot_prods = (w1 * w2).sum(dim=-1)
        penalties = torch.clamp(dot_prods - threshold, min=0.0)
        return penalties.sum()

    total_penalty = torch.tensor(0.0, device=prototypes_weight.device, dtype=prototypes_weight.dtype)
    for k in range(K):
        for m1 in range(M):
            for m2 in range(m1 + 1, M):
                dot_prod = torch.dot(w_norm[k, m1], w_norm[k, m2])
                total_penalty = total_penalty + torch.clamp(dot_prod - threshold, min=0.0)

    return total_penalty


def spherical_kmeans(features: torch.Tensor, k: int, max_iters: int = 50) -> torch.Tensor:
    """Sferyczny K-Means na hipersferze S^{D-1}."""
    n, d = features.shape
    if n <= k:
        centroids = F.normalize(features, p=2, dim=-1)
        if n < k:
            pad = torch.randn(k - n, d, device=features.device)
            centroids = torch.cat([centroids, F.normalize(pad, p=2, dim=-1)], dim=0)
        return centroids

    indices = torch.randperm(n)[:k]
    centroids = F.normalize(features[indices].clone(), p=2, dim=-1)
    for _ in range(max_iters):
        sims = torch.matmul(features, centroids.t())
        labels = torch.argmax(sims, dim=-1)
        new_centroids = torch.zeros_like(centroids)
        for j in range(k):
            mask = (labels == j)
            if mask.sum() > 0:
                new_centroids[j] = features[mask].sum(dim=0)
            else:
                new_centroids[j] = features[torch.randint(0, n, (1,))[0]]
        new_centroids = F.normalize(new_centroids, p=2, dim=-1)
        if torch.allclose(centroids, new_centroids, atol=1e-4):
            break
        centroids = new_centroids
    return centroids


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
    device: torch.device,
    prototypes_param: Optional[nn.Parameter] = None,
    smpc_scale: float = 16.0,
    ortho_lambda: float = 0.10,
    ortho_threshold: float = 0.30
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x_batch, y_batch in loader:
        x_batch = x_batch.to(device, non_blocking=True)
        y_batch = y_batch.to(device, non_blocking=True)

        optimizer.zero_grad()
        if prototypes_param is not None:
            dense_feat = model.extract_features(x_batch)
            z_norm = F.normalize(dense_feat, p=2, dim=-1)
            K, M, D = prototypes_param.shape
            w_norm = F.normalize(prototypes_param, p=2, dim=-1)
            w_flat = w_norm.view(K * M, D)
            sims_flat = torch.matmul(z_norm, w_flat.t())
            sims = sims_flat.view(-1, K, M)
            max_sims, _ = torch.max(sims, dim=-1)

            margin_loss = criterion(max_sims, y_batch)
            ortho_loss = compute_orthogonality_loss(prototypes_param, threshold=ortho_threshold)
            loss = margin_loss + ortho_lambda * ortho_loss
            logits = smpc_scale * max_sims
        else:
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
    device: torch.device,
    prototypes_param: Optional[nn.Parameter] = None,
    smpc_scale: float = 16.0
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

        if prototypes_param is not None:
            dense_feat = model.extract_features(x_batch)
            z_norm = F.normalize(dense_feat, p=2, dim=-1)
            K, M, D = prototypes_param.shape
            w_norm = F.normalize(prototypes_param, p=2, dim=-1)
            w_flat = w_norm.view(K * M, D)
            sims_flat = torch.matmul(z_norm, w_flat.t())
            sims = sims_flat.view(-1, K, M)
            max_sims, _ = torch.max(sims, dim=-1)
            loss = criterion(max_sims, y_batch)
            logits = smpc_scale * max_sims
        else:
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
    device_name: Optional[str] = None,
    use_smpc: bool = False,
    smpc_subcenters: int = 3,
    smpc_margin: float = 0.25,
    smpc_scale: float = 16.0,
    smpc_ortho_lambda: float = 0.10,
    smpc_ortho_threshold: float = 0.30
):
    torch.manual_seed(777)
    np.random.seed(777)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(777)

    print("=" * 70)
    print("TRENING MODELU 2D CNN W PYTORCH (ACOUSTIC HDD SENTINEL)")
    if use_smpc:
        print(f"TRYB: Sub-Center Multi-Prototype Cosine (SMPC, M={smpc_subcenters}, margin={smpc_margin}, scale={smpc_scale})")
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

    if use_smpc:
        prototypes_param = nn.Parameter(torch.randn(num_classes, smpc_subcenters, 128, device=device))
        for k in range(num_classes):
            q, _ = torch.linalg.qr(torch.randn(128, smpc_subcenters, device=device))
            prototypes_param.data[k] = q.t()

        criterion = SubCenterArcMarginLoss(
            scale=smpc_scale,
            margin=smpc_margin,
            class_weights=weights_tensor,
            label_smoothing=0.03
        )
        optimizer = torch.optim.AdamW([
            {"params": model.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": [prototypes_param], "lr": lr, "weight_decay": 0.0}
        ])
    else:
        prototypes_param = None
        criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.03)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            prototypes_param=prototypes_param, smpc_scale=smpc_scale,
            ortho_lambda=smpc_ortho_lambda, ortho_threshold=smpc_ortho_threshold
        )
        val_loss, val_acc, _, _ = evaluate_epoch(
            model, val_loader, criterion, device,
            prototypes_param=prototypes_param, smpc_scale=smpc_scale
        )
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
                "num_classes": num_classes,
                "use_smpc": use_smpc
            }
            if use_smpc and prototypes_param is not None:
                checkpoint["smpc_prototypes"] = prototypes_param.data.cpu()
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

    print(f"[5.5/6] Kalibracja prototypów Open-Set OOD dla uszkodzeń HDD...")
    model.eval()
    all_features = []
    all_targets = []
    with torch.no_grad():
        for x_b, y_b in train_loader:
            x_b = x_b.to(device)
            f_b = model.extract_features(x_b)
            all_features.append(f_b.cpu())
            all_targets.append(y_b.cpu())

    all_f = torch.cat(all_features, dim=0)
    all_t = torch.cat(all_targets, dim=0)

    # Indeksy klas uszkodzeń: beeping (0), clicking (1), scratching (3)
    fault_indices = [0, 1, 3]

    if use_smpc:
        M = smpc_subcenters
        prototypes = torch.zeros((3, M, 128), dtype=torch.float32)
        thresholds = torch.zeros(3, dtype=torch.float32)
        for i, f_idx in enumerate(fault_indices):
            mask = (all_t == f_idx)
            class_f = all_f[mask]
            class_f_norm = F.normalize(class_f, p=2, dim=1)

            if prototypes_param is not None:
                sub_centers = F.normalize(prototypes_param.data[f_idx].cpu(), p=2, dim=-1)
            else:
                sub_centers = spherical_kmeans(class_f_norm, k=M)

            prototypes[i] = sub_centers

            # Oblicz podobieństwo do najbliższego sub-centru
            sims_m = torch.matmul(class_f_norm, sub_centers.t())
            sims, _ = torch.max(sims_m, dim=-1)
            thresh = torch.quantile(sims, 0.05).item()
            thresh = max(0.35, min(0.65, thresh - 0.05))
            thresholds[i] = thresh
            cls_name = classes[f_idx]
            print(f"  - Prototyp SMPC (M={M}) {cls_name:12s}: obliczono z {len(class_f)} tensorów, próg: {thresh:.3f}")
    else:
        prototypes = torch.zeros((3, 128), dtype=torch.float32)
        thresholds = torch.zeros(3, dtype=torch.float32)
        for i, f_idx in enumerate(fault_indices):
            mask = (all_t == f_idx)
            class_f = all_f[mask]
            class_f_norm = F.normalize(class_f, p=2, dim=1)
            centroid = class_f_norm.mean(dim=0)
            centroid = F.normalize(centroid.unsqueeze(0), p=2, dim=1).squeeze(0)
            prototypes[i] = centroid

            sims = torch.matmul(class_f_norm, centroid)
            thresh = torch.quantile(sims, 0.05).item()
            thresh = max(0.35, min(0.65, thresh - 0.05))
            thresholds[i] = thresh
            cls_name = classes[f_idx]
            print(f"  - Prototyp {cls_name:12s}: obliczono z {len(class_f)} tensorów, próg cosinusowy: {thresh:.3f}")

    model.set_fault_prototypes(prototypes.to(device), thresholds.to(device), gamma=12.0)

    # Aktualizacja zapisanego checkpointu o skalibrowane prototypy OOD
    best_checkpoint["state_dict"] = model.state_dict()
    best_checkpoint["open_set_calibrated"] = True
    torch.save(best_checkpoint, best_model_path)
    print(f"      -> Zapisano checkpoint ze skalibrowanymi prototypami OOD: {best_model_path}")

    print(f"[6/6] Finalna ewaluacja na zbiorze testowym...")
    test_loss, test_acc, y_true, y_pred = evaluate_epoch(
        model, test_loader, criterion, device,
        prototypes_param=prototypes_param if use_smpc else None,
        smpc_scale=smpc_scale
    )

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
    parser.add_argument("--use-smpc", action="store_true", default=False, help="Włącz architekturę Sub-Center Multi-Prototype Cosine (SMPC)")
    parser.add_argument("--smpc-subcenters", type=int, default=3, help="Liczba podcentrów na klasę uszkodzenia (M)")
    parser.add_argument("--smpc-margin", type=float, default=0.25, help="Margines kątowy ArcFace m (w radianach)")
    parser.add_argument("--smpc-scale", type=float, default=16.0, help="Czynnik skalowania s dla podobieństw cosinusowych")
    parser.add_argument("--smpc-ortho-lambda", type=float, default=0.10, help="Waga straty ortogonalności podcentrów")
    parser.add_argument("--smpc-ortho-threshold", type=float, default=0.30, help="Próg odpychania podcentrów w stracie ortogonalności")
    args = parser.parse_args()

    ds_dir = resolve_dataset_dir(args.dataset_dir)
    out_dir = Path(args.output_dir)

    run_training(
        dataset_dir=ds_dir,
        output_dir=out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        use_smpc=args.use_smpc,
        smpc_subcenters=args.smpc_subcenters,
        smpc_margin=args.smpc_margin,
        smpc_scale=args.smpc_scale,
        smpc_ortho_lambda=args.smpc_ortho_lambda,
        smpc_ortho_threshold=args.smpc_ortho_threshold
    )


if __name__ == "__main__":
    main()
