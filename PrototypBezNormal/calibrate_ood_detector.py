#!/usr/bin/env python3
"""
calibrate_ood_detector.py
Kalibracja wielowymiarowego detektora Out-Of-Distribution (OOD) dla modelu Fault-Only.

Single Source of Truth: ADR-001, OOD Neyman-Pearson Decision Theory.
Zasada: Wszystkie parametry i progi są wyznaczane WYŁĄCZNIE na danych awarii
(zbiór treningowy i walidacyjny). Zero dostępu do próbek 'normal' podczas kalibracji!
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.model_no_normal import AcousticFaultOnlyCNN
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset


def extract_all_representations(model, loader, device):
    model.eval()
    all_embs = []
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            raw_e, norm_e = model.extract_features(x)
            h = F.relu(raw_e)
            logits = model.classifier(h)

            all_embs.append(norm_e.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
            all_targets.append(y.numpy())

    embs = np.concatenate(all_embs, axis=0)
    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    return embs, logits, targets


def calibrate_ood(
    weights_path: str = "PrototypBezNormal/output_v2/best_fault_cnn_v2.pt",
    output_config_path: str = "PrototypBezNormal/output_v2/ood_config_v2.json",
    tpr_target: float = 0.95
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kalibracja OOD na urządzeniu: {device}")

    ckpt = torch.load(weights_path, map_location=device, weights_only=False)
    model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    fault_classes = ["beeping", "clicking", "scratching"]

    # 1. Zbiór treningowy do wyznaczenia centroidów i kowariancji
    train_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "train", include_normal=False, augment=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
    tr_embs, tr_logits, tr_targets = extract_all_representations(model, train_loader, device)

    # Centroidy klasowe na sferze L2
    centroids = []
    class_covs = []
    pooled_cov = np.zeros((128, 128))

    for c in range(3):
        m = (tr_targets == c)
        cls_embs = tr_embs[m]
        mean_vec = np.mean(cls_embs, axis=0)
        mean_norm = np.linalg.norm(mean_vec)
        centroid = mean_vec / (mean_norm + 1e-9)
        centroids.append(centroid)

        diff = cls_embs - centroid
        cov_c = np.dot(diff.T, diff) / len(cls_embs)
        class_covs.append(cov_c)
        pooled_cov += np.dot(diff.T, diff)

    centroids = np.array(centroids)
    pooled_cov = pooled_cov / len(tr_embs) + 1e-4 * np.eye(128)
    inv_pooled_cov = np.linalg.inv(pooled_cov)

    print("Centroidy klas awarii obliczone pomyślnie.")

    # 2. Zbiór walidacyjny do wyznaczenia progów decyzyjnych (BEZ próbek 'normal'!)
    val_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "validate", include_normal=False, augment=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    val_embs, val_logits, val_targets = extract_all_representations(model, val_loader, device)

    # Obliczenie wskaźników pewności na zbiorze walidacyjnym awarii
    val_cosines = []
    val_mahals = []
    val_energies = []
    val_scores = []

    for i in range(len(val_embs)):
        z = val_embs[i]
        c_true = val_targets[i]

        # Cosinus do centroidu
        cos_sims = np.dot(centroids, z)
        max_cos = float(np.max(cos_sims))
        val_cosines.append(max_cos)

        # Mahalanobis
        diff = z - centroids[c_true]
        m_dist = float(np.sqrt(np.dot(np.dot(diff, inv_pooled_cov), diff.T)))
        val_mahals.append(m_dist)

        # Free Energy
        l = val_logits[i]
        energy = float(-np.log(np.sum(np.exp(l)) + 1e-9))
        val_energies.append(energy)

        # Połączony skoring (wyższy = bardziej pasuje do znanej awarii)
        # Cosinus [0, 1], Mahalanobis ~ [2, 10], Energy ~ [-6, 0]
        score = max_cos - 0.05 * m_dist - 0.10 * energy
        val_scores.append(score)

    val_cosines = np.array(val_cosines)
    val_mahals = np.array(val_mahals)
    val_energies = np.array(val_energies)
    val_scores = np.array(val_scores)

    # Wyznaczenie progu przy zadanym True Positive Rate awarii (np. 95%)
    cutoff_percentile = (1.0 - tpr_target) * 100.0  # np. 5.0%

    thresh_score = float(np.percentile(val_scores, cutoff_percentile))
    thresh_cos = float(np.percentile(val_cosines, cutoff_percentile))
    thresh_mahal = float(np.percentile(val_mahals, 100.0 - cutoff_percentile))
    thresh_energy = float(np.percentile(val_energies, 100.0 - cutoff_percentile))

    config = {
        "fault_classes": fault_classes,
        "eval_classes": ["beeping", "clicking", "normal", "scratching"],
        "centroids": centroids.tolist(),
        "inv_pooled_cov": inv_pooled_cov.tolist(),
        "thresholds": {
            "ood_score_threshold": thresh_score,
            "cosine_threshold": thresh_cos,
            "mahalanobis_threshold": thresh_mahal,
            "energy_threshold": thresh_energy,
            "tpr_target": tpr_target
        },
        "validation_stats": {
            "mean_val_score": float(np.mean(val_scores)),
            "std_val_score": float(np.std(val_scores)),
            "mean_val_cosine": float(np.mean(val_cosines)),
            "mean_val_mahal": float(np.mean(val_mahals)),
            "mean_val_energy": float(np.mean(val_energies))
        }
    }

    out_p = Path(output_config_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("\n" + "="*60)
    print("SKALIBROWANO PROGI OOD (FAULT-ONLY KALIBRACJA):")
    print("="*60)
    print(f"  Target TPR dla awarii:        {tpr_target*100:.1f}%")
    print(f"  Próg połączonego wyniku OOD:  {thresh_score:.4f}")
    print(f"  Próg Podobieństwa Cosinus:    {thresh_cos:.4f}")
    print(f"  Próg Dystansu Mahalanobisa:   {thresh_mahal:.4f}")
    print(f"  Próg Free Energy:             {thresh_energy:.4f}")
    print(f"Zapisano konfigurację do:       {out_p}")
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default="PrototypBezNormal/output_v2/best_fault_cnn_v2.pt")
    parser.add_argument("--output", default="PrototypBezNormal/output_v2/ood_config_v2.json")
    parser.add_argument("--tpr", type=float, default=0.95)
    args = parser.parse_args()

    calibrate_ood(args.weights, args.output, args.tpr)
