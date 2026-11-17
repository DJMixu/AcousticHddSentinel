#!/usr/bin/env python3
"""
train_gmm_ood.py
Trening i kalibracja estymatora gęstości prawdopodobieństwa GMM dla detekcji OOD.
Single Source of Truth: ADR-001, Fault-Only Learning.
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.model_no_normal import AcousticFaultOnlyCNN
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset


def extract_embeddings(model, loader, device):
    model.eval()
    all_embs = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            _, ne = model.extract_features(x)
            all_embs.append(ne.cpu().numpy())
            all_targets.append(y.numpy())
    return np.concatenate(all_embs, axis=0), np.concatenate(all_targets, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Trening i kalibracja GMM OOD")
    parser.add_argument("--weights", default="PrototypBezNormal/output_v2/best_fault_cnn_v2.pt")
    parser.add_argument("--output-dir", default="PrototypBezNormal/output_v2")
    parser.add_argument("--percentile", type=float, default=45.0, help="Percentyl odcięcia na zbiorze walidacyjnym")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie: {device}")

    # 1. Ładowanie modelu CNN
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # 2. Zbiór treningowy awarii (3 421 próbek)
    train_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "train", include_normal=False)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=False)
    tr_embs, tr_labels = extract_embeddings(model, train_loader, device)
    print(f"Załadowano {len(tr_embs)} embeddingów treningowych awarii.")

    # 3. Dopasowanie GMM (3 komponenty)
    print("Dopasowywanie modelu Gaussian Mixture Model (GMM)...")
    gmm = GaussianMixture(n_components=3, covariance_type="diag", max_iter=200, random_state=42)
    gmm.fit(tr_embs)
    print("Dopasowanie GMM zakończone sukcesem.")

    # 4. Kalibracja progu na zbiorze walidacyjnym awarii (131 próbek)
    val_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "validate", include_normal=False)
    val_loader = DataLoader(val_ds, batch_size=128, shuffle=False)
    val_embs, val_labels = extract_embeddings(model, val_loader, device)

    val_scores = gmm.score_samples(val_embs)
    threshold = float(np.percentile(val_scores, args.percentile))
    val_mean = float(np.mean(val_scores))
    val_std = float(np.std(val_scores))

    print("\n" + "="*60)
    print("PARAMETRY KALIBRACJI GMM OOD:")
    print("="*60)
    print(f"  Średnia gęstość w walidacji:  {val_mean:.2f} +/- {val_std:.2f}")
    print(f"  Percentyl kalibracji:         {args.percentile:.1f}%")
    print(f"  Wyznaczony próg OOD:          {threshold:.2f}")

    # 5. Zapis modelu i konfiguracji
    gmm_save_path = output_dir / "gmm_density_model.joblib"
    joblib.dump({
        "gmm": gmm,
        "threshold": threshold,
        "val_mean": val_mean,
        "val_std": val_std,
        "percentile": args.percentile
    }, gmm_save_path)
    print(f"Zapisano model GMM do: {gmm_save_path}")

    config_save_path = output_dir / "ood_gmm_config.json"
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold": threshold,
            "val_mean": val_mean,
            "val_std": val_std,
            "percentile": args.percentile,
            "training_samples": len(tr_embs),
            "validation_samples": len(val_embs)
        }, f, indent=2)
    print(f"Zapisano konfigurację do: {config_save_path}")


if __name__ == "__main__":
    main()
