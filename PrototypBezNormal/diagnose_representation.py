#!/usr/bin/env python3
"""
Diagnostyka reprezentacji przestrzeni ukrytej w PrototypBezNormal.
Bada zachowanie:
- Norm wektorów embeddingów ||z||
- Odległości euklidesowych i cosinusowych
- Energii swobodnej logitów E(x)
- Odległości Mahalanobisa
dla próbek treningowych awarii oraz próbek testowych (awarie vs normal).
"""

import os
import sys
import json
import torch
import numpy as np
import scipy.spatial.distance as dist

# Dołączenie ścieżki projektu
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from PrototypBezNormal.model_no_normal import AcousticFaultOnlyCNN
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    checkpoint_path = os.path.join(PROJECT_ROOT, "PrototypBezNormal", "output", "best_model_no_normal.pt")
    if not os.path.exists(checkpoint_path):
        print("Brak pliku wag!")
        return

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(DEVICE)
    state_dict = checkpoint.get("state_dict", checkpoint.get("model_state_dict"))
    model.load_state_dict(state_dict)
    model.eval()

    fault_classes = ["beeping", "clicking", "scratching"]

    # 1. Zbiór treningowy (same awarie)
    train_ds = FaultOnlyNpyDataset(os.path.join(PROJECT_ROOT, "dataset_encoded", "train"), include_normal=False)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=False)

    train_embeddings = {cls: [] for cls in fault_classes}
    train_logits = {cls: [] for cls in fault_classes}

    with torch.no_grad():
        for x, y in train_loader:
            x = x.to(DEVICE)
            logits, embs = model(x)
            for i in range(len(y)):
                cls_name = fault_classes[y[i].item()]
                train_embeddings[cls_name].append(embs[i].cpu().numpy())
                train_logits[cls_name].append(logits[i].cpu().numpy())

    for cls in fault_classes:
        train_embeddings[cls] = np.array(train_embeddings[cls])
        train_logits[cls] = np.array(train_logits[cls])

    # Centroidy i kowariancja
    centroids = {cls: np.mean(train_embeddings[cls], axis=0) for cls in fault_classes}
    all_train_embs = np.concatenate([train_embeddings[cls] for cls in fault_classes], axis=0)
    
    # Wspólna macierz kowariancji (pooled covariance) z regularyzacją
    pooled_cov = np.zeros((128, 128))
    for cls in fault_classes:
        diff = train_embeddings[cls] - centroids[cls]
        pooled_cov += np.dot(diff.T, diff)
    pooled_cov /= len(all_train_embs)
    pooled_cov += 1e-4 * np.eye(128)
    inv_pooled_cov = np.linalg.inv(pooled_cov)

    print("=== PARAMETRY TRENINGOWE AWARII ===")
    for cls in fault_classes:
        norms = np.linalg.norm(train_embeddings[cls], axis=1)
        mean_d = np.mean(np.linalg.norm(train_embeddings[cls] - centroids[cls], axis=1))
        std_d = np.std(np.linalg.norm(train_embeddings[cls] - centroids[cls], axis=1))
        # Free Energy: E = -log(sum(exp(l)))
        l = train_logits[cls]
        energies = -np.log(np.sum(np.exp(l), axis=1) + 1e-9)
        print(f"Klasa [{cls}] (N={len(train_embeddings[cls])}):")
        print(f"  Średnia norma ||z||: {np.mean(norms):.3f} +/- {np.std(norms):.3f}")
        print(f"  Dystans euklidesowy do centroidu: {mean_d:.3f} +/- {std_d:.3f}")
        print(f"  Free Energy E(x): {np.mean(energies):.3f} +/- {np.std(energies):.3f}")

    # 2. Zbiór testowy (zawiera awarie ORAZ normal)
    test_ds = FaultOnlyNpyDataset(os.path.join(PROJECT_ROOT, "dataset_encoded", "test"), include_normal=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    test_classes = ["beeping", "clicking", "scratching", "normal"]
    test_embeddings = {cls: [] for cls in test_classes}
    test_logits = {cls: [] for cls in test_classes}

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            logits, embs = model(x)
            for i in range(len(y)):
                idx = y[i].item()
                cls_name = test_classes[idx]
                test_embeddings[cls_name].append(embs[i].cpu().numpy())
                test_logits[cls_name].append(logits[i].cpu().numpy())

    for cls in test_classes:
        test_embeddings[cls] = np.array(test_embeddings[cls])
        test_logits[cls] = np.array(test_logits[cls])

    print("\n=== ANALIZA ZBIORU TESTOWEGO (AWARIE vs NORMAL) ===")
    for cls in test_classes:
        embs = test_embeddings[cls]
        l = test_logits[cls]
        if len(embs) == 0:
            continue
        norms = np.linalg.norm(embs, axis=1)
        
        # Min dystans euklidesowy do najbliższego centroidu
        min_eucl_dists = []
        # Max cosinusowe podobieństwo do centroidu
        max_cos_sims = []
        # Min odległość Mahalanobisa
        min_mahal_dists = []
        # Free Energy E(x)
        energies = -np.log(np.sum(np.exp(l), axis=1) + 1e-9)

        for z in embs:
            # Euklides
            eucls = [np.linalg.norm(z - centroids[c]) for c in fault_classes]
            min_eucl_dists.append(min(eucls))
            # Cosinus
            coss = [np.dot(z, centroids[c]) / (np.linalg.norm(z) * np.linalg.norm(centroids[c]) + 1e-9) for c in fault_classes]
            max_cos_sims.append(max(coss))
            # Mahalanobis
            mahals = [dist.mahalanobis(z, centroids[c], inv_pooled_cov) for c in fault_classes]
            min_mahal_dists.append(min(mahals))

        print(f"\nKlasa testowa: [{cls.upper()}] (N={len(embs)})")
        print(f"  Średnia norma ||z||:       {np.mean(norms):.3f} +/- {np.std(norms):.3f}")
        print(f"  Dystans Euklidesa min:     {np.mean(min_eucl_dists):.3f} +/- {np.std(min_eucl_dists):.3f} (min={np.min(min_eucl_dists):.3f}, max={np.max(min_eucl_dists):.3f})")
        print(f"  Podobieństwo Cosinus max:  {np.mean(max_cos_sims):.3f} +/- {np.std(max_cos_sims):.3f} (min={np.min(max_cos_sims):.3f}, max={np.max(max_cos_sims):.3f})")
        print(f"  Dystans Mahalanobisa min:  {np.mean(min_mahal_dists):.3f} +/- {np.std(min_mahal_dists):.3f} (min={np.min(min_mahal_dists):.3f}, max={np.max(min_mahal_dists):.3f})")
        print(f"  Free Energy E(x):          {np.mean(energies):.3f} +/- {np.std(energies):.3f} (min={np.min(energies):.3f}, max={np.max(energies):.3f})")

if __name__ == "__main__":
    main()
