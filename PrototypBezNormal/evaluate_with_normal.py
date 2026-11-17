#!/usr/bin/env python3
"""
evaluate_with_normal.py
Ewaluacja modelu trenowanego wyłącznie na awariach (Fault-Only) na pełnym zbiorze testowym
ZAWIERAJĄCYM klasę 'normal' (dysk sprawny).

Cel:
Zweryfikowanie czy mechanizm metrycznych prototypów i progowania energii potrafi
poprawnie rozpoznać sprawny dysk jako 'normal', zachowując jednocześnie wysoką
skuteczność klasyfikacji uszkodzeń (beeping, clicking, scratching).
"""

import sys
import json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

PROTOTYPE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOTYPE_DIR.parent
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_no_normal import AcousticFaultOnlyCNN
from dataset_no_normal import FaultOnlyNpyDataset, EVAL_CLASSES_4


def evaluate(
    weights_path: str = "PrototypBezNormal/output/best_model_no_normal.pt",
    dataset_encoded_dir: str = "dataset_encoded",
    output_dir: str = "PrototypBezNormal/output",
    threshold_override: float = None
):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[Ewaluacja] Używam urządzenia: {device}")

    print(f"[Ewaluacja] Wczytywanie wag z {weights_path}...")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    if threshold_override is not None:
        model.global_threshold.copy_(torch.tensor(threshold_override, device=device))
        print(f"[Parametr] Nadpisano próg odległości (Threshold Override): {threshold_override}")
    else:
        print(f"[Parametr] Używam wykalibrowanego progu odległości: {model.global_threshold.item():.4f}")

    # Wczytanie zbioru testowego z klasą 'normal'
    test_ds = FaultOnlyNpyDataset(
        Path(dataset_encoded_dir) / "test",
        include_normal=True,
        augment=False,
        preload_to_ram=True
    )
    print(f"[Dane] Załadowano zbiór testowy: {len(test_ds)} tensorów (.npy) dla 4 klas.")

    all_preds = []
    all_targets = []
    all_dists = []
    all_energies = []

    for i in range(len(test_ds)):
        x, y = test_ds[i]
        x_in = x.unsqueeze(0).to(device)
        res = model.predict_with_normal(x_in)

        pred_idx = res["pred_idx"][0]
        all_preds.append(pred_idx)
        all_targets.append(y)
        all_dists.append(float(res["min_distance"][0]))
        all_energies.append(float(res["energy"][0]))

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Indeksy w EVAL_CLASSES_4: 0: beeping, 1: clicking, 2: normal, 3: scratching
    report_dict = classification_report(
        all_targets, all_preds,
        target_names=EVAL_CLASSES_4,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        all_targets, all_preds,
        target_names=EVAL_CLASSES_4,
        zero_division=0
    )

    print("\n" + "="*70)
    print("OFICJALNY RAPORT KLASYFIKACJI (TEST Z KLASĄ NORMAL DLA MODELU FAULT-ONLY):")
    print("="*70)
    print(report_text)
    print("="*70)

    # Macierz pomyłek
    cm = confusion_matrix(all_targets, all_preds)
    print("Macierz pomyłek (Confusion Matrix):")
    print(cm)

    # Wykres macierzy pomyłek
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=EVAL_CLASSES_4,
        yticklabels=EVAL_CLASSES_4,
        title="Macierz Pomyłek – Model Fault-Only (Test z klasą Normal)",
        ylabel="Etykieta Rzeczywista",
        xlabel="Etykieta Przewidziana"
    )

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = (val / np.sum(cm[i, :])) * 100 if np.sum(cm[i, :]) > 0 else 0
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color="white" if val > thresh else "black"
            )

    fig.tight_layout()
    cm_path = out_path / "confusion_matrix_with_normal.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nZapisano macierz pomyłek do: {cm_path}")

    # Zapis raportu do JSON
    summary_path = out_path / "test_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"Zapisano podsumowanie metryk JSON do: {summary_path}")

    # Statystyki odległości dla 'normal' vs 'faults'
    is_norm = (all_targets == 2)
    dists_norm = [d for d, n in zip(all_dists, is_norm) if n]
    dists_fault = [d for d, n in zip(all_dists, is_norm) if not n]

    print("\n[Analiza Metryczna]")
    print(f"  Średnia odległość dla próbek NORMAL: {np.mean(dists_norm):.4f} (mediana: {np.median(dists_norm):.4f})")
    print(f"  Średnia odległość dla AWARII:        {np.mean(dists_fault):.4f} (mediana: {np.median(dists_fault):.4f})")
    print(f"  Separacja metryczna:                {np.mean(dists_norm) - np.mean(dists_fault):.4f}")

    return report_dict


if __name__ == "__main__":
    evaluate()
