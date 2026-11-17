#!/usr/bin/env python3
"""
Rygorystyczna ewaluacja modelu metrycznego OOD z uwzględnieniem klasy 'normal'.

Single Source of Truth: ADR-001, OOD Evaluation Standards (Hendrycks & Gimpel 2017, Lee et al. 2018).
Weryfikuje zachowanie modelu na pełnym zbiorze testowym zawierającym:
- 283 próbki zdrowych dysków (normal - stan nieobecny w treningu!)
- 245 autentycznych próbek uszkodzeń (beeping, clicking, scratching)
"""

import os
import sys
import json
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.model_metric_cnn import MetricFaultOnlyCNN
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset, EVAL_CLASSES_4


def plot_confusion_matrix_4x4(cm: np.ndarray, classes: list, save_path: Path):
    """Generuje estetyczny wykres macierzy pomyłek 4x4 z wartościami i procentami za pomocą Matplotlib."""
    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_yticklabels(classes, fontsize=10)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Macierz Pomyłek 4x4 (Model Metryczny OOD bez klasy 'normal' w treningu)", fontsize=12, pad=15, fontweight="bold")
    ax.set_ylabel("Rzeczywista klasa (Ground Truth)", fontsize=11, labelpad=10)
    ax.set_xlabel("Predykcja modelu OOD", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_ood_distributions(
    metrics_by_class: dict,
    thresholds: dict,
    save_path: Path
):
    """Rysuje rozkłady miar OOD (Cosinus, Mahalanobis, Free Energy) za pomocą Matplotlib."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=150)

    # 1. Cosinus similarity
    ax = axes[0]
    for cls, m in metrics_by_class.items():
        if len(m["cosines"]) > 0:
            color = "green" if cls == "normal" else None
            linestyle = "--" if cls == "normal" else "-"
            ax.hist(m["cosines"], bins=20, density=True, alpha=0.35, label=cls, color=color, histtype='stepfilled')
    ax.axvline(thresholds.get("global_cosine_threshold", 0.5), color="red", linestyle=":", label="Próg OOD")
    ax.set_title("Maksymalne podobieństwo cosinusowe ($S_{cos}$)", fontweight="bold")
    ax.set_xlabel("Cosinus kąta do klastra")
    ax.set_ylabel("Gęstość")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Mahalanobis distance
    ax = axes[1]
    for cls, m in metrics_by_class.items():
        if len(m["mahalanobis"]) > 0:
            color = "green" if cls == "normal" else None
            ax.hist(m["mahalanobis"], bins=20, density=True, alpha=0.35, label=cls, color=color, histtype='stepfilled')
    ax.axvline(thresholds.get("global_mahalanobis_threshold", 10.0), color="red", linestyle=":", label="Próg OOD")
    ax.set_title("Dystans Mahalanobisa ($D_M$)", fontweight="bold")
    ax.set_xlabel("Odległość Mahalanobisa")
    ax.set_ylabel("Gęstość")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Free Energy
    ax = axes[2]
    for cls, m in metrics_by_class.items():
        if len(m["energies"]) > 0:
            color = "green" if cls == "normal" else None
            ax.hist(m["energies"], bins=20, density=True, alpha=0.35, label=cls, color=color, histtype='stepfilled')
    ax.axvline(thresholds.get("global_energy_threshold", -1.5), color="red", linestyle=":", label="Próg OOD")
    ax.set_title("Free Energy logitów ($E(x)$)", fontweight="bold")
    ax.set_xlabel("Free Energy")
    ax.set_ylabel("Gęstość")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Rozkłady Miary Pewności OOD: Zdrowy Dysk (Normal) vs Mechaniczne Uszkodzenia", fontsize=14, y=1.02, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja OOD z klasą normal")
    parser.add_argument("--model-path", type=str, default="PrototypBezNormal/output_metric/best_metric_fault_model.pt")
    parser.add_argument("--output-dir", type=str, default="PrototypBezNormal/output_metric")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie ewaluacji: {device}")

    # 1. Ładowanie wag modelu
    checkpoint_path = Path(args.model_path)
    if not checkpoint_path.exists():
        print(f"Brak pliku wag: {checkpoint_path}")
        sys.exit(1)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = MetricFaultOnlyCNN(
        num_fault_classes=3,
        embedding_dim=128,
        scale=checkpoint.get("scale", 16.0),
        margin=checkpoint.get("margin", 0.35)
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # 2. Zbiór testowy zawierający klasę 'normal'
    test_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "test", include_normal=True)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    classes_4 = EVAL_CLASSES_4  # ["beeping", "clicking", "normal", "scratching"]

    print(f"Załadowano {len(test_ds)} tensorów testowych (w tym {sum(1 for _, y in test_ds if classes_4[y]=='normal')} próbek 'normal').")

    all_targets = []
    all_preds = []
    all_confidences = []
    
    # Metryki binarne dla OOD (Fault vs Normal)
    binary_ground_truth = []  # 1 dla uszkodzenia, 0 dla normal
    binary_fault_scores = []  # wyższy wynik = większe prawdopodobieństwo uszkodzenia

    metrics_by_class = {c: {"cosines": [], "mahalanobis": [], "energies": []} for c in classes_4}

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            results = model.predict_ood(x)
            if isinstance(results, dict):
                results = [results]

            for i in range(len(y)):
                target_cls = classes_4[y[i].item()]
                res = results[i]
                pred_cls = res["verdict"]

                all_targets.append(target_cls)
                all_preds.append(pred_cls)
                all_confidences.append(res["confidence"])

                # Zbieranie statystyk OOD
                m = res["metrics"]
                metrics_by_class[target_cls]["cosines"].append(m["cosine_similarity"])
                metrics_by_class[target_cls]["mahalanobis"].append(m["mahalanobis_distance"])
                metrics_by_class[target_cls]["energies"].append(m["free_energy"])

                # Skoring binarny OOD: cosinus jest bezpośrednim wyznacznikiem bycia uszkodzeniem
                is_actual_fault = 0 if target_cls == "normal" else 1
                binary_ground_truth.append(is_actual_fault)
                binary_fault_scores.append(m["cosine_similarity"])

    # 3. Raport klasyfikacji 4-klasowy
    print("\n" + "="*65)
    print("SZCZEGÓŁOWY RAPORT KLASYFIKACJI 4-KLASOWEJ (Z 'NORMAL')")
    print("="*65)
    report_dict = classification_report(all_targets, all_preds, target_names=classes_4, output_dict=True, zero_division=0)
    report_text = classification_report(all_targets, all_preds, target_names=classes_4, digits=4, zero_division=0)
    print(report_text)

    # 4. Metryki OOD (AUROC i AUPR dla detekcji Normal vs Uszkodzenia)
    binary_ground_truth = np.array(binary_ground_truth)
    binary_fault_scores = np.array(binary_fault_scores)

    auroc = float(roc_auc_score(binary_ground_truth, binary_fault_scores))
    precision_vals, recall_vals, _ = precision_recall_curve(binary_ground_truth, binary_fault_scores)
    aupr = float(auc(recall_vals, precision_vals))

    print("\n" + "="*65)
    print("METRYKI DETEKCJI OUT-OF-DISTRIBUTION (OOD: NORMAL vs USZKODZENIA)")
    print("="*65)
    print(f"  AUROC (Area Under ROC Curve):          {auroc*100:.2f}%")
    print(f"  AUPR  (Area Under PR Curve):           {aupr*100:.2f}%")
    
    normal_recall = report_dict["normal"]["recall"]
    normal_precision = report_dict["normal"]["precision"]
    normal_f1 = report_dict["normal"]["f1-score"]
    print(f"  Wykrywanie 'normal' Czułość (Recall):   {normal_recall*100:.2f}% ({int(normal_recall*report_dict['normal']['support'])}/{int(report_dict['normal']['support'])})")
    print(f"  Wykrywanie 'normal' Precyzja (Precision): {normal_precision*100:.2f}%")
    print(f"  Wykrywanie 'normal' F1-Score:           {normal_f1:.4f}")

    # 5. Macierz pomyłek
    cm = confusion_matrix(all_targets, all_preds, labels=classes_4)
    cm_path = output_dir / "confusion_matrix_metric_4x4.png"
    plot_confusion_matrix_4x4(cm, classes_4, cm_path)
    print(f"\nZapisano macierz pomyłek do: {cm_path}")

    # 6. Wykres rozkładów OOD
    dist_path = output_dir / "ood_distribution_histograms.png"
    thresholds = {
        "global_cosine_threshold": float(model.global_cosine_threshold.item()),
        "global_mahalanobis_threshold": float(model.global_mahalanobis_threshold.item()),
        "global_energy_threshold": float(model.global_energy_threshold.item())
    }
    plot_ood_distributions(metrics_by_class, thresholds, dist_path)
    print(f"Zapisano histogramy miar OOD do: {dist_path}")

    # 7. Zapis podsumowania JSON
    summary = {
        "classification_report": report_dict,
        "ood_metrics": {
            "auroc": auroc,
            "aupr": aupr,
            "normal_recall": normal_recall,
            "normal_precision": normal_precision,
            "normal_f1": normal_f1
        },
        "thresholds": thresholds,
        "confusion_matrix": cm.tolist()
    }
    summary_path = output_dir / "metric_evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Zapisano pełne podsumowanie JSON do: {summary_path}")


if __name__ == "__main__":
    main()
