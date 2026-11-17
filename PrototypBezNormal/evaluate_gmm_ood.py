#!/usr/bin/env python3
"""
evaluate_gmm_ood.py
Rygorystyczny benchmark zintegrowanego modelu OOD (CNN + GMM) na pełnym zbiorze testowym.
Single Source of Truth: ADR-001, OOD Evaluation (AUROC, AUPR, 4x4 Confusion Matrix).
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
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.ood_gmm_engine import GMMFaultOODDetector
from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset, EVAL_CLASSES_4


def plot_confusion_matrix_4x4(cm: np.ndarray, classes: list, save_path: Path):
    """Generuje estetyczny wykres macierzy pomyłek 4x4 z wartościami i procentami za pomocą Matplotlib."""
    fig, ax = plt.subplots(figsize=(8.5, 7.5), dpi=150)
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    cax = ax.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    plt.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, fontsize=11, fontweight="bold")
    ax.set_yticklabels(classes, fontsize=11, fontweight="bold")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center", color=color, fontsize=10)

    ax.set_title("Macierz Pomyłek 4x4 (Model OOD CNN + GMM bez 'normal' w treningu)", fontsize=12, pad=15, fontweight="bold")
    ax.set_ylabel("Rzeczywista klasa (Ground Truth)", fontsize=11, labelpad=10)
    ax.set_xlabel("Predykcja modelu OOD", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja OOD CNN + GMM")
    parser.add_argument("--weights", default="PrototypBezNormal/output_v2/best_fault_cnn_v2.pt")
    parser.add_argument("--gmm", default="PrototypBezNormal/output_v2/gmm_density_model.joblib")
    parser.add_argument("--output-dir", default="PrototypBezNormal/output_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Urządzenie ewaluacji: {device}")

    detector = GMMFaultOODDetector(
        model_weights_path=args.weights,
        gmm_model_path=args.gmm,
        device=device
    )

    test_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "test", include_normal=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    classes_4 = EVAL_CLASSES_4  # ["beeping", "clicking", "normal", "scratching"]

    print(f"Załadowano {len(test_ds)} tensorów testowych (w tym {sum(1 for _, y in test_ds if classes_4[y]=='normal')} próbek 'normal').")

    all_targets = []
    all_preds = []
    binary_gt = []
    binary_scores = []  # Wyższa wartość = większe prawdopodobieństwo bycia awarią

    for x, y in test_loader:
        results = detector.predict(x)
        if isinstance(results, dict):
            results = [results]

        for i in range(len(y)):
            target_cls = classes_4[y[i].item()]
            res = results[i]
            pred_cls = res["verdict"]

            all_targets.append(target_cls)
            all_preds.append(pred_cls)

            # Etykieta binarna OOD: 1 dla awarii, 0 dla normal
            is_fault = 0 if target_cls == "normal" else 1
            binary_gt.append(is_fault)
            binary_scores.append(res["log_likelihood"])

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    binary_gt = np.array(binary_gt)
    binary_scores = np.array(binary_scores)

    # 1. Raport klasyfikacji
    print("\n" + "="*65)
    print("SZCZEGÓŁOWY RAPORT KLASYFIKACJI 4-KLASOWEJ (CNN + GMM OOD)")
    print("="*65)
    report_dict = classification_report(all_targets, all_preds, target_names=classes_4, output_dict=True, zero_division=0)
    report_text = classification_report(all_targets, all_preds, target_names=classes_4, digits=4, zero_division=0)
    print(report_text)

    # 2. Metryki OOD (AUROC i AUPR)
    auroc = float(roc_auc_score(binary_gt, binary_scores))
    p_vals, r_vals, _ = precision_recall_curve(binary_gt, binary_scores)
    aupr = float(auc(r_vals, p_vals))

    norm_rec = report_dict["normal"]["recall"]
    norm_prec = report_dict["normal"]["precision"]
    norm_f1 = report_dict["normal"]["f1-score"]

    print("\n" + "="*65)
    print("METRYKI DETEKCJI OUT-OF-DISTRIBUTION (OOD: NORMAL vs USZKODZENIA):")
    print("="*65)
    print(f"  AUROC (Area Under ROC Curve):              {auroc*100:.2f}%")
    print(f"  AUPR  (Area Under PR Curve):               {aupr*100:.2f}%")
    print(f"  Wykrywanie 'normal' Czułość (Recall):       {norm_rec*100:.2f}% ({int(norm_rec*report_dict['normal']['support'])}/{int(report_dict['normal']['support'])})")
    print(f"  Wykrywanie 'normal' Precyzja (Precision):   {norm_prec*100:.2f}%")
    print(f"  Wykrywanie 'normal' F1-Score:               {norm_f1:.4f}")

    # 3. Macierz pomyłek 4x4
    cm = confusion_matrix(all_targets, all_preds, labels=classes_4)
    cm_path = output_dir / "confusion_matrix_gmm_4x4.png"
    plot_confusion_matrix_4x4(cm, classes_4, cm_path)
    print(f"\nZapisano macierz pomyłek do: {cm_path}")

    # 4. Wykres rozkładu log-likelihood
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for cname in classes_4:
        mask = (all_targets == cname)
        color = "green" if cname == "normal" else None
        linestyle = "--" if cname == "normal" else "-"
        ax.hist(binary_scores[mask], bins=25, alpha=0.4, label=cname, density=True, color=color, histtype='stepfilled')
    ax.axvline(detector.threshold, color="red", linestyle="--", linewidth=2, label=f"Próg OOD ({detector.threshold:.2f})")
    ax.set_title("Rozkład Log-Gęstości Prawdopodobieństwa GMM: Normal vs Awarie", fontweight="bold")
    ax.set_xlabel("Log-Likelihood $\log p(z | \\text{faults})$")
    ax.set_ylabel("Gęstość")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = output_dir / "gmm_log_likelihood_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Zapisano histogram do: {hist_path}")

    summary = {
        "classification_report": report_dict,
        "auroc": auroc,
        "aupr": aupr,
        "threshold": detector.threshold,
        "confusion_matrix": cm.tolist()
    }
    with open(output_dir / "gmm_test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
