#!/usr/bin/env python3
"""
evaluate_v2_ood.py
Kompletna ewaluacja modelu v2 z wielowymiarowym detektorem OOD na zbiorze testowym z klasą 'normal'.

Single Source of Truth: ADR-001, OOD Evaluation Metrics (AUROC, AUPR, FPR95, 4x4 Confusion Matrix).
Testuje:
- 303 próbki zdrowych dysków ('normal') - nienotowane w treningu ani walidacji!
- 213 autentycznych próbek awarii (63 beeping, 107 clicking, 43 scratching).
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

from PrototypBezNormal.model_no_normal import AcousticFaultOnlyCNN
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

    ax.set_title("Macierz Pomyłek 4x4 (Model OOD v2 bez klasy 'normal' w treningu)", fontsize=13, pad=15, fontweight="bold")
    ax.set_ylabel("Rzeczywista klasa (Ground Truth)", fontsize=11, labelpad=10)
    ax.set_xlabel("Predykcja modelu OOD", fontsize=11, labelpad=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Ewaluacja OOD v2")
    parser.add_argument("--weights", default="PrototypBezNormal/output_v2/best_fault_cnn_v2.pt")
    parser.add_argument("--config", default="PrototypBezNormal/output_v2/ood_config_v2.json")
    parser.add_argument("--output-dir", default="PrototypBezNormal/output_v2")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie ewaluacji: {device}")

    # Wczytanie wag i konfiguracji OOD
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    model = AcousticFaultOnlyCNN(num_fault_classes=3, embedding_dim=128).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    centroids = np.array(config["centroids"])  # (3, 128)
    inv_pooled_cov = np.array(config["inv_pooled_cov"])  # (128, 128)
    th_score = config["thresholds"]["ood_score_threshold"]
    th_cos = config["thresholds"]["cosine_threshold"]
    fault_classes = config["fault_classes"]
    classes_4 = EVAL_CLASSES_4  # ["beeping", "clicking", "normal", "scratching"]

    # Wczytanie zbioru testowego z klasą 'normal'
    test_ds = FaultOnlyNpyDataset(PROJECT_ROOT / "dataset_encoded" / "test", include_normal=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print(f"Liczba próbek testowych: {len(test_ds)}")

    all_targets = []
    all_preds = []
    binary_ground_truth = []
    binary_fault_scores = []
    ood_scores = []

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            raw_e, norm_e = model.extract_features(x)
            h = F.relu(raw_e)
            logits = model.classifier(h)
            probs = F.softmax(logits, dim=1)

            norm_e_np = norm_e.cpu().numpy()
            logits_np = logits.cpu().numpy()
            probs_np = probs.cpu().numpy()

            for i in range(len(y)):
                target_cls = classes_4[y[i].item()]
                z = norm_e_np[i]

                # 1. Cosinus do centroidów awarii
                cos_sims = np.dot(centroids, z)
                best_fault_idx = int(np.argmax(cos_sims))
                max_cos = float(cos_sims[best_fault_idx])

                # 2. Mahalanobis
                diff = z - centroids[best_fault_idx]
                m_dist = float(np.sqrt(np.dot(np.dot(diff, inv_pooled_cov), diff.T)))

                # 3. Free Energy
                energy = float(-np.log(np.sum(np.exp(logits_np[i])) + 1e-9))

                # 4. Łączny score OOD
                score = max_cos - 0.05 * m_dist - 0.10 * energy
                ood_scores.append(score)

                # Decyzja OOD:
                # Jeśli score jest poniżej progu kalibracyjnego -> NORMAL!
                if score < th_score:
                    pred_cls = "normal"
                else:
                    pred_cls = fault_classes[best_fault_idx]

                all_targets.append(target_cls)
                all_preds.append(pred_cls)

                is_fault = 0 if target_cls == "normal" else 1
                binary_ground_truth.append(is_fault)
                binary_fault_scores.append(score)

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    binary_ground_truth = np.array(binary_ground_truth)
    binary_fault_scores = np.array(binary_fault_scores)

    # Raport
    print("\n" + "="*65)
    print("SZCZEGÓŁOWY RAPORT KLASYFIKACJI 4-KLASOWEJ (MODEL v2 Z OOD)")
    print("="*65)
    report_dict = classification_report(all_targets, all_preds, target_names=classes_4, output_dict=True, zero_division=0)
    report_text = classification_report(all_targets, all_preds, target_names=classes_4, digits=4, zero_division=0)
    print(report_text)

    # AUROC i AUPR dla zadania OOD
    auroc = float(roc_auc_score(binary_ground_truth, binary_fault_scores))
    p_vals, r_vals, _ = precision_recall_curve(binary_ground_truth, binary_fault_scores)
    aupr = float(auc(r_vals, p_vals))

    print("\n" + "="*65)
    print("METRYKI DETEKCJI OUT-OF-DISTRIBUTION (OOD):")
    print("="*65)
    print(f"  AUROC (Area Under ROC Curve):            {auroc*100:.2f}%")
    print(f"  AUPR  (Area Under PR Curve):             {aupr*100:.2f}%")
    print(f"  Wykrywanie 'normal' Czułość (Recall):     {report_dict['normal']['recall']*100:.2f}%")
    print(f"  Wykrywanie 'normal' Precyzja (Precision): {report_dict['normal']['precision']*100:.2f}%")
    print(f"  Wykrywanie 'normal' F1-Score:             {report_dict['normal']['f1-score']:.4f}")

    # Macierz pomyłek
    cm = confusion_matrix(all_targets, all_preds, labels=classes_4)
    cm_path = output_dir / "confusion_matrix_v2_4x4.png"
    plot_confusion_matrix_4x4(cm, classes_4, cm_path)
    print(f"\nZapisano macierz pomyłek do: {cm_path}")

    # Wykres histogramu score OOD
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for cname in classes_4:
        mask = (all_targets == cname)
        color = "green" if cname == "normal" else None
        ax.hist(binary_fault_scores[mask], bins=25, alpha=0.4, label=cname, density=True, color=color)
    ax.axvline(th_score, color="red", linestyle="--", linewidth=2, label=f"Próg OOD ({th_score:.3f})")
    ax.set_title("Rozkład Łącznego Wskaźnika OOD: Normal vs Awarie", fontweight="bold")
    ax.set_xlabel("Wskaźnik OOD (Wyższy = Awaria, Niższy = Normal)")
    ax.set_ylabel("Gęstość")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = output_dir / "ood_scores_histogram.png"
    plt.savefig(hist_path)
    plt.close()
    print(f"Zapisano histogram score OOD do: {hist_path}")

    summary = {
        "classification_report": report_dict,
        "auroc": auroc,
        "aupr": aupr,
        "thresholds": config["thresholds"],
        "confusion_matrix": cm.tolist()
    }
    with open(output_dir / "test_evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
