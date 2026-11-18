#!/usr/bin/env python3
"""
calibrate_smpc.py
Kalibracja progu Out-Of-Distribution (OOD) dla modelu SMPC na zbiorze walidacyjnym awarii.

Single Source of Truth: ADR-001, AGENTS.md, PROJECT.md, SCOPE.md
Zasada działania:
1. Wczytuje najlepsze wagi best_smpc_model.pt.
2. Przeprowadza ewaluację na 131 próbkach dataset_encoded/validate (Zero Normal Leakage).
3. Dla każdej próbki wylicza ograniczony wynik kompozytowy OOD:
       S_OOD(x) = max_k cos(theta_k) + 0.10 * logsumexp(l_k / 2.0)
4. Ustala próg tau na poziomie 10. percentyla (dla target_recall = 0.90, gwarantując
   90% czułości na znane awarie na walidacji).
5. Zapisuje konfigurację detektora do smpc_ood_config.json.
"""

import argparse
import json
import sys
from pathlib import Path

# Zapewnienie dostępu do głównego katalogu repozytorium w sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from PrototypBezNormal.dataset_no_normal import FaultOnlyNpyDataset, FAULT_CLASSES_3
from PrototypBezNormal.model_smpc import AcousticSMPCModel


def parse_args():
    parser = argparse.ArgumentParser(description="Kalibracja progu OOD dla modelu SMPC")
    parser.add_argument("--weights", type=str, default="PrototypBezNormal/output_smpc/best_smpc_model.pt", help="Ścieżka do wag modelu")
    parser.add_argument("--output-dir", type=str, default="PrototypBezNormal/output_smpc", help="Katalog wyjściowy konfiguracji")
    parser.add_argument("--dataset-dir", type=str, default="dataset_encoded", help="Ścieżka do dataset_encoded")
    parser.add_argument("--target-recall", type=float, default=0.90, help="Docelowa czułość na awarie walidacyjne (domyślnie 90%)")
    parser.add_argument("--temperature", type=float, default=2.0, help="Temperatura energii w OOD")
    parser.add_argument("--alpha-energy", type=float, default=0.10, help="Współczynnik wagi składnika energii")
    parser.add_argument("--device", type=str, default="auto", help="Urządzenie (auto/cuda/cpu)")
    return parser.parse_args()


def calibrate():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"===========================================================")
    print(f" Kalibracja Progu OOD dla Modelu SMPC")
    print(f" Urządzenie: {device}")
    print(f" Wagi: {args.weights}")
    print(f" Docelowa czułość awarii (Target Recall): {args.target_recall * 100:.1f}%")
    print(f"===========================================================")

    # Wczytanie wag
    ckpt = torch.load(args.weights, map_location=device, weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict"))

    embedding_dim = ckpt.get("embedding_dim", 128)
    sub_centers = ckpt.get("sub_centers", 2)
    num_classes = ckpt.get("num_classes", len(FAULT_CLASSES_3))
    scale = ckpt.get("scale", 16.0)
    margin = ckpt.get("margin", 0.20)

    model = AcousticSMPCModel(
        num_fault_classes=num_classes,
        embedding_dim=embedding_dim,
        sub_centers=sub_centers,
        scale=scale
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Wczytanie walidacyjnego zbioru awarii (ŚCIŚLE ZERO normal)
    data_root = Path(args.dataset_dir)
    val_ds = FaultOnlyNpyDataset(
        data_root / "validate",
        include_normal=False,
        augment=False,
        preload_to_ram=True
    )

    # Niezmiennik braku wycieku danych (Zero Data Leakage Invariant)
    assert all(s[2] != "normal" for s in val_ds.samples), "BŁĄD: Wykryto normal w zbiorze walidacyjnym!"
    print(f"Załadowano próbki walidacyjne awarii: {len(val_ds)} (0 normal)")

    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    ood_scores = []
    cos_scores = []
    energy_scores = []
    per_class_scores = {c: [] for c in FAULT_CLASSES_3}

    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            logits, norm_embed = model(batch_x)
            cos_sims = model.compute_cosine_similarities(norm_embed)  # (B, 3)

            # S_cos = max_k cos(theta_k)
            max_cos, _ = torch.max(cos_sims, dim=-1)
            # S_energy = logsumexp(logits / T)
            energy = torch.logsumexp(logits / args.temperature, dim=-1)

            composite = max_cos + args.alpha_energy * energy

            comp_np = composite.cpu().numpy()
            cos_np = max_cos.cpu().numpy()
            energy_np = energy.cpu().numpy()
            y_np = batch_y.cpu().numpy()

            for i in range(len(comp_np)):
                sc = float(comp_np[i])
                ood_scores.append(sc)
                cos_scores.append(float(cos_np[i]))
                energy_scores.append(float(energy_np[i]))
                c_name = FAULT_CLASSES_3[y_np[i]]
                per_class_scores[c_name].append(sc)

    ood_scores = np.array(ood_scores, dtype=np.float32)

    # Obliczenie progu: dla target_recall = 0.90 próg to 10. percentyl
    percentile_q = (1.0 - args.target_recall) * 100.0
    threshold = float(np.percentile(ood_scores, percentile_q))

    # Statystyki wyników
    stats = {
        "count": len(ood_scores),
        "mean": float(np.mean(ood_scores)),
        "std": float(np.std(ood_scores)),
        "min": float(np.min(ood_scores)),
        "max": float(np.max(ood_scores)),
        "median": float(np.median(ood_scores)),
        "p05": float(np.percentile(ood_scores, 5.0)),
        "p10": float(np.percentile(ood_scores, 10.0)),
        "p25": float(np.percentile(ood_scores, 25.0)),
        "p75": float(np.percentile(ood_scores, 75.0)),
        "p90": float(np.percentile(ood_scores, 90.0)),
        "threshold": threshold
    }

    per_class_stats = {}
    for c_name, scores in per_class_scores.items():
        if len(scores) > 0:
            per_class_stats[c_name] = {
                "count": len(scores),
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
                "min": float(np.min(scores)),
                "max": float(np.max(scores)),
                "above_threshold_ratio": float(np.mean(np.array(scores) >= threshold))
            }

    print(f"\n--- Wyniki kalibracji OOD ---")
    print(f"Liczba próbek walidacji: {len(ood_scores)}")
    print(f"Średni wynik OOD: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"Zakres OOD: [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"Skalibrowany próg tau (10. percentyl): {threshold:.4f}")
    print(f"\nZachowanie per klasa na walidacji:")
    for c_name, c_s in per_class_stats.items():
        print(f"  {c_name:12s}: N={c_s['count']:2d} | Mean={c_s['mean']:.4f} | Recall: {c_s['above_threshold_ratio']*100:.1f}%")

    config = {
        "model_architecture": "AcousticSMPCModel",
        "threshold": threshold,
        "target_recall": args.target_recall,
        "percentile": percentile_q,
        "scale": scale,
        "margin": margin,
        "temperature": args.temperature,
        "alpha_energy": args.alpha_energy,
        "num_classes": num_classes,
        "sub_centers": sub_centers,
        "embedding_dim": embedding_dim,
        "classes_3": FAULT_CLASSES_3,
        "classes_4": ["beeping", "clicking", "normal", "scratching"],
        "weights_path": str(Path(args.weights).name),
        "val_stats": stats,
        "per_class_stats": per_class_stats
    }

    config_path = output_dir / "smpc_ood_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"\nZapisano konfigurację OOD do: {config_path}")

    # Wykres rozkładu OOD z zaznaczonym progiem
    plt.figure(figsize=(8, 4))
    plt.hist(ood_scores, bins=25, color="#1f77b4", alpha=0.7, edgecolor="black", label="Val Fault Scores")
    plt.axvline(threshold, color="#d62728", linestyle="--", linewidth=2, label=f"Threshold tau = {threshold:.4f} (Recall 90%)")
    plt.xlabel("Kompozytowy wynik OOD: S_OOD = max cos + 0.10 * logsumexp")
    plt.ylabel("Liczba próbek")
    plt.title(f"Kalibracja progu OOD (131 próbek awarii, Zero Normal)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "smpc_ood_calibration_hist.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()
    print(f"Zapisano wykres kalibracji do: {plot_path}")


if __name__ == "__main__":
    calibrate()
