"""
HDD Acoustic Sentinel — Porównanie Komitetów Modeli (Ensembles) z Modelem Bazowym
Single Source of Truth: ADR-001, AGENTS.md, NOTES.md

Skrypt:
1. Trenuje na GPU komitety modeli (Seed Ensembles z seeds=[42, 123, 777]) dla wybranych epok (np. 25, 30, 40).
2. Wczytuje i ewaluuje model bazowy ze stress testu (training_output/best_hdd_cnn_model.keras).
3. Porównuje wszystkie konfiguracje komitetów z modelem bazowym.
4. Generuje zbiorczy raport numeryczny (JSON, CSV) oraz wielopanelowy wykres (PNG).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import tensorflow as tf

from model_cnn import INPUT_SHAPE
from train_cnn import (
    resolve_dataset_dir,
    discover_classes_and_files,
    create_tf_dataset,
)
from train_ensemble import run_ensemble_pipeline, evaluate_predictions


def evaluate_single_keras_model(
    model_path: Path,
    test_ds: tf.data.Dataset,
    y_true: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Ewaluuje pojedynczy zapisany model Keras na zbiorze testowym."""
    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku modelu: {model_path}")
    
    print(f"Wczytywanie modelu bazowego: {model_path}")
    model = tf.keras.models.load_model(str(model_path))

    probs_list = []
    for x_b, _ in test_ds:
        p = model.predict(x_b, verbose=0)
        probs_list.append(p)
    probs = np.concatenate(probs_list, axis=0)

    metrics = evaluate_predictions(y_true, probs, class_names)
    del model
    tf.keras.backend.clear_session()
    return metrics


def run_full_comparison(
    epoch_configs: List[int],
    seeds: List[int],
    baseline_model_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.3
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("WIELOWYMIAROWE PORÓWNANIE KOMITETÓW MODELI Z MODELEM BAZOWYM ZE STRESS TESTU")
    print("=" * 75)
    print(f"Konfiguracje epok dla komitetów: {epoch_configs}")
    print(f"Ziarna losowości (Seeds):       {seeds} (N = {len(seeds)})")
    print(f"Model bazowy (Stress Test):     {baseline_model_path}")
    print(f"Katalog danych:                 {data_dir}")
    print(f"Katalog wyjściowy:              {output_dir}")

    # Sprawdzenie GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print(f"Urządzenie obliczeniowe:        GPU ({gpus[0].name}) [memory_growth=True]")
        except Exception as e:
            print(f"Urządzenie obliczeniowe:        GPU ({gpus[0].name}) ({e})")
    else:
        print("Urządzenie obliczeniowe:        CPU")

    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    train_classes, _, _ = discover_classes_and_files(train_dir)
    test_classes, test_files, test_labels = discover_classes_and_files(test_dir, expected_classes=train_classes)

    class_names = train_classes
    num_classes = len(class_names)

    test_ds = create_tf_dataset(
        test_files, test_labels, num_classes,
        batch_size=batch_size, is_training=False
    )
    y_true_all: List[int] = []
    for _, y_b in test_ds:
        y_true_all.extend(np.argmax(y_b.numpy(), axis=-1).tolist())
    y_true_arr = np.array(y_true_all)

    # 1. Ewaluacja modelu bazowego ze stress testu
    print("\n[1/4] Ewaluacja Modelu Bazowego ze Stress Testu...")
    baseline_metrics = evaluate_single_keras_model(
        model_path=baseline_model_path,
        test_ds=test_ds,
        y_true=y_true_arr,
        class_names=class_names
    )
    print(f"  -> Model Bazowy: Accuracy = {baseline_metrics['accuracy']:.4f}, Weighted F1 = {baseline_metrics['weighted_f1']:.4f}")
    for c in class_names:
        print(f"     {c:15s}: F1 = {baseline_metrics['per_class'][c]['f1']:.4f}")

    # 2. Trening i ewaluacja komitetów dla każdej konfiguracji epok
    ensemble_results: Dict[int, Dict] = {}
    for ep in epoch_configs:
        print(f"\n[2/4] Trening i ewaluacja Komitetu: {ep} epok...")
        ep_output_dir = output_dir / f"ensemble_{ep}_epochs"
        ens_data = run_ensemble_pipeline(
            seeds=seeds,
            epochs=ep,
            data_dir=data_dir,
            output_dir=ep_output_dir,
            batch_size=batch_size,
            lr=lr,
            dropout_rate=dropout_rate,
            fuse_model=True
        )
        ensemble_results[ep] = ens_data

    # 3. Zestawienie i analiza zysków
    print("\n" + "=" * 85)
    print("ZESTAWIENIE KOŃCOWE: MODEL BAZOWY VS KOMITETY MODELI (ENSEMBLES)")
    print("=" * 85)
    header = f"{'Wariant':<24} | {'Test Acc':>9} | {'Weighted F1':>12} | {'Clicking F1':>12} | {'Scratch F1':>11} | {'Beep F1':>8}"
    print(header)
    print("-" * len(header))

    # Wiersz modelu bazowego
    b_click = baseline_metrics["per_class"].get("clicking", {}).get("f1", 0.0)
    b_scratch = baseline_metrics["per_class"].get("scratching", {}).get("f1", 0.0)
    b_beep = baseline_metrics["per_class"].get("beeping", {}).get("f1", 0.0)
    print(f"{'Bazowy (Stress Test)':<24} | {baseline_metrics['accuracy']:>9.4f} | {baseline_metrics['weighted_f1']:>12.4f} | {b_click:>12.4f} | {b_scratch:>11.4f} | {b_beep:>8.4f}")
    print("-" * len(header))

    # Wiersze komitetów
    comparison_rows = []
    comparison_rows.append({
        "variant": "Bazowy (Stress Test)",
        "accuracy": baseline_metrics["accuracy"],
        "weighted_f1": baseline_metrics["weighted_f1"],
        "clicking_f1": b_click,
        "scratching_f1": b_scratch,
        "beeping_f1": b_beep,
        "normal_f1": baseline_metrics["per_class"].get("normal", {}).get("f1", 0.0)
    })

    best_variant = "Bazowy"
    best_f1 = baseline_metrics["weighted_f1"]

    for ep in epoch_configs:
        e_res = ensemble_results[ep]["ensemble_results"]
        c_f1 = e_res["per_class"].get("clicking", {}).get("f1", 0.0)
        s_f1 = e_res["per_class"].get("scratching", {}).get("f1", 0.0)
        bp_f1 = e_res["per_class"].get("beeping", {}).get("f1", 0.0)
        n_f1 = e_res["per_class"].get("normal", {}).get("f1", 0.0)
        v_name = f"Komitet {ep} epok (N={len(seeds)})"

        if e_res["weighted_f1"] > best_f1:
            best_f1 = e_res["weighted_f1"]
            best_variant = v_name

        print(f"{v_name:<24} | {e_res['accuracy']:>9.4f} | {e_res['weighted_f1']:>12.4f} | {c_f1:>12.4f} | {s_f1:>11.4f} | {bp_f1:>8.4f}")

        comparison_rows.append({
            "variant": v_name,
            "epochs": ep,
            "accuracy": e_res["accuracy"],
            "weighted_f1": e_res["weighted_f1"],
            "clicking_f1": c_f1,
            "scratching_f1": s_f1,
            "beeping_f1": bp_f1,
            "normal_f1": n_f1,
            "delta_f1_vs_baseline": e_res["weighted_f1"] - baseline_metrics["weighted_f1"],
            "delta_acc_vs_baseline": e_res["accuracy"] - baseline_metrics["accuracy"]
        })

    print("-" * len(header))
    print(f"\n🏆 Najwyższy wynik: {best_variant} z Weighted F1 = {best_f1:.4f}")

    # 4. Zapis do CSV
    csv_path = output_dir / "tabela_porownawcza_komitety.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("variant,accuracy,weighted_f1,clicking_f1,scratching_f1,beeping_f1,normal_f1,delta_f1_vs_baseline\n")
        for r in comparison_rows:
            d_f1 = r.get("delta_f1_vs_baseline", 0.0)
            f.write(f"{r['variant']},{r['accuracy']:.6f},{r['weighted_f1']:.6f},{r['clicking_f1']:.6f},{r['scratching_f1']:.6f},{r['beeping_f1']:.6f},{r['normal_f1']:.6f},{d_f1:+.6f}\n")
    print(f"[Artefakt] Zapisano tabelę CSV: {csv_path}")

    # 5. Zapis do JSON
    json_path = output_dir / "raport_porownawczy_komitety.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_model": {
                "path": str(baseline_model_path),
                "metrics": baseline_metrics
            },
            "ensemble_configs": epoch_configs,
            "seeds": seeds,
            "best_variant": best_variant,
            "best_weighted_f1": best_f1,
            "comparison": comparison_rows
        }, f, indent=2, ensure_ascii=False)
    print(f"[Artefakt] Zapisano raport JSON: {json_path}")

    # 6. Generowanie wielopanelowego wykresu porównawczego
    plot_comparison_dashboard(
        comparison_rows=comparison_rows,
        class_names=class_names,
        output_path=output_dir / "komitety_vs_bazowy_wykres.png"
    )

    print("\n" + "=" * 75)
    print("PEŁNE PORÓWNANIE UKOŃCZONE POMYŚLNIE!")
    print(f"Wszystkie wyniki i wykresy w: {output_dir.resolve()}")
    print("=" * 75)


def plot_comparison_dashboard(
    comparison_rows: List[Dict],
    class_names: List[str],
    output_path: Path
):
    """Tworzy 4-panelowy pulpit diagnostyczny porównujący komitety z modelem bazowym."""
    variants = [r["variant"] for r in comparison_rows]
    short_labels = [v.replace(" (N=3)", "").replace(" (Stress Test)", "") for v in variants]
    accuracies = [r["accuracy"] for r in comparison_rows]
    weighted_f1s = [r["weighted_f1"] for r in comparison_rows]

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=150)

    # Panel 1: Ogólna dokładność i Weighted F1
    ax1 = axes[0, 0]
    x = np.arange(len(variants))
    w = 0.35
    ax1.bar(x - w/2, accuracies, w, label="Accuracy", color="steelblue", alpha=0.85)
    ax1.bar(x + w/2, weighted_f1s, w, label="Weighted F1", color="darkorange", alpha=0.85)
    ax1.set_ylabel("Wartość metryki", fontsize=11, fontweight="bold")
    ax1.set_title("Dokładność (Accuracy) i Weighted F1-Score", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax1.set_ylim(0.5, 1.0)
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.legend(fontsize=9)
    for i in x:
        ax1.text(i - w/2, accuracies[i] + 0.01, f"{accuracies[i]:.3f}", ha="center", fontsize=8)
        ax1.text(i + w/2, weighted_f1s[i] + 0.01, f"{weighted_f1s[i]:.3f}", ha="center", fontsize=8)

    # Panel 2: F1-Score per klasa
    ax2 = axes[0, 1]
    cls_colors = {"beeping": "#1f77b4", "clicking": "#2ca02c", "normal": "#9467bd", "scratching": "#d62728"}
    w_cls = 0.8 / len(class_names)
    for j, c in enumerate(class_names):
        f1_vals = [r.get(f"{c}_f1", 0.0) for r in comparison_rows]
        ax2.bar(x + (j - len(class_names)/2 + 0.5) * w_cls, f1_vals, w_cls,
                label=f"Klasa: {c}", color=cls_colors.get(c, None), alpha=0.85)
    ax2.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax2.set_title("F1-Score per klasa awarii HDD", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax2.set_ylim(0.0, 1.1)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.legend(fontsize=8, loc="lower right")

    # Panel 3: Kluczowa klasa "clicking" (Stukanie głowicy)
    ax3 = axes[1, 0]
    click_f1s = [r["clicking_f1"] for r in comparison_rows]
    colors_c = ["dimgray"] + ["forestgreen" if f >= click_f1s[0] else "indianred" for f in click_f1s[1:]]
    bars = ax3.bar(x, click_f1s, 0.5, color=colors_c, alpha=0.85, edgecolor="black")
    ax3.axhline(0.60, color="red", linestyle="--", linewidth=1.5, label="Próg akceptacji (0.60)")
    ax3.axhline(click_f1s[0], color="dimgray", linestyle=":", linewidth=1.5, label=f"Baza: {click_f1s[0]:.3f}")
    ax3.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax3.set_title("Czułość na stukanie głowicy (Klasa: clicking)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax3.set_ylim(0.5, 1.0)
    ax3.grid(True, alpha=0.3, axis="y")
    ax3.legend(fontsize=8)
    for b in bars:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.01, f"{b.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

    # Panel 4: Zysk punktowy vs Model Bazowy (+Δ F1)
    ax4 = axes[1, 1]
    deltas_f1 = [r.get("delta_f1_vs_baseline", 0.0) * 100 for r in comparison_rows]
    colors_d = ["gray"] + ["green" if d >= 0 else "crimson" for d in deltas_f1[1:]]
    b4 = ax4.bar(x, deltas_f1, 0.5, color=colors_d, alpha=0.85, edgecolor="black")
    ax4.axhline(0.0, color="black", linewidth=1)
    ax4.set_ylabel("Różnica F1 (punkty procentowe p.p.)", fontsize=11, fontweight="bold")
    ax4.set_title("Zysk z Ensemblingu (Soft Voting) vs Model Bazowy", fontsize=12, fontweight="bold")
    ax4.set_xticks(x)
    ax4.set_xticklabels(short_labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax4.grid(True, alpha=0.3, axis="y")
    for b, d in zip(b4, deltas_f1):
        y_pos = b.get_height() + (0.3 if d >= 0 else -0.8)
        ax4.text(b.get_x() + b.get_width()/2, y_pos, f"{d:+.2f} p.p.", ha="center", fontsize=9, fontweight="bold")

    plt.suptitle("HDD Acoustic Fault Detection: Komitet Modeli (Ensemble) vs Model Bazowy",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Artefakt] Wykres porównawczy zapisano w: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Porównanie komitetów modeli z modelem bazowym.")
    parser.add_argument(
        "--epoch-configs", type=int, nargs="+", default=[25, 30, 40],
        help="Liczby epok dla kolejnych komitetów (domyślnie: 25 30 40)."
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 777],
        help="Ziarna losowości dla komitetu (domyślnie: 42 123 777)."
    )
    parser.add_argument(
        "--baseline-model", type=str, default="training_output/best_hdd_cnn_model.keras",
        help="Ścieżka do zapisanego modelu bazowego ze stress testu."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Katalog ze zbiorem danych dataset_encoded."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./ensemble_comparison_output",
        help="Katalog wyjściowy na wyniki porównania."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Rozmiar batcha (domyślnie: 32)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_directory = resolve_dataset_dir(args.data_dir)
    out_dir = Path(args.output_dir)
    base_model = Path(args.baseline_model)

    run_full_comparison(
        epoch_configs=args.epoch_configs,
        seeds=args.seeds,
        baseline_model_path=base_model,
        data_dir=data_directory,
        output_dir=out_dir,
        batch_size=args.batch_size
    )
