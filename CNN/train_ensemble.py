"""
HDD Acoustic Sentinel — Trening i Ewaluacja Komitetu Modeli (Seed Ensembling)
Single Source of Truth: ADR-001, AGENTS.md, NOTES.md

Moduł implementuje:
1. Trening N modeli 2D CNN o identycznej architekturze, ale z różnymi ziarnami losowości (seed).
2. Różną inicjalizację wag, różną kolejność próbek (shuffle) oraz różne maskowania SpecAugment/Dropout.
3. Inferencję komitetową (Soft Voting): uśrednianie wektorów prawdopodobieństw Softmax.
4. Ewaluację na zbiorze testowym z generowaniem macierzy pomyłek i raportu klasyfikacyjnego.
5. Zestawienie zysku: pojedyncze modele składowe vs Komitet (Ensemble).
6. Opcjonalną fuzję modeli do jednego grafu Keras (tf.keras.layers.Average) dla TFLite.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import tensorflow as tf

from model_cnn import build_hdd_cnn_model, compile_cnn_model, INPUT_SHAPE
from train_cnn import (
    resolve_dataset_dir,
    discover_classes_and_files,
    create_tf_dataset,
    calculate_balanced_class_weights,
    plot_and_save_confusion_matrix,
)


def train_single_ensemble_member(
    seed: int,
    epochs: int,
    train_files: List[Path],
    train_labels: List[int],
    val_files: List[Path],
    val_labels: List[int],
    num_classes: int,
    class_weights: Dict[int, float],
    output_dir: Path,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.3,
) -> Tuple[tf.keras.Model, Dict]:
    """
    Trenuje pojedynczy model składowy komitetu z unikalnym ziarnem losowości.
    """
    tf.random.set_seed(seed)
    np.random.seed(seed)

    print(f"\n--- Trening członka komitetu (seed = {seed}, epoki = {epochs}) ---")

    # Dedykowany dataset dla danego seeda (unikalny shuffle)
    train_ds = create_tf_dataset(
        train_files, train_labels, num_classes,
        batch_size=batch_size, is_training=True, seed=seed
    )
    val_ds = create_tf_dataset(
        val_files, val_labels, num_classes,
        batch_size=batch_size, is_training=False
    ) if val_files else None

    model = build_hdd_cnn_model(
        input_shape=INPUT_SHAPE,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        model_name=f"HddCNN_seed_{seed}"
    )
    model = compile_cnn_model(model, learning_rate=lr, label_mode="categorical")

    checkpoint_path = str(output_dir / f"ensemble_model_seed_{seed}.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="min",
            verbose=0
        ),
        tf.keras.callbacks.CSVLogger(
            filename=str(output_dir / f"history_seed_{seed}.csv"),
            separator=",",
            append=False
        ),
    ]

    t0 = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0
    )
    train_time = time.time() - t0

    # Załadowanie najlepszego checkpointu
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)

    stats = {
        "seed": seed,
        "epochs": epochs,
        "train_time_sec": train_time,
        "checkpoint_path": checkpoint_path,
        "final_loss": float(history.history["loss"][-1]),
        "final_val_loss": float(history.history["val_loss"][-1]) if "val_loss" in history.history else None,
        "best_val_loss": float(min(history.history["val_loss"])) if "val_loss" in history.history else None,
    }
    print(f"  [seed={seed}] Ukończono w {train_time:.1f}s | Min val_loss: {stats['best_val_loss']:.4f}")
    return model, stats


def evaluate_predictions(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: List[str]
) -> Dict:
    """Oblicza pełne metryki ewaluacyjne na podstawie wektora prawdopodobieństw."""
    y_pred = np.argmax(probabilities, axis=-1)
    rep = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    acc = float(np.mean(y_true == y_pred))
    return {
        "accuracy": acc,
        "weighted_f1": float(rep["weighted avg"]["f1-score"]),
        "weighted_precision": float(rep["weighted avg"]["precision"]),
        "weighted_recall": float(rep["weighted avg"]["recall"]),
        "per_class": {
            cls_name: {
                "precision": float(rep[cls_name]["precision"]),
                "recall": float(rep[cls_name]["recall"]),
                "f1": float(rep[cls_name]["f1-score"]),
                "support": int(rep[cls_name]["support"])
            }
            for cls_name in class_names
        },
        "y_pred": y_pred.tolist()
    }


def create_fused_ensemble_model(models: List[tf.keras.Model], model_name: str = "FusedHddEnsemble") -> tf.keras.Model:
    """
    Łączy listę modeli Keras w jeden nadrzędny graf z uśrednianiem (Average layer).
    Umożliwia pojedynczy eksport do .keras oraz konwersję do pojedynczego pliku .tflite.
    """
    inputs = tf.keras.layers.Input(shape=INPUT_SHAPE, name="audio_spectrogram_input")
    outputs = [m(inputs) for m in models]
    avg_output = tf.keras.layers.Average(name="ensemble_soft_voting")(outputs)
    fused_model = tf.keras.models.Model(inputs=inputs, outputs=avg_output, name=model_name)
    return fused_model


def run_ensemble_pipeline(
    seeds: List[int],
    epochs: int,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.3,
    fuse_model: bool = True
) -> Dict:
    """
    Główny potok Seed Ensemblingu.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TRENING I EWALUACJA KOMITETU MODELI (SEED ENSEMBLING)")
    print("=" * 70)
    print(f"Liczba modeli w komitecie: {len(seeds)}")
    print(f"Zestaw seedów:             {seeds}")
    print(f"Liczba epok na model:      {epochs}")
    print(f"Katalog wyjściowy:         {output_dir}")

    # Sprawdzenie GPU i włączenie dynamicznej alokacji pamięci
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Urządzenie obliczeniowe:   GPU ({gpus[0].name}) [memory_growth=True]")
        except Exception as e:
            print(f"Urządzenie obliczeniowe:   GPU ({gpus[0].name}) (set_memory_growth: {e})")
    else:
        print("Urządzenie obliczeniowe:   CPU")

    train_dir = data_dir / "train"
    val_dir = data_dir / "validate"
    test_dir = data_dir / "test"

    train_classes, train_files, train_labels = discover_classes_and_files(train_dir)
    val_classes, val_files, val_labels = discover_classes_and_files(val_dir, expected_classes=train_classes)
    test_classes, test_files, test_labels = discover_classes_and_files(test_dir, expected_classes=train_classes)

    class_names = train_classes
    num_classes = len(class_names)
    class_weights = calculate_balanced_class_weights(train_labels, num_classes)

    # Przygotowanie zbioru testowego do ewaluacji
    test_ds = create_tf_dataset(
        test_files, test_labels, num_classes,
        batch_size=batch_size, is_training=False
    )

    y_true_all: List[int] = []
    for _, y_b in test_ds:
        y_true_all.extend(np.argmax(y_b.numpy(), axis=-1).tolist())
    y_true_arr = np.array(y_true_all)

    # 1. Trening każdego modelu w komitecie
    trained_models: List[tf.keras.Model] = []
    member_stats: List[Dict] = []
    member_probs_list: List[np.ndarray] = []
    member_metrics_list: List[Dict] = []

    total_train_start = time.time()
    for s in seeds:
        model, stats = train_single_ensemble_member(
            seed=s,
            epochs=epochs,
            train_files=train_files,
            train_labels=train_labels,
            val_files=val_files,
            val_labels=val_labels,
            num_classes=num_classes,
            class_weights=class_weights,
            output_dir=models_dir,
            batch_size=batch_size,
            lr=lr,
            dropout_rate=dropout_rate,
        )
        trained_models.append(model)
        member_stats.append(stats)

        # Predykcje indywidualnego modelu
        probs_member = []
        for x_b, _ in test_ds:
            p = model.predict(x_b, verbose=0)
            probs_member.append(p)
        probs_member = np.concatenate(probs_member, axis=0)
        member_probs_list.append(probs_member)

        # Ewaluacja pojedynczego modelu
        met = evaluate_predictions(y_true_arr, probs_member, class_names)
        member_metrics_list.append(met)
        print(f"    -> Wynik pojedynczego seed={s}: Acc={met['accuracy']:.4f}, Weighted F1={met['weighted_f1']:.4f}")

    total_train_time = time.time() - total_train_start

    # 2. Fuzja predykcji komitetu (Soft Voting)
    print("\n" + "=" * 70)
    print("EWALUACJA KOMITETU (SOFT VOTING / PROBABILITY AVERAGING)")
    print("=" * 70)

    ensemble_probs = np.mean(member_probs_list, axis=0)
    ensemble_metrics = evaluate_predictions(y_true_arr, ensemble_probs, class_names)
    y_pred_ensemble = np.array(ensemble_metrics["y_pred"])

    # 3. Zapis macierzy pomyłek komitetu
    cm_path = str(output_dir / "ensemble_confusion_matrix.png")
    plot_and_save_confusion_matrix(
        y_true=y_true_arr,
        y_pred=y_pred_ensemble,
        class_names=class_names,
        output_path=cm_path,
        title=f"Macierz Pomyłek — Komitet {len(seeds)} Modeli ({epochs} epok)"
    )

    # 4. Opcjonalna fuzja do jednego modelu Keras
    fused_model_path = None
    if fuse_model and len(trained_models) > 1:
        fused = create_fused_ensemble_model(trained_models, model_name=f"HddEnsemble_{len(seeds)}models")
        fused_path = models_dir / "fused_ensemble_model.keras"
        fused.save(str(fused_path))
        fused_model_path = str(fused_path)
        print(f"[Model] Zapisano połączony graf Keras (Fused Ensemble): {fused_path}")

    # 5. Zestawienie wyników: Indywidualne vs Komitet
    print("\n" + "=" * 70)
    print("ZESTAWIENIE PORÓWNAWCZE: CZŁONKOWIE KOMITETU VS ENSEMBLE")
    print("=" * 70)
    header = f"{'Model':<20} | {'Test Acc':>9} | {'Weighted F1':>12} | {'Clicking F1':>12} | {'Scratch F1':>11}"
    print(header)
    print("-" * len(header))

    for idx, s in enumerate(seeds):
        m_met = member_metrics_list[idx]
        c_f1 = m_met["per_class"].get("clicking", {}).get("f1", 0.0)
        s_f1 = m_met["per_class"].get("scratching", {}).get("f1", 0.0)
        print(f"{f'Model (seed={s})':<20} | {m_met['accuracy']:>9.4f} | {m_met['weighted_f1']:>12.4f} | {c_f1:>12.4f} | {s_f1:>11.4f}")

    # Średnia pojedynczych modeli
    avg_ind_acc = float(np.mean([m["accuracy"] for m in member_metrics_list]))
    avg_ind_f1 = float(np.mean([m["weighted_f1"] for m in member_metrics_list]))
    print("-" * len(header))
    print(f"{'ŚREDNIA INDYWIDUALNA':<20} | {avg_ind_acc:>9.4f} | {avg_ind_f1:>12.4f} | {'-':>12} | {'-':>11}")

    # Wynik komitetu
    ens_c_f1 = ensemble_metrics["per_class"].get("clicking", {}).get("f1", 0.0)
    ens_s_f1 = ensemble_metrics["per_class"].get("scratching", {}).get("f1", 0.0)
    print(f"{'🏆 KOMITET (ENSEMBLE)':<20} | {ensemble_metrics['accuracy']:>9.4f} | {ensemble_metrics['weighted_f1']:>12.4f} | {ens_c_f1:>12.4f} | {ens_s_f1:>11.4f}")

    acc_gain = ensemble_metrics["accuracy"] - avg_ind_acc
    f1_gain = ensemble_metrics["weighted_f1"] - avg_ind_f1
    print(f"\nZYSK Z ENSEMBLINGU (vs średnia pojedynczych):")
    print(f"  Δ Dokładność (Accuracy): {acc_gain:+.4f} ({acc_gain*100:+.2f} p.p.)")
    print(f"  Δ Weighted F1-Score:    {f1_gain:+.4f} ({f1_gain*100:+.2f} p.p.)")

    # 6. Zapis raportu do JSON
    report_data = {
        "ensemble_config": {
            "num_models": len(seeds),
            "seeds": seeds,
            "epochs_per_model": epochs,
            "total_train_time_sec": total_train_time,
            "fused_model_path": fused_model_path
        },
        "individual_models": [
            {
                "seed": s,
                "stats": member_stats[i],
                "metrics": {
                    "accuracy": member_metrics_list[i]["accuracy"],
                    "weighted_f1": member_metrics_list[i]["weighted_f1"],
                    "per_class": member_metrics_list[i]["per_class"]
                }
            }
            for i, s in enumerate(seeds)
        ],
        "ensemble_results": {
            "accuracy": ensemble_metrics["accuracy"],
            "weighted_f1": ensemble_metrics["weighted_f1"],
            "weighted_precision": ensemble_metrics["weighted_precision"],
            "weighted_recall": ensemble_metrics["weighted_recall"],
            "per_class": ensemble_metrics["per_class"],
            "gain_vs_single_avg": {
                "accuracy_gain": acc_gain,
                "f1_gain": f1_gain
            }
        }
    }

    report_json_path = output_dir / "ensemble_summary.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Artefakt] Raport numeryczny komitetu: {report_json_path}")

    # Wykres porównawczy F1 per klasa
    plot_ensemble_comparison_chart(
        member_metrics=member_metrics_list,
        ensemble_metrics=ensemble_metrics,
        seeds=seeds,
        class_names=class_names,
        output_path=output_dir / "ensemble_comparison_bars.png"
    )

    return report_data


def plot_ensemble_comparison_chart(
    member_metrics: List[Dict],
    ensemble_metrics: Dict,
    seeds: List[int],
    class_names: List[str],
    output_path: Path
):
    """Generuje wykres słupkowy F1-Score dla każdej klasy: Indywidualne vs Komitet."""
    x = np.arange(len(class_names))
    width = 0.8 / (len(seeds) + 1)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for i, s in enumerate(seeds):
        f1s = [member_metrics[i]["per_class"].get(c, {}).get("f1", 0.0) for c in class_names]
        ax.bar(x + i * width, f1s, width, label=f"Model (seed={s})", alpha=0.7)

    # Komitet
    ens_f1s = [ensemble_metrics["per_class"].get(c, {}).get("f1", 0.0) for c in class_names]
    ax.bar(x + len(seeds) * width, ens_f1s, width, label="🏆 Komitet (Ensemble)", color="crimson", edgecolor="black", linewidth=1.5)

    ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_title("Porównanie F1-Score per klasa: Modele Składowe vs Komitet", fontsize=12, fontweight="bold")
    ax.set_xticks(x + (len(seeds) * width) / 2)
    ax.set_xticklabels(class_names, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"[Artefakt] Wykres słupkowy zysku komitetu: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Trening i ewaluacja komitetu modeli HDD CNN (Seed Ensembling).")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 123, 777],
        help="Lista ziaren losowości (domyślnie: 42 123 777)."
    )
    parser.add_argument(
        "--epochs", type=int, default=30,
        help="Liczba epok treningowych dla każdego członka komitetu (domyślnie: 30)."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Katalog ze zbiorem danych dataset_encoded."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./ensemble_output",
        help="Katalog wyjściowy na modele i raporty komitetu."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Rozmiar batcha (domyślnie: 32)."
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Początkowy Learning Rate (domyślnie: 0.001)."
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3,
        help="Współczynnik dropoutu (domyślnie: 0.3)."
    )
    parser.add_argument(
        "--no-fuse", action="store_true",
        help="Pomiń tworzenie połączonego modelu Keras (Fused Ensemble)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_directory = resolve_dataset_dir(args.data_dir)
    out_dir = Path(args.output_dir)

    run_ensemble_pipeline(
        seeds=args.seeds,
        epochs=args.epochs,
        data_dir=data_directory,
        output_dir=out_dir,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_rate=args.dropout,
        fuse_model=not args.no_fuse
    )
