#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moduł treningu i ewaluacji komitetu modeli (Seed Ensembling) dla systemu
HDD Acoustic Fault Detection (Acoustic HDD Sentinel).

Realizuje założenia ADR-001 oraz wytyczne Seed Ensembling (NOTES.md):
- Niezmiennik wejścia: tensor (128, 259, 3) float32 [Log-Mel, Delta, Delta-Delta]
- Trening N modeli 2D CNN z różnymi ziarnami losowości (seeds)
- Inferencja komitetowa metodą Soft Voting (Probability Averaging)
- Opcjonalna fuzja w pojedynczy graf Keras (fused_ensemble_model.keras) pod TFLite
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

# Importy z istniejących modułów projektu zgodnie ze specyfikacją
from model_cnn import (
    INPUT_SHAPE,
    build_hdd_cnn_model,
    compile_cnn_model,
)
from train_cnn import (
    calculate_balanced_class_weights,
    create_tf_dataset,
    discover_classes_and_files,
    plot_and_save_confusion_matrix,
    resolve_dataset_dir,
)

EXPECTED_INPUT_SHAPE: Tuple[int, int, int] = (128, 259, 3)


# ==============================================================================
# 1. KONFIGURACJA SPRZĘTOWA I DETERMINIZM (GPU ROCm / CUDA + SEEDS)
# ==============================================================================

def configure_gpu_memory_growth() -> None:
    """
    Konfiguruje dynamiczną alokację pamięci VRAM dla kart GPU (AMD ROCm / NVIDIA CUDA),
    zapobiegając natychmiastowemu zajęciu 100% pamięci przez proces TensorFlow.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("[INFO] Nie wykryto akceleratorów GPU. Trening odbędzie się na CPU.")
        return

    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[INFO] Włączono bezpieczny wzrost pamięci VRAM (memory_growth) dla: {gpu.name}")
        except RuntimeError as err:
            print(f"[WARN] Nie udało się ustawić memory_growth dla {gpu.name}: {err}")


def set_global_seed(seed: int) -> None:
    """
    Ustawia ziarno losowości dla Python, NumPy oraz TensorFlow.
    Gwarantuje odmienną inicjalizację wag, kolejność batchy (shuffle)
    oraz losowe maskowanie w locie (SpecAugment, SpatialDropout2D) dla każdego modelu.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)


def verify_adr001_invariants() -> None:
    """Weryfikuje zgodność stałej INPUT_SHAPE z niezmiennikiem ADR-001 (128, 259, 3)."""
    if tuple(INPUT_SHAPE) != EXPECTED_INPUT_SHAPE:
        raise ValueError(
            f"[ADR-001 VIOLATION] Oczekiwano INPUT_SHAPE={EXPECTED_INPUT_SHAPE}, "
            f"otrzymano {INPUT_SHAPE}."
        )


# ==============================================================================
# 2. ADAPTERY INTEGRACYJNE DLA MODUŁÓW PROJEKTU (train_cnn.py & model_cnn.py)
# ==============================================================================

def load_split_metadata(
    dataset_root: Path,
    split_name: str,
    reference_classes: Optional[List[str]] = None,
) -> Tuple[List[str], List[str], np.ndarray]:
    """
    Odczytuje pliki i etykiety dla wskazanego podzbioru (train / validate / test)
    przy użyciu funkcji `discover_classes_and_files` z `train_cnn.py`.
    """
    split_dir = dataset_root / split_name
    if not split_dir.exists() or not split_dir.is_dir():
        raise FileNotFoundError(
            f"Brak katalogu podziału '{split_name}' w ścieżce: {split_dir}"
        )

    sig = inspect.signature(discover_classes_and_files)
    params = list(sig.parameters.keys())

    if len(params) >= 2 and reference_classes is not None:
        result = discover_classes_and_files(split_dir, reference_classes)
    else:
        result = discover_classes_and_files(split_dir)

    # Obsługa kolejności zwracanych krotek z train_cnn.discover_classes_and_files
    if isinstance(result, tuple) and len(result) == 3:
        a, b, c = result
        # Rozpoznanie który element to class_names vs file_paths vs labels
        if (
            isinstance(a, list)
            and len(a) > 0
            and isinstance(a[0], str)
            and not a[0].endswith(".npy")
            and not Path(a[0]).exists()
        ):
            class_names = [str(x) for x in a]
            file_paths = [str(x) for x in b]
            labels = np.asarray(c, dtype=np.int32)
        else:
            file_paths = [str(x) for x in a]
            labels = np.asarray(b, dtype=np.int32)
            class_names = [str(x) for x in c]
    else:
        raise RuntimeError(
            "Funkcja `discover_classes_and_files` zwróciła nieoczekiwany format danych."
        )

    return class_names, file_paths, labels


def build_dataset_pipeline(
    file_paths: List[str],
    labels: np.ndarray,
    num_classes: int,
    batch_size: int,
    is_training: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Tworzy potok `tf.data.Dataset` za pomocą `create_tf_dataset` z `train_cnn.py`,
    dopasowując argumenty wywołania i weryfikując niezmiennik kształtu (128, 259, 3).
    """
    sig = inspect.signature(create_tf_dataset)
    kwargs: Dict[str, Any] = {}

    param_names = set(sig.parameters.keys())
    if "batch_size" in param_names:
        kwargs["batch_size"] = batch_size
    if "is_training" in param_names:
        kwargs["is_training"] = is_training
    elif "training" in param_names:
        kwargs["training"] = is_training
    elif "shuffle" in param_names:
        kwargs["shuffle"] = is_training

    if "num_classes" in param_names:
        kwargs["num_classes"] = num_classes
    if "seed" in param_names:
        kwargs["seed"] = seed
    elif "shuffle_seed" in param_names:
        kwargs["shuffle_seed"] = seed

    ds = create_tf_dataset(file_paths, labels, **kwargs)

    # Weryfikacja ADR-001 na wyjściu datasetu
    element_spec = ds.element_spec
    if isinstance(element_spec, tuple) and len(element_spec) >= 2:
        x_spec = element_spec[0]
        if x_spec.shape.rank == 4 and tuple(x_spec.shape[1:]) != EXPECTED_INPUT_SHAPE:
            raise ValueError(
                f"[ADR-001 VIOLATION] Potok tf.data zwraca kształt {x_spec.shape[1:]}, "
                f"wymagany: {EXPECTED_INPUT_SHAPE}"
            )
    return ds


def compute_sqrt_class_weights(
    train_labels: np.ndarray,
    num_classes: int,
) -> Dict[int, float]:
    """
    Oblicza wagi klas przy użyciu `calculate_balanced_class_weights` z `train_cnn.py`.
    Gwarantuje zgodność z regułą sqrt-balanced: np.clip(np.sqrt(w), 0.5, 3.5).
    """
    sig = inspect.signature(calculate_balanced_class_weights)
    if len(sig.parameters) >= 2:
        weights = calculate_balanced_class_weights(train_labels, num_classes)
    else:
        weights = calculate_balanced_class_weights(train_labels)

    if isinstance(weights, dict):
        return {int(k): float(np.clip( v, 0.5, 3.5 )) for k, v in weights.items()}

    weights_arr = np.asarray(weights, dtype=np.float32)
    return {i: float(np.clip(weights_arr[i], 0.5, 3.5)) for i in range(num_classes)}


def instantiate_and_compile_model(
    num_classes: int,
    dropout_rate: float,
    learning_rate: float,
) -> tf.keras.Model:
    """
    Buduje i kompiluje model 2D CNN przy użyciu funkcji z `model_cnn.py`.
    """
    build_sig = inspect.signature(build_hdd_cnn_model)
    build_kwargs: Dict[str, Any] = {}

    if "input_shape" in build_sig.parameters:
        build_kwargs["input_shape"] = INPUT_SHAPE
    if "num_classes" in build_sig.parameters:
        build_kwargs["num_classes"] = num_classes
    if "dropout_rate" in build_sig.parameters:
        build_kwargs["dropout_rate"] = dropout_rate
    elif "dropout" in build_sig.parameters:
        build_kwargs["dropout"] = dropout_rate

    model = build_hdd_cnn_model(**build_kwargs)

    compile_sig = inspect.signature(compile_cnn_model)
    compile_kwargs: Dict[str, Any] = {}
    if "learning_rate" in compile_sig.parameters:
        compile_kwargs["learning_rate"] = learning_rate
    elif "lr" in compile_sig.parameters:
        compile_kwargs["lr"] = learning_rate

    compiled_model = compile_cnn_model(model, **compile_kwargs)
    return compiled_model if compiled_model is not None else model


# ==============================================================================
# 3. TRENING, EWALUACJA I FUZJA KOMITETU (SEED ENSEMBLING)
# ==============================================================================

def train_single_seed_model(
    seed: int,
    train_files: List[str],
    train_labels: np.ndarray,
    val_ds: tf.data.Dataset,
    num_classes: int,
    class_weights: Dict[int, float],
    epochs: int,
    batch_size: int,
    learning_rate: float,
    dropout_rate: float,
    models_dir: Path,
) -> Tuple[tf.keras.Model, Path, Dict[str, List[float]]]:
    """
    Trenuje pojedynczy model 2D CNN dla zadanego ziarna `seed` i zapisuje najlepszy
    checkpoint do `output_dir/models/ensemble_model_seed_{seed}.keras`.
    """
    print("\n" + "=" * 80)
    print(f"🚀 START TRENINGU MODELU | SEED = {seed} | EPOCHS = {epochs}")
    print("=" * 80)

    set_global_seed(seed)

    # Osobna instancja datasetu treningowego dla danego seeda gwarantuje
    # unikalną permutację batchy (shuffle) oraz augmentacji w locie
    train_ds = build_dataset_pipeline(
        file_paths=train_files,
        labels=train_labels,
        num_classes=num_classes,
        batch_size=batch_size,
        is_training=True,
        seed=seed,
    )

    model = instantiate_and_compile_model(
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        learning_rate=learning_rate,
    )

    model_save_path = models_dir / f"ensemble_model_seed_{seed}.keras"

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    # Wczytanie najlepszego zapisanego stanu modelu z dysku
    if model_save_path.exists():
        print(f"[INFO] Wczytywanie najlepszego checkpointu z: {model_save_path}")
        best_model = tf.keras.models.load_model(str(model_save_path))
    else:
        model.save(str(model_save_path))
        best_model = model

    history_dict = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    return best_model, model_save_path, history_dict


def compute_metrics_from_probabilities(
    y_true: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Wylicza metryki klasyfikacji (Accuracy, Weighted F1 oraz F1 per klasa)
    na podstawie macierzy prawdopodobieństw Softmax (N_samples, N_classes).
    """
    y_pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    per_class_f1_arr = f1_score(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        average=None,
        zero_division=0,
    )
    per_class_f1 = {
        class_names[idx]: float(per_class_f1_arr[idx])
        for idx in range(len(class_names))
    }

    return {
        "accuracy": acc,
        "weighted_f1": weighted_f1,
        "macro_f1": macro_f1,
        "per_class_f1": per_class_f1,
        "y_pred": y_pred,
    }


def fuse_ensemble_models(
    models: Sequence[tf.keras.Model],
    seeds: Sequence[int],
    output_path: Path,
) -> tf.keras.Model:
    """
    Łączy wytrenowane modele w jeden nadrzędny graf Keras:
    `tf.keras.layers.Average()([model_1(inputs), model_2(inputs), ...])`
    i zapisuje go jako `fused_ensemble_model.keras` gotowy pod eksport TFLite.
    """
    inputs = tf.keras.layers.Input(shape=EXPECTED_INPUT_SHAPE, dtype=tf.float32, name="audio_tensor_input")

    submodel_outputs = []
    for idx, (model, seed) in enumerate(zip(models, seeds)):
        # Nadanie unikalnej nazwy każdemu podmodelowi w grafie Keras 3 / TF 2.x
        try:
            model.name = f"hdd_cnn_seed_{seed}"
        except AttributeError:
            model._name = f"hdd_cnn_seed_{seed}"  # pylint: disable=protected-access
        model.trainable = False
        submodel_outputs.append(model(inputs, training=False))

    if len(submodel_outputs) == 1:
        ensemble_output = submodel_outputs[0]
    else:
        ensemble_output = tf.keras.layers.Average(name="ensemble_soft_voting")(submodel_outputs)

    fused_model = tf.keras.Model(
        inputs=inputs,
        outputs=ensemble_output,
        name="fused_hdd_acoustic_ensemble",
    )
    fused_model.save(str(output_path))
    print(f"[INFO] Zapisano połączony model Keras (Fused Ensemble) do: {output_path}")
    return fused_model


# ==============================================================================
# 4. WIZUALIZACJA I RAPORTOWANIE (TABELA ASCII, PNG, JSON)
# ==============================================================================

def print_ascii_comparison_table(
    seeds: List[int],
    individual_metrics: Dict[int, Dict[str, Any]],
    ensemble_metrics: Dict[str, Any],
    class_names: List[str],
) -> Dict[str, float]:
    """
    Drukuje w konsoli czytelną tabelę ASCII zestawiającą:
    Modele indywidualne vs Średnia indywidualna vs 🏆 Komitet (Ensemble)
    wraz z podaniem zysku Δ F1 oraz Δ Accuracy.
    """
    target_order = ["clicking", "scratching", "beeping", "normal"]
    ordered_classes = [c for c in target_order if c in class_names] + [
        c for c in class_names if c not in target_order
    ]

    mean_acc = float(np.mean([individual_metrics[s]["accuracy"] for s in seeds]))
    mean_wf1 = float(np.mean([individual_metrics[s]["weighted_f1"] for s in seeds]))
    best_acc = float(np.max([individual_metrics[s]["accuracy"] for s in seeds]))
    best_wf1 = float(np.max([individual_metrics[s]["weighted_f1"] for s in seeds]))

    mean_class_f1 = {
        c: float(np.mean([individual_metrics[s]["per_class_f1"].get(c, 0.0) for s in seeds]))
        for c in ordered_classes
    }

    delta_acc_vs_mean = ensemble_metrics["accuracy"] - mean_acc
    delta_wf1_vs_mean = ensemble_metrics["weighted_f1"] - mean_wf1
    delta_acc_vs_best = ensemble_metrics["accuracy"] - best_acc
    delta_wf1_vs_best = ensemble_metrics["weighted_f1"] - best_wf1

    class_headers = " | ".join([f"{c[:10]:>10}" for c in ordered_classes])
    header = f"| {'Model / Konfiguracja':<26} | {'Accuracy':>9} | {'Weight F1':>9} | {class_headers} |"
    separator = "+" + "-" * (len(header) - 2) + "+"

    print("\n" + separator)
    print(f"| {'PODSUMOWANIE EWALUACJI SEED ENSEMBLING (ZBIÓR TESTOWY)':^{len(header) - 4}} |")
    print(separator)
    print(header)
    print(separator)

    for seed in seeds:
        m = individual_metrics[seed]
        cls_vals = " | ".join([f"{m['per_class_f1'].get(c, 0.0) * 100:>9.2f}%" for c in ordered_classes])
        print(
            f"| {f'Model (seed={seed})':<26} | {m['accuracy'] * 100:>8.2f}% | "
            f"{m['weighted_f1'] * 100:>8.2f}% | {cls_vals} |"
        )

    print(separator)
    mean_cls_vals = " | ".join([f"{mean_class_f1[c] * 100:>9.2f}%" for c in ordered_classes])
    print(
        f"| {'Średnia indywidualna':<26} | {mean_acc * 100:>8.2f}% | "
        f"{mean_wf1 * 100:>8.2f}% | {mean_cls_vals} |"
    )

    ens_cls_vals = " | ".join(
        [f"{ensemble_metrics['per_class_f1'].get(c, 0.0) * 100:>9.2f}%" for c in ordered_classes]
    )
    print(
        f"| {'🏆 KOMITET (Ensemble)':<25} | {ensemble_metrics['accuracy'] * 100:>8.2f}% | "
        f"{ensemble_metrics['weighted_f1'] * 100:>8.2f}% | {ens_cls_vals} |"
    )
    print(separator)

    delta_cls_vals = " | ".join(
        [
            f"{(ensemble_metrics['per_class_f1'].get(c, 0.0) - mean_class_f1[c]) * 100:>+9.2f}%"
            for c in ordered_classes
        ]
    )
    print(
        f"| {'Zysk Δ vs Średnia':<26} | {delta_acc_vs_mean * 100:>+8.2f}% | "
        f"{delta_wf1_vs_mean * 100:>+8.2f}% | {delta_cls_vals} |"
    )
    print(
        f"| {'Zysk Δ vs Najlepszy Seed':<26} | {delta_acc_vs_best * 100:>+8.2f}% | "
        f"{delta_wf1_vs_best * 100:>+8.2f}% | {'-':^{len(delta_cls_vals)}} |"
    )
    print(separator + "\n")

    return {
        "mean_individual_accuracy": mean_acc,
        "mean_individual_weighted_f1": mean_wf1,
        "best_individual_accuracy": best_acc,
        "best_individual_weighted_f1": best_wf1,
        "delta_accuracy_vs_mean": delta_acc_vs_mean,
        "delta_weighted_f1_vs_mean": delta_wf1_vs_mean,
        "delta_accuracy_vs_best": delta_acc_vs_best,
        "delta_weighted_f1_vs_best": delta_wf1_vs_best,
    }


def save_ensemble_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Zapisuje macierz pomyłek komitetu do `ensemble_confusion_matrix.png`,
    delegując do `plot_and_save_confusion_matrix` z `train_cnn.py`.
    """
    try:
        sig = inspect.signature(plot_and_save_confusion_matrix)
        params = list(sig.parameters.keys())
        cm = confusion_matrix(y_true, y_pred)

        if len(params) >= 3 and "y_true" in params[0].lower():
            plot_and_save_confusion_matrix(y_true, y_pred, class_names, str(output_path))
        elif len(params) == 3:
            plot_and_save_confusion_matrix(cm, class_names, str(output_path))
        else:
            plot_and_save_confusion_matrix(y_true, y_pred, class_names, output_path)

        if output_path.exists():
            print(f"[INFO] Zapisano macierz pomyłek komitetu do: {output_path}")
            return
    except Exception as exc:  # pylint: disable=broad-except
        print(f"[WARN] Fallback renderowania macierzy pomyłek ({exc}). Generowanie bezpośrednie...")

    # Fallback gwarantujący utworzenie pliku PNG nawet przy niestandardowej sygnaturze
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Ensemble Confusion Matrix (Soft Voting)",
        ylabel="Klasa rzeczywista (True Label)",
        xlabel="Klasa przewidziana (Predicted Label)",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0 if cm.size > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold",
            )
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] Zapisano macierz pomyłek komitetu do: {output_path}")


def plot_ensemble_comparison_bars(
    seeds: List[int],
    individual_metrics: Dict[int, Dict[str, Any]],
    ensemble_metrics: Dict[str, Any],
    class_names: List[str],
    output_path: Path,
) -> None:
    """
    Generuje wykres słupkowy porównujący F1-Score per klasa oraz Weighted F1
    dla poszczególnych seedów, średniej indywidualnej i komitetu (`ensemble_comparison_bars.png`).
    """
    categories = list(class_names) + ["Weighted F1"]
    x = np.arange(len(categories))

    num_series = len(seeds) + 2  # N seedów + Średnia + Ensemble
    total_width = 0.82
    bar_width = total_width / num_series

    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=150)
    palette = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974", "#64B5CD"]

    # Słupki dla pojedynczych modeli
    for idx, seed in enumerate(seeds):
        m = individual_metrics[seed]
        values = [m["per_class_f1"].get(c, 0.0) for c in class_names] + [m["weighted_f1"]]
        offset = (idx - num_series / 2.0 + 0.5) * bar_width
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=f"Seed {seed}",
            color=palette[idx % len(palette)],
            alpha=0.82,
            edgecolor="black",
            linewidth=0.5,
        )

    # Słupek średniej indywidualnej
    mean_values = [
        float(np.mean([individual_metrics[s]["per_class_f1"].get(c, 0.0) for s in seeds]))
        for c in class_names
    ] + [float(np.mean([individual_metrics[s]["weighted_f1"] for s in seeds]))]

    mean_offset = (len(seeds) - num_series / 2.0 + 0.5) * bar_width
    ax.bar(
        x + mean_offset,
        mean_values,
        width=bar_width,
        label="Średnia indywidualna",
        color="#7F7F7F",
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
        hatch="//",
    )

    # Słupek komitetu (Ensemble)
    ens_values = [
        ensemble_metrics["per_class_f1"].get(c, 0.0) for c in class_names
    ] + [ensemble_metrics["weighted_f1"]]

    ens_offset = (len(seeds) + 1 - num_series / 2.0 + 0.5) * bar_width
    ens_bars = ax.bar(
        x + ens_offset,
        ens_values,
        width=bar_width,
        label="🏆 Komitet (Ensemble)",
        color="#E5A93B",
        alpha=1.0,
        edgecolor="black",
        linewidth=1.2,
    )

    for rect in ens_bars:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2.0, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
        )

    ax.set_ylabel("F1-Score", fontsize=11, fontweight="bold")
    ax.set_title(
        "Porównanie F1-Score per klasa: Modele indywidualne vs Komitet (Seed Ensembling)",
        fontsize=13,
        fontweight="bold",
        pad=14,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10, fontweight="bold")
    ax.set_ylim(0.0, 1.08)
    ax.grid(axis="y", linestyle="--", alpha=0.45)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)

    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    print(f"[INFO] Zapisano wykres porównawczy F1 do: {output_path}")


# ==============================================================================
# 5. PARSER CLI I GŁÓWNY PRZEPŁYW WYKONANIA
# ==============================================================================

def parse_cli_arguments(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Buduje parser argumentów wiersza poleceń dla `train_ensemble.py`."""
    parser = argparse.ArgumentParser(
        description="HDD Acoustic Sentinel — Trening komitetu modeli 2D CNN (Seed Ensembling)."
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 777],
        help="Lista ziaren losowości dla modeli w komitecie (domyślnie: 42 123 777).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Liczba epok treningowych dla każdego modelu (domyślnie: 30).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="dataset_encoded",
        help="Ścieżka do katalogu dataset_encoded z podziałem train/validate/test.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ensemble_output",
        help="Katalog wyjściowy na wytrenowane modele i artefakty (domyślnie: ./ensemble_output).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rozmiar batcha treningowego i ewaluacyjnego (domyślnie: 32).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Początkowy współczynnik uczenia (learning rate, domyślnie: 0.001).",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Współczynnik regularyzacji Dropout (domyślnie: 0.3).",
    )
    parser.add_argument(
        "--no-fuse",
        action="store_true",
        help="Pomija budowę i zapis połączonego modelu fused_ensemble_model.keras.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Główna procedura wykonawcza modułu `train_ensemble.py`."""
    start_time = time.time()
    args = parse_cli_arguments(argv)

    verify_adr001_invariants()
    configure_gpu_memory_growth()

    output_dir = Path(args.output_dir).resolve()
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(resolve_dataset_dir(args.data_dir)).resolve()
    print(f"[INFO] Katalog danych (ADR-001): {dataset_root}")
    print(f"[INFO] Katalog wyjściowy artefaktów: {output_dir}")
    print(f"[INFO] Konfiguracja komitetu: N={len(args.seeds)}, seeds={args.seeds}")

    # 1. Odkrywanie klas i plików bez wycieku danych (train / validate / test)
    class_names, train_files, train_labels = load_split_metadata(dataset_root, "train")
    _, val_files, val_labels = load_split_metadata(dataset_root, "validate", class_names)
    _, test_files, test_labels = load_split_metadata(dataset_root, "test", class_names)

    num_classes = len(class_names)
    print(f"[INFO] Wykryto klasy ({num_classes}): {class_names}")
    print(
        f"[INFO] Liczebność zbiorów -> Train: {len(train_files)} | "
        f"Validate: {len(val_files)} | Test: {len(test_files)}"
    )

    # 2. Obliczenie wag klas (sqrt-balanced: np.clip(np.sqrt(w), 0.5, 3.5))
    class_weights = compute_sqrt_class_weights(train_labels, num_classes)
    print(f"[INFO] Wagi klas (sqrt-balanced): {class_weights}")

    # 3. Przygotowanie deterministycznych zbiorów walidacyjnego i testowego
    val_ds = build_dataset_pipeline(
        file_paths=val_files,
        labels=val_labels,
        num_classes=num_classes,
        batch_size=args.batch_size,
        is_training=False,
        seed=args.seeds[0],
    )
    test_ds = build_dataset_pipeline(
        file_paths=test_files,
        labels=test_labels,
        num_classes=num_classes,
        batch_size=args.batch_size,
        is_training=False,
        seed=args.seeds[0],
    )

    # 4. Pętla treningowa komitetu (N modeli) i indywidualna predykcja na zbiorze testowym
    trained_models: List[tf.keras.Model] = []
    saved_model_paths: Dict[int, str] = {}
    individual_probabilities: List[np.ndarray] = []
    individual_metrics: Dict[int, Dict[str, Any]] = {}

    for seed in args.seeds:
        model, model_path, _ = train_single_seed_model(
            seed=seed,
            train_files=train_files,
            train_labels=train_labels,
            val_ds=val_ds,
            num_classes=num_classes,
            class_weights=class_weights,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            dropout_rate=args.dropout,
            models_dir=models_dir,
        )
        trained_models.append(model)
        saved_model_paths[seed] = str(model_path)

        print(f"[INFO] Ewaluacja modelu (seed={seed}) na zbiorze testowym...")
        probs = model.predict(test_ds, verbose=0)
        individual_probabilities.append(probs)

        metrics = compute_metrics_from_probabilities(test_labels, probs, class_names)
        individual_metrics[seed] = metrics
        print(
            f"[RESULT] Seed {seed} -> Accuracy: {metrics['accuracy'] * 100:.2f}% | "
            f"Weighted F1: {metrics['weighted_f1'] * 100:.2f}%"
        )

    # 5. Inferencja komitetowa (Soft Voting / Probability Averaging)
    # P_ensemble(c) = (1 / N) * sum_{i=1}^N P_model_i(c)
    stacked_probs = np.stack(individual_probabilities, axis=0)
    ensemble_probs = np.mean(stacked_probs, axis=0)
    ensemble_metrics = compute_metrics_from_probabilities(
        test_labels, ensemble_probs, class_names
    )

    # 6. Drukowanie tabeli porównawczej ASCII oraz pełnego raportu klasyfikacji
    deltas = print_ascii_comparison_table(
        seeds=args.seeds,
        individual_metrics=individual_metrics,
        ensemble_metrics=ensemble_metrics,
        class_names=class_names,
    )

    print("[INFO] Szczegółowy raport klasyfikacji dla Komitetu (Soft Voting):")
    print(
        classification_report(
            test_labels,
            ensemble_metrics["y_pred"],
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    # 7. Generowanie artefaktów wizualnych (PNG)
    cm_path = output_dir / "ensemble_confusion_matrix.png"
    save_ensemble_confusion_matrix(
        y_true=test_labels,
        y_pred=ensemble_metrics["y_pred"],
        class_names=class_names,
        output_path=cm_path,
    )

    bars_path = output_dir / "ensemble_comparison_bars.png"
    plot_ensemble_comparison_bars(
        seeds=args.seeds,
        individual_metrics=individual_metrics,
        ensemble_metrics=ensemble_metrics,
        class_names=class_names,
        output_path=bars_path,
    )

    # 8. Opcjonalna fuzja modeli w pojedynczy graf Keras pod TFLite
    fused_model_path: Optional[Path] = None
    if not args.no_fuse:
        fused_model_path = output_dir / "fused_ensemble_model.keras"
        fuse_ensemble_models(
            models=trained_models,
            seeds=args.seeds,
            output_path=fused_model_path,
        )

    # 9. Zapis raportu numerycznego JSON (ensemble_summary.json)
    elapsed_sec = time.time() - start_time
    summary_payload: Dict[str, Any] = {
        "project": "HDD Acoustic Fault Detection (Acoustic HDD Sentinel)",
        "invariant_input_shape": list(EXPECTED_INPUT_SHAPE),
        "hyperparameters": {
            "seeds": args.seeds,
            "num_models": len(args.seeds),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "dropout_rate": args.dropout,
            "class_weights": class_weights,
        },
        "classes": class_names,
        "dataset_counts": {
            "train": len(train_files),
            "validate": len(val_files),
            "test": len(test_files),
        },
        "individual_models": {
            str(seed): {
                "model_path": saved_model_paths[seed],
                "accuracy": individual_metrics[seed]["accuracy"],
                "weighted_f1": individual_metrics[seed]["weighted_f1"],
                "macro_f1": individual_metrics[seed]["macro_f1"],
                "per_class_f1": individual_metrics[seed]["per_class_f1"],
            }
            for seed in args.seeds
        },
        "ensemble_soft_voting": {
            "accuracy": ensemble_metrics["accuracy"],
            "weighted_f1": ensemble_metrics["weighted_f1"],
            "macro_f1": ensemble_metrics["macro_f1"],
            "per_class_f1": ensemble_metrics["per_class_f1"],
            "confusion_matrix": confusion_matrix(
                test_labels, ensemble_metrics["y_pred"]
            ).tolist(),
        },
        "comparison_deltas": deltas,
        "artifacts": {
            "confusion_matrix_png": str(cm_path),
            "comparison_bars_png": str(bars_path),
            "fused_keras_model": str(fused_model_path) if fused_model_path else None,
        },
        "execution_time_seconds": round(elapsed_sec, 2),
    }

    summary_json_path = output_dir / "ensemble_summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Zapisano raport JSON do: {summary_json_path}")
    print(f"[DONE] Proces Seed Ensembling zakończony pomyślnie w {elapsed_sec:.1f}s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())