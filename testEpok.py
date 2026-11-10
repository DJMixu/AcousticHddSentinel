"""
HDD Acoustic Sentinel — Porównanie modeli CNN w zależności od liczby epok treningowych
Single Source of Truth: ADR-001 & AGENTS.md

Skrypt automatycznie:
1. Trenuje oddzielne modele 2D CNN dla każdej zadanej liczby epok
   (domyślnie: 20, 25, 30, 35, 40, 45, 50, 60, 70).
2. Po każdym treningu wykonuje krótką walidację na zbiorze testowym
   (Classification Report, Confusion Matrix, metryki per klasa).
3. Przeprowadza test przeuczenia (Overfitting Diagnostic):
   - Porównuje Train Accuracy vs Test Accuracy (Δ > 15% = alarm).
   - Porównuje Train Loss vs Test Loss (ratio > 2.0 = alarm).
4. Zapisuje zbiorczy raport porównawczy (JSON + wykres + tabela CSV).

Uruchomienie:
    # CPU (domyślnie):
    python testEpok.py

    # GPU ROCm:
    ROCM_PATH=/opt/rocm HSA_OVERRIDE_GFX_VERSION=12.0.0 \
    ./.venv_rocm/bin/python testEpok.py

    # Niestandardowe epoki i parametry:
    python testEpok.py --epochs 10 20 30 40 50 --batch-size 64 --lr 0.0005
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

# Dodanie katalogu skryptu do sys.path
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

import tensorflow as tf

# Import modułów projektu (współdzielone z train_cnn.py)
from model_cnn import build_hdd_cnn_model, compile_cnn_model, INPUT_SHAPE
from train_cnn import (
    resolve_dataset_dir,
    discover_classes_and_files,
    create_tf_dataset,
    calculate_balanced_class_weights,
    apply_spec_augment,
)


# =============================================================================
# 1. TRENING POJEDYNCZEGO MODELU DLA ZADANEJ LICZBY EPOK
# =============================================================================

def train_single_model(
    num_epochs: int,
    train_ds: tf.data.Dataset,
    val_ds: Optional[tf.data.Dataset],
    num_classes: int,
    class_weights: Dict[int, float],
    output_dir: Path,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.3,
    seed: int = 42,
) -> Tuple[tf.keras.Model, tf.keras.callbacks.History]:
    """
    Buduje nowy model od zera i trenuje go na zadaną liczbę epok.
    EarlyStopping jest WYŁĄCZONY celowo, aby każdy model trenował
    się dokładnie przez podaną liczbę epok (test wpływu epok).
    """
    # Ustawienie seeda dla powtarzalności
    tf.random.set_seed(seed)
    np.random.seed(seed)

    model = build_hdd_cnn_model(
        input_shape=INPUT_SHAPE,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        model_name=f"HddCNN_epochs{num_epochs}"
    )
    model = compile_cnn_model(model, learning_rate=lr, label_mode="categorical")

    checkpoint_path = str(output_dir / f"best_model_ep{num_epochs}.keras")

    callbacks = [
        # ModelCheckpoint: Zapisz najlepszą epokę (wg val_loss)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0
        ),
        # ReduceLROnPlateau: Adaptacyjne obniżanie LR
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            mode="min",
            verbose=0
        ),
        # CSVLogger: Historia epok
        tf.keras.callbacks.CSVLogger(
            filename=str(output_dir / f"history_ep{num_epochs}.csv"),
            separator=",",
            append=False
        ),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=0  # Cisza — wyniki podsumowujemy po treningu
    )

    # Załaduj najlepsze wagi (z najniższą val_loss)
    if os.path.exists(checkpoint_path):
        model = tf.keras.models.load_model(checkpoint_path)

    return model, history


# =============================================================================
# 2. EWALUACJA MODELU NA ZBIORZE TESTOWYM
# =============================================================================

def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    train_ds: tf.data.Dataset,
    class_names: List[str],
    history: tf.keras.callbacks.History,
) -> Dict:
    """
    Wykonuje ewaluację modelu na zbiorze testowym i zwraca słownik metryk
    wraz z diagnostyką przeuczenia (overfitting).
    """
    # Metryki na zbiorze testowym
    test_results = model.evaluate(test_ds, verbose=0)
    metric_names = model.metrics_names
    test_metrics = {name: float(val) for name, val in zip(metric_names, test_results)}

    # Metryki na zbiorze treningowym (do oceny przeuczenia)
    train_results = model.evaluate(train_ds, verbose=0)
    train_metrics = {name: float(val) for name, val in zip(metric_names, train_results)}

    # Predykcje dla raportu klasyfikacyjnego
    y_true_all: List[int] = []
    y_pred_all: List[int] = []

    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred_all.extend(np.argmax(preds, axis=-1).tolist())
        y_true_all.extend(np.argmax(y_batch.numpy(), axis=-1).tolist())

    report_dict = classification_report(
        y_true_all, y_pred_all,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )

    # Gwarancja spójności kluczy metryk (niezależnie od wersji Keras/TF)
    test_acc_computed = float(np.mean(np.array(y_true_all) == np.array(y_pred_all)))
    test_metrics["accuracy"] = test_acc_computed
    if "loss" not in test_metrics and len(test_results) > 0:
        test_metrics["loss"] = float(test_results[0])

    if "accuracy" not in train_metrics:
        for k in ["categorical_accuracy", "compile_metrics_accuracy", "acc"]:
            if k in train_metrics:
                train_metrics["accuracy"] = train_metrics[k]
                break
        if "accuracy" not in train_metrics and len(train_results) > 1:
            train_metrics["accuracy"] = float(train_results[1])
    if "loss" not in train_metrics and len(train_results) > 0:
        train_metrics["loss"] = float(train_results[0])

    # Diagnostyka przeuczenia (Overfitting)
    hist = history.history
    final_train_loss = hist["loss"][-1]
    final_val_loss = hist["val_loss"][-1] if "val_loss" in hist else final_train_loss
    best_val_loss = min(hist["val_loss"]) if "val_loss" in hist else final_train_loss
    best_val_epoch = int(np.argmin(hist["val_loss"]) + 1) if "val_loss" in hist else len(hist["loss"])

    acc_key = "accuracy" if "accuracy" in hist else "categorical_accuracy"
    val_acc_key = f"val_{acc_key}"
    final_train_acc = hist[acc_key][-1] if acc_key in hist else 0.0
    final_val_acc = hist[val_acc_key][-1] if val_acc_key in hist else 0.0

    train_test_acc_gap = final_train_acc - test_metrics.get("accuracy", 0.0)
    loss_ratio = final_train_loss / (test_metrics.get("loss", 1.0) + 1e-9)

    overfit_flags = []
    if train_test_acc_gap > 0.15:
        overfit_flags.append(f"ALARM: Train-Test Accuracy gap = {train_test_acc_gap:.2%} (>15%)")
    if loss_ratio < 0.5:
        overfit_flags.append(f"ALARM: Train/Test Loss ratio = {loss_ratio:.3f} (<0.5 — model zbytnio dopasowany)")
    if final_val_loss > best_val_loss * 1.5:
        overfit_flags.append(f"UWAGA: Końcowa val_loss ({final_val_loss:.4f}) > 1.5× najlepsza ({best_val_loss:.4f})")

    return {
        "test_metrics": test_metrics,
        "train_metrics": train_metrics,
        "per_class": {
            name: {
                "precision": report_dict[name]["precision"],
                "recall": report_dict[name]["recall"],
                "f1-score": report_dict[name]["f1-score"],
                "support": report_dict[name]["support"],
            }
            for name in class_names
        },
        "weighted_avg": {
            "precision": report_dict["weighted avg"]["precision"],
            "recall": report_dict["weighted avg"]["recall"],
            "f1-score": report_dict["weighted avg"]["f1-score"],
        },
        "overfitting_diagnostic": {
            "train_accuracy": final_train_acc,
            "test_accuracy": test_metrics.get("accuracy", 0.0),
            "train_test_acc_gap": train_test_acc_gap,
            "train_loss": final_train_loss,
            "test_loss": test_metrics.get("loss", 0.0),
            "loss_ratio": loss_ratio,
            "best_val_loss": best_val_loss,
            "best_val_epoch": best_val_epoch,
            "final_val_loss": final_val_loss,
            "overfit_flags": overfit_flags,
            "is_overfitting": len(overfit_flags) > 0,
        },
    }


# =============================================================================
# 3. WYKRESY PORÓWNAWCZE
# =============================================================================

def plot_epoch_comparison(all_results: Dict[int, Dict], output_dir: Path):
    """
    Generuje zbiorczy wykres porównawczy wszystkich wytrenowanych modeli:
    - Panel 1: Test Accuracy i Weighted F1-Score vs. Epochs
    - Panel 2: Train vs Test Accuracy (luka = przeuczenie)
    - Panel 3: F1-Score per klasa vs. Epochs
    - Panel 4: Train Loss vs Test Loss vs. Epochs
    """
    epochs_list = sorted(all_results.keys())
    test_accs = [all_results[e]["test_metrics"]["accuracy"] for e in epochs_list]
    weighted_f1s = [all_results[e]["weighted_avg"]["f1-score"] for e in epochs_list]
    train_accs = [all_results[e]["overfitting_diagnostic"]["train_accuracy"] for e in epochs_list]
    train_losses = [all_results[e]["overfitting_diagnostic"]["train_loss"] for e in epochs_list]
    test_losses = [all_results[e]["test_metrics"]["loss"] for e in epochs_list]

    # Klasy
    sample_result = all_results[epochs_list[0]]
    class_names = list(sample_result["per_class"].keys())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)

    # ---- Panel 1: Test Accuracy & Weighted F1 ----
    ax1 = axes[0, 0]
    ax1.plot(epochs_list, test_accs, "o-", color="royalblue", linewidth=2, markersize=7, label="Test Accuracy")
    ax1.plot(epochs_list, weighted_f1s, "s--", color="darkorange", linewidth=2, markersize=7, label="Weighted F1-Score")
    best_f1_idx = int(np.argmax(weighted_f1s))
    ax1.axvline(epochs_list[best_f1_idx], color="green", linestyle=":", alpha=0.7, label=f"Best F1 @ {epochs_list[best_f1_idx]} epok")
    ax1.set_title("Dokładność i F1-Score na zbiorze testowym", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Liczba epok treningowych")
    ax1.set_ylabel("Wartość metryki")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs_list)

    # ---- Panel 2: Train vs Test Accuracy (Overfitting) ----
    ax2 = axes[0, 1]
    ax2.plot(epochs_list, train_accs, "^-", color="forestgreen", linewidth=2, markersize=7, label="Train Accuracy")
    ax2.plot(epochs_list, test_accs, "o-", color="crimson", linewidth=2, markersize=7, label="Test Accuracy")
    ax2.fill_between(epochs_list, test_accs, train_accs, alpha=0.15, color="red", label="Luka (Overfitting)")
    ax2.set_title("Diagnostyka przeuczenia: Train vs Test Accuracy", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Liczba epok treningowych")
    ax2.set_ylabel("Accuracy")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs_list)

    # ---- Panel 3: F1-Score per klasa ----
    ax3 = axes[1, 0]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
    for i, cls_name in enumerate(class_names):
        f1_per_epoch = [all_results[e]["per_class"][cls_name]["f1-score"] for e in epochs_list]
        ax3.plot(epochs_list, f1_per_epoch, "o-", color=colors[i % len(colors)],
                 linewidth=2, markersize=6, label=cls_name)
    ax3.set_title("F1-Score per klasa vs. liczba epok", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Liczba epok treningowych")
    ax3.set_ylabel("F1-Score")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(epochs_list)

    # ---- Panel 4: Train Loss vs Test Loss ----
    ax4 = axes[1, 1]
    ax4.plot(epochs_list, train_losses, "^-", color="teal", linewidth=2, markersize=7, label="Train Loss (final)")
    ax4.plot(epochs_list, test_losses, "o-", color="purple", linewidth=2, markersize=7, label="Test Loss")
    ax4.set_title("Funkcja straty: Train vs Test", fontsize=13, fontweight="bold")
    ax4.set_xlabel("Liczba epok treningowych")
    ax4.set_ylabel("Loss")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_xticks(epochs_list)

    plt.suptitle("Analiza wpływu liczby epok treningowych na jakość modelu HDD CNN",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()

    chart_path = output_dir / "porownanie_epok.png"
    plt.savefig(chart_path, bbox_inches="tight")
    plt.close()
    print(f"\n[Artefakt] Wykres porównawczy zapisano w: {chart_path}")


def plot_individual_confusion_matrix(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: List[str],
    num_epochs: int,
    output_dir: Path,
):
    """Generuje macierz pomyłek dla pojedynczego modelu."""
    y_true, y_pred = [], []
    for x_batch, y_batch in test_ds:
        preds = model.predict(x_batch, verbose=0)
        y_pred.extend(np.argmax(preds, axis=-1).tolist())
        y_true.extend(np.argmax(y_batch.numpy(), axis=-1).tolist())

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(7, 6), dpi=120)
    cax = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    ax.set_title(f"Macierz Pomyłek — {num_epochs} epok", fontsize=12, fontweight="bold")
    ticks = np.arange(len(class_names))
    ax.set_xticks(ticks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(ticks)
    ax.set_yticklabels(class_names, fontsize=10)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]*100:.1f}%)",
                    ha="center", va="center", color=color, fontsize=9)
    ax.set_ylabel("Rzeczywista klasa")
    ax.set_xlabel("Przewidziana klasa")
    plt.tight_layout()
    plt.savefig(output_dir / f"confusion_matrix_ep{num_epochs}.png")
    plt.close()


# =============================================================================
# 4. GŁÓWNA PROCEDURA PORÓWNAWCZA
# =============================================================================

def run_epoch_comparison(
    epoch_list: List[int],
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.3,
    seed: int = 42,
):
    """
    Pełny pipeline porównania modeli CNN trenowanych na różnych liczbach epok.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ANALIZA WPŁYWU LICZBY EPOK TRENINGOWYCH NA JAKOŚĆ MODELU HDD CNN")
    print("=" * 70)
    print(f"Epoki do przetestowania: {epoch_list}")
    print(f"Katalog danych:          {data_dir}")
    print(f"Katalog wyników:         {output_dir}")
    print(f"Batch size: {batch_size} | LR: {lr} | Dropout: {dropout_rate} | Seed: {seed}")

    # Sprawdzenie dostępności GPU i dynamiczna alokacja pamięci VRAM
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Urządzenie obliczeniowe: GPU ({gpus[0].name}) [memory_growth=True]")
        except Exception as e:
            print(f"Urządzenie obliczeniowe: GPU ({gpus[0].name}) (set_memory_growth: {e})")
    else:
        print("Urządzenie obliczeniowe: CPU")

    # Wczytanie danych
    train_dir = data_dir / "train"
    val_dir = data_dir / "validate"
    test_dir = data_dir / "test"

    train_classes, train_files, train_labels = discover_classes_and_files(train_dir)
    val_classes, val_files, val_labels = discover_classes_and_files(val_dir, expected_classes=train_classes)
    test_classes, test_files, test_labels = discover_classes_and_files(test_dir, expected_classes=train_classes)

    if not train_files:
        raise FileNotFoundError(f"Brak plików .npy w {train_dir}. Uruchom hdd_audio_encoder.py.")

    class_names = train_classes
    num_classes = len(class_names)

    print(f"\nWykryte klasy ({num_classes}): {class_names}")
    print(f"Próbki: TRAIN={len(train_files)} | VALIDATE={len(val_files)} | TEST={len(test_files)}")

    # Wagi klas
    class_weights = calculate_balanced_class_weights(train_labels, num_classes)
    print(f"Wagi klas (sqrt-balanced):")
    for idx, name in enumerate(class_names):
        print(f"  [{idx}] {name:15s}: {class_weights[idx]:.3f} (n={train_labels.count(idx)})")

    # Tworzenie pipeline'ów tf.data.Dataset (współdzielone między modelami)
    train_ds = create_tf_dataset(train_files, train_labels, num_classes,
                                  batch_size=batch_size, is_training=True, seed=seed)
    val_ds = create_tf_dataset(val_files, val_labels, num_classes,
                                batch_size=batch_size, is_training=False) if val_files else None
    test_ds = create_tf_dataset(test_files, test_labels, num_classes,
                                 batch_size=batch_size, is_training=False) if test_files else None

    # Dataset treningowy BEZ augmentacji (do oceny przeuczenia — czysta dokładność na danych treningowych)
    train_eval_ds = create_tf_dataset(train_files, train_labels, num_classes,
                                       batch_size=batch_size, is_training=False)

    # Pętla główna: trening i ewaluacja dla każdej liczby epok
    all_results: Dict[int, Dict] = {}
    total_start = time.time()

    for i, num_epochs in enumerate(epoch_list, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(epoch_list)}] TRENING MODELU: {num_epochs} EPOK")
        print(f"{'=' * 70}")

        ep_dir = output_dir / f"epochs_{num_epochs}"
        ep_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        model, history = train_single_model(
            num_epochs=num_epochs,
            train_ds=train_ds,
            val_ds=val_ds,
            num_classes=num_classes,
            class_weights=class_weights,
            output_dir=ep_dir,
            batch_size=batch_size,
            lr=lr,
            dropout_rate=dropout_rate,
            seed=seed,
        )
        train_time = time.time() - t0

        # Ewaluacja
        results = evaluate_model(model, test_ds, train_eval_ds, class_names, history)
        results["training_time_sec"] = train_time
        results["num_epochs"] = num_epochs

        all_results[num_epochs] = results

        # Macierz pomyłek
        plot_individual_confusion_matrix(model, test_ds, class_names, num_epochs, ep_dir)

        # Podsumowanie w konsoli
        diag = results["overfitting_diagnostic"]
        print(f"  Czas treningu:     {train_time:.1f}s ({train_time/60:.1f} min)")
        print(f"  Test Accuracy:     {results['test_metrics']['accuracy']:.4f}")
        print(f"  Weighted F1-Score: {results['weighted_avg']['f1-score']:.4f}")
        print(f"  Train Accuracy:    {diag['train_accuracy']:.4f}")
        print(f"  Luka Train-Test:   {diag['train_test_acc_gap']:.4f} "
              f"({'⚠️ OVERFITTING' if diag['is_overfitting'] else '✅ OK'})")
        print(f"  Najlepsza val_loss: ep.{diag['best_val_epoch']} ({diag['best_val_loss']:.4f})")

        for cls_name in class_names:
            cls_f1 = results["per_class"][cls_name]["f1-score"]
            print(f"    {cls_name:15s}: F1 = {cls_f1:.4f}")

        if diag["overfit_flags"]:
            for flag in diag["overfit_flags"]:
                print(f"  ⚠️  {flag}")

        sys.stdout.flush()

        # Zwolnienie pamięci GPU po każdym modelu
        del model
        tf.keras.backend.clear_session()

    total_time = time.time() - total_start

    # =========================================================================
    # 5. PODSUMOWANIE ZBIORCZE
    # =========================================================================
    print(f"\n{'=' * 70}")
    print("PODSUMOWANIE PORÓWNANIA MODELI (WSZYSTKIE EPOKI)")
    print(f"{'=' * 70}")
    print(f"Łączny czas obliczeń: {total_time:.1f}s ({total_time/60:.1f} min)")

    # Tabela zbiorcza
    header = f"{'Epoki':>6} | {'Test Acc':>9} | {'W. F1':>7} | {'Train Acc':>10} | {'Gap':>6} | {'Overfit':>10} | {'Czas (s)':>8}"
    print(f"\n{header}")
    print("-" * len(header))

    for ep in sorted(all_results.keys()):
        r = all_results[ep]
        d = r["overfitting_diagnostic"]
        status = "⚠️ TAK" if d["is_overfitting"] else "✅ NIE"
        print(
            f"{ep:>6} | "
            f"{r['test_metrics']['accuracy']:>9.4f} | "
            f"{r['weighted_avg']['f1-score']:>7.4f} | "
            f"{d['train_accuracy']:>10.4f} | "
            f"{d['train_test_acc_gap']:>6.4f} | "
            f"{status:>10} | "
            f"{r['training_time_sec']:>8.1f}"
        )

    # Najlepszy model
    best_ep = max(all_results.keys(), key=lambda e: all_results[e]["weighted_avg"]["f1-score"])
    print(f"\n🏆 Najlepszy model wg Weighted F1-Score: {best_ep} epok "
          f"(F1 = {all_results[best_ep]['weighted_avg']['f1-score']:.4f}, "
          f"Acc = {all_results[best_ep]['test_metrics']['accuracy']:.4f})")

    # Wykresy porównawcze
    plot_epoch_comparison(all_results, output_dir)

    # Zapis CSV ze zbiorczą tabelą
    csv_path = output_dir / "porownanie_epok.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("epochs,test_accuracy,weighted_f1,train_accuracy,train_test_gap,is_overfitting,"
                "best_val_epoch,best_val_loss,train_loss,test_loss,training_time_sec")
        for cls_name in class_names:
            f.write(f",{cls_name}_f1")
        f.write("\n")
        for ep in sorted(all_results.keys()):
            r = all_results[ep]
            d = r["overfitting_diagnostic"]
            f.write(
                f"{ep},{r['test_metrics']['accuracy']:.6f},{r['weighted_avg']['f1-score']:.6f},"
                f"{d['train_accuracy']:.6f},{d['train_test_acc_gap']:.6f},"
                f"{int(d['is_overfitting'])},"
                f"{d['best_val_epoch']},{d['best_val_loss']:.6f},"
                f"{d['train_loss']:.6f},{r['test_metrics']['loss']:.6f},"
                f"{r['training_time_sec']:.1f}"
            )
            for cls_name in class_names:
                f.write(f",{r['per_class'][cls_name]['f1-score']:.6f}")
            f.write("\n")
    print(f"[Artefakt] Tabela CSV: {csv_path}")

    # Zapis pełnego raportu JSON
    json_path = output_dir / "porownanie_epok.json"
    # Konwersja numpy typów na natywne Python
    def _clean(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "epoch_list": epoch_list,
            "best_epoch_count": best_ep,
            "best_weighted_f1": all_results[best_ep]["weighted_avg"]["f1-score"],
            "total_training_time_sec": total_time,
            "device": "GPU" if gpus else "CPU",
            "results": {str(k): _clean(v) for k, v in all_results.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"[Artefakt] Raport JSON: {json_path}")

    print(f"\n{'=' * 70}")
    print("ANALIZA ZAKOŃCZONA POMYŚLNIE!")
    print(f"Wszystkie wyniki w: {output_dir.resolve()}")
    print(f"{'=' * 70}")


# =============================================================================
# 6. PARSER ARGUMENTÓW CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Porównanie modeli HDD CNN trenowanych na różnej liczbie epok."
    )
    parser.add_argument(
        "--epochs", type=int, nargs="+",
        default=[20, 25, 30, 35, 40, 45, 50, 60, 70],
        help="Lista liczb epok do przetestowania (domyślnie: 20 25 30 35 40 45 50 60 70)."
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Ścieżka do katalogu dataset_encoded."
    )
    parser.add_argument(
        "--output-dir", type=str, default="./epoch_comparison_output",
        help="Katalog wyjściowy na wyniki porównania (domyślnie: ./epoch_comparison_output)."
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
        "--seed", type=int, default=42,
        help="Ziarno generatora losowego dla powtarzalności (domyślnie: 42)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_directory = resolve_dataset_dir(args.data_dir)

    run_epoch_comparison(
        epoch_list=sorted(args.epochs),
        data_dir=data_directory,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_rate=args.dropout,
        seed=args.seed,
    )
