"""
HDD Acoustic Fault Detection - CNN Training Pipeline
Single Source of Truth: ADR-001, AGENTS.md, .agents/rules/adr-001-audio-pipeline.md

Skrypt treningowy dla sieci 2D CNN (Keras / TensorFlow) z:
1. Loaderem tf.data.Dataset wczytującym pliki .npy z dataset_encoded/{train,validate,test}
   zgodnie z regułą tf.numpy_function i tensor.set_shape([128, 259, 3]).
2. Obsługą class_weight (zrównoważenie rzadkich awarii HDD).
3. Callbackami: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau.
4. Ewaluacją na zbiorze testowym i generowaniem macierzy pomyłek (Confusion Matrix).
5. Zachowaniem izolacji normalizacji (import z hdd_audio_encoder.py dla inferencji).
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Dodanie katalogu skryptu oraz katalogu nadrzędnego do sys.path,
# aby umożliwić importy zarówno przy uruchamianiu z root, jak i z CNN/
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))
parent_dir = current_dir.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

# Import architektury modelu
from model_cnn import build_hdd_cnn_model, compile_cnn_model, INPUT_SHAPE

# Invariant 4 z AGENTS.md: Izolacja normalizacji
# Podczas pisania skryptów ewaluacyjnych i inferencyjnych importuj funkcje bezpośrednio z hdd_audio_encoder.py
try:
    from hdd_audio_encoder import (
        encode_audio_chunk,
        normalize_waveform_peak,
        normalize_tensor_zscore,
        TARGET_SR
    )
except Exception:
    try:
        from CNN.hdd_audio_encoder import (
            encode_audio_chunk,
            normalize_waveform_peak,
            normalize_tensor_zscore,
            TARGET_SR
        )
    except Exception:
        encode_audio_chunk = None
        normalize_waveform_peak = None
        normalize_tensor_zscore = None
        TARGET_SR = 22050


# =============================================================================
# 1. WYSZUKIWANIE I WALIDACJA KATALOGU ZBIORU DANYCH
# =============================================================================

def resolve_dataset_dir(custom_path: Optional[str] = None) -> Path:
    """
    Automatycznie lokalizuje katalog `dataset_encoded` sprawdzając typowe lokalizacje.
    """
    if custom_path:
        p = Path(custom_path)
        if p.exists():
            return p.resolve()
        raise FileNotFoundError(f"Podany katalog danych nie istnieje: {custom_path}")

    candidates = [
        Path("dataset_encoded"),
        Path("CNN/dataset_encoded"),
        current_dir / "dataset_encoded",
        parent_dir / "dataset_encoded",
        parent_dir / "CNN" / "dataset_encoded",
    ]

    for c in candidates:
        if c.exists() and (c / "train").exists():
            return c.resolve()

    # Domyślny fallback
    return (current_dir / "dataset_encoded").resolve()


def discover_classes_and_files(split_dir: Path) -> Tuple[List[str], List[Path], List[int]]:
    """
    Skanuje katalog danego podziału (train/validate/test), wykrywa podfoldery klas,
    mapuje klasy alfabetycznie i zbiera listę plików .npy wraz z etykietami całkowitymi.
    """
    if not split_dir.exists():
        return [], [], []

    # Klasy to podfoldery w katalogu danego splitu
    subdirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
    
    if not subdirs:
        # Fallback jeśli wszystkie pliki leżą płasko (np. pojedyncza klasa)
        npy_files = sorted(list(split_dir.glob("*.npy")))
        labels = [0] * len(npy_files)
        return ["default"], npy_files, labels

    class_names = [d.name for d in subdirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    all_files: List[Path] = []
    all_labels: List[int] = []

    for class_name in class_names:
        class_folder = split_dir / class_name
        files = sorted(list(class_folder.glob("*.npy")))
        all_files.extend(files)
        all_labels.extend([class_to_idx[class_name]] * len(files))

    return class_names, all_files, all_labels


# =============================================================================
# 2. LOADER tf.data.Dataset DLA PLIKÓW .NPY (ZGODNIE Z ADR-001)
# =============================================================================

def load_npy_tensor(filepath_input) -> np.ndarray:
    """
    Czysta funkcja Python do wczytania pliku .npy (wywoływana wewnątrz tf.numpy_function).
    """
    if isinstance(filepath_input, bytes):
        path_str = filepath_input.decode("utf-8")
    elif hasattr(filepath_input, "numpy"):
        val = filepath_input.numpy()
        path_str = val.decode("utf-8") if isinstance(val, bytes) else str(val)
    else:
        path_str = str(filepath_input)

    tensor = np.load(path_str).astype(np.float32)
    return tensor


def create_tf_dataset(
    file_paths: List[Path],
    labels: List[int],
    num_classes: int,
    batch_size: int = 32,
    is_training: bool = False,
    shuffle_buffer: int = 1000,
    seed: int = 42
) -> tf.data.Dataset:
    """
    Tworzy zoptymalizowany pipeline tf.data.Dataset wczytujący tensory .npy.
    
    Niezmiennik techniczny (.agents/rules/adr-001-audio-pipeline.md):
    Wczytując pliki .npy, używaj `tf.numpy_function` i jawnie ustawiaj
    `tensor.set_shape([128, 259, 3])`, aby kompilator grafu TF znał wymiary przed treningiem.
    """
    str_paths = [str(p) for p in file_paths]
    
    paths_ds = tf.data.Dataset.from_tensor_slices(str_paths)
    labels_ds = tf.data.Dataset.from_tensor_slices(labels)
    dataset = tf.data.Dataset.zip((paths_ds, labels_ds))

    if is_training:
        dataset = dataset.shuffle(buffer_size=min(len(file_paths), shuffle_buffer), seed=seed, reshuffle_each_iteration=True)

    def _map_entry(path_elem, label_elem):
        # tf.numpy_function do bezpiecznego wczytania pliku numpy
        [tensor] = tf.numpy_function(
            func=load_npy_tensor,
            inp=[path_elem],
            Tout=[tf.float32]
        )
        # Niezmiennik: jawne ustawienie kształtu tensora
        tensor.set_shape(INPUT_SHAPE)  # [128, 259, 3]

        # Konwersja etykiety do wektora one-hot
        one_hot_label = tf.one_hot(label_elem, depth=num_classes)
        one_hot_label.set_shape([num_classes])

        return tensor, one_hot_label

    dataset = dataset.map(_map_entry, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


# =============================================================================
# 3. OBSŁUGA class_weight (ZRÓWNOWAŻENIE KLAS)
# =============================================================================

def calculate_balanced_class_weights(labels: List[int], num_classes: int) -> Dict[int, float]:
    """
    Oblicza wagi klas w formule 'balanced' ze scikit-learn, aby zniwelować
    asymetrię liczby próbek w poszczególnych rodzajach uszkodzeń dysków HDD.
    """
    unique_present = np.unique(labels)
    computed_weights = compute_class_weight(
        class_weight="balanced",
        classes=unique_present,
        y=labels
    )
    
    weights_dict: Dict[int, float] = {}
    for cls_idx in range(num_classes):
        if cls_idx in unique_present:
            pos = np.where(unique_present == cls_idx)[0][0]
            weights_dict[cls_idx] = float(computed_weights[pos])
        else:
            weights_dict[cls_idx] = 1.0

    return weights_dict


# =============================================================================
# 4. CALLBACKI TRENINGOWE
# =============================================================================

def setup_training_callbacks(
    checkpoint_filepath: str,
    log_dir: str,
    early_stopping_patience: int = 15,
    reduce_lr_patience: int = 5
) -> List[tf.keras.callbacks.Callback]:
    """
    Konfiguruje zestaw kluczowych callbacków Keras:
    - EarlyStopping: zatrzymuje trening przy braku poprawy val_loss i przywraca najlepsze wagi.
    - ModelCheckpoint: zapisuje najlepszy model na podstawie minimalnego val_loss.
    - ReduceLROnPlateau: dynamicznie redukuje LR o 50% przy stagnacji val_loss.
    - CSVLogger: zapisuje historię metryk do pliku CSV.
    """
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_filepath)), exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=early_stopping_patience,
            restore_best_weights=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_filepath,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-6,
            mode="min",
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(log_dir, "training_history.csv"),
            separator=",",
            append=False
        )
    ]
    return callbacks


# =============================================================================
# 5. MACIERZ POMYŁEK I WYKRESY EWALUACYJNE
# =============================================================================

def plot_and_save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    output_path: str,
    title: str = "Macierz Pomyłek (HDD Fault Classification)"
) -> np.ndarray:
    """
    Generuje i zapisuje profesjonalny wykres macierzy pomyłek z liczbami próbek
    oraz znormalizowanymi procentami per klasa rzeczywista.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / (cm.sum(axis=1)[:, np.newaxis] + 1e-9)

    fig, ax = plt.subplots(figsize=(8, 7), dpi=150)
    cax = ax.imshow(cm_normalized, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    fig.colorbar(cax, fraction=0.046, pad=0.04)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names, fontsize=10)

    # Wypisanie wartości i procentów w komórkach
    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_normalized[i, j] * 100
            text_color = "white" if cm_normalized[i, j] > thresh else "black"
            ax.text(
                j, i, f"{val}\n({pct:.1f}%)",
                ha="center", va="center",
                color=text_color,
                fontsize=9, fontweight="medium"
            )

    ax.set_ylabel("Rzeczywista klasa (True Label)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Przewidziana klasa (Predicted Label)", fontsize=11, fontweight="bold")
    plt.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()

    print(f"[Ewaluacja] Wykres macierzy pomyłek zapisano w: {output_path}")
    return cm


def plot_training_curves(history: tf.keras.callbacks.History, output_path: str):
    """
    Rysuje i zapisuje krzywe uczenia (funkcja straty oraz dokładność) dla Train i Validation.
    """
    hist = history.history
    epochs_range = range(1, len(hist["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    # Loss
    ax1.plot(epochs_range, hist["loss"], label="Train Loss", color="royalblue", linewidth=2)
    if "val_loss" in hist:
        ax1.plot(epochs_range, hist["val_loss"], label="Val Loss", color="crimson", linewidth=2, linestyle="--")
    ax1.set_title("Krzywa funkcji straty (Loss)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoka")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    acc_key = "accuracy" if "accuracy" in hist else "categorical_accuracy"
    val_acc_key = f"val_{acc_key}"
    if acc_key in hist:
        ax2.plot(epochs_range, hist[acc_key], label="Train Accuracy", color="forestgreen", linewidth=2)
    if val_acc_key in hist:
        ax2.plot(epochs_range, hist[val_acc_key], label="Val Accuracy", color="darkorange", linewidth=2, linestyle="--")
    ax2.set_title("Krzywa dokładności (Accuracy)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoka")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[Ewaluacja] Wykres krzywych uczenia zapisano w: {output_path}")


# =============================================================================
# 6. FUNKCJA INFERENCJI ZGODNA Z INVARIANTEM 4 (IMPORT NORMALIZACJI)
# =============================================================================

def predict_audio_chunk(
    model: tf.keras.Model,
    y_chunk: np.ndarray,
    class_names: List[str],
    sr: int = 22050
) -> Tuple[str, float, np.ndarray]:
    """
    Wykonuje inferencję bezpośrednio na surowym 3-sekundowym wycinku audio.
    Zgodnie z Invariantem 4 z AGENTS.md, używa wyizolowanych funkcji normalizacji
    i enkodera z hdd_audio_encoder.py bez duplikowania kodu.
    """
    if encode_audio_chunk is None:
        raise ImportError("Brak modułu hdd_audio_encoder.py. Upewnij się, że plik znajduje się w ścieżce.")

    tensor = encode_audio_chunk(
        y_chunk=y_chunk,
        sr=sr,
        waveform_norm_fn=normalize_waveform_peak,
        tensor_norm_fn=normalize_tensor_zscore,
        tensor_format="tf"
    )
    # Rozszerzenie wymiaru o batch: (1, 128, 259, 3)
    batch_tensor = np.expand_dims(tensor, axis=0)
    probs = model.predict(batch_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = class_names[pred_idx]
    confidence = float(probs[pred_idx])

    return pred_class, confidence, probs


# =============================================================================
# 7. GŁÓWNA PROCEDURA TRENINGOWA
# =============================================================================

def train_and_evaluate(
    data_dir: Path,
    output_dir: Path,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    dropout_rate: float = 0.3
):
    """
    Pełny potok: wczytanie danych, budowa modelu, trening z class_weight,
    ewaluacja na zbiorze testowym oraz wygenerowanie macierzy pomyłek.
    """
    print("=" * 70)
    print("ROZPOCZĘCIE TRENINGU SIECI 2D CNN (HDD ACOUSTIC SENTINEL)")
    print("=" * 70)
    print(f"Katalog zbioru danych: {data_dir}")
    print(f"Katalog wyjściowy:     {output_dir}")

    train_dir = data_dir / "train"
    val_dir = data_dir / "validate"
    test_dir = data_dir / "test"

    # Weryfikacja katalogów
    train_classes, train_files, train_labels = discover_classes_and_files(train_dir)
    val_classes, val_files, val_labels = discover_classes_and_files(val_dir)
    test_classes, test_files, test_labels = discover_classes_and_files(test_dir)

    if not train_files:
        raise FileNotFoundError(
            f"Nie znaleziono plików .npy w katalogu treningowym '{train_dir}'. "
            f"Uruchom najpierw enkoder: python hdd_audio_encoder.py"
        )

    # Klasy powinny być spójne (bazujemy na katalogu train)
    class_names = train_classes
    num_classes = len(class_names)

    print(f"\nWykryte klasy ({num_classes}): {class_names}")
    print(f"Liczba próbek .npy:")
    print(f"  - TRAIN:    {len(train_files)} tensorów")
    print(f"  - VALIDATE: {len(val_files)} tensorów")
    print(f"  - TEST:     {len(test_files)} tensorów")

    # Obliczenie class_weight
    class_weights = calculate_balanced_class_weights(train_labels, num_classes)
    print(f"\nObliczone wagi klas (class_weight):")
    for idx, c_name in enumerate(class_names):
        print(f"  [{idx}] {c_name:15s}: waga = {class_weights[idx]:.3f} (próbek: {train_labels.count(idx)})")

    # Tworzenie pipeline'ów tf.data.Dataset
    print("\nBudowanie loaderów tf.data.Dataset z wymiarem (128, 259, 3)...")
    train_ds = create_tf_dataset(train_files, train_labels, num_classes, batch_size=batch_size, is_training=True)
    val_ds = create_tf_dataset(val_files, val_labels, num_classes, batch_size=batch_size, is_training=False) if val_files else None
    test_ds = create_tf_dataset(test_files, test_labels, num_classes, batch_size=batch_size, is_training=False) if test_files else None

    # Budowa i kompilacja modelu 2D CNN
    print("\nInicjalizacja modelu 2D CNN...")
    model = build_hdd_cnn_model(
        input_shape=INPUT_SHAPE,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        model_name="AcousticHddSentinelCNN"
    )
    model = compile_cnn_model(model, learning_rate=lr, label_mode="categorical")
    model.summary()

    # Callbacki
    checkpoint_path = str(output_dir / "best_hdd_cnn_model.keras")
    callbacks = setup_training_callbacks(
        checkpoint_filepath=checkpoint_path,
        log_dir=str(output_dir)
    )

    # Trening
    print("\n" + "=" * 70)
    print("START TRENINGU (model.fit)")
    print("=" * 70)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )

    # Zapis wykresów uczenia
    training_curves_path = str(output_dir / "training_curves.png")
    plot_training_curves(history, training_curves_path)

    # Zapis mapowania klas do JSON
    class_mapping_path = output_dir / "classes.json"
    with open(class_mapping_path, "w", encoding="utf-8") as f:
        json.dump({"classes": class_names, "input_shape": list(INPUT_SHAPE)}, f, indent=2)

    # Ewaluacja na zbiorze testowym
    if test_ds is not None and len(test_files) > 0:
        print("\n" + "=" * 70)
        print("EWALUACJA NA ZBIORZE TESTOWYM (TEST SET)")
        print("=" * 70)

        # Wczytanie wag najlepszego modelu zapisanego przez ModelCheckpoint
        if os.path.exists(checkpoint_path):
            print(f"Wczytywanie wag najlepszego checkpointu: {checkpoint_path}")
            model = tf.keras.models.load_model(checkpoint_path)

        eval_results = model.evaluate(test_ds, verbose=1)
        metric_names = model.metrics_names
        print("\nWyniki metryk na zbiorze testowym:")
        eval_dict = {}
        for name, val in zip(metric_names, eval_results):
            eval_dict[name] = float(val)
            print(f"  - {name}: {val:.4f}")

        # Predykcje dla całego zbioru testowego do macierzy pomyłek
        print("\nGenerowanie predykcji dla macierzy pomyłek...")
        y_true_all: List[int] = []
        y_pred_all: List[int] = []

        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch, verbose=0)
            y_pred_batch = np.argmax(preds, axis=-1)
            y_true_batch = np.argmax(y_batch.numpy(), axis=-1)
            y_true_all.extend(y_true_batch)
            y_pred_all.extend(y_pred_batch)

        y_true_arr = np.array(y_true_all)
        y_pred_arr = np.array(y_pred_all)

        # Raport klasyfikacji
        report_str = classification_report(
            y_true_arr,
            y_pred_arr,
            target_names=class_names,
            digits=4,
            zero_division=0
        )
        print("\nRaport Klasyfikacji (Classification Report):")
        print(report_str)

        # Zapis raportu do pliku tekstowego
        report_path = output_dir / "classification_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_str)

        # Wygenerowanie i zapis macierzy pomyłek
        cm_path = str(output_dir / "confusion_matrix.png")
        plot_and_save_confusion_matrix(
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            class_names=class_names,
            output_path=cm_path
        )

        # Zapis podsumowania metryk do JSON
        summary_path = output_dir / "test_evaluation_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "test_metrics": eval_dict,
                "classes": class_names,
                "total_test_samples": len(test_files)
            }, f, indent=2)

    print("\n" + "=" * 70)
    print("TRENING I EWALUACJA ZAKOŃCZONE POMYŚLNIE!")
    print(f"Wszystkie artefakty zapisano w katalogu: {output_dir.resolve()}")
    print("=" * 70)


# =============================================================================
# 8. PARSER ARGUMENTÓW CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Trening modelu 2D CNN do akustycznej detekcji awarii dysków HDD."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Ścieżka do katalogu dataset_encoded zawierającego train/, validate/, test/."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./training_output",
        help="Katalog do zapisu modeli, wykresów i logów (domyślnie: ./training_output)."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maksymalna liczba epok treningowych (domyślnie: 50)."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Rozmiar batcha (domyślnie: 32)."
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Początkowy współczynnik uczenia Learning Rate (domyślnie: 0.001)."
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Współczynnik dropoutu w warstwach modelu (domyślnie: 0.3)."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data_directory = resolve_dataset_dir(args.data_dir)
    out_directory = Path(args.output_dir)
    out_directory.mkdir(parents=True, exist_ok=True)

    train_and_evaluate(
        data_dir=data_directory,
        output_dir=out_directory,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        dropout_rate=args.dropout
    )
