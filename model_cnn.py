"""
HDD Acoustic Fault Detection - 2D CNN Model Architecture
Single Source of Truth: ADR-001 & AGENTS.md

Moduł definiujący architekturę sieci 2D CNN w TensorFlow / Keras przystosowaną
do przetwarzania 3-kanałowych reprezentacji czasowo-częstotliwościowych [Log-Mel, Delta, Delta-Delta]
wycinków audio dysków HDD o stałym kształcie (128, 259, 3).
"""

from typing import Tuple, Optional
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers


# Stałe DSP / wejściowe zgodne z ADR-001
INPUT_SHAPE: Tuple[int, int, int] = (128, 259, 3)  # (n_mels, time_frames, channels)
DEFAULT_NUM_CLASSES: int = 4                       # np. clicking, beeping, scratching, normal


def build_hdd_cnn_model(
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    dropout_rate: float = 0.3,
    l2_reg: float = 1e-4,
    model_name: str = "AcousticHddSentinelCNN"
) -> tf.keras.Model:
    """
    Tworzy model 2D CNN dedykowany do akustycznej detekcji i klasyfikacji uszkodzeń HDD.

    Architektura oparta na 4 blokach konwolucyjnych z Batch Normalization,
    warstwami poolingowymi oraz Global Average Pooling, co zapobiega przeuczeniu
    (overfitting) na małych i średnich zbiorach akustycznych, zachowując wysoką
    czułość na impulsowe transjenty (kanał Delta) oraz wąskopasmowe piski (Log-Mel).

    Parametry:
        input_shape: Kształt tensora wejściowego (128, 259, 3) - Channel-Last.
        num_classes: Liczba klas wyjściowych (>= 2 dla softmax, 1 dla binarnego sigmoid).
        dropout_rate: Bazowy współczynnik regularyzacji dropout.
        l2_reg: Współczynnik regularyzacji wag L2.
        model_name: Nazwa modelu Keras.

    Zwraca:
        tf.keras.Model: Nieskompilowany lub gotowy do kompilacji model Keras.
    """
    # Niezmiennik techniczny ADR-001 / .agents/rules/adr-001-audio-pipeline.md:
    # Zawsze jawnie definiuj layers.Input(shape=(128, 259, 3))
    inputs = layers.Input(shape=input_shape, name="audio_spectrogram_input")

    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    # =========================================================================
    # BLOK 1: Niskopoziomowa ekstrakcja cech widmowych i transjentów
    # Wejście: (128, 259, 3) -> Wyjście po poolingu: (64, 129, 32)
    # =========================================================================
    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv1_1")(inputs)
    x = layers.BatchNormalization(name="bn1_1")(x)
    x = layers.Activation("relu", name="relu1_1")(x)

    x = layers.Conv2D(32, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv1_2")(x)
    x = layers.BatchNormalization(name="bn1_2")(x)
    x = layers.Activation("relu", name="relu1_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool1")(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.5, name="s_dropout1")(x)

    # =========================================================================
    # BLOK 2: Średniopoziomowe wzorce czasowo-częstotliwościowe (np. klikanie głowicy)
    # Wejście: (64, 129, 32) -> Wyjście po poolingu: (32, 64, 64)
    # =========================================================================
    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv2_1")(x)
    x = layers.BatchNormalization(name="bn2_1")(x)
    x = layers.Activation("relu", name="relu2_1")(x)

    x = layers.Conv2D(64, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv2_2")(x)
    x = layers.BatchNormalization(name="bn2_2")(x)
    x = layers.Activation("relu", name="relu2_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool2")(x)
    x = layers.SpatialDropout2D(dropout_rate * 0.75, name="s_dropout2")(x)

    # =========================================================================
    # BLOK 3: Wysokopoziomowa dynamika zmian (zatarte łożyska, rezonanse, zarysowania)
    # Wejście: (32, 64, 64) -> Wyjście po poolingu: (16, 32, 128)
    # =========================================================================
    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv3_1")(x)
    x = layers.BatchNormalization(name="bn3_1")(x)
    x = layers.Activation("relu", name="relu3_1")(x)

    x = layers.Conv2D(128, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv3_2")(x)
    x = layers.BatchNormalization(name="bn3_2")(x)
    x = layers.Activation("relu", name="relu3_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool3")(x)
    x = layers.SpatialDropout2D(dropout_rate, name="s_dropout3")(x)

    # =========================================================================
    # BLOK 4: Semantyczna reprezentacja akustyczna
    # Wejście: (16, 32, 128) -> Wyjście po poolingu: (8, 16, 256)
    # =========================================================================
    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv4_1")(x)
    x = layers.BatchNormalization(name="bn4_1")(x)
    x = layers.Activation("relu", name="relu4_1")(x)

    x = layers.Conv2D(256, kernel_size=(3, 3), padding="same", kernel_regularizer=reg, name="conv4_2")(x)
    x = layers.BatchNormalization(name="bn4_2")(x)
    x = layers.Activation("relu", name="relu4_2")(x)

    x = layers.MaxPooling2D(pool_size=(2, 2), name="pool4")(x)
    x = layers.Dropout(dropout_rate, name="dropout4")(x)

    # =========================================================================
    # AGREGACJA PRZESTRZENNO-CZASOWA (Global Average Pooling)
    # Zastępuje gigantyczną warstwę Flatten, redukuje liczbę parametrów z milionów
    # do bezpiecznego zakresu, eliminując ryzyko przeuczenia.
    # Wyjście: wektor 256-elementowy
    # =========================================================================
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)

    # Gęsta warstwa projekcyjna (Dense Projection Layer)
    x = layers.Dense(128, kernel_regularizer=reg, name="dense_features")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Activation("relu", name="relu_dense")(x)
    x = layers.Dropout(dropout_rate * 1.33, name="dense_dropout")(x)

    # =========================================================================
    # WARSTWA WYJŚCIOWA (Klasyfikator)
    # =========================================================================
    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name=model_name)
    return model


def build_transfer_learning_model(
    backbone_name: str = "mobilenetv2",
    input_shape: Tuple[int, int, int] = INPUT_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
    freeze_backbone: bool = True,
    dropout_rate: float = 0.3,
    model_name: Optional[str] = None
) -> tf.keras.Model:
    """
    Buduje model transfer learningu wykorzystując 3-kanałowy format tensora
    (Log-Mel, Delta, Delta-Delta) jako wejście do sieci pretrenowanej na ImageNet
    (zgodnie z sekcją 2 i 4 ADR-001).

    Parametry:
        backbone_name: Wybór szkieletu ('mobilenetv2', 'efficientnetb0', 'resnet50').
        input_shape: (128, 259, 3).
        num_classes: Liczba klas wyjściowych.
        freeze_backbone: Czy zablokować wagi bazowe podczas wstępnego treningu.
        dropout_rate: Stopień dropoutu w głowicy klasyfikatora.
        model_name: Opcjonalna nazwa modelu.
    """
    inputs = layers.Input(shape=input_shape, name="audio_spectrogram_input")

    backbone_name_lower = backbone_name.lower()
    if backbone_name_lower == "mobilenetv2":
        base_model = tf.keras.applications.MobileNetV2(
            input_tensor=inputs,
            include_top=False,
            weights="imagenet"
        )
    elif backbone_name_lower == "efficientnetb0":
        base_model = tf.keras.applications.EfficientNetB0(
            input_tensor=inputs,
            include_top=False,
            weights="imagenet"
        )
    elif backbone_name_lower == "resnet50":
        base_model = tf.keras.applications.ResNet50(
            input_tensor=inputs,
            include_top=False,
            weights="imagenet"
        )
    else:
        raise ValueError(f"Nieobsługiwany szkielet: {backbone_name}. Wybierz: mobilenetv2, efficientnetb0, resnet50.")

    base_model.trainable = not freeze_backbone

    x = base_model.output
    x = layers.GlobalAveragePooling2D(name="transfer_gap")(x)
    x = layers.BatchNormalization(name="transfer_bn")(x)
    x = layers.Dropout(dropout_rate, name="transfer_dropout")(x)
    x = layers.Dense(128, activation="relu", name="transfer_dense")(x)
    x = layers.Dropout(dropout_rate, name="transfer_dense_dropout")(x)

    if num_classes == 1:
        outputs = layers.Dense(1, activation="sigmoid", name="predictions")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    name = model_name or f"HddSentinel_{backbone_name}"
    model = models.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def compile_cnn_model(
    model: tf.keras.Model,
    learning_rate: float = 1e-3,
    label_mode: str = "categorical"
) -> tf.keras.Model:
    """
    Kompiluje model Keras z optymalizatorem Adam, odpowiednią funkcją straty i metrykami.

    Parametry:
        model: Model zwrócony przez `build_hdd_cnn_model` lub `build_transfer_learning_model`.
        learning_rate: Początkowy współczynnik uczenia (Learning Rate).
        label_mode: 'categorical' (dla wektorów one-hot), 'sparse' (dla indeksów int)
                    lub 'binary' (dla klasyfikacji 1-wyjściowej).
    """
    optimizer = optimizers.Adam(learning_rate=learning_rate)

    if label_mode == "sparse":
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy")
        ]
    elif label_mode == "binary":
        loss = tf.keras.losses.BinaryCrossentropy()
        metrics = [
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc")
        ]
    else:  # categorical (one-hot)
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        metrics = [
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall")
        ]

    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    return model


if __name__ == "__main__":
    print("=" * 70)
    print("AKUSTYCZNA DETEKCJA AWARII HDD - ARCHITEKTURA 2D CNN")
    print("=" * 70)
    
    # Przykładowa budowa modelu dla 4 klas awarii
    model = build_hdd_cnn_model(input_shape=(128, 259, 3), num_classes=4)
    model = compile_cnn_model(model, learning_rate=1e-3, label_mode="categorical")
    
    model.summary()
    print("\n[OK] Model 2D CNN utworzony i skompilowany pomyślnie z tensorem wejściowym (128, 259, 3).")
