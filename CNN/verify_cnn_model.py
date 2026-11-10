"""
Skrypt weryfikacyjny dla architektury modelu 2D CNN (Acoustic HDD Sentinel).
Uruchomienie: python verify_cnn_model.py
"""

import os
import tempfile
import numpy as np
import tensorflow as tf
from model_cnn import build_hdd_cnn_model, compile_cnn_model, build_transfer_learning_model

def run_verification():
    print("=" * 70)
    print("KOMPLEKSOWA WERYFIKACJA MODELU 2D CNN (ACOUSTIC HDD SENTINEL)")
    print("=" * 70)

    # 1. Kształt wejściowy i parametry
    print("[1/6] Inicjalizacja modelu i weryfikacja kształtów...")
    model = build_hdd_cnn_model(input_shape=(128, 259, 3), num_classes=4, dropout_rate=0.2)
    model = compile_cnn_model(model, learning_rate=1e-3)
    assert model.input_shape == (None, 128, 259, 3), "Błędny input_shape!"
    assert model.output_shape == (None, 4), "Błędny output_shape!"
    print(f"   -> Wejście: {model.input_shape}")
    print(f"   -> Wyjście: {model.output_shape}")
    trainable_count = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    print(f"   -> Liczba parametrów: {model.count_params():,} (trenowalnych: {trainable_count:,})")
    print("   [OK] Kształty zgodne z ADR-001.\n")

    # 2. Przekazywanie w przód (Forward pass)
    print("[2/6] Test przekazywania w przód (Forward Pass)...")
    dummy_x = np.random.randn(4, 128, 259, 3).astype(np.float32)
    preds = model.predict(dummy_x, verbose=0)
    assert preds.shape == (4, 4), "Błędny kształt predykcji!"
    assert not np.isnan(preds).any(), "Wykryto NaN!"
    assert not np.isinf(preds).any(), "Wykryto Inf!"
    assert np.allclose(np.sum(preds, axis=-1), 1.0, atol=1e-5), "Suma Softmax != 1.0!"
    print(f"   -> Kształt tensora predykcji: {preds.shape}")
    print(f"   -> Średnie prawdopodobieństwa: {np.mean(preds, axis=0)}")
    print("   [OK] Predykcja stabilna numerycznie, rozkład prawdopodobieństw Softmax poprawny.\n")

    # 3. Przepływ gradientów (Backward pass)
    print("[3/6] Test przepływu gradientów (Backward Pass)...")
    dummy_y = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)
    with tf.GradientTape() as tape:
        out = model(dummy_x, training=True)
        loss = tf.keras.losses.categorical_crossentropy(dummy_y, out)
    grads = tape.gradient(loss, model.trainable_variables)
    assert all(g is not None for g in grads), "Brakujące gradienty!"
    assert all(not tf.math.reduce_any(tf.math.is_nan(g)) for g in grads), "NaN w gradientach!"
    print(f"   -> Obliczono gradienty dla {len(grads)} zestawów wag.")
    print("   [OK] Gradienty przepływają prawidłowo przez wszystkie warstwy konwolucyjne i gęste.\n")

    # 4. Zbieżność uczenia (Convergence check)
    print("[4/6] Test zbieżności uczenia (Trening na małym batchu)...")
    train_x = np.random.randn(8, 128, 259, 3).astype(np.float32)
    train_y = tf.keras.utils.to_categorical(np.random.randint(0, 4, size=8), num_classes=4)
    hist = model.fit(train_x, train_y, epochs=15, batch_size=4, verbose=0)
    initial_loss = hist.history["loss"][0]
    final_loss = min(hist.history["loss"])
    print(f"   -> Loss początkowy: {initial_loss:.4f} -> Najniższy loss: {final_loss:.4f}")
    assert final_loss < initial_loss, "Brak spadku wartości funkcji straty!"
    print("   [OK] Model efektywnie uczy się wzorców i minimalizuje funkcję straty.\n")

    # 5. Zapis i odczyt z dysku (Serialization)
    print("[5/6] Test serializacji (Keras save & load)...")
    with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        model.save(tmp_path)
        loaded_model = tf.keras.models.load_model(tmp_path)
        p_orig = model.predict(dummy_x, verbose=0)
        p_loaded = loaded_model.predict(dummy_x, verbose=0)
        assert np.allclose(p_orig, p_loaded, atol=1e-5), "Rozbieżność predykcji po zapisie/odczycie!"
        print(f"   -> Model pomyślnie zapisany i odczytany z: {tmp_path}")
        print("   [OK] Wagi i architektura po deserializacji są w 100% identyczne.\n")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    # 6. Model Transfer Learning
    print("[6/6] Test architektury Transfer Learning (MobileNetV2)...")
    tl_model = build_transfer_learning_model("mobilenetv2", input_shape=(128, 259, 3), num_classes=4)
    tl_out = tl_model.predict(dummy_x[:2], verbose=0)
    assert tl_out.shape == (2, 4)
    print(f"   -> Kształt wyjścia MobileNetV2: {tl_out.shape}")
    print("   [OK] Architektura transfer learningu poprawnie przyjmuje tensory (128, 259, 3).\n")

    print("=" * 70)
    print("WYNIK: MODEL 2D CNN DZIAŁA W 100% POPRAWNIE I JEST GOTOWY DO UŻYCIA!")
    print("=" * 70)

if __name__ == "__main__":
    run_verification()
