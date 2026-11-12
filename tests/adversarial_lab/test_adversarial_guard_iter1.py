"""
Test walidacyjny Iteracji 1: Weryfikacja obrony przed GPU Coil Whine i Metronomem.
Single Source of Truth: ADR-001 & AGENTS.md
"""

import os
import sys
import numpy as np
import soundfile as sf
import onnxruntime as ort
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard, CLASSES
from ml.adversarial_outlier_generator import generate_gpu_coil_whine, generate_synthetic_metronome

ONNX_MODEL_PATH = "exported_models/best_hdd_cnn_model.onnx"


def test_adversarial_guard_empirical():
    assert os.path.isfile(ONNX_MODEL_PATH), f"Brak modelu ONNX: {ONNX_MODEL_PATH}"
    sess = ort.InferenceSession(ONNX_MODEL_PATH)
    inp_name = sess.get_inputs()[0].name
    guard = DSPAdversarialGuard()

    print("=" * 75)
    print("🧪 ROZPOCZĘCIE TESTÓW WALIDACYJNYCH ITERACJI 1 (DSP ADVERSARIAL GUARD)")
    print("=" * 75)

    def evaluate_audio(name: str, audio: np.ndarray, expected_normal: bool):
        chunk = audio[:int(22050 * 3.0)]
        tensor = encode_audio_chunk(chunk)
        inp = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
        raw_probs = sess.run(None, {inp_name: inp})[0][0]

        sanitized_probs, was_sanitized, reason = guard.protect_and_sanitize_predictions(raw_probs, tensor)
        pred_raw = CLASSES[np.argmax(raw_probs)]
        pred_san = CLASSES[np.argmax(sanitized_probs)]

        print(f"\n[Test: {name}]")
        print(f"  Surowy ONNX:      {pred_raw:12s} | P({pred_raw}) = {np.max(raw_probs)*100:.1f}% (normal: {raw_probs[2]*100:.1f}%)")
        print(f"  Po DSP Guard:     {pred_san:12s} | P({pred_san}) = {np.max(sanitized_probs)*100:.1f}% (normal: {sanitized_probs[2]*100:.1f}%)")
        if was_sanitized:
            print(f"  🛡️ OBRONA:         {reason}")

        if expected_normal:
            assert pred_san == "normal", f"BŁĄD: {name} nie został zneutralizowany do 'normal'! Pred: {pred_san}"
            assert sanitized_probs[2] >= 0.70, f"BŁĄD: Pewność 'normal' zbyt niska: {sanitized_probs[2]:.3f}"
        return pred_raw, pred_san

    # 1. Real YouTube Coil Whine
    cw_path = "tests/adversarial_lab/samples/coil_whine_yt.wav"
    cw_sig, _ = sf.read(cw_path)
    if cw_sig.ndim > 1:
        cw_sig = np.mean(cw_sig, axis=-1)
    evaluate_audio("Real YouTube GPU Coil Whine", cw_sig, expected_normal=True)

    # 2. Real YouTube Metronome
    metro_path = "tests/adversarial_lab/samples/metronome_click_yt.wav"
    metro_sig, _ = sf.read(metro_path)
    if metro_sig.ndim > 1:
        metro_sig = np.mean(metro_sig, axis=-1)
    evaluate_audio("Real YouTube 120 BPM Metronome", metro_sig, expected_normal=True)

    # 3. Syntetyczny GPU Coil Whine (wygenerowany przez moduł)
    syn_cw = generate_gpu_coil_whine(duration=3.5)
    evaluate_audio("Synthetic GPU Coil Whine (Qwen Coder)", syn_cw, expected_normal=True)

    # 4. Syntetyczny Metronome (wygenerowany przez moduł)
    syn_metro = generate_synthetic_metronome(duration=3.5, bpm=120)
    evaluate_audio("Synthetic Metronome 120 BPM (Qwen Coder)", syn_metro, expected_normal=True)

    # 5. Sprawdzenie, czy autentyczny zbiór testowy HDD nie został uszkodzony!
    test_dir = Path("dataset_encoded/test")
    assert test_dir.is_dir(), "Brak katalogu testowego dataset_encoded/test"

    print("\n" + "-" * 75)
    print("🔍 TEST REGRESJI: Weryfikacja autentycznych uszkodzeń HDD...")
    print("-" * 75)

    fault_classes = ["beeping", "clicking", "normal", "scratching"]
    for cls in fault_classes:
        files = list((test_dir / cls).glob("*.npy"))[:10]  # sprawdź 10 pierwszych tensorów
        correct_raw = 0
        correct_san = 0
        for f in files:
            tensor = np.load(f)
            inp = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
            raw_probs = sess.run(None, {inp_name: inp})[0][0]
            san_probs, _, _ = guard.protect_and_sanitize_predictions(raw_probs, tensor)
            
            if CLASSES[np.argmax(raw_probs)] == cls:
                correct_raw += 1
            if CLASSES[np.argmax(san_probs)] == cls:
                correct_san += 1
        print(f"  Klasa '{cls:12s}': Dokładność przed Guardem: {correct_raw}/{len(files)}, po Guardzie: {correct_san}/{len(files)}")
        assert correct_san >= correct_raw, f"Regresja na autentycznej klasie HDD {cls}!"

    print("\n" + "=" * 75)
    print("✅ WSZYSTKIE TESTY WALIDACYJNE ITERACJI 1 ZAKOŃCZONE PEŁNYM SUKCESEM!")
    print("=" * 75)


if __name__ == "__main__":
    test_adversarial_guard_empirical()
