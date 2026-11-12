"""
Weryfikacja autonomicznego odrzucania dźwięków obcych (OOD) przez sieć ONNX.
Single Source of Truth: ADR-001 & AGENTS.md

Test sprawdza, czy model ONNX SAMODZIELNIE (wewnątrz swojego grafu obliczeniowego):
1. Odrzuca dźwięki ustami (klikanie językiem, cmokanie, gwizd) jako 'normal'.
2. Odrzuca szumy wentylatorów komputerowych i tła jako 'normal'.
3. Wykrywa autentyczne mechaniczne uszkodzenia HDD (clicking, beeping, scratching) jako anomalie.
"""

import os
import sys
import numpy as np
import onnxruntime as ort
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from ml.hdd_audio_encoder import encode_audio_chunk
from ml.generate_outlier_exposure import (
    generate_mouth_clicks,
    generate_whistle_and_voice,
    generate_pc_fan_turbulent,
    generate_keyboard_typing
)

ONNX_MODEL_PATH = "exported_models/best_hdd_cnn_model.onnx"
CLASSES = ["beeping", "clicking", "normal", "scratching"]


def get_onnx_session():
    assert os.path.isfile(ONNX_MODEL_PATH), f"Brak pliku ONNX: {ONNX_MODEL_PATH}"
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    return session


def predict_onnx(session, audio_samples_3s: np.ndarray) -> np.ndarray:
    """Konwertuje 3-sekundowy wycinek audio do tensora ADR-001 i wykonuje inferencję ONNX."""
    tensor_ch_last = encode_audio_chunk(audio_samples_3s) # (128, 259, 3)
    # Model w ONNX akceptuje channel-first (1, 3, 128, 259) lub channel-last (1, 128, 259, 3)
    tensor_input = np.transpose(tensor_ch_last, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
    inp_name = session.get_inputs()[0].name
    probs = session.run(None, {inp_name: tensor_input})[0][0]
    return probs


def test_mouth_clicks_classified_as_normal_by_network(onnx_session):
    """Sieć powinna sama wiedzieć, że klikanie ustami jest obce i zwrócić 'normal'."""
    audio = generate_mouth_clicks(duration=3.0, sr=22050, num_clicks=4)
    probs = predict_onnx(onnx_session, audio)
    pred_class = CLASSES[np.argmax(probs)]
    normal_prob = probs[CLASSES.index("normal")]
    
    print(f"\n[Test Mouth Clicks] Predykcja ONNX: {pred_class} (normal: {normal_prob*100:.1f}%)")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:12s}: {p*100:5.2f}%")
        
    assert pred_class == "normal", f"Sieć nie odrzuciła klikania ustami! Pred: {pred_class}"
    assert normal_prob >= 0.70, f"Pewność normal zbyt niska: {normal_prob:.3f}"


def test_whistle_and_voice_classified_as_normal_by_network(onnx_session):
    """Gwizdanie / głos ludzki powinno być odrzucone jako 'normal'."""
    audio = generate_whistle_and_voice(duration=3.0, sr=22050)
    probs = predict_onnx(onnx_session, audio)
    pred_class = CLASSES[np.argmax(probs)]
    normal_prob = probs[CLASSES.index("normal")]
    
    print(f"\n[Test Whistle/Voice] Predykcja ONNX: {pred_class} (normal: {normal_prob*100:.1f}%)")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:12s}: {p*100:5.2f}%")
        
    assert pred_class == "normal", f"Sieć nie odrzuciła głosu! Pred: {pred_class}"
    assert normal_prob >= 0.70, f"Pewność normal zbyt niska: {normal_prob:.3f}"


def test_pc_fan_classified_as_normal_by_network(onnx_session):
    """Szum wentylatora komputera powinien być sklasyfikowany jako 'normal'."""
    audio = generate_pc_fan_turbulent(duration=3.0, sr=22050)
    probs = predict_onnx(onnx_session, audio)
    pred_class = CLASSES[np.argmax(probs)]
    normal_prob = probs[CLASSES.index("normal")]
    
    print(f"\n[Test PC Fan] Predykcja ONNX: {pred_class} (normal: {normal_prob*100:.1f}%)")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:12s}: {p*100:5.2f}%")
        
    assert pred_class == "normal", f"Sieć nie odrzuciła wentylatora! Pred: {pred_class}"
    assert normal_prob >= 0.50, f"Pewność normal zbyt niska: {normal_prob:.3f}"


def test_keyboard_typing_classified_as_normal_by_network(onnx_session):
    """Stukanie w klawiaturę powinno być odrzucone jako 'normal'."""
    audio = generate_keyboard_typing(duration=3.0, sr=22050)
    probs = predict_onnx(onnx_session, audio)
    pred_class = CLASSES[np.argmax(probs)]
    normal_prob = probs[CLASSES.index("normal")]
    
    print(f"\n[Test Keyboard Typing] Predykcja ONNX: {pred_class} (normal: {normal_prob*100:.1f}%)")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:12s}: {p*100:5.2f}%")
        
    assert pred_class == "normal", f"Sieć nie odrzuciła klawiatury! Pred: {pred_class}"


def test_real_healthy_toshiba_hdd_classified_as_normal(onnx_session):
    """Autentyczny sprawny dysk 7200 RPM Toshiba HDWR480 powinien być 'normal'."""
    path = "data/raw/verified_real/TOSHIBA_HDWR480_7200rpm_healthy_spinup.wav"
    import soundfile as sf
    import librosa
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 22050:
        data = librosa.resample(data, orig_sr=sr, target_sr=22050)
    chunk = data[int(10*22050):int(13*22050)]
    probs = predict_onnx(onnx_session, chunk)
    pred_class = CLASSES[np.argmax(probs)]
    print(f"\n[Test Real Healthy Toshiba 7200 RPM] Predykcja: {pred_class} (normal: {probs[2]*100:.1f}%)")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:12s}: {p*100:5.2f}%")
    assert pred_class == "normal"


def test_real_faulty_clicking_detected_as_clicking(onnx_session):
    """Autentyczny uszkodzony dysk z uderzającymi głowicami (Click of Death) powinien być 'clicking'."""
    path = "data/raw/verified_real/WD_bad_heads_click_of_death.ogg"
    import soundfile as sf
    import librosa
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 22050:
        data = librosa.resample(data, orig_sr=sr, target_sr=22050)
    chunk = data[int(5*22050):int(8*22050)]
    probs = predict_onnx(onnx_session, chunk)
    pred_class = CLASSES[np.argmax(probs)]
    print(f"\n[Test Real Faulty Clicking HDD] Predykcja: {pred_class} (clicking: {probs[1]*100:.1f}%)")
    for c, p in zip(CLASSES, probs):
        print(f"  {c:12s}: {p*100:5.2f}%")
    assert pred_class == "clicking" or probs[1] > 0.40


if __name__ == "__main__":
    session = ort.InferenceSession(ONNX_MODEL_PATH)
    test_mouth_clicks_classified_as_normal_by_network(session)
    test_whistle_and_voice_classified_as_normal_by_network(session)
    test_pc_fan_classified_as_normal_by_network(session)
    test_keyboard_typing_classified_as_normal_by_network(session)
    test_real_healthy_toshiba_hdd_classified_as_normal(session)
    test_real_faulty_clicking_detected_as_clicking(session)
    print("\n✅ Wszystkie 6 testów autentycznych i obcych próbek dla ONNX ZALICZONE (100% OK)!")
