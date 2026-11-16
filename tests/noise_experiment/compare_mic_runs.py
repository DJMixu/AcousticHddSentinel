#!/usr/bin/env python3
"""
Porównanie wpływu dwóch niezależnych nagrań szumu z mikrofonu użytkownika na model Acoustic HDD Sentinel.
"""

import sys
import os
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard

TARGET_SR = 22050
CHUNK_SAMPLES = 66150  # 3.0 s
CLASSES = ["beeping", "clicking", "normal", "scratching"]

def load_resampled(p: Path) -> np.ndarray:
    y, sr = sf.read(str(p))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != TARGET_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y.astype(np.float32)

def mix(sig: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    if len(noise) < len(sig):
        reps = int(np.ceil(len(sig) / len(noise)))
        noise_ext = np.tile(noise, reps)[:len(sig)]
    else:
        noise_ext = noise[:len(sig)]

    sp = np.mean(sig ** 2)
    npow = np.mean(noise_ext ** 2)
    if sp < 1e-9 or npow < 1e-9:
        return sig

    target_np = sp / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_np / npow)
    m = sig + noise_ext * scale
    maxv = np.max(np.abs(m))
    if maxv > 0.98:
        m = m / maxv * 0.95
    return m.astype(np.float32)

def evaluate_samples(samples, sess, guard, noise=None, snr=None):
    tp, fn, tn, fp = 0, 0, 0, 0
    class_stats = {c: {"total": 0, "correct": 0, "avg_fault_p": 0.0} for c in CLASSES}

    for true_cls, files in samples.items():
        for f in files:
            try:
                y = load_resampled(f)
            except Exception:
                continue
            if len(y) < CHUNK_SAMPLES:
                continue
            mid = max(0, (len(y) - CHUNK_SAMPLES) // 2)
            chunk = y[mid : mid + CHUNK_SAMPLES]
            if noise is not None and snr is not None:
                eval_c = mix(chunk, noise, snr)
            else:
                eval_c = chunk

            t = encode_audio_chunk(eval_c, sr=TARGET_SR, tensor_format="pytorch")
            raw = sess.run(None, {"input_audio_tensor": np.expand_dims(t, 0).astype(np.float32)})[0][0]
            san, _, _ = guard.protect_and_sanitize_predictions(raw, t, waveform=eval_c)
            fault_p = float(san[0] + san[1] + san[3])
            verdict_fault = (fault_p >= 0.70)

            class_stats[true_cls]["total"] += 1
            class_stats[true_cls]["avg_fault_p"] += fault_p

            if true_cls == "normal":
                if verdict_fault:
                    fp += 1
                else:
                    tn += 1
                    class_stats[true_cls]["correct"] += 1
            else:
                if verdict_fault:
                    tp += 1
                    class_stats[true_cls]["correct"] += 1
                else:
                    fn += 1

    for c in CLASSES:
        if class_stats[c]["total"] > 0:
            class_stats[c]["avg_fault_p"] /= class_stats[c]["total"]

    recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
    prec = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
    fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
    acc = ((tp + tn) / (tp + tn + fp + fn) * 100.0) if (tp + tn + fp + fn) > 0 else 0.0
    return {"acc": acc, "recall": recall, "prec": prec, "fpr": fpr, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "stats": class_stats}

def main():
    model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model.onnx"
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    guard = DSPAdversarialGuard()

    noise1 = load_resampled(PROJECT_ROOT / "tests/noise_experiment/ambient_mic_noise_10s.wav")
    noise2 = load_resampled(PROJECT_ROOT / "tests/noise_experiment/ambient_mic_noise_run2.wav")

    print(f"Noise Run 1: RMS={np.sqrt(np.mean(noise1**2)):.6f}")
    print(f"Noise Run 2: RMS={np.sqrt(np.mean(noise2**2)):.6f}")

    # Test czystego szumu Run 2
    n_windows = len(noise2) // CHUNK_SAMPLES
    noise2_faults = 0
    print("\n--- Test czystego szumu Run 2 ---")
    for nw in range(max(1, n_windows)):
        n_c = noise2[nw * CHUNK_SAMPLES : (nw + 1) * CHUNK_SAMPLES]
        t = encode_audio_chunk(n_c, sr=TARGET_SR, tensor_format="pytorch")
        raw = sess.run(None, {"input_audio_tensor": np.expand_dims(t, 0).astype(np.float32)})[0][0]
        san, _, _ = guard.protect_and_sanitize_predictions(raw, t, waveform=n_c)
        fp = float(san[0] + san[1] + san[3])
        if fp >= 0.70:
            noise2_faults += 1
        print(f"Okno {nw}: P(fault)={fp*100:.1f}% | Normal={san[2]*100:.1f}%")
    print(f"Fałszywe alarmy na czystym szumie Run 2: {noise2_faults}/{n_windows} (FPR = {noise2_faults/n_windows*100:.1f}%)")

    # Wybór próbek z datasetu
    cat_dir = PROJECT_ROOT / "dataset_categorized"
    samples = {c: [] for c in CLASSES}
    for c in CLASSES:
        if (cat_dir / c).exists():
            fl = sorted(list((cat_dir / c).glob("*.wav")) + list((cat_dir / c).glob("*.mp3")))[:10]
            samples[c] = fl
    study_files = list((PROJECT_ROOT / "tests/recordings_4min_study").glob("*/*.wav"))[:8]
    samples["normal"].extend(study_files)

    # Ewaluacja porównawcza
    res_clean = evaluate_samples(samples, sess, guard)
    res_n1_18 = evaluate_samples(samples, sess, guard, noise=noise1, snr=18.0)
    res_n2_18 = evaluate_samples(samples, sess, guard, noise=noise2, snr=18.0)

    res_n1_10 = evaluate_samples(samples, sess, guard, noise=noise1, snr=10.0)
    res_n2_10 = evaluate_samples(samples, sess, guard, noise=noise2, snr=10.0)

    res_n1_4 = evaluate_samples(samples, sess, guard, noise=noise1, snr=4.0)
    res_n2_4 = evaluate_samples(samples, sess, guard, noise=noise2, snr=4.0)

    print("\n" + "=" * 80)
    print("ZESTAWIENIE PORÓWNAWCZE: RUN 1 vs RUN 2")
    print("=" * 80)
    print(f"{'WARIANT TESTU':<35} | {'ACC':<7} | {'RECALL':<8} | {'PRECISION':<10} | {'FPR':<6}")
    print("-" * 75)
    print(f"{'Czysty dźwięk (Clean)':<35} | {res_clean['acc']:5.1f}% | {res_clean['recall']:6.1f}%  | {res_clean['prec']:7.1f}%   | {res_clean['fpr']:4.1f}%")
    print("-" * 75)
    print(f"{'Szum Run 1 (SNR +18 dB)':<35} | {res_n1_18['acc']:5.1f}% | {res_n1_18['recall']:6.1f}%  | {res_n1_18['prec']:7.1f}%   | {res_n1_18['fpr']:4.1f}%")
    print(f"{'Szum Run 2 (SNR +18 dB)':<35} | {res_n2_18['acc']:5.1f}% | {res_n2_18['recall']:6.1f}%  | {res_n2_18['prec']:7.1f}%   | {res_n2_18['fpr']:4.1f}%")
    print("-" * 75)
    print(f"{'Szum Run 1 (SNR +10 dB)':<35} | {res_n1_10['acc']:5.1f}% | {res_n1_10['recall']:6.1f}%  | {res_n1_10['prec']:7.1f}%   | {res_n1_10['fpr']:4.1f}%")
    print(f"{'Szum Run 2 (SNR +10 dB)':<35} | {res_n2_10['acc']:5.1f}% | {res_n2_10['recall']:6.1f}%  | {res_n2_10['prec']:7.1f}%   | {res_n2_10['fpr']:4.1f}%")
    print("-" * 75)
    print(f"{'Szum Run 1 (SNR +4 dB)':<35}  | {res_n1_4['acc']:5.1f}% | {res_n1_4['recall']:6.1f}%  | {res_n1_4['prec']:7.1f}%   | {res_n1_4['fpr']:4.1f}%")
    print(f"{'Szum Run 2 (SNR +4 dB)':<35}  | {res_n2_4['acc']:5.1f}% | {res_n2_4['recall']:6.1f}%  | {res_n2_4['prec']:7.1f}%   | {res_n2_4['fpr']:4.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    main()
