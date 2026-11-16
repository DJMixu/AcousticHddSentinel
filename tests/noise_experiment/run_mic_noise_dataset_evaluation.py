#!/usr/bin/env python3
"""
Eksperyment Diagnostyczny: Weryfikacja Modelu na Plikach z Datasetu
oraz Odporności na Rzeczywisty Szum Otoczenia Nagrany Mikrofonem.

Single Source of Truth: ADR-001, ADR-002, AGENTS.md
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import soundfile as sf
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard

TARGET_SR = 22050
CHUNK_SAMPLES = 66150  # 3.0 s
CLASSES = ["beeping", "clicking", "normal", "scratching"]


def load_and_resample(path: Path) -> np.ndarray:
    y, sr = sf.read(str(path))
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if sr != TARGET_SR:
        import librosa
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    return y.astype(np.float32)


def mix_audio_with_noise(signal: np.ndarray, noise: np.ndarray, target_snr_db: float) -> np.ndarray:
    """Miesza sygnał HDD z szumem mikrofonu na zadanym poziomie SNR w decybelach."""
    sig_len = len(signal)
    if len(noise) < sig_len:
        # Powielenie szumu jeśli jest krótszy niż sygnał
        repeats = int(np.ceil(sig_len / len(noise)))
        noise_extended = np.tile(noise, repeats)[:sig_len]
    else:
        noise_extended = noise[:sig_len]

    sig_power = np.mean(signal ** 2)
    noise_power = np.mean(noise_extended ** 2)

    if sig_power < 1e-9 or noise_power < 1e-9:
        return signal

    # SNR = 10 * log10(sig_power / noise_power)
    # target_noise_power = sig_power / (10 ** (target_snr_db / 10))
    target_noise_power = sig_power / (10.0 ** (target_snr_db / 10.0))
    scaling_factor = np.sqrt(target_noise_power / noise_power)

    scaled_noise = noise_extended * scaling_factor
    mixed = signal + scaled_noise

    # Ochrona przed clippingiem
    max_val = np.max(np.abs(mixed))
    if max_val > 0.98:
        mixed = mixed / max_val * 0.95

    return mixed.astype(np.float32)


def run_pipeline():
    print("=" * 80)
    print("EKSPERYMENT: EWALUACJA MODELU Z SZUMEM Z MIKROFONU")
    print("=" * 80)

    model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model.onnx"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model_int8.onnx"

    print(f"1. Ładowanie modelu ONNX: {model_path.name}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    guard = DSPAdversarialGuard()

    # 2. Wczytanie nagranego szumu mikrofonu
    mic_noise_path = PROJECT_ROOT / "tests" / "noise_experiment" / "ambient_mic_noise_10s.wav"
    if not mic_noise_path.exists():
        mic_noise_path = PROJECT_ROOT / "tests" / "noise_experiment" / "ambient_mic_noise.wav"
    print(f"2. Wczytywanie nagranego szumu z mikrofonu: {mic_noise_path.name}")
    mic_noise = load_and_resample(mic_noise_path)
    mic_rms = float(np.sqrt(np.mean(mic_noise ** 2)))
    print(f"   - Długość szumu: {len(mic_noise)/TARGET_SR:.2f} s")
    print(f"   - Energia RMS szumu mikrofonu: {mic_rms:.6f}")

    # 3. TEST 0: Sam czysty szum z mikrofonu przepuszczony przez model
    print("\n" + "-" * 80)
    print("[ETAP 0] Test samego czystego szumu z mikrofonu użytkownika...")
    num_noise_windows = len(mic_noise) // CHUNK_SAMPLES
    noise_fault_count = 0
    noise_evaluations = []

    for nw in range(max(1, num_noise_windows)):
        n_chunk = mic_noise[nw * CHUNK_SAMPLES : (nw + 1) * CHUNK_SAMPLES]
        if len(n_chunk) < CHUNK_SAMPLES:
            n_chunk = np.pad(n_chunk, (0, CHUNK_SAMPLES - len(n_chunk)))

        tensor = encode_audio_chunk(n_chunk, sr=TARGET_SR, tensor_format="pytorch")
        raw_probs = sess.run(None, {"input_audio_tensor": np.expand_dims(tensor, 0).astype(np.float32)})[0][0]
        san_probs, guarded, reason = guard.protect_and_sanitize_predictions(raw_probs, tensor, waveform=n_chunk)
        fault_p = float(san_probs[0] + san_probs[1] + san_probs[3])

        noise_evaluations.append({
            "window": nw,
            "raw_probs": {CLASSES[i]: float(raw_probs[i]) for i in range(4)},
            "san_probs": {CLASSES[i]: float(san_probs[i]) for i in range(4)},
            "fault_p": fault_p,
            "guarded": guarded,
            "reason": reason
        })
        if fault_p >= 0.70:
            noise_fault_count += 1
        print(f"   - Okno szumu {nw}: P(fault)={fault_p*100:.1f}% | Beep={san_probs[0]*100:.1f}%, Click={san_probs[1]*100:.1f}%, Norm={san_probs[2]*100:.1f}%, Scratch={san_probs[3]*100:.1f}%")

    noise_fpr = (noise_fault_count / len(noise_evaluations)) * 100.0
    print(f"   -> Fałszywe alarmy na czystym szumie mikrofonu: {noise_fault_count}/{len(noise_evaluations)} (FPR = {noise_fpr:.1f}%)")
    if noise_fault_count == 0:
        print("   ✅ Czysty szum mikrofonu został poprawnie zakwalifikowany (BRAK FAŁSZYWYCH ALARMÓW)")

    # 4. Wybór reprezentatywnego zbioru plików z datasetu dla wszystkich 4 klas
    print("\n" + "-" * 80)
    print("[ETAP 1 & 2] Przygotowanie próbek testowych z datasetu dla 4 klas...")

    dataset_cat_dir = PROJECT_ROOT / "dataset_categorized"
    test_samples: Dict[str, List[Path]] = {cls: [] for cls in CLASSES}

    for cls in CLASSES:
        cls_dir = dataset_cat_dir / cls
        if cls_dir.exists():
            files = sorted(list(cls_dir.glob("*.wav")) + list(cls_dir.glob("*.mp3")) + list(cls_dir.glob("*.ogg")))
            # Filtrujemy reprezentatywne autentyczne pliki (po max 10 na klasę dla czytelnego benchmarku)
            test_samples[cls] = files[:10]

    # Dodanie autentycznych nagrań 4 dysków study do klasy normal
    study_dir = PROJECT_ROOT / "tests" / "recordings_4min_study"
    if study_dir.exists():
        study_files = list(study_dir.glob("*/*.wav"))[:8]
        test_samples["normal"].extend(study_files)

    total_files = sum(len(v) for v in test_samples.values())
    print(f"Wybrano łącznie {total_files} próbek referencyjnych:")
    for cls, flist in test_samples.items():
        print(f"  - {cls:12s}: {len(flist)} plików")

    # Warianty do zbadania:
    # 1. Czysty dźwięk (Clean)
    # 2. Szum otoczenia realistyczny (SNR = +18 dB)
    # 3. Szum otoczenia umiarkowany (SNR = +10 dB)
    # 4. Szum otoczenia silny (SNR = +4 dB)
    snr_variants = [
        ("Clean (Bez szumu)", None),
        ("Realistyczny szum mic (SNR +18 dB)", 18.0),
        ("Umiarkowany szum mic (SNR +10 dB)", 10.0),
        ("Silny szum mic (SNR +4 dB)", 4.0)
    ]

    all_benchmark_results = {}

    for variant_name, snr_val in snr_variants:
        print("\n" + "=" * 80)
        print(f"BENCHMARK: {variant_name}")
        print("=" * 80)

        tp = 0  # True Positives (awaria wykryta gdy była awaria)
        fn = 0  # False Negatives (awaria przeoczona)
        tn = 0  # True Negatives (sprawny dysk bez alarmu)
        fp = 0  # False Positives (sprawny dysk z fałszywym alarmem)
        class_stats = {cls: {"total": 0, "detected_correct": 0, "avg_fault_p": 0.0} for cls in CLASSES}

        for true_cls, file_list in test_samples.items():
            for fpath in file_list:
                try:
                    y = load_and_resample(fpath)
                except Exception as e:
                    continue

                if len(y) < CHUNK_SAMPLES:
                    continue

                # Bierzemy pierwsze 3-sekundowe okno (lub środkowe)
                mid_start = max(0, (len(y) - CHUNK_SAMPLES) // 2)
                chunk = y[mid_start : mid_start + CHUNK_SAMPLES]

                # Jeśli badamy wariant z szumem:
                if snr_val is not None:
                    # Mieszamy z nagranym szumem mikrofonu
                    eval_chunk = mix_audio_with_noise(chunk, mic_noise, snr_val)
                else:
                    eval_chunk = chunk

                tensor = encode_audio_chunk(eval_chunk, sr=TARGET_SR, tensor_format="pytorch")
                raw_probs = sess.run(None, {"input_audio_tensor": np.expand_dims(tensor, 0).astype(np.float32)})[0][0]
                san_probs, guarded, reason = guard.protect_and_sanitize_predictions(raw_probs, tensor, waveform=eval_chunk)

                # Werdykt binarny: P(fault) >= 70%
                fault_p = float(san_probs[0] + san_probs[1] + san_probs[3])
                is_fault_verdict = (fault_p >= 0.70)

                class_stats[true_cls]["total"] += 1
                class_stats[true_cls]["avg_fault_p"] += fault_p

                if true_cls == "normal":
                    if is_fault_verdict:
                        fp += 1
                    else:
                        tn += 1
                        class_stats[true_cls]["detected_correct"] += 1
                else:
                    # Klasa uszkodzenia
                    if is_fault_verdict:
                        tp += 1
                        class_stats[true_cls]["detected_correct"] += 1
                    else:
                        fn += 1

        for cls in CLASSES:
            if class_stats[cls]["total"] > 0:
                class_stats[cls]["avg_fault_p"] /= class_stats[cls]["total"]

        recall = (tp / (tp + fn) * 100.0) if (tp + fn) > 0 else 0.0
        precision = (tp / (tp + fp) * 100.0) if (tp + fp) > 0 else 0.0
        fpr = (fp / (fp + tn) * 100.0) if (fp + tn) > 0 else 0.0
        accuracy = ((tp + tn) / (tp + tn + fp + fn) * 100.0) if (tp + tn + fp + fn) > 0 else 0.0

        all_benchmark_results[variant_name] = {
            "snr_db": snr_val,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "recall": recall,
            "precision": precision,
            "fpr": fpr,
            "accuracy": accuracy,
            "class_stats": class_stats
        }

        print(f"Wyniki dla [{variant_name}]:")
        print(f"  - Dokładność binarna (Accuracy):     {accuracy:6.2f}%")
        print(f"  - Czułość na uszkodzenia (Recall):    {recall:6.2f}%  (TP={tp}, FN={fn})")
        print(f"  - Precyzja (Precision):               {precision:6.2f}%  (TP={tp}, FP={fp})")
        print(f"  - Wskaźnik fałszywych alarmów (FPR):  {fpr:6.2f}%  (FP={fp}, TN={tn})")
        print("  - Skuteczność per klasa:")
        for cls in CLASSES:
            cs = class_stats[cls]
            pct = (cs["detected_correct"] / cs["total"] * 100.0) if cs["total"] > 0 else 0.0
            print(f"    * {cls:12s}: {cs['detected_correct']:2d}/{cs['total']:2d} poprawnych ({pct:5.1f}%) | Średnie P(fault)={cs['avg_fault_p']*100:5.1f}%")

    # 5. Zapis raportu i wykresu
    output_dir = PROJECT_ROOT / "tests" / "noise_experiment"
    report_json_path = output_dir / "mic_noise_benchmark_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(all_benchmark_results, f, indent=2)
    print(f"\n[Raport] Zapisano raport JSON do: {report_json_path}")

    # Wykres słupkowy porównawczy
    variants = list(all_benchmark_results.keys())
    recalls = [all_benchmark_results[v]["recall"] for v in variants]
    fprs = [all_benchmark_results[v]["fpr"] for v in variants]
    precisions = [all_benchmark_results[v]["precision"] for v in variants]

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    x = np.arange(len(variants))
    width = 0.25

    rects1 = ax.bar(x - width, recalls, width, label="Czułość (Recall uszkodzeń %)", color="#22C55E")
    rects2 = ax.bar(x, precisions, width, label="Precyzja (Precision %)", color="#3B82F6")
    rects3 = ax.bar(x + width, fprs, width, label="Fałszywe alarmy (FPR %)", color="#EF4444")

    ax.set_ylabel("Wartość w procentach (%)", fontsize=11, fontweight="bold")
    ax.set_title("Wpływ Rzeczywistego Szumu Mikrofonu na Czułość i Odporność Modelu HDD", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([v.replace(" ", "\n") for v in variants], fontsize=9)
    ax.set_ylim(0, 115)
    ax.axhline(70, color="gray", linestyle="--", linewidth=1, label="Wymóg SLA Recall >= 70%")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)

    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f"{height:.1f}%",
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()
    chart_path = output_dir / "mic_noise_benchmark_chart.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"[Wykres] Zapisano wykres porównawczy: {chart_path}")

    print("\n" + "=" * 80)
    print("EKSPERYMENT ZAKOŃCZONY POMYŚLNIE!")
    print("=" * 80)


if __name__ == "__main__":
    run_pipeline()
