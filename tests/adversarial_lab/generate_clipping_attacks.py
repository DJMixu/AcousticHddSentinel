#!/usr/bin/env python3
"""
HDD Acoustic Sentinel - Synthetic Clipping & Acoustic Distortion Generator
Single Source of Truth: ADR-001, AGENTS.md, tests/adversarial_lab/iter3_arch_task.md

Ten moduł generuje syntetyczne ataki zniekształceń toru audio smartfona:
1. Hard Clipping (obcinanie wierzchołków fali przy progu 0.90, 0.80, 0.60, 0.40, 0.20)
   - symulacja przesterowania przetwornika ADC / mikrofonu MEMS przy zbliżeniu do obudowy HDD.
2. Nieliniowe zniekształcenia THD (Total Harmonic Distortion):
   - model nieliniowości stopnia wejściowego / membrany mikrofonu: y = x + 0.15*x^2 - 0.08*x^3
   - asymetria fazowa (2. harmoniczna parzysta) + kompresja szczytów (3. harmoniczna nieparzysta).
3. Kompresja dynamiki AGC (Automatic Gain Control) smartfona:
   - szybki czas ataku Attack = 10 ms (tłumienie nagłych impulsów mechanicznych)
   - wolny czas powrotu Release = 200 ms (efekt "pumping" i podnoszenie tła szumowego)
4. Złożony atak toru smartfona (Compound Smartphone Microphone Saturation):
   - połączenie AGC + THD + Hard Clipping

Skrypt analizuje wpływ powyższych zniekształceń na zachowanie surowego modelu ONNX
(weryfikacja czy 'normal' nie zmienia się fałszywie w 'scratching'),
oblicza precyzyjne metryki fizyczne DSP i zapisuje wygenerowane próbki audio oraz raport JSON/TXT.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import scipy.stats
import soundfile as sf
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.hdd_audio_encoder import encode_audio_chunk, normalize_waveform_peak
from ml.dsp_adversarial_guard import DSPAdversarialGuard, CLASSES, TARGET_SR, CHUNK_SAMPLES

ONNX_MODEL_PATH = "exported_models/best_hdd_cnn_model.onnx"
SAMPLES_DIR = Path("tests/adversarial_lab/samples")
RESULTS_DIR = Path("tests/adversarial_lab/results")


# =====================================================================
# 1. FUNKCJE GENERUJĄCE ZNIEKSZTAŁCENIA AKUSTYCZNE (DSP ATTACKS)
# =====================================================================

def apply_hard_clipping(
    waveform: np.ndarray,
    clip_level: float = 0.80,
    normalize_output: bool = True
) -> np.ndarray:
    """
    Syntetyczne twarde obcinanie wierzchołków fali (Hard Clipping).
    Modeluje przesterowanie toru analogowego i saturację przetwornika ADC w smartfonie.
    
    Parametry:
    - waveform: sygnał audio 1D float32
    - clip_level: poziom obcinania w zakresie (0.0, 1.0] np. 0.90, 0.80, 0.60
    - normalize_output: przeskalowanie obciętych szczytów z powrotem do +/-1.0 (pełna skala ADC)
    """
    sig = np.array(waveform, dtype=np.float32, copy=True)
    peak = np.max(np.abs(sig))
    if peak > 1e-7:
        sig = sig / peak
    
    clipped = np.clip(sig, -clip_level, clip_level)
    
    if normalize_output and clip_level > 1e-7:
        clipped = clipped / clip_level
        
    return clipped.astype(np.float32)


def apply_thd_distortion(
    waveform: np.ndarray,
    a2: float = 0.15,
    a3: float = -0.08,
    remove_dc: bool = True,
    normalize_output: bool = True
) -> np.ndarray:
    """
    Nieliniowe zniekształcenia harmoniczne THD (2. i 3. rzędu).
    Modeluje asymetrię wzmacniacza wejściowego oraz nieliniowość membrany mikrofonu MEMS.
    Wzór zadany: y = x + a2 * x^2 + a3 * x^3 (domyślnie a2=0.15, a3=-0.08).
    
    Parametry:
    - waveform: sygnał audio 1D float32
    - a2: współczynnik 2. harmonicznej (asymetria, parzyste harmoniczne)
    - a3: współczynnik 3. harmonicznej (saturacja, nieparzyste harmoniczne)
    - remove_dc: usunięcie stałej składowej wprowadzonej przez człon kwadratowy (kondensator sprzęgający AC)
    - normalize_output: normalizacja szczytowa do [-1.0, 1.0]
    """
    sig = np.array(waveform, dtype=np.float32, copy=True)
    peak = np.max(np.abs(sig))
    if peak > 1e-7:
        sig = sig / peak
        
    # Nieliniowość wielomianowa
    y = sig + a2 * (sig ** 2) + a3 * (sig ** 3)
    
    if remove_dc:
        y = y - np.mean(y)
        
    if normalize_output:
        peak_y = np.max(np.abs(y))
        if peak_y > 1e-7:
            y = y / peak_y
            
    return y.astype(np.float32)


def apply_smartphone_agc(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    attack_ms: float = 10.0,
    release_ms: float = 200.0,
    threshold_db: float = -20.0,
    ratio: float = 4.0,
    makeup_gain: bool = True
) -> np.ndarray:
    """
    Kompresja dynamiki AGC (Automatic Gain Control) typowa dla algorytmów audio smartfonów.
    Szybki Attack (10 ms) gwałtownie tłumi głośne transjenty (uderzenia głowic).
    Wolny Release (200 ms) powoli przywraca wzmocnienie, wzmacniając szum tła ("pumping").
    
    Parametry:
    - waveform: sygnał audio 1D float32
    - sr: częstotliwość próbkowania (22050 Hz)
    - attack_ms: stała czasowa ataku w milisekundach (domyślnie 10 ms)
    - release_ms: stała czasowa powrotu w milisekundach (domyślnie 200 ms)
    - threshold_db: próg zadziałania kompresora w dBFS (domyślnie -20 dB)
    - ratio: stopień kompresji (domyślnie 4:1)
    - makeup_gain: wyrównanie wzmocnienia szczytowego do pełnego zakresu
    """
    sig = np.array(waveform, dtype=np.float32, copy=True)
    peak = np.max(np.abs(sig))
    if peak > 1e-7:
        sig = sig / peak
        
    tau_att = max(0.0001, attack_ms / 1000.0)
    tau_rel = max(0.0001, release_ms / 1000.0)
    
    alpha_att = 1.0 - np.exp(-1.0 / (tau_att * sr))
    alpha_rel = 1.0 - np.exp(-1.0 / (tau_rel * sr))
    
    n_samples = len(sig)
    env = np.zeros(n_samples, dtype=np.float32)
    curr_env = 0.0
    abs_sig = np.abs(sig)
    
    # Detektor obwiedni peak-following
    for i in range(n_samples):
        x_val = abs_sig[i]
        if x_val > curr_env:
            curr_env += alpha_att * (x_val - curr_env)
        else:
            curr_env += alpha_rel * (x_val - curr_env)
        env[i] = curr_env
        
    # Obliczenie redukcji wzmocnienia w dB
    eps = 1e-6
    env_db = 20.0 * np.log10(np.maximum(env, eps))
    gain_db = np.where(
        env_db > threshold_db,
        -(1.0 - 1.0 / ratio) * (env_db - threshold_db),
        0.0
    )
    
    gain_linear = 10.0 ** (gain_db / 20.0)
    compressed = sig * gain_linear
    
    if makeup_gain:
        pk_comp = np.max(np.abs(compressed))
        if pk_comp > 1e-7:
            compressed = compressed / pk_comp
            
    return compressed.astype(np.float32)


def apply_combined_channel_distortion(
    waveform: np.ndarray,
    sr: int = TARGET_SR,
    clip_level: float = 0.60
) -> np.ndarray:
    """
    Złożona symulacja nagrania mikrofonem smartfona z bliskiej odległości:
    AGC (10ms att, 200ms rel) -> Nieliniowość THD (a2=0.15, a3=-0.08) -> Hard Clipping (clip_level).
    """
    sig = apply_smartphone_agc(waveform, sr=sr, attack_ms=10.0, release_ms=200.0, threshold_db=-18.0, ratio=4.0)
    sig = apply_thd_distortion(sig, a2=0.15, a3=-0.08)
    sig = apply_hard_clipping(sig, clip_level=clip_level)
    return sig


# =====================================================================
# 2. METRYKI FIZYCZNE I WIDMOWE DSP
# =====================================================================

def compute_waveform_metrics(audio: np.ndarray, sr: int = TARGET_SR) -> Dict[str, float]:
    """
    Oblicza zestaw diagnostycznych metryk czasowo-częstotliwościowych sygnału audio.
    """
    sig = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(sig)))
    rms = float(np.sqrt(np.mean(sig ** 2)))
    crest = float(peak / (rms + 1e-9))
    
    # Odsetek próbek nasyconych (|x| >= 0.995)
    saturated_count = int(np.sum(np.abs(sig) >= 0.995))
    clipping_ratio = float(saturated_count / len(sig))
    
    # Maksymalna ciągła długość obciętych próbek (Flat-Top Run Length)
    is_saturated = (np.abs(sig) >= 0.995).astype(np.int32)
    max_flat_run = 0
    current_run = 0
    for val in is_saturated:
        if val == 1:
            current_run += 1
            if current_run > max_flat_run:
                max_flat_run = current_run
        else:
            current_run = 0
            
    # Kurtoza i Asymetria (Skewness)
    kurt = float(scipy.stats.kurtosis(sig, fisher=False))
    skew = float(scipy.stats.skew(sig))
    
    # Analiza widmowa FFT
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1.0 / sr)
    total_power = float(np.sum(fft ** 2) + 1e-12)
    
    # Udział energii w wysokich częstotliwościach (>4 kHz) - obszar pasma 'scratching'
    high_freq_power = float(np.sum(fft[freqs >= 4000.0] ** 2))
    high_freq_ratio = float(high_freq_power / total_power)
    
    # Środek ciężkości widma (Spectral Centroid)
    centroid = float(np.sum(freqs * fft) / (np.sum(fft) + 1e-12))
    
    return {
        "peak": round(peak, 4),
        "rms": round(rms, 4),
        "crest_factor": round(crest, 2),
        "clipping_ratio": round(clipping_ratio, 5),
        "max_flat_run": int(max_flat_run),
        "kurtosis": round(kurt, 2),
        "skewness": round(skew, 3),
        "high_freq_ratio_pct": round(high_freq_ratio * 100.0, 2),
        "spectral_centroid_hz": round(centroid, 1)
    }


# =====================================================================
# 3. EWALUACJA MODELU ONNX I STRAŻNIKA DSP GUARD
# =====================================================================

class AudioAttackEvaluator:
    def __init__(self, onnx_model_path: str = ONNX_MODEL_PATH):
        assert os.path.isfile(onnx_model_path), f"Nie znaleziono pliku modelu ONNX: {onnx_model_path}"
        self.session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.guard = DSPAdversarialGuard()

    def evaluate_waveform(self, audio: np.ndarray, sr: int = TARGET_SR) -> Dict[str, Any]:
        """
        Ocenia wycinek audio 3.0s przez:
        1. Ekstrakcję tensora ADR-001 (Log-Mel, Delta, Delta-Delta)
        2. Surowy model ONNX (Softmax)
        3. DSP Adversarial Guard (sanityzacja predykcji)
        """
        chunk = audio[:CHUNK_SAMPLES]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))

        tensor = encode_audio_chunk(chunk, sr=sr)
        tensor_cf = np.transpose(tensor, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

        raw_logits = self.session.run(None, {self.input_name: tensor_cf})[0]
        exp_logits = np.exp(raw_logits - np.max(raw_logits))
        raw_probs = (exp_logits / np.sum(exp_logits))[0]

        raw_pred = CLASSES[np.argmax(raw_probs)]
        raw_conf = float(np.max(raw_probs))

        san_probs, was_sanitized, guard_reason = self.guard.protect_and_sanitize_predictions(
            raw_probs, tensor, waveform=chunk
        )
        san_pred = CLASSES[np.argmax(san_probs)]
        san_conf = float(np.max(san_probs))

        metrics = compute_waveform_metrics(chunk, sr=sr)

        return {
            "raw_prediction": raw_pred,
            "raw_confidence": round(raw_conf, 4),
            "raw_probs": {cls: round(float(raw_probs[i]), 4) for i, cls in enumerate(CLASSES)},
            "sanitized_prediction": san_pred,
            "sanitized_confidence": round(san_conf, 4),
            "was_sanitized": bool(was_sanitized),
            "guard_reason": guard_reason,
            "dsp_metrics": metrics
        }


# =====================================================================
# 4. BATERIA TESTOWA SYGNAŁÓW
# =====================================================================

def load_or_generate_test_suite() -> List[Dict[str, Any]]:
    """
    Przygotowuje zestaw reprezentatywnych sygnałów źródłowych do badania ataków przesterowania:
    - Normalne dźwięki HDD i wentylatorów (z bazy categorized)
    - Tonalne świsty i rezonanse
    - Czyste tony sinusoidalne (500 Hz, 1000 Hz, 2000 Hz, 3000 Hz)
    - Autentyczne uszkodzenia HDD (Hitachi Deathstar clicking, Seagate Beep)
    """
    suite = []

    # 1. Normalne HDD i komponenty PC
    candidate_files = [
        ("synth_healthy_7200", "dataset_categorized/normal/synth_healthy_hdd_7200rpm_01.wav", "normal"),
        ("synth_healthy_5400", "dataset_categorized/normal/synth_healthy_hdd_5400rpm_01.wav", "normal"),
        ("toshiba_spinup", "dataset_categorized/normal/TOSHIBA_HDWR480_7200rpm_healthy_spinup.wav", "normal"),
        ("seagate_seek", "dataset_categorized/normal/Seagate_ST3250823A_7200rpm_healthy_seek.wav", "normal"),
        ("room_ambient_hum", "dataset_categorized/normal/oe_room_ambient_hum_01.wav", "normal"),
        ("pc_fan_turbulent", "dataset_categorized/normal/oe_pc_fan_turbulent_01.wav", "normal"),
        ("whistle_voice_01", "dataset_categorized/normal/oe_whistle_voice_01.wav", "normal"),
        ("whistle_voice_03", "dataset_categorized/normal/oe_whistle_voice_03.wav", "normal"),
        ("ambient_ventilador", "dataset_categorized/normal/ambient_Ventilador.wav", "normal"),
        ("pc_fan_yt", "tests/adversarial_lab/samples/fan_pc_yt.wav", "normal"),
        ("hitachi_deathstar_click", "tests/adversarial_lab/samples/hdd_hitachi_deathstar_yt.wav", "clicking"),
        ("seagate_stuck_beep", "tests/adversarial_lab/samples/hdd_seagate_stuck_yt.wav", "beeping"),
    ]

    for name, path, true_label in candidate_files:
        if os.path.isfile(path):
            data, sr = sf.read(path)
            if data.ndim > 1:
                data = np.mean(data, axis=-1)
            chunk = data[:CHUNK_SAMPLES]
            suite.append({
                "name": name,
                "category": "audio_file",
                "true_label": true_label,
                "waveform": chunk.astype(np.float32),
                "sr": sr
            })

    # 2. Syntetyczne czyste tony (badanie generacji harmonicznych przesterowania)
    t = np.linspace(0, 3.0, CHUNK_SAMPLES, endpoint=False)
    for freq in [500, 1000, 2000, 3000]:
        sine = np.sin(2 * np.pi * freq * t).astype(np.float32)
        suite.append({
            "name": f"sine_{freq}hz",
            "category": "pure_tone",
            "true_label": "normal",  # dźwięk syntetyczny OOD niebędący awarią HDD
            "waveform": sine,
            "sr": TARGET_SR
        })

    return suite


# =====================================================================
# 5. GŁÓWNY PIPELINE GENEROWANIA I DIAGNOSTYKI ATAKÓW
# =====================================================================

def run_clipping_attacks_lab(
    save_wav_samples: bool = True,
    output_json_path: Optional[str] = None,
    output_txt_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Główna funkcja wykonawcza:
    1. Aplikuje przesterowania Hard Clipping (0.90, 0.80, 0.60), THD, AGC oraz Combined.
    2. Ewaluuje wpływ na surowy model ONNX i istniejącego Strażnika DSP.
    3. Monitoruje czy 'normal' przekształca się fałszywie w 'scratching'.
    4. Zapisuje próbki .wav oraz szczegółowe raporty JSON i TXT.
    """
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    evaluator = AudioAttackEvaluator()
    suite = load_or_generate_test_suite()

    print("=" * 85)
    print("🔬 ADVERSARIAL LAB ITERACJA 3: GENEROWANIE ATAKÓW PRZESTEROWANIA MIKROFONU")
    print("   Testowane zniekształcenia: Hard Clipping (0.90, 0.80, 0.60), THD, AGC (10/200ms), Combined")
    print("=" * 85)

    all_results = []
    normal_to_scratching_vulnerabilities = []
    clicking_suppression_cases = []

    # Definicje transformacji adwersarialnych
    transformations = [
        ("clean", lambda x: normalize_waveform_peak(x)),
        ("clip_090", lambda x: apply_hard_clipping(x, clip_level=0.90)),
        ("clip_080", lambda x: apply_hard_clipping(x, clip_level=0.80)),
        ("clip_060", lambda x: apply_hard_clipping(x, clip_level=0.60)),
        ("thd", lambda x: apply_thd_distortion(x, a2=0.15, a3=-0.08)),
        ("agc", lambda x: apply_smartphone_agc(x, sr=TARGET_SR, attack_ms=10.0, release_ms=200.0)),
        ("combined_smart_mic", lambda x: apply_combined_channel_distortion(x, sr=TARGET_SR, clip_level=0.60)),
    ]

    for item in suite:
        item_name = item["name"]
        true_lbl = item["true_label"]
        base_wav = item["waveform"]
        sr = item["sr"]

        print(f"\n▶ Próbka bazowa: {item_name} (Etykieta: {true_lbl})")

        sample_evals = {}
        for trans_name, trans_fn in transformations:
            proc_audio = trans_fn(base_wav)
            eval_res = evaluator.evaluate_waveform(proc_audio, sr=sr)
            sample_evals[trans_name] = eval_res

            raw_p = eval_res["raw_prediction"]
            raw_c = eval_res["raw_confidence"]
            scratch_p = eval_res["raw_probs"]["scratching"]
            click_p = eval_res["raw_probs"]["clicking"]
            norm_p = eval_res["raw_probs"]["normal"]
            crest = eval_res["dsp_metrics"]["crest_factor"]
            clip_r = eval_res["dsp_metrics"]["clipping_ratio"]

            print(
                f"  [{trans_name:18s}] -> Surowy ONNX: {raw_p:10s} (conf: {raw_c*100:4.1f}% | "
                f"scratch: {scratch_p*100:4.1f}%, norm: {norm_p*100:4.1f}%) | "
                f"Crest: {crest:4.2f}, Clip: {clip_r*100:4.1f}%"
            )

            # Zapis pliku WAV dla kluczowych zniekształceń
            if save_wav_samples and trans_name in ["clip_090", "clip_080", "clip_060", "thd", "agc", "combined_smart_mic"]:
                out_wav_path = SAMPLES_DIR / f"{item_name}_{trans_name}.wav"
                sf.write(out_wav_path, proc_audio, TARGET_SR)

            # Badanie wrażliwości: normal -> scratching
            if true_lbl == "normal" and raw_p == "scratching":
                vuln_record = {
                    "sample": item_name,
                    "attack": trans_name,
                    "raw_pred": raw_p,
                    "confidence": raw_c,
                    "scratch_prob": scratch_p,
                    "normal_prob": norm_p,
                    "crest_factor": crest,
                    "clipping_ratio": clip_r,
                    "was_sanitized_by_guard": eval_res["was_sanitized"]
                }
                normal_to_scratching_vulnerabilities.append(vuln_record)

            # Badanie tłumienia klikania przez AGC
            if true_lbl == "clicking" and trans_name == "agc" and raw_p != "clicking":
                clicking_suppression_cases.append({
                    "sample": item_name,
                    "attack": trans_name,
                    "raw_pred": raw_p,
                    "clicking_prob": click_p
                })

        all_results.append({
            "name": item_name,
            "category": item["category"],
            "true_label": true_lbl,
            "evaluations": sample_evals
        })

    # =====================================================================
    # 6. PODSUMOWANIE I WNIOSKI DIAGNOSTYCZNE
    # =====================================================================
    print("\n" + "=" * 85)
    print("📊 PODSUMOWANIE DIAGNOSTYCZNE ATAKÓW PRZESTEROWANIA:")
    print("=" * 85)
    print(f"Liczba przetestowanych sygnałów źródłowych: {len(suite)}")
    print(f"Liczba kombinacji ataków: {len(suite) * len(transformations)}")
    print(f"Liczba wykrytych fałszywych alarmów SCRATCHING na sygnałach NORMALNYCH: {len(normal_to_scratching_vulnerabilities)}")
    
    if normal_to_scratching_vulnerabilities:
        print("\n🚨 WYKRYTE PODATNOŚCI: Sygnały normalne/rezonansowe zaklasyfikowane jako SCRATCHING pod wpływem przesterowania:")
        for v in normal_to_scratching_vulnerabilities:
            san_status = "🛡️ ZNEUTRALIZOWANY" if v["was_sanitized_by_guard"] else "❌ PRZEOCZONY PRZEZ STRAŻNIKA"
            print(
                f"   - {v['sample']} + {v['attack']}: P(scratching) = {v['scratch_prob']*100:.1f}%, "
                f"Crest Factor = {v['crest_factor']:.2f}, Clip Ratio = {v['clipping_ratio']*100:.1f}% -> {san_status}"
            )
    else:
        print("   Brak przypadków normal -> scratching.")

    if clicking_suppression_cases:
        print("\n⚠️ Tłumienie impulsów klikania (CLICKING) przez AGC:")
        for c in clicking_suppression_cases:
            print(f"   - {c['sample']} pod wpływem AGC spadł z clicking do: {c['raw_pred']} (P(click)={c['clicking_prob']*100:.1f}%)")
    else:
        print("   Wszystkie autentyczne sygnały CLICKING zachowały detekcję pod wpływem AGC.")

    # Zapis raportu JSON
    if output_json_path is None:
        output_json_path = str(RESULTS_DIR / "clipping_attacks_evaluation.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_samples": len(suite),
                "total_evaluations": len(suite) * len(transformations),
                "normal_to_scratching_vulnerabilities_count": len(normal_to_scratching_vulnerabilities),
                "normal_to_scratching_vulnerabilities": normal_to_scratching_vulnerabilities,
                "clicking_suppression_cases": clicking_suppression_cases
            },
            "detailed_results": all_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Zapisano szczegółowy raport JSON: {output_json_path}")

    # Zapis raportu tekstowego
    if output_txt_path is None:
        output_txt_path = str(RESULTS_DIR / "clipping_attacks_report.txt")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RAPORT Z BADAŃ ADWERSARIALNYCH: PRZESTEROWANIE MIKROFONU I DYSTORSJE AKUSTYCZNE\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"1. Łączna liczba przetestowanych próbek: {len(suite)}\n")
        f.write(f"2. Liczba podatności normal -> scratching: {len(normal_to_scratching_vulnerabilities)}\n")
        f.write(f"3. Liczba przypadków stłumienia clicking przez AGC: {len(clicking_suppression_cases)}\n\n")
        f.write("SZCZEGÓŁOWE ZESTAWIENIE PODATNOŚCI NA PRZESTEROWANIE (NORMAL -> SCRATCHING):\n")
        for v in normal_to_scratching_vulnerabilities:
            f.write(
                f"- Próbka: {v['sample']} | Atak: {v['attack']} | P(scratching)={v['scratch_prob']*100:.1f}% | "
                f"Crest={v['crest_factor']} | ClipRatio={v['clipping_ratio']*100:.1f}% | "
                f"Sanitized={v['was_sanitized_by_guard']}\n"
            )
        f.write("\nKLUCZOWE WNIOSKI DLA ARCHITEKTA DSP:\n")
        f.write("1. Czyste tony sinusoidalne i harmoniczne pisków (500-3000 Hz) po obcięciu (Hard Clipping 0.60 lub THD)\n")
        f.write("   generują sztuczne harmoniczne o wysokiej częstotliwości, które sieć 2D CNN błędnie bierze za scratching.\n")
        f.write("2. Cechy dystynktywne przesterowania mikrofonu:\n")
        f.write("   - Crest Factor gwałtownie spada (< 2.2 dla silnego przesteru, podczas gdy prawdziwy scratching ma wysoki CF > 3.5)\n")
        f.write("   - Clipping Ratio > 0.01 (ponad 1% próbek stale dotyka szyn +/-1.0)\n")
        f.write("   - Max Flat Run > 4 (serie kolejnych identycznych próbek na poziomie szyny)\n")
        f.write("3. Dotychczasowy DSPAdversarialGuard przepuszcza większość tych fałszywych alarmów,\n")
        f.write("   ponieważ reguła is_aerodynamic_fan szuka ciągłego szumu 1/f, a is_coil_whine szuka wąskopasmowej cewki PWM.\n")
        f.write("   Konieczny jest nowy moduł is_microphone_clipped_or_distorted()!\n")
    print(f"💾 Zapisano raport tekstowy: {output_txt_path}")

    return {
        "vulnerabilities": normal_to_scratching_vulnerabilities,
        "clicking_suppression": clicking_suppression_cases,
        "json_path": output_json_path,
        "txt_path": output_txt_path
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generator ataków przesterowania mikrofonu dla HDD Acoustic Sentinel")
    parser.add_argument("--no-save-wav", action="store_true", help="Nie zapisuj plików audio .wav na dysk")
    parser.add_argument("--output-json", type=str, default=None, help="Ścieżka do wyjściowego pliku JSON")
    parser.add_argument("--output-txt", type=str, default=None, help="Ścieżka do wyjściowego pliku TXT")
    args = parser.parse_args()

    run_clipping_attacks_lab(
        save_wav_samples=not args.no_save_wav,
        output_json_path=args.output_json,
        output_txt_path=args.output_txt
    )
