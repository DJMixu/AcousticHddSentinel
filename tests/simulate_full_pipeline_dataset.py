#!/usr/bin/env python3
"""
HDD Acoustic Sentinel - Symulacja Pełnego Potoku Aplikacji dla Całego Datasetu
Single Source of Truth: ADR-001, ADR-002, AGENTS.md

Cel skryptu:
1. Symulacja nagrywania mikrofonowego dla każdego z 388 plików audio w całym zbiorze (dataset_categorized).
2. Przepuszczenie sygnału przez kompletny potok DSP:
   - Resampling do 22050 Hz (1 kanał)
   - Bramka ciszy cyfrowej ADR-001 (RMS < 0.002)
   - Wycinki 3.0s (66 150 próbek)
   - Normalizacja szczytowa fali (Peak Normalization)
   - Enkoder Log-Mel STFT [128][259] + Delta + Delta-Delta
   - Z-Score standardisation z eps = 1e-6
   - Analizator harmonicznych RPM (5400/7200/10000/15000) i pisków Stall Beep
   - Fizyczny strażnik DSP (DSPAdversarialGuard / DspPhysicalGuard)
3. Decyzja modelu ONNX (AcousticHddSentinelCNN):
   - Wariant BEZ DSP (surowy Softmax modelu neuronowego)
   - Wariant PO UŻYCIU DSP (hybrydowy potok decyzyjny aplikacji)
4. Dwa poziomy ewaluacji:
   - Poziom A: Okienkowy (Window-level, 3.0s okna, krok 3.0s, N=2539 wycinków)
   - Poziom B: Sesyjny (Session-level, streaming adaptive listening z oknami co 1.5s,
               dokładnie tak jak w Android MainViewModel.java z Early Stopping i konsensusem)
5. Generowanie pełnych macierzy pomyłek (Confusion Matrix 4x4), raportów klasyfikacji,
   wskaźników wykrywalności usterek (Fault Detection Rate) oraz wykresów graficznych.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import soundfile as sf
import librosa
import onnxruntime as ort
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from ml.hdd_audio_encoder import (
    split_files,
    slice_waveform,
    encode_audio_chunk,
    TARGET_SR,
    CHUNK_DURATION,
    INPUT_DIR
)
from ml.rpm_harmonics_analyzer import analyze_frequency_peaks
from ml.dsp_adversarial_guard import DSPAdversarialGuard

CLASSES = ["beeping", "clicking", "normal", "scratching"]
CHUNK_SAMPLES = int(CHUNK_DURATION * TARGET_SR)  # 66150 próbek


class FullPipelineSimulator:
    """
    Symulator zachowania aplikacji mobilnej Android Acoustic HDD Sentinel.
    """
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.guard = DSPAdversarialGuard()
        self.input_name = self.session.get_inputs()[0].name
        print(f"[Init] Załadowano model ONNX: {model_path.name}")

    def process_and_classify_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """
        Przetwarza pojedynczy 3-sekundowy wycinek fali dźwiękowej (66 150 próbek)
        dokładnie tak, jak robi to aplikacja mobilna Android (AudioPreprocessor + Model + DSP Guard).
        """
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        # 1. Enkoder DSP -> Tensor [3, 128, 259]
        tensor = encode_audio_chunk(chunk, sr=TARGET_SR, tensor_format="pytorch")
        tensor_in = np.expand_dims(tensor, axis=0).astype(np.float32)

        # 2. Surowa inferencja modelu neuronowego ONNX
        raw_probs = self.session.run(None, {self.input_name: tensor_in})[0][0].astype(np.float32)
        raw_pred_idx = int(np.argmax(raw_probs))
        raw_pred_class = CLASSES[raw_pred_idx]
        raw_fault_prob = float(raw_probs[0] + raw_probs[1] + raw_probs[3])
        raw_is_fault = (raw_fault_prob >= 0.70)

        # 3. Potok DSP aplikacji (dokładnie jak w AudioPreprocessor, RpmHarmonicsAnalyzer i DspPhysicalGuard):
        if rms < 0.002:
            # Bramka ciszy cyfrowej ADR-001 (RMS < 0.002)
            dsp_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            dsp_pred_class = "normal"
            dsp_is_fault = False
            rpm_res = {"has_rotation": False, "detected_rpm": None, "has_stall_beep": False}
            guard_reason = "ADR-001 Silence Gate (RMS < 0.002)"
        else:
            # Analiza harmonicznych obrotowych RPM oraz tonów zablokowanego stojana (Stall Beep)
            rpm_res = analyze_frequency_peaks(chunk, sr=TARGET_SR)

            # Fizyczny strażnik DSP (weta szumu wentylatorów, tła miejskiego, przesterowań, wzmocnienie pisków)
            san_probs, guarded, guard_reason = self.guard.protect_and_sanitize_predictions(
                raw_probs, tensor, waveform=chunk, rpm_result=rpm_res
            )

            total_p = np.sum(san_probs)
            if total_p > 0:
                dsp_probs = san_probs / total_p
            else:
                dsp_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

            dsp_fault_prob = float(dsp_probs[0] + dsp_probs[1] + dsp_probs[3])
            dsp_is_fault = (dsp_fault_prob >= 0.70)

            if dsp_is_fault:
                fault_indices = [0, 1, 3]  # beeping, clicking, scratching
                best_fault_idx = fault_indices[int(np.argmax([dsp_probs[i] for i in fault_indices]))]
                dsp_pred_class = CLASSES[best_fault_idx]
            else:
                dsp_pred_class = "normal"

        return {
            "rms": rms,
            "raw_probs": raw_probs,
            "raw_pred_class": raw_pred_class,
            "raw_is_fault": raw_is_fault,
            "raw_fault_prob": raw_fault_prob,
            "dsp_probs": dsp_probs,
            "dsp_pred_class": dsp_pred_class,
            "dsp_is_fault": dsp_is_fault,
            "dsp_fault_prob": float(dsp_probs[0] + dsp_probs[1] + dsp_probs[3]),
            "rpm_res": rpm_res,
            "guard_reason": guard_reason
        }

    def simulate_session_recording(self, waveform: np.ndarray, max_windows: int = 5) -> Dict[str, Any]:
        """
        Symuluje pełną sesję nasłuchu mikrofonowego w aplikacji Android (MainViewModel.java).
        - Próbkowanie z oknem 3.0s i krokiem 1.5s (streaming sliding window)
        - Obsługa do max_windows okien (standardowo do 9.0s nasłuchu)
        - Logika Early Stopping (awaria w >= 2 oknach, lub rotacja RPM >= 80%, lub konsensus 3 okien)
        - Agregacja końcowa i werdykt sesyjny
        """
        step_samples = int(1.5 * TARGET_SR)  # 33075 próbek
        total_samples = len(waveform)

        window_results = []
        fault_windows_count = 0
        detected_rpm = None
        best_anomaly_res = None
        stopped_early = False
        early_stop_reason = ""

        # Wyliczamy dostępne okna streamingowe
        n_windows = 0
        offset = 0
        while offset + CHUNK_SAMPLES <= total_samples and n_windows < max_windows:
            window_chunk = waveform[offset:offset + CHUNK_SAMPLES]
            eval_res = self.process_and_classify_chunk(window_chunk)
            window_results.append(eval_res)

            # Detekcja RPM
            if eval_res["rpm_res"].get("has_rotation", False) and eval_res["rpm_res"].get("detected_rpm"):
                detected_rpm = eval_res["rpm_res"]["detected_rpm"]

            # Obsługa awarii
            if eval_res["dsp_is_fault"]:
                fault_windows_count += 1
                if best_anomaly_res is None or eval_res["dsp_probs"][CLASSES.index(eval_res["dsp_pred_class"])] > best_anomaly_res["dsp_probs"][CLASSES.index(best_anomaly_res["dsp_pred_class"])]:
                    best_anomaly_res = eval_res

            # Logika Early Stopping z Android MainViewModel.java:
            # 1. Zatrzymanie przy awarii po potwierdzeniu w co najmniej 2 oknach
            if fault_windows_count >= 2:
                stopped_early = True
                early_stop_reason = f"Early Stopping: Potwierdzono awarię w >= 2 oknach (okno {n_windows})"
                break

            # 2. Potwierdzony dysk sprawny z wykrytą rotacją RPM (>= 80% pewności)
            confidence = eval_res["dsp_probs"][2]
            if not eval_res["dsp_is_fault"] and eval_res["rpm_res"].get("has_rotation", False) and confidence >= 0.80:
                stopped_early = True
                early_stop_reason = f"Early Stopping: Dysk sprawny z potwierdzoną rotacją RPM ({confidence*100:.1f}%) w oknie {n_windows}"
                break

            # 3. Konsensus 3 kolejnych okien sprawnych (>= 70%)
            if n_windows >= 2 and not eval_res["dsp_is_fault"]:
                prev1 = window_results[n_windows - 1]
                prev2 = window_results[n_windows - 2]
                if not prev1["dsp_is_fault"] and not prev2["dsp_is_fault"] and confidence >= 0.70:
                    stopped_early = True
                    early_stop_reason = f"Early Stopping: Konsensus 3 okien sprawnych w oknie {n_windows}"
                    break

            offset += step_samples
            n_windows += 1

        # Jeśli plik był krótszy niż 3.0s, oceniamy pojedyncze okno z dopełnieniem zerami
        if not window_results:
            padded = np.pad(waveform, (0, max(0, CHUNK_SAMPLES - total_samples)))[:CHUNK_SAMPLES]
            eval_res = self.process_and_classify_chunk(padded)
            window_results.append(eval_res)
            if eval_res["dsp_is_fault"]:
                fault_windows_count += 1
                best_anomaly_res = eval_res

        # =========================================================================
        # Finalna agregacja diagnostyczna (dokładnie jak finalizeDiagnostic w Android)
        # =========================================================================
        num_evaluated = len(window_results)
        avg_dsp_probs = np.mean([w["dsp_probs"] for w in window_results], axis=0)
        avg_raw_probs = np.mean([w["raw_probs"] for w in window_results], axis=0)

        # Reguła aplikacji Android: Werdykt "AWARIA" wyłącznie po potwierdzeniu w min. 2 oknach
        # (lub w jedynym oknie, jeśli nagranie było bardzo krótkie < 4.5s i miało 1 okno)
        if num_evaluated == 1:
            is_final_dsp_fault = (fault_windows_count >= 1 and best_anomaly_res is not None)
        else:
            is_final_dsp_fault = (fault_windows_count >= 2 and best_anomaly_res is not None)

        if is_final_dsp_fault:
            final_dsp_class = best_anomaly_res["dsp_pred_class"]
            final_dsp_conf = float(best_anomaly_res["dsp_probs"][CLASSES.index(final_dsp_class)])
        else:
            final_dsp_class = "normal"
            final_dsp_conf = float(avg_dsp_probs[2])

        # Surowy model bez logiki DSP i bez reguły 2 okien (czysta średnia Softmax):
        raw_final_idx = int(np.argmax(avg_raw_probs))
        final_raw_class = CLASSES[raw_final_idx]
        final_raw_conf = float(avg_raw_probs[raw_final_idx])

        return {
            "num_windows": num_evaluated,
            "stopped_early": stopped_early,
            "early_stop_reason": early_stop_reason,
            "detected_rpm": detected_rpm,
            "fault_windows_count": fault_windows_count,
            "final_dsp_class": final_dsp_class,
            "final_dsp_conf": final_dsp_conf,
            "is_final_dsp_fault": is_final_dsp_fault,
            "final_raw_class": final_raw_class,
            "final_raw_conf": final_raw_conf,
            "avg_dsp_probs": avg_dsp_probs,
            "avg_raw_probs": avg_raw_probs
        }


def plot_side_by_side_confusion_matrices(
    cm_raw: np.ndarray,
    cm_dsp: np.ndarray,
    classes: List[str],
    title_suffix: str,
    output_path: Path
):
    """
    Rysuje elegancką podwójną macierz pomyłek: BEZ DSP (Raw Model) vs PO UŻYCIU DSP.
    Pokazuje surowe liczby próbek oraz procenty znormalizowane w wierszach (Recall).
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, cm, title, cmap in zip(
        axes,
        [cm_raw, cm_dsp],
        [f"BEZ DSP (Surowy Model CNN)\n{title_suffix}", f"PO UŻYCIU DSP (Potok Aplikacji)\n{title_suffix}"],
        ["Blues", "Greens"]
    ):
        # Normalizacja procentowa per klasa rzeczywista
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.zeros_like(cm, dtype=np.float32)
        for i in range(len(row_sums)):
            if row_sums[i] > 0:
                cm_norm[i] = cm[i] / row_sums[i]

        im = ax.imshow(cm_norm, interpolation="nearest", cmap=cmap, vmin=0.0, vmax=1.0)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

        # Etykiety osi
        tick_marks = np.arange(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(classes, fontsize=11, fontweight="medium")
        ax.set_yticklabels(classes, fontsize=11, fontweight="medium")
        ax.set_xlabel("Klasa Przewidziana (Predicted Class)", fontsize=11, labelpad=8)
        ax.set_ylabel("Klasa Rzeczywista (True Class)", fontsize=11, labelpad=8)

        # Adnotacje tekstowe w komórkach (Liczba + Procent)
        thresh = 0.5
        for i in range(len(classes)):
            for j in range(len(classes)):
                count = cm[i, j]
                pct = cm_norm[i, j] * 100.0
                text = f"{count}\n({pct:.1f}%)"
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=10, fontweight="bold")

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[Wykres] Zapisano macierz pomyłek do: {output_path}")


def run_full_pipeline_simulation():
    print("=" * 85)
    print("SYMULACJA PEŁNEGO POTOKU APLIKACJI DLA CAŁEGO DATASETU (388 PLIKÓW WAV)")
    print("Potok: Mikrofon -> Resampling -> DSP Enkoder -> ONNX CNN -> RPM Guard -> Werdykt")
    print("=" * 85)

    model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model.onnx"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model_int8.onnx"
    simulator = FullPipelineSimulator(model_path)

    # Przygotowanie podziału plików źródłowych ADR-001 (Zero Data Leakage)
    print("\n[Krok 1] Podział bazy 388 nagrań źródłowych na splity...")
    splits = split_files(INPUT_DIR)
    all_files = []
    file_to_split = {}
    for s_name in ["test", "validate", "train"]:
        for f_path, c_name in splits[s_name]:
            all_files.append((f_path, c_name))
            file_to_split[str(f_path)] = s_name

    print(f"  -> Całkowita liczba nagrań WAV: {len(all_files)}")
    print(f"     * Train:    {len(splits['train'])} plików")
    print(f"     * Validate: {len(splits['validate'])} plików")
    print(f"     * Test:     {len(splits['test'])} plików")

    # =========================================================================
    # STRUKTURY DANYCH DLA WYNIKÓW
    # =========================================================================
    # Poziom A: Okienkowy (Wszystkie 3.0s wycinki, krok 3.0s)
    window_y_true = []
    window_y_raw = []
    window_y_dsp = []
    window_splits = []

    # Poziom B: Sesyjny (Każde nagranie jako pełna sesja mikrofonowa z adaptive listening)
    file_y_true = []
    file_y_raw = []
    file_y_dsp = []
    file_splits = []
    file_early_stops = []
    file_detected_rpms = []

    start_time = time.time()
    processed_count = 0

    print("\n[Krok 2] Uruchamianie symulacji mikrofonu dla każdego nagrania audio...")

    for f_path, true_cls in all_files:
        split_name = file_to_split[str(f_path)]
        try:
            waveform, sr = sf.read(str(f_path))
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            if sr != TARGET_SR:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=TARGET_SR)
            waveform = waveform.astype(np.float32)

            # -----------------------------------------------------------------
            # A. Symulacja poziomu sesyjnego aplikacji (Streaming Adaptive Listening)
            # -----------------------------------------------------------------
            session_res = simulator.simulate_session_recording(waveform, max_windows=5)
            file_y_true.append(true_cls)
            file_y_raw.append(session_res["final_raw_class"])
            file_y_dsp.append(session_res["final_dsp_class"])
            file_splits.append(split_name)
            file_early_stops.append(session_res["stopped_early"])
            file_detected_rpms.append(session_res["detected_rpm"])

            # -----------------------------------------------------------------
            # B. Symulacja poziomu pojedynczych okien (Wszystkie 3-sekundowe wycinki)
            # -----------------------------------------------------------------
            chunks = slice_waveform(waveform, sr=TARGET_SR, chunk_dur=CHUNK_DURATION, overlap_dur=0.0, is_train=False)
            for chunk in chunks:
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                elif len(chunk) > CHUNK_SAMPLES:
                    chunk = chunk[:CHUNK_SAMPLES]

                chunk_res = simulator.process_and_classify_chunk(chunk)
                window_y_true.append(true_cls)
                window_y_raw.append(chunk_res["raw_pred_class"])
                window_y_dsp.append(chunk_res["dsp_pred_class"])
                window_splits.append(split_name)

            processed_count += 1
            if processed_count % 50 == 0 or processed_count == len(all_files):
                print(f"  -> Przetworzono {processed_count}/{len(all_files)} plików audio...")

        except Exception as e:
            print(f"  [Błąd pliku] {f_path.name}: {e}")

    total_time = time.time() - start_time
    print(f"\n[Gotowe] Przetworzono {len(all_files)} plików i {len(window_y_true)} wycinków w {total_time:.2f} s ({total_time/len(all_files)*1000:.1f} ms/plik)")

    # =========================================================================
    # OBLICZENIE MACIERZY POMYŁEK I METRYK
    # =========================================================================
    # 1. Macierze pomyłek na poziomie OKIENKOWYM (Window-level)
    cm_window_raw = confusion_matrix(window_y_true, window_y_raw, labels=CLASSES)
    cm_window_dsp = confusion_matrix(window_y_true, window_y_dsp, labels=CLASSES)

    # 2. Macierze pomyłek na poziomie SESYJNYM (File-level / Session-level)
    cm_file_raw = confusion_matrix(file_y_true, file_y_raw, labels=CLASSES)
    cm_file_dsp = confusion_matrix(file_y_true, file_y_dsp, labels=CLASSES)

    # Raporty klasyfikacji per-klasa i makro
    win_raw_p, win_raw_r, win_raw_f1, _ = precision_recall_fscore_support(window_y_true, window_y_raw, labels=CLASSES, average="macro", zero_division=0)
    win_dsp_p, win_dsp_r, win_dsp_f1, _ = precision_recall_fscore_support(window_y_true, window_y_dsp, labels=CLASSES, average="macro", zero_division=0)
    win_raw_acc = float(np.mean(np.array(window_y_raw) == np.array(window_y_true)))
    win_dsp_acc = float(np.mean(np.array(window_y_dsp) == np.array(window_y_true)))

    file_raw_p, file_raw_r, file_raw_f1, _ = precision_recall_fscore_support(file_y_true, file_y_raw, labels=CLASSES, average="macro", zero_division=0)
    file_dsp_p, file_dsp_r, file_dsp_f1, _ = precision_recall_fscore_support(file_y_true, file_y_dsp, labels=CLASSES, average="macro", zero_division=0)
    file_raw_acc = float(np.mean(np.array(file_y_raw) == np.array(file_y_true)))
    file_dsp_acc = float(np.mean(np.array(file_y_dsp) == np.array(file_y_true)))

    # Metryki per klasa
    win_raw_pc = precision_recall_fscore_support(window_y_true, window_y_raw, labels=CLASSES, average=None, zero_division=0)
    win_dsp_pc = precision_recall_fscore_support(window_y_true, window_y_dsp, labels=CLASSES, average=None, zero_division=0)
    file_raw_pc = precision_recall_fscore_support(file_y_true, file_y_raw, labels=CLASSES, average=None, zero_division=0)
    file_dsp_pc = precision_recall_fscore_support(file_y_true, file_y_dsp, labels=CLASSES, average=None, zero_division=0)

    # =========================================================================
    # PREZENTACJA WYNIKÓW W TERMINALU
    # =========================================================================
    print("\n" + "=" * 85)
    print("WYNIKI 1: POZIOM SESYJNY APLIKACJI (SYMULACJA MIKROFONU NA 388 NAGRANIACH)")
    print("=" * 85)
    print(f"Całkowita liczba nagrań: {len(file_y_true)}")
    print(f"Liczba wczesnych zatrzymań (Early Stopping): {sum(file_early_stops)}/{len(file_y_true)} ({sum(file_early_stops)/len(file_y_true)*100:.1f}%)")
    print(f"Dokładność ogólna (Accuracy):")
    print(f"  * BEZ DSP (Surowy Model CNN):  {file_raw_acc*100:.2f}%")
    print(f"  * PO UŻYCIU DSP (Aplikacja):    {file_dsp_acc*100:.2f}% (Poprawa: +{(file_dsp_acc-file_raw_acc)*100:+.2f} p.p.)")
    print(f"Macro F1-Score:")
    print(f"  * BEZ DSP: {file_raw_f1*100:.2f}%")
    print(f"  * PO UŻYCIU DSP: {file_dsp_f1*100:.2f}%\n")

    print("MACIERZ POMYŁEK NA POZIOMIE SESYJNYM (388 PLIKÓW) - PO UŻYCIU DSP:")
    print(f"{'Klasa Rzeczywista':<18} | {'beeping':>8} {'clicking':>9} {'normal':>8} {'scratching':>11} | {'Recall':>8}")
    print("-" * 75)
    for i, cls in enumerate(CLASSES):
        row = cm_file_dsp[i]
        rec = (row[i] / sum(row) * 100) if sum(row) > 0 else 0.0
        print(f"{cls:<18} | {row[0]:>8} {row[1]:>9} {row[2]:>8} {row[3]:>11} | {rec:>7.1f}%")

    print("\n" + "-" * 85)
    print("SZCZEGÓŁOWE METRYKI PER-KLASA NA POZIOMIE SESYJNYM (Z DSP):")
    print(f"{'Klasa':<14} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Liczba (Support)':>15}")
    print("-" * 75)
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<14} | {file_dsp_pc[0][i]*100:>9.2f}% | {file_dsp_pc[1][i]*100:>9.2f}% | {file_dsp_pc[2][i]*100:>9.2f}% | {file_dsp_pc[3][i]:>15}")

    print("\n" + "=" * 85)
    print(f"WYNIKI 2: POZIOM OKIENKOWY (WSZYSTKIE {len(window_y_true)} WYCINKÓW 3.0s)")
    print("=" * 85)
    print(f"Dokładność ogólna (Accuracy):")
    print(f"  * BEZ DSP: {win_raw_acc*100:.2f}%")
    print(f"  * PO UŻYCIU DSP: {win_dsp_acc*100:.2f}% (Poprawa: +{(win_dsp_acc-win_raw_acc)*100:+.2f} p.p.)")
    print(f"Macro F1-Score:")
    print(f"  * BEZ DSP: {win_raw_f1*100:.2f}%")
    print(f"  * PO UŻYCIU DSP: {win_dsp_f1*100:.2f}%\n")

    print("MACIERZ POMYŁEK NA POZIOMIE OKIENKOWYM (2539 OKIEN) - PO UŻYCIU DSP:")
    print(f"{'Klasa Rzeczywista':<18} | {'beeping':>8} {'clicking':>9} {'normal':>8} {'scratching':>11} | {'Recall':>8}")
    print("-" * 75)
    for i, cls in enumerate(CLASSES):
        row = cm_window_dsp[i]
        rec = (row[i] / sum(row) * 100) if sum(row) > 0 else 0.0
        print(f"{cls:<18} | {row[0]:>8} {row[1]:>9} {row[2]:>8} {row[3]:>11} | {rec:>7.1f}%")

    print("\n" + "-" * 85)
    print("SZCZEGÓŁOWE METRYKI PER-KLASA NA POZIOMIE OKIENKOWYM (Z DSP):")
    print(f"{'Klasa':<14} | {'Precision':>10} | {'Recall':>10} | {'F1-Score':>10} | {'Liczba (Support)':>15}")
    print("-" * 75)
    for i, cls in enumerate(CLASSES):
        print(f"{cls:<14} | {win_dsp_pc[0][i]*100:>9.2f}% | {win_dsp_pc[1][i]*100:>9.2f}% | {win_dsp_pc[2][i]*100:>9.2f}% | {win_dsp_pc[3][i]:>15}")

    # =========================================================================
    # ROZBICIE NA SPLITY (TEST, VALIDATE, TRAIN)
    # =========================================================================
    print("\n" + "=" * 85)
    print("PORÓWNANIE WSKAŹNIKÓW W ROZBICIU NA SPLITY ADR-001 (Zero Data Leakage):")
    print("=" * 85)
    split_metrics = {}
    for s_name in ["test", "validate", "train"]:
        # Filtrujemy dane
        idx_f = [i for i, s in enumerate(file_splits) if s == s_name]
        s_true = [file_y_true[i] for i in idx_f]
        s_raw = [file_y_raw[i] for i in idx_f]
        s_dsp = [file_y_dsp[i] for i in idx_f]

        s_raw_acc = float(np.mean(np.array(s_raw) == np.array(s_true)))
        s_dsp_acc = float(np.mean(np.array(s_dsp) == np.array(s_true)))
        _, _, s_dsp_f1, _ = precision_recall_fscore_support(s_true, s_dsp, labels=CLASSES, average="macro", zero_division=0)
        s_cm_dsp = confusion_matrix(s_true, s_dsp, labels=CLASSES)

        split_metrics[s_name] = {
            "num_files": len(s_true),
            "raw_acc": s_raw_acc,
            "dsp_acc": s_dsp_acc,
            "dsp_f1": s_dsp_f1,
            "cm_dsp": s_cm_dsp.tolist()
        }

        print(f"Split {s_name.upper():<8} ({len(s_true):>3} plików):")
        print(f"  * Accuracy: Raw={s_raw_acc*100:.1f}% -> DSP={s_dsp_acc*100:.1f}% | Macro F1={s_dsp_f1*100:.1f}%")

    # =========================================================================
    # RYSOWANIE I ZAPIS WYKRESÓW MACIERZY POMYŁEK
    # =========================================================================
    out_dir = PROJECT_ROOT / "training_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_win_path = out_dir / "confusion_matrix_full_dataset_windows.png"
    plot_file_path = out_dir / "confusion_matrix_full_dataset_files.png"

    plot_side_by_side_confusion_matrices(
        cm_window_raw, cm_window_dsp, CLASSES,
        f"(Cały Dataset: {len(window_y_true)} wycinków 3.0s)",
        plot_win_path
    )

    plot_side_by_side_confusion_matrices(
        cm_file_raw, cm_file_dsp, CLASSES,
        f"(Sesje Mikrofonowe: {len(file_y_true)} nagrań WAV)",
        plot_file_path
    )

    # Zapis danych do JSON
    json_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_files": len(file_y_true),
        "total_windows": len(window_y_true),
        "classes": CLASSES,
        "session_level": {
            "accuracy_raw": file_raw_acc,
            "accuracy_dsp": file_dsp_acc,
            "macro_f1_raw": file_raw_f1,
            "macro_f1_dsp": file_dsp_f1,
            "confusion_matrix_raw": cm_file_raw.tolist(),
            "confusion_matrix_dsp": cm_file_dsp.tolist(),
            "early_stopping_count": sum(file_early_stops),
            "per_class_dsp": {
                cls: {
                    "precision": float(file_dsp_pc[0][i]),
                    "recall": float(file_dsp_pc[1][i]),
                    "f1": float(file_dsp_pc[2][i]),
                    "support": int(file_dsp_pc[3][i])
                } for i, cls in enumerate(CLASSES)
            }
        },
        "window_level": {
            "accuracy_raw": win_raw_acc,
            "accuracy_dsp": win_dsp_acc,
            "macro_f1_raw": win_raw_f1,
            "macro_f1_dsp": win_dsp_f1,
            "confusion_matrix_raw": cm_window_raw.tolist(),
            "confusion_matrix_dsp": cm_window_dsp.tolist(),
            "per_class_dsp": {
                cls: {
                    "precision": float(win_dsp_pc[0][i]),
                    "recall": float(win_dsp_pc[1][i]),
                    "f1": float(win_dsp_pc[2][i]),
                    "support": int(win_dsp_pc[3][i])
                } for i, cls in enumerate(CLASSES)
            }
        },
        "splits": split_metrics
    }

    json_path = out_dir / "full_pipeline_simulation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    print(f"\n[Zapis] Zapisano podsumowanie JSON do: {json_path}")
    print("=" * 85)


if __name__ == "__main__":
    run_full_pipeline_simulation()
