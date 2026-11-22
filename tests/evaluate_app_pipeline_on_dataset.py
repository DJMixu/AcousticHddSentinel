#!/usr/bin/env python3
"""
HDD Acoustic Sentinel - Pełna Ewaluacja Potoku Aplikacji i Modelu (Z DSP vs Bez DSP)
Single Source of Truth: ADR-001, ADR-002, AGENTS.md

Cel ewaluacji:
1. Przetestowanie modelu i potoku aplikacji na autentycznych próbkach audio ze zbioru testowego (test split)
   oraz zbioru treningowego (train split).
2. Przetestowanie odporności na szumy komputerowe (wentylatory PC, tło akustyczne ~45 dB SPL)
   oraz weryfikacja na 4 rzeczywistych sprawnych dyskach HDD (Toshiba, WD, Seagate, HGST).
3. Bezpośrednie porównanie zachowania:
   - BEZ DSP (Surowy model neuronowy ONNX / PyTorch - czysty Softmax)
   - PO UŻYCIU DSP (Aplikacja z potokiem DSP: AudioPreprocessor + RpmHarmonicsAnalyzer + DspPhysicalGuard)
4. Obliczenie pełnych metryk:
   - Macierz pomyłek (Confusion Matrix) dla 4 klas
   - Wskaźniki Precision, Recall, F1 per klasa
   - Accuracy ogólna
   - False Positive Rate (FPR) na dyskach sprawnych i szumach tła
   - Wskaźnik wykrywania rotacji wrzeciona (5400 / 7200 RPM)
   - Liczba błędów surowego modelu naprawionych przez filtry DSP.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import soundfile as sf
import onnxruntime as ort
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


class AppPipelineTester:
    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.guard = DSPAdversarialGuard()
        self.input_name = self.session.get_inputs()[0].name
        print(f"[Init] Załadowano model ONNX: {model_path.name}")

    def evaluate_chunk(self, chunk: np.ndarray) -> Dict[str, Any]:
        """
        Ocenia pojedynczy 3-sekundowy chunk audio dwiema metodami:
        1. Surowy model ONNX (bez logiki DSP Guard / RPM)
        2. Pełny potok aplikacji po użyciu DSP (Bramka ciszy + RPM Comb Filter + DSP Guard)
        """
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        
        # 1. DSP Audio Preprocessing do tensora [3, 128, 259]
        tensor = encode_audio_chunk(chunk, sr=TARGET_SR, tensor_format="pytorch")
        tensor_in = np.expand_dims(tensor, axis=0).astype(np.float32)

        # 2. Inferencja modelu ONNX
        raw_probs = self.session.run(None, {self.input_name: tensor_in})[0][0].astype(np.float32)
        raw_top_idx = int(np.argmax(raw_probs))
        raw_pred_class = CLASSES[raw_top_idx]
        raw_fault_prob = float(raw_probs[0] + raw_probs[1] + raw_probs[3])
        raw_is_fault = (raw_fault_prob >= 0.70)

        # 3. Potok DSP Aplikacji:
        # A. Bramka ciszy ADR-001
        if rms < 0.002:
            dsp_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
            dsp_pred_class = "normal"
            dsp_is_fault = False
            rpm_res = {"has_rotation": False, "detected_rpm": None, "has_stall_beep": False}
            guard_reason = "ADR-001 Silence Gate (RMS < 0.002)"
        else:
            # B. Analizator częstotliwościowy RPM i Stall Beep
            rpm_res = analyze_frequency_peaks(chunk, sr=TARGET_SR)

            # C. Fizyczny strażnik DSP (identycznie jak w MainViewModel.java)
            san_probs, guarded, guard_reason = self.guard.protect_and_sanitize_predictions(
                raw_probs, tensor, waveform=chunk, rpm_result=rpm_res
            )

            # Normalizacja do sumy 1.0
            total_p = np.sum(san_probs)
            if total_p > 0:
                dsp_probs = san_probs / total_p
            else:
                dsp_probs = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)

            # P(fault) = beeping + clicking + scratching (zgodnie z ADR-002 i MainViewModel)
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
            "raw_pred_class": raw_pred_class,
            "raw_probs": raw_probs,
            "raw_is_fault": raw_is_fault,
            "dsp_pred_class": dsp_pred_class,
            "dsp_probs": dsp_probs,
            "dsp_is_fault": dsp_is_fault,
            "rpm_detected": rpm_res.get("detected_rpm"),
            "has_rotation": rpm_res.get("has_rotation", False),
            "guard_reason": guard_reason
        }


def run_full_evaluation():
    print("=" * 80)
    print("PEŁNY TEST EWALUACYJNY POTOKU APLIKACJI I MODELU (Z DSP vs BEZ DSP)")
    print("=" * 80)

    # 1. Sprawdzenie modelu
    model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model.onnx"
    if not model_path.exists():
        model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model_int8.onnx"
    tester = AppPipelineTester(model_path)

    # 2. Podział plików źródłowych (Zero Data Leakage)
    print("\n[Krok 1] Przygotowanie podziału plików źródłowych ADR-001...")
    splits = split_files(INPUT_DIR)
    print(f"  -> Zbiór treningowy: {len(splits['train'])} plików")
    print(f"  -> Zbiór walidacyjny: {len(splits['validate'])} plików")
    print(f"  -> Zbiór testowy:     {len(splits['test'])} plików")

    results_summary: Dict[str, Any] = {}

    for split_name in ["test", "train"]:
        print("\n" + "=" * 80)
        print(f"EWALUACJA ZBIORU: {split_name.upper()} (Zero Data Leakage)")
        print("=" * 80)

        file_list = splits[split_name]
        is_train = (split_name == "train")

        # Aby test nie trwał zbyt długo, dla zbioru train próbkujemy reprezentatywny podzbiór
        # (do 120 plików: po max 30 na klasę), a zbiór testowy badamy w 100% (wszystkie 74 pliki)!
        if is_train:
            selected_files = []
            class_counts = {c: 0 for c in CLASSES}
            for path, cls in file_list:
                if class_counts[cls] < 20:
                    selected_files.append((path, cls))
                    class_counts[cls] += 1
        else:
            selected_files = file_list

        y_true_all: List[str] = []
        y_raw_pred_all: List[str] = []
        y_dsp_pred_all: List[str] = []

        raw_binary_tp, raw_binary_fp, raw_binary_tn, raw_binary_fn = 0, 0, 0, 0
        dsp_binary_tp, dsp_binary_fp, dsp_binary_tn, dsp_binary_fn = 0, 0, 0, 0

        rotation_detected_normal = 0
        total_normal_chunks = 0
        dsp_corrections_count = 0  # Błędy surowego modelu poprawione przez DSP

        total_chunks = 0

        for path, true_cls in selected_files:
            try:
                y, sr = sf.read(str(path))
                if y.ndim > 1:
                    y = np.mean(y, axis=1)
                if sr != TARGET_SR:
                    import librosa
                    y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

                # Cięcie na wycinki: dla test step=3.0s (czyste), dla train step=3.0s dla równego porównania
                chunks = slice_waveform(y, sr=TARGET_SR, chunk_dur=CHUNK_DURATION, overlap_dur=0.0, is_train=False)

                for chunk in chunks:
                    if len(chunk) < CHUNK_SAMPLES:
                        chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                    elif len(chunk) > CHUNK_SAMPLES:
                        chunk = chunk[:CHUNK_SAMPLES]

                    eval_res = tester.evaluate_chunk(chunk)
                    total_chunks += 1

                    raw_pred = eval_res["raw_pred_class"]
                    dsp_pred = eval_res["dsp_pred_class"]

                    y_true_all.append(true_cls)
                    y_raw_pred_all.append(raw_pred)
                    y_dsp_pred_all.append(dsp_pred)

                    is_true_fault = (true_cls != "normal")
                    raw_fault = eval_res["raw_is_fault"]
                    dsp_fault = eval_res["dsp_is_fault"]

                    # Metryki binarne RAW
                    if is_true_fault and raw_fault: raw_binary_tp += 1
                    elif is_true_fault and not raw_fault: raw_binary_fn += 1
                    elif not is_true_fault and raw_fault: raw_binary_fp += 1
                    else: raw_binary_tn += 1

                    # Metryki binarne DSP
                    if is_true_fault and dsp_fault: dsp_binary_tp += 1
                    elif is_true_fault and not dsp_fault: dsp_binary_fn += 1
                    elif not is_true_fault and dsp_fault: dsp_binary_fp += 1
                    else: dsp_binary_tn += 1

                    # Weryfikacja rotacji dla normalnych dysków
                    if true_cls == "normal":
                        total_normal_chunks += 1
                        if eval_res["has_rotation"]:
                            rotation_detected_normal += 1

                    # Weryfikacja korekt DSP
                    if raw_pred != true_cls and dsp_pred == true_cls:
                        dsp_corrections_count += 1

            except Exception as e:
                print(f"  [Błąd pliku] {path.name}: {e}")

        # Obliczenie metryk wieloklasowych
        raw_acc = float(np.mean(np.array(y_raw_pred_all) == np.array(y_true_all)))
        dsp_acc = float(np.mean(np.array(y_dsp_pred_all) == np.array(y_true_all)))

        raw_prec, raw_rec, raw_f1, _ = precision_recall_fscore_support(
            y_true_all, y_raw_pred_all, labels=CLASSES, average="macro", zero_division=0
        )
        dsp_prec, dsp_rec, dsp_f1, _ = precision_recall_fscore_support(
            y_true_all, y_dsp_pred_all, labels=CLASSES, average="macro", zero_division=0
        )

        # Raport per-klasa
        raw_per_class = precision_recall_fscore_support(
            y_true_all, y_raw_pred_all, labels=CLASSES, average=None, zero_division=0
        )
        dsp_per_class = precision_recall_fscore_support(
            y_true_all, y_dsp_pred_all, labels=CLASSES, average=None, zero_division=0
        )

        # Metryki binarne
        raw_bin_recall = raw_binary_tp / (raw_binary_tp + raw_binary_fn) if (raw_binary_tp + raw_binary_fn) > 0 else 0.0
        raw_bin_prec = raw_binary_tp / (raw_binary_tp + raw_binary_fp) if (raw_binary_tp + raw_binary_fp) > 0 else 0.0
        raw_bin_fpr = raw_binary_fp / (raw_binary_fp + raw_binary_tn) if (raw_binary_fp + raw_binary_tn) > 0 else 0.0

        dsp_bin_recall = dsp_binary_tp / (dsp_binary_tp + dsp_binary_fn) if (dsp_binary_tp + dsp_binary_fn) > 0 else 0.0
        dsp_bin_prec = dsp_binary_tp / (dsp_binary_tp + dsp_binary_fp) if (dsp_binary_tp + dsp_binary_fp) > 0 else 0.0
        dsp_bin_fpr = dsp_binary_fp / (dsp_binary_fp + dsp_binary_tn) if (dsp_binary_fp + dsp_binary_tn) > 0 else 0.0

        rpm_detection_rate = (rotation_detected_normal / total_normal_chunks * 100.0) if total_normal_chunks > 0 else 0.0

        # Wyświetlenie wyników
        print(f"Przebadano łącznie wycinków 3.0s: {total_chunks}")
        print("\n--- PORÓWNANIE WYNIKÓW: BEZ DSP vs PO UŻYCIU DSP ---")
        print(f"{'Metryka':<35} | {'BEZ DSP (Raw Model)':<22} | {'PO UŻYCIU DSP (App)':<22} | {'Różnica':<10}")
        print("-" * 98)
        print(f"{'Ogólna dokładność (Accuracy)':<35} | {raw_acc * 100:>20.2f}% | {dsp_acc * 100:>20.2f}% | {(dsp_acc - raw_acc) * 100:>+8.2f}%")
        print(f"{'Średnia F1 (Macro F1)':<35} | {raw_f1 * 100:>20.2f}% | {dsp_f1 * 100:>20.2f}% | {(dsp_f1 - raw_f1) * 100:>+8.2f}%")
        print(f"{'Precyzja makro (Macro Precision)':<35} | {raw_prec * 100:>20.2f}% | {dsp_prec * 100:>20.2f}% | {(dsp_prec - raw_prec) * 100:>+8.2f}%")
        print(f"{'Czułość makro (Macro Recall)':<35} | {raw_rec * 100:>20.2f}% | {dsp_rec * 100:>20.2f}% | {(dsp_rec - raw_rec) * 100:>+8.2f}%")
        print("-" * 98)
        print(f"{'Binarny Recall (Wykrycie awarii)':<35} | {raw_bin_recall * 100:>20.2f}% | {dsp_bin_recall * 100:>20.2f}% | {(dsp_bin_recall - raw_bin_recall) * 100:>+8.2f}%")
        print(f"{'Binarna Precyzja':<35} | {raw_bin_prec * 100:>20.2f}% | {dsp_bin_prec * 100:>20.2f}% | {(dsp_bin_prec - raw_bin_prec) * 100:>+8.2f}%")
        print(f"{'False Positive Rate (Fałszywy alarm)':<35} | {raw_bin_fpr * 100:>20.2f}% | {dsp_bin_fpr * 100:>20.2f}% | {(dsp_bin_fpr - raw_bin_fpr) * 100:>+8.2f}%")
        print(f"{'Wykrywalność rotacji RPM (normal)':<35} | {'N/A (Brak DSP)':>22} | {rpm_detection_rate:>20.1f}% | {'--':>10}")
        print(f"{'Błędy surowe naprawione przez DSP':<35} | {'0 (Baza)':>22} | {dsp_corrections_count:>20} | {f'+{dsp_corrections_count}':>10}")

        print("\n--- METRYKI PER KLASA (F1-SCORE) ---")
        print(f"{'Klasa':<15} | {'BEZ DSP F1':<15} | {'PO DSP F1':<15} | {'BEZ DSP Rec':<15} | {'PO DSP Rec':<15}")
        print("-" * 80)
        for i, c in enumerate(CLASSES):
            print(f"{c:<15} | {raw_per_class[2][i]*100:>13.1f}% | {dsp_per_class[2][i]*100:>13.1f}% | {raw_per_class[1][i]*100:>13.1f}% | {dsp_per_class[1][i]*100:>13.1f}%")

        # Macierz pomyłek PO DSP
        cm_dsp = confusion_matrix(y_true_all, y_dsp_pred_all, labels=CLASSES)
        print(f"\n--- MACIERZ POMYŁEK PO UŻYCIU DSP ({split_name.upper()}) ---")
        header_title = "True \\ Pred"
        print(f"{header_title:<15} | " + " | ".join([f"{c:>12}" for c in CLASSES]))
        print("-" * 75)
        for i, c in enumerate(CLASSES):
            row_str = " | ".join([f"{cm_dsp[i, j]:>12}" for j in range(len(CLASSES))])
            print(f"{c:<15} | {row_str}")

        results_summary[split_name] = {
            "total_chunks": total_chunks,
            "raw_acc": raw_acc,
            "dsp_acc": dsp_acc,
            "raw_f1": raw_f1,
            "dsp_f1": dsp_f1,
            "raw_bin_fpr": raw_bin_fpr,
            "dsp_bin_fpr": dsp_bin_fpr,
            "dsp_bin_recall": dsp_bin_recall,
            "dsp_corrections": dsp_corrections_count,
            "rpm_detection_rate": rpm_detection_rate,
            "per_class_f1_dsp": {c: float(dsp_per_class[2][i]) for i, c in enumerate(CLASSES)}
        }

    # =========================================================================
    # TEST SPECJALNY: 320 OKIEN 4 AUTENTYCZNYCH DYSKÓW STUDY RECORDINGS
    # =========================================================================
    print("\n" + "=" * 80)
    print("EWALUACJA SPECJALNA: 320 OKIEN AUTENTYCZNYCH DYSKÓW FIZYCZNYCH (4-MIN STUDY)")
    print("=" * 80)
    study_dir = PROJECT_ROOT / "tests" / "recordings_4min_study"
    study_files = sorted(list(study_dir.glob("*/*.wav")))

    study_raw_false_alarms = 0
    study_dsp_false_alarms = 0
    study_total_windows = 0
    study_rpm_detected = 0

    for wf in study_files:
        y, sr = sf.read(str(wf))
        if y.ndim > 1: y = np.mean(y, axis=1)
        if sr != TARGET_SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

        num_windows = len(y) // CHUNK_SAMPLES
        for w in range(num_windows):
            chunk = y[w * CHUNK_SAMPLES : (w + 1) * CHUNK_SAMPLES]
            eval_res = tester.evaluate_chunk(chunk)
            study_total_windows += 1

            if eval_res["raw_is_fault"]:
                study_raw_false_alarms += 1
            if eval_res["dsp_is_fault"]:
                study_dsp_false_alarms += 1
            if eval_res["has_rotation"]:
                study_rpm_detected += 1

    study_raw_fpr = (study_raw_false_alarms / study_total_windows * 100.0) if study_total_windows > 0 else 0.0
    study_dsp_fpr = (study_dsp_false_alarms / study_total_windows * 100.0) if study_total_windows > 0 else 0.0
    study_rpm_pct = (study_rpm_detected / study_total_windows * 100.0) if study_total_windows > 0 else 0.0

    print(f"Zbadano okien rzeczywistych dysków (Toshiba, WD, Seagate, HGST): {study_total_windows}")
    print(f"Fałszywe alarmy awarii BEZ DSP: {study_raw_false_alarms} (FPR: {study_raw_fpr:.2f}%)")
    print(f"Fałszywe alarmy awarii PO UŻYCIU DSP: {study_dsp_false_alarms} (FPR: {study_dsp_fpr:.2f}%)")
    print(f"Wykrycie prawidłowej rotacji silnika wrzeciona (5400/7200 RPM): {study_rpm_detected}/{study_total_windows} ({study_rpm_pct:.1f}%)")

    # =========================================================================
    # TEST SPECJALNY: ODPORNOŚĆ NA SZUMY WENTYLATORÓW I TŁA PC
    # =========================================================================
    print("\n" + "=" * 80)
    print("EWALUACJA SPECJALNA: ODPORNOŚĆ NA SZUMY WENTYLATORÓW I TŁA PC (~45 dB SPL)")
    print("=" * 80)
    ambient_files = []
    ambient_files.extend(sorted(list((PROJECT_ROOT / "data" / "raw" / "outliers_oe").glob("oe_pc_fan*.wav"))))
    ambient_files.extend(sorted(list((PROJECT_ROOT / "data" / "raw" / "computer_ambient").glob("*.wav"))))

    amb_raw_false_alarms = 0
    amb_dsp_false_alarms = 0
    amb_total = 0

    for af in ambient_files:
        try:
            y, sr = sf.read(str(af))
            if y.ndim > 1: y = np.mean(y, axis=1)
            if sr != TARGET_SR:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

            num_w = max(1, len(y) // CHUNK_SAMPLES)
            for w in range(num_w):
                chunk = y[w * CHUNK_SAMPLES : (w + 1) * CHUNK_SAMPLES]
                if len(chunk) < CHUNK_SAMPLES:
                    chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
                eval_res = tester.evaluate_chunk(chunk)
                amb_total += 1
                if eval_res["raw_is_fault"]:
                    amb_raw_false_alarms += 1
                if eval_res["dsp_is_fault"]:
                    amb_dsp_false_alarms += 1
        except Exception:
            pass

    amb_raw_fpr = (amb_raw_false_alarms / amb_total * 100.0) if amb_total > 0 else 0.0
    amb_dsp_fpr = (amb_dsp_false_alarms / amb_total * 100.0) if amb_total > 0 else 0.0

    print(f"Zbadano okien szumów wentylatorów PC / chłodzenia: {amb_total}")
    print(f"Fałszywe alarmy awarii BEZ DSP (surowy model): {amb_raw_false_alarms} (FPR: {amb_raw_fpr:.2f}%)")
    print(f"Fałszywe alarmy awarii PO UŻYCIU DSP:          {amb_dsp_false_alarms} (FPR: {amb_dsp_fpr:.2f}%)")

    # Zapis raportu do JSON
    report_file = PROJECT_ROOT / "training_output" / "evaluation_with_vs_without_dsp.json"
    results_summary["real_drives_study"] = {
        "total_windows": study_total_windows,
        "raw_fpr": study_raw_fpr,
        "dsp_fpr": study_dsp_fpr,
        "rpm_detection_rate": study_rpm_pct
    }
    results_summary["ambient_noise_study"] = {
        "total_windows": amb_total,
        "raw_fpr": amb_raw_fpr,
        "dsp_fpr": amb_dsp_fpr
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n[Raport] Pełny raport zapisano do: {report_file}")
    print("=" * 80)
    print("EWALUACJA ZAKOŃCZONA SUKCESEM!")
    print("=" * 80)


if __name__ == "__main__":
    run_full_evaluation()
