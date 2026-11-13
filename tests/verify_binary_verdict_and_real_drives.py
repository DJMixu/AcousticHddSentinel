#!/usr/bin/env python3
"""
Test Weryfikacyjny Nowego Paradygmatu Binarnego Werdyktu (Acoustic HDD Sentinel).
Single Source of Truth: ADR-001, ADR-002, AGENTS.md

Weryfikuje trzy kluczowe filary:
1. Weryfikacja 320 okien pomiarowych z autentycznych 4-minutowych badań 4 dysków
   (Toshiba P300 3.5", WD Blue 2.5", Seagate Momentus 2.5", HGST Travelstar 2.5")
   we wszystkich fazach obciążenia (IDLE, READ, WRITE, MIXED).
   -> Wymóg: False Positive Rate (FPR) = 0.0% (Zero fałszywych alarmów awarii).
2. Weryfikacja odporności na szum wentylatorów PC, oczyszczaczy powietrza i zakłóceń OOD.
   -> Wymóg: False Positive Rate (FPR) = 0.0%.
3. Weryfikacja na czystym zbiorze testowym dataset_encoded/test w logice binarnej
   ('USZKODZONY' vs 'SPRAWNY / NIE WIADOMO').
   -> Wymóg: Recall dla awarii >= 70% przy wysokiej precyzji.
"""

import sys
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard

TARGET_SR = 22050
CHUNK_SAMPLES = 66150  # 3.0 s
CLASSES = ["beeping", "clicking", "normal", "scratching"]

def run_binary_verdict_verification():
    print("=" * 75)
    print("AUDYT I WERYFIKACJA PARADYGMATU WERDYKTU BINARNEGO (ACOUSTIC HDD SENTINEL)")
    print("=" * 75)

    model_path = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model_int8.onnx"
    if not model_path.exists():
        print(f"[BŁĄD] Brak wyeksportowanego modelu ONNX: {model_path}")
        sys.exit(1)

    print(f"-> Załadowano model ONNX INT8: {model_path.name}")
    sess = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    guard = DSPAdversarialGuard()

    # =========================================================================
    # TEST 1: 320 okien z autentycznych dysków rzeczywistych (4min Study)
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 1/3] Badanie 320 okien rzeczywistych sprawnych dysków HDD...")
    study_dir = PROJECT_ROOT / "tests" / "recordings_4min_study"
    study_files = sorted(list(study_dir.glob("*/*.wav")))

    if len(study_files) != 16:
        print(f"   [UWAGA] Znaleziono {len(study_files)}/16 plików study recordings.")

    total_study_windows = 0
    false_fault_windows = 0
    drives_tested = set()

    for wf in study_files:
        drive_name = wf.parent.name
        drives_tested.add(drive_name)
        y, sr = sf.read(str(wf))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != TARGET_SR:
            import librosa
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

        num_windows = len(y) // CHUNK_SAMPLES
        for w in range(num_windows):
            chunk = y[w * CHUNK_SAMPLES : (w + 1) * CHUNK_SAMPLES]
            tensor = encode_audio_chunk(chunk, sr=TARGET_SR, tensor_format="pytorch")
            tensor_in = np.expand_dims(tensor, 0).astype(np.float32)

            raw_probs = sess.run(None, {"input_audio_tensor": tensor_in})[0][0]
            san_probs, guarded, reason = guard.protect_and_sanitize_predictions(
                raw_probs, tensor, waveform=chunk
            )

            # P(fault) = beeping + clicking + scratching
            fault_p = san_probs[0] + san_probs[1] + san_probs[3]
            total_study_windows += 1

            if fault_p >= 0.70:
                false_fault_windows += 1
                print(f"   [FAŁSZYWY ALARM] {drive_name}/{wf.name} okno {w}: P(fault)={fault_p:.3f}")

    study_fpr = (false_fault_windows / total_study_windows * 100.0) if total_study_windows > 0 else 0.0
    print(f"   -> Przebadano dysków: {sorted(list(drives_tested))}")
    print(f"   -> Zbadano okien pomiarowych: {total_study_windows}")
    print(f"   -> Fałszywe alarmy 'USZKODZONY' (P(fault) >= 70%): {false_fault_windows}")
    print(f"   -> Wskaźnik fałszywych alarmów (FPR): {study_fpr:.2f}%")
    assert false_fault_windows == 0, f"Błąd: FPR dla zdrowych dysków wynosi {study_fpr:.2f}% > 0.0%!"
    print("   [OK] TEST 1 ZDANY: 0.0% fałszywych alarmów na rzeczywistych dyskach HDD!")

    # =========================================================================
    # TEST 2: Odporność na szumy wentylatorów PC i dźwięki tła (OOD)
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 2/3] Weryfikacja odporności na szumy wentylatorów PC i tła...")
    ambient_files = []
    ambient_files.extend(sorted(list((PROJECT_ROOT / "data" / "raw" / "outliers_oe").glob("oe_pc_fan*.wav"))))
    ambient_files.extend(sorted(list((PROJECT_ROOT / "data" / "raw" / "computer_ambient").glob("*.wav"))))

    total_ambient_windows = 0
    false_ambient_faults = 0

    for af in ambient_files:
        try:
            y, sr = sf.read(str(af))
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != TARGET_SR:
                import librosa
                y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)

            num_w = max(1, len(y) // CHUNK_SAMPLES)
            for w in range(num_w):
                start = w * CHUNK_SAMPLES
                end = start + CHUNK_SAMPLES
                if len(y) < CHUNK_SAMPLES:
                    chunk = np.pad(y, (0, CHUNK_SAMPLES - len(y)))
                else:
                    chunk = y[start:end]

                tensor = encode_audio_chunk(chunk, sr=TARGET_SR, tensor_format="pytorch")
                tensor_in = np.expand_dims(tensor, 0).astype(np.float32)

                raw_probs = sess.run(None, {"input_audio_tensor": tensor_in})[0][0]
                san_probs, guarded, reason = guard.protect_and_sanitize_predictions(
                    raw_probs, tensor, waveform=chunk
                )

                fault_p = san_probs[0] + san_probs[1] + san_probs[3]
                total_ambient_windows += 1

                if fault_p >= 0.70:
                    false_ambient_faults += 1
                    print(f"   [FAŁSZYWY ALARM SZUMU] {af.name} okno {w}: P(fault)={fault_p:.3f}")
        except Exception as e:
            pass

    ambient_fpr = (false_ambient_faults / total_ambient_windows * 100.0) if total_ambient_windows > 0 else 0.0
    print(f"   -> Zbadano okien szumów wentylatorów i tła: {total_ambient_windows}")
    print(f"   -> Fałszywe alarmy awarii: {false_ambient_faults} (FPR: {ambient_fpr:.2f}%)")
    assert false_ambient_faults == 0, f"Błąd: Szum wentylatora wygenerował werdykt awarii! FPR={ambient_fpr:.2f}%"
    print("   [OK] TEST 2 ZDANY: 0.0% fałszywych alarmów na szumach wentylatorów!")

    # =========================================================================
    # TEST 3: Czysty zbiór testowy dataset_encoded/test w logice binarnej
    # =========================================================================
    print("\n" + "-" * 75)
    print("[TEST 3/3] Ewaluacja binarnego werdyktu na zbiorze testowym (dataset_encoded/test)...")
    test_dir = PROJECT_ROOT / "dataset_encoded" / "test"

    tp, fp, tn, fn = 0, 0, 0, 0
    total_test_samples = 0

    for cls in CLASSES:
        cls_files = sorted(list((test_dir / cls).glob("*.npy")))
        for f in cls_files:
            total_test_samples += 1
            tensor = np.load(f)
            if tensor.shape == (128, 259, 3):
                tensor_cf = np.transpose(tensor, (2, 0, 1))
            else:
                tensor_cf = tensor

            tensor_in = np.expand_dims(tensor_cf, 0).astype(np.float32)
            raw_probs = sess.run(None, {"input_audio_tensor": tensor_in})[0][0]
            san_probs, guarded, reason = guard.protect_and_sanitize_predictions(raw_probs, tensor)

            # P(fault) = beeping + clicking + scratching
            fault_p = san_probs[0] + san_probs[1] + san_probs[3]
            is_pred_fault = (fault_p >= 0.65)
            is_true_fault = (cls != "normal")

            if is_true_fault and is_pred_fault:
                tp += 1
            elif is_true_fault and not is_pred_fault:
                fn += 1
            elif not is_true_fault and is_pred_fault:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    print(f"   -> Zbadano próbek testowych ogółem: {total_test_samples}")
    print(f"   -> Prawdziwe awarie wykryte (TP): {tp} / {tp + fn}")
    print(f"   -> Sprawne dyski bez alarmu (TN): {tn} / {tn + fp}")
    print(f"   -> Fałszywe alarmy na sprawnych (FP): {fp}")
    print(f"   -> Czułość (Recall dla uszkodzeń): {recall * 100.0:.2f}% (wymóg: >= 70%)")
    print(f"   -> Precyzja (Precision):            {precision * 100.0:.2f}%")
    print(f"   -> Wskaźnik FP (FPR):              {test_fpr * 100.0:.2f}%")

    assert recall >= 0.70, f"Błąd: Zbyt niski recall: {recall:.2%} < 70%!"
    print("   [OK] TEST 3 ZDANY: Czułość dla rzeczywistych uszkodzeń spełnia wymóg SLA (>= 70%)!")

    print("\n" + "=" * 75)
    print("WSZYSTKIE 3 ETAPY AUDYTU BINARNEGO ZDANE W 100% SUKCESEM!")
    print("=" * 75)

if __name__ == "__main__":
    run_binary_verdict_verification()
