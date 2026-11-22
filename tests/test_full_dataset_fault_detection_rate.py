#!/usr/bin/env python3
"""
HDD Acoustic Sentinel - Weryfikacja Wykrywalności Awarii na 100% Całego Datasetu
Single Source of Truth: ADR-001, ADR-002, AGENTS.md

Cel skryptu:
Odpowiedź na pytanie: "Czy wykrywalność błędów (awarii) wynosi 50% na całym datasiecie?"
1. Testuje 100% tensorów w dataset_encoded (łącznie 6 840 tensorów):
   - Zbiór TEST:     516 tensorów (213 awarii, 303 sprawne)
   - Zbiór VALIDATE: 279 tensorów (131 awarii, 148 sprawne)
   - Zbiór TRAIN:    6 045 tensorów (3 421 awarii, 2 624 sprawne)
   - CAŁY DATASET:   6 840 tensorów (3 765 awarii, 3 075 sprawne)
2. Sprawdza wykrywalność awarii:
   - Surowy model ONNX (Raw CNN)
   - Po użyciu DSP (DSP Guard & Sanitization)
   - Dla różnych progów decyzyjnych:
     * Próg operacyjny ADR-002: P(fault) >= 0.70
     * Standardowy próg binarny: P(fault) >= 0.50
     * Klasyfikacja dominująca: argmax(probs) != 'normal'
3. Generuje pełną macierz pomyłek i raport numeryczny.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import onnxruntime as ort

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))

from ml.dsp_adversarial_guard import DSPAdversarialGuard, CLASSES

ENCODED_DIR = PROJECT_ROOT / "dataset_encoded"
MODEL_PATH = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model.onnx"
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "exported_models" / "best_hdd_cnn_model_int8.onnx"


def run_full_dataset_test():
    print("=" * 80)
    print("WERYFIKACJA WYKRYWALNOŚCI AWARII NA 100% CAŁEGO DATASETU (6 840 TENSORÓW)")
    print("=" * 80)
    print(f"Model ONNX: {MODEL_PATH.name}")

    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    guard = DSPAdversarialGuard()

    splits = ["test", "validate", "train"]
    overall_stats: Dict[str, Any] = {}

    total_dataset_samples = 0
    total_dataset_faults = 0
    total_raw_detected_70 = 0
    total_raw_detected_50 = 0
    total_raw_detected_argmax = 0
    total_dsp_detected_70 = 0
    total_dsp_detected_50 = 0
    total_dsp_detected_argmax = 0

    class_stats = {
        c: {"total": 0, "raw_det_70": 0, "dsp_det_70": 0, "raw_det_50": 0, "dsp_det_50": 0}
        for c in CLASSES
    }

    start_time = time.time()

    for split in splits:
        split_dir = ENCODED_DIR / split
        print(f"\n--- BADANIE ZBIORU: {split.upper()} ---")

        split_faults = 0
        split_normal = 0
        split_raw_det_70 = 0
        split_raw_det_50 = 0
        split_dsp_det_70 = 0
        split_dsp_det_50 = 0
        split_raw_fp_70 = 0
        split_dsp_fp_70 = 0

        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.exists():
                continue
            files = sorted(list(cls_dir.glob("*.npy")))
            is_fault = (cls != "normal")

            for f in files:
                tensor = np.load(f)
                if tensor.shape == (128, 259, 3):
                    tensor_cf = np.transpose(tensor, (2, 0, 1))
                else:
                    tensor_cf = tensor

                tensor_in = np.expand_dims(tensor_cf, axis=0).astype(np.float32)

                # 1. Inferencja surowego modelu ONNX
                raw_probs = session.run(None, {input_name: tensor_in})[0][0].astype(np.float32)
                raw_fault_prob = float(raw_probs[0] + raw_probs[1] + raw_probs[3])
                raw_argmax_idx = int(np.argmax(raw_probs))
                raw_argmax_class = CLASSES[raw_argmax_idx]

                # 2. Potok DSP
                san_probs, guarded, _ = guard.protect_and_sanitize_predictions(raw_probs, tensor_cf)
                san_total = float(np.sum(san_probs))
                dsp_probs = (san_probs / san_total) if san_total > 0 else san_probs
                dsp_fault_prob = float(dsp_probs[0] + dsp_probs[1] + dsp_probs[3])
                dsp_argmax_idx = int(np.argmax(dsp_probs))
                dsp_argmax_class = CLASSES[dsp_argmax_idx]

                # Zliczanie statystyk
                total_dataset_samples += 1
                class_stats[cls]["total"] += 1

                if is_fault:
                    split_faults += 1
                    total_dataset_faults += 1

                    if raw_fault_prob >= 0.70:
                        split_raw_det_70 += 1
                        total_raw_detected_70 += 1
                        class_stats[cls]["raw_det_70"] += 1

                    if raw_fault_prob >= 0.50:
                        split_raw_det_50 += 1
                        total_raw_detected_50 += 1
                        class_stats[cls]["raw_det_50"] += 1

                    if raw_argmax_class != "normal":
                        total_raw_detected_argmax += 1

                    if dsp_fault_prob >= 0.70:
                        split_dsp_det_70 += 1
                        total_dsp_detected_70 += 1
                        class_stats[cls]["dsp_det_70"] += 1

                    if dsp_fault_prob >= 0.50:
                        split_dsp_det_50 += 1
                        total_dsp_detected_50 += 1
                        class_stats[cls]["dsp_det_50"] += 1

                    if dsp_argmax_class != "normal":
                        total_dsp_detected_argmax += 1

                else:
                    split_normal += 1
                    if raw_fault_prob >= 0.70:
                        split_raw_fp_70 += 1
                    if dsp_fault_prob >= 0.70:
                        split_dsp_fp_70 += 1

        raw_rec_70 = (split_raw_det_70 / split_faults * 100.0) if split_faults > 0 else 0.0
        dsp_rec_70 = (split_dsp_det_70 / split_faults * 100.0) if split_faults > 0 else 0.0
        raw_rec_50 = (split_raw_det_50 / split_faults * 100.0) if split_faults > 0 else 0.0
        dsp_rec_50 = (split_dsp_det_50 / split_faults * 100.0) if split_faults > 0 else 0.0

        raw_fpr_70 = (split_raw_fp_70 / split_normal * 100.0) if split_normal > 0 else 0.0
        dsp_fpr_70 = (split_dsp_fp_70 / split_normal * 100.0) if split_normal > 0 else 0.0

        print(f"  Liczba próbek w zbiorze: {split_faults + split_normal} (Awarie: {split_faults}, Sprawne: {split_normal})")
        print(f"  Wykrywalność awarii BEZ DSP (P(fault) >= 0.70): {split_raw_det_70}/{split_faults} ({raw_rec_70:.2f}%)")
        print(f"  Wykrywalność awarii PO DSP  (P(fault) >= 0.70): {split_dsp_det_70}/{split_faults} ({dsp_rec_70:.2f}%)")
        print(f"  Wykrywalność awarii PO DSP  (P(fault) >= 0.50): {split_dsp_det_50}/{split_faults} ({dsp_rec_50:.2f}%)")
        print(f"  Fałszywe alarmy na sprawnych (FPR) PO DSP:     {split_dsp_fp_70}/{split_normal} ({dsp_fpr_70:.2f}%)")

        overall_stats[split] = {
            "total_samples": split_faults + split_normal,
            "faults": split_faults,
            "normal": split_normal,
            "raw_rec_70": raw_rec_70,
            "dsp_rec_70": dsp_rec_70,
            "dsp_rec_50": dsp_rec_50,
            "dsp_fpr_70": dsp_fpr_70
        }

    elapsed = time.time() - start_time

    # =========================================================================
    # PODSUMOWANIE DLA 100% CAŁEGO DATASETU
    # =========================================================================
    overall_raw_rec_70 = (total_raw_detected_70 / total_dataset_faults * 100.0) if total_dataset_faults > 0 else 0.0
    overall_dsp_rec_70 = (total_dsp_detected_70 / total_dataset_faults * 100.0) if total_dataset_faults > 0 else 0.0
    overall_dsp_rec_50 = (total_dsp_detected_50 / total_dataset_faults * 100.0) if total_dataset_faults > 0 else 0.0
    overall_dsp_rec_argmax = (total_dsp_detected_argmax / total_dataset_faults * 100.0) if total_dataset_faults > 0 else 0.0

    print("\n" + "=" * 80)
    print("GLOBALNA ODPOWIEDŹ: WYKRYWALNOŚĆ AWARII DLA 100% CAŁEGO DATASETU")
    print("=" * 80)
    print(f"Łącznie zbadano tensorów: {total_dataset_samples}")
    print(f"  - Próbki rzeczywistych AWARII: {total_dataset_faults}")
    print(f"  - Próbki dysków SPRAWNYCH:    {total_dataset_samples - total_dataset_faults}")
    print(f"Czas ewaluacji 6 840 tensorów: {elapsed:.2f} s ({elapsed / total_dataset_samples * 1000:.3f} ms / chunk)")

    print("\n--- WYKRYWALNOŚĆ AWARII (RECALL) DLA CAŁEGO DATASETU ---")
    print(f"1. BEZ DSP (Surowy model, P(fault) >= 0.70):   {total_raw_detected_70}/{total_dataset_faults} ({overall_raw_rec_70:.2f}%)")
    print(f"2. PO UŻYCIU DSP (Próg operacyjny >= 0.70):   {total_dsp_detected_70}/{total_dataset_faults} ({overall_dsp_rec_70:.2f}%)")
    print(f"3. PO UŻYCIU DSP (Standardowy próg >= 0.50):   {total_dsp_detected_50}/{total_dataset_faults} ({overall_dsp_rec_50:.2f}%)")
    print(f"4. PO UŻYCIU DSP (Klasa dominująca argmax):    {total_dsp_detected_argmax}/{total_dataset_faults} ({overall_dsp_rec_argmax:.2f}%)")

    print("\n--- WYKRYWALNOŚĆ PER KLASA USZKODZENIA (PO UŻYCIU DSP, P >= 0.70) ---")
    print(f"{'Typ Uszkodzenia':<25} | {'Łącznie próbek':<15} | {'Wykryte PO DSP':<15} | {'Wykrywalność %':<15}")
    print("-" * 75)
    for c in ["beeping", "clicking", "scratching"]:
        tot = class_stats[c]["total"]
        det = class_stats[c]["dsp_det_70"]
        pct = (det / tot * 100.0) if tot > 0 else 0.0
        print(f"{c:<25} | {tot:<15} | {det:<15} | {pct:>13.2f}%")

    # Zapis do JSON
    report_file = PROJECT_ROOT / "training_output" / "full_dataset_fault_detection_summary.json"
    summary_data = {
        "total_dataset_samples": total_dataset_samples,
        "total_faults": total_dataset_faults,
        "overall_raw_recall_70": overall_raw_rec_70,
        "overall_dsp_recall_70": overall_dsp_rec_70,
        "overall_dsp_recall_50": overall_dsp_rec_50,
        "overall_dsp_recall_argmax": overall_dsp_rec_argmax,
        "splits": overall_stats,
        "class_breakdown": {
            c: {
                "total": class_stats[c]["total"],
                "raw_det_70": class_stats[c]["raw_det_70"],
                "dsp_det_70": class_stats[c]["dsp_det_70"],
                "dsp_recall_pct": (class_stats[c]["dsp_det_70"] / class_stats[c]["total"] * 100.0) if class_stats[c]["total"] > 0 else 0.0
            }
            for c in CLASSES
        }
    }
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nRaport zapisano do: {report_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_full_dataset_test()
