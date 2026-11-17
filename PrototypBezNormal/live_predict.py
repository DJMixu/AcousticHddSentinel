#!/usr/bin/env python3
"""
live_predict.py
Inferencja i diagnostyka audio dla modeli Fault-Only (PrototypBezNormal).
Obsługuje silnik GMMFaultOODDetector (v2) oraz fallback do v1.

Single Source of Truth: ADR-001, ADR-002.
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import librosa
import torch

PROTOTYPE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOTYPE_DIR.parent
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PrototypBezNormal.ood_gmm_engine import GMMFaultOODDetector
from ml.hdd_audio_encoder import encode_audio_chunk, TARGET_SR, CHUNK_DURATION
CHUNK_SAMPLES = int(CHUNK_DURATION * TARGET_SR)


def analyze_audio_file(
    audio_path: str or Path,
    weights_path: str = "PrototypBezNormal/output_v2/best_fault_cnn_v2.pt",
    gmm_path: str = "PrototypBezNormal/output_v2/gmm_density_model.joblib",
    device: str = "cpu"
) -> dict:
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {audio_path}")

    # Inicjalizacja detektora GMM OOD
    detector = GMMFaultOODDetector(
        model_weights_path=weights_path,
        gmm_model_path=gmm_path,
        device=device
    )

    # Wczytanie audio
    y, sr = librosa.load(str(audio_path), sr=TARGET_SR, mono=True)
    total_duration = len(y) / TARGET_SR

    if len(y) < CHUNK_SAMPLES:
        pad_len = CHUNK_SAMPLES - len(y)
        y = np.pad(y, (0, pad_len), mode="constant")

    # Podział na okna 3.0 s z krokiem 1.5 s (50% overlap)
    hop = int(1.5 * TARGET_SR)
    chunks = []
    chunk_timestamps = []

    for start_idx in range(0, len(y) - CHUNK_SAMPLES + 1, hop):
        c = y[start_idx:start_idx + CHUNK_SAMPLES]
        rms = float(np.sqrt(np.mean(c ** 2)))
        if rms >= 0.002:
            chunks.append(c)
            chunk_timestamps.append(start_idx / TARGET_SR)

    if len(chunks) == 0:
        chunks = [y[:CHUNK_SAMPLES]]
        chunk_timestamps = [0.0]

    results = []

    for t_sec, c in zip(chunk_timestamps, chunks):
        tensor_np = encode_audio_chunk(c)
        res = detector.predict(tensor_np)

        results.append({
            "timestamp_s": round(t_sec, 2),
            "predicted_class": res["verdict"],
            "confidence": round(res["confidence"] * 100, 1),
            "is_fault": res["is_fault"],
            "log_likelihood": round(res["log_likelihood"], 2),
            "threshold": round(res["threshold"], 2),
            "fault_candidate": res["fault_candidate"],
            "probabilities": {k: round(v * 100, 1) for k, v in res["fault_probabilities"].items()}
        })

    # Zbiorczy werdykt
    fault_windows = [r for r in results if r["predicted_class"] != "normal"]
    if len(fault_windows) > 0:
        faults_detected = [r["predicted_class"] for r in fault_windows]
        overall_verdict = max(set(faults_detected), key=faults_detected.count)
        avg_confidence = np.mean([r["confidence"] for r in results if r["predicted_class"] == overall_verdict])
    else:
        overall_verdict = "normal"
        avg_confidence = np.mean([r["confidence"] for r in results if r["predicted_class"] == "normal"])

    summary = {
        "file": str(audio_path),
        "duration_s": round(total_duration, 2),
        "total_windows_analyzed": len(results),
        "overall_verdict": overall_verdict,
        "overall_confidence": round(float(avg_confidence), 2),
        "status": "HEALTHY / NORMAL" if overall_verdict == "normal" else f"FAULT DETECTED: {overall_verdict.upper()}",
        "window_details": results
    }

    return summary


def print_summary(res: dict):
    print("\n" + "="*80)
    print("RAPORT DIAGNOSTYCZNY AUDIO (MODEL FAULT-ONLY V2 + GMM OOD)")
    print("="*80)
    print(f"Plik wejściowy:       {res['file']}")
    print(f"Czas trwania:         {res['duration_s']} s")
    print(f"Zbadanych okien 3.0s: {res['total_windows_analyzed']}")
    print(f"OGÓLNY WERDYKT:       {res['status']}")
    print(f"Pewność diagnozy:     {res['overall_confidence']} %")
    print("-" * 80)
    print("Szczegóły per okno czasowe:")
    print(f"  {'Czas [s]':8s} | {'Predykcja':12s} | {'Pewność %':9s} | {'Log-Likelihood':15s} | {'Kandydat':10s} | {'P_Fault':8s}")
    print("-" * 80)
    for w in res["window_details"]:
        cand = w["fault_candidate"]
        p_cand = w["probabilities"].get(cand, 0.0)
        print(f"  {w['timestamp_s']:6.2f}s  | {w['predicted_class']:12s} | "
              f"{w['confidence']:7.1f}%  | {w['log_likelihood']:10.2f} (>{w['threshold']:.1f}) | "
              f"{cand:10s} | {p_cand:6.1f}%")
    print("="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analiza audio modelem Fault-Only v2 GMM.")
    parser.add_argument("audio", help="Ścieżka do pliku audio (.wav, .mp3, .ogg)")
    parser.add_argument("--weights", default="PrototypBezNormal/output_v2/best_fault_cnn_v2.pt", help="Ścieżka do wag modelu")
    parser.add_argument("--gmm", default="PrototypBezNormal/output_v2/gmm_density_model.joblib", help="Ścieżka do modelu GMM")
    parser.add_argument("--device", default="cpu", help="Urządzenie (cpu lub cuda:0)")
    args = parser.parse_args()

    res = analyze_audio_file(args.audio, weights_path=args.weights, gmm_path=args.gmm, device=args.device)
    print_summary(res)
