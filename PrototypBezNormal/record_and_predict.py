#!/usr/bin/env python3
"""
record_and_predict.py
Interaktywne narzędzie do nagrywania dźwięku dysku fizycznym mikrofonem i natychmiastowej
analizy modelem PrototypBezNormal (trenowanym wyłącznie na błędach).

Niezmiennik sprzętowy (AGENTS.md):
Fizyczny mikrofon: alsa_input.pci-0000_7b_00.6.analog-stereo (Realtek ALC1220 Analog).
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

PROTOTYPE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PROTOTYPE_DIR.parent
if str(PROTOTYPE_DIR) not in sys.path:
    sys.path.insert(0, str(PROTOTYPE_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from live_predict import analyze_audio_file, print_summary

PHYSICAL_MIC_SOURCE = "alsa_input.pci-0000_7b_00.6.analog-stereo"
TARGET_SR = 22050


def record_from_physical_mic(duration_sec: float = 4.5, output_wav: str = "PrototypBezNormal/output_v2/mic_recording.wav") -> str:
    out_p = Path(output_wav)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"🎙️  REJESTRACJA AUDIO Z FIZYCZNEGO MIKROFONU")
    print("="*70)
    print(f"Urządzenie źródłowe: {PHYSICAL_MIC_SOURCE}")
    print(f"Czas trwania:        {duration_sec} sekund")
    print(f"Próbkowanie:         {TARGET_SR} Hz (Mono WAV)")
    print(f"Plik wyjściowy:      {out_p}")
    print("\n🔴 NAGRYWANIE... Zbliż mikrofon do dysku HDD!")

    # Wywołanie ffmpeg z jawnym źródłem PulseAudio
    cmd = [
        "ffmpeg", "-y",
        "-f", "pulse",
        "-i", PHYSICAL_MIC_SOURCE,
        "-t", str(duration_sec),
        "-ar", str(TARGET_SR),
        "-ac", "1",
        str(out_p)
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)
        print("✅ Nagrywanie zakończone pomyślnie!\n")
        return str(out_p)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode("utf-8", errors="ignore")
        print(f"❌ Błąd podczas nagrywania: {err}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Nagraj dźwięk dysku mikrofonem i przeanalizuj modelem PrototypBezNormal.")
    parser.add_argument("--duration", type=float, default=4.5, help="Czas nagrywania w sekundach (domyślnie: 4.5 s)")
    parser.add_argument("--output", default="PrototypBezNormal/output_v2/mic_recording.wav", help="Ścieżka do nagranego pliku WAV")
    parser.add_argument("--weights", default="PrototypBezNormal/output_v2/best_fault_cnn_v2.pt", help="Ścieżka do wag modelu")
    parser.add_argument("--gmm", default="PrototypBezNormal/output_v2/gmm_density_model.joblib", help="Ścieżka do modelu GMM")
    parser.add_argument("--device", default="cpu", help="Urządzenie do inferencji (cpu / cuda:0)")
    args = parser.parse_args()

    recorded_file = record_from_physical_mic(duration_sec=args.duration, output_wav=args.output)
    res = analyze_audio_file(recorded_file, weights_path=args.weights, gmm_path=args.gmm, device=args.device)
    print_summary(res)


if __name__ == "__main__":
    main()
