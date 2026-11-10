"""
Skrypt weryfikujący i demonstrujący działanie modułu normalizacji dźwięku
na rzeczywistych nagraniach z katalogu dataset/.
"""

import os
import glob
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audio_processor import AudioNormalizer

def run_verification():
    normalizer = AudioNormalizer(
        target_sample_rate=44100,
        chunk_duration_sec=5.0,
        n_mels=64,
        n_fft=1024,
        hop_length=512,
        peak_target=0.95
    )

    dataset_dir = "/home/DJMixu/Pulpit/AcousticHddSentinel/CNN/dataset"
    wav_files = sorted(glob.glob(os.path.join(dataset_dir, "*.wav")))

    if not wav_files:
        print("Nie znaleziono plików .wav w katalogu:", dataset_dir)
        return

    print(f"Znaleziono {len(wav_files)} plików audio.")
    print("=" * 70)
    print("WERYFIKACJA NORMALIZACJI NA PRÓBKACH RZECZYWISTYCH")
    print("=" * 70)

    # Wybierzmy 3 reprezentatywne pliki:
    # 1. Cichy
    # 2. Głośny / z klikami
    # 3. Z dużym DC offsetem
    selected_indices = [0, 10, 20]
    sample_files = [wav_files[i] for i in selected_indices if i < len(wav_files)]

    fig, axes = plt.subplots(len(sample_files), 3, figsize=(16, 4 * len(sample_files)))
    if len(sample_files) == 1:
        axes = np.expand_dims(axes, 0)

    for idx, fpath in enumerate(sample_files):
        fname = os.path.basename(fpath)
        # Surowy sygnał
        raw_wave, orig_sr = normalizer.load_wav(fpath)
        raw_dc = torch.mean(raw_wave).item()
        raw_peak = torch.max(torch.abs(raw_wave)).item()
        raw_rms = torch.sqrt(torch.mean(raw_wave ** 2)).item()

        # Pełne przetwarzanie i normalizacja
        chunks_data = normalizer.process_file(fpath, normalization_method="peak")
        first_chunk_wave, first_mel = chunks_data[0]

        norm_dc = torch.mean(first_chunk_wave).item()
        norm_peak = torch.max(torch.abs(first_chunk_wave)).item()
        norm_rms = torch.sqrt(torch.mean(first_chunk_wave ** 2)).item()

        print(f"\nPlik: {fname[:45]}...")
        print(f"  [PRZED NORMALIZACJĄ] SR: {orig_sr} Hz, Długość: {raw_wave.shape[-1]/orig_sr:.2f}s")
        print(f"    - DC Offset: {raw_dc:+.6f}")
        print(f"    - Peak Amp:  {raw_peak:.4f}")
        print(f"    - RMS:       {raw_rms:.5f}")
        print(f"  [PO NORMALIZACJI (okno 5s)] SR: {normalizer.target_sample_rate} Hz, Kształt Mel: {list(first_mel.shape)}")
        print(f"    - DC Offset: {norm_dc:+.6f} (zredukowany do ~0)")
        print(f"    - Peak Amp:  {norm_peak:.4f} (znormalizowany do {normalizer.peak_target})")
        print(f"    - RMS:       {norm_rms:.5f}")
        print(f"    - Mel min/max: min={first_mel.min().item():.3f}, max={first_mel.max().item():.3f} (zakres [0, 1])")

        # Rysowanie wykresów
        # Kolumna 1: Raw Waveform (pierwsze 5s)
        t_raw = np.linspace(0, min(5.0, raw_wave.shape[-1]/orig_sr), int(orig_sr * min(5.0, raw_wave.shape[-1]/orig_sr)))
        raw_slice = raw_wave[0, :len(t_raw)].numpy()
        axes[idx, 0].plot(t_raw, raw_slice, color='darkorange', alpha=0.8, linewidth=0.5)
        axes[idx, 0].set_title(f"Raw Audio (SR={orig_sr}Hz, Peak={raw_peak:.2f})")
        axes[idx, 0].set_ylabel("Amplituda")
        axes[idx, 0].set_ylim(-1.1, 1.1)
        axes[idx, 0].grid(True, alpha=0.3)

        # Kolumna 2: Normalized Waveform (5s @ 44.1kHz)
        t_norm = np.linspace(0, 5.0, normalizer.chunk_samples)
        norm_slice = first_chunk_wave[0].numpy()
        axes[idx, 1].plot(t_norm, norm_slice, color='royalblue', alpha=0.8, linewidth=0.5)
        axes[idx, 1].set_title(f"Znormalizowane 5s (Peak={norm_peak:.2f}, DC~0)")
        axes[idx, 1].set_ylim(-1.1, 1.1)
        axes[idx, 1].grid(True, alpha=0.3)

        # Kolumna 3: Znormalizowany Mel-spektrogram (64 mels x 431 frames)
        im = axes[idx, 2].imshow(
            first_mel[0].numpy(),
            origin='lower',
            aspect='auto',
            cmap='magma',
            vmin=0.0,
            vmax=1.0
        )
        axes[idx, 2].set_title(f"Mel-spektrogram [0, 1] ({first_mel.shape[1]}x{first_mel.shape[2]})")
        axes[idx, 2].set_ylabel("Mel pasma")
        axes[idx, 2].set_xlabel("Ramki czasowe")
        fig.colorbar(im, ax=axes[idx, 2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    report_path = "/home/DJMixu/Pulpit/AcousticHddSentinel/CNN/normalization_report.png"
    plt.savefig(report_path, dpi=120)
    plt.close()
    print(f"\nWygenerowano wykres porównawczy: {report_path}")

if __name__ == "__main__":
    run_verification()
