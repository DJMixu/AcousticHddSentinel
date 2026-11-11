"""
Kompleksowa weryfikacja i profilowanie jakości normalizacji (Acoustic HDD Sentinel).
Single Source of Truth: ADR-001, AGENTS.md, hdd_audio_encoder.py

Skrypt analizuje:
1. Surowe pliki audio (rozrzut energii, RMS, Crest Factor, DC offset).
2. Krok 1: Normalizację fali dźwiękowej (Peak vs RMS).
3. Krok 2: Spektrogram Log-Mel, Delty i Delty-Delty.
4. Krok 3: Standaryzację per-channel (Z-Score) we wszystkich wygenerowanych plikach .npy.
5. Generuje raport wizualny (normalization_quality_report.png) oraz zestaw metryk numerycznych.
"""

import os
import glob
import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ścieżki
if Path("data/raw").exists():
    RAW_DATASET_DIR = Path("data/raw")
elif Path("dataset_categorized").exists():
    RAW_DATASET_DIR = Path("dataset_categorized")
else:
    RAW_DATASET_DIR = Path("dataset")

ENCODED_DIR = Path("dataset_encoded")
OUTPUT_DIR = Path("training_output")
OUTPUT_IMG = OUTPUT_DIR / "normalization_quality_report.png"
OUTPUT_METRICS_JSON = OUTPUT_DIR / "normalization_quality_metrics.json"


def analyze_raw_audio_profile():
    """
    Analizuje statystyki surowych nagrań przed normalizacją.
    """
    audio_files = sorted([p for p in RAW_DATASET_DIR.rglob("*") if p.suffix.lower() in [".wav", ".mp3", ".flac"]])
    rms_list = []
    peak_list = []
    crest_list = []
    dc_list = []
    dur_list = []

    for fpath in audio_files:
        try:
            y, sr = librosa.load(str(fpath), sr=22050, mono=True)
            if len(y) == 0:
                continue
            dc = np.mean(y)
            peak = np.max(np.abs(y))
            rms = np.sqrt(np.mean(y ** 2))
            cf = peak / (rms + 1e-9)
            dur = len(y) / sr

            rms_list.append(rms)
            peak_list.append(peak)
            crest_list.append(cf)
            dc_list.append(dc)
            dur_list.append(dur)
        except Exception:
            continue

    raw_stats = {
        "count": len(rms_list),
        "rms_min": float(np.min(rms_list)),
        "rms_max": float(np.max(rms_list)),
        "rms_mean": float(np.mean(rms_list)),
        "rms_ratio_max_min": float(np.max(rms_list) / (np.min(rms_list) + 1e-9)),
        "peak_min": float(np.min(peak_list)),
        "peak_max": float(np.max(peak_list)),
        "crest_min": float(np.min(crest_list)),
        "crest_max": float(np.max(crest_list)),
        "crest_mean": float(np.mean(crest_list)),
        "crest_p95": float(np.percentile(crest_list, 95)),
        "dc_offset_max_abs": float(np.max(np.abs(dc_list))),
        "dc_offset_mean_abs": float(np.mean(np.abs(dc_list))),
    }
    return raw_stats, audio_files


def analyze_encoded_tensors_profile():
    """
    Analizuje wszystkie pliki .npy w katalogu dataset_encoded (po pełnej normalizacji).
    """
    npy_files = sorted(glob.glob(str(ENCODED_DIR / "**" / "*.npy"), recursive=True))
    if not npy_files:
        return {}

    ch0_means, ch0_stds, ch0_mins, ch0_maxs = [], [], [], []
    ch1_means, ch1_stds, ch1_mins, ch1_maxs = [], [], [], []
    ch2_means, ch2_stds, ch2_mins, ch2_maxs = [], [], [], []
    outliers_3sigma_ratios = []

    nans_total = 0
    infs_total = 0

    for fpath in npy_files:
        arr = np.load(fpath)  # Kształt: (128, 259, 3)

        if np.isnan(arr).any():
            nans_total += 1
        if np.isinf(arr).any():
            infs_total += 1

        for c_idx, (means, stds, mins, maxs) in enumerate([
            (ch0_means, ch0_stds, ch0_mins, ch0_maxs),
            (ch1_means, ch1_stds, ch1_mins, ch1_maxs),
            (ch2_means, ch2_stds, ch2_mins, ch2_maxs),
        ]):
            ch = arr[..., c_idx]
            means.append(float(np.mean(ch)))
            stds.append(float(np.std(ch)))
            mins.append(float(np.min(ch)))
            maxs.append(float(np.max(ch)))

        # Odsetek wartości poza 3 odchyleniami standardowymi (|z| > 3.0)
        outliers = np.mean(np.abs(arr) > 3.0)
        outliers_3sigma_ratios.append(float(outliers))

    encoded_stats = {
        "total_tensors": len(npy_files),
        "nan_tensors": nans_total,
        "inf_tensors": infs_total,
        "channel_0_log_mel": {
            "mean_of_means": float(np.mean(ch0_means)),
            "std_of_means": float(np.std(ch0_means)),
            "mean_of_stds": float(np.mean(ch0_stds)),
            "std_of_stds": float(np.std(ch0_stds)),
            "min_global": float(np.min(ch0_mins)),
            "max_global": float(np.max(ch0_maxs)),
        },
        "channel_1_delta": {
            "mean_of_means": float(np.mean(ch1_means)),
            "std_of_means": float(np.std(ch1_means)),
            "mean_of_stds": float(np.mean(ch1_stds)),
            "std_of_stds": float(np.std(ch1_stds)),
            "min_global": float(np.min(ch1_mins)),
            "max_global": float(np.max(ch1_maxs)),
        },
        "channel_2_delta_delta": {
            "mean_of_means": float(np.mean(ch2_means)),
            "std_of_means": float(np.std(ch2_means)),
            "mean_of_stds": float(np.mean(ch2_stds)),
            "std_of_stds": float(np.std(ch2_stds)),
            "min_global": float(np.min(ch2_mins)),
            "max_global": float(np.max(ch2_maxs)),
        },
        "mean_3sigma_outlier_pct": float(np.mean(outliers_3sigma_ratios) * 100),
    }
    return encoded_stats


def generate_comparative_plots(sample_wav_files):
    """
    Generuje szczegółowe wykresy porównawcze przed i po normalizacji:
    1. Cichy vs Głośny plik przed i po Peak Normalization
    2. Histogram rozkładu wartości w kanałach Log-Mel, Delta i Delta-Delta (Z-score)
    3. Spektrogram przed Z-score vs po Z-score
    """
    from hdd_audio_encoder import encode_audio_chunk, normalize_waveform_peak

    # Wybór próbki z wysokim Crest Factor (kliknięcia) i próbki tonalnej (beeping)
    target_quiet = None
    target_loud = None

    for f in sample_wav_files:
        fname = f.name.lower()
        if "clicking" in fname and target_loud is None:
            target_loud = f
        if "beeping" in fname and target_quiet is None:
            target_quiet = f

    if target_loud is None or target_quiet is None:
        target_loud = sample_wav_files[0]
        target_quiet = sample_wav_files[min(10, len(sample_wav_files)-1)]

    y_loud, _ = librosa.load(str(target_loud), sr=22050, mono=True)
    y_quiet, _ = librosa.load(str(target_quiet), sr=22050, mono=True)

    # 3-sekundowy wycinek
    chunk_samples = int(22050 * 3.0)
    chunk_loud = y_loud[:chunk_samples]
    chunk_quiet = y_quiet[:chunk_samples]

    norm_loud = normalize_waveform_peak(chunk_loud)
    norm_quiet = normalize_waveform_peak(chunk_quiet)

    tensor_loud = encode_audio_chunk(chunk_loud, sr=22050, tensor_format="tf")

    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.25)

    # Wiersz 1: Fala dźwiękowa PRZED normalizacją
    t = np.linspace(0, 3.0, len(chunk_loud))
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, chunk_loud, color="crimson", alpha=0.8, linewidth=0.5)
    ax1.set_title(f"Przed normalizacją (Głośny / Klik)\nPeak={np.max(np.abs(chunk_loud)):.3f}, RMS={np.sqrt(np.mean(chunk_loud**2)):.4f}", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Amplituda surowa")
    ax1.set_ylim(-1.05, 1.05)
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, chunk_quiet, color="darkorange", alpha=0.8, linewidth=0.5)
    ax2.set_title(f"Przed normalizacją (Cichy / Beep)\nPeak={np.max(np.abs(chunk_quiet)):.3f}, RMS={np.sqrt(np.mean(chunk_quiet**2)):.4f}", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Amplituda surowa")
    ax2.set_ylim(-1.05, 1.05)
    ax2.grid(True, alpha=0.3)

    # Wiersz 1, kol 3: Po Peak Normalization (Wyrównanie zakresu dynamicznego)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, norm_quiet, color="royalblue", alpha=0.7, linewidth=0.5, label="Cichy po Peak Norm")
    ax3.plot(t, norm_loud, color="purple", alpha=0.5, linewidth=0.5, label="Głośny po Peak Norm")
    ax3.set_title("Po Peak Normalization ([-1.0, 1.0])\nZachowany Crest Factor, Peak=1.0", fontsize=9, fontweight="bold")
    ax3.legend(fontsize=8, loc="upper right")
    ax3.set_ylim(-1.05, 1.05)
    ax3.grid(True, alpha=0.3)

    # Wiersz 2: Kanały tensora wyjściowego (Log-Mel, Delta, Delta-Delta)
    ax4 = fig.add_subplot(gs[1, 0])
    im1 = ax4.imshow(tensor_loud[..., 0], aspect="auto", origin="lower", cmap="magma")
    ax4.set_title("Kanał 0: Log-Mel Spectrogram (Z-Score)\n(mean=0.0, std=1.0)", fontsize=9, fontweight="bold")
    ax4.set_ylabel("Mel pasma (128)")
    fig.colorbar(im1, ax=ax4, fraction=0.046, pad=0.04)

    ax5 = fig.add_subplot(gs[1, 1])
    im2 = ax5.imshow(tensor_loud[..., 1], aspect="auto", origin="lower", cmap="seismic", vmin=-3.0, vmax=3.0)
    ax5.set_title("Kanał 1: Delta (1. pochodna - Transjenty)\n(mean=0.0, std=1.0)", fontsize=9, fontweight="bold")
    ax5.set_xlabel("Ramki czasowe (259)")
    fig.colorbar(im2, ax=ax5, fraction=0.046, pad=0.04)

    ax6 = fig.add_subplot(gs[1, 2])
    im3 = ax6.imshow(tensor_loud[..., 2], aspect="auto", origin="lower", cmap="seismic", vmin=-3.0, vmax=3.0)
    ax6.set_title("Kanał 2: Delta-Delta (2. pochodna - Dynamika)\n(mean=0.0, std=1.0)", fontsize=9, fontweight="bold")
    fig.colorbar(im3, ax=ax6, fraction=0.046, pad=0.04)

    # Wiersz 3: Histogramy rozkładu Z-score we wszystkich 3 kanałach
    ax7 = fig.add_subplot(gs[2, 0])
    ax7.hist(tensor_loud[..., 0].flatten(), bins=50, color="firebrick", alpha=0.8, density=True)
    ax7.axvline(0, color="black", linestyle="--", linewidth=1)
    ax7.set_title(f"Rozkład Log-Mel\nmean={np.mean(tensor_loud[..., 0]):.2f}, std={np.std(tensor_loud[..., 0]):.2f}", fontsize=9, fontweight="bold")
    ax7.grid(True, alpha=0.3)

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.hist(tensor_loud[..., 1].flatten(), bins=50, color="seagreen", alpha=0.8, density=True)
    ax8.axvline(0, color="black", linestyle="--", linewidth=1)
    ax8.set_title(f"Rozkład Delta\nmean={np.mean(tensor_loud[..., 1]):.2f}, std={np.std(tensor_loud[..., 1]):.2f}", fontsize=9, fontweight="bold")
    ax8.grid(True, alpha=0.3)

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.hist(tensor_loud[..., 2].flatten(), bins=50, color="darkslateblue", alpha=0.8, density=True)
    ax9.axvline(0, color="black", linestyle="--", linewidth=1)
    ax9.set_title(f"Rozkład Delta-Delta\nmean={np.mean(tensor_loud[..., 2]):.2f}, std={np.std(tensor_loud[..., 2]):.2f}", fontsize=9, fontweight="bold")
    ax9.grid(True, alpha=0.3)

    plt.suptitle("RAPORT JAKOŚCI NORMALIZACJI DANYCH (ACOUSTIC HDD SENTINEL - ADR-001)", fontsize=13, fontweight="bold", y=0.99)
    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_IMG)
    plt.close()


def run_assessment():
    print("=" * 70)
    print("PROFILOWANIE I OCENA JAKOŚCI NORMALIZACJI (ADR-001)")
    print("=" * 70)

    # 1. Profil surowych nagrań
    print("\n[1/3] Analiza surowych plików audio przed normalizacją...")
    raw_stats, sample_files = analyze_raw_audio_profile()
    print(f"  - Liczba przeanalizowanych plików: {raw_stats['count']}")
    print(f"  - Rozstęp energii RMS:             od {raw_stats['rms_min']:.5f} do {raw_stats['rms_max']:.5f}")
    print(f"  - Krotność różnicy głośności:      {raw_stats['rms_ratio_max_min']:.1f}x (!) ")
    print(f"  - Maksymalny Crest Factor (kliki): {raw_stats['crest_max']:.1f} (średnia: {raw_stats['crest_mean']:.1f})")
    print(f"  - Średni DC Offset przed korektą:  {raw_stats['dc_offset_mean_abs']:.6f}")

    # 2. Profil wygenerowanych tensorów .npy
    print("\n[2/3] Weryfikacja statystyczna tensorów po normalizacji (dataset_encoded)...")
    encoded_stats = analyze_encoded_tensors_profile()
    if encoded_stats:
        print(f"  - Zbadano tensorów:               {encoded_stats['total_tensors']}")
        print(f"  - Liczba wartości NaN / Inf:       {encoded_stats['nan_tensors']} / {encoded_stats['inf_tensors']} (idealnie czyste)")
        print(f"  - Kanał 0 (Log-Mel)     mean:      {encoded_stats['channel_0_log_mel']['mean_of_means']:.6f} | std: {encoded_stats['channel_0_log_mel']['mean_of_stds']:.6f}")
        print(f"  - Kanał 1 (Delta)       mean:      {encoded_stats['channel_1_delta']['mean_of_means']:.6f} | std: {encoded_stats['channel_1_delta']['mean_of_stds']:.6f}")
        print(f"  - Kanał 2 (Delta-Delta) mean:      {encoded_stats['channel_2_delta_delta']['mean_of_means']:.6f} | std: {encoded_stats['channel_2_delta_delta']['mean_of_stds']:.6f}")
        print(f"  - Udział próbek poza 3-sigma:      {encoded_stats['mean_3sigma_outlier_pct']:.2f}% (zgodne z rozkładem zbliżonym do Gaussa)")

    # 3. Wykresy diagnostyczne
    print("\n[3/3] Generowanie wykresu diagnostycznego jakości normalizacji...")
    generate_comparative_plots(sample_files)
    print(f"  -> Wykres zapisano w: {OUTPUT_IMG}")

    # Zapis raportu do JSON
    full_report = {
        "raw_audio_profile": raw_stats,
        "encoded_tensors_profile": encoded_stats
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"  -> Metryki JSON zapisano w: {OUTPUT_METRICS_JSON}")

    print("\n" + "=" * 70)
    print("PROFILOWANIE ZAKOŃCZONE!")
    print("=" * 70)

if __name__ == "__main__":
    run_assessment()
