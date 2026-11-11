"""
HDD Acoustic Sentinel - 1h/24h Continuous Production Stress Test
Single Source of Truth: ADR-001 & AGENTS.md

Moduł testów obciążeniowych weryfikujący:
1. Pamięć RAM / wycieki zasobów (Memory Leaks via RSS monitoring).
2. Odporność na dane skrajne i zdegenerowane (Adversarial & Edge Cases: cisza, clipping, szum, impulsy).
3. Latencję w czasie rzeczywistym (Real-time latency: DSP + CNN inference p50/p95/p99).
4. Determinizm i brak dryfu predykcji (Canary chunk drift).
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Dodanie ścieżek
PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "ml") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "ml"))
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

def get_process_memory_mb() -> float:
    """
    Odczytuje zużycie pamięci RSS procesu w MB (działa niezależnie przez /proc lub psutil).
    """
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        # Format: VmRSS:     12345 kB
                        parts = line.split()
                        return float(parts[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


def generate_edge_case_audio(case_type: str, sr: int = 22050, duration: float = 3.0) -> np.ndarray:
    """
    Generuje skrajne, zdegenerowane próbki audio do testowania odporności potoku.
    """
    num_samples = int(sr * duration)
    if case_type == "silence":
        # Całkowite cyfrowe zero (test dzielenia przez zero / eps)
        return np.zeros(num_samples, dtype=np.float32)
    elif case_type == "clipping":
        # Maksymalny clipping (prostokątny sygnał przesterowany)
        t = np.linspace(0, duration, num_samples, endpoint=False)
        return np.sign(np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    elif case_type == "extreme_noise":
        # Agresywny biały szum o dużej energii
        return np.clip(np.random.randn(num_samples) * 0.8, -1.0, 1.0).astype(np.float32)
    elif case_type == "dirac_impulse":
        # Pojedynczy ostry impuls (np. trzask elektryczny w mikrofonie)
        sig = np.zeros(num_samples, dtype=np.float32)
        sig[num_samples // 2] = 1.0
        return sig
    elif case_type == "dc_offset_heavy":
        # Sygnał ze skrajnym przesunięciem DC (+0.6)
        return (np.sin(2 * np.pi * 200 * np.linspace(0, duration, num_samples)) * 0.3 + 0.6).astype(np.float32)
    elif case_type == "short_burst":
        # Niepełny bufor (np. 1.2s zamiast 3.0s)
        return (np.random.randn(int(sr * 1.2)) * 0.1).astype(np.float32)
    else:
        return np.random.randn(num_samples).astype(np.float32) * 0.05


def run_stress_test(
    duration_seconds: float,
    model_path: Path,
    dataset_dir: Path,
    output_dir: Path,
    log_interval_sec: float = 60.0
):
    import torch
    from model_cnn import build_hdd_cnn_model
    from hdd_audio_encoder import (
        encode_audio_chunk,
        normalize_waveform_peak,
        normalize_tensor_zscore,
        TARGET_SR
    )
    import librosa
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ROZPOCZĘCIE TESTU STRESU / CIĄGŁEGO OBCIĄŻENIA (STRESS TEST - PYTORCH)")
    print("=" * 70)
    print(f"Czas trwania:     {duration_seconds / 3600:.2f} godz. ({duration_seconds:.0f} s)")
    print(f"Model:            {model_path}")
    print(f"Katalog wyników:  {output_dir}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Urządzenie:       {device}")
    print(f"Pamięć startowa:  {get_process_memory_mb():.2f} MB")

    # Wczytanie modelu
    if not model_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku wag modelu: {model_path}")

    print("\nŁadowanie modelu PyTorch...")
    checkpoint = torch.load(model_path, weights_only=True)
    model = build_hdd_cnn_model(num_classes=4)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Wczytanie listy plików audio
    wav_files = sorted(list(dataset_dir.glob("*.wav"))) if dataset_dir.exists() else []
    print(f"Znaleziono {len(wav_files)} nagrań referencyjnych w {dataset_dir}.")

    # Przygotowanie próbki kontrolnej (Canary Chunk) do badania dryfu determinizmu
    if wav_files:
        canary_audio, _ = librosa.load(str(wav_files[0]), sr=TARGET_SR, mono=True)
        canary_chunk = canary_audio[:int(TARGET_SR * 3.0)]
        if len(canary_chunk) < int(TARGET_SR * 3.0):
            canary_chunk = np.pad(canary_chunk, (0, int(TARGET_SR * 3.0) - len(canary_chunk)), mode="reflect")
    else:
        canary_chunk = np.random.randn(int(TARGET_SR * 3.0)).astype(np.float32) * 0.05

    # Baseline predykcji Canary Chunk
    canary_tensor = encode_audio_chunk(
        y_chunk=canary_chunk,
        sr=TARGET_SR,
        waveform_norm_fn=normalize_waveform_peak,
        tensor_norm_fn=normalize_tensor_zscore,
        tensor_format="tf"
    )
    canary_batch = torch.from_numpy(canary_tensor).unsqueeze(0).to(device)
    with torch.no_grad():
        canary_baseline_preds = model(canary_batch).cpu().numpy()[0]
    print(f"Predykcja bazowa Canary Chunk: {np.round(canary_baseline_preds, 4)}")

    # Metryki profilowania
    time_series_timestamps: List[float] = []
    time_series_ram_mb: List[float] = []
    time_series_canary_drift: List[float] = []
    dsp_latencies_ms: List[float] = []
    cnn_latencies_ms: List[float] = []
    total_latencies_ms: List[float] = []

    edge_cases_tested = {
        "silence": {"count": 0, "nans": 0, "infs": 0, "errors": 0},
        "clipping": {"count": 0, "nans": 0, "infs": 0, "errors": 0},
        "extreme_noise": {"count": 0, "nans": 0, "infs": 0, "errors": 0},
        "dirac_impulse": {"count": 0, "nans": 0, "infs": 0, "errors": 0},
        "dc_offset_heavy": {"count": 0, "nans": 0, "infs": 0, "errors": 0},
        "short_burst": {"count": 0, "nans": 0, "infs": 0, "errors": 0},
    }

    edge_case_types = list(edge_cases_tested.keys())
    edge_idx = 0

    start_time = time.time()
    last_log_time = start_time
    total_chunks_processed = 0
    wav_idx = 0

    # Rozgrzewka (Warmup)
    for _ in range(10):
        with torch.no_grad():
            _ = model(canary_batch)

    print("\n" + "=" * 70)
    print("START PĘTLI TESTU STRESU")
    print("=" * 70)

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= duration_seconds:
                break

            total_chunks_processed += 1

            # Co 40 iteracji wstrzykujemy przypadek skrajny (Edge Case)
            if total_chunks_processed % 40 == 0:
                c_type = edge_case_types[edge_idx % len(edge_case_types)]
                edge_idx += 1
                audio_input = generate_edge_case_audio(c_type, sr=TARGET_SR, duration=3.0)
                is_edge_case = True
                active_edge_type = c_type
            else:
                is_edge_case = False
                active_edge_type = None
                if wav_files:
                    fpath = wav_files[wav_idx % len(wav_files)]
                    wav_idx += 1
                    try:
                        y_raw, _ = librosa.load(str(fpath), sr=TARGET_SR, mono=True)
                        if len(y_raw) >= int(TARGET_SR * 3.0):
                            audio_input = y_raw[:int(TARGET_SR * 3.0)]
                        else:
                            audio_input = np.pad(y_raw, (0, int(TARGET_SR * 3.0) - len(y_raw)), mode="reflect")
                    except Exception:
                        audio_input = np.random.randn(int(TARGET_SR * 3.0)).astype(np.float32) * 0.05
                else:
                    audio_input = np.random.randn(int(TARGET_SR * 3.0)).astype(np.float32) * 0.05

            # 1. Pomiar czasu DSP
            t0 = time.perf_counter()
            try:
                # Wyrównanie długości w razie potrzeby
                if len(audio_input) < int(TARGET_SR * 3.0):
                    audio_input = np.pad(audio_input, (0, int(TARGET_SR * 3.0) - len(audio_input)), mode="reflect")
                else:
                    audio_input = audio_input[:int(TARGET_SR * 3.0)]

                tensor = encode_audio_chunk(
                    y_chunk=audio_input,
                    sr=TARGET_SR,
                    waveform_norm_fn=normalize_waveform_peak,
                    tensor_norm_fn=normalize_tensor_zscore,
                    tensor_format="tf"
                )
            except Exception as e:
                if is_edge_case:
                    edge_cases_tested[active_edge_type]["errors"] += 1
                continue

            t1 = time.perf_counter()

            # 2. Pomiar czasu inferencji CNN
            batch_t = torch.from_numpy(tensor).unsqueeze(0).to(device)
            with torch.no_grad():
                preds = model(batch_t).cpu().numpy()[0]
            t2 = time.perf_counter()

            dsp_ms = (t1 - t0) * 1000.0
            cnn_ms = (t2 - t1) * 1000.0
            total_ms = (t2 - t0) * 1000.0

            dsp_latencies_ms.append(dsp_ms)
            cnn_latencies_ms.append(cnn_ms)
            total_latencies_ms.append(total_ms)

            # Weryfikacja numeryczna (NaN / Inf)
            has_nan = bool(np.isnan(preds).any() or np.isnan(tensor).any())
            has_inf = bool(np.isinf(preds).any() or np.isinf(tensor).any())

            if is_edge_case:
                edge_cases_tested[active_edge_type]["count"] += 1
                if has_nan:
                    edge_cases_tested[active_edge_type]["nans"] += 1
                if has_inf:
                    edge_cases_tested[active_edge_type]["infs"] += 1

            # Okresowy test Canary Chunk (co 50 chunks)
            if total_chunks_processed % 50 == 0:
                with torch.no_grad():
                    canary_curr_preds = model(canary_batch).cpu().numpy()[0]
                drift = float(np.max(np.abs(canary_curr_preds - canary_baseline_preds)))
                current_ram = get_process_memory_mb()

                time_series_timestamps.append(elapsed)
                time_series_ram_mb.append(current_ram)
                time_series_canary_drift.append(drift)

            # Raportowanie w konsoli co log_interval_sec
            if now - last_log_time >= log_interval_sec:
                last_log_time = now
                curr_ram = get_process_memory_mb()
                p50_lat = np.percentile(total_latencies_ms[-500:], 50) if total_latencies_ms else 0.0
                p95_lat = np.percentile(total_latencies_ms[-500:], 95) if total_latencies_ms else 0.0
                p99_lat = np.percentile(total_latencies_ms[-500:], 99) if total_latencies_ms else 0.0
                progress_pct = (elapsed / duration_seconds) * 100

                print(
                    f"[{progress_pct:5.1f}% | {elapsed/60:5.1f} min] "
                    f"Przetworzono: {total_chunks_processed:6d} chunków | "
                    f"RAM: {curr_ram:6.1f} MB | "
                    f"Latencja (p50/p95/p99): {p50_lat:.1f}/{p95_lat:.1f}/{p99_lat:.1f} ms"
                )

    except KeyboardInterrupt:
        print("\n[ZATRZYMANIE] Test przerwany ręcznie przez użytkownika.")

    total_test_time = time.time() - start_time
    final_ram_mb = get_process_memory_mb()
    start_ram_mb = time_series_ram_mb[0] if time_series_ram_mb else final_ram_mb
    ram_growth_mb = final_ram_mb - start_ram_mb

    print("\n" + "=" * 70)
    print("PODSUMOWANIE WYNIKÓW TESTU STRESU")
    print("=" * 70)
    print(f"Rzeczywisty czas testu:        {total_test_time/60:.2f} minut ({total_test_time:.1f} s)")
    print(f"Łącznie przetworzonych chunków:{total_chunks_processed}")
    print(f"Przepustowość (Throughput):    {total_chunks_processed / total_test_time:.2f} chunków/s")
    print(f"Pamięć RAM (Start -> Koniec):  {start_ram_mb:.2f} MB -> {final_ram_mb:.2f} MB (Delta: {ram_growth_mb:+.2f} MB)")
    print(f"Latencja łączna (Średnia):     {np.mean(total_latencies_ms):.2f} ms")
    print(f"Latencja p50 / p95 / p99:      {np.percentile(total_latencies_ms, 50):.2f} ms / {np.percentile(total_latencies_ms, 95):.2f} ms / {np.percentile(total_latencies_ms, 99):.2f} ms")
    print(f"Maksymalny dryf Canary Chunk:  {np.max(time_series_canary_drift) if time_series_canary_drift else 0.0:.2e}")

    print("\nOdporność na przypadki skrajne (Edge Cases):")
    for c_name, stats in edge_cases_tested.items():
        print(f"  - {c_name:18s}: zbadano {stats['count']:3d} | NaN: {stats['nans']} | Inf: {stats['infs']} | Błędy: {stats['errors']}")

    # Zapis raportu do JSON
    report_dict = {
        "duration_seconds": total_test_time,
        "total_chunks_processed": total_chunks_processed,
        "throughput_chunks_per_sec": total_chunks_processed / total_test_time if total_test_time > 0 else 0,
        "ram_memory_mb": {
            "initial": start_ram_mb,
            "final": final_ram_mb,
            "growth": ram_growth_mb,
            "max": float(np.max(time_series_ram_mb)) if time_series_ram_mb else final_ram_mb
        },
        "latencies_ms": {
            "dsp_mean": float(np.mean(dsp_latencies_ms)),
            "cnn_mean": float(np.mean(cnn_latencies_ms)),
            "total_mean": float(np.mean(total_latencies_ms)),
            "total_p50": float(np.percentile(total_latencies_ms, 50)),
            "total_p95": float(np.percentile(total_latencies_ms, 95)),
            "total_p99": float(np.percentile(total_latencies_ms, 99)),
            "total_max": float(np.max(total_latencies_ms))
        },
        "max_canary_drift": float(np.max(time_series_canary_drift)) if time_series_canary_drift else 0.0,
        "edge_cases": edge_cases_tested
    }

    report_json_path = output_dir / "stress_test_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Artefakt] Zapisano raport numeryczny: {report_json_path}")

    # Wygenerowanie wykresu diagnostycznego
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Wykres RAM w czasie
    if time_series_timestamps:
        time_minutes = np.array(time_series_timestamps) / 60.0
        ax1.plot(time_minutes, time_series_ram_mb, color="crimson", linewidth=1.5, label="Pamięć RSS (MB)")
        ax1.axhline(start_ram_mb, color="black", linestyle="--", alpha=0.5, label="Baza startowa")
        ax1.set_title("Profil pamięci RAM procesu w czasie (Brak wycieków = płaska linia)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Czas testu (minuty)")
        ax1.set_ylabel("Pamięć RSS (MB)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Histogram latencji
    if total_latencies_ms:
        ax2.hist(total_latencies_ms, bins=50, color="royalblue", alpha=0.8, edgecolor="black")
        p95 = np.percentile(total_latencies_ms, 95)
        ax2.axvline(p95, color="darkorange", linestyle="--", linewidth=2, label=f"p95 = {p95:.1f} ms")
        ax2.set_title("Rozkład latencji inferencji (DSP + CNN forward pass)", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Czas przetwarzania 3s okna (ms)")
        ax2.set_ylabel("Liczba próbek")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "stress_test_report.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"[Artefakt] Wykres testu stresu zapisano w: {chart_path}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="Test obciążeniowy modelu HDD Acoustic Sentinel.")
    parser.add_argument("--hours", type=float, default=1.0, help="Czas trwania w godzinach (domyślnie: 1.0)")
    parser.add_argument("--seconds", type=float, default=None, help="Opcjonalny czas trwania w sekundach (nadpisuje --hours)")
    parser.add_argument("--model-path", type=str, default="training_output/best_hdd_cnn_model.pt", help="Ścieżka do modelu PyTorch (.pt)")
    default_dataset = "data/raw/aesonlabs" if Path("data/raw/aesonlabs").exists() else ("data/raw" if Path("data/raw").exists() else "dataset")
    parser.add_argument("--dataset-dir", type=str, default=default_dataset, help="Katalog z surowymi plikami WAV")
    parser.add_argument("--output-dir", type=str, default="stress_test_output", help="Katalog na wyniki testu")
    parser.add_argument("--log-interval", type=float, default=60.0, help="Interwał logowania w konsoli (sekundy)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total_sec = args.seconds if args.seconds is not None else args.hours * 3600.0
    run_stress_test(
        duration_seconds=total_sec,
        model_path=Path(args.model_path),
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        log_interval_sec=args.log_interval
    )
