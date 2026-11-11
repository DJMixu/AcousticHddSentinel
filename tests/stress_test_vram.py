"""
HDD Acoustic Sentinel - VRAM-Resident GPU Stress Test
Single Source of Truth: ADR-001 & AGENTS.md

Moduł testów obciążeniowych VRAM:
1. Wszystkie zdekodowane tensory (128, 259, 3) oraz przypadki skrajne są wstępnie załadowane w 100% do pamięci VRAM karty graficznej AMD Radeon RX 9060 XT.
2. Zero obciążenia CPU (brak librosa.load, brak dyskowego I/O, brak FFT w pętli).
3. Pętla inferencji przetwarza duże paczki (Batch: 64/128) bezpośrednio z pamięci VRAM GPU, maksymalizując wykorzystanie procesorów strumieniowych (Compute Units) karty graficznej.
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

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
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        try:
            with open("/proc/self/status", "r") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        return float(line.split()[1]) / 1024.0
        except Exception:
            return 0.0
    return 0.0


def create_pre_encoded_edge_cases() -> np.ndarray:
    """
    Tworzy zbiór przetestowanych matematycznie przypadków skrajnych w kształcie (128, 259, 3)
    gotowych do załadowania do VRAM.
    """
    cases = []
    # 1. Cyfrowe zero (Cisza)
    cases.append(np.zeros((128, 259, 3), dtype=np.float32))
    # 2. Skrajny przesterowany sygnał (Clipping)
    cases.append(np.ones((128, 259, 3), dtype=np.float32) * 5.0)
    # 3. Silny szum o wysokiej wariancji
    cases.append(np.random.randn(128, 259, 3).astype(np.float32) * 4.0)
    # 4. Impuls Diraca w spektrogramie
    imp = np.zeros((128, 259, 3), dtype=np.float32)
    imp[64, 129, :] = 10.0
    cases.append(imp)
    # 5. Silne przesunięcie DC
    cases.append(np.ones((128, 259, 3), dtype=np.float32) * -3.0)
    return np.array(cases, dtype=np.float32)


def load_all_tensors_to_ram(data_dir: Path, target_pool_size: int = 5000) -> np.ndarray:
    """
    Wczytuje wszystkie pliki .npy do pamięci RAM, powiela je do zadanego rozmiaru bufora (np. 5000 tensorów = ~2 GB)
    i dołącza przypadki skrajne.
    """
    npy_files = sorted(list(data_dir.rglob("*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"Nie znaleziono tensorów .npy w {data_dir}. Uruchom hdd_audio_encoder.py.")

    print(f"Wczytywanie {len(npy_files)} plików .npy z dysku do bufora RAM...")
    raw_list = [np.load(f).astype(np.float32) for f in npy_files]
    base_arr = np.array(raw_list, dtype=np.float32)

    # Dołączenie przypadków skrajnych
    edge_cases = create_pre_encoded_edge_cases()
    combined = np.concatenate([base_arr, edge_cases], axis=0)

    # Rozszerzenie bufora do zadanego rozmiaru VRAM (np. 5000 tensorów)
    if len(combined) < target_pool_size:
        repeats = int(np.ceil(target_pool_size / len(combined)))
        pool = np.tile(combined, (repeats, 1, 1, 1))[:target_pool_size]
    else:
        pool = combined[:target_pool_size]

    return pool


def run_vram_stress_test(
    duration_seconds: float,
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    batch_size: int = 64,
    pool_size: int = 5000,
    log_interval_sec: float = 60.0
):
    import torch
    from model_cnn import build_hdd_cnn_model
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_csv_path = output_dir / "stress_test_timeline.csv"
    with open(timeline_csv_path, "w", encoding="utf-8") as f:
        f.write("elapsed_sec,elapsed_min,batches,chunks,throughput_fps,p50_ms,p95_ms,ram_mb,nan_count,inf_count\n")

    print("=" * 70)
    print("VRAM-RESIDENT GPU STRESS TEST (PYTORCH)")
    print("=" * 70)
    print(f"Czas trwania:     {duration_seconds / 3600:.2f} godz. ({duration_seconds:.0f} s)")
    print(f"Rozmiar Batcha:   {batch_size} tensorów na krok GPU")
    print(f"Docelowy bufor:   {pool_size} tensorów w VRAM")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Aktywne urządzenie: {device}")

    # 1. Wczytanie modelu PyTorch
    print("\n[1/3] Ładowanie modelu PyTorch do pamięci...")
    checkpoint = torch.load(model_path, weights_only=True)
    model = build_hdd_cnn_model(num_classes=4)
    state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 2. Załadowanie bazy tensorów do RAM, a następnie alokacja w VRAM
    print("\n[2/3] Alokacja bazy danych w pamięci urządzenia...")
    t_start_load = time.time()
    ram_pool = load_all_tensors_to_ram(data_dir=data_dir, target_pool_size=pool_size)
    pool_vram_bytes = ram_pool.nbytes
    pool_vram_mb = pool_vram_bytes / (1024 * 1024)
    print(f"  -> Rozmiar puli tensorów: {ram_pool.shape} ({pool_vram_mb:.1f} MB)")

    gpu_tensor_pool = torch.from_numpy(ram_pool).float().to(device)
    canary_input = gpu_tensor_pool[:1]
    with torch.no_grad():
        canary_baseline_preds = model(canary_input).cpu().numpy()[0]

    del ram_pool
    print(f"  -> Pula tensorów pomyślnie zaalokowana na {device} w czasie {time.time() - t_start_load:.2f}s!")
    print(f"  -> Canary Chunk Baseline: {np.round(canary_baseline_preds, 4)}")
    print(f"  -> Pamięć RAM procesu (CPU): {get_process_memory_mb():.2f} MB")

    # 3. Pętla inferencji GPU
    print("\n[3/3] Start pętli testu obciążeniowego GPU...")
    print("=" * 70)

    num_samples_in_pool = int(gpu_tensor_pool.shape[0])
    num_batches_in_pool = num_samples_in_pool // batch_size
    usable_samples = num_batches_in_pool * batch_size
    print(f"  -> Użytecznych próbek: {usable_samples} ({num_batches_in_pool} stałych batchy)")

    def gpu_infer_step(batch_tensor):
        with torch.no_grad():
            predictions = model(batch_tensor)
            has_nan = torch.isnan(predictions).any().item()
            has_inf = torch.isinf(predictions).any().item()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            return has_nan, has_inf, predictions

    time_series_timestamps: List[float] = []
    time_series_ram_mb: List[float] = []
    time_series_canary_drift: List[float] = []
    sampled_latencies_ms: List[float] = []

    total_chunks_processed = 0
    total_batches_processed = 0
    total_nan_count = 0
    total_inf_count = 0

    # Rozgrzewka GPU (Warmup & Kompilacja grafu tf.function)
    print("  -> Rozgrzewka GPU i kompilacja grafu tf.function...")
    with tf.device(device_name):
        warmup_batch = gpu_tensor_pool[:batch_size]
        for _ in range(10):
            _ = gpu_infer_step(warmup_batch)

    print("  -> Graf tf.function skompilowany na GPU. Rozpoczynanie pętli ciągłej!")

    start_time = time.time()
    last_log_time = start_time

    try:
        while True:
            now = time.time()
            elapsed = now - start_time
            if elapsed >= duration_seconds:
                break

            # Indeks paczki w VRAM (rotacja bez alokacji nowych obiektów)
            b_idx = total_batches_processed % num_batches_in_pool
            batch_vram = gpu_tensor_pool[b_idx * batch_size : (b_idx + 1) * batch_size]

            # Inferencja bezpośrednio w GPU VRAM
            t0 = time.perf_counter()
            has_nan, has_inf, _ = gpu_infer_step(batch_vram)
            t1 = time.perf_counter()

            b_latency_ms = (t1 - t0) * 1000.0

            # Próbkowanie latencji (co 5 batchy) dla reprezentatywnego profilowania bez rozrostu RAM
            if total_batches_processed % 5 == 0:
                sampled_latencies_ms.append(b_latency_ms)

            if has_nan:
                total_nan_count += 1
            if has_inf:
                total_inf_count += 1

            total_chunks_processed += batch_size
            total_batches_processed += 1

            # Okresowa weryfikacja Canary Chunk i pamięci procesu CPU (co 250 batchy)
            if total_batches_processed % 250 == 0:
                with torch.no_grad():
                    canary_curr = model(canary_input).cpu().numpy()[0]
                drift = float(np.max(np.abs(canary_curr - canary_baseline_preds)))
                curr_ram = get_process_memory_mb()

                time_series_timestamps.append(elapsed)
                time_series_ram_mb.append(curr_ram)
                time_series_canary_drift.append(drift)

            # Logowanie w konsoli i do pliku co log_interval_sec
            if now - last_log_time >= log_interval_sec:
                last_log_time = now
                curr_ram = get_process_memory_mb()
                recent_latencies = sampled_latencies_ms[-200:] if sampled_latencies_ms else [b_latency_ms]
                p50_lat = np.percentile(recent_latencies, 50)
                p95_lat = np.percentile(recent_latencies, 95)
                fps = total_chunks_processed / elapsed if elapsed > 0 else 0.0
                progress_pct = (elapsed / duration_seconds) * 100

                log_line = (
                    f"[{progress_pct:5.1f}% | {elapsed/60:5.1f} min] "
                    f"GPU Batch: {total_batches_processed:7d} ({total_chunks_processed:8d} chunków) | "
                    f"Wydajność GPU: {fps:7.1f} chunków/s | "
                    f"Latencja: {p50_lat:4.2f} ms (p50) / {p95_lat:4.2f} ms (p95) | "
                    f"RAM procesu: {curr_ram:5.1f} MB | "
                    f"NaN/Inf: {total_nan_count}/{total_inf_count}"
                )
                print(log_line)
                sys.stdout.flush()

                # Zapis do CSV na żywo
                with open(timeline_csv_path, "a", encoding="utf-8") as f_csv:
                    f_csv.write(f"{elapsed:.2f},{elapsed/60:.2f},{total_batches_processed},{total_chunks_processed},{fps:.2f},{p50_lat:.2f},{p95_lat:.2f},{curr_ram:.2f},{total_nan_count},{total_inf_count}\n")

    except KeyboardInterrupt:
        print("\n[ZATRZYMANIE] Test przerwany ręcznie przez użytkownika.")

    total_test_time = time.time() - start_time
    final_ram_mb = get_process_memory_mb()
    start_ram_mb = time_series_ram_mb[0] if time_series_ram_mb else final_ram_mb
    ram_growth = final_ram_mb - start_ram_mb
    overall_fps = total_chunks_processed / total_test_time if total_test_time > 0 else 0.0

    print("\n" + "=" * 70)
    print("PODSUMOWANIE VRAM-RESIDENT GPU STRESS TEST")
    print("=" * 70)
    print(f"Rzeczywisty czas testu:        {total_test_time/60:.2f} minut ({total_test_time:.1f} s)")
    print(f"Przetworzonych paczek GPU:     {total_batches_processed}")
    print(f"Przetworzonych chunków audio:  {total_chunks_processed}")
    print(f"Maksymalna wydajność GPU:      {overall_fps:.1f} wycinków/sekundę (!)")
    print(f"Średnia latencja paczki {batch_size}:  {np.mean(sampled_latencies_ms) if sampled_latencies_ms else 0.0:.2f} ms")
    print(f"Latencja p50 / p95 / p99:      {np.percentile(sampled_latencies_ms, 50) if sampled_latencies_ms else 0.0:.2f} ms / {np.percentile(sampled_latencies_ms, 95) if sampled_latencies_ms else 0.0:.2f} ms / {np.percentile(sampled_latencies_ms, 99) if sampled_latencies_ms else 0.0:.2f} ms")
    print(f"Pamięć RAM procesu CPU:        {start_ram_mb:.2f} MB -> {final_ram_mb:.2f} MB (Delta: {ram_growth:+.2f} MB)")
    print(f"Maksymalny dryf Canary Chunk:  {np.max(time_series_canary_drift) if time_series_canary_drift else 0.0:.2e}")
    print(f"Błędy NaN / Inf:               {total_nan_count} / {total_inf_count}")

    # Zapis raportu do JSON
    report_dict = {
        "device": device_name,
        "batch_size": batch_size,
        "vram_pool_tensors": pool_size,
        "vram_pool_mb": pool_vram_mb,
        "duration_seconds": total_test_time,
        "total_batches_processed": total_batches_processed,
        "total_chunks_processed": total_chunks_processed,
        "throughput_chunks_per_sec": overall_fps,
        "ram_memory_mb": {
            "initial": start_ram_mb,
            "final": final_ram_mb,
            "growth": ram_growth
        },
        "gpu_batch_latencies_ms": {
            "mean": float(np.mean(sampled_latencies_ms)) if sampled_latencies_ms else 0.0,
            "p50": float(np.percentile(sampled_latencies_ms, 50)) if sampled_latencies_ms else 0.0,
            "p95": float(np.percentile(sampled_latencies_ms, 95)) if sampled_latencies_ms else 0.0,
            "p99": float(np.percentile(sampled_latencies_ms, 99)) if sampled_latencies_ms else 0.0,
            "max": float(np.max(sampled_latencies_ms)) if sampled_latencies_ms else 0.0
        },
        "max_canary_drift": float(np.max(time_series_canary_drift)) if time_series_canary_drift else 0.0,
        "nan_errors": total_nan_count,
        "inf_errors": total_inf_count
    }

    report_json_path = output_dir / "vram_stress_test_report.json"
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[Artefakt] Zapisano raport numeryczny: {report_json_path}")

    # Wykres diagnostyczny
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    if time_series_timestamps:
        time_min = np.array(time_series_timestamps) / 60.0
        ax1.plot(time_min, time_series_ram_mb, color="darkorange", linewidth=1.5, label="Pamięć procesu (MB)")
        ax1.set_title("Stabilność pamięci procesu (Brak wycieków = linia płaska)", fontsize=11, fontweight="bold")
        ax1.set_xlabel("Czas testu (minuty)")
        ax1.set_ylabel("RAM (MB)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    if sampled_latencies_ms:
        ax2.hist(sampled_latencies_ms, bins=50, color="crimson", alpha=0.8, edgecolor="black")
        p95 = np.percentile(sampled_latencies_ms, 95)
        ax2.axvline(p95, color="navy", linestyle="--", linewidth=2, label=f"p95 = {p95:.1f} ms")
        ax2.set_title(f"Rozkład latencji inferencji GPU (Batch = {batch_size})", fontsize=11, fontweight="bold")
        ax2.set_xlabel("Czas wykonania batcha na GPU (ms)")
        ax2.set_ylabel("Liczba próbek")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_dir / "vram_stress_test_report.png"
    plt.savefig(chart_path)
    plt.close()
    print(f"[Artefakt] Wykres testu GPU zapisano w: {chart_path}")
    print("=" * 70)


def parse_args():
    parser = argparse.ArgumentParser(description="VRAM-Resident GPU Stress Test na karcie AMD Radeon.")
    parser.add_argument("--hours", type=float, default=1.0, help="Czas trwania w godzinach (domyślnie: 1.0)")
    parser.add_argument("--seconds", type=float, default=None, help="Czas trwania w sekundach (nadpisuje --hours)")
    parser.add_argument("--model-path", type=str, default="training_output/best_hdd_cnn_model.pt", help="Ścieżka do wag modelu PyTorch (.pt)")
    parser.add_argument("--data-dir", type=str, default="dataset_encoded", help="Katalog z tensorami .npy")
    parser.add_argument("--output-dir", type=str, default="stress_test_output", help="Katalog na wyniki testu")
    parser.add_argument("--batch-size", type=int, default=64, help="Rozmiar paczki na krok GPU (domyślnie: 64)")
    parser.add_argument("--pool-size", type=int, default=5000, help="Liczba tensorów trzymanych w VRAM (domyślnie: 5000 = ~2GB)")
    parser.add_argument("--log-interval", type=float, default=60.0, help="Interwał logowania w konsoli (sekundy)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    total_sec = args.seconds if args.seconds is not None else args.hours * 3600.0
    run_vram_stress_test(
        duration_seconds=total_sec,
        model_path=Path(args.model_path),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        log_interval_sec=args.log_interval
    )
