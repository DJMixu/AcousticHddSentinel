#!/usr/bin/env python3
"""
Pełne 4-minutowe badanie akustyczne dysku HGST Travelstar 5K500.B 2.5" (HTS545050A7E680)
zgodne z ADR-001 i wymogami bezpieczeństwa/prywatności.
"""
import os
import sys
import time
import json
import random
import threading
import subprocess
import numpy as np
import soundfile as sf
import torch
import onnxruntime as ort
import matplotlib.pyplot as plt
from scipy.signal import welch

# Dodaj ścieżkę ml do importów DSP i modelu
sys.path.insert(0, '/home/DJMixu/Pulpit/AcousticHddSentinel')
from ml.hdd_audio_encoder import encode_audio_chunk
from ml.model_cnn import AcousticHddSentinelCNN

OUTPUT_DIR = '/home/DJMixu/Pulpit/AcousticHddSentinel/tests/recordings_4min_study/HGST'
os.makedirs(OUTPUT_DIR, exist_ok=True)

PARTITION_DEV = '/dev/sdb2'
MOUNT_DIR = '/run/media/DJMixu/Nowy'
TEMP_FILE_PATH = os.path.join(MOUNT_DIR, '.benchmark_stress_temp.bin')

# Flaga kontrolna dla wątków obciążeniowych
stop_stress = threading.Event()

def run_write_stress():
    """Wątek generujący realne zapisy i ruchy siłownika VCM."""
    try:
        with open(TEMP_FILE_PATH, 'wb') as f:
            chunk = os.urandom(64 * 1024) # 64 KB
            while not stop_stress.is_set():
                f.write(chunk)
                f.flush()
                os.fdatasync(f.fileno())
                time.sleep(random.uniform(0.01, 0.04))
    except Exception as e:
        print(f"[Stress Write] Info: {e}")

def run_read_stress():
    """Wątek generujący losowe przeskoki głowic (seek read) bez cache."""
    try:
        with open(TEMP_FILE_PATH, 'wb') as f:
            f.write(os.urandom(30 * 1024 * 1024))
            f.flush()
            os.fdatasync(f.fileno())
        
        file_size = 30 * 1024 * 1024
        with open(TEMP_FILE_PATH, 'rb') as f:
            fd = f.fileno()
            while not stop_stress.is_set():
                pos = random.randint(0, file_size - 64 * 1024)
                f.seek(pos)
                os.posix_fadvise(fd, pos, 64 * 1024, os.POSIX_FADV_DONTNEED)
                _ = f.read(random.choice([4096, 16384, 65536]))
                time.sleep(random.uniform(0.005, 0.025))
    except Exception as e:
        print(f"[Stress Read] Info: {e}")

def run_mixed_stress():
    """Wątek mieszany: naprzemienny losowy odczyt i zapis."""
    try:
        file_size = 30 * 1024 * 1024
        with open(TEMP_FILE_PATH, 'wb') as f:
            f.write(os.urandom(file_size))
            f.flush()
            os.fdatasync(f.fileno())
            
        with open(TEMP_FILE_PATH, 'r+b') as f:
            fd = f.fileno()
            while not stop_stress.is_set():
                pos = random.randint(0, file_size - 64 * 1024)
                f.seek(pos)
                os.posix_fadvise(fd, pos, 64 * 1024, os.POSIX_FADV_DONTNEED)
                if random.random() < 0.5:
                    _ = f.read(random.choice([4096, 16384, 32768]))
                else:
                    f.write(os.urandom(random.choice([4096, 16384, 32768])))
                    f.flush()
                    os.fdatasync(fd)
                time.sleep(random.uniform(0.008, 0.03))
    except Exception as e:
        print(f"[Stress Mixed] Info: {e}")

def record_audio_phase(phase_name, duration_sec=60):
    """Nagrywa audio bezpośrednio przez ALSA kernel plughw:2,0 (22050 Hz mono)."""
    raw_tmp = f'/tmp/hgst_raw_{phase_name}.wav'
    out_path = os.path.join(OUTPUT_DIR, f'HGST_5400rpm_healthy_{phase_name}_60s.wav')
    
    print(f"\n[Nagrywanie ALSA] Faza: {phase_name.upper()} ({duration_sec} sekund)...")
    record_dur = duration_sec + 1
    cmd = ['arecord', '-D', 'plughw:2,0', '-f', 'S16_LE', '-c', '1', '-r', '22050', '-d', str(record_dur), raw_tmp]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    data, sr = sf.read(raw_tmp)
    start_idx = int(1.0 * sr)
    target_samples = duration_sec * sr
    trimmed_data = data[start_idx : start_idx + target_samples]
    
    sf.write(out_path, trimmed_data, sr)
    if os.path.exists(raw_tmp):
        os.remove(raw_tmp)
        
    rms = np.sqrt(np.mean(trimmed_data**2))
    peak = np.max(np.abs(trimmed_data))
    print(f"[Zapisano] {out_path}")
    print(f"  Próbki: {len(trimmed_data)} ({len(trimmed_data)/sr:.1f}s), RMS: {rms:.6f} ({20*np.log10(rms+1e-12):.1f} dBFS), Peak: {peak:.6f}")
    return out_path, trimmed_data

def mount_disk():
    print("[Dysk] Montowanie /dev/sdb2...")
    subprocess.run(['udisksctl', 'mount', '-b', PARTITION_DEV], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(0.5)

def unmount_disk():
    print("[Dysk] Usuwanie plików testowych i odmontowywanie...")
    if os.path.exists(TEMP_FILE_PATH):
        try:
            os.remove(TEMP_FILE_PATH)
        except:
            pass
    subprocess.run(['sync'])
    subprocess.run(['udisksctl', 'unmount', '-b', PARTITION_DEV], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def main():
    print("=" * 70)
    print("ROZPOCZĘCIE 4-MINUTOWEGO BADANIA AKUSTYCZNEGO HGST TRAVELSTAR 5K500.B")
    print("=" * 70)
    
    phases_data = {}
    phases = ['idle', 'write', 'read', 'mixed']
    all_exist = all(os.path.exists(os.path.join(OUTPUT_DIR, f'HGST_5400rpm_healthy_{p}_60s.wav')) for p in phases)
    
    if all_exist:
        print("\n[Info] Znaleziono gotowe nagrania 4 faz na dysku. Wczytywanie...")
        for p in phases:
            data, sr_read = sf.read(os.path.join(OUTPUT_DIR, f'HGST_5400rpm_healthy_{p}_60s.wav'))
            phases_data[p] = data
            print(f"  Załadowano {p}: {len(data)} próbek ({len(data)/sr_read:.1f}s)")
    else:
        try:
            # FAZA 1: IDLE
            print("\n--- 1/4: FAZA SPOCZYNKU (IDLE) ---")
            p_idle, d_idle = record_audio_phase('idle', 60)
            phases_data['idle'] = d_idle
            
            # FAZA 2: WRITE
            print("\n--- 2/4: FAZA ZAPISU (WRITE STRESS) ---")
            mount_disk()
            stop_stress.clear()
            t_stress = threading.Thread(target=run_write_stress)
            t_stress.start()
            time.sleep(1.0)
            p_write, d_write = record_audio_phase('write', 60)
            stop_stress.set()
            t_stress.join()
            if os.path.exists(TEMP_FILE_PATH):
                os.remove(TEMP_FILE_PATH)
            phases_data['write'] = d_write
            
            # FAZA 3: READ
            print("\n--- 3/4: FAZA ODCZYTU (READ / RANDOM SEEK) ---")
            stop_stress.clear()
            t_stress = threading.Thread(target=run_read_stress)
            t_stress.start()
            time.sleep(1.0)
            p_read, d_read = record_audio_phase('read', 60)
            stop_stress.set()
            t_stress.join()
            if os.path.exists(TEMP_FILE_PATH):
                os.remove(TEMP_FILE_PATH)
            phases_data['read'] = d_read
            
            # FAZA 4: MIXED
            print("\n--- 4/4: FAZA MIESZANA (MIXED I/O) ---")
            stop_stress.clear()
            t_stress = threading.Thread(target=run_mixed_stress)
            t_stress.start()
            time.sleep(1.0)
            p_mixed, d_mixed = record_audio_phase('mixed', 60)
            stop_stress.set()
            t_stress.join()
            phases_data['mixed'] = d_mixed
            
        finally:
            unmount_disk()
            
    print("\n" + "=" * 70)
    print("NAGRANIA UKOŃCZONE. ROZPOCZĘCIE EWALUACJI MODELI I DSP...")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_pt = AcousticHddSentinelCNN(num_classes=4).to(device)
    ckpt = torch.load('/home/DJMixu/Pulpit/AcousticHddSentinel/training_output/best_hdd_cnn_model.pt', map_location=device)
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        model_pt.load_state_dict(ckpt['state_dict'], strict=False)
    else:
        model_pt.load_state_dict(ckpt, strict=False)
    model_pt.eval()
    
    onnx_session = ort.InferenceSession('/home/DJMixu/Pulpit/AcousticHddSentinel/exported_models/best_hdd_cnn_model.onnx', providers=['CPUExecutionProvider'])
    
    classes = ['beeping', 'clicking', 'normal', 'scratching']
    results = {'phases': {}, 'summary': {}}
    
    sr = 22050
    chunk_samples = 3 * sr # 66 150
    
    total_pt_normal = 0
    total_onnx_normal = 0
    total_windows = 0
    
    window_predictions = []
    
    for phase_name, audio in phases_data.items():
        n_chunks = len(audio) // chunk_samples
        phase_pt_preds = []
        phase_onnx_preds = []
        phase_pt_confs = []
        phase_onnx_confs = []
        
        for i in range(n_chunks):
            chunk = audio[i * chunk_samples : (i + 1) * chunk_samples]
            tensor_np = encode_audio_chunk(chunk, sr=sr) # (128, 259, 3)
            
            # --- PyTorch Eval ---
            tensor_pt = torch.from_numpy(tensor_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                logits = model_pt(tensor_pt)
                probs_pt = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_pt_idx = int(np.argmax(probs_pt))
                pred_pt_cls = classes[pred_pt_idx]
                conf_pt = float(probs_pt[pred_pt_idx])
                
            # --- ONNX Eval ---
            tensor_onnx = np.transpose(tensor_np, (2, 0, 1))[np.newaxis, ...].astype(np.float32)
            onnx_inputs = {onnx_session.get_inputs()[0].name: tensor_onnx}
            onnx_out = onnx_session.run(None, onnx_inputs)[0]
            exp_out = np.exp(onnx_out - np.max(onnx_out, axis=1, keepdims=True))
            probs_onnx = (exp_out / np.sum(exp_out, axis=1, keepdims=True))[0]
            pred_onnx_idx = int(np.argmax(probs_onnx))
            pred_onnx_cls = classes[pred_onnx_idx]
            conf_onnx = float(probs_onnx[pred_onnx_idx])
            
            phase_pt_preds.append(pred_pt_cls)
            phase_pt_confs.append(conf_pt)
            phase_onnx_preds.append(pred_onnx_cls)
            phase_onnx_confs.append(conf_onnx)
            
            if pred_pt_cls == 'normal': total_pt_normal += 1
            if pred_onnx_cls == 'normal': total_onnx_normal += 1
            total_windows += 1
            
            window_predictions.append({
                'phase': phase_name,
                'chunk_idx': i,
                'time_start': i * 3.0,
                'time_end': (i + 1) * 3.0,
                'pt_pred': pred_pt_cls,
                'pt_conf': conf_pt,
                'pt_probs': {classes[c]: float(probs_pt[c]) for c in range(4)},
                'onnx_pred': pred_onnx_cls,
                'onnx_conf': conf_onnx,
                'onnx_probs': {classes[c]: float(probs_onnx[c]) for c in range(4)}
            })
            
        results['phases'][phase_name] = {
            'n_chunks': n_chunks,
            'pt_normal_count': phase_pt_preds.count('normal'),
            'pt_accuracy': phase_pt_preds.count('normal') / n_chunks,
            'pt_mean_conf': float(np.mean(phase_pt_confs)),
            'onnx_normal_count': phase_onnx_preds.count('normal'),
            'onnx_accuracy': phase_onnx_preds.count('normal') / n_chunks,
            'onnx_mean_conf': float(np.mean(phase_onnx_confs)),
            'pt_predictions': phase_pt_preds,
            'onnx_predictions': phase_onnx_preds
        }
        
    results['summary'] = {
        'total_windows': total_windows,
        'pt_total_accuracy': total_pt_normal / total_windows,
        'onnx_total_accuracy': total_onnx_normal / total_windows,
        'false_alarms_pt': total_windows - total_pt_normal,
        'false_alarms_onnx': total_windows - total_onnx_normal
    }
    
    report_json_path = os.path.join(OUTPUT_DIR, 'evaluation_report.json')
    with open(report_json_path, 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'window_details': window_predictions}, f, indent=2)
    print(f"\n[Raport JSON] Zapisano w {report_json_path}")
    
    # --- WIZUALIZACJA ---
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    
    # 1. Oscylogram 4 minut
    full_audio = np.concatenate([phases_data['idle'], phases_data['write'], phases_data['read'], phases_data['mixed']])
    t_full = np.arange(len(full_audio)) / sr
    axes[0].plot(t_full, full_audio, color='#1f77b4', lw=0.5)
    axes[0].set_title('1. Rzeczywisty oscylogram 4-minutowego badania HGST Travelstar 2.5" (HTS545050A7E680)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Amplituda [-1, 1]')
    axes[0].set_xlim(0, 240)
    for p_idx, (p_name, col) in enumerate([('IDLE (0-60s)', 'gray'), ('WRITE (60-120s)', 'red'), ('READ (120-180s)', 'blue'), ('MIXED (180-240s)', 'purple')]):
        axes[0].axvspan(p_idx * 60, (p_idx + 1) * 60, color=col, alpha=0.1, label=p_name)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)
    
    # 2. Spektrogram całościowy (0 - 2000 Hz)
    Pxx, freqs_sp, bins_sp, im = axes[1].specgram(full_audio, NFFT=2048, Fs=sr, noverlap=1024, cmap='viridis')
    axes[1].set_title('2. Spektrogram czasowo-częstotliwościowy (0 - 2000 Hz) — Praca silnika 5400 RPM i impulsy głowic', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Częstotliwość [Hz]')
    axes[1].set_ylim(0, 2000)
    axes[1].set_xlim(0, 240)
    plt.colorbar(im, ax=axes[1], format='%+2.0f dB', pad=0.01, label='Moc [dBFS]')
    
    # 3. Welch PSD
    for p_name, col in [('idle', '#2ca02c'), ('write', '#d62728'), ('read', '#1f77b4'), ('mixed', '#9467bd')]:
        f_p, p_p = welch(phases_data[p_name], sr, nperseg=8192)
        mask = (f_p >= 20) & (f_p <= 2000)
        axes[2].plot(f_p[mask], 10 * np.log10(p_p[mask] + 1e-12), label=f'Faza {p_name.upper()} (RMS: {np.sqrt(np.mean(phases_data[p_name]**2)):.4f})', color=col, lw=1.2)
    axes[2].set_title('3. Gęstość widmowa mocy (Welch PSD) — Wyraźna podstawa rotacji 91.5 Hz i profil obrotowy', fontsize=11, fontweight='bold')
    axes[2].set_xlabel('Częstotliwość [Hz]')
    axes[2].set_ylabel('Moc [dBFS]')
    axes[2].set_xlim(20, 2000)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right', fontsize=9)
    
    # 4. Trajektoria pewności
    confs_pt = [w['pt_conf'] * 100 for w in window_predictions]
    times_w = [w['time_start'] + 1.5 + (['idle','write','read','mixed'].index(w['phase'])*60) for w in window_predictions]
    classes_w = [w['pt_pred'] for w in window_predictions]
    
    colors_map = {'normal': '#2ca02c', 'clicking': '#d62728', 'scratching': '#ff7f0e', 'beeping': '#1f77b4'}
    for i, (t_w, c_w, p_cls) in enumerate(zip(times_w, confs_pt, classes_w)):
        axes[3].scatter(t_w, c_w, color=colors_map[p_cls], s=35, zorder=3)
    axes[3].plot(times_w, confs_pt, color='#2ca02c', alpha=0.5, linestyle=':', lw=1.0)
    axes[3].set_title(f'4. Trajektoria pewności klasyfikacji modelu produkcyjnego PyTorch ({total_pt_normal}/{total_windows} = {total_pt_normal/total_windows*100:.1f}% normal)', fontsize=11, fontweight='bold')
    axes[3].set_xlabel('Czas badania [s]')
    axes[3].set_ylabel('Pewność predykcji [%]')
    axes[3].set_ylim(0, 105)
    axes[3].set_xlim(0, 240)
    axes[3].axhline(50, color='red', linestyle='--', alpha=0.5, label='Próg 50%')
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc='lower left', fontsize=9)
    
    plt.tight_layout()
    chart_path = os.path.join(OUTPUT_DIR, 'hgst_acoustic_study_report.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[Wykres] Zapisano w {chart_path}")
    
    print("\n" + "=" * 70)
    print("PODSUMOWANIE METRYK MODELI DLA HGST TRAVELSTAR:")
    print(f"  PyTorch Model Produkcyjny: {total_pt_normal}/{total_windows} ({total_pt_normal/total_windows*100:.2f}%) trafień 'normal'")
    print(f"  ONNX Model Bazowy:        {total_onnx_normal}/{total_windows} ({total_onnx_normal/total_windows*100:.2f}%) trafień 'normal'")
    print(f"  Fałszywe alarmy PyTorch:  {total_windows - total_pt_normal}")
    print(f"  Fałszywe alarmy ONNX:     {total_windows - total_onnx_normal}")
    print("=" * 70)

if __name__ == '__main__':
    main()
