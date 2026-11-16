import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import glob
import soundfile as sf
import numpy as np
import torch
from ml.model_cnn import AcousticHddSentinelCNN
from ml.hdd_audio_encoder import encode_audio_chunk
from ml.dsp_adversarial_guard import DSPAdversarialGuard
from ml.rpm_harmonics_analyzer import analyze_frequency_peaks

def main():
    noise_path = "tests/noise_experiment/new_ambient_mic_noise_10s.wav"
    noise_data, sr = sf.read(noise_path)
    if len(noise_data.shape) > 1:
        noise_data = noise_data[:, 0]

    model = AcousticHddSentinelCNN(num_classes=4)
    state = torch.load("training_output/best_hdd_cnn_model.pt", map_location="cpu")
    model.load_state_dict(state["state_dict"])
    model.eval()

    guard = DSPAdversarialGuard()
    chunk_len = 66150
    classes = ["beeping", "clicking", "normal", "scratching"]

    # Wybór próbek reprezentatywnych dla każdej klasy
    test_cases = [
        # 1. Sprawne dyski studyjne (healthy)
        {"class": "normal", "path": "tests/recordings_4min_study/HGST/HGST_5400rpm_healthy_read_60s.wav", "label": "HGST 5400rpm Healthy"},
        {"class": "normal", "path": "tests/recordings_4min_study/WD/WD_5400rpm_healthy_read_60s.wav", "label": "WD 5400rpm Healthy"},
        {"class": "normal", "path": "tests/recordings_4min_study/Toshiba/Toshiba_5400rpm_healthy_read_60s.wav", "label": "Toshiba 5400rpm Healthy"},
        {"class": "normal", "path": "tests/recordings_4min_study/Seagate/Seagate_5400rpm_healthy_read_60s.wav", "label": "Seagate 5400rpm Healthy"},
        # 2. Rzeczywiste awarie mechaniczne
        {"class": "clicking", "path": "dataset_categorized/clicking/northwind_wd_head_click_1.wav", "label": "WD Head Crash Clicking"},
        {"class": "beeping", "path": "dataset_categorized/beeping/seagate_barracuda_seized_stiction.wav", "label": "Seagate Seized Beeping"},
        {"class": "scratching", "path": "dataset_categorized/scratching/aesonlabs_head_grinding_platter.wav", "label": "Aesonlabs Head Scratching"},
    ]

    print("=" * 80)
    print("EWALUACJA ZBIORU DANYCH Z NAŁOŻONYM NOWYM SZUMEM MIKROFONOWYM (SNR ~ 12 dB)")
    print("=" * 80)

    # Przygotowanie wycinka szumu
    noise_chunk = noise_data[:chunk_len].astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(noise_chunk ** 2)))

    correct_pure = 0
    correct_mixed = 0
    total = len(test_cases)

    for tc in test_cases:
        p = tc["path"]
        if not os.path.exists(p):
            # Fallback wyszukania innego pliku danej klasy
            alt_files = glob.glob(f"dataset_categorized/{tc['class']}/*.wav")
            if alt_files:
                p = alt_files[0]
            else:
                continue

        audio_raw, s_sr = sf.read(p)
        if len(audio_raw.shape) > 1:
            audio_raw = audio_raw[:, 0]
        drive_chunk = audio_raw[:chunk_len].astype(np.float32)
        if len(drive_chunk) < chunk_len:
            drive_chunk = np.pad(drive_chunk, (0, chunk_len - len(drive_chunk)))

        drive_rms = float(np.sqrt(np.mean(drive_chunk ** 2)))

        # 1. Test czystego sygnału
        t_pure = encode_audio_chunk(drive_chunk, sr)
        t_pure_torch = torch.from_numpy(t_pure.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            probs_pure = torch.softmax(model(t_pure_torch), dim=1).numpy()[0]
        san_pure, _, _ = guard.protect_and_sanitize_predictions(probs_pure, t_pure, waveform=drive_chunk)

        # 2. Test sygnału zmiksowanego z szumem mikrofonowym (współczynnik miksowania realistyczny 25% szumu)
        mixed_chunk = drive_chunk + (noise_chunk * 0.25 * (drive_rms / max(noise_rms, 1e-6)))
        mixed_chunk = mixed_chunk / max(float(np.max(np.abs(mixed_chunk))), 1.0) # Zapobieganie przesterowaniu

        t_mixed = encode_audio_chunk(mixed_chunk, sr)
        t_mixed_torch = torch.from_numpy(t_mixed.transpose(2, 0, 1)).unsqueeze(0)
        with torch.no_grad():
            probs_mixed = torch.softmax(model(t_mixed_torch), dim=1).numpy()[0]
        san_mixed, _, reason_mix = guard.protect_and_sanitize_predictions(probs_mixed, t_mixed, waveform=mixed_chunk)

        target_cls = tc["class"]
        pred_pure_cls = classes[int(np.argmax(san_pure))]
        pred_mix_cls = classes[int(np.argmax(san_mixed))]

        fault_prob_pure = float(san_pure[0] + san_pure[1] + san_pure[3]) * 100.0
        fault_prob_mix = float(san_mixed[0] + san_mixed[1] + san_mixed[3]) * 100.0

        is_pure_ok = (pred_pure_cls == target_cls) if target_cls != "normal" else (fault_prob_pure < 70.0)
        is_mix_ok = (pred_mix_cls == target_cls) if target_cls != "normal" else (fault_prob_mix < 70.0)

        if is_pure_ok:
            correct_pure += 1
        if is_mix_ok:
            correct_mixed += 1

        print(f"\n[{tc['label']}] (Oczekiwano: {target_cls.upper()}):")
        print(f"  Czysty dźwięk:    P(fault)={fault_prob_pure:.1f}%, Pred={pred_pure_cls.upper()} [{'OK' if is_pure_ok else 'FAIL'}]")
        print(f"  Zmiksowany z szumem: P(fault)={fault_prob_mix:.1f}%, Pred={pred_mix_cls.upper()} [{'OK' if is_mix_ok else 'FAIL'}]")
        if reason_mix:
            print(f"  Akcja Strażnika na miksie: {reason_mix}")

    print("\n" + "=" * 80)
    print(f"PODSUMOWANIE DOKŁADNOŚCI:")
    print(f"  Czyste próbki studyjne:       {correct_pure} / {total} ({(correct_pure/total)*100:.1f}%)")
    print(f"  Próbki zmiksowane z szumem:   {correct_mixed} / {total} ({(correct_mixed/total)*100:.1f}%)")
    print("=" * 80)

if __name__ == "__main__":
    main()
