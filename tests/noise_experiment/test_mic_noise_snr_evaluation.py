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

    # Reprezentatywne próbki
    test_drives = [
        {"class": "normal", "name": "HGST Healthy", "path": "tests/recordings_4min_study/HGST/HGST_5400rpm_healthy_read_60s.wav"},
        {"class": "normal", "name": "WD Healthy", "path": "tests/recordings_4min_study/WD/WD_5400rpm_healthy_read_60s.wav"},
        {"class": "normal", "name": "Toshiba Healthy", "path": "tests/recordings_4min_study/Toshiba/Toshiba_5400rpm_healthy_read_60s.wav"},
        {"class": "clicking", "name": "Archive HDD Clicking", "path": "dataset_categorized/clicking/archive_hdd_clicking_failure.wav"},
        {"class": "beeping", "name": "Seagate Seized Beeping", "path": "dataset_categorized/beeping/seagate_barracuda_seized_stiction.wav"},
        {"class": "scratching", "name": "Aesonlabs Scratching", "path": "dataset_categorized/scratching/Aesonlabs Data Recovery_DiamondMax 200GB Maxtor Grinding.wav"}
    ]

    # Używamy fragmentu szumu z okna 1-4 (gdzie mikrofon nie jest przesterowany)
    cleaner_noise = noise_data[66150//2 : 66150//2 + chunk_len].astype(np.float32)
    noise_rms = float(np.sqrt(np.mean(cleaner_noise ** 2)))

    print("=" * 80)
    print("TEST ODPORNOŚCI NA NOWY SZUM MIKROFONU PRZY RÓŻNYCH POZIOMACH SNR")
    print("=" * 80)

    for snr_db in [20, 15]:
        print(f"\n--- POZIOM SNR: {snr_db} dB ---")
        correct = 0
        total = 0

        for d in test_drives:
            p = d["path"]
            if not os.path.exists(p):
                # fallback
                alts = glob.glob(f"dataset_categorized/{d['class']}/*.wav")
                if alts:
                    p = alts[0]
                else:
                    continue

            audio, _ = sf.read(p)
            if len(audio.shape) > 1:
                audio = audio[:, 0]
            chunk = audio[:chunk_len].astype(np.float32)
            if len(chunk) < chunk_len:
                chunk = np.pad(chunk, (0, chunk_len - len(chunk)))

            drive_rms = float(np.sqrt(np.mean(chunk ** 2)))
            scale = (drive_rms / max(noise_rms, 1e-6)) / (10 ** (snr_db / 20.0))
            mixed = chunk + (cleaner_noise * scale)
            mixed = mixed / max(float(np.max(np.abs(mixed))), 1.0)

            t = encode_audio_chunk(mixed, sr)
            t_torch = torch.from_numpy(t.transpose(2, 0, 1)).unsqueeze(0)
            with torch.no_grad():
                probs = torch.softmax(model(t_torch), dim=1).numpy()[0]
            san_p, _, reason = guard.protect_and_sanitize_predictions(probs, t, waveform=mixed)

            pred_idx = int(np.argmax(san_p))
            pred_cls = classes[pred_idx]
            fault_prob = float(san_p[0] + san_p[1] + san_p[3]) * 100.0

            target = d["class"]
            is_ok = (fault_prob < 70.0) if target == "normal" else (fault_prob >= 70.0)
            if is_ok:
                correct += 1
            total += 1

            status = "OK" if is_ok else "FAIL"
            print(f"  {d['name']} ({target.upper()}): P(fault)={fault_prob:.1f}%, Pred={pred_cls.upper()} [{status}]")

        print(f"Wynik dla SNR {snr_db} dB: {correct} / {total} ({(correct/total)*100:.1f}%)")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
