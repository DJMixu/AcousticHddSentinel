import json
from pathlib import Path
import torch
import librosa

RAW_DIR = Path("data/raw/freesound_hdd")
with open(RAW_DIR / "metadata.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

# Załadowanie modelu Silero VAD
print("Ładowanie modelu Silero VAD...")
model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
get_speech_timestamps, _, _, _, _ = utils

speech_results = {}
sr_vad = 16000 # Silero VAD wymaga 16kHz lub 8kHz

print(f"Skanowanie {len(meta)} plików pod kątem mowy ludzkiej (Silero VAD @ 16kHz)...")

for sid, info in meta.items():
    fpath = RAW_DIR / info["file"]
    if not fpath.exists():
        continue
    
    y, sr = librosa.load(str(fpath), sr=sr_vad, mono=True)
    wav_tensor = torch.from_numpy(y)
    
    # Wykrywanie fragmentów mowy
    speech_timestamps = get_speech_timestamps(wav_tensor, model, sampling_rate=sr_vad, threshold=0.5)
    
    total_speech_samples = sum(ts['end'] - ts['start'] for ts in speech_timestamps)
    speech_duration_s = total_speech_samples / sr_vad
    total_dur_s = len(y) / sr_vad
    speech_pct = (speech_duration_s / total_dur_s * 100.0) if total_dur_s > 0 else 0.0
    
    speech_results[sid] = {
        "file": info["file"],
        "title": info["title"],
        "total_dur_s": round(total_dur_s, 2),
        "speech_dur_s": round(speech_duration_s, 2),
        "speech_pct": round(speech_pct, 2),
        "timestamps": [{"start_s": round(ts['start'] / sr_vad, 2), "end_s": round(ts['end'] / sr_vad, 2)} for ts in speech_timestamps]
    }
    
    if speech_pct > 0.0:
        print(f"  [WYKRYTO MOWĘ] [{sid}] {info['title']}: {speech_pct:.1f}% mowy ({speech_duration_s:.1f}s / {total_dur_s:.1f}s)")
    else:
        print(f"  [Czysty dźwięk - brak mowy] [{sid}] {info['title']}")

out_path = RAW_DIR / "speech_vad_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(speech_results, f, indent=2, ensure_ascii=False)

print(f"\nZapisano wyniki VAD do: {out_path}")
