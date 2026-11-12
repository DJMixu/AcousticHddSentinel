import json
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

RAW_DIR = Path("data/raw/freesound_hdd")
META_PATH = RAW_DIR / "metadata_enriched.json"

with open(META_PATH, "r", encoding="utf-8") as f:
    meta = json.load(f)

print(f"Total files in metadata: {len(meta)}")

results = []

for sid, info in meta.items():
    fname = info["file"]
    fpath = RAW_DIR / fname
    if not fpath.exists():
        print(f"File missing: {fname}")
        continue
    
    # 1. Load audio with librosa
    y, sr = librosa.load(str(fpath), sr=22050, mono=True)
    dur = len(y) / sr
    
    # 2. RMS, Peak, Crest Factor
    rms = float(np.sqrt(np.mean(y ** 2)))
    peak = float(np.max(np.abs(y)))
    crest_factor = (peak / rms) if rms > 1e-7 else 0.0
    
    # 3. Spectral features
    centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    rolloff_85 = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
    rolloff_95 = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.95)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(y=y)))
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    
    # 4. Check chunking and silent chunks
    target_samples = int(3.0 * sr) # 66150
    step_samples = int(1.5 * sr) # 33075
    valid_chunks_train = 0
    valid_chunks_test = 0
    silent_chunks = 0
    
    if len(y) >= target_samples:
        # train chunks (50% overlap)
        for s in range(0, len(y) - target_samples + 1, step_samples):
            chunk = y[s:s + target_samples]
            c_rms = np.sqrt(np.mean(chunk ** 2))
            if c_rms >= 0.002:
                valid_chunks_train += 1
            else:
                silent_chunks += 1
        # test chunks (0% overlap)
        for s in range(0, len(y) - target_samples + 1, target_samples):
            chunk = y[s:s + target_samples]
            c_rms = np.sqrt(np.mean(chunk ** 2))
            if c_rms >= 0.002:
                valid_chunks_test += 1
    else:
        if rms >= 0.002:
            valid_chunks_train = 1
            valid_chunks_test = 1
        else:
            silent_chunks = 1

    # 5. Speech detection heuristics:
    # Human speech characteristics:
    # - Formants in 300 - 3400 Hz
    # - Pitch fundamental f0 typically 85 - 255 Hz
    # - Syllabic rhythm / modulation around 3 - 6 Hz in RMS energy envelope
    # Let's compute RMS envelope modulation and energy in speech band:
    hop_env = 512
    frame_rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_env)[0]
    frame_rms_var = float(np.var(frame_rms) / (np.mean(frame_rms) ** 2 + 1e-7))
    
    # Energy ratios in bands:
    # Low (<300Hz), Speech (300-3400Hz), High (>3400Hz)
    spec = np.abs(librosa.stft(y, n_fft=1024, hop_length=hop_env))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=1024)
    low_band = np.sum(spec[(freqs < 300), :])
    speech_band = np.sum(spec[(freqs >= 300) & (freqs < 3400), :])
    high_band = np.sum(spec[(freqs >= 3400), :])
    total_energy = low_band + speech_band + high_band + 1e-7
    speech_ratio = float(speech_band / total_energy)
    
    # 6. Check descriptions and tags
    desc = info.get("description", "").lower()
    tags = " ".join(info.get("tags", [])).lower()
    title = info.get("title", "").lower()

    results.append({
        "id": sid,
        "file": fname,
        "title": info.get("title"),
        "user": info.get("user"),
        "duration_s": round(dur, 2),
        "rms": round(rms, 6),
        "peak": round(peak, 4),
        "crest_factor": round(crest_factor, 2),
        "centroid": round(centroid, 1),
        "rolloff_95": round(rolloff_95, 1),
        "flatness": round(flatness, 6),
        "zcr": round(zcr, 4),
        "valid_chunks_train": valid_chunks_train,
        "valid_chunks_test": valid_chunks_test,
        "silent_chunks": silent_chunks,
        "speech_ratio": round(speech_ratio, 3),
        "frame_rms_var": round(frame_rms_var, 3),
        "tags": info.get("tags", []),
        "desc_snippet": info.get("description", "")[:120].replace("\n", " ")
    })

with open("data/raw/freesound_hdd/acoustic_analysis_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Saved acoustic analysis for {len(results)} files to data/raw/freesound_hdd/acoustic_analysis_results.json")
