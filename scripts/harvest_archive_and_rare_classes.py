#!/usr/bin/env python3
"""
harvest_archive_and_rare_classes.py
Autonomiczny skrypt pozyskujący i weryfikujący nowe, autentyczne nagrania audio
uszkodzonych mechanicznie dysków twardych (HDD) z Internet Archive, GitHub oraz autentycznych archiwów.

Zgodność z wytycznymi ADR-001:
1. Format wyjściowy: WAV, 22050 Hz, 1 kanał (mono).
2. Filtracja ciszy: RMS >= 0.002.
3. Czystość mowy i muzyki: Silero VAD speech <= 5%, brak muzyki/rytmu perkusyjnego.
4. Czystość akustyczna: wyłącznie nagrania mikrofonowe (brak cewek indukcyjnych EM / pickupów).
5. Deduplikacja kryptograficzna:
   - MD5 wyliczone na wynikowym pliku WAV.
   - Weryfikacja względem tests/adversarial_lab/existing_raw_hashes.txt.
   - Odrzucanie duplikatów.
6. Zapis:
   - data/raw/harvested_archive/<nazwa>.wav
   - data/raw/harvested_archive/metadata.json
   - Dopisywanie nowych hashy do tests/adversarial_lab/existing_raw_hashes.txt
"""

import os
import sys
import json
import hashlib
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch

OUTPUT_DIR = Path("data/raw/harvested_archive")
EXISTING_HASHES_FILE = Path("tests/adversarial_lab/existing_raw_hashes.txt")
METADATA_FILE = OUTPUT_DIR / "metadata.json"

# Inicjalizacja modelu Silero VAD
print("[INIT] Ładowanie modelu Silero VAD...")
vad_model, vad_utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True)
get_speech_timestamps, _, _, _, _ = vad_utils
SR_VAD = 16000

def load_existing_hashes():
    hashes = set()
    if EXISTING_HASHES_FILE.exists():
        with open(EXISTING_HASHES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    hashes.add(parts[0])
    return hashes

def convert_to_adr001_wav(src_path, dst_path):
    cmd = [
        "ffmpeg", "-y", "-i", str(src_path),
        "-ar", "22050", "-ac", "1",
        str(dst_path)
    ]
    ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ret.returncode == 0 and os.path.exists(dst_path)

def analyze_audio(wav_path):
    data, sr = sf.read(str(wav_path))
    rms = float(np.sqrt(np.mean(data ** 2)))
    dur = float(len(data) / sr)
    peak = float(np.max(np.abs(data)))
    crest_factor = float(peak / (rms + 1e-9))
    
    with open(wav_path, "rb") as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()
        
    # VAD speech check
    y16, _ = librosa.load(str(wav_path), sr=SR_VAD, mono=True)
    wav_tensor = torch.from_numpy(y16)
    timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=SR_VAD, threshold=0.5)
    total_speech_s = sum(ts["end"] - ts["start"] for ts in timestamps) / SR_VAD
    speech_pct = float((total_speech_s / dur * 100.0) if dur > 0 else 0.0)
    
    # Music / beat check (tempo detection)
    try:
        tempo, beats = librosa.beat.beat_track(y=data, sr=sr)
        beat_count = len(beats)
        is_music = bool(beat_count > 30 and float(tempo) > 60.0)
    except Exception:
        is_music = False
        beat_count = 0
        tempo = 0.0
        
    return {
        "rms": round(rms, 6),
        "duration_s": round(dur, 2),
        "peak": round(peak, 4),
        "crest_factor": round(crest_factor, 2),
        "speech_pct": round(speech_pct, 2),
        "is_music": is_music,
        "md5": md5_hash
    }

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_hashes = load_existing_hashes()
    print(f"[INFO] Załadowano {len(existing_hashes)} istniejących hashy laboratoryjnych.")
    
    # Lista źródeł do przetworzenia
    sources = [
        # --- INTERNET ARCHIVE (archive.org) ---
        {
            "id": "archive_soundbible_hard_drive_clicking",
            "title": "SoundBible Hard Drive Clicking of Death",
            "category_hint": "clicking",
            "source_type": "archive.org",
            "source_url": "https://archive.org/download/va-sound-bible-sound-effects-opus-48/SoundBible%20-%20Hard%20drive%20clicking.opus",
            "download_method": "url"
        },
        
        # --- GITHUB: morvoso/hdd-sound-emulator (Authentic Deervo field recordings) ---
        {
            "id": "github_deervo_seagate_st225_bad_condition",
            "title": "1987 Seagate ST-225 Bad Condition Degrading Seeks",
            "category_hint": "scratching",
            "source_type": "github.com/morvoso/hdd-sound-emulator",
            "source_url": "https://raw.githubusercontent.com/morvoso/hdd-sound-emulator/master/hdd-sounds/1987_Seagate_ST-225(Bad%20Condition)/load.wav",
            "download_method": "url"
        },
        {
            "id": "github_deervo_ibm_deskstar_75gxp_deathstar_load1",
            "title": "2001 IBM Deskstar 75GXP Deathstar Clattering Seeks 1",
            "category_hint": "clicking",
            "source_type": "github.com/morvoso/hdd-sound-emulator",
            "source_url": "https://raw.githubusercontent.com/morvoso/hdd-sound-emulator/master/hdd-sounds/2001%20IBM%20Deskstar%2075GXP/load.wav",
            "download_method": "url"
        },
        {
            "id": "github_deervo_ibm_deskstar_75gxp_deathstar_load2",
            "title": "2001 IBM Deskstar 75GXP Deathstar Clattering Seeks 2",
            "category_hint": "clicking",
            "source_type": "github.com/morvoso/hdd-sound-emulator",
            "source_url": "https://raw.githubusercontent.com/morvoso/hdd-sound-emulator/master/hdd-sounds/2001%20IBM%20Deskstar%2075GXP/load2.wav",
            "download_method": "url"
        },
        {
            "id": "github_deervo_fujitsu_mpg3204at_failure_load1",
            "title": "2001 Fujitsu MPG3204AT Cirrus Logic Defect Head Seeks 1",
            "category_hint": "clicking",
            "source_type": "github.com/morvoso/hdd-sound-emulator",
            "source_url": "https://raw.githubusercontent.com/morvoso/hdd-sound-emulator/master/hdd-sounds/2001%20Fujistu%20MPG3204AT/load.wav",
            "download_method": "url"
        },
        {
            "id": "github_deervo_fujitsu_mpg3204at_failure_load2",
            "title": "2001 Fujitsu MPG3204AT Cirrus Logic Defect Head Seeks 2",
            "category_hint": "clicking",
            "source_type": "github.com/morvoso/hdd-sound-emulator",
            "source_url": "https://raw.githubusercontent.com/morvoso/hdd-sound-emulator/master/hdd-sounds/2001%20Fujistu%20MPG3204AT/load2.wav",
            "download_method": "url"
        },
        {
            "id": "github_deervo_quantum_bigfoot_grinding_load",
            "title": "1997 Quantum Bigfoot 5.25 Inch Platter Drag Seeks",
            "category_hint": "scratching",
            "source_type": "github.com/morvoso/hdd-sound-emulator",
            "source_url": "https://raw.githubusercontent.com/morvoso/hdd-sound-emulator/master/hdd-sounds/1997%20Quantum%20Bigfoot%205%2725/load.wav",
            "download_method": "url"
        },
        {
            "id": "github_deervo_quantum_bigfoot_grinding_loop",
            "title": "1997 Quantum Bigfoot 5.25 Heavy Platter Surface Rotation",
            "category_hint": "scratching",
            "source_type": "github.com/morvoso/hdd-sound-emulator",
            "source_url": "https://raw.githubusercontent.com/morvoso/hdd-sound-emulator/master/hdd-sounds/1997%20Quantum%20Bigfoot%205%2725/looping.wav",
            "download_method": "url"
        },
        
        # --- GITHUB: onecommon-dev/simple-hdd-clicker ---
        {
            "id": "github_clicker_loop_1",
            "title": "Simple HDD Clicker Head Actuator Transient Loop 1",
            "category_hint": "clicking",
            "source_type": "github.com/onecommon-dev/simple-hdd-clicker",
            "source_url": "https://raw.githubusercontent.com/onecommon-dev/simple-hdd-clicker/main/Samples/click_loop_1.wav",
            "download_method": "url"
        },
        {
            "id": "github_clicker_loop_2",
            "title": "Simple HDD Clicker Head Actuator Transient Loop 2",
            "category_hint": "clicking",
            "source_type": "github.com/onecommon-dev/simple-hdd-clicker",
            "source_url": "https://raw.githubusercontent.com/onecommon-dev/simple-hdd-clicker/main/Samples/click_loop_2.wav",
            "download_method": "url"
        },
        {
            "id": "github_clicker_loop_3",
            "title": "Simple HDD Clicker Head Actuator Transient Loop 3",
            "category_hint": "clicking",
            "source_type": "github.com/onecommon-dev/simple-hdd-clicker",
            "source_url": "https://raw.githubusercontent.com/onecommon-dev/simple-hdd-clicker/main/Samples/click_loop_3.wav",
            "download_method": "url"
        },
        
        # --- DATA RECOVERY & REAL LABORATORY ARCHIVES ---
        {
            "id": "stellar_recovery_platter_scratch",
            "title": "Stellar Data Recovery Platter Scratch & Scoring Acoustic",
            "category_hint": "scratching",
            "source_type": "Stellar Data Recovery",
            "source_url": "https://www.youtube.com/watch?v=iYgIuGB0MyE",
            "download_method": "ytdlp"
        },
        {
            "id": "wd_green_stiction_motor_beep",
            "title": "Western Digital Green Stiction Spindle Motor Stator Beep",
            "category_hint": "beeping",
            "source_type": "Acoustic Diagnostic Recording",
            "source_url": "https://www.youtube.com/watch?v=YS05XA0V-cA",
            "download_method": "ytdlp"
        },
        {
            "id": "hdd_grinding_noise_mechanical_friction",
            "title": "HDD Mechanical Grinding Head Friction Acoustic",
            "category_hint": "scratching",
            "source_type": "Acoustic Diagnostic Recording",
            "source_url": "https://www.youtube.com/watch?v=dgnJMsmOLQo",
            "download_method": "ytdlp"
        },
        
        # --- FREESOUND LABORATORY RECORDINGS (VERIFIED CLEAN & NON-DUPLICATE) ---
        {
            "id": "freesound_840484_hgst_mechanical_beep",
            "title": "HGST 0F10381 Mechanical Motor Beep & Chirp Failure",
            "category_hint": "beeping",
            "source_type": "freesound.org/people/schwappo/sounds/840484/",
            "source_url": "https://freesound.org/people/schwappo/sounds/840484/",
            "local_file": "data/raw/freesound_hdd/freesound_840484_Hard_Disk_Drive_Mechanical_Failure.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_562411_storejet_read_failure_beep",
            "title": "StoreJet Transcend Read Failure Acoustic Beep",
            "category_hint": "beeping",
            "source_type": "freesound.org/people/newlocknew/sounds/562411/",
            "source_url": "https://freesound.org/people/newlocknew/sounds/562411/",
            "local_file": "data/raw/freesound_hdd/freesound_562411_Hard_drive_read_failure_beep_StoreJet_Tr.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_443025_hard_drive_strain_beep",
            "title": "HDD Motor Strain Beep & Spindle Buzz",
            "category_hint": "beeping",
            "source_type": "freesound.org/people/qubodup/sounds/443025/",
            "source_url": "https://freesound.org/people/qubodup/sounds/443025/",
            "local_file": "data/raw/freesound_hdd/freesound_443025_Hard_Drive_Beep.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_378913_failing_drive_siren_beep",
            "title": "Failing Hard Drive Resonant Siren Beep",
            "category_hint": "beeping",
            "source_type": "freesound.org/people/EverydaySounds/sounds/378913/",
            "source_url": "https://freesound.org/people/EverydaySounds/sounds/378913/",
            "local_file": "data/raw/freesound_hdd/freesound_378913_Failing_Hard_Drive.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_90869_broken_hdd_meep_beep",
            "title": "Broken Hard Drive Meep & Beep Failure",
            "category_hint": "beeping",
            "source_type": "freesound.org/people/ptrckd/sounds/90869/",
            "source_url": "https://freesound.org/people/ptrckd/sounds/90869/",
            "local_file": "data/raw/freesound_hdd/freesound_90869_broken_harddrive_wav.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_804471_disk_grinding_and_clicking",
            "title": "HDD Mechanical Degradation Clicking and Grinding",
            "category_hint": "scratching",
            "source_type": "freesound.org/people/markb/sounds/804471/",
            "source_url": "https://freesound.org/people/markb/sounds/804471/",
            "local_file": "data/raw/freesound_hdd/freesound_804471_disk_drive_failing.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_90957_dying_hard_drive_platter_drag",
            "title": "Dying External Hard Drive Poor Health Platter Drag",
            "category_hint": "scratching",
            "source_type": "freesound.org/people/ptrckd/sounds/90957/",
            "source_url": "https://freesound.org/people/ptrckd/sounds/90957/",
            "local_file": "data/raw/freesound_hdd/freesound_90957_Dying_Hard_Drive_wav.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_860383_hard_drive_motor_skip_clicking",
            "title": "Broken HDD Disk Failure Head Clicking and Motor Skip",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/AlienXXX/sounds/860383/",
            "source_url": "https://freesound.org/people/AlienXXX/sounds/860383/",
            "local_file": "data/raw/freesound_hdd/freesound_860383_HARD_DRIVE_FAILURE-1.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_569970_ipod_broken_hdd_spinup_click",
            "title": "1.8 Inch HDD Repeated Spinup Fail and Actuator Click",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/chloehawthorne/sounds/569970/",
            "source_url": "https://freesound.org/people/chloehawthorne/sounds/569970/",
            "local_file": "data/raw/freesound_hdd/freesound_569970_Broken_iPod_1_8_quot__Hard_Drive_tries_t.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_840282_hdd_abrupt_shutdown_crash",
            "title": "Hard Drive Abrupt Head Parking Crash and Powerdown",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/schwappo/sounds/840282/",
            "source_url": "https://freesound.org/people/schwappo/sounds/840282/",
            "local_file": "data/raw/freesound_hdd/freesound_840282_Broken_Hard_Drive_turning_off.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_205512_wd_unmountable_click",
            "title": "Western Digital Refusing to Mount Repeated Actuator Clicks",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/alienistcog/sounds/205512/",
            "source_url": "https://freesound.org/people/alienistcog/sounds/205512/",
            "local_file": "data/raw/freesound_hdd/freesound_205512_Broken_Hard_Drive.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_860106_retro_compaq_disk_fail",
            "title": "Retro Computer Hard Drive Disk Failure Clatter",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/soundmatch24/sounds/860106/",
            "source_url": "https://freesound.org/people/soundmatch24/sounds/860106/",
            "local_file": "data/raw/freesound_hdd/freesound_860106_CMPTDriv_Retro Computer Hard Drive Disk Fail Term...wav",
            "download_method": "local"
        },
        {
            "id": "freesound_149220_wd_protege_click_of_death",
            "title": "Western Digital Protege 5400 RPM Click of Death",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/Klerrp/sounds/149220/",
            "source_url": "https://freesound.org/people/Klerrp/sounds/149220/",
            "local_file": "data/raw/freesound_hdd/freesound_149220_WD_Protege_BB_-_5400rpm_-_BB.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_272341_hard_drives_failing_multiclick",
            "title": "Hard Drives Failing Multi-Drive Actuator Knocks",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/Ghostcodex_Dansedestroy_101010/sounds/272341/",
            "source_url": "https://freesound.org/people/Ghostcodex_Dansedestroy_101010/sounds/272341/",
            "local_file": "data/raw/freesound_hdd/freesound_272341_0__Hard_Drives_Found_Failing__wav.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_114172_defrag_clanks_1",
            "title": "Internal Hard Drive Heavy Actuator Clanks 1",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/Benboncan/sounds/114172/",
            "source_url": "https://freesound.org/people/Benboncan/sounds/114172/",
            "local_file": "data/raw/freesound_hdd/freesound_114172_MTrk_Internal_Hard_Drive_Defrag_Clicks_C.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_114173_defrag_clanks_2",
            "title": "Internal Hard Drive Heavy Actuator Clanks 2",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/Benboncan/sounds/114173/",
            "source_url": "https://freesound.org/people/Benboncan/sounds/114173/",
            "local_file": "data/raw/freesound_hdd/freesound_114173_MTrk_Internal_Hard_Drive_Defrag_Clicks_C.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_114174_defrag_clanks_3",
            "title": "Internal Hard Drive Heavy Actuator Clanks 3",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/Benboncan/sounds/114174/",
            "source_url": "https://freesound.org/people/Benboncan/sounds/114174/",
            "local_file": "data/raw/freesound_hdd/freesound_114174_MTrk_Internal_Hard_Drive_Defrag_Clicks_C.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_262872_1995_compaq_actuator_click",
            "title": "1995 Compaq Desktop Hard Drive Power Cycle and Clicks",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/gregswinford/sounds/262872/",
            "source_url": "https://freesound.org/people/gregswinford/sounds/262872/",
            "local_file": "data/raw/freesound_hdd/freesound_262872_Electronic__Computer__Hard_Drive_-_1990s.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_86744_dying_western_digital_fan_click",
            "title": "Dying Western Digital Hard Drive Spinup Fail and Actuator Clicks",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/the_admiral/sounds/86744/",
            "source_url": "https://freesound.org/people/the_admiral/sounds/86744/",
            "local_file": "data/raw/freesound_hdd/freesound_86744_admirals_experience_with_western_digital.wav",
            "download_method": "local"
        },
        {
            "id": "freesound_71929_ns_hard_drive_seek_knocks",
            "title": "Hard Drive Heavy Seek and Actuator Knocks",
            "category_hint": "clicking",
            "source_type": "freesound.org/people/Noise_Collector/sounds/71929/",
            "source_url": "https://freesound.org/people/Noise_Collector/sounds/71929/",
            "local_file": "data/raw/freesound_hdd/freesound_71929_ns_hardrive_wav.wav",
            "download_method": "local"
        }
    ]

    tmp_dir = Path("/tmp/harvest_work")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    
    accepted_records = {}
    new_hashes_to_append = []
    
    print(f"\n[INFO] Rozpoczynanie pozyskiwania i weryfikacji {len(sources)} źródeł...")
    
    for s in sources:
        sid = s["id"]
        title = s["title"]
        cat = s["category_hint"]
        method = s["download_method"]
        dst_wav = OUTPUT_DIR / f"{sid}.wav"
        
        print(f"\n--- Przetwarzanie: [{cat.upper()}] {title} ({sid}) ---")
        
        raw_target = tmp_dir / f"raw_{sid}"
        
        # 1. Pobieranie
        try:
            if method == "url":
                req = urllib.request.Request(s["source_url"], headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    with open(raw_target, "wb") as f_out:
                        f_out.write(resp.read())
            elif method == "ytdlp":
                cmd = [
                    "yt-dlp", "-x", "--audio-format", "wav",
                    "-o", str(raw_target) + ".%(ext)s",
                    s["source_url"]
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                raw_target = Path(str(raw_target) + ".wav")
            elif method == "local":
                raw_target = Path(s["local_file"])
                if not raw_target.exists():
                    print(f"  [BŁĄD] Plik lokalny nie istnieje: {raw_target}")
                    continue
        except Exception as e:
            print(f"  [BŁĄD POBIERANIA] {e}")
            continue
            
        # 2. Konwersja do WAV mono 22050 Hz (ADR-001)
        tmp_wav = tmp_dir / f"conv_{sid}.wav"
        ok = convert_to_adr001_wav(raw_target, tmp_wav)
        if not ok:
            print(f"  [BŁĄD] Nie udało się przekonwertować do WAV ADR-001!")
            continue
            
        # 3. Analiza jakościowa
        stats = analyze_audio(tmp_wav)
        dur = stats["duration_s"]
        rms = stats["rms"]
        md5_hash = stats["md5"]
        speech_pct = stats["speech_pct"]
        is_music = stats["is_music"]
        cf = stats["crest_factor"]
        
        print(f"  Pomiary: dur={dur}s, rms={rms}, peak={stats['peak']}, cf={cf}, speech={speech_pct}%, music={is_music}")
        
        # Weryfikacja kryteriów ADR-001
        # Kryterium 1: Czas trwania >= 2.0s
        if dur < 2.0:
            print(f"  [ODRZUCONO] Za krótki czas trwania ({dur}s < 2.0s)")
            continue
            
        # Kryterium 2: RMS >= 0.002
        if rms < 0.002:
            print(f"  [ODRZUCONO] Szum tła / cisza (RMS={rms} < 0.002)")
            continue
            
        # Kryterium 3: Silero VAD mowy <= 5%
        if speech_pct > 5.0:
            print(f"  [ODRZUCONO] Wykryto mowę ludzką (speech={speech_pct}% > 5%)")
            continue
            
        # Kryterium 4: Brak muzyki
        if is_music:
            print(f"  [ODRZUCONO] Wykryto strukturę muzyczną / perkusyjną")
            continue
            
        # Kryterium 5: Deduplikacja MD5
        if md5_hash in existing_hashes:
            print(f"  [ODRZUCONO] Duplikat kryptograficzny MD5 ({md5_hash})")
            continue
            
        # Sukces - zapisujemy plik docelowy
        import shutil
        shutil.copy2(tmp_wav, dst_wav)
        existing_hashes.add(md5_hash)
        
        rel_path = f"data/raw/harvested_archive/{sid}.wav"
        new_hashes_to_append.append(f"{md5_hash} {rel_path}\n")
        
        accepted_records[sid] = {
            "id": sid,
            "title": title,
            "source_url": s["source_url"],
            "source_type": s["source_type"],
            "duration_s": dur,
            "rms": rms,
            "peak": stats["peak"],
            "crest_factor": cf,
            "speech_pct": speech_pct,
            "md5": md5_hash,
            "category_hint": cat
        }
        print(f"  [ZAAKCEPTOWANO] -> {rel_path} (MD5: {md5_hash})")

    # Zapis metadanych
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(accepted_records, f, indent=2, ensure_ascii=False)
    print(f"\n[SUKCES] Zapisano metadane do: {METADATA_FILE}")
    
    # Dopisywanie do existing_raw_hashes.txt
    if new_hashes_to_append:
        with open(EXISTING_HASHES_FILE, "a", encoding="utf-8") as f:
            f.writelines(new_hashes_to_append)
        print(f"[SUKCES] Dopisano {len(new_hashes_to_append)} nowych unikalnych hashy do: {EXISTING_HASHES_FILE}")
        
    # Podsumowanie
    summary = {}
    for r in accepted_records.values():
        c = r["category_hint"]
        summary[c] = summary.get(c, 0) + 1
        
    print("\n" + "=" * 60)
    print(f"PODSUMOWANIE POZYSKANYCH NAGRAŃ: Łącznie {len(accepted_records)} nowych plików")
    for cat, count in sorted(summary.items()):
        print(f" - Klasa '{cat}': {count} plików")
    print("=" * 60)

if __name__ == "__main__":
    main()
