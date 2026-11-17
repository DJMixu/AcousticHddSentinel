#!/usr/bin/env python3
"""
Skrypt przeszukujący i pobierający autentyczne nagrania uszkodzeń HDD z bazy Freesound.org.
Zgodny z ADR-001 oraz wytycznymi projektu AcousticHddSentinel:
1. Format: WAV mono 22050 Hz.
2. Filtracja ciszy: RMS >= 0.002.
3. Czystość akustyczna: VAD mowy <= 5% (Silero VAD na CPU).
4. Zakaz pickupów EM (electromagnetic / inductor / emf / telephone pickup).
5. Deduplikacja kryptograficzna: MD5 względem tests/adversarial_lab/existing_raw_hashes.txt oraz istniejących plików.
6. Zapis do data/raw/harvested_freesound/ z metadata.json oraz aktualizacją existing_raw_hashes.txt.
"""

import os
import re
import sys
import json
import time
import hashlib
import urllib.request
import urllib.parse
import urllib.error
import subprocess
from pathlib import Path
import numpy as np
import soundfile as sf
import librosa
import torch

OUTPUT_DIR = Path("data/raw/harvested_freesound")
METADATA_FILE = OUTPUT_DIR / "metadata.json"
EXISTING_HASHES_FILE = Path("tests/adversarial_lab/existing_raw_hashes.txt")

SEARCH_QUERIES = [
    # 1. Scratching
    "hard drive grinding",
    "platter scratch",
    "hard drive screech",
    "head crash",
    "scratched platter",
    "hard drive scratch",
    "hdd scratch",
    "hdd grinding",
    "hard drive scraping",
    "drive grinding",
    "disk grinding",
    "drive screech",
    
    # 2. Beeping
    "seized hard drive",
    "hdd beep",
    "stuck spindle",
    "hard drive stiction",
    "buzzing drive",
    "hard drive beep",
    "hard drive beeping",
    "hdd beeping",
    "hdd buzz",
    "hard drive buzz",
    "hard drive buzzing",
    "stuck hard drive",
    "hdd stuck",
    "hdd motor stuck",
    
    # 3. Clicking
    "hard drive click of death",
    "broken hard drive click",
    "head knock hdd",
    "dying hdd click",
    "hard drive click",
    "hdd click",
    "hard drive clicking",
    "hdd clicking",
    "broken hdd click",
    "click of death hdd"
]

TAG_QUERIES = [
    "hard-drive",
    "hdd",
    "harddrive"
]

EM_KEYWORDS = [
    "electromagnetic",
    "inductor",
    "emf",
    "telephone pickup",
    "telecoil",
    "soma ether",
    "coil pickup",
    "inductive"
]

NON_HDD_EXCLUDES = [
    "cymbal", "snare", "drum", "percussion", "synthesizer",
    "hip hop", "dance", "edm", "footsteps", "rain", "thunder",
    "car crash", "door slam", "gunshot", "water drop"
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:124.0) Gecko/20100101 Firefox/124.0"
}


def load_existing_hashes():
    hashes = set()
    if EXISTING_HASHES_FILE.exists():
        with open(EXISTING_HASHES_FILE, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    hashes.add(parts[0].lower())
    return hashes


def load_existing_freesound_ids():
    fs_ids = set()
    for p in Path("data/raw").glob("**/*.wav"):
        m = re.search(r"freesound_(\d+)", p.name)
        if m:
            fs_ids.add(m.group(1))
    return fs_ids


def fetch_url_html(url, retries=3):
    req = urllib.request.Request(url, headers=HEADERS)
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                final_url = resp.geturl()
                content = resp.read().decode("utf-8", errors="ignore")
                return content, final_url
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait_time = 5.0 * (attempt + 1)
                print(f"    [HTTP 429 Rate Limit] Czekam {wait_time}s przed ponowieniem...")
                time.sleep(wait_time)
            elif e.code == 404:
                return None, None
            else:
                print(f"    [HTTP {e.code}] Błąd dla {url}")
                time.sleep(2.0)
        except Exception as e:
            print(f"    [Błąd sieci] {e} dla {url}")
            time.sleep(2.0)
    return None, None


def search_candidates():
    candidates = {} # sid -> dict
    print("=== ETAP 1: Przeszukiwanie Freesound.org ===")
    
    # 1. Przeszukiwanie fraz
    for q in SEARCH_QUERIES:
        print(f"  Szukanie frazy: '{q}'...")
        for page in range(1, 4):
            url = f"https://freesound.org/search/?q={urllib.parse.quote_plus(q)}&page={page}"
            html, _ = fetch_url_html(url)
            if not html:
                break
            
            matches = re.findall(r"/people/([^/]+)/sounds/(\d+)/", html)
            page_unique = 0
            for u, sid in matches:
                if sid not in candidates:
                    candidates[sid] = {"user": u, "matched_queries": [q]}
                    page_unique += 1
                else:
                    if q not in candidates[sid]["matched_queries"]:
                        candidates[sid]["matched_queries"].append(q)
            
            time.sleep(1.2)
            if len(set(matches)) < 10:
                break

    # 2. Przeglądanie tagów
    for tag in TAG_QUERIES:
        print(f"  Przeglądanie tagu: '{tag}'...")
        for page in range(1, 6): # pierwsze 5 stron
            url = f"https://freesound.org/browse/tags/{tag}/?page={page}"
            html, _ = fetch_url_html(url)
            if not html:
                break
            
            matches = re.findall(r"/people/([^/]+)/sounds/(\d+)/", html)
            for u, sid in matches:
                if sid not in candidates:
                    candidates[sid] = {"user": u, "matched_queries": [f"tag:{tag}"]}
                else:
                    tag_str = f"tag:{tag}"
                    if tag_str not in candidates[sid]["matched_queries"]:
                        candidates[sid]["matched_queries"].append(tag_str)
            
            time.sleep(1.2)
            if len(set(matches)) < 10:
                break

    print(f"\nŁącznie zebrano {len(candidates)} unikalnych identyfikatorów kandydatów.")
    return candidates


def inspect_sound(sid):
    """Pobiera pełną stronę dźwięku z freesound.org/s/{sid}/ i parsuje metadane"""
    short_url = f"https://freesound.org/s/{sid}/"
    html, canonical_url = fetch_url_html(short_url)
    if not html:
        return None
    
    # Tytuł
    title_m = re.search(r"<meta\s+property=\"og:title\"\s+content=\"([^\"]+)\"", html)
    if not title_m:
        title_m = re.search(r"<title>(.*?)</title>", html)
    title = title_m.group(1).replace(" - Freesound", "").strip() if title_m else f"sound_{sid}"
    
    # Opis
    desc_m = re.search(r"<meta\s+property=\"og:description\"\s+content=\"([^\"]+)\"", html)
    desc = desc_m.group(1).strip() if desc_m else ""
    
    # Tagi
    tags_raw = re.findall(r"/tags/([^/\"\\?#]+)/", html)
    tags = [t.lower().strip() for t in set(tags_raw) if not t.startswith("_")]
    
    # Audio preview MP3
    previews = re.findall(r"https://cdn\.freesound\.org/previews/[^\"]+\.mp3", html)
    hq_previews = [p for p in previews if "-hq.mp3" in p]
    mp3_url = hq_previews[0] if hq_previews else (previews[0] if previews else None)
    
    return {
        "id": sid,
        "title": title,
        "canonical_url": canonical_url or short_url,
        "desc": desc,
        "tags": tags,
        "mp3_url": mp3_url
    }


def is_em_pickup(meta):
    """Sprawdza czy nagranie to pickup elektromagnetyczny (zakazany przez ADR-001)"""
    combined = (meta["title"] + " " + meta["desc"] + " " + " ".join(meta["tags"])).lower()
    for kw in EM_KEYWORDS:
        if kw in combined:
            return True, kw
    return False, ""


def is_valid_hdd_candidate(meta):
    """Sprawdza czy nagranie dotyczy HDD i nie jest instrumentem / szumem otoczenia"""
    combined = (meta["title"] + " " + meta["desc"] + " " + " ".join(meta["tags"])).lower()
    
    # Wykluczenia
    for excl in NON_HDD_EXCLUDES:
        if excl in meta["tags"] or (excl in combined and "hard drive" not in combined and "hdd" not in combined):
            return False, f"Wykluczenie non-HDD: '{excl}'"
            
    # Wymagane słowa kluczowe HDD
    hdd_keywords = [
        "hard drive", "harddrive", "hard-drive", "hdd",
        "hard disk", "hard-disk", "harddisk", "disk drive",
        "platter", "stuck spindle", "head crash", "stiction"
    ]
    has_hdd_kw = any(kw in combined for kw in hdd_keywords)
    if not has_hdd_kw:
        return False, "Brak słów kluczowych związanych z dyskiem twardym"
        
    return True, "OK"


def determine_category_hint(meta):
    combined = (meta["title"] + " " + meta["desc"] + " " + " ".join(meta["tags"])).lower()
    
    scratching_kw = ["scratch", "grind", "grinding", "screech", "scrape", "scraping", "friction", "head crash", "platter scratch"]
    beeping_kw = ["beep", "beeping", "buzz", "buzzing", "stuck", "seized", "spindle", "stiction", "siren", "chirp", "melody"]
    clicking_kw = ["click", "clicking", "click of death", "head knock", "knocking", "tick", "ticking", "clack", "failing hard drive"]
    
    scores = {
        "scratching": sum(1 for kw in scratching_kw if kw in combined),
        "beeping": sum(1 for kw in beeping_kw if kw in combined),
        "clicking": sum(1 for kw in clicking_kw if kw in combined)
    }
    
    # Sprawdź czy to dźwięk normalny
    normal_kw = ["spin up", "idle", "read and write", "accessing", "normal", "healthy", "laptop startup"]
    if all(score == 0 for score in scores.values()) and any(kw in combined for kw in normal_kw):
        return "normal"
        
    # Wybierz klasę z najwyższym wynikiem
    best_cat = max(scores, key=scores.get)
    if scores[best_cat] > 0:
        return best_cat
        
    # Domyślnie jeśli niejednoznaczne
    if "crash" in combined:
        return "scratching"
    return "clicking"


def check_speech_vad(wav_path, vad_model, vad_utils):
    """Sprawdza procent mowy w pliku audio za pomocą Silero VAD na CPU"""
    get_speech_timestamps, _, _, _, _ = vad_utils
    sr_vad = 16000
    y, sr = librosa.load(str(wav_path), sr=sr_vad, mono=True)
    if len(y) == 0:
        return 0.0
        
    wav_tensor = torch.from_numpy(y)
    with torch.no_grad():
        timestamps = get_speech_timestamps(wav_tensor, vad_model, sampling_rate=sr_vad, threshold=0.5)
        
    total_speech_samples = sum(ts['end'] - ts['start'] for ts in timestamps)
    speech_duration_s = total_speech_samples / sr_vad
    total_dur_s = len(y) / sr_vad
    speech_pct = (speech_duration_s / total_dur_s * 100.0) if total_dur_s > 0 else 0.0
    return float(speech_pct)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_hashes = load_existing_hashes()
    known_fs_ids = load_existing_freesound_ids()
    
    print(f"Załadowano {len(existing_hashes)} istniejących hashy MD5 z {EXISTING_HASHES_FILE}.")
    print(f"Załadowano {len(known_fs_ids)} znanych ID Freesound z repozytorium.")
    
    # Ładowanie metadanych jeśli już istnieją w OUTPUT_DIR
    metadata = {}
    if METADATA_FILE.exists():
        try:
            with open(METADATA_FILE, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    # Ładowanie Silero VAD na CPU
    print("Ładowanie modelu Silero VAD na CPU...")
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True
    )
    vad_model.to("cpu")
    print("Model Silero VAD gotowy.\n")
    
    # Krok 1: Wyszukiwanie kandydatów
    candidates = search_candidates()
    
    # Filtruj kandydatów, którzy są już pobrani lub w metadata
    filtered_sids = [sid for sid in candidates if sid not in known_fs_ids and sid not in metadata]
    print(f"\nDo zbadania pozostało {len(filtered_sids)} nowych kandydatów (pominięto {len(candidates) - len(filtered_sids)} już znanych).")
    
    stats = {
        "processed": 0,
        "em_rejected": 0,
        "non_hdd_rejected": 0,
        "no_preview": 0,
        "silence_rejected": 0,
        "duplicate_rejected": 0,
        "speech_rejected": 0,
        "accepted": 0,
        "by_category": {"scratching": 0, "beeping": 0, "clicking": 0, "normal": 0}
    }
    
    new_hashes_to_append = []
    
    for idx, sid in enumerate(filtered_sids, 1):
        print(f"\n--- [{idx}/{len(filtered_sids)}] Badanie dźwięku ID: {sid} ---")
        meta = inspect_sound(sid)
        time.sleep(1.2)
        if not meta:
            print("  -> Nie udało się pobrać metadanych.")
            continue
            
        stats["processed"] += 1
        print(f"  Tytuł: {meta['title']}")
        print(f"  Tagi:  {meta['tags'][:6]}")
        
        # 1. Sprawdzenie EM pickup
        is_em, em_term = is_em_pickup(meta)
        if is_em:
            print(f"  ❌ ODRZUCONO (EM pickup: '{em_term}')")
            stats["em_rejected"] += 1
            continue
            
        # 2. Sprawdzenie czy to HDD
        is_hdd, reason = is_valid_hdd_candidate(meta)
        if not is_hdd:
            print(f"  ❌ ODRZUCONO ({reason})")
            stats["non_hdd_rejected"] += 1
            continue
            
        # 3. Sprawdzenie linku MP3
        if not meta["mp3_url"]:
            print(f"  ❌ ODRZUCONO (brak linku do preview MP3)")
            stats["no_preview"] += 1
            continue
            
        # Pobieranie do pliku tymczasowego
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', meta["title"])[:45]
        temp_mp3 = OUTPUT_DIR / f"temp_{sid}.mp3"
        out_wav = OUTPUT_DIR / f"freesound_{sid}_{safe_title}.wav"
        
        try:
            req = urllib.request.Request(meta["mp3_url"], headers=HEADERS)
            with urllib.request.urlopen(req, timeout=20) as resp, open(temp_mp3, "wb") as f_out:
                f_out.write(resp.read())
        except Exception as e:
            print(f"  ❌ Błąd pobierania MP3: {e}")
            if temp_mp3.exists():
                temp_mp3.unlink()
            continue
            
        # Konwersja do WAV mono 22050 Hz (ADR-001)
        cmd = [
            "ffmpeg", "-y", "-i", str(temp_mp3),
            "-ar", "22050", "-ac", "1",
            str(out_wav)
        ]
        ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if temp_mp3.exists():
            temp_mp3.unlink()
            
        if ret.returncode != 0 or not out_wav.exists():
            print("  ❌ Błąd konwersji ffmpeg.")
            continue
            
        # Analiza akustyczna
        try:
            data, sr = sf.read(str(out_wav))
            duration = float(len(data) / sr)
            rms = float(np.sqrt(np.mean(data ** 2)))
            
            # Suma MD5
            with open(out_wav, "rb") as f_in:
                file_hash = hashlib.md5(f_in.read()).hexdigest().lower()
                
            # 4. Sprawdzenie deduplikacji
            if file_hash in existing_hashes:
                print(f"  ❌ ODRZUCONO (Duplikat MD5: {file_hash})")
                out_wav.unlink()
                stats["duplicate_rejected"] += 1
                continue
                
            # 5. Filtracja ciszy (RMS >= 0.002)
            if rms < 0.002:
                print(f"  ❌ ODRZUCONO (Cisza / niski RMS: {rms:.5f} < 0.002)")
                out_wav.unlink()
                stats["silence_rejected"] += 1
                continue
                
            # 6. Sprawdzenie czystości akustycznej (VAD mowy <= 5%)
            speech_pct = check_speech_vad(out_wav, vad_model, vad_utils)
            if speech_pct > 5.0:
                print(f"  ❌ ODRZUCONO (Wykryto mowę ludzką: {speech_pct:.1f}% > 5.0%)")
                out_wav.unlink()
                stats["speech_rejected"] += 1
                continue
                
            # Określenie category_hint
            category_hint = determine_category_hint(meta)
            
            # Zaakceptowano!
            existing_hashes.add(file_hash)
            known_fs_ids.add(sid)
            
            metadata[sid] = {
                "id": sid,
                "title": meta["title"],
                "source_url": meta["canonical_url"],
                "duration_s": round(duration, 3),
                "rms": round(rms, 6),
                "md5": file_hash,
                "category_hint": category_hint
            }
            
            rel_wav_path = f"data/raw/harvested_freesound/{out_wav.name}"
            new_hashes_to_append.append(f"{file_hash} {rel_wav_path}")
            
            stats["accepted"] += 1
            stats["by_category"][category_hint] = stats["by_category"].get(category_hint, 0) + 1
            
            print(f"  ✅ ZAAKCEPTOWANO: {out_wav.name}")
            print(f"     Czas: {duration:.2f}s | RMS: {rms:.4f} | Mowa: {speech_pct:.1f}% | Kategoria: {category_hint}")
            
            # Zapisz metadane na bieżąco
            with open(METADATA_FILE, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
                
            # Dopisuj hashe do pliku
            with open(EXISTING_HASHES_FILE, "a", encoding="utf-8") as f:
                f.write(f"{file_hash} {rel_wav_path}\n")
                
        except Exception as e:
            print(f"  ❌ Błąd przetwarzania audio: {e}")
            if out_wav.exists():
                out_wav.unlink()

    # Podsumowanie
    print("\n" + "=" * 60)
    print("PODSUMOWANIE PROCESU HARVESTINGU FREESOUND:")
    print(f"  Zbadanych dźwięków:            {stats['processed']}")
    print(f"  Odrzuconych (EM pickup):        {stats['em_rejected']}")
    print(f"  Odrzuconych (non-HDD / inne):   {stats['non_hdd_rejected']}")
    print(f"  Odrzuconych (brak preview):     {stats['no_preview']}")
    print(f"  Odrzuconych (duplikaty MD5):    {stats['duplicate_rejected']}")
    print(f"  Odrzuconych (cisza RMS < 0.002):{stats['silence_rejected']}")
    print(f"  Odrzuconych (mowa VAD > 5%):    {stats['speech_rejected']}")
    print(f"  ✅ ŁĄCZNIE POBRANYCH NOWYCH:    {stats['accepted']}")
    print(f"     - scratching:                {stats['by_category'].get('scratching', 0)}")
    print(f"     - beeping:                   {stats['by_category'].get('beeping', 0)}")
    print(f"     - clicking:                  {stats['by_category'].get('clicking', 0)}")
    print(f"     - normal:                    {stats['by_category'].get('normal', 0)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
