#!/usr/bin/env python3
"""
Skrypt zbierający autentyczne nagrania uszkodzeń HDD z bazy Freesound (spoza YouTube).
- Wyszukuje po frazach: hard drive failure, dying hard drive, hard drive click, hdd beep, head crash
- Pobiera pliki audio w najwyższej dostępnej jakości (HQ preview MP3)
- Konwertuje do WAV mono 22050 Hz (zgodnie z ADR-001)
- Sprawdza unikalność hashy (MD5) względem istniejącego datasetu
- Odrzuca ciszę i duplikaty
"""

import os
import re
import json
import hashlib
import urllib.request
import subprocess
import soundfile as sf
import numpy as np

OUTPUT_DIR = "data/raw/freesound_hdd"
EXISTING_HASHES_FILE = "tests/adversarial_lab/existing_raw_hashes.txt"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

SEARCH_QUERIES = [
    "hard drive failure",
    "failing hard drive",
    "broken hard drive",
    "dying hard drive",
    "hard drive click",
    "hdd click of death",
    "hard drive beep",
    "hard drive scratching",
    "head crash hard drive",
    "damaged hard drive"
]

def load_existing_hashes():
    hashes = set()
    if os.path.exists(EXISTING_HASHES_FILE):
        with open(EXISTING_HASHES_FILE, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    hashes.add(parts[0])
    return hashes

def get_html(url):
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:120.0) Gecko/20100101 Firefox/120.0"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"Błąd pobierania {url}: {e}")
        return ""

def search_freesound(query):
    encoded_query = urllib.parse.quote_plus(query)
    url = f"https://freesound.org/search/?q={encoded_query}&f=duration:%5B1+TO+120%5D"
    html = get_html(url)
    
    # Znajdź pary (user, sound_id, title)
    # Wzorce na stronie freesound:
    # <a href="/people/{user}/sounds/{id}/" ... title="{title}">
    matches = re.findall(r'href="/people/([^/]+)/sounds/(\d+)/"[^>]*>([^<]+)</a>', html)
    results = []
    seen_ids = set()
    for user, sound_id, title in matches:
        if sound_id not in seen_ids and sound_id.isdigit():
            seen_ids.add(sound_id)
            title_clean = title.strip().replace("\n", " ")
            results.append({
                "id": sound_id,
                "user": user,
                "title": title_clean,
                "page_url": f"https://freesound.org/people/{user}/sounds/{sound_id}/"
            })
    return results

def get_hq_audio_url(page_url, sound_id):
    html = get_html(page_url)
    # Szukaj bezpośrednich linków do cdn.freesound.org w HTML strony
    m = re.findall(r'https://cdn\.freesound\.org/previews/[^\"]+\.mp3', html)
    if m:
        # Preferuj -hq.mp3
        hq = [x for x in m if "-hq.mp3" in x]
        if hq:
            return hq[0]
        return m[0]
    return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_hashes = load_existing_hashes()
    print(f"Załadowano {len(existing_hashes)} istniejących hashy.")
    
    metadata = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
            
    all_candidates = []
    seen_cand_ids = set()
    
    for q in SEARCH_QUERIES:
        print(f"Szukanie: '{q}'...")
        res = search_freesound(q)
        for item in res:
            if item["id"] not in seen_cand_ids:
                seen_cand_ids.add(item["id"])
                all_candidates.append(item)
                
    print(f"\nZnaleziono {len(all_candidates)} unikalnych kandydatów na Freesound.")
    
    downloaded_count = 0
    duplicate_count = 0
    silence_count = 0
    
    for item in all_candidates:
        sid = item["id"]
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', item["title"])[:40]
        out_wav = os.path.join(OUTPUT_DIR, f"freesound_{sid}_{safe_title}.wav")
        
        if os.path.exists(out_wav):
            continue
            
        print(f"\n[ID {sid}] {item['title']} (by {item['user']})")
        audio_url = get_hq_audio_url(item["page_url"], sid)
        if not audio_url:
            print(f"  -> Nie udało się odnaleźć linku audio.")
            continue
            
        temp_mp3 = os.path.join(OUTPUT_DIR, f"temp_{sid}.mp3")
        try:
            req = urllib.request.Request(audio_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp, open(temp_mp3, "wb") as f_out:
                f_out.write(resp.read())
        except Exception as e:
            print(f"  -> Błąd pobierania MP3: {e}")
            if os.path.exists(temp_mp3):
                os.remove(temp_mp3)
            continue
            
        # Konwersja do WAV 22050 Hz mono zgodnie z ADR-001
        cmd = [
            "ffmpeg", "-y", "-i", temp_mp3,
            "-ar", "22050", "-ac", "1",
            out_wav
        ]
        ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)
            
        if ret.returncode != 0 or not os.path.exists(out_wav):
            print("  -> Błąd ffmpeg.")
            continue
            
        # Sprawdzenie audio (RMS, MD5)
        try:
            data, sr = sf.read(out_wav)
            rms = float(np.sqrt(np.mean(data ** 2)))
            duration = float(len(data) / sr)
            
            with open(out_wav, "rb") as f_in:
                file_hash = hashlib.md5(f_in.read()).hexdigest()
                
            if file_hash in existing_hashes:
                print(f"  -> DUPLIKAT (hash {file_hash}). Usunięto.")
                os.remove(out_wav)
                duplicate_count += 1
                continue
                
            if rms < 0.002:
                print(f"  -> CISZA (RMS {rms:.5f} < 0.002). Usunięto.")
                os.remove(out_wav)
                silence_count += 1
                continue
                
            existing_hashes.add(file_hash)
            metadata[sid] = {
                "title": item["title"],
                "user": item["user"],
                "url": item["page_url"],
                "file": os.path.basename(out_wav),
                "duration_s": duration,
                "rms": rms,
                "md5": file_hash
            }
            downloaded_count += 1
            print(f"  -> ✅ POBRANO: {os.path.basename(out_wav)} ({duration:.1f}s, RMS={rms:.4f})")
            
        except Exception as e:
            print(f"  -> Błąd analizy WAV: {e}")
            if os.path.exists(out_wav):
                os.remove(out_wav)
                
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        
    print("\n=======================================================")
    print(f"Zakończono pobieranie z Freesound:")
    print(f"  Nowych unikalnych plików HDD: {downloaded_count}")
    print(f"  Odrzuconych duplikatów:       {duplicate_count}")
    print(f"  Odrzuconej ciszy:             {silence_count}")
    print(f"  Łącznie w bazie Freesound:    {len(metadata)}")
    print("=======================================================")

if __name__ == "__main__":
    main()
