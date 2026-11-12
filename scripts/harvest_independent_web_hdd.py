#!/usr/bin/env python3
"""
harvest_independent_web_hdd.py
Agresywny scraper i przetwornik próbek audio dysków HDD z niezależnych źródeł:
- Internet Archive (archive.org)
- Repozytoria GitHub (MaverickUK/HDDSynth, glandais/winchester)
- Freesound.org (rozszerzona baza unikalnych zapytań i paginacji)

Zgodność z ADR-001:
- Format: WAV, 22050 Hz, 1 kanał (mono)
- Weryfikacja RMS >= 0.002 (odrzucanie ciszy)
- Weryfikacja unikalności MD5 względem tests/adversarial_lab/existing_raw_hashes.txt
- Zapis do data/raw/independent_web_hdd/ oraz data/raw/independent_web_hdd/metadata.json
"""

import os
import re
import json
import hashlib
import subprocess
import urllib.request
import urllib.parse
import numpy as np
import soundfile as sf

OUTPUT_DIR = "data/raw/independent_web_hdd"
EXISTING_HASHES_FILE = "tests/adversarial_lab/existing_raw_hashes.txt"
METADATA_FILE = os.path.join(OUTPUT_DIR, "metadata.json")

# 1. Źródła bezpośrednie z Archive.org i GitHub
DIRECT_SOURCES = [
    {
        "id": "archive_ipod_dead_hdd_01",
        "title": "Clicking iPod 1.8 inch HDD Spinup Fail and Spindown",
        "source": "Internet Archive (dead_ipod_01)",
        "url": "https://archive.org/download/dead_ipod_01/dead_ipod_01.wav",
        "category_hint": "clicking"
    },
    {
        "id": "archive_hdd_clicking_failure",
        "title": "Hard Drive Disk Repetitive Clicking Death",
        "source": "Internet Archive (HddClicking)",
        "url": "https://archive.org/download/HddClicking/HddClicking.mp3",
        "category_hint": "clicking"
    },
    {
        "id": "archive_slowly_degenerating_hdd",
        "title": "Slowly Degenerating HDD Acoustic Breakdown",
        "source": "Internet Archive (rp_20250308_20250308_1318)",
        "url": "https://archive.org/download/rp_20250308_20250308_1318/delete.mp3",
        "category_hint": "scratching"
    },
    {
        "id": "archive_hdd3_drive_noises",
        "title": "Hard Drive Mechanical Noises",
        "source": "Internet Archive (hdd-3)",
        "url": "https://archive.org/download/hdd-3/HDD2.mp3",
        "category_hint": "normal"
    },
    {
        "id": "github_hdds_access",
        "title": "HDDSynth HDD Access Head Clicks",
        "source": "GitHub (MaverickUK/HDDSynth)",
        "url": "https://raw.githubusercontent.com/MaverickUK/HDDSynth/main/samples/original/hdd_access.wav",
        "category_hint": "normal"
    },
    {
        "id": "github_hdds_idle",
        "title": "HDDSynth HDD Idle Platter Whir",
        "source": "GitHub (MaverickUK/HDDSynth)",
        "url": "https://raw.githubusercontent.com/MaverickUK/HDDSynth/main/samples/original/hdd_idle.wav",
        "category_hint": "normal"
    },
    {
        "id": "github_hdds_idle_long",
        "title": "HDDSynth HDD Long Idle Rotation",
        "source": "GitHub (MaverickUK/HDDSynth)",
        "url": "https://raw.githubusercontent.com/MaverickUK/HDDSynth/main/samples/original/hdd_idle_long.wav",
        "category_hint": "normal"
    },
    {
        "id": "github_hdds_spinup",
        "title": "HDDSynth HDD Platter Spinup Cycle",
        "source": "GitHub (MaverickUK/HDDSynth)",
        "url": "https://raw.githubusercontent.com/MaverickUK/HDDSynth/main/samples/original/hdd_spinup.wav",
        "category_hint": "normal"
    }
]

# Dodajmy źródła z glandais/winchester (z lokalnego klonu /tmp/winchester_repo)
WINCHESTER_DIR = "/tmp/winchester_repo/docs/writing/simulating-a-hard-disk/media"
if os.path.exists(WINCHESTER_DIR):
    for f in sorted(os.listdir(WINCHESTER_DIR)):
        if f.endswith(".m4a"):
            DIRECT_SOURCES.append({
                "id": f"github_winchester_{f.replace('.m4a', '')}",
                "title": f"Winchester Real HDD Acoustic {f}",
                "source": "GitHub (glandais/winchester)",
                "local_file": os.path.join(WINCHESTER_DIR, f),
                "category_hint": "normal"
            })

# 2. Zapytania Freesound poszerzone
FREESOUND_QUERIES = [
    "hard disk drive",
    "hdd click",
    "hard drive motor",
    "hard drive spin",
    "disk head",
    "hard drive seek",
    "hard disk click",
    "seagate hard drive",
    "western digital hard drive",
    "bad hard drive",
    "hard drive read",
    "hard drive write",
    "computer hard drive",
    "external hard drive",
    "hard drive dying",
    "hard drive buzz",
    "hard drive screech"
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
        with urllib.request.urlopen(req, timeout=12) as resp:
            return resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        print(f"  [GET Error] {url}: {e}")
        return ""

def search_freesound(query, max_pages=2):
    results = []
    seen = set()
    for page in range(1, max_pages + 1):
        url = f"https://freesound.org/search/?q={urllib.parse.quote_plus(query)}&page={page}&f=duration:%5B1+TO+120%5D"
        html = get_html(url)
        matches = re.findall(r'href=\"/people/([^/]+)/sounds/(\d+)/\"[^>]*>([^<]+)</a>', html)
        for user, sid, title in matches:
            if sid.isdigit() and sid not in seen:
                seen.add(sid)
                results.append({
                    "id": sid,
                    "user": user,
                    "title": title.strip().replace("\n", " "),
                    "page_url": f"https://freesound.org/people/{user}/sounds/{sid}/"
                })
    return results

def get_freesound_audio_url(page_url):
    html = get_html(page_url)
    m = re.findall(r'https://cdn\.freesound\.org/previews/[^\"]+\.mp3', html)
    if m:
        hq = [x for x in m if "-hq.mp3" in x]
        return hq[0] if hq else m[0]
    return None

def process_file_to_adr001(src_path, out_wav):
    cmd = [
        "ffmpeg", "-y", "-i", src_path,
        "-ar", "22050", "-ac", "1",
        out_wav
    ]
    ret = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return ret.returncode == 0 and os.path.exists(out_wav)

def check_audio_quality(wav_path):
    try:
        data, sr = sf.read(wav_path)
        rms = float(np.sqrt(np.mean(data ** 2)))
        duration = float(len(data) / sr)
        with open(wav_path, "rb") as f:
            md5_hash = hashlib.md5(f.read()).hexdigest()
        return rms, duration, md5_hash
    except Exception as e:
        print(f"  [WAV Check Error] {e}")
        return 0.0, 0.0, ""

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    existing_hashes = load_existing_hashes()
    print(f"Załadowano {len(existing_hashes)} istniejących hashy laboratoryjnych.")
    
    metadata = {}
    if os.path.exists(METADATA_FILE):
        try:
            with open(METADATA_FILE, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}

    downloaded = 0
    duplicates = 0
    silent = 0

    # KROK 1: Przetwarzanie DIRECT_SOURCES (Archive.org, GitHub)
    print("\n=== KROK 1: Pobieranie z Archive.org oraz GitHub ===")
    for item in DIRECT_SOURCES:
        sid = item["id"]
        out_wav = os.path.join(OUTPUT_DIR, f"{sid}.wav")
        if os.path.exists(out_wav):
            continue

        print(f"\n[DIRECT] {item['title']} ({item['source']})")
        temp_file = os.path.join(OUTPUT_DIR, f"temp_{sid}")
        
        if "local_file" in item and os.path.exists(item["local_file"]):
            src_file = item["local_file"]
        else:
            try:
                req = urllib.request.Request(item["url"], headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as resp, open(temp_file, "wb") as f_out:
                    f_out.write(resp.read())
                src_file = temp_file
            except Exception as e:
                print(f"  -> Błąd pobierania: {e}")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                continue

        ok = process_file_to_adr001(src_file, out_wav)
        if os.path.exists(temp_file):
            os.remove(temp_file)

        if not ok:
            print("  -> Błąd konwersji ffmpeg.")
            continue

        rms, duration, file_hash = check_audio_quality(out_wav)
        if file_hash in existing_hashes:
            print(f"  -> DUPLIKAT (hash {file_hash}). Usunięto.")
            os.remove(out_wav)
            duplicates += 1
            continue

        if rms < 0.002:
            print(f"  -> CISZA (RMS {rms:.5f} < 0.002). Usunięto.")
            os.remove(out_wav)
            silent += 1
            continue

        existing_hashes.add(file_hash)
        metadata[sid] = {
            "title": item["title"],
            "source": item["source"],
            "url": item.get("url", item.get("local_file")),
            "file": os.path.basename(out_wav),
            "duration_s": round(duration, 3),
            "rms": round(rms, 6),
            "md5": file_hash,
            "category_hint": item.get("category_hint", "normal")
        }
        downloaded += 1
        print(f"  -> ✅ POBRANO: {os.path.basename(out_wav)} ({duration:.2f}s, RMS={rms:.4f}, MD5={file_hash})")

    # KROK 2: Poszerzone wyszukiwanie na Freesound.org
    print("\n=== KROK 2: Poszerzone pobieranie z Freesound.org ===")
    seen_fids = set()
    freesound_candidates = []
    
    # Sprawdź które ID już mamy w bazie freesound_hdd
    existing_freesound_ids = set()
    if os.path.exists("data/raw/freesound_hdd/metadata.json"):
        with open("data/raw/freesound_hdd/metadata.json") as f:
            existing_freesound_ids = set(json.load(f).keys())

    for q in FREESOUND_QUERIES:
        print(f"Szukanie: '{q}'...")
        res = search_freesound(q, max_pages=2)
        for cand in res:
            cid = cand["id"]
            if cid not in seen_fids and cid not in existing_freesound_ids:
                # Sprawdź czy tytuł nawiązuje do HDD
                t_low = cand["title"].lower()
                if any(w in t_low for w in ['hard drive', 'harddrive', 'hdd', 'hard disk', 'harddisk', 'head', 'platter', 'disk click', 'spindle', 'seagate', 'western digital', 'hitachi', 'toshiba', 'maxtor']):
                    seen_fids.add(cid)
                    freesound_candidates.append(cand)

    print(f"\nZnaleziono {len(freesound_candidates)} nowych unikalnych kandydatów Freesound.")

    for cand in freesound_candidates:
        sid = cand["id"]
        safe_title = re.sub(r'[^a-zA-Z0-9_-]', '_', cand["title"])[:35]
        out_wav = os.path.join(OUTPUT_DIR, f"freesound_{sid}_{safe_title}.wav")
        if os.path.exists(out_wav):
            continue

        print(f"\n[FS {sid}] {cand['title']} (by {cand['user']})")
        audio_url = get_freesound_audio_url(cand["page_url"])
        if not audio_url:
            print("  -> Nie znaleziono preview audio.")
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

        ok = process_file_to_adr001(temp_mp3, out_wav)
        if os.path.exists(temp_mp3):
            os.remove(temp_mp3)

        if not ok:
            print("  -> Błąd ffmpeg.")
            continue

        rms, duration, file_hash = check_audio_quality(out_wav)
        if file_hash in existing_hashes:
            print(f"  -> DUPLIKAT (hash {file_hash}). Usunięto.")
            os.remove(out_wav)
            duplicates += 1
            continue

        if rms < 0.002:
            print(f"  -> CISZA (RMS {rms:.5f} < 0.002). Usunięto.")
            os.remove(out_wav)
            silent += 1
            continue

        existing_hashes.add(file_hash)
        metadata[sid] = {
            "title": cand["title"],
            "user": cand["user"],
            "source": "Freesound.org",
            "url": cand["page_url"],
            "file": os.path.basename(out_wav),
            "duration_s": round(duration, 3),
            "rms": round(rms, 6),
            "md5": file_hash
        }
        downloaded += 1
        print(f"  -> ✅ POBRANO: {os.path.basename(out_wav)} ({duration:.2f}s, RMS={rms:.4f})")

    # Zapisz metadata.json
    with open(METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Zaktualizuj existing_raw_hashes.txt o nowo pobrane unikalne hashe
    with open(EXISTING_HASHES_FILE, "a", encoding="utf-8") as f_hash:
        for sid, info in metadata.items():
            f_hash.write(f"{info['md5']} {os.path.join(OUTPUT_DIR, info['file'])}\n")

    print("\n=======================================================")
    print(f"PODSUMOWANIE POBIERANIA Z NIEZALEŻNYCH STRON (INDEPENDENT WEB HDD):")
    print(f"  Nowych unikalnych plików HDD: {downloaded}")
    print(f"  Odrzuconych duplikatów:       {duplicates}")
    print(f"  Odrzuconej ciszy (< 0.002):   {silent}")
    print(f"  Łącznie w independent_web_hdd: {len(metadata)}")
    print(f"  Zapisano metadane do:          {METADATA_FILE}")
    print("=======================================================")

if __name__ == "__main__":
    main()
