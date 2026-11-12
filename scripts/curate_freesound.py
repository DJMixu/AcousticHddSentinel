import os
import sys
import re
import json
import urllib.request
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf

RAW_FREESOUND_DIR = Path("data/raw/freesound_hdd")
META_PATH = RAW_FREESOUND_DIR / "metadata.json"
OUTPUT_ENRICHED_PATH = RAW_FREESOUND_DIR / "metadata_enriched.json"

TARGET_SR = 22050
CHUNK_DURATION = 3.0
OVERLAP_DURATION = 1.5

def fetch_freesound_info():
    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    enriched = {}
    print(f"Pobieranie metadanych dla {len(meta)} nagrań z Freesound...")
    
    for sid, info in meta.items():
        url = info.get("url")
        desc = ""
        tags = []
        if url:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"})
                with urllib.request.urlopen(req, timeout=10) as resp:
                    html = resp.read().decode("utf-8", errors="ignore")
                    
                    # Szukanie opisu
                    desc_match = re.search(r'class="sound_description"[^>]*>(.*?)</div>', html, re.DOTALL)
                    if not desc_match:
                        desc_match = re.search(r'id="sound_description"[^>]*>(.*?)</div>', html, re.DOTALL)
                    if desc_match:
                        desc = re.sub(r'<[^>]+>', ' ', desc_match.group(1)).strip()
                        desc = re.sub(r'\s+', ' ', desc)
                    
                    # Szukanie tagów
                    found_tags = re.findall(r'/tags/([^/"\?#]+)/', html)
                    if not found_tags:
                        found_tags = re.findall(r'tag">([^<]+)</a>', html)
                    tags = list(dict.fromkeys([t.lower().strip() for t in found_tags if t.strip()]))
            except Exception as e:
                print(f"  [Błąd pobierania {sid}]: {e}")
        
        enriched[sid] = {
            "title": info.get("title"),
            "user": info.get("user"),
            "file": info.get("file"),
            "url": url,
            "duration_s": info.get("duration_s"),
            "rms_meta": info.get("rms"),
            "description": desc,
            "tags": tags[:15]
        }
    
    with open(OUTPUT_ENRICHED_PATH, "w", encoding="utf-8") as f:
        json.dump(enriched, f, indent=2, ensure_ascii=False)
    print(f"Zapisano wzbogacone metadane do: {OUTPUT_ENRICHED_PATH}")
    return enriched

if __name__ == "__main__":
    fetch_freesound_info()
