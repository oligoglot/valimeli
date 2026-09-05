"""
Script: download_canonical_tamil_wavs.py
Downloads the canonical native post-nasal recordings (தம்பி, மருந்து, குரங்கு) and updates the acoustic analysis.
"""

import urllib.request
import urllib.parse
import json
import os
import ssl
import wave
import numpy as np

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'ValiMeliAcousticPhonetics/1.0 (academic phonetic analysis; contact: research@example.org)'
}

AUDIO_DIR = "artifacts/audio_samples"
os.makedirs(AUDIO_DIR, exist_ok=True)

NEW_WORDS = [
    ("தம்பி", "LL-Q5885 (tam)-Sriveenkat-தம்பி.wav"),
    ("மருந்து", "LL-Q5885 (tam)-Sriveenkat-மருந்து.wav"),
    ("குரங்கு", "LL-Q5885 (tam)-Sriveenkat-குரங்கு.wav"),
]

def get_commons_file_url(filename: str) -> str:
    encoded_title = urllib.parse.quote("File:" + filename)
    url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={encoded_title}&prop=imageinfo&iiprop=url&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            pages = data.get("query", {}).get("pages", {})
            for pid, pdata in pages.items():
                imageinfo = pdata.get("imageinfo", [])
                if imageinfo:
                    return imageinfo[0].get("url", "")
    except Exception as e:
        print(f"Error getting URL for {filename}: {e}")
    return ""

def download_file(url: str, dest_path: str) -> bool:
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp, open(dest_path, "wb") as out:
            out.write(resp.read())
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

if __name__ == "__main__":
    print("=" * 80)
    print("DOWNLOADING CANONICAL TAMIL POST-NASAL AUDIO (THAMBI, MARUNDHU, KURANGU)")
    print("=" * 80)
    
    for word, fname in NEW_WORDS:
        print(f"Fetching URL for '{word}' -> {fname}...")
        url = get_commons_file_url(fname)
        if url:
            dest = os.path.join(AUDIO_DIR, fname)
            success = download_file(url, dest)
            if success:
                print(f"  ✓ Downloaded {fname} ({os.path.getsize(dest):,d} bytes)")
        else:
            print(f"  ❌ URL not found for {fname}")
