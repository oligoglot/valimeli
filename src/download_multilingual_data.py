#!/usr/bin/env python3
"""
Project ValiMeli — Multilingual Aksharantar Downloader & Extractor
Downloads and extracts raw Aksharantar JSON datasets from HuggingFace.
"""

import os
import sys
import zipfile
import urllib.request

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli", "data")
os.makedirs(DATA_DIR, exist_ok=True)

LANGUAGES = ["tel", "kan", "hin", "ben", "guj", "mar"]

def download_and_extract(lang: str):
    zip_path = os.path.join(DATA_DIR, f"{lang}.zip")
    extract_dir = os.path.join(DATA_DIR, f"extracted_{lang}")
    
    if os.path.exists(extract_dir) and len(os.listdir(extract_dir)) >= 3:
        print(f" -> [{lang}] Already extracted at {extract_dir}", flush=True)
        return
        
    url = f"https://huggingface.co/datasets/ai4bharat/Aksharantar/resolve/main/{lang}.zip"
    print(f" -> [{lang}] Downloading {url}...", flush=True)
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())
        print(f"    Saved {zip_path} ({os.path.getsize(zip_path):,} bytes)", flush=True)
        
        print(f" -> [{lang}] Extracting to {extract_dir}...", flush=True)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
            
        # Clean up zip
        os.remove(zip_path)
        print(f"    ⭐ Successfully extracted [{lang}]", flush=True)
    except Exception as e:
        print(f"    ❌ Failed to download/extract {lang}: {e}", flush=True)

def main():
    print("\n" + "=" * 80)
    print(" PROJECT VALIMELI — MULTILINGUAL DATASET INGESTION")
    print(" Ingesting Aksharantar Data for Dravidian + Indo-Aryan Scaling Grid")
    print("=" * 80)
    for lang in LANGUAGES:
        download_and_extract(lang)

if __name__ == "__main__":
    main()
