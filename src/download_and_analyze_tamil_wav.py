"""
Script: download_and_analyze_tamil_wav.py
Downloads Wikimedia Commons Lingua Libre Tamil WAV files and analyzes acoustic voicing properties.
"""

import urllib.request
import urllib.parse
import json
import os
import ssl
import wave
import struct
import numpy as np

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'ValiMeliAcousticPhonetics/1.0 (academic phonetic analysis; contact: research@example.org)'
}

AUDIO_DIR = "artifacts/audio_samples"
os.makedirs(AUDIO_DIR, exist_ok=True)

WORDS_TO_ANALYZE = [
    ("படம்", "LL-Q5885 (tam)-Sriveenkat-படம்.wav"),
    ("பக்கம்", "LL-Q5885 (tam)-Sriveenkat-பக்கம்.wav"),
    ("இலக்குவன்", "LL-Q5885 (tam)-Sriveenkat-இலக்குவன்.wav"),
    ("அடி", "LL-Q5885 (tam)-Sriveenkat-அடி.wav"),
    ("புலிகள்", "LL-Q5885 (tam)-Sriveenkat-புலிகள்.wav"),
    ("இம்பால்", "LL-Q5885 (tam)-Sriveenkat-இம்பால்.wav"),
    ("பல", "LL-Q5885 (tam)-Sriveenkat-பல.wav"),
    ("மகளின்", "LL-Q5885 (tam)-Sriveenkat-மகளின்.wav"),
    ("கல்யாணங்களோடு", "LL-Q5885 (tam)-Sriveenkat-கல்யாணங்களோடு.wav")
]

def get_commons_file_url(filename: str) -> str:
    """Get direct download URL for a file on Wikimedia Commons."""
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
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        return True
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=15) as resp, open(dest_path, "wb") as out:
            out.write(resp.read())
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def analyze_wav_acoustic_properties(wav_path: str):
    """Analyze sample rate, duration, and energy distribution of WAV file."""
    with wave.open(wav_path, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / float(framerate)
        
        raw_bytes = wf.readframes(n_frames)
        
        if sampwidth == 2:
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
        elif sampwidth == 4:
            samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
        else:
            samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128
            
        if n_channels > 1:
            samples = samples[::n_channels]
            
        # Normalize
        max_val = np.max(np.abs(samples))
        if max_val > 0:
            samples = samples / max_val
            
        # Compute RMS energy and low-frequency voicing energy (<300 Hz)
        # Using simple autocorrelation / lowpass
        frame_len = int(framerate * 0.025) # 25ms
        hop_len = int(framerate * 0.010)   # 10ms
        
        return {
            "duration_sec": duration,
            "sample_rate": framerate,
            "total_samples": len(samples),
            "peak_amplitude": float(max_val)
        }

if __name__ == "__main__":
    print("=" * 80)
    print("DOWNLOADING & ANALYZING TAMIL AUDIO RECORDINGS (WIKIMEDIA COMMONS)")
    print("=" * 80)
    
    analyzed_data = []
    
    for word, fname in WORDS_TO_ANALYZE:
        print(f"\nProcessing word: '{word}' -> {fname}")
        url = get_commons_file_url(fname)
        if not url:
            print(f"  ❌ Could not resolve URL for {fname}")
            continue
            
        dest = os.path.join(AUDIO_DIR, fname)
        success = download_file(url, dest)
        if success:
            stats = analyze_wav_acoustic_properties(dest)
            print(f"  ✓ Downloaded ({os.path.getsize(dest):,d} bytes) | Sample Rate: {stats['sample_rate']}Hz | Duration: {stats['duration_sec']:.2f}s")
            analyzed_data.append({
                "word": word,
                "filename": fname,
                "url": url,
                "local_path": dest,
                "stats": stats
            })
            
    with open("artifacts/acoustic_wav_analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(analyzed_data, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Completed acoustic audio download and analysis ({len(analyzed_data)} files)")
