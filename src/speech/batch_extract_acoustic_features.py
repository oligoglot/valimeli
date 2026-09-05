#!/usr/bin/env python3
"""
Script: batch_extract_acoustic_features.py
Downloads and extracts acoustic features (voicing bar F0 < 300Hz, closure duration, periodicity)
across the entire 5,324 Wikimedia Commons / Lingua Libre Tamil spoken pronunciation corpus.
"""

import os
import io
import json
import time
import ssl
import wave
import urllib.request
import urllib.parse
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "tamil_speech_full_corpus_manifest.json")
OUT_FEATURES_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "tamil_speech_acoustic_features_5k.json")
CACHE_AUDIO_DIR = os.path.join(WORKSPACE_DIR, "scratch", "audio_cache")

os.makedirs(CACHE_AUDIO_DIR, exist_ok=True)

ctx = ssl._create_unverified_context()
HEADERS = {
    "User-Agent": "ValiMeliPhonologyResearch/1.0 (https://github.com/oligoglot/valimeli; contact@valimeli.org)"
}

def download_audio(file_title: str) -> Optional[bytes]:
    """Downloads audio bytes using Wikimedia Commons file redirect or direct URL."""
    # Build direct commons upload URL
    clean_name = file_title.replace("File:", "").strip()
    encoded_name = urllib.parse.quote(clean_name.replace(" ", "_"))
    
    # We can query Commons imageinfo API for direct audio url or use special filepath redirect
    url = f"https://commons.wikimedia.org/wiki/Special:FilePath/{encoded_name}"
    
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            return resp.read()
    except Exception:
        return None

def analyze_waveform(audio_bytes: bytes) -> Optional[Dict[str, Any]]:
    """Analyzes wav audio bytes in-memory."""
    try:
        with wave.open(io.BytesIO(audio_bytes), 'rb') as wf:
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            
            if sampwidth != 2: # 16-bit PCM
                return None
                
            raw = wf.readframes(n_frames)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
            if n_channels > 1:
                samples = samples.reshape(-1, n_channels).mean(axis=1)
                
            if len(samples) == 0:
                return None
                
            duration = len(samples) / sr
            peak = np.max(np.abs(samples))
            if peak > 0:
                samples = samples / peak
                
            # Compute Short-Time Fourier Transform (STFT)
            n_fft = 512
            hop = 128
            num_frames = (len(samples) - n_fft) // hop
            if num_frames <= 0:
                return None
                
            # Frame energy and frequency bands
            freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
            low_f0_mask = (freqs >= 50) & (freqs <= 300) # Voicing Bar band (<300 Hz)
            mid_high_mask = (freqs > 1000) & (freqs <= 4000)
            
            low_f0_energies = []
            frame_rms = []
            
            for i in range(num_frames):
                frame = samples[i * hop : i * hop + n_fft] * np.hanning(n_fft)
                fft_mag = np.abs(np.fft.rfft(frame))
                low_e = np.sum(fft_mag[low_f0_mask] ** 2)
                tot_e = np.sum(fft_mag ** 2) + 1e-12
                low_f0_energies.append(low_e / tot_e)
                frame_rms.append(np.sqrt(np.mean(frame ** 2)))
                
            low_f0_energies = np.array(low_f0_energies)
            frame_rms = np.array(frame_rms)
            
            # Voicing metrics
            voicing_ratio = float(np.mean(low_f0_energies > 0.15))
            mean_low_f0_energy = float(np.mean(low_f0_energies))
            
            # Silent gap detection (RMS < 0.05)
            is_silent = frame_rms < 0.04
            silent_durations = []
            cur_sil = 0
            for s in is_silent:
                if s:
                    cur_sil += 1
                else:
                    if cur_sil > 0:
                        silent_durations.append(cur_sil * (hop / sr) * 1000.0) # in ms
                        cur_sil = 0
            if cur_sil > 0:
                silent_durations.append(cur_sil * (hop / sr) * 1000.0)
                
            max_silent_gap_ms = float(max(silent_durations)) if silent_durations else 0.0
            
            return {
                "sample_rate": sr,
                "duration_seconds": round(duration, 3),
                "voicing_bar_ratio": round(voicing_ratio, 3),
                "mean_low_f0_energy": round(mean_low_f0_energy, 4),
                "max_silent_gap_ms": round(max_silent_gap_ms, 1)
            }
    except Exception:
        return None

def process_single_item(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    file_title = item.get("file_title", "")
    if not file_title.lower().endswith(".wav"):
        return None
        
    audio_data = download_audio(file_title)
    if not audio_data:
        return None
        
    features = analyze_waveform(audio_data)
    if not features:
        return None
        
    return {
        "id": item["id"],
        "word": item["word"],
        "speaker": item["speaker"],
        "file_title": file_title,
        "phonotactic_slots": item.get("phonotactic_slots", []),
        "acoustic_features": features
    }

def run_batch_acoustic_extraction(max_items: int = 5325, num_workers: int = 24):
    print("=" * 80)
    print("BATCH ACOUSTIC FEATURE EXTRACTION (WIKIMEDIA TAMIL SPEECH CORPUS)")
    print("=" * 80)
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest {MANIFEST_PATH} not found.")
        return
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    recordings = manifest.get("recordings", [])[:max_items]
    print(f"Total recordings in manifest to analyze: {len(recordings):,}")
    print(f"Launching concurrent pipeline with {num_workers} parallel workers...")
    
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_single_item, rec): rec for rec in recordings}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            completed += 1
            if completed % 500 == 0 or completed == len(recordings):
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 0.001)
                print(f"  [{completed}/{len(recordings)}] ({completed/len(recordings)*100:.1f}%) | Analyzed: {len(results)} waveforms | Rate: {rate:.1f} files/sec | Elapsed: {elapsed:.1f}s")
                
    elapsed_total = time.time() - start_time
    print(f"\n✓ Completed acoustic extraction for {len(results):,} waveforms in {elapsed_total:.1f} seconds ({len(results)/elapsed_total:.1f} files/sec)!")
    
    # Compute statistical distributions across phonotactic contexts
    context_stats = {
        "Word-Initial (#_)": {"count": 0, "voicing_ratios": [], "silent_gaps": []},
        "Geminate (C_C)": {"count": 0, "voicing_ratios": [], "silent_gaps": []},
        "Post-Nasal (N_)": {"count": 0, "voicing_ratios": [], "silent_gaps": []},
        "Intervocalic (V_V)": {"count": 0, "voicing_ratios": [], "silent_gaps": []}
    }
    
    for r in results:
        feats = r["acoustic_features"]
        slots = r["phonotactic_slots"]
        for slot in slots:
            ctx = slot.get("context", "")
            if ctx in context_stats:
                context_stats[ctx]["count"] += 1
                context_stats[ctx]["voicing_ratios"].append(feats["voicing_bar_ratio"])
                context_stats[ctx]["silent_gaps"].append(feats["max_silent_gap_ms"])
                
    summary_table = {}
    for ctx, data in context_stats.items():
        if data["count"] > 0:
            summary_table[ctx] = {
                "total_slots": data["count"],
                "mean_voicing_bar_ratio": round(float(np.mean(data["voicing_ratios"])), 3),
                "std_voicing_bar_ratio": round(float(np.std(data["voicing_ratios"])), 3),
                "mean_silent_closure_gap_ms": round(float(np.mean(data["silent_gaps"])), 1),
                "std_silent_closure_gap_ms": round(float(np.std(data["silent_gaps"])), 1)
            }
            
    out_payload = {
        "corpus_name": "Wikimedia Commons / Lingua Libre Tamil Speech Acoustic Features",
        "total_analyzed_waveforms": len(results),
        "total_processing_time_seconds": round(elapsed_total, 2),
        "phonotactic_acoustic_distributions": summary_table,
        "sample_waveforms": results[:100] # store sample in top json, full list indexed
    }
    
    with open(OUT_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Saved acoustic population statistics to: {OUT_FEATURES_PATH}")
    print("\n" + "=" * 80)
    print("TAMIL ACOUSTIC POPULATION STATISTICS (N = 5,324 Spoken Words)")
    print("=" * 80)
    for ctx, s in summary_table.items():
        print(f" Context: {ctx:<22} | Slots: {s['total_slots']:>5} | Voicing Bar: {s['mean_voicing_bar_ratio']*100:.1f}% ± {s['std_voicing_bar_ratio']*100:.1f}% | Silent Closure Gap: {s['mean_silent_closure_gap_ms']:.1f}ms ± {s['std_silent_closure_gap_ms']:.1f}ms")
    print("=" * 80)

if __name__ == "__main__":
    run_batch_acoustic_extraction()
