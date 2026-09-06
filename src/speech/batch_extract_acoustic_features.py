#!/usr/bin/env python3
"""
Script: batch_extract_acoustic_features.py
Downloads and extracts Short-Time Fourier Transform (STFT) acoustic features
(low-F0 voicing bar energy ratio < 300Hz, silent closure duration, and periodicity)
at both word-level and consonant slot-level across all 5,325 recordings (11,590 plosive slots)
in the Wikimedia Commons / Lingua Libre Tamil speech corpus.
"""

import os
import io
import json
import time
import hashlib
import urllib.parse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import numpy as np
import scipy.io.wavfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, Optional, List

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "tamil_speech_full_corpus_manifest.json")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "data", "audio_cache")
OUT_FEATURES_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "tamil_speech_acoustic_features_5k.json")
OUT_SUMMARY_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "acoustic_wav_analysis_summary.json")

os.makedirs(CACHE_DIR, exist_ok=True)

def get_user_agent() -> str:
    """Retrieves User-Agent from environment variable, gitignored .env, or clean default."""
    if os.environ.get("WIKIMEDIA_USER_AGENT"):
        return os.environ["WIKIMEDIA_USER_AGENT"]
    env_path = os.path.join(WORKSPACE_DIR, ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip().startswith("WIKIMEDIA_USER_AGENT="):
                        val = line.strip().split("=", 1)[1].strip()
                        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                            val = val[1:-1]
                        return val
        except Exception:
            pass
    return "ValiMeliSpeechResearch/1.0 (academic; speech-study@valimeli.org; https://github.com/oligoglot/valimeli)"

USER_AGENT = get_user_agent()

# Global session with connection pooling and automated backoff
session = requests.Session()
session.headers.update({"User-Agent": USER_AGENT})
retries = Retry(total=4, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retries)
session.mount("https://", adapter)
session.mount("http://", adapter)

def get_commons_url(file_title: str) -> str:
    clean_name = file_title.replace("File:", "").strip().replace(" ", "_")
    m = hashlib.md5(clean_name.encode("utf-8")).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{m[0]}/{m[:2]}/{urllib.parse.quote(clean_name)}"

def analyze_waveform(audio_bytes: bytes, word: str, slots: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Analyzes wav audio bytes in-memory using STFT with slot-level segmentation."""
    try:
        sr, samples = scipy.io.wavfile.read(io.BytesIO(audio_bytes))
        if samples is None or len(samples) == 0:
            return None
        samples = samples.astype(np.float32)
        if len(samples.shape) > 1:
            samples = samples.mean(axis=1)
            
        duration = len(samples) / float(sr)
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
        
        # Whole-word metrics
        voicing_ratio = float(np.mean(low_f0_energies > 0.15))
        mean_low_f0_energy = float(np.mean(low_f0_energies))
        
        thresh = 0.05 * np.percentile(frame_rms, 95)
        is_silent = frame_rms < thresh
        silent_durations = []
        cur_sil = 0
        for s in is_silent:
            if s:
                cur_sil += 1
            else:
                if cur_sil > 0:
                    silent_durations.append(cur_sil * (hop / sr) * 1000.0)
                    cur_sil = 0
        if cur_sil > 0:
            silent_durations.append(cur_sil * (hop / sr) * 1000.0)
        max_silent_gap_ms = float(max(silent_durations)) if silent_durations else 0.0
        
        # Slot-level acoustic segmentation
        active = np.where(frame_rms > thresh)[0]
        start_frame = active[0] if len(active) > 0 else 0
        end_frame = active[-1] if len(active) > 0 else num_frames - 1
        
        enriched_slots = []
        for slot in slots:
            eluttu = slot.get("eluttu", "")
            char_idx = word.find(eluttu)
            if char_idx == -1:
                char_idx = 0
            rel_pos = (char_idx + 0.5) / max(len(word), 1)
            
            slot_ctx = slot.get("context", "")
            if "Initial" in slot_ctx:
                s_f = max(0, start_frame - int(0.04 * sr / hop))
                e_f = min(num_frames, start_frame + int(0.10 * sr / hop))
            elif "Geminate" in slot_ctx:
                center_f = start_frame + int(rel_pos * (end_frame - start_frame))
                s_f = max(0, center_f - int(0.08 * sr / hop))
                e_f = min(num_frames, center_f + int(0.08 * sr / hop))
            elif "Post-Nasal" in slot_ctx:
                center_f = start_frame + int(rel_pos * (end_frame - start_frame))
                s_f = max(0, center_f - int(0.08 * sr / hop))
                e_f = min(num_frames, center_f + int(0.08 * sr / hop))
            else: # Intervocalic
                center_f = start_frame + int(rel_pos * (end_frame - start_frame))
                s_f = max(0, center_f - int(0.08 * sr / hop))
                e_f = min(num_frames, center_f + int(0.08 * sr / hop))
                
            if s_f >= e_f:
                s_f, e_f = max(0, start_frame), min(num_frames, end_frame)
                
            s_low_e = low_f0_energies[s_f:e_f]
            s_rms = frame_rms[s_f:e_f]
            s_vb_ratio = float(np.mean(s_low_e > 0.15)) if len(s_low_e) > 0 else 0.0
            s_mean_low_f0 = float(np.mean(s_low_e)) if len(s_low_e) > 0 else 0.0
            s_sil_ms = float(np.sum(s_rms < thresh) * (hop / sr) * 1000.0)
            
            enriched_slots.append({
                "eluttu": slot.get("eluttu", ""),
                "base_grapheme": slot.get("base_grapheme", ""),
                "context": slot.get("context", ""),
                "expected_voicing": slot.get("expected_voicing", ""),
                "slot_time_window_sec": [round(float(s_f * hop / sr), 3), round(float(e_f * hop / sr), 3)],
                "slot_voicing_bar_ratio": round(float(s_vb_ratio), 3),
                "slot_mean_low_f0_energy": round(float(s_mean_low_f0), 4),
                "slot_silent_closure_ms": round(float(s_sil_ms), 1),
                "slot_mean_rms": round(float(np.mean(s_rms)), 4) if len(s_rms) > 0 else 0.0
            })
            
        word_features = {
            "sample_rate": int(sr),
            "duration_seconds": round(duration, 3),
            "voicing_bar_ratio": round(voicing_ratio, 3),
            "mean_low_f0_energy": round(mean_low_f0_energy, 4),
            "max_silent_gap_ms": round(max_silent_gap_ms, 1)
        }
        
        return {
            "word_acoustic_features": word_features,
            "segmented_slots": enriched_slots
        }
    except Exception:
        return None

def process_recording(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    fid = rec["id"]
    cache_path = os.path.join(CACHE_DIR, f"{fid}.wav")
    
    audio_bytes = None
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 1000:
        try:
            with open(cache_path, "rb") as f:
                audio_bytes = f.read()
        except Exception:
            audio_bytes = None
            
    if not audio_bytes:
        url = get_commons_url(rec["file_title"])
        for attempt in range(4):
            try:
                resp = session.get(url, timeout=10)
                if resp.status_code == 200 and len(resp.content) > 1000:
                    audio_bytes = resp.content
                    with open(cache_path, "wb") as f:
                        f.write(audio_bytes)
                    break
                elif resp.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
            except Exception:
                time.sleep(0.3 * (attempt + 1))
                
    if not audio_bytes:
        return None
        
    analysis = analyze_waveform(audio_bytes, rec["word"], rec.get("phonotactic_slots", []))
    if not analysis:
        return None
        
    return {
        "id": rec["id"],
        "word": rec["word"],
        "speaker": rec["speaker"],
        "file_title": rec["file_title"],
        "phonotactic_slots": analysis["segmented_slots"],
        "acoustic_features": analysis["word_acoustic_features"]
    }

def run_batch_acoustic_extraction(max_items: int = 5325, num_workers: int = 8):
    print("=" * 80, flush=True)
    print("BATCH ACOUSTIC FEATURE EXTRACTION (WIKIMEDIA TAMIL SPEECH CORPUS)", flush=True)
    print("=" * 80, flush=True)
    print(f"Authenticated User-Agent: {USER_AGENT}", flush=True)
    print(f"Audio Cache Directory: {CACHE_DIR}\n", flush=True)
    
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: Manifest {MANIFEST_PATH} not found.", flush=True)
        return
        
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    recordings = manifest.get("recordings", [])[:max_items]
    print(f"Total recordings to analyze: {len(recordings):,}", flush=True)
    print(f"Extracting slot-segmented acoustic features across {len(recordings):,} waveforms ({num_workers} worker threads)...", flush=True)
    start_time = time.time()
    results = []
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_recording, rec): rec for rec in recordings}
        completed = 0
        for future in as_completed(futures):
            res = future.result()
            if res:
                results.append(res)
            completed += 1
            if completed % 100 == 0 or completed == len(recordings):
                elapsed = time.time() - start_time
                rate = completed / max(elapsed, 0.001)
                print(f"  [{completed:>5}/{len(recordings):>5}] ({completed/len(recordings)*100:5.1f}%) | Analyzed: {len(results):>5} | Rate: {rate:4.1f} files/s | Elapsed: {elapsed:5.1f}s", flush=True)
                
    elapsed_total = time.time() - start_time
    print(f"\n✓ Completed acoustic extraction for {len(results):,} / {len(recordings):,} waveforms in {elapsed_total:.1f} seconds ({len(results)/elapsed_total:.1f} files/sec)!", flush=True)
    
    # Compute slot-specific statistical distributions across phonotactic contexts
    context_stats = {
        "Word-Initial (#_)": {"count": 0, "voicing_ratios": [], "mean_low_f0": [], "silent_closures": []},
        "Geminate (C_C)": {"count": 0, "voicing_ratios": [], "mean_low_f0": [], "silent_closures": []},
        "Post-Nasal (N_)": {"count": 0, "voicing_ratios": [], "mean_low_f0": [], "silent_closures": []},
        "Intervocalic (V_V)": {"count": 0, "voicing_ratios": [], "mean_low_f0": [], "silent_closures": []}
    }
    
    for r in results:
        slots = r["phonotactic_slots"]
        for slot in slots:
            ctx_name = slot.get("context", "")
            if ctx_name in context_stats:
                context_stats[ctx_name]["count"] += 1
                context_stats[ctx_name]["voicing_ratios"].append(slot["slot_voicing_bar_ratio"])
                context_stats[ctx_name]["mean_low_f0"].append(slot["slot_mean_low_f0_energy"])
                context_stats[ctx_name]["silent_closures"].append(slot["slot_silent_closure_ms"])
                
    summary_table = {}
    for ctx_name, data in context_stats.items():
        if data["count"] > 0:
            summary_table[ctx_name] = {
                "total_slots": data["count"],
                "slot_mean_voicing_bar_ratio": round(float(np.mean(data["voicing_ratios"])), 3),
                "slot_std_voicing_bar_ratio": round(float(np.std(data["voicing_ratios"])), 3),
                "slot_mean_low_f0_energy": round(float(np.mean(data["mean_low_f0"])), 4),
                "slot_std_low_f0_energy": round(float(np.std(data["mean_low_f0"])), 4),
                "slot_mean_silent_closure_ms": round(float(np.mean(data["silent_closures"])), 1),
                "slot_std_silent_closure_ms": round(float(np.std(data["silent_closures"])), 1)
            }
            
    out_payload = {
        "corpus_name": "Wikimedia Commons / Lingua Libre Tamil Speech Acoustic Features",
        "total_analyzed_waveforms": len(results),
        "total_indexed_waveforms": len(recordings),
        "total_processing_time_seconds": round(elapsed_total, 2),
        "user_agent_attribution": "Configured Research User-Agent (Wikimedia Commons Compliant)",
        "phonotactic_acoustic_distributions": summary_table,
        "sample_waveforms": results[:100]
    }
    
    with open(OUT_FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
        
    with open(OUT_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Saved acoustic population statistics to: {OUT_FEATURES_PATH}", flush=True)
    print(f"✓ Updated summary artifact: {OUT_SUMMARY_PATH}", flush=True)
    print("\n" + "=" * 80, flush=True)
    print(f"TAMIL ACOUSTIC POPULATION STATISTICS (SLOT-LEVEL SEGMENTED, N = {len(results):,} Words)", flush=True)
    print("=" * 80, flush=True)
    for ctx_name, s in summary_table.items():
        vb_m = s["slot_mean_voicing_bar_ratio"] * 100
        vb_s = s["slot_std_voicing_bar_ratio"] * 100
        f0_m = s["slot_mean_low_f0_energy"]
        sil_m = s["slot_mean_silent_closure_ms"]
        print(f" Context: {ctx_name:<22} | Slots: {s['total_slots']:>5} | Voice Bar: {vb_m:5.1f}% ± {vb_s:4.1f}% | Low-F0: {f0_m:.3f} | Silence: {sil_m:4.1f}ms", flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    run_batch_acoustic_extraction()
