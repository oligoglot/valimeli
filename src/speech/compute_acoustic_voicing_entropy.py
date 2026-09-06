#!/usr/bin/env python3
"""
Script: compute_acoustic_voicing_entropy.py
Computes the Acoustic Voicing Entropy H(V_acoustic | C) directly from speech waveforms.
Tests the user's profound insight: In spoken Tamil, does phonotactic context C determine
physical acoustic voicing V_acoustic with near-zero conditional entropy H(V_acoustic | C),
proving that the single graphemic series is mathematically parsimonious and sufficient?
"""

import os
import sys
import json
import math
import numpy as np
from collections import defaultdict

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src", "speech"))
from batch_extract_acoustic_features import analyze_waveform

MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "tamil_speech_full_corpus_manifest.json")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "data", "audio_cache")
OUT_JSON = os.path.join(WORKSPACE_DIR, "artifacts", "acoustic_voicing_entropy_results.json")

def binary_entropy(p: float) -> float:
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))

def compute_acoustic_entropy():
    print("=" * 80)
    print("EMPIRICAL ACOUSTIC VOICING ENTROPY AUDIT (SPOKEN TAMIL CORPUS)")
    print("=" * 80)
    
    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    recordings = manifest.get("recordings", [])
    print(f"Loaded {len(recordings):,} recordings from manifest.")
    
    # Process cached waveforms
    cached_slots = defaultdict(list)
    total_analyzed = 0
    
    for rec in recordings:
        fid = rec["id"]
        cache_file = os.path.join(CACHE_DIR, f"{fid}.wav")
        if not os.path.exists(cache_file) or os.path.getsize(cache_file) < 1000:
            continue
            
        with open(cache_file, "rb") as f:
            audio_bytes = f.read()
            
        analysis = analyze_waveform(audio_bytes, rec["word"], rec.get("phonotactic_slots", []))
        if not analysis:
            continue
            
        total_analyzed += 1
        for slot in analysis["segmented_slots"]:
            ctx = slot.get("context", "")
            vb_ratio = slot.get("slot_voicing_bar_ratio", 0.0)
            low_f0 = slot.get("slot_mean_low_f0_energy", 0.0)
            closure_ms = slot.get("slot_silent_closure_ms", 0.0)
            
            # Physical voicing determination:
            # A slot has glottal voicing if it maintains continuous low-F0 energy (vb_ratio >= 0.50 or low_f0 >= 0.25)
            # and lacks a silent voiceless closure gap (closure_ms < 35ms).
            is_voiced = (vb_ratio >= 0.50 or low_f0 >= 0.25) and (closure_ms < 35.0)
            
            cached_slots[ctx].append({
                "word": rec["word"],
                "eluttu": slot.get("eluttu", ""),
                "is_voiced": is_voiced,
                "vb_ratio": vb_ratio,
                "low_f0": low_f0,
                "closure_ms": closure_ms
            })
            
        if total_analyzed % 500 == 0:
            print(f"  Analyzed {total_analyzed:,} cached waveforms...")
            
    print(f"\nCompleted analysis across {total_analyzed:,} cached speech recordings.")
    
    # Calculate conditional probabilities and entropy
    all_slots_count = sum(len(v) for v in cached_slots.values())
    total_voiced = sum(sum(1 for s in v if s["is_voiced"]) for v in cached_slots.values())
    p_voiced_uncond = total_voiced / all_slots_count if all_slots_count > 0 else 0.0
    h_uncond = binary_entropy(p_voiced_uncond)
    
    context_results = {}
    h_cond = 0.0
    correct_argmax_tokens = 0
    
    print("\n" + "=" * 80)
    print(f"{'Context':<24} | {'Slots':>6} | {'P(Voiced)':>9} | {'Entropy H(V|C)':>14} | {'Phonological Prediction'}")
    print("-" * 80)
    
    for ctx, items in sorted(cached_slots.items()):
        n = len(items)
        n_voiced = sum(1 for s in items if s["is_voiced"])
        p_v = n_voiced / n if n > 0 else 0.0
        h_c = binary_entropy(p_v)
        weight = n / all_slots_count
        h_cond += weight * h_c
        
        # Expected phonological voicing
        expected_voiced = ("Post-Nasal" in ctx or "Intervocalic" in ctx)
        predicted_voiced = (p_v >= 0.50)
        matches_expectation = (predicted_voiced == expected_voiced)
        
        # Argmax accuracy for this context under phonological rule
        if expected_voiced:
            correct_argmax_tokens += n_voiced
        else:
            correct_argmax_tokens += (n - n_voiced)
            
        context_results[ctx] = {
            "n_slots": n,
            "n_voiced": n_voiced,
            "n_voiceless": n - n_voiced,
            "p_voiced": round(p_v, 4),
            "entropy_bits": round(h_c, 4),
            "context_weight": round(weight, 4),
            "mean_vb_ratio": round(float(np.mean([s["vb_ratio"] for s in items])), 3),
            "mean_low_f0": round(float(np.mean([s["low_f0"] for s in items])), 4),
            "mean_closure_ms": round(float(np.mean([s["closure_ms"] for s in items])), 1),
            "phonological_match": matches_expectation
        }
        
        status = "✓ Matches Theory" if matches_expectation else "Discrepancy"
        print(f"{ctx:<24} | {n:>6,} | {p_v*100:>8.2f}% | {h_c:>11.4f} bits | {status}")
        
    mutual_info = h_uncond - h_cond
    entropy_reduction_pct = (mutual_info / h_uncond * 100) if h_uncond > 0 else 0.0
    overall_argmax_acc = correct_argmax_tokens / all_slots_count * 100 if all_slots_count > 0 else 0.0
    
    print("=" * 80)
    print(f"ACOUSTIC VOICING INFORMATION SUMMARY:")
    print(f"  Total Plosive Consonant Slots: {all_slots_count:,} across {total_analyzed:,} audio files")
    print(f"  Unconditional Voicing Entropy H(V):      {h_uncond:.4f} bits (P(voiced) = {p_voiced_uncond*100:.2f}%)")
    print(f"  Conditional Acoustic Entropy H(V | C):  {h_cond:.4f} bits")
    print(f"  Acoustic Mutual Information I(V; C):     {mutual_info:.4f} bits")
    print(f"  Acoustic Entropy Reduction:              {entropy_reduction_pct:.2f}%")
    print(f"  Phonotactic Argmax Prediction Accuracy:  {overall_argmax_acc:.2f}%")
    print("=" * 80)
    
    out_data = {
        "corpus": "Wikimedia Commons / Lingua Libre Tamil Spoken Speech",
        "total_analyzed_waveforms": total_analyzed,
        "total_plosive_slots": all_slots_count,
        "acoustic_voicing_entropy": {
            "H_V_unconditional_bits": round(h_uncond, 4),
            "H_V_given_C_bits": round(h_cond, 4),
            "I_V_C_mutual_info_bits": round(mutual_info, 4),
            "entropy_reduction_pct": round(entropy_reduction_pct, 2),
            "unconditional_p_voiced": round(p_voiced_uncond, 4),
            "phonotactic_rule_accuracy_pct": round(overall_argmax_acc, 2)
        },
        "per_context_distributions": context_results
    }
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out_data, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Saved acoustic entropy results to: {OUT_JSON}")

if __name__ == "__main__":
    compute_acoustic_entropy()
