#!/usr/bin/env python3
"""
Script: validate_voicing_detector.py
Computes an internal heuristic threshold sensitivity and consistency check for the automated
acoustic voicing detector across a stratified sample of 160 consonant slots (40 per context).
NOTE: Baseline labels are rule-assigned based on phonotactic heuristics (not independent human 
expert annotation) to measure threshold sensitivity and quantify the geminate formant-bleed rate.
Outputs: artifacts/acoustic_detector_validation.json
"""

import os
import sys
import json
import math
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src", "speech"))
from batch_extract_acoustic_features import analyze_waveform

MANIFEST_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "tamil_speech_full_corpus_manifest.json")
CACHE_DIR = os.path.join(WORKSPACE_DIR, "data", "audio_cache")
OUT_JSON = os.path.join(WORKSPACE_DIR, "artifacts", "acoustic_detector_validation.json")

def main():
    print("=" * 80)
    print("VALIDATING AUTOMATED VOICING DETECTOR (STRATIFIED SAMPLE N=160)")
    print("=" * 80)

    with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    recordings = manifest.get("recordings", [])
    
    # Target 40 slots per context: Word-Initial, Geminate, Post-Nasal, Intervocalic
    targets = {
        "Word-Initial (#_)": 40,
        "Geminate (C_C)": 40,
        "Post-Nasal (N_)": 40,
        "Intervocalic (V_V)": 40
    }
    
    collected_slots = {k: [] for k in targets}
    
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
            
        for slot in analysis.get("segmented_slots", []):
            ctx = slot.get("context", "")
            if ctx in targets and len(collected_slots[ctx]) < targets[ctx]:
                vb_ratio = slot.get("slot_voicing_bar_ratio", 0.0)
                low_f0 = slot.get("slot_mean_low_f0_energy", 0.0)
                closure_ms = slot.get("slot_silent_closure_ms", 0.0)
                
                # Automated detector prediction (as in compute_acoustic_voicing_entropy.py)
                detector_pred = (vb_ratio >= 0.50 or low_f0 >= 0.25) and (closure_ms < 35.0)
                
                # Ground truth acoustic annotation:
                # 1. Word-Initial (#_): In citation forms, initial stops lack pre-voicing (true voiceless).
                #    True voiced only if continuous voice bar precedes release (rare loan).
                # 2. Geminate (C_C): Fortis occlusion gap > 35ms with voiceless release (true voiceless).
                # 3. Post-Nasal (N_): True voiced stop with continuous glottal periodicity across boundary.
                # 4. Intervocalic (V_V): True voiced if lenis tap with continuous pitch periodicity; voiceless if distinct closure silence >= 25ms
                if ctx == "Word-Initial (#_)":
                    ground_truth = (vb_ratio > 0.70 and low_f0 > 0.40 and closure_ms == 0.0)
                elif ctx == "Geminate (C_C)":
                    ground_truth = False
                elif ctx == "Post-Nasal (N_)":
                    ground_truth = True
                elif ctx == "Intervocalic (V_V)":
                    ground_truth = (closure_ms < 25.0 and low_f0 > 0.20)
                else:
                    ground_truth = False

                collected_slots[ctx].append({
                    "word": rec["word"],
                    "eluttu": slot.get("eluttu", ""),
                    "context": ctx,
                    "vb_ratio": round(vb_ratio, 3),
                    "low_f0": round(low_f0, 3),
                    "closure_ms": round(closure_ms, 1),
                    "detector_pred": bool(detector_pred),
                    "ground_truth": bool(ground_truth)
                })

        if all(len(v) >= targets[k] for k, v in collected_slots.items()):
            break

    # Compute validation metrics per context and overall
    results = {
        "sample_size": sum(len(v) for v in collected_slots.values()),
        "per_context_metrics": {},
        "overall_metrics": {}
    }
    
    overall_tp = 0
    overall_fp = 0
    overall_tn = 0
    overall_fn = 0
    
    print("\nPER-CONTEXT VALIDATION RESULTS:")
    print(f"{'Context':<22} | {'N':<4} | {'TP':<4} | {'FP':<4} | {'TN':<4} | {'FN':<4} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<6} | {'Specificity':<11}")
    print("-" * 95)
    
    for ctx, slots in collected_slots.items():
        tp = sum(1 for s in slots if s["detector_pred"] and s["ground_truth"])
        fp = sum(1 for s in slots if s["detector_pred"] and not s["ground_truth"])
        tn = sum(1 for s in slots if not s["detector_pred"] and not s["ground_truth"])
        fn = sum(1 for s in slots if not s["detector_pred"] and s["ground_truth"])
        
        overall_tp += tp
        overall_fp += fp
        overall_tn += tn
        overall_fn += fn
        
        n = len(slots)
        acc = (tp + tn) / n if n > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        results["per_context_metrics"][ctx] = {
            "n": n,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "specificity": round(spec, 4),
            "false_positive_rate": round(1.0 - spec, 4)
        }
        
        print(f"{ctx:<22} | {n:<4} | {tp:<4} | {fp:<4} | {tn:<4} | {fn:<4} | {acc*100:6.2f}%  | {prec*100:7.2f}%  | {rec*100:5.2f}% | {spec*100:9.2f}%")

    tot_n = overall_tp + overall_fp + overall_tn + overall_fn
    tot_acc = (overall_tp + overall_tn) / tot_n if tot_n > 0 else 0.0
    tot_prec = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    tot_rec = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    tot_spec = overall_tn / (overall_tn + overall_fp) if (overall_tn + overall_fp) > 0 else 0.0
    tot_f1 = 2 * tot_prec * tot_rec / (tot_prec + tot_rec) if (tot_prec + tot_rec) > 0 else 0.0
    
    results["overall_metrics"] = {
        "n": tot_n,
        "tp": overall_tp,
        "fp": overall_fp,
        "tn": overall_tn,
        "fn": overall_fn,
        "accuracy": round(tot_acc, 4),
        "precision": round(tot_prec, 4),
        "recall": round(tot_rec, 4),
        "specificity": round(tot_spec, 4),
        "f1_score": round(tot_f1, 4),
        "false_positive_rate": round(1.0 - tot_spec, 4)
    }
    
    print("-" * 95)
    print(f"{'OVERALL':<22} | {tot_n:<4} | {overall_tp:<4} | {overall_fp:<4} | {overall_tn:<4} | {overall_fn:<4} | {tot_acc*100:6.2f}%  | {tot_prec*100:7.2f}%  | {tot_rec*100:5.2f}% | {tot_spec*100:9.2f}%")
    print(f"Overall F1-Score: {tot_f1*100:.2f}%")
    
    results["sample_slots"] = collected_slots
    
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"\nSaved validation report to: {OUT_JSON}")

if __name__ == "__main__":
    main()
