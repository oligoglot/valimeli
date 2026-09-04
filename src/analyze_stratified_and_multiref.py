"""
Script: analyze_stratified_and_multiref.py
Audits instance predictions across Native vs. Loanword strata, and measures Multi-Reference Exact Match.
"""

import json
import os
import glob
from typing import Dict, List, Set

PREDICTIONS_DIR = "artifacts/predictions"

# Tamil Grantha loan characters: ja, ssa, sa, ha, sha, and Sri ligature
TAM_GRANTHA = {'\u0b9c', '\u0bb7', '\u0bb8', '\u0bb9', '\u0bb6', '\u0bf9'}
# Malayalam Grantha voiced/aspirated stop characters & sibilants
MAL_GRANTHA = {'\u0d16', '\u0d17', '\u0d18', '\u0d1a', '\u0d1b', '\u0d1c', '\u0d1d', '\u0d20', '\u0d21', '\u0d22', '\u0d25', '\u0d26', '\u0d27', '\u0d2b', '\u0d2c', '\u0d2d', '\u0d36', '\u0d37', '\u0d38', '\u0d39'}

def is_native_tam(word: str) -> bool:
    return not any(c in TAM_GRANTHA for c in word)

def is_native_mal(word: str) -> bool:
    return not any(c in MAL_GRANTHA for c in word)

def run_stratified_audit():
    print("=" * 80)
    print("STRATIFIED NATIVE VS LOANWORD ACCURACY AUDIT")
    print("=" * 80)
    
    results = {}
    
    for lang, is_native_fn in [("tam", is_native_tam), ("mal", is_native_mal)]:
        results[lang] = {}
        for arm in ["A0", "A1-MT"]:
            results[lang][arm] = {
                "native_correct": [],
                "native_total": 0,
                "loan_correct": [],
                "loan_total": 0
            }
            files = sorted(glob.glob(f"{PREDICTIONS_DIR}/predictions_{lang}_{arm}_25k_seed*.jsonl"))
            for fpath in files:
                n_nat_tot, n_nat_cor = 0, 0
                n_loan_tot, n_loan_cor = 0, 0
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        obj = json.loads(line)
                        target = obj.get("gold", "")
                        pred = obj.get("pred", "")
                        correct = obj.get("correct", False)
                        
                        if is_native_fn(target):
                            n_nat_tot += 1
                            if correct:
                                n_nat_cor += 1
                        else:
                            n_loan_tot += 1
                            if correct:
                                n_loan_cor += 1
                
                results[lang][arm]["native_correct"].append(n_nat_cor / max(1, n_nat_tot) * 100)
                results[lang][arm]["native_total"] = n_nat_tot
                results[lang][arm]["loan_correct"].append(n_loan_cor / max(1, n_loan_tot) * 100)
                results[lang][arm]["loan_total"] = n_loan_tot
                
        # Compute summary
        print(f"\n--- Language: {lang.upper()} ---")
        for arm in ["A0", "A1-MT"]:
            nat_vals = results[lang][arm]["native_correct"]
            loan_vals = results[lang][arm]["loan_correct"]
            nat_mean = sum(nat_vals) / len(nat_vals)
            loan_mean = sum(loan_vals) / len(loan_vals)
            print(f"  Arm {arm:>5s}: Native ({results[lang][arm]['native_total']:,d} words) = {nat_mean:.2f}% | Loan ({results[lang][arm]['loan_total']:,d} words) = {loan_mean:.2f}%")
            
    # Save artifact
    out_file = "artifacts/stratified_native_loan_analysis.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved stratified analysis to {out_file}")

if __name__ == "__main__":
    run_stratified_audit()
