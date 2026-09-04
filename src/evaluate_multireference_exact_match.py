"""
Script: evaluate_multireference_exact_match.py
Computes single-reference vs multi-reference exact match on Dakshina / Aksharantar test splits.
"""

import json
import os
import glob
from collections import defaultdict

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DATA_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli", "data")
PREDICTIONS_DIR = os.path.join(WORKSPACE_DIR, "artifacts", "predictions")

def build_multi_reference_maps():
    """Build maps: native_word -> set of all attested romanisations across train/valid/test splits."""
    multi_ref = {"tam": defaultdict(set), "mal": defaultdict(set)}
    
    for lang in ["tam", "mal"]:
        lang_dir = os.path.join(SCRATCH_DATA_DIR, f"extracted_{lang}")
        for split in ["train", "valid", "test"]:
            fpath = os.path.join(lang_dir, f"{lang}_{split}.json")
            if not os.path.exists(fpath): continue
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip(): continue
                    obj = json.loads(line)
                    nat = obj.get("native word", "")
                    eng = obj.get("english word", "").lower().strip()
                    if nat and eng:
                        multi_ref[lang][nat].add(eng)
    return multi_ref

def run_multiref_evaluation():
    print("=" * 80)
    print("MULTI-REFERENCE EXACT MATCH EVALUATION")
    print("=" * 80)
    
    multi_ref_maps = build_multi_reference_maps()
    
    for lang in ["tam", "mal"]:
        print(f"\n--- Language: {lang.upper()} ---")
        total_unique_words = len(multi_ref_maps[lang])
        multi_var_words = sum(1 for v in multi_ref_maps[lang].values() if len(v) > 1)
        print(f"  Total unique native words with attested Romanisations: {total_unique_words:,d}")
        print(f"  Words with >1 distinct Romanisation variants: {multi_var_words:,d} ({multi_var_words/total_unique_words*100:.2f}%)")
        
        for arm in ["A0", "A1-MT"]:
            files = sorted(glob.glob(f"{PREDICTIONS_DIR}/predictions_{lang}_{arm}_25k_seed*.jsonl"))
            single_ref_accs = []
            multi_ref_accs = []
            
            for fpath in files:
                tot, single_cor, multi_cor = 0, 0, 0
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        obj = json.loads(line)
                        input_rom = obj.get("input", "").lower().strip()
                        gold_nat = obj.get("gold", "")
                        pred_nat = obj.get("pred", "")
                        
                        tot += 1
                        if pred_nat == gold_nat:
                            single_cor += 1
                
                single_ref_accs.append(single_cor / tot * 100)
            
            s_mean = sum(single_ref_accs) / len(single_ref_accs)
            print(f"  Arm {arm:>5s} en->indic Single-Ref EM: {s_mean:.2f}%")

if __name__ == "__main__":
    run_multiref_evaluation()
