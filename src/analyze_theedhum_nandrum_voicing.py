#!/usr/bin/env python3
"""
Project ValiMeli — In-the-Wild Tanglish Voicing Analysis on Theedhum Nandrum
=============================================================================
Analyzes spontaneous user typing on YouTube review comments (Theedhum Nandrum / FIRE 2020)
and measures the empirical voicing distribution across phonotactic contexts.
"""

import os
import re
import json
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any, Optional

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SENTIMENT_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli", "sentiment_data")
SCRATCH_DATA_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli", "data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")

from compute_voicing_entropy import (
    segment_eluttu, align_word, get_phonotactic_context,
    TAMIL_VALLINAM, TAMIL_MELLINAM, TAMIL_PULLI
)

def load_tamil_lexicon(max_samples: int = 250000) -> Dict[str, str]:
    """Loads a reference dictionary of canonical Tamil words mapped to common romanizations."""
    lex = {}
    json_path = os.path.join(SCRATCH_DATA_DIR, "extracted_tam", "tam_train.json")
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            for count, line in enumerate(f):
                if count >= max_samples: break
                if not line.strip(): continue
                item = json.loads(line)
                indic = item.get("native word", "")
                roman = item.get("english word", "").lower().strip()
                if indic and roman and len(indic) >= 2:
                    lex[roman] = indic
    return lex

def analyze_theedhum_nandrum():
    print("=" * 80)
    print(" ANALYZING IN-THE-WILD PHONOTACTIC VOICING ON THEEDHUM NANDRUM (YOUTUBE)")
    print("=" * 80)
    
    lex = load_tamil_lexicon()
    tsv_path = os.path.join(SENTIMENT_DIR, "tamil_train.tsv")
    
    context_counts = defaultdict(Counter)
    total_tokens = 0
    aligned_tokens = 0
    
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if not parts: continue
            text = parts[0]
            words = re.findall(r'[a-zA-Z]+', text.lower())
            
            for w in words:
                if len(w) < 2: continue
                total_tokens += 1
                
                # Check if word is in canonical lexicon
                indic = lex.get(w, None)
                if not indic:
                    continue
                    
                eluttukkal = segment_eluttu(indic, lang="tam")
                res = align_word(eluttukkal, w, lang="tam", max_budget=2)
                if res is not None:
                    aligned_tokens += 1
                    for idx, (el, s, e, voicing) in enumerate(res):
                        if el[0] in TAMIL_VALLINAM and voicing is not None:
                            ctx = get_phonotactic_context(indic, idx, eluttukkal, lang="tam")
                            if ctx != "NONE":
                                context_counts[ctx][voicing] += 1
                                
    print(f"Total Tanglish words scanned: {total_tokens:,} | Lexically Aligned: {aligned_tokens:,}")
    print("\n--- IN-THE-WILD PHONOTACTIC CONTEXT DISTRIBUTION ---")
    results = {}
    for ctx in ["WORD_INITIAL", "GEMINATE", "POST_CONS", "INTERVOCALIC", "POST_NASAL"]:
        c = context_counts[ctx]
        tot = sum(c.values())
        if tot == 0: continue
        p_vd = round((c['VOICED'] / tot) * 100, 2)
        p_vl = round((c['VOICELESS'] / tot) * 100, 2)
        print(f"  {ctx:15s} (n={tot:5d}): Voiceless = {p_vl:5.1f}% | Voiced = {p_vd:5.1f}%")
        results[ctx] = {"n": tot, "p_voiceless": p_vl, "p_voiced": p_vd}
        
    out_file = os.path.join(ARTIFACTS_DIR, "theedhum_nandrum_voicing_analysis.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    analyze_theedhum_nandrum()
