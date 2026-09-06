#!/usr/bin/env python3
"""
Script: audit_malayalam_alignment.py
Performs an empirical audit of the 34.69% unaligned pairs in Malayalam Aksharantar (52,035 / 150,000).
Tests whether unaligned pairs are disproportionately Sanskrit / Grantha loanwords or complex conjuncts.
"""

import os
import sys
import json
import scipy.stats as stats
from collections import Counter

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
import compute_voicing_entropy as cve

DATA_PATH = os.path.join(WORKSPACE_DIR, "scratch", "valimeli", "data", "extracted_mal", "mal_train.json")
OUT_PATH = os.path.join(WORKSPACE_DIR, "artifacts", "malayalam_alignment_audit.json")

# Sanskrit / Grantha loan phonemes in Malayalam orthography
# (Aspirated series, independent voiced series, sibilants, vocalic liquids)
# Note: ksha (ക്ഷ) decomposes to ka + virama + ssa; ssa (ഷ) is already included.
SANSKRIT_CONSONANTS = set("ഖഘഛഝഠഢഥധഫഭശഷസഹ")
INDEPENDENT_VOICED_STOPS = set("ഗജഡദബ")
SANSKRIT_VOCALIC = set("ഋൠഌൡ")
ALL_LOAN_MARKERS = SANSKRIT_CONSONANTS | INDEPENDENT_VOICED_STOPS | SANSKRIT_VOCALIC

def run_audit(max_pairs: int = 150000):
    print("=" * 80)
    print("EMPIRICAL AUDIT: MALAYALAM ALIGNMENT COVERAGE & LOANWORD COMPOSITION")
    print("=" * 80)
    
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
        
    pairs = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_pairs:
                break
            obj = json.loads(line.strip())
            pairs.append((obj["native word"], obj["english word"]))
            
    print(f"Loaded {len(pairs):,} Malayalam-English training pairs.")
    
    aligned_count = 0
    unaligned_count = 0
    
    aligned_loan = 0
    unaligned_loan = 0
    
    aligned_conjuncts = 0
    unaligned_conjuncts = 0
    
    loan_char_counts = Counter()
    
    for i, (mal, eng) in enumerate(pairs):
        units = cve.segment_eluttu(mal, lang="mal")
        alignment = cve.align_word(units, eng.lower(), lang="mal")
        
        has_loan = any(c in ALL_LOAN_MARKERS for c in mal)
        virama_count = mal.count("\u0d4d")
        has_complex_conjunct = virama_count >= 2
        
        if alignment is not None:
            aligned_count += 1
            if has_loan:
                aligned_loan += 1
            if has_complex_conjunct:
                aligned_conjuncts += 1
        else:
            unaligned_count += 1
            if has_loan:
                unaligned_loan += 1
            if has_complex_conjunct:
                unaligned_conjuncts += 1
                
        for c in mal:
            if c in ALL_LOAN_MARKERS:
                loan_char_counts[c] += 1
                
        if (i + 1) % 25000 == 0 or (i + 1) == len(pairs):
            print(f"  Processed {i+1:>6,} / {len(pairs):,} pairs... (Aligned: {aligned_count:,})")
            
    p_aligned_loan = (aligned_loan / aligned_count) * 100
    p_unaligned_loan = (unaligned_loan / unaligned_count) * 100
    
    p_aligned_conjunct = (aligned_conjuncts / aligned_count) * 100
    p_unaligned_conjunct = (unaligned_conjuncts / unaligned_count) * 100
    
    # Chi-square test of independence for loanword marker presence
    contingency_table = [
        [unaligned_loan, unaligned_count - unaligned_loan],
        [aligned_loan, aligned_count - aligned_loan]
    ]
    chi2_res = stats.chi2_contingency(contingency_table)
    
    odds_ratio = (unaligned_loan / (unaligned_count - unaligned_loan)) / (aligned_loan / (aligned_count - aligned_loan))
    
    results = {
        "dataset": "Aksharantar Malayalam Training Set (first 150k)",
        "total_evaluated_pairs": len(pairs),
        "aligned_pairs": aligned_count,
        "alignment_coverage_pct": round(aligned_count / len(pairs) * 100, 2),
        "unaligned_pairs": unaligned_count,
        "unaligned_pct": round(unaligned_count / len(pairs) * 100, 2),
        "loanword_marker_analysis": {
            "loanword_graphemes_tested": sorted(list(ALL_LOAN_MARKERS)),
            "aligned_loanword_rate_pct": round(p_aligned_loan, 2),
            "unaligned_loanword_rate_pct": round(p_unaligned_loan, 2),
            "odds_ratio": round(float(odds_ratio), 2),
            "chi2_statistic": round(float(chi2_res.statistic), 1),
            "p_value": float(chi2_res.pvalue)
        },
        "complex_conjunct_analysis": {
            "aligned_complex_conjunct_rate_pct": round(p_aligned_conjunct, 2),
            "unaligned_complex_conjunct_rate_pct": round(p_unaligned_conjunct, 2)
        },
        "top_loanword_graphemes_in_corpus": loan_char_counts.most_common(10)
    }
    
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print("\n" + "=" * 80)
    print("AUDIT RESULTS & STATISTICAL FINDINGS")
    print("=" * 80)
    print(f"Total Pairs: {len(pairs):,}")
    print(f"Aligned Coverage: {aligned_count:,} ({aligned_count/len(pairs)*100:.2f}%)")
    print(f"Unaligned Stratum: {unaligned_count:,} ({unaligned_count/len(pairs)*100:.2f}%)")
    print(f"Loanword Rate: Aligned = {p_aligned_loan:.2f}% vs. Unaligned = {p_unaligned_loan:.2f}%")
    print(f"Odds Ratio: {odds_ratio:.2f}x higher odds of loanword graphemes in unaligned words")
    print(f"Chi-square: χ² = {chi2_res.statistic:.1f} (p = {chi2_res.pvalue:.2e})")
    print(f"Complex Conjuncts (≥2 viramas): Aligned = {p_aligned_conjunct:.2f}% vs. Unaligned = {p_unaligned_conjunct:.2f}%")
    print(f"✓ Saved results to: {OUT_PATH}")
    print("=" * 80)

if __name__ == "__main__":
    run_audit()
