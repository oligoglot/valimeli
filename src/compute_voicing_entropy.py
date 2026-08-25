#!/usr/bin/env python3
"""
Project ValiMeli — Information-Theoretic Conditional Voicing Entropy Audit
==========================================================================
Computes the empirical conditional entropy H(Voicing | Phonotactic Context)
for plosives in Tamil and Malayalam using alignment over Aksharantar data.

Outputs verifiable entropy metrics and aligner coverage to artifacts/voicing_entropy_results.json.
"""

import os
import sys
import json
import math
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DATA_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli", "data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")

TAMIL_PLOSIVES = {'க', 'ச', 'ட', 'த', 'ப', 'ற'}
TAMIL_NASALS = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}
TAMIL_VOWEL_SIGNS = {
    '\u0bbe', '\u0bbf', '\u0bc0', '\u0bc1', '\u0bc2',
    '\u0bc6', '\u0bc7', '\u0bc8', '\u0bca', '\u0bcb', '\u0bcc'
}
TAMIL_VIRAMA = '\u0bcd'

MALAYALAM_PLOSIVES = {'ക', 'ച', 'ട', 'ത', 'പ', 'റ'}
MALAYALAM_NASALS = {'ങ', 'ഞ', 'ണ', 'ന', 'മ'}
MALAYALAM_VOWEL_SIGNS = {
    '\u0d3e', '\u0d3f', '\u0d40', '\u0d41', '\u0d42', '\u0d43', '\u0d44',
    '\u0d46', '\u0d47', '\u0d48', '\u0d4a', '\u0d4b', '\u0d4c'
}
MALAYALAM_VIRAMA = '\u0d4d'

# Plosive phonetic voicing dictionaries
PLOSIVE_MAP = {
    'க': {'voiceless': ['k', 'c', 'q', 'ck', 'kh', 'x'], 'voiced': ['g', 'gh', 'h']},
    'ക': {'voiceless': ['k', 'c', 'q', 'ck', 'kh'], 'voiced': ['g', 'gh', 'h']},
    'ச': {'voiceless': ['ch', 'c', 's', 'sh', 'ts'], 'voiced': ['j', 'z', 'jh']},
    'ച': {'voiceless': ['ch', 'c', 's', 'sh'], 'voiced': ['j', 'z', 'jh']},
    'ட': {'voiceless': ['t', 'tt', 'th'], 'voiced': ['d', 'dd', 'dh', 'r']},
    'ട': {'voiceless': ['t', 'tt', 'th'], 'voiced': ['d', 'dd', 'dh', 'r']},
    'த': {'voiceless': ['th', 't'], 'voiced': ['dh', 'd']},
    'ത': {'voiceless': ['th', 't'], 'voiced': ['dh', 'd']},
    'ப': {'voiceless': ['p', 'pp', 'ph', 'f'], 'voiced': ['b', 'bb', 'bh', 'v']},
    'പ': {'voiceless': ['p', 'pp', 'ph', 'f'], 'voiced': ['b', 'bb', 'bh', 'v']},
    'ற': {'voiceless': ['t', 'tr', 'tt', 'r', 'rh'], 'voiced': ['d', 'dr', 'r']}
}

def segment_aksharas(word: str, lang: str = "tam") -> List[str]:
    vowel_signs = TAMIL_VOWEL_SIGNS if lang == "tam" else MALAYALAM_VOWEL_SIGNS
    virama = TAMIL_VIRAMA if lang == "tam" else MALAYALAM_VIRAMA
    units = []
    current = ""
    for char in word:
        if char in vowel_signs or char == virama:
            current += char
        else:
            if current:
                units.append(current)
            current = char
    if current:
        units.append(current)
    return units

def get_phonotactic_context(word: str, akshara_idx: int, aksharas: List[str], lang: str = "tam") -> str:
    plosives = TAMIL_PLOSIVES if lang == "tam" else MALAYALAM_PLOSIVES
    nasals = TAMIL_NASALS if lang == "tam" else MALAYALAM_NASALS
    virama = TAMIL_VIRAMA if lang == "tam" else MALAYALAM_VIRAMA
    
    current = aksharas[akshara_idx]
    if current[0] not in plosives:
        return "NONE"
        
    if virama in current:
        if akshara_idx + 1 < len(aksharas) and aksharas[akshara_idx + 1][0] == current[0]:
            return "GEMINATE"
            
    if akshara_idx > 0:
        prev = aksharas[akshara_idx - 1]
        if virama in prev and prev[0] == current[0]:
            return "GEMINATE"
        if virama in prev and prev[0] in nasals:
            return "POST_NASAL"
            
    if akshara_idx == 0:
        return "WORD_INITIAL"
        
    if akshara_idx > 0:
        prev = aksharas[akshara_idx - 1]
        if virama not in prev:
            return "INTERVOCALIC"
            
    return "OTHER"

def align_and_classify_voicing(indic_word: str, roman_word: str, lang: str = "tam") -> List[Tuple[str, str, str]]:
    """
    Returns list of (plosive_char, context, voicing_status)
    voicing_status in {'VOICELESS', 'VOICED'} or None if unaligned.
    """
    aksharas = segment_aksharas(indic_word, lang=lang)
    plosives = TAMIL_PLOSIVES if lang == "tam" else MALAYALAM_PLOSIVES
    
    results = []
    roman_lower = roman_word.lower().strip()
    
    # Track positions of plosive aksharas
    for idx, ak in enumerate(aksharas):
        p_char = ak[0]
        if p_char in plosives:
            context = get_phonotactic_context(indic_word, idx, aksharas, lang=lang)
            if context == "NONE":
                continue
                
            # Alignment heuristic by relative position and phonetic target candidate match
            mapping = PLOSIVE_MAP.get(p_char, {})
            voiceless_cands = mapping.get('voiceless', [])
            voiced_cands = mapping.get('voiced', [])
            
            # Substring search based on relative position
            rel_pos = idx / max(len(aksharas), 1)
            target_char_idx = int(rel_pos * len(roman_lower))
            search_window = roman_lower[max(0, target_char_idx - 2): min(len(roman_lower), target_char_idx + 4)]
            
            found_voicing = None
            
            # Check voiced matches first (e.g., 'b', 'd', 'g', 'dh', 'zh')
            for v in sorted(voiced_cands, key=len, reverse=True):
                if v in search_window:
                    found_voicing = 'VOICED'
                    break
            
            if not found_voicing:
                for vl in sorted(voiceless_cands, key=len, reverse=True):
                    if vl in search_window:
                        found_voicing = 'VOICELESS'
                        break
                        
            if found_voicing:
                results.append((p_char, context, found_voicing))
            else:
                results.append((p_char, context, 'UNALIGNED'))
                
    return results

def compute_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for count in counts.values():
        if count > 0:
            p = count / total
            h -= p * math.log2(p)
    return h

def run_entropy_analysis(lang: str, max_samples: int = 150000) -> Dict[str, Any]:
    json_path = os.path.join(SCRATCH_DATA_DIR, f"extracted_{lang}", f"{lang}_train.json")
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Data not found: {json_path}")
        
    pairs = []
    with open(json_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            indic = item.get("native word", "")
            roman = item.get("english word", "")
            if indic and roman:
                pairs.append((indic, roman))
            if len(pairs) >= max_samples:
                break
                
    total_plosives = 0
    aligned_plosives = 0
    
    global_counts = Counter()
    context_counts = defaultdict(Counter)
    
    for indic, roman in pairs:
        decisions = align_and_classify_voicing(indic, roman, lang=lang)
        for p_char, ctx, status in decisions:
            total_plosives += 1
            if status != 'UNALIGNED':
                aligned_plosives += 1
                global_counts[status] += 1
                context_counts[ctx][status] += 1
                
    coverage_pct = (aligned_plosives / max(total_plosives, 1)) * 100.0
    unconditioned_h = compute_entropy(global_counts)
    
    context_entropies = {}
    weighted_conditional_h = 0.0
    
    for ctx in ["WORD_INITIAL", "GEMINATE", "POST_NASAL", "INTERVOCALIC", "OTHER"]:
        c_counts = context_counts[ctx]
        n_ctx = sum(c_counts.values())
        h_ctx = compute_entropy(c_counts)
        weight = n_ctx / max(aligned_plosives, 1)
        weighted_conditional_h += weight * h_ctx
        context_entropies[ctx] = {
            "entropy_bits": round(h_ctx, 4),
            "sample_count": n_ctx,
            "p_voiced": round(c_counts['VOICED'] / max(n_ctx, 1), 4),
            "p_voiceless": round(c_counts['VOICELESS'] / max(n_ctx, 1), 4)
        }
        
    mutual_info = unconditioned_h - weighted_conditional_h
    entropy_reduction_pct = (mutual_info / max(unconditioned_h, 1e-9)) * 100.0
    
    return {
        "language": lang,
        "samples_evaluated": len(pairs),
        "total_plosive_tokens": total_plosives,
        "aligned_plosive_tokens": aligned_plosives,
        "aligner_coverage_percent": round(coverage_pct, 2),
        "unconditioned_entropy_bits": round(unconditioned_h, 4),
        "conditional_entropy_bits": round(weighted_conditional_h, 4),
        "mutual_information_bits": round(mutual_info, 4),
        "entropy_reduction_percent": round(entropy_reduction_pct, 2),
        "contexts": context_entropies
    }

def main():
    print("=" * 80)
    print(" PROJECT VALIMELI — INFORMATION-THEORETIC CONDITIONAL VOICING ENTROPY AUDIT")
    print("=" * 80)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_file = os.path.join(ARTIFACTS_DIR, "voicing_entropy_results.json")
    
    results = {}
    for lang in ["tam", "mal"]:
        print(f"\n -> Computing Voicing Entropy for {lang.upper()} (up to 150k pairs)...", flush=True)
        res = run_entropy_analysis(lang, max_samples=150000)
        results[lang] = res
        print(f"    * Aligner Coverage:         {res['aligner_coverage_percent']}% ({res['aligned_plosive_tokens']:,}/{res['total_plosive_tokens']:,} plosives)")
        print(f"    * Unconditioned H(Voicing): {res['unconditioned_entropy_bits']} bits")
        print(f"    * Conditional H(V | C):     {res['conditional_entropy_bits']} bits")
        print(f"    * Mutual Information I(V;C):{res['mutual_information_bits']} bits ({res['entropy_reduction_percent']}% reduction)")
        for ctx, data in res["contexts"].items():
            print(f"      - {ctx:<15}: H = {data['entropy_bits']:.4f} bits (n={data['sample_count']:,}, voiceless={data['p_voiceless']*100:.1f}%, voiced={data['p_voiced']*100:.1f}%)")
            
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n⭐ Results saved with full provenance to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
