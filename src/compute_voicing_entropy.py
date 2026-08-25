#!/usr/bin/env python3
"""
Project ValiMeli — Information-Theoretic Conditional Voicing Entropy Audit
==========================================================================
Computes the empirical conditional entropy H(Voicing | Phonotactic Context)
and argmax error reduction for plosives in Tamil and Malayalam using a 
Dynamic Programming (DP) akshara aligner over Aksharantar data.

Outputs verified entropy metrics and coverage to artifacts/voicing_entropy_results.json.
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
    '\u0bbe': ['aa', 'a'], '\u0bbf': ['i', 'ee', 'e'], '\u0bc0': ['ee', 'ii', 'i'],
    '\u0bc1': ['u', 'oo'], '\u0bc2': ['oo', 'uu', 'u'], '\u0bc6': ['e'],
    '\u0bc7': ['e', 'ee', 'ae'], '\u0bc8': ['ai', 'ay', 'ey'],
    '\u0bca': ['o'], '\u0bcb': ['o', 'oo', 'oa'], '\u0bcc': ['au', 'av', 'ow']
}
TAMIL_VIRAMA = '\u0bcd'

MALAYALAM_PLOSIVES = {'ക', 'ച', 'ട', 'ത', 'പ', 'റ'}
MALAYALAM_NASALS = {'ങ', 'ഞ', 'ണ', 'ന', 'മ'}
MALAYALAM_VOWEL_SIGNS = {
    '\u0d3e': ['aa', 'a'], '\u0d3f': ['i', 'ee', 'e'], '\u0d40': ['ee', 'ii', 'i'],
    '\u0d41': ['u', 'oo'], '\u0d42': ['oo', 'uu', 'u'], '\u0d43': ['ri', 'ru'], '\u0d44': ['ri'],
    '\u0d46': ['e'], '\u0d47': ['e', 'ee', 'ae'], '\u0d48': ['ai', 'ay', 'ey'],
    '\u0d4a': ['o'], '\u0d4b': ['o', 'oo', 'oa'], '\u0d4c': ['au', 'av', 'ow']
}
MALAYALAM_VIRAMA = '\u0d4d'

CONSONANT_MAP_TAMIL = {
    'க': {'voiceless': ['k', 'c', 'q', 'ck', 'kh', 'x'], 'voiced': ['g', 'gh']},
    'ச': {'voiceless': ['ch', 'c', 's', 'sh', 'ts'], 'voiced': ['j', 'z', 'jh']},
    'ட': {'voiceless': ['t', 'tt', 'th'], 'voiced': ['d', 'dd', 'dh']},
    'த': {'voiceless': ['th', 't'], 'voiced': ['dh', 'd']},
    'ப': {'voiceless': ['p', 'pp'], 'voiced': ['b', 'bb', 'bh', 'v']},
    'ற': {'voiceless': ['tr', 'tt', 't', 'rh'], 'voiced': ['dr', 'd']},
    'ங': {'all': ['ng', 'n']}, 'ஞ': {'all': ['gn', 'nj', 'ny', 'n']},
    'ண': {'all': ['n', 'nn']}, 'ந': {'all': ['n', 'nh']}, 'ம': {'all': ['m', 'mm']}, 'ன': {'all': ['n', 'nn']},
    'ய': {'all': ['y', 'yy']}, 'ர': {'all': ['r', 'rr']}, 'ல': {'all': ['l', 'll']}, 'வ': {'all': ['v', 'w']},
    'ழ': {'all': ['zh', 'z', 'l', 'r']}, 'ள': {'all': ['l', 'll']},
    'ஶ': {'all': ['sh', 's']}, 'ஷ': {'all': ['sh', 's']}, 'ஸ': {'all': ['s', 'sh']}, 'ஹ': {'all': ['h']}, 'ஜ': {'all': ['j']}
}

CONSONANT_MAP_MALAYALAM = {
    'ക': {'voiceless': ['k', 'c', 'q', 'ck', 'kh'], 'voiced': ['g', 'gh']},
    'ച': {'voiceless': ['ch', 'c', 's', 'sh'], 'voiced': ['j', 'z', 'jh']},
    'ട': {'voiceless': ['t', 'tt', 'th'], 'voiced': ['d', 'dd', 'dh']},
    'ത': {'voiceless': ['th', 't'], 'voiced': ['dh', 'd']},
    'പ': {'voiceless': ['p', 'pp'], 'voiced': ['b', 'bb', 'bh', 'v']},
    'റ': {'voiceless': ['tr', 'tt', 't', 'rh'], 'voiced': ['dr', 'd']},
    'ങ': {'all': ['ng', 'n']}, 'ഞ': {'all': ['nj', 'ny', 'n']},
    'ണ': {'all': ['n', 'nn']}, 'ന': {'all': ['n', 'nh']}, 'മ': {'all': ['m', 'mm']},
    'യ': {'all': ['y']}, 'ര': {'all': ['r', 'rr']}, 'ല': {'all': ['l', 'll']}, 'വ': {'all': ['v', 'w']},
    'ഴ': {'all': ['zh', 'z', 'l', 'r']}, 'ള': {'all': ['l', 'll']},
    'ശ': {'all': ['sh', 's']}, 'ഷ': {'all': ['sh', 's']}, 'സ': {'all': ['s']}, 'ഹ': {'all': ['h']}, 'ജ': {'all': ['j']}
}

VOWEL_LETTERS_TAMIL = {
    'அ': ['a', 'u'], 'ஆ': ['aa', 'a'], 'இ': ['i', 'e'], 'ஈ': ['ee', 'ii', 'i'],
    'உ': ['u', 'oo'], 'ஊ': ['oo', 'uu', 'u'], 'எ': ['e'], 'ஏ': ['e', 'ee', 'ae'],
    'ஐ': ['ai', 'ay'], 'ஒ': ['o'], 'ஓ': ['o', 'oo'], 'ஔ': ['au', 'av', 'ow']
}

VOWEL_LETTERS_MALAYALAM = {
    'അ': ['a', 'u'], 'ആ': ['aa', 'a'], 'ഇ': ['i', 'e'], 'ഈ': ['ee', 'ii', 'i'],
    'ഉ': ['u', 'oo'], 'ഊ': ['oo', 'uu', 'u'], 'ഋ': ['ri', 'ru'], 'എ': ['e'],
    'ഏ': ['e', 'ee', 'ae'], 'ഐ': ['ai', 'ay'], 'ഒ': ['o'], 'ഓ': ['o', 'oo'], 'ഔ': ['au', 'av', 'ow']
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

def get_akshara_candidates(ak: str, lang: str = "tam") -> List[Tuple[str, Optional[str]]]:
    virama = TAMIL_VIRAMA if lang == "tam" else MALAYALAM_VIRAMA
    vowel_signs = TAMIL_VOWEL_SIGNS if lang == "tam" else MALAYALAM_VOWEL_SIGNS
    c_map = CONSONANT_MAP_TAMIL if lang == "tam" else CONSONANT_MAP_MALAYALAM
    v_letters = VOWEL_LETTERS_TAMIL if lang == "tam" else VOWEL_LETTERS_MALAYALAM
    
    if ak in v_letters:
        return [(v, None) for v in v_letters[ak]]
        
    base_char = ak[0]
    has_virama = virama in ak
    
    vowel_cands = [""] if has_virama else ["a", "u", ""]
    for char in ak[1:]:
        if char in vowel_signs:
            vowel_cands = vowel_signs[char]
            
    c_info = c_map.get(base_char, None)
    if not c_info:
        return [("", None)]
        
    candidates = []
    if 'voiceless' in c_info:
        for vl in c_info['voiceless']:
            for vw in vowel_cands:
                candidates.append((vl + vw, 'VOICELESS'))
        for vd in c_info['voiced']:
            for vw in vowel_cands:
                candidates.append((vd + vw, 'VOICED'))
    else:
        all_c = c_info.get('all', [base_char])
        for c in all_c:
            for vw in vowel_cands:
                candidates.append((c + vw, None))
                
    return candidates

def edit_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def align_word(aksharas: List[str], roman: str, lang: str = "tam", max_budget: int = 2) -> Optional[List[Tuple[str, int, int, Optional[str]]]]:
    n_ak = len(aksharas)
    n_rom = len(roman)
    
    ak_cands = [get_akshara_candidates(ak, lang=lang) for ak in aksharas]
    dp = [[(float('inf'), -1, -1) for _ in range(n_rom + 1)] for _ in range(n_ak + 1)]
    dp[0][0] = (0, 0, -1)
    
    for i in range(n_ak):
        for j in range(n_rom + 1):
            curr_cost, _, _ = dp[i][j]
            if curr_cost == float('inf'):
                continue
                
            for c_idx, (cand_str, _) in enumerate(ak_cands[i]):
                len_cand = len(cand_str)
                for k in range(j, min(n_rom + 1, j + len_cand + 3)):
                    span = roman[j:k]
                    cost = edit_distance(cand_str, span)
                    if curr_cost + cost < dp[i + 1][k][0]:
                        dp[i + 1][k] = (curr_cost + cost, j, c_idx)
                        
    best_total_cost, prev_j, best_cand_idx = dp[n_ak][n_rom]
    if best_total_cost > max_budget:
        return None
        
    alignment = []
    curr_j = n_rom
    for i in range(n_ak, 0, -1):
        cost, p_j, c_idx = dp[i][curr_j]
        cand_str, voicing = ak_cands[i - 1][c_idx]
        alignment.append((aksharas[i - 1], p_j, curr_j, voicing))
        curr_j = p_j
        
    alignment.reverse()
    return alignment

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
        if virama in prev:
            return "POST_CONS"
            
    if akshara_idx == 0:
        return "WORD_INITIAL"
        
    if akshara_idx > 0:
        prev = aksharas[akshara_idx - 1]
        if virama not in prev:
            return "INTERVOCALIC"
            
    return "OTHER"

def compute_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0: return 0.0
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
            if not line.strip(): continue
            item = json.loads(line)
            indic = item.get("native word", "")
            roman = item.get("english word", "")
            src = item.get("source", "unknown")
            if indic and roman:
                pairs.append((indic, roman, src))
            if len(pairs) >= max_samples:
                break
                
    total_words = len(pairs)
    aligned_words = 0
    total_plosives = 0
    labelled_plosives = 0
    
    global_counts = Counter()
    context_counts = defaultdict(Counter)
    plosives_set = TAMIL_PLOSIVES if lang == "tam" else MALAYALAM_PLOSIVES
    
    for indic, roman, src in pairs:
        aksharas = segment_aksharas(indic, lang=lang)
        for idx, ak in enumerate(aksharas):
            if ak[0] in plosives_set:
                total_plosives += 1
                
        res = align_word(aksharas, roman.lower().strip(), lang=lang, max_budget=2)
        if res is not None:
            aligned_words += 1
            for idx, (ak, start_idx, end_idx, voicing) in enumerate(res):
                if ak[0] in plosives_set and voicing is not None:
                    labelled_plosives += 1
                    ctx = get_phonotactic_context(indic, idx, aksharas, lang=lang)
                    global_counts[voicing] += 1
                    context_counts[ctx][voicing] += 1
                    
    word_cov = round((aligned_words / total_words) * 100, 2)
    plos_cov = round((labelled_plosives / total_plosives) * 100, 2)
    h_uncond = round(compute_entropy(global_counts), 4)
    
    context_entropies = {}
    weighted_h_cond = 0.0
    
    for ctx in ["WORD_INITIAL", "GEMINATE", "POST_CONS", "INTERVOCALIC", "POST_NASAL", "OTHER"]:
        c = context_counts[ctx]
        n = sum(c.values())
        if n == 0: continue
        h_ctx = compute_entropy(c)
        weight = n / labelled_plosives
        weighted_h_cond += weight * h_ctx
        context_entropies[ctx] = {
            "entropy_bits": round(h_ctx, 4),
            "sample_count": n,
            "p_voiced": round(c['VOICED'] / n, 4),
            "p_voiceless": round(c['VOICELESS'] / n, 4)
        }
        
    weighted_h_cond = round(weighted_h_cond, 4)
    mi = round(h_uncond - weighted_h_cond, 4)
    entropy_red_pct = round((mi / h_uncond) * 100, 2) if h_uncond > 0 else 0.0
    
    # Argmax accuracies
    global_acc = round((global_counts['VOICELESS'] / labelled_plosives) * 100, 2)
    ctx_correct = sum(max(c.values()) for c in context_counts.values() if c)
    ctx_acc = round((ctx_correct / labelled_plosives) * 100, 2)
    
    return {
        "language": lang,
        "total_words_evaluated": total_words,
        "aligned_words": aligned_words,
        "word_alignment_rate_percent": word_cov,
        "total_plosive_tokens": total_plosives,
        "labelled_plosive_tokens": labelled_plosives,
        "plosive_coverage_percent": plos_cov,
        "unconditioned_entropy_bits": h_uncond,
        "conditional_entropy_bits": weighted_h_cond,
        "mutual_information_bits": mi,
        "entropy_reduction_percent": entropy_red_pct,
        "argmax_accuracy_always_voiceless_percent": global_acc,
        "argmax_accuracy_context_rules_percent": ctx_acc,
        "argmax_error_reduction_gain_percent": round(ctx_acc - global_acc, 2),
        "contexts": context_entropies
    }

def main():
    print("=" * 80)
    print(" PROJECT VALIMELI — DP-ALIGNED CONDITIONAL VOICING ENTROPY AUDIT")
    print("=" * 80)
    
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_file = os.path.join(ARTIFACTS_DIR, "voicing_entropy_results.json")
    
    results = {}
    for lang in ["tam", "mal"]:
        res = run_entropy_analysis(lang, max_samples=150000)
        results[lang] = res
        print(f"\n -> {lang.upper()}: Word Coverage = {res['word_alignment_rate_percent']}% | Plosive Coverage = {res['plosive_coverage_percent']}%")
        print(f"    * H(Voicing): {res['unconditioned_entropy_bits']} -> H(V|C): {res['conditional_entropy_bits']} bits (MI: {res['mutual_information_bits']})")
        print(f"    * Always-Voiceless Argmax: {res['argmax_accuracy_always_voiceless_percent']}% | Context Rules Argmax: {res['argmax_accuracy_context_rules_percent']}% (Gain: {res['argmax_error_reduction_gain_percent']}%)")
        
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print(f"\n⭐ Verified results saved to: {out_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
