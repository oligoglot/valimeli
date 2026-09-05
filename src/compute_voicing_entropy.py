#!/usr/bin/env python3
"""
Project ValiMeli — Information-Theoretic Conditional Voicing Entropy Audit
==========================================================================
Computes the empirical conditional entropy H(Voicing | Phonotactic Context)
and argmax error reduction for plosives in Tamil and Malayalam using a 
Dynamic Programming (DP) eḻuttu aligner over Aksharantar data.

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

# Tamil Grammatical Terms (Tolkāppiyam, Niklas 1988)
# - Eḻuttu (எழுத்து): Basic orthographic / syllabic grapheme unit
# - Puḷḷi (புள்ளி): Virama / dot indicating pure consonant
# - Vallinam (வல்லினம்): Hard stops / plosives
# - Mellinam (மெல்லினம்): Soft nasals
# - Puṇarcci (புணர்ச்சி): Morphophonemic juncture / sandhi

TAMIL_VALLINAM = {'க', 'ச', 'ட', 'த', 'ப', 'ற'}
TAMIL_PLOSIVES = TAMIL_VALLINAM
TAMIL_MELLINAM = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}
TAMIL_NASALS = TAMIL_MELLINAM
TAMIL_VOWEL_SIGNS = {
    '\u0bbe': ['aa', 'a'], '\u0bbf': ['i', 'ee', 'e'], '\u0bc0': ['ee', 'ii', 'i'],
    '\u0bc1': ['u', 'oo'], '\u0bc2': ['oo', 'uu', 'u'], '\u0bc6': ['e'],
    '\u0bc7': ['e', 'ee', 'ae'], '\u0bc8': ['ai', 'ay', 'ey'],
    '\u0bca': ['o'], '\u0bcb': ['o', 'oo', 'oa'], '\u0bcc': ['au', 'av', 'ow']
}
TAMIL_PULLI = '\u0bcd'
TAMIL_VIRAMA = TAMIL_PULLI

MALAYALAM_PLOSIVES = {'ക', 'ച', 'ട', 'ത', 'പ', 'റ'}
MALAYALAM_NASALS = {'ങ', 'ഞ', 'ണ', 'ന', 'മ'}
MALAYALAM_VOWEL_SIGNS = {
    '\u0d3e': ['aa', 'a'], '\u0d3f': ['i', 'ee', 'e'], '\u0d40': ['ee', 'ii', 'i'],
    '\u0d41': ['u', 'oo'], '\u0d42': ['oo', 'uu', 'u'], '\u0d43': ['ri', 'ru'], '\u0d44': ['ri'],
    '\u0d46': ['e'], '\u0d47': ['e', 'ee', 'ae'], '\u0d48': ['ai', 'ay', 'ey'],
    '\u0d4a': ['o'], '\u0d4b': ['o', 'oo', 'oa'], '\u0d4c': ['au', 'av', 'ow']
}
MALAYALAM_CHANDRAKKALA = '\u0d4d'
MALAYALAM_VIRAMA = MALAYALAM_CHANDRAKKALA

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
    'ழ': {'all': ['zh', 'z', 'l', 'r']}, 'ള': {'all': ['l', 'll']},
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
    'ഏ': ['e', 'ee', 'ae'], 'ഐ': ['ai', 'ay'], 'ஒ': ['o'], 'ഓ': ['o', 'oo'], 'ഔ': ['au', 'av', 'ow']
}

def segment_eluttu(word: str, lang: str = "tam") -> List[str]:
    """Segments a native word into eḻuttu (syllabic graphemic units)."""
    vowel_signs = TAMIL_VOWEL_SIGNS if lang == "tam" else MALAYALAM_VOWEL_SIGNS
    pulli = TAMIL_PULLI if lang == "tam" else MALAYALAM_CHANDRAKKALA
    units = []
    current = ""
    for char in word:
        if char in vowel_signs or char == pulli:
            current += char
        else:
            if current:
                units.append(current)
            current = char
    if current:
        units.append(current)
    return units

# Backwards compatibility alias
segment_aksharas = segment_eluttu

def get_eluttu_candidates(el: str, lang: str = "tam") -> List[Tuple[str, Optional[str]]]:
    """Generates Romanization candidates for an individual eḻuttu."""
    pulli = TAMIL_PULLI if lang == "tam" else MALAYALAM_CHANDRAKKALA
    vowel_signs = TAMIL_VOWEL_SIGNS if lang == "tam" else MALAYALAM_VOWEL_SIGNS
    c_map = CONSONANT_MAP_TAMIL if lang == "tam" else CONSONANT_MAP_MALAYALAM
    v_letters = VOWEL_LETTERS_TAMIL if lang == "tam" else VOWEL_LETTERS_MALAYALAM
    
    if el in v_letters:
        return [(v, None) for v in v_letters[el]]
        
    base_char = el[0]
    has_pulli = pulli in el
    
    vowel_cands = [""] if has_pulli else ["a", "u", ""]
    for char in el[1:]:
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

# Backwards compatibility alias
get_akshara_candidates = get_eluttu_candidates

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

def align_word(eluttukkal: List[str], roman: str, lang: str = "tam", max_budget: int = 2) -> Optional[List[Tuple[str, int, int, Optional[str]]]]:
    """Aligns an eḻuttu sequence to a Latin transliteration string via dynamic programming."""
    n_el = len(eluttukkal)
    n_rom = len(roman)
    
    el_cands = [get_eluttu_candidates(el, lang=lang) for el in eluttukkal]
    dp = [[(float('inf'), -1, -1) for _ in range(n_rom + 1)] for _ in range(n_el + 1)]
    dp[0][0] = (0, 0, -1)
    
    for i in range(n_el):
        for j in range(n_rom + 1):
            curr_cost, _, _ = dp[i][j]
            if curr_cost == float('inf'):
                continue
                
            for c_idx, (cand_str, _) in enumerate(el_cands[i]):
                len_cand = len(cand_str)
                for k in range(j, min(n_rom + 1, j + len_cand + 3)):
                    span = roman[j:k]
                    cost = edit_distance(cand_str, span)
                    if curr_cost + cost < dp[i + 1][k][0]:
                        dp[i + 1][k] = (curr_cost + cost, j, c_idx)
                        
    best_total_cost, prev_j, best_cand_idx = dp[n_el][n_rom]
    if best_total_cost > max_budget:
        return None
        
    alignment = []
    curr_j = n_rom
    for i in range(n_el, 0, -1):
        cost, p_j, c_idx = dp[i][curr_j]
        cand_str, voicing = el_cands[i - 1][c_idx]
        alignment.append((eluttukkal[i - 1], p_j, curr_j, voicing))
        curr_j = p_j
        
    alignment.reverse()
    return alignment

def get_phonotactic_context(word: str, eluttu_idx: int, eluttukkal: List[str], lang: str = "tam") -> str:
    """Computes the phonotactic context of a plosive (vallinam) eḻuttu."""
    plosives = TAMIL_VALLINAM if lang == "tam" else MALAYALAM_PLOSIVES
    nasals = TAMIL_MELLINAM if lang == "tam" else MALAYALAM_NASALS
    pulli = TAMIL_PULLI if lang == "tam" else MALAYALAM_CHANDRAKKALA
    
    current = eluttukkal[eluttu_idx]
    if current[0] not in plosives:
        return "NONE"
        
    # If this is the pure consonant first half of a geminate, skip it (label on second half)
    if pulli in current:
        if eluttu_idx + 1 < len(eluttukkal) and eluttukkal[eluttu_idx + 1][0] == current[0]:
            return "NONE"
            
    if eluttu_idx > 0:
        prev = eluttukkal[eluttu_idx - 1]
        if pulli in prev and prev[0] == current[0]:
            return "GEMINATE"
        if pulli in prev and prev[0] in nasals:
            return "POST_NASAL"
        if pulli in prev:
            return "POST_CONS"
            
    if eluttu_idx == 0:
        return "WORD_INITIAL"
        
    if eluttu_idx > 0:
        prev = eluttukkal[eluttu_idx - 1]
        if pulli not in prev:
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
    pulli = TAMIL_PULLI if lang == "tam" else MALAYALAM_CHANDRAKKALA
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
    plosives_set = TAMIL_VALLINAM if lang == "tam" else MALAYALAM_PLOSIVES
    
    for indic, roman, src in pairs:
        eluttukkal = segment_eluttu(indic, lang=lang)
        for idx, el in enumerate(eluttukkal):
            if el[0] in plosives_set:
                # Do not count the first half of a geminate separately
                if pulli in el and idx + 1 < len(eluttukkal) and eluttukkal[idx + 1][0] == el[0]:
                    continue
                total_plosives += 1
                
        res = align_word(eluttukkal, roman.lower().strip(), lang=lang, max_budget=2)
        if res is not None:
            aligned_words += 1
            for idx, (el, start_idx, end_idx, voicing) in enumerate(res):
                if el[0] in plosives_set and voicing is not None:
                    ctx = get_phonotactic_context(indic, idx, eluttukkal, lang=lang)
                    if ctx == "NONE":
                        continue
                    labelled_plosives += 1
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
