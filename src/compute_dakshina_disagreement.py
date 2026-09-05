#!/usr/bin/env python3
"""
Compute Pairwise Annotator Disagreement over Dakshina Multi-Annotator Words
===========================================================================
Measures the empirical disagreement rate between independent Latin transliterations
of identical native-script words across phonotactic slots in Dakshina.
"""

import os
import json
import glob
from collections import defaultdict, Counter
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
    'അ': ['a', 'u'], 'ആ': ['aa', 'a'], 'ഇ': ['i', 'e'], 'ஈ': ['ee', 'ii', 'i'],
    'ഉ': ['u', 'oo'], 'ഊ': ['oo', 'uu', 'u'], 'ഋ': ['ri', 'ru'], 'എ': ['e'],
    'ഏ': ['e', 'ee', 'ae'], 'ഐ': ['ai', 'ay'], 'ഒ': ['o'], 'ഓ': ['o', 'oo'], 'ഔ': ['au', 'av', 'ow']
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

def compute_dakshina_disagreement(lang: str):
    # Search for multi-reference files in Dakshina
    pattern = os.path.join(SCRATCH_DATA_DIR, f"extracted_{lang}", "*dakshina*.json")
    files = glob.glob(pattern)
    if not files:
        pattern = os.path.join(SCRATCH_DATA_DIR, f"extracted_{lang}", f"{lang}_train.json")
        files = glob.glob(pattern)

    word_to_romans = defaultdict(set)
    for fpath in files:
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line)
                indic = item.get("native word", "")
                roman = item.get("english word", "").strip().lower()
                src = item.get("source", "")
                if "dakshina" in src.lower() and indic and roman:
                    word_to_romans[indic].add(roman)

    multi_ref_words = {k: v for k, v in word_to_romans.items() if len(v) > 1}
    plosives_set = TAMIL_VALLINAM if lang == "tam" else MALAYALAM_PLOSIVES

    context_disagree = defaultdict(int)
    context_total = defaultdict(int)

    for indic, romans in multi_ref_words.items():
        eluttukkal = segment_eluttu(indic, lang=lang)
        slot_alignments = defaultdict(list)
        
        for r in romans:
            res = align_word(eluttukkal, r, lang=lang, max_budget=2)
            if res is not None:
                for idx, (el, s, e, voicing) in enumerate(res):
                    if el[0] in plosives_set and voicing is not None:
                        ctx = get_phonotactic_context(indic, idx, eluttukkal, lang=lang)
                        if ctx != "NONE":
                            slot_alignments[idx].append((ctx, voicing))

        for idx, observations in slot_alignments.items():
            if len(observations) >= 2:
                ctx = observations[0][0]
                voicings = [obs[1] for obs in observations]
                for i in range(len(voicings)):
                    for j in range(i + 1, len(voicings)):
                        context_total[ctx] += 1
                        if voicings[i] != voicings[j]:
                            context_disagree[ctx] += 1

    results = {}
    for ctx in ["WORD_INITIAL", "GEMINATE", "POST_CONS", "INTERVOCALIC", "POST_NASAL"]:
        tot = context_total[ctx]
        dis = context_disagree[ctx]
        rate = round((dis / tot) * 100, 2) if tot > 0 else 0.0
        results[ctx] = {"disagreements": dis, "total_pairs": tot, "rate_pct": rate}

    return results

def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    out_file = os.path.join(ARTIFACTS_DIR, "valimeli_disagreement_recomputed.json")
    all_res = {}
    for lang in ["tam", "mal"]:
        all_res[lang] = compute_dakshina_disagreement(lang)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_res, f, indent=2)
    print(f"Saved: {out_file}")

if __name__ == "__main__":
    main()
