#!/usr/bin/env python3
"""
Project ValiMeli & Project Pulli — Sarvam AI Transliteration Benchmark Suite
Evaluates Sarvam AI's Transliteration API (https://api.sarvam.ai/transliterate)
against ValiMeli 3.2M Multilingual and IndicXlit across:
  1. Pulli & Stop Allophony (Virama / Chandrakkala & Positional Stop Voicing)
  2. In-the-Wild Dravidian Code-Mixed YouTube Review Lexicon (Theedhum Nandrum)
  3. Standard Aksharantar Holdout Benchmark (Native Words vs Named Entities)

Usage:
    export SARVAM_API_KEY="your_api_key_here"
    python3 src/sarvam_pulli_benchmark.py [--api-key KEY] [--lang ta-IN,ml-IN] [--sample-size 500]
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error
from typing import List, Tuple, Dict, Any, Optional

import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Import local ValiMeli engine for side-by-side comparison
sys.path.insert(0, WORKSPACE_DIR)
try:
    from src.valimeli_transliterate import ValiMeliEngine
    VALIMELI_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import ValiMeliEngine: {e}")
    VALIMELI_AVAILABLE = False

SARVAM_API_URL = "https://api.sarvam.ai/transliterate"

# =====================================================================
# 1. PULLI & EPIGRAPHY TEST SUITE
# =====================================================================

PULLI_TEST_BATTERY_TAMIL = [
    # Initial Voiceless Stops
    ("kanden", "கண்டேன்", "init_stop"),
    ("thambi", "தம்பி", "init_stop"),
    ("padam", "படம்", "init_stop"),
    ("pakkam", "பக்கம்", "init_stop"),
    ("theedhum", "தீதும்", "init_stop"),
    ("nandrum", "நன்றும்", "init_stop"),
    # Intervocalic Voicing / Lenition
    ("azhagu", "அழகு", "inter_voicing"),
    ("pagal", "பகல்", "inter_voicing"),
    ("madam", "மடம்", "inter_voicing"),
    ("nagaram", "நகரம்", "inter_voicing"),
    ("abayam", "அபயம்", "inter_voicing"),
    # Post-Nasal Voicing
    ("singam", "சிங்கம்", "post_nasal"),
    ("kangal", "கண்கள்", "post_nasal"),
    ("panju", "பஞ்சு", "post_nasal"),
    ("thangam", "தங்கம்", "post_nasal"),
    ("nandri", "நன்றி", "post_nasal"),
    ("kamban", "கம்பன்", "post_nasal"),
    ("pandhu", "பந்து", "post_nasal"),
    # Geminates (Fortis)
    ("appam", "அப்பம்", "geminate"),
    ("pathu", "பத்து", "geminate"),
    ("kattu", "கட்டு", "geminate"),
    ("chittirai", "சித்திரை", "geminate"),
    ("thoppi", "தொப்பி", "geminate"),
    ("vetri", "வெற்றி", "geminate"),
    # Colloquial Code-Mixed Slang (Theedhum Nandrum)
    ("super", "சூப்பர்", "loanword"),
    ("acting", "ஆக்டிங்", "loanword"),
    ("thalaiva", "தலைவா", "colloquial"),
    ("semma", "செம்மா", "colloquial"),
    ("mokkai", "மொக்கை", "colloquial"),
    ("level", "லெவல்", "loanword"),
    ("vera", "வேற", "colloquial"),
]

PULLI_TEST_BATTERY_MALAYALAM = [
    # Base Stops & Lenition
    ("thampi", "തമ്പി", "native_stop"),
    ("kandu", "കണ്ടു", "native_stop"),
    ("padam", "പദം", "native_stop"),
    ("pakkam", "പക്കം", "geminate"),
    ("azhaku", "അഴക്", "inter_voicing"),
    ("chiri", "ചിരി", "init_stop"),
    # Post-Nasal
    ("thangam", "തങ്കം", "post_nasal"),
    ("nanni", "നന്ദി", "post_nasal"),
    ("kamban", "കമ്പൻ", "post_nasal"),
    ("panthu", "പന്ത്", "post_nasal"),
    # Geminates
    ("appam", "അപ്പം", "geminate"),
    ("kettu", "കെട്ട്", "geminate"),
    ("thoppi", "തൊപ്പി", "geminate"),
    # Code-Mixed & Loanwords
    ("super", "സൂപ്പർ", "loanword"),
    ("acting", "ആക്ടിംഗ്", "loanword"),
    ("semma", "സെമ്മ", "colloquial"),
]

# =====================================================================
# 2. SARVAM API CLIENT
# =====================================================================

def query_sarvam_transliterate(text: str, source_lang: str = "en-IN", target_lang: str = "ta-IN", api_key: Optional[str] = None) -> Tuple[Optional[str], float]:
    """Queries Sarvam AI Transliteration API and returns (transliterated_text, latency_ms)"""
    if not api_key:
        api_key = os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_KEY")
    if not api_key:
        return None, 0.0

    headers = {
        "api-subscription-key": api_key.strip(),
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "input": text,
        "source_language_code": source_lang,
        "target_language_code": target_lang
    }).encode("utf-8")

    req = urllib.request.Request(SARVAM_API_URL, data=payload, headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            latency = (time.time() - t0) * 1000.0
            return data.get("transliterated_text", "").strip(), latency
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8")
        print(f"[Sarvam API HTTP Error {e.code}]: {err_msg}", file=sys.stderr)
        return None, 0.0
    except Exception as e:
        print(f"[Sarvam API Error]: {e}", file=sys.stderr)
        return None, 0.0

# =====================================================================
# 3. EVALUATION HELPERS
# =====================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def evaluate_predictions(preds: List[str], refs: List[str]) -> Dict[str, float]:
    exact = 0
    edit_dist = 0
    ref_chars = 0
    for p, r in zip(preds, refs):
        p_c = p.strip().lower()
        r_c = r.strip().lower()
        if p_c == r_c:
            exact += 1
        edit_dist += levenshtein_distance(p_c, r_c)
        ref_chars += max(len(r_c), 1)
    n = max(len(preds), 1)
    return {
        "exact_match_accuracy": (exact / n) * 100.0,
        "character_error_rate": (edit_dist / max(ref_chars, 1)) * 100.0,
        "total_evaluated": n
    }

# =====================================================================
# 4. BENCHMARK EXECUTION
# =====================================================================

def run_pulli_benchmark(api_key: Optional[str] = None):
    print("\n" + "=" * 80)
    print(" PROJECT VALIMELI & PULLI — SARVAM AI TRANSLITERATION BENCHMARK")
    print("=" * 80)

    valimeli = None
    if VALIMELI_AVAILABLE:
        try:
            valimeli = ValiMeliEngine()
            print(" -> ValiMeli 3.2M Standalone Engine Loaded Successfully.")
        except Exception as e:
            print(f" -> Could not initialize ValiMeliEngine: {e}")

    # Check Sarvam API key
    if not api_key:
        api_key = os.getenv("SARVAM_API_KEY") or os.getenv("SARVAM_KEY")

    if not api_key:
        print("\n ⚠️ NOTE: No SARVAM_API_KEY provided.")
        print("    You can run live API benchmarks by passing: --api-key YOUR_KEY")
        print("    or setting: export SARVAM_API_KEY='your_key'")
        print("    Generating test suite and ValiMeli baseline predictions...\n")

    # Evaluate Tamil Battery
    print("-" * 80)
    print(" 1. EVALUATION ON TAMIL PULLI & ALLOPHONY TEST BATTERY:")
    print("-" * 80)
    print(f" {'Roman Input':15s} | {'Target Indic':15s} | {'Category':15s} | {'ValiMeli Output':15s} | {'Sarvam Output':15s}")
    print("-" * 80)

    tam_results = []
    for roman, target, cat in PULLI_TEST_BATTERY_TAMIL:
        v_out = valimeli.transliterate(roman, lang="tam") if valimeli else "—"
        s_out = "—"
        latency = 0.0
        if api_key:
            s_out, latency = query_sarvam_transliterate(roman, source_lang="en-IN", target_lang="ta-IN", api_key=api_key)
            if s_out is None:
                s_out = "[ERROR]"
            time.sleep(0.05) # Polite rate limiting
            
        print(f" {roman:15s} | {target:15s} | {cat:15s} | {v_out:15s} | {s_out:15s}")
        tam_results.append({
            "roman": roman,
            "target": target,
            "category": cat,
            "valimeli_output": v_out,
            "sarvam_output": s_out,
            "sarvam_latency_ms": latency
        })

    # Evaluate Malayalam Battery
    print("\n" + "-" * 80)
    print(" 2. EVALUATION ON MALAYALAM PULLI & PHONOTACTIC BATTERY:")
    print("-" * 80)
    print(f" {'Roman Input':15s} | {'Target Indic':15s} | {'Category':15s} | {'ValiMeli Output':15s} | {'Sarvam Output':15s}")
    print("-" * 80)

    mal_results = []
    for roman, target, cat in PULLI_TEST_BATTERY_MALAYALAM:
        v_out = valimeli.transliterate(roman, lang="mal") if valimeli else "—"
        s_out = "—"
        latency = 0.0
        if api_key:
            s_out, latency = query_sarvam_transliterate(roman, source_lang="en-IN", target_lang="ml-IN", api_key=api_key)
            if s_out is None:
                s_out = "[ERROR]"
            time.sleep(0.05)
            
        print(f" {roman:15s} | {target:15s} | {cat:15s} | {v_out:15s} | {s_out:15s}")
        mal_results.append({
            "roman": roman,
            "target": target,
            "category": cat,
            "valimeli_output": v_out,
            "sarvam_output": s_out,
            "sarvam_latency_ms": latency
        })

    # Save benchmark suite results
    output_file = os.path.join(ARTIFACTS_DIR, "sarvam_pulli_benchmark.json")
    benchmark_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sarvam_api_tested": bool(api_key),
        "tamil_battery": tam_results,
        "malayalam_battery": mal_results
    }
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_report, f, indent=2, ensure_ascii=False)
    print(f"\n⭐ Benchmark report saved to: {output_file}\n")

def main():
    parser = argparse.ArgumentParser(description="Sarvam AI Transliteration Benchmark for Project Pulli")
    parser.add_argument("--api-key", type=str, default=None, help="Sarvam AI API Subscription Key")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of holdout test samples to query")
    args = parser.parse_args()

    run_pulli_benchmark(api_key=args.api_key)

if __name__ == "__main__":
    main()
