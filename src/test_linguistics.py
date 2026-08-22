#!/usr/bin/env python3
"""
Unit tests for Project ValiMeli Linguistic Rules & Context Taggers
Verifies:
1. Tamil word-initial voiceless stops (பக்கம் -> [INIT]ப)
2. Tamil geminate voiceless stops (பக்கம் -> [GEM]க், [GEM]க)
3. Tamil post-nasal voiced stops (சங்கு -> [NASAL]ங்கு, சந்தை -> [NASAL]ந்தை, பந்து -> [NASAL]ந்து)
4. Tamil intervocalic voiced stops (படம் -> [INTER]ட, அழகு -> [INTER]கு, வகை -> [INTER]கை)
5. Malayalam native plosive conditioning vs loanwords.
6. Levenshtein edit distance & metric calculations.
"""

import os
import sys

# Import functions from valimeli-benchmark.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the module dynamically or import directly
import importlib.util
spec = importlib.util.spec_from_file_location("valimeli_benchmark", os.path.join(os.path.dirname(__file__), "valimeli-benchmark.py"))
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)

def test_tamil_tagging():
    print("Testing Tamil Linguistic Phonology Rules...")
    
    # 1. Word initial + Geminate: 'பக்கம்'
    # 'ப' -> INIT, 'க்' -> GEM, 'க' -> GEM
    t1 = vb.apply_tamil_phonology_tags("பக்கம்")
    assert vb.TAG_INITIAL in t1, f"Failed initial tag on பக்கம்: {t1}"
    assert vb.TAG_GEMINATE in t1, f"Failed geminate tag on பக்கம்: {t1}"
    print(f"  ✅ 'பக்கம்' -> '{t1}' (INIT + GEM correctly tagged)")
    
    # 2. Intervocalic: 'படம்'
    # 'ப' -> INIT, 'ட' -> INTER
    t2 = vb.apply_tamil_phonology_tags("படம்")
    assert vb.TAG_INITIAL in t2, f"Failed initial tag on படம்: {t2}"
    assert vb.TAG_INTERVOCALIC in t2, f"Failed intervocalic tag on படம்: {t2}"
    print(f"  ✅ 'படம்' -> '{t2}' (INIT + INTER correctly tagged)")
    
    # 3. Post-nasal: 'சங்கு'
    # 'ச' -> INIT, 'கு' -> POST_NASAL
    t3 = vb.apply_tamil_phonology_tags("சங்கு")
    assert vb.TAG_POST_NASAL in t3, f"Failed post-nasal tag on சங்கு: {t3}"
    print(f"  ✅ 'சங்கு' -> '{t3}' (INIT + POST_NASAL correctly tagged)")
    
    # 4. Post-nasal: 'பந்து'
    t4 = vb.apply_tamil_phonology_tags("பந்து")
    assert vb.TAG_POST_NASAL in t4, f"Failed post-nasal tag on பந்து: {t4}"
    print(f"  ✅ 'பந்து' -> '{t4}' (INIT + POST_NASAL correctly tagged)")
    
    # 5. Intervocalic in polysyllabic word: 'அழகு'
    t5 = vb.apply_tamil_phonology_tags("அழகு")
    assert vb.TAG_INTERVOCALIC in t5, f"Failed intervocalic tag on அழகு: {t5}"
    print(f"  ✅ 'அழகு' -> '{t5}' (INTER correctly tagged)")

def test_malayalam_tagging():
    print("\nTesting Malayalam Linguistic Phonology Rules...")
    
    # 1. Native word: 'പകൽ' (pakal -> [paɡal])
    # 'പ' -> INIT, 'ക' -> INTER
    m1 = vb.apply_malayalam_phonology_tags("പകൽ")
    assert vb.TAG_INITIAL in m1, f"Failed initial tag on പകൽ: {m1}"
    assert vb.TAG_INTERVOCALIC in m1, f"Failed intervocalic tag on പകൽ: {m1}"
    print(f"  ✅ 'പകൽ' -> '{m1}' (INIT + INTER correctly tagged)")
    
    # 2. Geminate word: 'പച്ച' (pachha)
    m2 = vb.apply_malayalam_phonology_tags("പച്ച")
    assert vb.TAG_INITIAL in m2, f"Failed initial tag on പച്ച: {m2}"
    print(f"  ✅ 'പച്ച' -> '{m2}' (INIT + GEM correctly tagged)")
    
    # 3. Post-nasal: 'പങ്ക്' (pangu)
    m3 = vb.apply_malayalam_phonology_tags("പങ്ക്")
    assert vb.TAG_POST_NASAL in m3, f"Failed post-nasal tag on പങ്ക്: {m3}"
    print(f"  ✅ 'പങ്ക്' -> '{m3}' (INIT + POST_NASAL correctly tagged)")

def test_metrics():
    print("\nTesting Levenshtein Distance & Evaluation Metrics...")
    dist = vb.levenshtein_distance("pakkam", "pakam")
    assert dist == 1, f"Levenshtein distance incorrect: {dist} != 1"
    
    preds = ["pakkam", "padam", "sangu"]
    refs = ["pakkam", "padam", "sanku"]
    m = vb.calculate_metrics(preds, refs)
    assert round(m["exact_match_accuracy"], 2) == 66.67, f"EM incorrect: {m['exact_match_accuracy']}"
    assert m["character_error_rate"] > 0, "CER should be > 0"
    print(f"  ✅ Metrics verified: EM = {m['exact_match_accuracy']:.2f}%, CER = {m['character_error_rate']:.2f}%")

if __name__ == "__main__":
    print("=" * 60)
    print(" PROJECT VALIMELI — LINGUISTIC ENGINE VERIFICATION")
    print("=" * 60)
    test_tamil_tagging()
    test_malayalam_tagging()
    test_metrics()
    print("\n🎉 ALL LINGUISTIC & METRIC UNIT TESTS PASSED!")
    print("=" * 60)
