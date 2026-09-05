#!/usr/bin/env python3
"""
Pan-Indic Media & OTT Entity Corpus Generator for Pulli-Pro (Server Scale)
Generates high-priority cultural, cinema, music, and OTT search entity pairs
across all 8 Indic languages with extensive Romanized typing variants.

Copyright (c) 2026 BalaSundaraRaman Lakshmanan (oligoglot). All Rights Reserved.
"""

import os
import json
import gzip
from typing import List, Dict, Tuple

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PULLI_DIR = os.path.abspath(os.path.join(WORKSPACE_DIR, "..", "pulli"))
OUT_FILE = os.path.join(PULLI_DIR, "data", "pulli_pro_media_entities.jsonl")

# Core Pan-Indic Iconic Entities (Cinema, Music, Literature, Toponyms, Cultural Icons)
MEDIA_CATALOG = {
    "ta-IN": [
        ("பொன்னியின் செல்வன்", ["ponniyin selvan", "ponniyinselvan", "ponniyin selvan 1", "ponniyin selvan 2", "ps1", "ps2"]),
        ("விக்ரம்", ["vikram", "vikram vedha", "vikram movie"]),
        ("லியோ", ["leo", "leo movie", "leo das"]),
        ("ஜெயிலர்", ["jailer", "jailer rajini", "jailer movie"]),
        ("தளபதி", ["thalapathy", "thalapathi", "dalapathi"]),
        ("தலைவர்", ["thalaivar", "thalaiva", "thalaivar 170"]),
        ("அனிருத்", ["anirudh", "aniruth", "anirudh ravichander"]),
        ("இளையராஜா", ["ilaiyaraaja", "ilayaraja", "ilaiyaraja", "isaignani"]),
        ("ரஹ்மான்", ["rahman", "ar rahman", "arrahman"]),
        ("மரகத நாணயம்", ["maragadha naanayam", "maragatha nanayam"]),
        ("சூரரைப் போற்று", ["soorarai pottru", "sooraraipottru"]),
        ("கண்ணே கலைமானே", ["kanne kalaimaane", "kanne kalaimane"]),
        ("முத்து", ["muthu", "muthu movie"]),
        ("பாட்ஷா", ["baasha", "baadshah", "basha"]),
        ("அந்நியன்", ["anniyan", "anniyan movie"]),
        ("சிவாஜி", ["sivaji", "sivaji the boss", "shivaji"]),
        ("காப்பான்", ["kaappaan", "kappan"]),
        ("மாஸ்டர்", ["master", "master vijay", "master movie"]),
        ("அமரன்", ["amaran", "amaran movie"]),
        ("கங்குவா", ["kanguva", "kanguva suriya"])
    ],
    "ml-IN": [
        ("മണിച്ചിത്രത്താഴ്", ["manichitrathazhu", "manichitrathazu", "manichitrathal"]),
        ("ദൃശ്യം", ["drishyam", "drishyam 2", "drishyam 3", "dhrishyam"]),
        ("മോഹൻലാൽ", ["mohanlal", "lalettan", "mohanlal movies"]),
        ("മമ്മൂട്ടി", ["mammootty", "mammookka", "mammooty"]),
        ("കുമ്പളങ്ങി നൈറ്റ്സ്", ["kumbalangi nights", "kumbalangi"]),
        ("പ്രേമം", ["premam", "premam nivin", "premam songs"]),
        ("ലൂസിഫർ", ["lucifer", "lucifer mohanlal", "empuraan"]),
        ("ആടുജീവിതം", ["aadujeevitham", "the goat life", "adujeevitham"]),
        ("മഞ്ഞുമ്മൽ ബോയ്സ്", ["manjummel boys", "manjummel"]),
        ("ആവേശം", ["aavesham", "avesham", "fahadh faasil"]),
        ("ചിത്രം", ["chithram", "chitram"]),
        ("കിലുക്കം", ["kilukkam", "kilukkam comedy"]),
        ("ദേവാസുരം", ["devasuram", "devaasuram"]),
        ("സ്ഫടികം", ["sphadikam", "spadikam"]),
        ("ബാംഗ്ലൂർ ഡേയ്സ്", ["bangalore days", "bangaloredays"])
    ],
    "te-IN": [
        ("బాహుబలి", ["baahubali", "bahubali", "baahubali 2", "bahubali the beginning"]),
        ("ఆర్ఆర్ఆర్", ["rrr", "rrr movie", "rajamouli rrr"]),
        ("పుష్ప", ["pushpa", "pushpa 2", "pushpa the rule", "pushpa the rise"]),
        ("దేవర", ["devara", "devara ntr", "devara movie"]),
        ("కల్కి", ["kalki", "kalki 2898 ad", "kalki movie"]),
        ("మగధీర", ["magadheera", "magadhira"]),
        ("చిరంజీవి", ["chiranjeevi", "megastar", "chiru"]),
        ("మహేష్ బాబు", ["mahesh babu", "mahesh", "superstar mahesh"]),
        ("ప్రభాస్", ["prabhas", "darling prabhas"]),
        ("అల్లు అర్జున్", ["allu arjun", "stylish star", "icon star"]),
        ("జగదేక వీరుడు అతిలోక సుందరి", ["jagadeka veerudu athiloka sundari"]),
        ("మాయాబజార్", ["mayabazar", "maya bazaar"]),
        ("శంకరాభరణం", ["sankarabharanam", "shankarabharanam"]),
        ("సీతారామం", ["sita ramam", "sitharamam"]),
        ("అల వైకుంఠపురములో", ["ala vaikunthapurramuloo", "ala vaikuntapuramlo"])
    ],
    "kn-IN": [
        ("ಕಾಂತಾರ", ["kantara", "kantara chapter 1", "kantara movie", "rishab shetty"]),
        ("ಕೆಜಿಎಫ್", ["kgf", "kgf chapter 1", "kgf chapter 2", "yash kgf"]),
        ("ರಾಜಕುಮಾರ", ["raajakumara", "rajakumara", "puneeth rajkumar"]),
        ("ಉಪ್ಪಿಟ್ಟು", ["uppittu", "upendra"]),
        ("ಕಿರಿಕ್ ಪಾರ್ಟಿ", ["kirik party", "kirikparty"]),
        ("ಚಾರ್ಲಿ", ["777 charlie", "charlie 777"]),
        ("ಕ್ರಾಂತಿವೀರ ಸಂಗೊಳ್ಳಿ ರಾಯಣ್ಣ", ["krantiveera sangolli rayanna"]),
        ("ಮುಂಗಾರು ಮಳೆ", ["mungaru male", "mungarumale"]),
        ("ಓಂ", ["om", "om shivrajkumar", "om movie"]),
        ("ಅಪ್ಪು", ["appu", "powerstar appu"])
    ],
    "hi-IN": [
        ("शोले", ["sholay", "sholay movie", "gabbar singh"]),
        ("दंगल", ["dangal", "dangal aamir khan"]),
        ("दिलवाले दुल्हनिया ले जाएंगे", ["ddlj", "dilwale dulhania le jayenge"]),
        ("लगान", ["lagaan", "lagan"]),
        ("थ्री इडियट्स", ["3 idiots", "three idiots"]),
        ("शोले", ["sholay"]),
        ("जवान", ["jawan", "jawan srk", "jawaan"]),
        ("पठान", ["pathaan", "pathan srk"]),
        ("स्त्री", ["stree", "stree 2", "stree movie"]),
        ("शोले", ["sholay"]),
        ("एनिमल", ["animal", "animal movie", "ranbir animal"])
    ],
    "bn-IN": [
        ("পথের পাঁচালী", ["pather panchali", "satyajit ray"]),
        ("চারুলতা", ["charulata", "charulata ray"]),
        ("সোনার কেল্লা", ["shonar kella", "sonar kella", "feluda"]),
        ("অপরাজিত", ["aparajito", "apur sansar"]),
        ("গুপি গাইন বাঘা বাইন", ["goopy gyne bagha byne", "gupi gayen"]),
        ("বেলাশেষে", ["belaseshe", "bela seshe"]),
        ("ভূতের ভবিষ্যৎ", ["bhooter bhobishyot", "bhuter bhobishyot"])
    ],
    "gu-IN": [
        ("હેલ્લारो", ["hellaro", "hellaro movie", "national award"]),
        ("ચલ મન જીતવા જઈએ", ["chal man jeetva jaiye"]),
        ("છેલ્લો શો", ["chhello show", "last film show"]),
        ("ગુજ્જુભાઈ ધ ગ્રેટ", ["gujjubhai the great", "gujjubhai"]),
        ("લવની ભવાઈ", ["love ni bhavai"])
    ],
    "mr-IN": [
        ("सैराट", ["sairat", "sairat movie", "zingaat"]),
        ("नટસમ્રાટ", ["natsamrat", "nana patekar"]),
        ("कट्यार काळजात घुसली", ["katyar kaljat ghusli"]),
        ("पावनखिंड", ["pawankhind", "pawankhind movie"]),
        ("हर हर महादेव", ["har har mahadev", "chhatrapati shivaji maharaj"])
    ]
}

def generate_media_dataset():
    print("=" * 80)
    print(" 🎬 GENERATING PAN-INDIC MEDIA & OTT SEARCH CORPUS FOR PULLI-PRO")
    print("=" * 80)

    total_pairs = 0
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        for lang_code, entries in MEDIA_CATALOG.items():
            lang_prefix = f"__{lang_code[:2]}__"
            lang_count = 0
            for target_indic, roman_variants in entries:
                # Add base words
                words_indic = target_indic.split()
                
                # Add full phrase variants
                for r_var in roman_variants:
                    row = {
                        "roman": r_var.lower(),
                        "indic": target_indic,
                        "lang": lang_prefix,
                        "type": "media_entity"
                    }
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    total_pairs += 1
                    lang_count += 1

                # Also add individual title tokens if multi-word
                for w_ind in words_indic:
                    if len(w_ind) > 2:
                        row = {
                            "roman": w_ind.lower(),
                            "indic": w_ind,
                            "lang": lang_prefix,
                            "type": "media_token"
                        }
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                        total_pairs += 1
                        lang_count += 1
            print(f" -> {lang_code}: {lang_count} media entity pairs generated.")

    print(f"\n ⭐ Total Media Entity Pairs: {total_pairs:,}")
    print(f" 💾 Saved to: {OUT_FILE}")

if __name__ == "__main__":
    generate_media_dataset()
