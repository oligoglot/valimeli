#!/usr/bin/env python3
"""
Script: batch_index_commons_speech.py
Batch indexes all 5,000+ Tamil native pronunciation recordings from Wikimedia Commons / Lingua Libre (LL-Q5885).
Extracts:
- Clean Tamil word
- Native speaker (Sriveenkat, Manimaran96, etc.)
- License: Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)
- Commons page URL & Direct audio stream URL
- Automated ValiMeli phonotactic tagging (Initial, Geminate, Post-Nasal, Intervocalic)
"""

import os
import re
import json
import time
import ssl
import urllib.request
import urllib.parse
from typing import List, Dict, Any

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_MANIFEST = os.path.join(WORKSPACE_DIR, "artifacts", "tamil_speech_full_corpus_manifest.json")
ATTRIBUTION_MD = os.path.join(WORKSPACE_DIR, "artifacts", "AUDIO_ATTRIBUTION.md")

TAMIL_VALLINAM = {'க', 'ச', 'ட', 'த', 'ப', 'ற'}
TAMIL_MELLINAM = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}
TAMIL_VOWEL_SIGNS = {
    '\u0bbe', '\u0bbf', '\u0bc0', '\u0bc1', '\u0bc2', 
    '\u0bc6', '\u0bc7', '\u0bc8', '\u0bca', '\u0bcb', '\u0bcc'
}
TAMIL_PULLI = '\u0bcd'

def segment_eluttu(word: str) -> List[str]:
    units = []
    current = ""
    for char in word:
        if char in TAMIL_VOWEL_SIGNS or char == TAMIL_PULLI:
            current += char
        else:
            if current:
                units.append(current)
            current = char
    if current:
        units.append(current)
    return units

def classify_word_phonotactics(word: str) -> Dict[str, Any]:
    """Classifies Tamil plosive phonotactic environments at the eḻuttu level."""
    eluttukkal = segment_eluttu(word)
    slots = []
    
    idx = 0
    while idx < len(eluttukkal):
        el = eluttukkal[idx]
        if not el or el[0] not in TAMIL_VALLINAM:
            idx += 1
            continue
            
        base_char = el[0]
        
        # Check if geminate cluster (pure consonant followed by identical plosive)
        if TAMIL_PULLI in el and idx + 1 < len(eluttukkal) and eluttukkal[idx + 1][0] == base_char:
            full_unit = el + eluttukkal[idx + 1]
            slots.append({
                "eluttu": full_unit,
                "base_grapheme": base_char,
                "context": "Geminate (C_C)",
                "expected_voicing": "Voiceless Fortis"
            })
            idx += 2
            continue
            
        # Check if post-nasal cluster (preceding nasal with pulli)
        if idx > 0 and TAMIL_PULLI in eluttukkal[idx - 1] and eluttukkal[idx - 1][0] in TAMIL_MELLINAM:
            full_unit = eluttukkal[idx - 1] + el
            slots.append({
                "eluttu": full_unit,
                "base_grapheme": base_char,
                "context": "Post-Nasal (N_)",
                "expected_voicing": "Voiced"
            })
            idx += 1
            continue
            
        # If pure consonant without forming a recognized cluster (e.g. coda stop)
        if TAMIL_PULLI in el:
            slots.append({
                "eluttu": el,
                "base_grapheme": base_char,
                "context": "Coda / Pure Consonant",
                "expected_voicing": "Voiceless Default"
            })
            idx += 1
            continue
            
        # Word-Initial
        if idx == 0:
            slots.append({
                "eluttu": el,
                "base_grapheme": base_char,
                "context": "Word-Initial (#_)",
                "expected_voicing": "Voiceless"
            })
            idx += 1
            continue
            
        # Intervocalic (preceded by vowel or vowel-bearing uyirmey without pulli)
        prev = eluttukkal[idx - 1]
        if TAMIL_PULLI not in prev:
            slots.append({
                "eluttu": el,
                "base_grapheme": base_char,
                "context": "Intervocalic (V_V)",
                "expected_voicing": "Voiced / Lenis"
            })
        else:
            slots.append({
                "eluttu": el,
                "base_grapheme": base_char,
                "context": "Post-Consonantal / Other",
                "expected_voicing": "Voiceless Default"
            })
        idx += 1
            
    return {
        "has_plosives": len(slots) > 0,
        "slots": slots
    }

def fetch_all_commons_tamil_audio() -> List[Dict[str, Any]]:
    print("=" * 80)
    print("BATCH INDEXING WIKIMEDIA COMMONS / LINGUA LIBRE TAMIL SPEECH CORPUS")
    print("=" * 80)
    
    ctx = ssl._create_unverified_context()
    base_url = "https://commons.wikimedia.org/w/api.php"
    headers = {
        "User-Agent": "ValiMeliPhonologyResearch/1.0 (https://github.com/oligoglot/valimeli; contact@valimeli.org)"
    }
    
    all_files = []
    sroffset = 0
    batch_num = 1
    
    while True:
        params = {
            "action": "query",
            "list": "search",
            "srsearch": "LL-Q5885 (tam)",
            "srnamespace": "6", # File: namespace
            "srlimit": "500",
            "sroffset": str(sroffset),
            "format": "json"
        }
        url = f"{base_url}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers=headers)
        
        try:
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                search_results = data.get('query', {}).get('search', [])
                if not search_results:
                    break
                    
                print(f"Batch {batch_num:02d}: Retrieved {len(search_results)} recordings (Offset: {sroffset})...")
                all_files.extend(search_results)
                
                if 'continue' in data and 'sroffset' in data['continue']:
                    sroffset = data['continue']['sroffset']
                    batch_num += 1
                    time.sleep(0.3) # Respect API rate limits
                else:
                    break
        except Exception as e:
            print(f"Error fetching batch {batch_num}: {e}")
            break
            
    print(f"\n✓ Successfully fetched {len(all_files)} total Commons audio metadata records.")
    return all_files

def parse_and_build_manifest(raw_files: List[Dict[str, Any]]):
    manifest_records = []
    speaker_counts = {}
    context_counts = {"Word-Initial (#_)": 0, "Geminate (C_C)": 0, "Post-Nasal (N_)": 0, "Intervocalic (V_V)": 0}
    
    for item in raw_files:
        title = item.get("title", "")
        # Format typically: File:LL-Q5885 (tam)-<Speaker>-<TamilWord>.wav
        clean_title = title.replace("File:", "").strip()
        
        speaker = "Unknown"
        word = ""
        
        # Regex matching Lingua Libre pattern
        match = re.search(r"LL-Q5885\s*\(tam\)-([^-]+)-(.+?)\.(wav|ogg|mp3|flac)", clean_title, re.IGNORECASE)
        if match:
            speaker = match.group(1).strip()
            word = match.group(2).strip()
        else:
            # Fallback for Ta-*.ogg
            if clean_title.startswith("Ta-"):
                word = clean_title[3:].split(".")[0].strip()
                speaker = "Wikimedia Community"
            else:
                word = clean_title.split(".")[0].strip()
                
        # Clean any HTML entities or URL encodings
        word = urllib.parse.unquote(word)
        
        # Verify contains Tamil characters
        if not any(0x0B80 <= ord(c) <= 0x0BFF for c in word):
            continue
            
        speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        phonotactics = classify_word_phonotactics(word)
        
        for slot in phonotactics.get("slots", []):
            ctx_name = slot["context"]
            if ctx_name in context_counts:
                context_counts[ctx_name] += 1
                
        commons_url = f"https://commons.wikimedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
        
        manifest_records.append({
            "id": len(manifest_records) + 1,
            "word": word,
            "speaker": speaker,
            "file_title": title,
            "commons_url": commons_url,
            "license": "CC BY-SA 4.0 (Creative Commons Attribution-ShareAlike 4.0 International)",
            "license_url": "https://creativecommons.org/licenses/by-sa/4.0/",
            "phonotactic_slots": phonotactics["slots"],
            "has_plosives": phonotactics["has_plosives"]
        })
        
    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
    with open(OUT_MANIFEST, "w", encoding="utf-8") as f:
        json.dump({
            "corpus_name": "Wikimedia Commons / Lingua Libre Tamil Speech Pronunciation Corpus",
            "total_indexed_recordings": len(manifest_records),
            "license": "CC BY-SA 4.0",
            "speaker_distribution": speaker_counts,
            "phonotactic_context_counts": context_counts,
            "recordings": manifest_records
        }, f, indent=2, ensure_ascii=False)
        
    print(f"✓ Saved complete manifest ({len(manifest_records)} words) to: {OUT_MANIFEST}")
    
    # Generate Attribution Markdown Document
    with open(ATTRIBUTION_MD, "w", encoding="utf-8") as f:
        f.write("# Audio Data Attribution & License Notice\n\n")
        f.write("This project utilizes spoken audio pronunciations from the **Wikimedia Commons / Lingua Libre** repository.\n\n")
        f.write("## License Terms\n")
        f.write("All audio recordings indexed and analyzed in this study are licensed under the **[Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/)**.\n\n")
        f.write("## Contributors & Speakers\n")
        f.write("We gratefully acknowledge the Lingua Libre project contributors and native Tamil speakers for their open-access recordings:\n\n")
        for spk, count in sorted(speaker_counts.items(), key=lambda x: -x[1]):
            f.write(f"- **User:{spk}**: {count:,} audio recordings ([Lingua Libre Profile](https://commons.wikimedia.org/wiki/User:{urllib.parse.quote(spk)}))\n")
        f.write("\n## Corpus Summary\n")
        f.write(f"- **Total Indexed Audio Files**: {len(manifest_records):,}\n")
        f.write(f"- **Word-Initial Plosive Environments**: {context_counts['Word-Initial (#_)']:,}\n")
        f.write(f"- **Geminate Fortis Environments**: {context_counts['Geminate (C_C)']:,}\n")
        f.write(f"- **Post-Nasal Voiced Environments**: {context_counts['Post-Nasal (N_)']:,}\n")
        f.write(f"- **Intervocalic Lenis Environments**: {context_counts['Intervocalic (V_V)']:,}\n\n")
        f.write("Under the terms of CC BY-SA 4.0, any derivative acoustic representations, spectrograms, or alignment models published in this research are similarly attributed and shared.\n")
        
    print(f"✓ Saved Attribution documentation to: {ATTRIBUTION_MD}")

if __name__ == "__main__":
    files = fetch_all_commons_tamil_audio()
    if files:
        parse_and_build_manifest(files)
