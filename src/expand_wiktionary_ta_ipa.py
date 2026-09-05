"""
Expand Wiktionary template {{ta-IPA}} via MediaWiki API with rate limit handling
"""
import urllib.request
import urllib.parse
import json
import time
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'ValiMeliDravidianPhonologyBot/1.0 (academic research on Tamil orthography and phonetics; contact: research@example.org)'
}

WORDS = ["படம்", "பக்கம்", "தம்பி", "பந்து", "அழகு", "கண்", "அடி", "இலக்குவன்", "புலிகள்", "கொட்டிவாக்கம்", "சங்கு"]

def expand_ta_ipa(word):
    url = "https://en.wiktionary.org/w/api.php"
    params = {
        "action": "expandtemplates",
        "text": f"{{{{ta-IPA|{word}}}}}",
        "prop": "wikitext",
        "format": "json"
    }
    encoded_params = urllib.parse.urlencode(params).encode('utf-8')
    req = urllib.request.Request(url, data=encoded_params, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return data.get("expandtemplates", {}).get("wikitext", "")
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("=" * 80)
    print("LIVE WIKTIONARY {{ta-IPA}} TEMPLATE EXPANSION")
    print("=" * 80)
    
    results = {}
    for w in WORDS:
        time.sleep(1.5) # respectful delay to avoid 429
        res = expand_ta_ipa(w)
        print(f"Word: {w:<12} -> Wiktionary IPA Output: {res}")
        results[w] = res
        
    with open("artifacts/wiktionary_rendered_ipas.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
