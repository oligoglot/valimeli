"""
Fetch Module:ta-IPA from English Wiktionary to inspect its exact phonological rules
"""
import urllib.request
import urllib.parse
import json
import ssl

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

url = "https://en.wiktionary.org/w/api.php?action=parse&page=Module:ta-IPA&prop=wikitext&format=json"
req = urllib.request.Request(url, headers=HEADERS)

try:
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        wikitext = data.get('parse', {}).get('wikitext', {}).get('*', '')
        with open("artifacts/wiktionary_ta_ipa_module.lua", "w", encoding="utf-8") as f:
            f.write(wikitext)
        print("✓ Successfully saved Wiktionary Module:ta-IPA to artifacts/wiktionary_ta_ipa_module.lua")
        print(f"File size: {len(wikitext)} characters")
except Exception as e:
    print(f"Error: {e}")
