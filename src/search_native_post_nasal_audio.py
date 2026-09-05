"""
Search Wikimedia Commons for audio recordings of common native Tamil post-nasal words.
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

CANDIDATES = [
    "தம்பி", "பந்து", "சங்கு", "வண்டு", "அம்பு", "கொம்பு", "பஞ்சு", "கம்பு", "நண்டு", "கன்று", 
    "சிங்கம்", "தங்கம்", "மங்கு", "மண்டபம்", "சந்தனம்", "விருந்து", "மருந்து", "குரங்கு"
]

def search_word_audio(word):
    query = f"File: {word}"
    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&srnamespace=6&srlimit=10&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            results = [m['title'] for m in data.get('query', {}).get('search', []) if any(ext in m['title'].lower() for ext in ['.wav', '.ogg', '.oga', '.webm'])]
            return results
    except Exception as e:
        return []

if __name__ == "__main__":
    print("=" * 80)
    print("SEARCHING COMMONS FOR NATIVE POST-NASAL AUDIO RECORDINGS")
    print("=" * 80)
    
    found = {}
    for c in CANDIDATES:
        res = search_word_audio(c)
        if res:
            print(f"Word: {c:<10} -> Found: {res}")
            found[c] = res
        else:
            # Also try search LL-Q5885
            res2 = search_word_audio(f"LL-Q5885 (tam) {c}")
            if res2:
                print(f"Word: {c:<10} -> Found: {res2}")
                found[c] = res2
