"""
Search Wikimedia Commons for all Tamil audio files (LL-Q5885 and Ta-*)
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

def search_commons_files(query="LL-Q5885 (tam)", limit=50):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&srnamespace=6&srlimit={limit}&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return [m['title'] for m in data.get('query', {}).get('search', [])]
    except Exception as e:
        print(f"Error searching Commons: {e}")
        return []

if __name__ == "__main__":
    files1 = search_commons_files("LL-Q5885 (tam)", 50)
    print(f"Found {len(files1)} LL-Q5885 (tam) files:")
    for f in files1[:20]:
        print(" ", f)
        
    files2 = search_commons_files("Ta- .ogg", 50)
    print(f"\nFound {len(files2)} Ta-*.ogg files:")
    for f in files2[:20]:
        print(" ", f)
