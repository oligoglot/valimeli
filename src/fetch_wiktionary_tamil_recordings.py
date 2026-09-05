"""
Script: fetch_wiktionary_tamil_recordings.py
Queries Wikimedia Commons / Wiktionary API to retrieve Tamil audio recordings and IPA transcriptions.
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

def query_commons(cmtitle="Category:Tamil_pronunciation_by_native_speakers"):
    url = f"https://commons.wikimedia.org/w/api.php?action=query&list=categorymembers&cmtitle={urllib.parse.quote(cmtitle)}&cmtype=file&cmlimit=50&format=json"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            return [m['title'] for m in data.get('query', {}).get('categorymembers', [])]
    except Exception as e:
        print(f"Error querying Commons API: {e}")
        return []

if __name__ == "__main__":
    files = query_commons("Category:Tamil_pronunciation")
    print(f"Found {len(files)} files in Category:Tamil_pronunciation:")
    for f in files[:15]:
        print(" ", f)
        
    subcats = query_commons("Category:Tamil_pronunciation_of_words")
    if not subcats:
        subcats = query_commons("Category:Tamil_pronunciation_by_native_speakers")
    print(f"\nFound {len(subcats)} files in subcategory:")
    for f in subcats[:15]:
        print(" ", f)
