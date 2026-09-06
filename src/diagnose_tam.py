"""
Diagnostic to inspect characters in Tamil test set
"""
import json
from collections import Counter

TAM_GRANTHA = set("ஜஷஸஹஶஸ்ரீ")

c = Counter()
with open("artifacts/predictions/predictions_tam_A0_25k_seed42.jsonl", "r") as f:
    for line in f:
        obj = json.loads(line)
        gold = obj.get("gold", "")
        chars = [ch for ch in gold if ch in TAM_GRANTHA]
        if chars:
            for ch in chars:
                c[ch] += 1

print("Grantha occurrences in Tamil test set:", c)
print("Total words with Grantha in Tamil:", sum(1 for line in open("artifacts/predictions/predictions_tam_A0_25k_seed42.jsonl") if any(ch in TAM_GRANTHA for ch in json.loads(line).get("gold", ""))))
