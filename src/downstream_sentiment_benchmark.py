#!/usr/bin/env python3
"""
Project ValiMeli — Downstream Sentiment Classification Benchmark (v2 - Dual-Stream Augmentation)
Dataset: DravidianCodeMix FIRE 2020 (Tamil-English and Malayalam-English YouTube Comments)

Evaluating 3 Dual-Stream Feature Augmented Configurations:
  Arm 0: Raw Code-Mixed Text Only (Baseline)
  Arm 1: Raw Code-Mixed Text + Standard Transliteration Stream (A0 Augmented)
  Arm 2: Raw Code-Mixed Text + ValiMeli Phonology-Aware Stream (A1 Augmented)

Key Characteristics:
  - Preserves exact English Latin tokens (best, bad, super, waste) to retain English sentiment polarity.
  - Augments with phonologically normalized Dravidian n-grams (padam, thambi, nalla, semma).
  - Class-Balanced Loss Weighting & Unweighted Macro F1 reporting to avoid class skew distortion.
"""

import os
import sys
import json
import math
import time
import re
import urllib.request
from collections import Counter
from typing import List, Tuple, Dict, Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "sentiment_data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

URLS = {
    "tamil_train": "https://raw.githubusercontent.com/oligoglot/theedhum-nandrum/master/resources/data/tamil_train.tsv",
    "tamil_dev": "https://raw.githubusercontent.com/oligoglot/theedhum-nandrum/master/resources/data/tamil_dev.tsv",
    "malayalam_train": "https://raw.githubusercontent.com/oligoglot/theedhum-nandrum/master/resources/data/malayalam_train.tsv",
    "malayalam_dev": "https://raw.githubusercontent.com/oligoglot/theedhum-nandrum/master/resources/data/malayalam_dev.tsv",
}

def download_data():
    for key, url in URLS.items():
        fname = f"{key}.tsv"
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 100:
            print(f" -> Downloading {fname}...", flush=True)
            try:
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response, open(fpath, 'wb') as out_file:
                    out_file.write(response.read())
                print(f"    Saved {fname} ({os.path.getsize(fpath):,} bytes)", flush=True)
            except Exception as e:
                print(f"    Failed to download {fname}: {e}", flush=True)

def load_tsv_corpus(filepath: str) -> List[Tuple[str, str]]:
    samples = []
    if not os.path.exists(filepath):
        return samples
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                text = parts[0].strip()
                label = parts[1].strip()
                if text and label:
                    samples.append((text, label))
            elif len(parts) == 1 and "\t" not in line:
                sub = line.strip().rsplit(" ", 1)
                if len(sub) == 2:
                    samples.append((sub[0].strip(), sub[1].strip()))
    return samples

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def transliterate_valimeli(text: str) -> str:
    # Phonology-aware normalization for colloquial Dravidian roots
    words = text.split()
    norm = []
    for w in words:
        w_t = re.sub(r"([aeiou])th([aeiou])", r"\1d\2", w)
        w_t = re.sub(r"^th", "t", w_t)
        w_t = re.sub(r"dh", "d", w_t)
        w_t = re.sub(r"bh", "b", w_t)
        w_t = re.sub(r"kh", "k", w_t)
        w_t = re.sub(r"gh", "g", w_t)
        w_t = re.sub(r"aa+", "aa", w_t)
        w_t = re.sub(r"ee+", "ee", w_t)
        w_t = re.sub(r"oo+", "oo", w_t)
        norm.append(w_t)
    return " ".join(norm)

def transliterate_standard(text: str) -> str:
    # Standard character-by-character mapping
    words = text.split()
    norm = []
    for w in words:
        w_s = w.replace("th", "t").replace("dh", "t")
        norm.append(w_s)
    return " ".join(norm)

def prepare_augmented_text(text: str, mode: str = "raw") -> str:
    c_text = clean_text(text)
    if mode == "raw":
        return c_text
    elif mode == "standard_augmented":
        # Dual-stream: Original Latin String + Standard Transliteration Stream
        return f"{c_text} [STANDARD_TRANS] {transliterate_standard(c_text)}"
    elif mode == "valimeli_augmented":
        # Dual-stream: Original Latin String + ValiMeli Phonology Stream
        return f"{c_text} [VALIMELI_PHONO] {transliterate_valimeli(c_text)}"
    return c_text

def train_and_eval_sentiment(lang: str = "tam", mode: str = "raw") -> Dict[str, Any]:
    full_lang = "tamil" if lang == "tam" else "malayalam"
    train_path = os.path.join(DATA_DIR, f"{full_lang}_train.tsv")
    dev_path = os.path.join(DATA_DIR, f"{full_lang}_dev.tsv")
    
    raw_train = load_tsv_corpus(train_path)
    raw_dev = load_tsv_corpus(dev_path)
    
    if not raw_train or not raw_dev:
        print(f"❌ Could not load {lang} dataset from {train_path} / {dev_path}.", flush=True)
        return {}
        
    train_texts = [prepare_augmented_text(t, mode=mode) for t, _ in raw_train]
    train_y = [l for _, l in raw_train]
    dev_texts = [prepare_augmented_text(t, mode=mode) for t, _ in raw_dev]
    dev_y = [l for _, l in raw_dev]
    
    labels = sorted(list(set(train_y)))
    
    # Feature Extraction: Word + Character-in-Word N-Grams
    vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb", min_df=2, max_features=75000)
    X_train = vec.fit_transform(train_texts)
    X_dev = vec.transform(dev_texts)
    
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_y)
    
    preds = clf.predict(X_dev)
    
    macro_f1 = f1_score(dev_y, preds, average="macro") * 100.0
    weighted_f1 = f1_score(dev_y, preds, average="weighted") * 100.0
    accuracy = (np.mean([p == t for p, t in zip(preds, dev_y)])) * 100.0
    
    rep = classification_report(dev_y, preds, output_dict=True, zero_division=0)
    class_details = {}
    for lbl in labels:
        if lbl in rep:
            class_details[lbl] = {
                "precision": rep[lbl]["precision"] * 100.0,
                "recall": rep[lbl]["recall"] * 100.0,
                "f1_score": rep[lbl]["f1-score"] * 100.0,
                "support": rep[lbl]["support"]
            }
            
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "class_details": class_details
    }

def run_downstream_grid():
    download_data()
    print("\n" + "=" * 80, flush=True)
    print(" PROJECT VALIMELI — THEEDHUM NANDRUM DOWNSTREAM SENTIMENT BENCHMARK (AUGMENTED)", flush=True)
    print(" Evaluating Dual-Stream Feature Augmentation on DravidianCodeMix FIRE 2020", flush=True)
    print("=" * 80, flush=True)
    
    results = {}
    modes = [
        ("raw", "Raw Code-Mixed Text (Baseline)"),
        ("standard_augmented", "Raw Text + Standard Transliteration (A0 Augmented)"),
        ("valimeli_augmented", "Raw Text + ValiMeli Phonology Stream (A1 Augmented)")
    ]
    
    for lang in ["tam", "mal"]:
        results[lang] = {}
        print(f"\n >>> Evaluating Language: {lang.upper()} (DravidianCodeMix FIRE 2020)...", flush=True)
        for mode_key, mode_name in modes:
            print(f"  -> Running Arm: {mode_name}...", flush=True)
            res = train_and_eval_sentiment(lang=lang, mode=mode_key)
            results[lang][mode_key] = res
            print(f"     ⭐ Macro F1: {res['macro_f1']:.2f}% | Weighted F1: {res['weighted_f1']:.2f}% | Accuracy: {res['accuracy']:.2f}%", flush=True)
            
    print("\n" + "*" * 80, flush=True)
    print(" FINAL DUAL-STREAM AUGMENTED DOWNSTREAM SUMMARY (DRAVIDIANCODEMIX):", flush=True)
    print(" TAMIL-ENGLISH SENTIMENT:")
    for m_key, m_name in modes:
        print(f"   ► {m_key:22s}: Macro F1 = {results['tam'][m_key]['macro_f1']:.2f}% | Weighted F1 = {results['tam'][m_key]['weighted_f1']:.2f}% | Acc = {results['tam'][m_key]['accuracy']:.2f}%", flush=True)
    print("-" * 80, flush=True)
    print(" MALAYALAM-ENGLISH SENTIMENT:")
    for m_key, m_name in modes:
        print(f"   ► {m_key:22s}: Macro F1 = {results['mal'][m_key]['macro_f1']:.2f}% | Weighted F1 = {results['mal'][m_key]['weighted_f1']:.2f}% | Acc = {results['mal'][m_key]['accuracy']:.2f}%", flush=True)
    print("*" * 80 + "\n", flush=True)
    
    artifact_path = os.path.join(ARTIFACTS_DIR, "downstream_sentiment_augmented_results.json")
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f" -> Artifacts saved to {artifact_path}", flush=True)

if __name__ == "__main__":
    run_downstream_grid()
