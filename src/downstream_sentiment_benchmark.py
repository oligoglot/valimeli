#!/usr/bin/env python3
"""
Project ValiMeli — Downstream Sentiment Classification Benchmark
Dataset: DravidianCodeMix FIRE 2020 (Tamil-English and Malayalam-English YouTube Comments)
Comparing:
  Arm 0: Raw Code-Mixed Text (Baseline)
  Arm 1: Standard Transliteration (A0)
  Arm 2: ValiMeli Phonology-Aware Transliteration (A1 / A1-MT)

Metrics:
  - Macro F1-Score (Primary metric to avoid class-imbalance distortion)
  - Weighted F1-Score
  - Per-Class F1 (Positive, Negative, Mixed_feelings, Neutral, not-Tamil)
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "sentiment_data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# 1. Download official splits if not present
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
                # Sometimes comma or space delimited
                sub = line.strip().rsplit(" ", 1)
                if len(sub) == 2:
                    samples.append((sub[0].strip(), sub[1].strip()))
    return samples

# 2. Text Normalisation & Feature Extraction
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def extract_ngrams(text: str, char_n_range=(2, 5), word_n_range=(1, 2)) -> Counter:
    counts = Counter()
    words = text.split()
    # Word n-grams
    for n in range(word_n_range[0], word_n_range[1] + 1):
        for i in range(len(words) - n + 1):
            counts["W_" + "_".join(words[i:i+n])] += 1
            
    # Char n-grams
    clean = " " + text + " "
    for n in range(char_n_range[0], char_n_range[1] + 1):
        for i in range(len(clean) - n + 1):
            counts["C_" + clean[i:i+n]] += 1
            
    return counts

# 3. Simple rule-based phonological transliteration emulator / transliterator
def transliterate_word_mock(word: str, mode: str = "raw") -> str:
    if mode == "raw":
        return word
    # Simple phonology-aware mapping simulating plosive normalization
    if mode == "valimeli":
        # Harmonize common plosive variations: dh->t/d, th->t, ph->p, bh->b, kh->k
        w = re.sub(r"([aeiou])th([aeiou])", r"\1d\2", word)
        w = re.sub(r"^th", "t", w)
        w = re.sub(r"dh", "d", w)
        w = re.sub(r"bh", "b", w)
        w = re.sub(r"kh", "k", w)
        w = re.sub(r"gh", "g", w)
        w = re.sub(r"aa+", "aa", w)
        w = re.sub(r"ee+", "ee", w)
        w = re.sub(r"oo+", "oo", w)
        return w
    elif mode == "standard":
        # Standard character-by-character mapping (strict)
        w = word.replace("th", "t").replace("dh", "t")
        return w
    return word

def preprocess_corpus(samples: List[Tuple[str, str]], mode: str = "raw") -> List[Tuple[str, str]]:
    processed = []
    for text, label in samples:
        c_text = clean_text(text)
        words = c_text.split()
        trans_words = [transliterate_word_mock(w, mode=mode) for w in words]
        processed.append((" ".join(trans_words), label))
    return processed

# 4. TF-IDF & PyTorch SGD Classifier
class TFIDFVectorizerPyTorch:
    def __init__(self, max_features: int = 50000, min_df: int = 2):
        self.max_features = max_features
        self.min_df = min_df
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.array([])
        
    def fit(self, texts: List[str]):
        doc_counts = Counter()
        N = len(texts)
        for t in texts:
            ng = extract_ngrams(t)
            for feat in ng.keys():
                doc_counts[feat] += 1
                
        valid_feats = [f for f, cnt in doc_counts.items() if cnt >= self.min_df]
        valid_feats.sort(key=lambda f: doc_counts[f], reverse=True)
        top_feats = valid_feats[:self.max_features]
        
        self.vocab = {f: idx for idx, f in enumerate(top_feats)}
        self.idf = np.zeros(len(self.vocab), dtype=np.float32)
        for f, idx in self.vocab.items():
            df = doc_counts[f]
            self.idf[idx] = math.log((1.0 + N) / (1.0 + df)) + 1.0
            
    def transform(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Return sparse indices and values for batch processing
        rows, cols, vals = [], [], []
        for r, t in enumerate(texts):
            ng = extract_ngrams(t)
            row_feats = {}
            for f, cnt in ng.items():
                if f in self.vocab:
                    c = self.vocab[f]
                    row_feats[c] = cnt * self.idf[c]
                    
            norm = math.sqrt(sum(v*v for v in row_feats.values())) if row_feats else 1.0
            for c, val in row_feats.items():
                rows.append(r)
                cols.append(c)
                vals.append(val / norm)
                
        indices = torch.tensor([rows, cols], dtype=torch.long)
        values = torch.tensor(vals, dtype=torch.float32)
        return indices, values, (len(texts), len(self.vocab))

def compute_f1_metrics(y_true: List[int], y_pred: List[int], label_names: List[str]) -> Dict[str, Any]:
    num_classes = len(label_names)
    tp = [0] * num_classes
    fp = [0] * num_classes
    fn = [0] * num_classes
    support = [0] * num_classes
    
    for t, p in zip(y_true, y_pred):
        support[t] += 1
        if t == p:
            tp[t] += 1
        else:
            fp[p] += 1
            fn[t] += 1
            
    class_f1s = []
    class_reports = {}
    for c in range(num_classes):
        prec = tp[c] / max(tp[c] + fp[c], 1)
        rec = tp[c] / max(tp[c] + fn[c], 1)
        f1 = (2 * prec * rec) / max(prec + rec, 1e-6)
        class_f1s.append(f1)
        class_reports[label_names[c]] = {
            "precision": prec * 100.0,
            "recall": rec * 100.0,
            "f1_score": f1 * 100.0,
            "support": support[c]
        }
        
    macro_f1 = (sum(class_f1s) / num_classes) * 100.0
    weighted_f1 = (sum(f1 * sup for f1, sup in zip(class_f1s, support)) / max(sum(support), 1)) * 100.0
    accuracy = (sum(tp) / max(len(y_true), 1)) * 100.0
    
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "class_details": class_reports
    }

def train_and_eval_sentiment(lang: str = "tam", mode: str = "raw") -> Dict[str, Any]:
    full_lang = "tamil" if lang == "tam" else "malayalam"
    train_path = os.path.join(DATA_DIR, f"{full_lang}_train.tsv")
    dev_path = os.path.join(DATA_DIR, f"{full_lang}_dev.tsv")
    
    raw_train = load_tsv_corpus(train_path)
    raw_dev = load_tsv_corpus(dev_path)
    
    if not raw_train or not raw_dev:
        print(f"❌ Could not load {lang} dataset from {train_path} / {dev_path}.", flush=True)
        return {}
        
    train_samples = preprocess_corpus(raw_train, mode=mode)
    dev_samples = preprocess_corpus(raw_dev, mode=mode)
    
    # Label encoding
    label_counts = Counter(lbl for _, lbl in train_samples)
    labels = sorted(label_counts.keys())
    label2idx = {l: i for i, l in enumerate(labels)}
    
    train_texts = [t for t, _ in train_samples]
    train_labels = [label2idx[l] for _, l in train_samples]
    dev_texts = [t for t, _ in dev_samples]
    dev_labels = [label2idx.get(l, 0) for _, l in dev_samples]
    
    vectorizer = TFIDFVectorizerPyTorch(max_features=50000, min_df=2)
    vectorizer.fit(train_texts)
    
    # Compute inverse class weights
    total_samples = len(train_labels)
    class_weights = torch.zeros(len(labels), dtype=torch.float32)
    for l, idx in label2idx.items():
        cnt = label_counts[l]
        class_weights[idx] = total_samples / (len(labels) * max(cnt, 1))
        
    # Logistic Regression Classifier in PyTorch
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    num_features = len(vectorizer.vocab)
    num_classes = len(labels)
    
    model = nn.Linear(num_features, num_classes).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.AdamW(model.parameters(), lr=0.01, weight_decay=1e-4)
    
    # Prepare batch sparse tensors
    batch_size = 512
    num_epochs = 15
    
    for epoch in range(num_epochs):
        model.train()
        permutation = np.random.permutation(len(train_texts))
        
        for i in range(0, len(train_texts), batch_size):
            batch_indices = permutation[i:i+batch_size]
            b_texts = [train_texts[idx] for idx in batch_indices]
            b_labels = torch.tensor([train_labels[idx] for idx in batch_indices], dtype=torch.long, device=device)
            
            ind, val, shape = vectorizer.transform(b_texts)
            sp_tensor = torch.sparse_coo_tensor(ind, val, shape).to_dense().to(device)
            
            optimizer.zero_grad()
            logits = model(sp_tensor)
            loss = criterion(logits, b_labels)
            loss.backward()
            optimizer.step()
            
    # Evaluation on Holdout Dev/Test Split
    model.eval()
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(dev_texts), batch_size):
            b_texts = dev_texts[i:i+batch_size]
            ind, val, shape = vectorizer.transform(b_texts)
            sp_tensor = torch.sparse_coo_tensor(ind, val, shape).to_dense().to(device)
            logits = model(sp_tensor)
            preds = logits.argmax(dim=-1).cpu().tolist()
            all_preds.extend(preds)
            
    metrics = compute_f1_metrics(dev_labels, all_preds, labels)
    return metrics

def run_downstream_grid():
    download_data()
    print("\n" + "=" * 80, flush=True)
    print(" PROJECT VALIMELI — THEEDHUM NANDRUM DOWNSTREAM SENTIMENT BENCHMARK", flush=True)
    print(" Evaluating In-the-Wild YouTube Review Code-Mixed Comments (Tamil & Malayalam)", flush=True)
    print("=" * 80, flush=True)
    
    results = {}
    for lang in ["tam", "mal"]:
        results[lang] = {}
        print(f"\n >>> Evaluating Language: {lang.upper()} (DravidianCodeMix FIRE 2020)...", flush=True)
        for mode in ["raw", "standard", "valimeli"]:
            mode_name = {
                "raw": "Raw Code-Mixed Text (No Transliteration)",
                "standard": "Standard Transliteration Baseline (A0)",
                "valimeli": "ValiMeli Phonology-Aware Frontend (A1)"
            }[mode]
            
            print(f"  -> Running Arm: {mode_name}...", flush=True)
            res = train_and_eval_sentiment(lang=lang, mode=mode)
            results[lang][mode] = res
            print(f"     ⭐ Macro F1: {res['macro_f1']:.2f}% | Weighted F1: {res['weighted_f1']:.2f}% | Accuracy: {res['accuracy']:.2f}%", flush=True)
            
    print("\n" + "*" * 80, flush=True)
    print(" FINAL DOWNSTREAM BENCHMARK SUMMARY (DRAVIDIANCODEMIX):", flush=True)
    print(" TAMIL-ENGLISH SENTIMENT:")
    for m in ["raw", "standard", "valimeli"]:
        print(f"   ► {m.upper():10s}: Macro F1 = {results['tam'][m]['macro_f1']:.2f}% | Weighted F1 = {results['tam'][m]['weighted_f1']:.2f}% | Acc = {results['tam'][m]['accuracy']:.2f}%", flush=True)
    print("-" * 80, flush=True)
    print(" MALAYALAM-ENGLISH SENTIMENT:")
    for m in ["raw", "standard", "valimeli"]:
        print(f"   ► {m.upper():10s}: Macro F1 = {results['mal'][m]['macro_f1']:.2f}% | Weighted F1 = {results['mal'][m]['weighted_f1']:.2f}% | Acc = {results['mal'][m]['accuracy']:.2f}%", flush=True)
    print("*" * 80 + "\n", flush=True)
    
    artifact_path = os.path.join(ARTIFACTS_DIR, "downstream_sentiment_results.json")
    with open(artifact_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f" -> Artifacts saved to {artifact_path}", flush=True)

if __name__ == "__main__":
    run_downstream_grid()
