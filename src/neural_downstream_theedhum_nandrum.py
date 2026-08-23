#!/usr/bin/env python3
"""
Project ValiMeli — Neural Downstream Theedhum Nandrum Benchmark
Uses our actual trained 11.0M Transformer checkpoints (IndicXlit Baseline vs ValiMeli)
to neural-transliterate real-world code-mixed YouTube comments and evaluate downstream sentiment.

Comparing:
  1. Raw Code-Mixed Text Only (Baseline)
  2. Raw Text + Neural IndicXlit Transliteration (A0 Augmented)
  3. Raw Text + Neural ValiMeli Transliteration (A1 Augmented)
"""

import os
import sys
import json
import math
import time
import re
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "sentiment_data")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")

# =====================================================================
# 1. MODEL ARCHITECTURE FOR INFERENCE
# =====================================================================

class CharVocabulary:
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    def __init__(self):
        self.char2idx = {self.PAD_TOKEN: 0, self.SOS_TOKEN: 1, self.EOS_TOKEN: 2, self.UNK_TOKEN: 3}
        self.idx2char = {0: self.PAD_TOKEN, 1: self.SOS_TOKEN, 2: self.EOS_TOKEN, 3: self.UNK_TOKEN}
        self.num_chars = 4
    def encode(self, text: str) -> List[int]:
        tokens = [self.char2idx[self.SOS_TOKEN]]
        for c in text:
            tokens.append(self.char2idx.get(c, self.char2idx[self.UNK_TOKEN]))
        tokens.append(self.char2idx[self.EOS_TOKEN])
        return tokens
    def decode(self, indices: List[int]) -> str:
        chars = []
        for idx in indices:
            char = self.idx2char.get(idx, "")
            if char in {self.PAD_TOKEN, self.SOS_TOKEN}:
                continue
            if char == self.EOS_TOKEN:
                break
            chars.append(char)
        return "".join(chars)

import __main__
__main__.CharVocabulary = CharVocabulary

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class TransformerSeq2SeqInference(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=128)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def greedy_decode(self, src: torch.Tensor, max_len: int = 35) -> List[List[int]]:
        self.eval()
        device = src.device
        batch_size = src.size(0)
        with torch.no_grad():
            src_mask = (src == 0)
            src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
            memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
            tgt_indices = torch.full((batch_size, 1), 1, dtype=torch.long, device=device)
            decoded = [[] for _ in range(batch_size)]
            finished = [False] * batch_size
            for _ in range(max_len):
                tgt_emb = self.pos_encoder(self.tgt_embed(tgt_indices) * math.sqrt(self.d_model))
                causal_mask = self.generate_square_subsequent_mask(tgt_indices.size(1), device)
                out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_mask)
                next_tok = self.out_proj(out[:, -1]).argmax(dim=-1)
                for b in range(batch_size):
                    if not finished[b]:
                        tok = next_tok[b].item()
                        if tok == 2:
                            finished[b] = True
                        else:
                            decoded[b].append(tok)
                if all(finished):
                    break
                tgt_indices = torch.cat((tgt_indices, next_tok.unsqueeze(1)), dim=1)
        return decoded

# =====================================================================
# 2. NEURAL TRANSLITERATOR ENGINE
# =====================================================================

class NeuralTransliterator:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.vocab_src = ckpt["vocab_src"]
        self.vocab_tgt = ckpt["vocab_tgt"]
        
        self.model = TransformerSeq2SeqInference(
            self.vocab_src.num_chars, self.vocab_tgt.num_chars,
            d_model=256, nhead=4, num_layers=6, dim_feedforward=1024
        ).to(device)
        
        state_dict = ckpt["model_state_dict"]
        # Filter out aux keys if any
        filtered_state = {k: v for k, v in state_dict.items() if not k.startswith("aux_phono_proj")}
        self.model.load_state_dict(filtered_state, strict=False)
        self.model.eval()
        
    def transliterate_words_batch(self, words: List[str], batch_size: int = 512) -> List[str]:
        results = []
        for i in range(0, len(words), batch_size):
            batch_w = [w[:35] for w in words[i:i+batch_size]]
            encoded = [self.vocab_src.encode(w.lower()) for w in batch_w]
            max_len = max(len(e) for e in encoded)
            tensor = torch.zeros(len(encoded), max_len, dtype=torch.long, device=self.device)
            for j, e in enumerate(encoded):
                tensor[j, :len(e)] = torch.tensor(e, dtype=torch.long)
                
            decoded_indices = self.model.greedy_decode(tensor)
            for d in decoded_indices:
                raw_out = self.vocab_tgt.decode(d)
                # Strip any internal PUA tags if present
                clean_out = "".join(c for c in raw_out if ord(c) < 0xE000 or ord(c) > 0xF8FF)
                results.append(clean_out)
        return results

    def transliterate_sentence(self, sentence: str, word_cache: Dict[str, str]) -> str:
        words = re.findall(r"[a-zA-Z]+|[^\s\w]+", sentence.lower())
        out_words = []
        missing_words = []
        for w in words:
            if re.match(r"^[a-zA-Z]+$", w):
                if w in word_cache:
                    out_words.append(word_cache[w])
                else:
                    missing_words.append(w)
                    out_words.append(w) # placeholder
            else:
                out_words.append(w)
        return " ".join(out_words)

# =====================================================================
# 3. BENCHMARK PIPELINE
# =====================================================================

def load_tsv_corpus(filepath: str) -> List[Tuple[str, str]]:
    samples = []
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                samples.append((parts[0].strip(), parts[1].strip()))
    return samples

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def run_neural_downstream_benchmark(lang: str = "tam"):
    full_lang = "tamil" if lang == "tam" else "malayalam"
    train_path = os.path.join(DATA_DIR, f"{full_lang}_train.tsv")
    dev_path = os.path.join(DATA_DIR, f"{full_lang}_dev.tsv")
    
    train_raw = load_tsv_corpus(train_path)
    dev_raw = load_tsv_corpus(dev_path)
    
    print("\n" + "=" * 80, flush=True)
    print(f" NEURAL DOWNSTREAM BENCHMARK: {full_lang.upper()} SENTIMENT CLASSIFICATION", flush=True)
    print(" Using Actual Trained 11M Transformer Seq2Seq Checkpoints (IndicXlit vs ValiMeli)", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" -> Compute Device: {device}", flush=True)
    
    # Checkpoint paths
    a0_ckpt = os.path.join(RUNS_DIR, f"{lang}_A0_en-indic_indicxlit_rep", "checkpoint_best.pt")
    a1_ckpt = os.path.join(RUNS_DIR, f"{lang}_A1_en-indic_indicxlit_rep", "checkpoint_best.pt")
    
    print(f" -> Loading IndicXlit Baseline (A0) Checkpoint: {a0_ckpt}...", flush=True)
    a0_trans = NeuralTransliterator(a0_ckpt, device)
    print(f" -> Loading ValiMeli (A1) Checkpoint: {a1_ckpt}...", flush=True)
    a1_trans = NeuralTransliterator(a1_ckpt, device)
    
    # Pre-compute word transliteration caches
    all_texts = [clean_text(t) for t, _ in train_raw + dev_raw]
    all_words = list(set([w for t in all_texts for w in re.findall(r"[a-zA-Z]+", t)]))
    print(f" -> Pre-computing Neural Transliterations for {len(all_words):,} unique code-mixed vocabulary words...", flush=True)
    
    a0_translit_words = a0_trans.transliterate_words_batch(all_words)
    a1_translit_words = a1_trans.transliterate_words_batch(all_words)
    
    a0_cache = dict(zip(all_words, a0_translit_words))
    a1_cache = dict(zip(all_words, a1_translit_words))
    
    # Sample inspection
    print("\n -> Sample Neural Transliterations on Real Code-Mixed Slang:", flush=True)
    for sample_w in ["thambi", "padam", "semma", "mokkai", "super", "acting", "vera", "level", "thalaiva"][:6]:
        if sample_w in a0_cache and sample_w in a1_cache:
            print(f"    '{sample_w:10s}' -> IndicXlit A0: '{a0_cache[sample_w]}' | ValiMeli A1: '{a1_cache[sample_w]}'", flush=True)
            
    # Prepare datasets for 3 arms
    # Arm 0: Raw Code-Mixed Text Only
    # Arm 1: Dual-Stream (Raw Text + Neural IndicXlit Stream)
    # Arm 2: Dual-Stream (Raw Text + Neural ValiMeli Stream)
    
    train_raw_texts = [clean_text(t) for t, _ in train_raw]
    dev_raw_texts = [clean_text(t) for t, _ in dev_raw]
    
    def apply_trans(texts, cache):
        out = []
        for t in texts:
            words = re.findall(r"[a-zA-Z]+|[^\s\w]+", t)
            t_words = [cache.get(w, w) for w in words]
            out.append(" ".join(t_words))
        return out
        
    train_a0_texts = apply_trans(train_raw_texts, a0_cache)
    dev_a0_texts = apply_trans(dev_raw_texts, a0_cache)
    
    train_a1_texts = apply_trans(train_raw_texts, a1_cache)
    dev_a1_texts = apply_trans(dev_raw_texts, a1_cache)
    
    arms = {
        "1. Raw Code-Mixed Text (No Transliteration)": {
            "train": train_raw_texts,
            "dev": dev_raw_texts
        },
        "2. Raw Text + Neural IndicXlit Transliteration (A0)": {
            "train": [f"{r} [INDIC_XLIT] {t}" for r, t in zip(train_raw_texts, train_a0_texts)],
            "dev": [f"{r} [INDIC_XLIT] {t}" for r, t in zip(dev_raw_texts, dev_a0_texts)]
        },
        "3. Raw Text + Neural ValiMeli Transliteration (A1)": {
            "train": [f"{r} [VALIMELI] {t}" for r, t in zip(train_raw_texts, train_a1_texts)],
            "dev": [f"{r} [VALIMELI] {t}" for r, t in zip(dev_raw_texts, dev_a1_texts)]
        }
    }
    
    train_y = [l for _, l in train_raw]
    dev_y = [l for _, l in dev_raw]
    labels = sorted(list(set(train_y)))
    
    results = {}
    
    for arm_name, arm_data in arms.items():
        vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb", min_df=2, max_features=80000)
        X_train = vec.fit_transform(arm_data["train"])
        X_dev = vec.transform(arm_data["dev"])
        
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
        clf.fit(X_train, train_y)
        
        preds = clf.predict(X_dev)
        
        macro_f1 = f1_score(dev_y, preds, average="macro") * 100.0
        weighted_f1 = f1_score(dev_y, preds, average="weighted") * 100.0
        acc = (np.mean([p == t for p, t in zip(preds, dev_y)])) * 100.0
        
        rep = classification_report(dev_y, preds, output_dict=True, zero_division=0)
        neg_f1 = rep.get("Negative", {}).get("f1-score", 0.0) * 100.0
        mix_f1 = rep.get("Mixed_feelings", {}).get("f1-score", 0.0) * 100.0
        pos_f1 = rep.get("Positive", {}).get("f1-score", 0.0) * 100.0
        
        results[arm_name] = {
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "accuracy": acc,
            "positive_f1": pos_f1,
            "negative_f1": neg_f1,
            "mixed_f1": mix_f1
        }
        
        print(f"\n ► {arm_name:52s}", flush=True)
        print(f"    Macro F1: {macro_f1:.2f}% | Weighted F1: {weighted_f1:.2f}% | Accuracy: {acc:.2f}%", flush=True)
        print(f"    Class F1: Positive={pos_f1:.2f}% | Negative={neg_f1:.2f}% | Mixed={mix_f1:.2f}%", flush=True)
        
    return results

def run_all_neural_downstream():
    final_res = {}
    for l in ["tam", "mal"]:
        final_res[l] = run_neural_downstream_benchmark(l)
        
    print("\n" + "*" * 80, flush=True)
    print(" FINAL END-TO-END NEURAL DOWNSTREAM BENCHMARK SUMMARY (DRAVIDIANCODEMIX):", flush=True)
    for l in ["tam", "mal"]:
        name = "TAMIL-ENGLISH" if l == "tam" else "MALAYALAM-ENGLISH"
        print(f"\n {name} SENTIMENT:")
        for arm_n, r in final_res[l].items():
            print(f"   ► {arm_n:52s}: Macro F1 = {r['macro_f1']:.2f}% | Weighted F1 = {r['weighted_f1']:.2f}% | Acc = {r['accuracy']:.2f}%", flush=True)
    print("*" * 80 + "\n", flush=True)
    
    with open(os.path.join(ARTIFACTS_DIR, "neural_downstream_theedhum_nandrum_results.json"), "w") as f:
        json.dump(final_res, f, indent=2)

if __name__ == "__main__":
    run_all_neural_downstream()
