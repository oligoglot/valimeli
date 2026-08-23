#!/usr/bin/env python3
"""
Project ValiMeli — Theedhum Nandrum Evaluation using 3.2M Multilingual Pre-trained Model
Evaluates the downstream sentiment classification gains of the 3.2M Multilingual Transformer
(Tamil, Malayalam, Telugu, Kannada, Hindi, Bengali, Gujarati, Marathi) on DravidianCodeMix FIRE 2020.
"""

import os
import sys
import json
import math
import time
import re
from collections import Counter
from typing import List, Tuple, Dict, Any

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
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# =====================================================================
# 1. MODEL ARCHITECTURE & VOCAB
# =====================================================================

class MultilingualVocab:
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    def __init__(self):
        self.t2i = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.i2t = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
        self.n = 4
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.t2i[self.SOS]] + [self.t2i.get(t, self.t2i[self.UNK]) for t in tokens] + [self.t2i[self.EOS]]
    def decode(self, indices: List[int]) -> str:
        out = []
        for idx in indices:
            if idx in {0, 1}:
                continue
            if idx == 2:
                break
            out.append(self.i2t.get(idx, ""))
        return "".join(out)

import __main__
__main__.MultilingualVocab = MultilingualVocab

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultilingualTransformerSeq2SeqInference(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=256)
        
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
# 2. TRANSLITERATOR WRAPPER
# =====================================================================

class ThreePointTwoMTransliterator:
    def __init__(self, checkpoint_path: str, device: torch.device):
        self.device = device
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.src_vocab = ckpt["src_vocab"]
        self.tgt_vocab = ckpt["tgt_vocab"]
        
        self.model = MultilingualTransformerSeq2SeqInference(
            self.src_vocab.n, self.tgt_vocab.n,
            d_model=256, nhead=4, num_layers=6, dim_feedforward=1024
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def transliterate_words(self, words: List[str], lang_prefix: str = "__ta__", batch_size: int = 512) -> List[str]:
        results = []
        for i in range(0, len(words), batch_size):
            batch_w = [w[:35] for w in words[i:i+batch_size]]
            encoded = [self.src_vocab.encode([lang_prefix] + list(w.lower())) for w in batch_w]
            max_len = max(len(e) for e in encoded)
            tensor = torch.zeros(len(encoded), max_len, dtype=torch.long, device=self.device)
            for j, e in enumerate(encoded):
                tensor[j, :len(e)] = torch.tensor(e, dtype=torch.long)
                
            decoded_indices = self.model.greedy_decode(tensor)
            for d in decoded_indices:
                clean_out = self.tgt_vocab.decode(d)
                results.append(clean_out)
        return results

# =====================================================================
# 3. DOWNSTREAM SENTIMENT EVALUATION
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

def run_evaluation(lang: str = "tam"):
    full_lang = "tamil" if lang == "tam" else "malayalam"
    prefix = "__ta__" if lang == "tam" else "__ml__"
    
    train_path = os.path.join(DATA_DIR, f"{full_lang}_train.tsv")
    dev_path = os.path.join(DATA_DIR, f"{full_lang}_dev.tsv")
    
    train_raw = load_tsv_corpus(train_path)
    dev_raw = load_tsv_corpus(dev_path)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt_path = os.path.join(RUNS_DIR, "multilingual_scaling_best.pt")
    
    print("\n" + "=" * 80, flush=True)
    print(f" EVALUATING 3.2M MULTILINGUAL MODEL ON {full_lang.upper()} SENTIMENT (THEEDHUM NANDRUM)", flush=True)
    print("=" * 80, flush=True)
    
    engine = ThreePointTwoMTransliterator(ckpt_path, device)
    
    all_texts = [clean_text(t) for t, _ in train_raw + dev_raw]
    all_words = list(set([w for t in all_texts for w in re.findall(r"[a-zA-Z]+", t)]))
    print(f" -> Generating 3.2M Neural Transliterations for {len(all_words):,} unique words...", flush=True)
    
    trans_words = engine.transliterate_words(all_words, lang_prefix=prefix)
    word_cache = dict(zip(all_words, trans_words))
    
    print("\n -> Sample 3.2M Neural Transliterations on Code-Mixed Slang:", flush=True)
    for sample_w in ["thambi", "padam", "semma", "mokkai", "super", "acting", "vera", "level", "thalaiva"][:6]:
        if sample_w in word_cache:
            print(f"    '{sample_w:10s}' -> 3.2M Multilingual: '{word_cache[sample_w]}'", flush=True)
            
    def apply_trans(texts):
        out = []
        for t in texts:
            words = re.findall(r"[a-zA-Z]+|[^\s\w]+", t)
            t_words = [word_cache.get(w, w) for w in words]
            out.append(" ".join(t_words))
        return out
        
    train_raw_texts = [clean_text(t) for t, _ in train_raw]
    dev_raw_texts = [clean_text(t) for t, _ in dev_raw]
    
    train_trans_texts = apply_trans(train_raw_texts)
    dev_trans_texts = apply_trans(dev_raw_texts)
    
    # Dual-stream concatenation: Raw Text + 3.2M Transliteration
    train_augmented = [f"{r} [MULTI_3.2M] {t}" for r, t in zip(train_raw_texts, train_trans_texts)]
    dev_augmented = [f"{r} [MULTI_3.2M] {t}" for r, t in zip(dev_raw_texts, dev_trans_texts)]
    
    train_y = [l for _, l in train_raw]
    dev_y = [l for _, l in dev_raw]
    labels = sorted(list(set(train_y)))
    
    vec = TfidfVectorizer(ngram_range=(1, 3), analyzer="char_wb", min_df=2, max_features=80000)
    X_train = vec.fit_transform(train_augmented)
    X_dev = vec.transform(dev_augmented)
    
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, C=1.0, random_state=42)
    clf.fit(X_train, train_y)
    
    preds = clf.predict(X_dev)
    
    macro_f1 = f1_score(dev_y, preds, average="macro") * 100.0
    weighted_f1 = f1_score(dev_y, preds, average="weighted") * 100.0
    acc = (np.mean([p == t for p, t in zip(preds, dev_y)])) * 100.0
    
    rep = classification_report(dev_y, preds, output_dict=True, zero_division=0)
    pos_f1 = rep.get("Positive", {}).get("f1-score", 0.0) * 100.0
    neg_f1 = rep.get("Negative", {}).get("f1-score", 0.0) * 100.0
    mix_f1 = rep.get("Mixed_feelings", {}).get("f1-score", 0.0) * 100.0
    
    print(f"\n ► 3.2M MULTILINGUAL AUGMENTED SENTIMENT RESULTS ({full_lang.upper()}):", flush=True)
    print(f"    Macro F1:    {macro_f1:.2f}%", flush=True)
    print(f"    Weighted F1: {weighted_f1:.2f}%", flush=True)
    print(f"    Accuracy:    {acc:.2f}%", flush=True)
    print(f"    Class F1:    Positive={pos_f1:.2f}% | Negative={neg_f1:.2f}% | Mixed={mix_f1:.2f}%\n", flush=True)
    
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": acc,
        "positive_f1": pos_f1,
        "negative_f1": neg_f1,
        "mixed_f1": mix_f1
    }

def main():
    tam_res = run_evaluation("tam")
    mal_res = run_evaluation("mal")
    
    # Save standalone persisted model for downstream & Pulli work
    src_ckpt = os.path.join(RUNS_DIR, "multilingual_scaling_best.pt")
    dst_ckpt = os.path.join(MODELS_DIR, "valimeli_multilingual_3.2m.pt")
    import shutil
    shutil.copyfile(src_ckpt, dst_ckpt)
    print(f"⭐ Persisted 3.2M model weights to: {dst_ckpt}", flush=True)
    
    all_res = {
        "tam": tam_res,
        "mal": mal_res
    }
    with open(os.path.join(ARTIFACTS_DIR, "theedhum_nandrum_3.2m_results.json"), "w") as f:
        json.dump(all_res, f, indent=2)

if __name__ == "__main__":
    main()
