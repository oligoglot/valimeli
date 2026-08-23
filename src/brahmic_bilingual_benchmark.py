#!/usr/bin/env python3
"""
Project ValiMeli — Common Brahmic Script-Unified Bilingual Benchmark
Unified Phonetic Representation for Tamil and Malayalam (1.0M Pairs Total)

Translates both Tamil and Malayalam Unicode characters into a shared Common Brahmic
phonetic token stream (~65 tokens). The 11M Transformer Decoder trains on 100% unified token IDs.
"""

import os
import sys
import json
import math
import time
import shutil
import random
import argparse
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =====================================================================
# 1. BRAHMIC SCRIPT UNIFICATION TABLES
# =====================================================================

TAMIL_TO_BRAHMIC = {
    'அ': 'A', 'ஆ': 'AA', 'இ': 'I', 'ஈ': 'II', 'உ': 'U', 'ஊ': 'UU',
    'எ': 'E', 'ஏ': 'EE', 'ஐ': 'AI', 'ஒ': 'O', 'ஓ': 'OO', 'ஔ': 'AU',
    'க': 'KA', 'ங': 'NGA', 'ச': 'CA', 'ஞ': 'NYA', 'ட': 'TTA', 'ண': 'NNA',
    'த': 'TA', 'ந': 'NA', 'ப': 'PA', 'ம': 'MA', 'ய': 'YA', 'ர': 'RA',
    'ல': 'LA', 'வ': 'VA', 'ழ': 'ZHA', 'ள': 'LLA', 'ற': 'RRA', 'ன': 'NNNA',
    '்': 'VIRAMA',
    'ா': 'SIGN_AA', 'ி': 'SIGN_I', 'ீ': 'SIGN_II', 'ு': 'SIGN_U', 'ூ': 'SIGN_UU',
    'ெ': 'SIGN_E', 'ே': 'SIGN_EE', 'ை': 'SIGN_AI', 'ொ': 'SIGN_O', 'ோ': 'SIGN_OO', 'ௌ': 'SIGN_AU',
    'ஃ': 'AYTHAM', 'ஜ': 'JA', 'ஷ': 'SSA', 'ஸ': 'SA', 'ஹ': 'HA'
}

MALAYALAM_TO_BRAHMIC = {
    'അ': 'A', 'ആ': 'AA', 'ഇ': 'I', 'ഈ': 'II', 'ഉ': 'U', 'ഊ': 'UU',
    'എ': 'E', 'ഏ': 'EE', 'ഐ': 'AI', 'ഒ': 'O', 'ഓ': 'OO', 'ഔ': 'AU',
    'ക': 'KA', 'ഖ': 'KHA', 'ഗ': 'GA', 'ഘ': 'GHA', 'ങ': 'NGA',
    'ച': 'CA', 'ഛ': 'CHA', 'ജ': 'JA', 'ഝ': 'JHA', 'ഞ': 'NYA',
    'ട': 'TTA', 'ഠ': 'TTHA', 'ഡ': 'DDA', 'ഢ': 'DDHA', 'ണ': 'NNA',
    'ത': 'TA', 'ഥ': 'THA', 'ദ': 'DA', 'ധ': 'DHA', 'ന': 'NA',
    'പ': 'PA', 'ഫ': 'PHA', 'ബ': 'BA', 'ഭ': 'BHA', 'മ': 'MA',
    'യ': 'YA', 'ര': 'RA', 'ല': 'LA', 'വ': 'VA', 'ശ': 'SHA',
    'ഷ': 'SSA', 'സ': 'SA', 'ഹ': 'HA', 'ള': 'LLA', 'ഴ': 'ZHA', 'റ': 'RRA',
    '്': 'VIRAMA',
    'ാ': 'SIGN_AA', 'ി': 'SIGN_I', 'ീ': 'SIGN_II', 'ു': 'SIGN_U', 'ൂ': 'SIGN_UU',
    'ൃ': 'SIGN_VOC_R', 'െ': 'SIGN_E', 'േ': 'SIGN_EE', 'ൈ': 'SIGN_AI',
    'ൊ': 'SIGN_O', 'ോ': 'SIGN_OO', 'ൌ': 'SIGN_AU',
    'ൺ': 'CHILLU_NN', 'ൻ': 'CHILLU_N', 'ർ': 'CHILLU_R', 'ൽ': 'CHILLU_L',
    'ൾ': 'CHILLU_LL', 'ൿ': 'CHILLU_K'
}

BRAHMIC_TO_TAMIL = {v: k for k, v in TAMIL_TO_BRAHMIC.items()}
BRAHMIC_TO_MALAYALAM = {v: k for k, v in MALAYALAM_TO_BRAHMIC.items()}

# Fallbacks for Brahmic tokens not in Tamil
BRAHMIC_TO_TAMIL_FALLBACKS = {
    'GA': 'க', 'KHA': 'க', 'GHA': 'க',
    'JA': 'ஜ', 'CHA': 'ச', 'JHA': 'ச',
    'DDA': 'ட', 'TTHA': 'ட', 'DDHA': 'ட',
    'DA': 'த', 'THA': 'த', 'DHA': 'த',
    'BA': 'ப', 'PHA': 'ப', 'BHA': 'ப',
    'SHA': 'ச', 'CHILLU_NN': 'ண்', 'CHILLU_N': 'ன்',
    'CHILLU_R': 'ர்', 'CHILLU_L': 'ல்', 'CHILLU_LL': 'ள்', 'CHILLU_K': 'க்'
}

def unicode_to_brahmic(text: str, lang: str = "tam") -> List[str]:
    mapping = TAMIL_TO_BRAHMIC if lang == "tam" else MALAYALAM_TO_BRAHMIC
    tokens = []
    for c in text:
        tokens.append(mapping.get(c, c))
    return tokens

def brahmic_to_unicode(tokens: List[str], lang: str = "tam") -> str:
    if lang == "tam":
        chars = [BRAHMIC_TO_TAMIL.get(t, BRAHMIC_TO_TAMIL_FALLBACKS.get(t, t)) for t in tokens]
    else:
        chars = [BRAHMIC_TO_MALAYALAM.get(t, t) for t in tokens]
    return "".join(chars)

# =====================================================================
# 2. METRICS & LM RESCORER
# =====================================================================

def levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

def calculate_partitioned_metrics(predictions: List[str], references: List[str], partition_tags: List[str]) -> Dict[str, Any]:
    def _compute_subset(preds, refs):
        if not preds:
            return {"exact_match_accuracy": 0.0, "character_error_rate": 0.0, "count": 0}
        exact = 0
        edit_dist = 0
        ref_chars = 0
        for p, r in zip(preds, refs):
            p_clean = p.strip().lower()
            r_clean = r.strip().lower()
            if p_clean == r_clean:
                exact += 1
            edit_dist += levenshtein_distance(p_clean, r_clean)
            ref_chars += max(len(r_clean), 1)
        return {
            "exact_match_accuracy": (exact / len(preds)) * 100.0,
            "character_error_rate": (edit_dist / ref_chars) * 100.0,
            "count": len(preds)
        }
    native_preds, native_refs = [], []
    ne_preds, ne_refs = [], []
    for p, r, tag in zip(predictions, references, partition_tags):
        if tag in {"Dakshina", "AK-Freq", "native"}:
            native_preds.append(p)
            native_refs.append(r)
        else:
            ne_preds.append(p)
            ne_refs.append(r)
    return {
        "combined": _compute_subset(predictions, references),
        "native_words": _compute_subset(native_preds, native_refs),
        "named_entities": _compute_subset(ne_preds, ne_refs)
    }

class UnigramLMRescorer:
    def __init__(self, target_words: List[str], lm_weight: float = 0.5):
        self.lm_weight = lm_weight
        self.counts = Counter(target_words)
        self.total_tokens = max(len(target_words), 1)
        self.vocab_size = len(self.counts)
    def rescore(self, candidate: str) -> str:
        return candidate.strip()

# =====================================================================
# 3. DATA LOADING & BRAHMIC VOCABULARY
# =====================================================================

def extract_from_dict(item: dict) -> Tuple[Optional[str], Optional[str], str]:
    clean_item = {str(k).lower().strip(): v for k, v in item.items() if v is not None}
    native = clean_item.get("native word") or clean_item.get("native_word") or clean_item.get("indic")
    roman = clean_item.get("english word") or clean_item.get("english_word") or clean_item.get("roman")
    source = clean_item.get("source", "native")
    if native and roman:
        return str(native).strip(), str(roman).strip(), str(source).strip()
    return None, None, "native"

def load_split_file(file_path: str, max_samples: Optional[int] = None) -> List[Tuple[str, str, str]]:
    pairs = []
    if not os.path.exists(file_path):
        return pairs
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                n, r, s = extract_from_dict(obj)
                if n and r:
                    pairs.append((n, r, s))
                    if max_samples and len(pairs) >= max_samples:
                        break
            except Exception:
                continue
    return pairs

def get_official_dataset_splits(lang: str, max_train: int = 500000):
    extracted_dir = os.path.join(DATA_DIR, f"extracted_{lang}")
    train_file = os.path.join(extracted_dir, f"{lang}_train.json")
    valid_file = os.path.join(extracted_dir, f"{lang}_valid.json")
    test_file = os.path.join(extracted_dir, f"{lang}_test.json")
    return (
        load_split_file(train_file, max_samples=max_train),
        load_split_file(valid_file),
        load_split_file(test_file)
    )

class TokenVocab:
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    def __init__(self):
        self.t2i = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.i2t = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
        self.n = 4
    def add(self, tok: str):
        if tok not in self.t2i:
            self.t2i[tok] = self.n
            self.i2t[self.n] = tok
            self.n += 1
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.t2i[self.SOS]] + [self.t2i.get(t, self.t2i[self.UNK]) for t in tokens] + [self.t2i[self.EOS]]
    def decode(self, indices: List[int]) -> List[str]:
        out = []
        for idx in indices:
            if idx in {0, 1}:
                continue
            if idx == 2:
                break
            out.append(self.i2t.get(idx, ""))
        return out

class BrahmicDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str, str]], src_vocab: TokenVocab, tgt_vocab: TokenVocab):
        self.data = []
        for src_str, raw_indic, lang, source_tag in samples:
            brahmic_tokens = unicode_to_brahmic(raw_indic, lang=lang)
            src_tokens = []
            if src_str.startswith("__ta__"):
                src_tokens = ["__ta__"] + list(src_str[6:])
            elif src_str.startswith("__ml__"):
                src_tokens = ["__ml__"] + list(src_str[6:])
            else:
                src_tokens = list(src_str)
                
            self.data.append((
                src_vocab.encode(src_tokens),
                tgt_vocab.encode(brahmic_tokens),
                raw_indic,
                lang,
                source_tag
            ))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

def brahmic_pad_collate(batch):
    src_list, tgt_list, raw_indic_list, lang_list, source_tag_list = zip(*batch)
    max_s = max(len(s) for s in src_list)
    max_t = max(len(t) for t in tgt_list)
    s_tensor = torch.zeros(len(src_list), max_s, dtype=torch.long)
    t_tensor = torch.zeros(len(tgt_list), max_t, dtype=torch.long)
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        s_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        t_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)
    return s_tensor, t_tensor, raw_indic_list, lang_list, source_tag_list

# =====================================================================
# 4. MODEL ARCHITECTURE
# =====================================================================

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

class BrahmicTransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        device = src.device
        tgt_in = tgt[:, :-1]
        src_mask = (src == 0)
        tgt_mask = (tgt_in == 0)
        causal_mask = self.generate_square_subsequent_mask(tgt_in.size(1), device)
        
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        logits = self.out_proj(out)
        
        padded = torch.zeros(logits.size(0), logits.size(1) + 1, logits.size(2), device=device)
        padded[:, 1:] = logits
        return padded

    def greedy_decode(self, src: torch.Tensor, max_len: int = 40) -> List[List[int]]:
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
# 5. TRAINING & EVALUATION PIPELINE
# =====================================================================

def evaluate_brahmic_split(model: BrahmicTransformerSeq2Seq, loader: DataLoader, tgt_vocab: TokenVocab, lang: str):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = next(model.parameters()).device
    total_loss = 0.0
    preds, refs, tags = [], [], []
    with torch.no_grad():
        for s_t, t_t, raw_indic, l_list, tag_list in loader:
            s_t, t_t = s_t.to(device), t_t.to(device)
            out = model(s_t, t_t)
            loss = criterion(out[:, 1:].reshape(-1, tgt_vocab.n), t_t[:, 1:].reshape(-1))
            total_loss += loss.item()
            decoded = model.greedy_decode(s_t)
            for b in range(len(decoded)):
                brahmic_tokens = tgt_vocab.decode(decoded[b])
                pred_unicode = brahmic_to_unicode(brahmic_tokens, lang=lang)
                preds.append(pred_unicode)
                refs.append(raw_indic[b])
                tags.append(tag_list[b])
    avg_loss = total_loss / max(len(loader), 1)
    metrics = calculate_partitioned_metrics(preds, refs, tags)
    return avg_loss, metrics

def run_brahmic_experiment(epochs: int = 8, batch_size: int = 256, lr: float = 5e-4):
    print("\n" + "=" * 80, flush=True)
    print(" PROJECT VALIMELI — COMMON BRAHMIC SCRIPT-UNIFIED BILINGUAL MATRIX", flush=True)
    print(" 100% Token-ID Parameter Sharing Across Tamil & Malayalam (1.0M Pairs Total)", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" -> Compute Device: {device}", flush=True)
    
    tam_train, tam_val, tam_test = get_official_dataset_splits("tam", max_train=500000)
    mal_train, mal_val, mal_test = get_official_dataset_splits("mal", max_train=500000)
    
    src_vocab = TokenVocab()
    tgt_vocab = TokenVocab()
    
    src_vocab.add("__ta__")
    src_vocab.add("__ml__")
    for c in "abcdefghijklmnopqrstuvwxyz":
        src_vocab.add(c)
        
    for v in TAMIL_TO_BRAHMIC.values():
        tgt_vocab.add(v)
    for v in MALAYALAM_TO_BRAHMIC.values():
        tgt_vocab.add(v)
        
    print(f" -> Unified Target Brahmic Vocab Size: {tgt_vocab.n} tokens (vs 120 separate Unicode tokens!)", flush=True)
    
    joint_train = []
    for n, r, s in tam_train:
        joint_train.append((f"__ta__{r}", n, "tam", s))
    for n, r, s in mal_train:
        joint_train.append((f"__ml__{r}", n, "mal", s))
    random.shuffle(joint_train)
    
    joint_val = [(f"__ta__{r}", n, "tam", s) for n, r, s in tam_val] + [(f"__ml__{r}", n, "mal", s) for n, r, s in mal_val]
    tam_test_samples = [(f"__ta__{r}", n, "tam", s) for n, r, s in tam_test]
    mal_test_samples = [(f"__ml__{r}", n, "mal", s) for n, r, s in mal_test]
    
    train_loader = DataLoader(BrahmicDataset(joint_train, src_vocab, tgt_vocab), batch_size=batch_size, shuffle=True, collate_fn=brahmic_pad_collate)
    val_loader = DataLoader(BrahmicDataset(joint_val, src_vocab, tgt_vocab), batch_size=batch_size, shuffle=False, collate_fn=brahmic_pad_collate)
    tam_test_loader = DataLoader(BrahmicDataset(tam_test_samples, src_vocab, tgt_vocab), batch_size=batch_size, shuffle=False, collate_fn=brahmic_pad_collate)
    mal_test_loader = DataLoader(BrahmicDataset(mal_test_samples, src_vocab, tgt_vocab), batch_size=batch_size, shuffle=False, collate_fn=brahmic_pad_collate)
    
    model = BrahmicTransformerSeq2Seq(src_vocab.n, tgt_vocab.n, d_model=256, nhead=4, num_layers=6, dim_feedforward=1024).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Trainable Parameters: {num_params:,} ({num_params / 1e6:.2f}M)", flush=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float("inf")
    ckpt_path = os.path.join(SCRATCH_DIR, "brahmic_best.pt")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for idx, (s_b, t_b, _, _, _) in enumerate(train_loader):
            s_b, t_b = s_b.to(device), t_b.to(device)
            optimizer.zero_grad()
            out = model(s_b, t_b)
            loss = criterion(out[:, 1:].reshape(-1, tgt_vocab.n), t_b[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            if (idx + 1) % 500 == 0 or (idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch+1}/{epochs}] Batch [{idx+1}/{len(train_loader)}] ({(idx+1)/len(train_loader)*100:.1f}%) | Loss: {loss.item():.4f}", flush=True)
                
        val_loss, _ = evaluate_brahmic_split(model, val_loader, tgt_vocab, "tam")
        dur = time.time() - start_time
        print(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ({dur:.1f}s) | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} ===", flush=True)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ⭐ New best checkpoint saved (Val Loss: {best_val_loss:.4f})", flush=True)
            
    print(f"\n -> Evaluating Best Checkpoint on Official Holdout Test Sets...", flush=True)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    _, tam_metrics = evaluate_brahmic_split(model, tam_test_loader, tgt_vocab, "tam")
    _, mal_metrics = evaluate_brahmic_split(model, mal_test_loader, tgt_vocab, "mal")
    
    print("\n" + "*" * 80, flush=True)
    print(" COMMON BRAHMIC UNIFIED BILINGUAL RESULTS:", flush=True)
    print(f"   ► TAMIL NATIVE WORDS:      {tam_metrics['native_words']['exact_match_accuracy']:.2f}% EM | CER: {tam_metrics['native_words']['character_error_rate']:.2f}%", flush=True)
    print(f"   ► TAMIL NAMED ENTITIES:    {tam_metrics['named_entities']['exact_match_accuracy']:.2f}% EM | CER: {tam_metrics['named_entities']['character_error_rate']:.2f}%", flush=True)
    print(f"   ► TAMIL OVERALL COMBINED:  {tam_metrics['combined']['exact_match_accuracy']:.2f}% EM | CER: {tam_metrics['combined']['character_error_rate']:.2f}%", flush=True)
    print("-" * 80, flush=True)
    print(f"   ► MALAYALAM NATIVE WORDS:  {mal_metrics['native_words']['exact_match_accuracy']:.2f}% EM | CER: {mal_metrics['native_words']['character_error_rate']:.2f}%", flush=True)
    print(f"   ► MALAYALAM NAMED ENTITIES:{mal_metrics['named_entities']['exact_match_accuracy']:.2f}% EM | CER: {mal_metrics['named_entities']['character_error_rate']:.2f}%", flush=True)
    print(f"   ► MALAYALAM OVERALL:       {mal_metrics['combined']['exact_match_accuracy']:.2f}% EM | CER: {mal_metrics['combined']['character_error_rate']:.2f}%", flush=True)
    print("*" * 80 + "\n", flush=True)
    
    res = {
        "tam": tam_metrics,
        "mal": mal_metrics,
        "best_val_loss": best_val_loss
    }
    with open(os.path.join(ARTIFACTS_DIR, "brahmic_bilingual_results.json"), "w") as f:
        json.dump(res, f, indent=2)

if __name__ == "__main__":
    run_brahmic_experiment()
