#!/usr/bin/env python3
"""
Project ValiMeli — Multi-Seed Rigorous Benchmark & Paired McNemar Analysis
==========================================================================
Executes multiple randomized seeds (42, 43, 44, ...) for:
  - 25k Low-Resource Cells (Tamil & Malayalam: A0 vs A1-MT)
  - 250k Standard Cells (Tamil & Malayalam: A0 vs A1-MT)

Research Hygiene & Auditing Standards:
  1. Full deterministic seeding across torch (CPU + MPS), numpy, and random.
  2. Provenance tracking: Git commit SHA, git dirty status, PyTorch & Python versions, device.
  3. Instance-level prediction dumps (JSONL) with ID, input, target, pred, correctness, and partition tag.
  4. Exact paired McNemar's tests (2x2 contingency matrix, chi2, exact binomial two-tailed p-value).
  5. Multi-metric evaluation: Top-1 Exact Match (EM), Character Error Rate (CER), Native EM, and Named Entity EM.
  6. Aggregated mean ± standard deviation across seeds saved to artifacts/multiseed_rigorous_results.json.
"""

import os
import sys
import json
import math
import time
import shutil
import random
import argparse
import subprocess
import unicodedata
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from scipy import stats

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs_multiseed")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
PREDICTIONS_DIR = os.path.join(ARTIFACTS_DIR, "predictions")

os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(PREDICTIONS_DIR, exist_ok=True)

def get_git_info(repo_path: str) -> Dict[str, Any]:
    try:
        env = dict(os.environ)
        env["GIT_CONFIG_GLOBAL"] = "/dev/null"
        env["GIT_CONFIG_SYSTEM"] = "/dev/null"
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_path, env=env, capture_output=True, text=True, check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain"], cwd=repo_path, env=env, capture_output=True, text=True, check=True).stdout.strip()
        return {"commit_sha": commit, "is_dirty": len(status) > 0}
    except Exception as e:
        return {"commit_sha": "unknown", "is_dirty": True, "error": str(e)}

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def compute_levenshtein_distance(s1: str, s2: str) -> int:
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    return dp[m][n]

# =====================================================================
# LINGUISTIC PHONOLOGY LABELLING
# =====================================================================

# Tamil Grammatical Terms (Tolkāppiyam, Niklas 1988)
# - Eḻuttu (எழுத்து): Basic orthographic / syllabic grapheme unit
# - Puḷḷi (புள்ளி): Virama / dot indicating pure consonant
# - Vallinam (வல்லினம்): Hard stops / plosives
# - Mellinam (மெல்லினம்): Soft nasals
# - Puṇarcci (புணர்ச்சி): Morphophonemic juncture / sandhi

TAMIL_VALLINAM = {'க', 'ச', 'ட', 'த', 'ப', 'ற'}
TAMIL_PLOSIVES = TAMIL_VALLINAM
TAMIL_MELLINAM = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}
TAMIL_NASALS = TAMIL_MELLINAM
TAMIL_VOWEL_SIGNS = {
    '\u0bbe', '\u0bbf', '\u0bc0', '\u0bc1', '\u0bc2', 
    '\u0bc6', '\u0bc7', '\u0bc8', '\u0bca', '\u0bcb', '\u0bcc'
}
TAMIL_PULLI = '\u0bcd'
TAMIL_VIRAMA = TAMIL_PULLI

MALAYALAM_PLOSIVES = {'ക', 'ച', 'ട', 'ത', 'പ', 'റ'}
MALAYALAM_NASALS = {'ങ', 'ഞ', 'ണ', 'ന', 'മ'}
MALAYALAM_VOWEL_SIGNS = {
    '\u0d3e', '\u0d3f', '\u0d40', '\u0d41', '\u0d42', '\u0d43', '\u0d44',
    '\u0d46', '\u0d47', '\u0d48', '\u0d4a', '\u0d4b', '\u0d4c'
}
MALAYALAM_CHANDRAKKALA = '\u0d4d'
MALAYALAM_VIRAMA = MALAYALAM_CHANDRAKKALA

PHONO_NONE = 0
PHONO_INIT = 1
PHONO_GEM = 2
PHONO_NASAL = 3
PHONO_INTER = 4
PHONO_DEF = 5

def segment_eluttu(word: str, lang: str = "tam") -> List[str]:
    """Segments a native word into eḻuttu (syllabic graphemic units)."""
    vowel_signs = TAMIL_VOWEL_SIGNS if lang == "tam" else MALAYALAM_VOWEL_SIGNS
    pulli = TAMIL_PULLI if lang == "tam" else MALAYALAM_CHANDRAKKALA
    units = []
    current = ""
    for char in word:
        if char in vowel_signs or char == pulli:
            current += char
        else:
            if current:
                units.append(current)
            current = char
    if current:
        units.append(current)
    return units

# Backwards compatibility alias
segment_aksharas = segment_eluttu

def get_phonotactic_label(word: str, eluttu_idx: int, eluttukkal: List[str], lang: str = "tam") -> int:
    """Computes the phonotactic label for a plosive (vallinam) eḻuttu."""
    plosives = TAMIL_VALLINAM if lang == "tam" else MALAYALAM_PLOSIVES
    nasals = TAMIL_MELLINAM if lang == "tam" else MALAYALAM_NASALS
    pulli = TAMIL_PULLI if lang == "tam" else MALAYALAM_CHANDRAKKALA
    
    current = eluttukkal[eluttu_idx]
    if current[0] not in plosives:
        return PHONO_NONE
        
    if pulli in current:
        if eluttu_idx + 1 < len(eluttukkal) and eluttukkal[eluttu_idx + 1][0] == current[0]:
            return PHONO_GEM
            
    if eluttu_idx > 0:
        prev = eluttukkal[eluttu_idx - 1]
        if pulli in prev and prev[0] == current[0]:
            return PHONO_GEM
        if pulli in prev and prev[0] in nasals:
            return PHONO_NASAL
            
    if pulli in current:
        return PHONO_DEF
    if eluttu_idx == 0:
        return PHONO_INIT
    if eluttu_idx > 0:
        prev = eluttukkal[eluttu_idx - 1]
        if pulli not in prev:
            return PHONO_INTER
            
    return PHONO_DEF

def generate_char_level_phonology_labels(word: str, lang: str = "tam") -> List[int]:
    eluttukkal = segment_eluttu(word, lang=lang)
    labels = []
    for idx, el in enumerate(eluttukkal):
        plabel = get_phonotactic_label(word, idx, eluttukkal, lang=lang)
        for _ in el:
            labels.append(plabel)
    return labels

# =====================================================================
# DATASET & VOCABULARY
# =====================================================================

def load_split_file(filepath: str, max_samples: Optional[int] = None) -> List[Tuple[str, str, str]]:
    pairs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
                n = item.get("native word", "").strip()
                r = item.get("english word", "").strip().lower()
                s = item.get("source", "unknown")
                if n and r:
                    pairs.append((n, r, s))
                    if max_samples and len(pairs) >= max_samples:
                        break
            except Exception:
                continue
    return pairs

def get_official_splits(lang: str, max_train_samples: Optional[int] = None):
    extracted_dir = os.path.join(DATA_DIR, f"extracted_{lang}")
    train_file = os.path.join(extracted_dir, f"{lang}_train.json")
    valid_file = os.path.join(extracted_dir, f"{lang}_valid.json")
    test_file = os.path.join(extracted_dir, f"{lang}_test.json")
    
    train_pairs = load_split_file(train_file, max_samples=max_train_samples)
    valid_pairs = load_split_file(valid_file, max_samples=None)
    test_pairs = load_split_file(test_file, max_samples=None)
    return train_pairs, valid_pairs, test_pairs

class CharVocab:
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    def __init__(self):
        self.c2i = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.i2c = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
        self.size = 4
        
    def add(self, c: str):
        if c not in self.c2i:
            self.c2i[c] = self.size
            self.i2c[self.size] = c
            self.size += 1
            
    def encode(self, text: str) -> List[int]:
        return [1] + [self.c2i.get(c, 3) for c in text] + [2]
        
    def decode(self, indices: List[int]) -> str:
        chars = []
        for idx in indices:
            if idx in (0, 1): continue
            if idx == 2: break
            chars.append(self.i2c.get(idx, ""))
        return "".join(chars)

class TranslitDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], vocab_src: CharVocab, vocab_tgt: CharVocab, lang: str = "tam"):
        self.data = []
        for indic, roman, src_tag in pairs:
            phono = generate_char_level_phonology_labels(indic, lang=lang)
            phono_enc = [0] + phono + [0]
            self.data.append((
                vocab_src.encode(roman),
                vocab_tgt.encode(indic),
                phono_enc,
                indic,
                roman,
                src_tag
            ))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def pad_collate_fn(batch):
    srcs, tgts, phonos, raw_indics, raw_romans, src_tags = zip(*batch)
    max_src = max(len(s) for s in srcs)
    max_tgt = max(len(t) for t in tgts)
    
    src_tensor = torch.zeros(len(srcs), max_src, dtype=torch.long)
    tgt_tensor = torch.zeros(len(tgts), max_tgt, dtype=torch.long)
    phono_tensor = torch.zeros(len(phonos), max_tgt, dtype=torch.long)
    
    for i, (s, t, p) in enumerate(zip(srcs, tgts, phonos)):
        src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        phono_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long)
        
    return src_tensor, tgt_tensor, phono_tensor, raw_indics, raw_romans, src_tags

# =====================================================================
# TRANSFORMER ARCHITECTURE (MPS Compatible, enable_nested_tensor=False)
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

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, num_phono_classes: int = 6,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True, dropout=0.1
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.char_head = nn.Linear(d_model, tgt_vocab_size)
        self.phono_head = nn.Linear(d_model, num_phono_classes)
        
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        device = src.device
        tgt_in = tgt[:, :-1]
        
        src_mask = (src == 0)
        tgt_mask = (tgt_in == 0)
        causal_mask = self.generate_square_subsequent_mask(tgt_in.size(1), device)
        
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt_in) * math.sqrt(self.d_model))
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        char_logits = self.char_head(out)
        phono_logits = self.phono_head(out)
        
        batch_size, seq_len, vocab_size = char_logits.size()
        padded_char = torch.zeros(batch_size, seq_len + 1, vocab_size, device=device)
        padded_char[:, 1:] = char_logits
        
        padded_phono = torch.zeros(batch_size, seq_len + 1, 6, device=device)
        padded_phono[:, 1:] = phono_logits
        
        return padded_char, padded_phono

    def generate_step(self, src: torch.Tensor, curr_tgt: torch.Tensor):
        device = src.device
        src_mask = (src == 0)
        tgt_mask = (curr_tgt == 0)
        causal_mask = self.generate_square_subsequent_mask(curr_tgt.size(1), device)
        
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_embed(curr_tgt) * math.sqrt(self.d_model))
        
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask
        )
        return self.char_head(out)

# =====================================================================
# EVALUATION & METRICS
# =====================================================================

def evaluate_on_split(model: nn.Module, data_loader: DataLoader, vocab_tgt: CharVocab, device: torch.device):
    model.eval()
    predictions = []
    references = []
    sources = []
    src_tags = []
    total_edit_dist = 0
    total_ref_chars = 0
    
    with torch.no_grad():
        for src_batch, tgt_batch, phono_batch, raw_indics, raw_romans, tags in data_loader:
            src_batch = src_batch.to(device)
            bsz = src_batch.size(0)
            
            curr_tgt = torch.full((bsz, 1), 1, dtype=torch.long, device=device)
            finished = torch.zeros(bsz, dtype=torch.bool, device=device)
            
            for _ in range(40):
                char_logits = model.generate_step(src_batch, curr_tgt)
                next_tok = char_logits[:, -1, :].argmax(dim=-1, keepdim=True)
                curr_tgt = torch.cat([curr_tgt, next_tok], dim=1)
                
                finished |= (next_tok.squeeze(1) == 2)
                if finished.all():
                    break
                    
            for b in range(bsz):
                pred_str = vocab_tgt.decode(curr_tgt[b].tolist())
                ref_str = raw_indics[b]
                predictions.append(pred_str)
                references.append(ref_str)
                sources.append(raw_romans[b])
                src_tags.append(tags[b])
                
                total_edit_dist += compute_levenshtein_distance(pred_str, ref_str)
                total_ref_chars += max(len(ref_str), 1)
                
    total = len(references)
    em_hits = sum(1 for p, r in zip(predictions, references) if p == r)
    em_acc = round((em_hits / total) * 100, 2)
    cer = round((total_edit_dist / max(total_ref_chars, 1)) * 100, 2)
    
    native_tot = sum(1 for t in src_tags if "dakshina" in t.lower() or "native" in t.lower() or "wiki" in t.lower())
    native_hits = sum(1 for p, r, t in zip(predictions, references, src_tags) if p == r and ("dakshina" in t.lower() or "native" in t.lower() or "wiki" in t.lower()))
    native_acc = round((native_hits / max(native_tot, 1)) * 100, 2)
    
    ne_tot = total - native_tot
    ne_hits = em_hits - native_hits
    ne_acc = round((ne_hits / max(ne_tot, 1)) * 100, 2) if ne_tot > 0 else 0.0
    
    records = []
    for idx, (src, ref, pred, tag) in enumerate(zip(sources, references, predictions, src_tags)):
        records.append({
            "id": idx,
            "input": src,
            "gold": ref,
            "pred": pred,
            "correct": (pred == ref),
            "source_tag": tag
        })
        
    return {
        "exact_match_acc": em_acc,
        "character_error_rate": cer,
        "native_words_acc": native_acc,
        "named_entities_acc": ne_acc,
        "total_samples": total,
        "records": records
    }

# =====================================================================
# SINGLE CELL EXECUTION
# =====================================================================

def train_single_run(lang: str, arm: str, max_train_samples: int, seed: int, epochs: int = 8, batch_size: int = 256, aux_lambda: float = 0.3):
    start_time = time.time()
    set_seed(seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    train_pairs, valid_pairs, test_pairs = get_official_splits(lang, max_train_samples=max_train_samples)
    
    vocab_src = CharVocab()
    vocab_tgt = CharVocab()
    for _, r, _ in train_pairs:
        for c in r: vocab_src.add(c)
    for n, _, _ in train_pairs:
        for c in n: vocab_tgt.add(c)
        
    train_ds = TranslitDataset(train_pairs, vocab_src, vocab_tgt, lang=lang)
    val_ds = TranslitDataset(valid_pairs, vocab_src, vocab_tgt, lang=lang)
    test_ds = TranslitDataset(test_pairs, vocab_src, vocab_tgt, lang=lang)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    model = TransformerSeq2Seq(vocab_src.size, vocab_tgt.size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    char_crit = nn.CrossEntropyLoss(ignore_index=0)
    phono_crit = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_loss = float("inf")
    best_model_weights = None
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for src, tgt, phono, _, _, _ in train_loader:
            src, tgt, phono = src.to(device), tgt.to(device), phono.to(device)
            optimizer.zero_grad()
            char_logits, phono_logits = model(src, tgt)
            
            loss_char = char_crit(char_logits[:, 1:].reshape(-1, vocab_tgt.size), tgt[:, 1:].reshape(-1))
            loss_phono = phono_crit(phono_logits[:, 1:].reshape(-1, 6), phono[:, 1:].reshape(-1))
            
            loss = loss_char + (aux_lambda * loss_phono) if "A1" in arm else loss_char
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss_char.item()
            
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt, _, _, _, _ in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                char_logits, _ = model(src, tgt)
                v_loss = char_crit(char_logits[:, 1:].reshape(-1, vocab_tgt.size), tgt[:, 1:].reshape(-1))
                val_loss += v_loss.item()
        val_loss /= max(len(val_loader), 1)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            
    # Load best model for test evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_model_weights.items()})
    eval_res = evaluate_on_split(model, test_loader, vocab_tgt, device)
    
    duration = round(time.time() - start_time, 1)
    pred_file = os.path.join(PREDICTIONS_DIR, f"predictions_{lang}_{arm}_{max_train_samples//1000}k_seed{seed}.jsonl")
    with open(pred_file, "w", encoding="utf-8") as f:
        for rec in eval_res["records"]:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            
    print(f"[{lang.upper()} | {arm} | {max_train_samples//1000}k | Seed {seed}] EM: {eval_res['exact_match_acc']}% | CER: {eval_res['character_error_rate']}% | Native: {eval_res['native_words_acc']}% | Time: {duration}s", flush=True)
    
    return {
        "lang": lang,
        "arm": arm,
        "scale": max_train_samples,
        "seed": seed,
        "duration_seconds": duration,
        "test_em": eval_res["exact_match_acc"],
        "test_cer": eval_res["character_error_rate"],
        "native_em": eval_res["native_words_acc"],
        "named_entities_em": eval_res["named_entities_acc"],
        "prediction_file": pred_file
    }

# =====================================================================
# PAIRED MCNEMAR'S TEST
# =====================================================================

def compute_mcnemar_paired_test(file_a0: str, file_a1: str):
    recs_a0 = []
    with open(file_a0, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): recs_a0.append(json.loads(line))
            
    recs_a1 = []
    with open(file_a1, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip(): recs_a1.append(json.loads(line))
            
    assert len(recs_a0) == len(recs_a1), "Prediction lists must be paired!"
    
    n00, n01, n10, n11 = 0, 0, 0, 0
    for r0, r1 in zip(recs_a0, recs_a1):
        c0, c1 = r0["correct"], r1["correct"]
        if not c0 and not c1: n00 += 1
        elif not c0 and c1: n01 += 1
        elif c0 and not c1: n10 += 1
        else: n11 += 1
        
    b, c = n01, n10
    total_discordant = b + c
    
    if total_discordant > 0:
        p_val = stats.binomtest(b, total_discordant, 0.5, alternative="two-sided").pvalue
        chi2_stat = ((abs(b - c) - 1) ** 2) / total_discordant
    else:
        p_val = 1.0
        chi2_stat = 0.0
        
    return {
        "contingency_matrix": {"n00": n00, "n01_A1_wins": b, "n10_A0_wins": c, "n11": n11},
        "discordant_total": total_discordant,
        "mcnemar_chi2": round(chi2_stat, 4),
        "exact_mcnemar_p_value": float(f"{p_val:.6f}"),
        "net_gain": b - c
    }

# =====================================================================
# MAIN RUNNER
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-seed Rigorous Benchmark Runner")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--scales", nargs="+", type=int, default=[25000])
    parser.add_argument("--langs", nargs="+", type=str, default=["tam", "mal"])
    args = parser.parse_args()
    
    git_info = get_git_info(WORKSPACE_DIR)
    
    print("=" * 80, flush=True)
    print(" PROJECT VALIMELI — MULTI-SEED RIGOROUS BENCHMARK & MCNEMAR ANALYSIS", flush=True)
    print(f" Git Commit: {git_info['commit_sha']} (Dirty: {git_info['is_dirty']})", flush=True)
    print(f" PyTorch: {torch.__version__} | Device: {'mps' if torch.backends.mps.is_available() else 'cpu'}", flush=True)
    print(f" Seeds: {args.seeds} | Scales: {args.scales} | Langs: {args.langs}", flush=True)
    print("=" * 80, flush=True)
    
    all_runs = []
    
    for scale in args.scales:
        for lang in args.langs:
            for seed in args.seeds:
                print(f"\n>>> Running {lang.upper()} Scale: {scale//1000}k | Seed: {seed} | Arm: A0 (Baseline)...", flush=True)
                res_a0 = train_single_run(lang, "A0", scale, seed)
                print(f">>> Running {lang.upper()} Scale: {scale//1000}k | Seed: {seed} | Arm: A1-MT (Multi-Task)...", flush=True)
                res_a1 = train_single_run(lang, "A1-MT", scale, seed)
                all_runs.extend([res_a0, res_a1])
                
    summary = {}
    mcnemar_results = {}
    
    for scale in args.scales:
        scale_k = f"{scale//1000}k"
        summary[scale_k] = {}
        mcnemar_results[scale_k] = {}
        
        for lang in args.langs:
            a0_runs = [r for r in all_runs if r["lang"] == lang and r["arm"] == "A0" and r["scale"] == scale]
            a1_runs = [r for r in all_runs if r["lang"] == lang and r["arm"] == "A1-MT" and r["scale"] == scale]
            
            a0_ems = [r["test_em"] for r in a0_runs]
            a1_ems = [r["test_em"] for r in a1_runs]
            a0_cers = [r["test_cer"] for r in a0_runs]
            a1_cers = [r["test_cer"] for r in a1_runs]
            a0_natives = [r["native_em"] for r in a0_runs]
            a1_natives = [r["native_em"] for r in a1_runs]
            
            summary[scale_k][lang] = {
                "A0": {
                    "mean_em": round(float(np.mean(a0_ems)), 2),
                    "std_em": round(float(np.std(a0_ems)), 2),
                    "mean_cer": round(float(np.mean(a0_cers)), 2),
                    "mean_native_em": round(float(np.mean(a0_natives)), 2),
                    "runs_em": a0_ems
                },
                "A1-MT": {
                    "mean_em": round(float(np.mean(a1_ems)), 2),
                    "std_em": round(float(np.std(a1_ems)), 2),
                    "mean_cer": round(float(np.mean(a1_cers)), 2),
                    "mean_native_em": round(float(np.mean(a1_natives)), 2),
                    "runs_em": a1_ems
                },
                "delta_em": round(float(np.mean(a1_ems) - np.mean(a0_ems)), 2)
            }
            
            # Compute McNemar's paired test for seed 42
            p_a0_seed42 = [r["prediction_file"] for r in a0_runs if r["seed"] == 42][0]
            p_a1_seed42 = [r["prediction_file"] for r in a1_runs if r["seed"] == 42][0]
            mcn = compute_mcnemar_paired_test(p_a0_seed42, p_a1_seed42)
            mcnemar_results[scale_k][lang] = mcn
            
    final_output = {
        "metadata": {
            "git_commit": git_info["commit_sha"],
            "git_dirty": git_info["is_dirty"],
            "pytorch_version": torch.__version__,
            "python_version": sys.version.split()[0],
            "seeds": args.seeds,
            "scales": args.scales,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "summary": summary,
        "mcnemar_paired_tests": mcnemar_results,
        "all_individual_runs": all_runs
    }
    
    out_file = os.path.join(ARTIFACTS_DIR, "multiseed_rigorous_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)
        
    print("\n" + "=" * 80, flush=True)
    print(f"⭐ MULTI-SEED RIGOROUS BENCHMARK COMPLETE!", flush=True)
    print(f" Results saved to: {out_file}", flush=True)
    print(json.dumps(final_output, indent=2), flush=True)
    print("=" * 80, flush=True)

if __name__ == "__main__":
    main()
