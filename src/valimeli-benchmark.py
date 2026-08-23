#!/usr/bin/env python3
"""
Project ValiMeli (வலி–മെலி) — Standardized Benchmarking Pipeline (v9 - Bilingual & Multi-Task Benchmark)
Joint Bilingual Dravidian Modeling (Tamil + Malayalam) & Monolingual Benchmarks

Features:
1. Joint Bilingual Dravidian Model (Tamil + Malayalam):
   - 1.0 Million combined training pairs (500k Tamil + 500k Malayalam)
   - Language conditioning prefix tags (__ta__, __ml__)
   - Unified shared character vocabulary
   - Joint evaluation on full official holdout test sets for both Tamil (11,499) and Malayalam (12,451)
2. Comparison of Bilingual-A0 (Baseline) vs Bilingual-A1-MT (Multi-Task Phonology).
3. Detailed partitioned reporting on Native Words vs Named Entities (NER Cross-Lingual Transfer).
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
from typing import List, Tuple, Dict, Generator, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
os.environ["MPLCONFIGDIR"] = os.path.join(SCRATCH_DIR, "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
DOCS_DIR = os.path.join(WORKSPACE_DIR, "docs")

MAX_DISK_WRITE_GB = 180.0

def check_disk_usage(additional_bytes_to_write: int = 0):
    total_size_bytes = 0
    if os.path.exists(SCRATCH_DIR):
        for dirpath, _, filenames in os.walk(SCRATCH_DIR):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                try:
                    total_size_bytes += os.path.getsize(fp)
                except OSError:
                    continue
                
    total_size_gb = (total_size_bytes + additional_bytes_to_write) / (1024 ** 3)
    try:
        total, used, free = shutil.disk_usage(WORKSPACE_DIR)
    except Exception:
        total, used, free = 500 * (1024**3), 50 * (1024**3), 450 * (1024**3)
    free_gb = free / (1024 ** 3)
    
    if total_size_gb > MAX_DISK_WRITE_GB:
        print(f"\n[CRITICAL ERROR] Disk guard triggered! ValiMeli scratch size: {total_size_gb:.2f} GB > {MAX_DISK_WRITE_GB} GB.")
        sys.exit(1)
    if free_gb < 2.0 and total > 0:
        print(f"\n[CRITICAL ERROR] Host disk space low: {free_gb:.2f} GB free. Halting.")
        sys.exit(1)

os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

# =====================================================================
# 2. LINGUISTIC RULES & PHONOLOGY TARGET GENERATOR
# =====================================================================

TAMIL_PLOSIVES = {'க', 'ச', 'ட', 'த', 'ப', 'ற'}
TAMIL_NASALS = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}
TAMIL_VOWEL_SIGNS = {
    '\u0bbe', '\u0bbf', '\u0bc0', '\u0bc1', '\u0bc2', 
    '\u0bc6', '\u0bc7', '\u0bc8', '\u0bca', '\u0bcb', '\u0bcc'
}
TAMIL_VIRAMA = '\u0bcd'

MALAYALAM_PLOSIVES = {'ക', 'ച', 'ട', 'ത', 'പ', 'റ'}
MALAYALAM_NASALS = {'ങ', 'ഞ', 'ണ', 'ന', 'മ'}
MALAYALAM_VOWEL_SIGNS = {
    '\u0d3e', '\u0d3f', '\u0d40', '\u0d41', '\u0d42', '\u0d43', '\u0d44',
    '\u0d46', '\u0d47', '\u0d48', '\u0d4a', '\u0d4b', '\u0d4c'
}
MALAYALAM_VIRAMA = '\u0d4d'

PHONO_NONE = 0
PHONO_INIT = 1
PHONO_GEM = 2
PHONO_NASAL = 3
PHONO_INTER = 4
PHONO_DEF = 5

def segment_aksharas(word: str, lang: str = "tam") -> List[str]:
    vowel_signs = TAMIL_VOWEL_SIGNS if lang == "tam" else MALAYALAM_VOWEL_SIGNS
    virama = TAMIL_VIRAMA if lang == "tam" else MALAYALAM_VIRAMA
    units = []
    current = ""
    for char in word:
        if char in vowel_signs or char == virama:
            current += char
        else:
            if current:
                units.append(current)
            current = char
    if current:
        units.append(current)
    return units

def get_phonotactic_label(word: str, akshara_idx: int, aksharas: List[str], lang: str = "tam") -> int:
    plosives = TAMIL_PLOSIVES if lang == "tam" else MALAYALAM_PLOSIVES
    nasals = TAMIL_NASALS if lang == "tam" else MALAYALAM_NASALS
    virama = TAMIL_VIRAMA if lang == "tam" else MALAYALAM_VIRAMA
    
    current = aksharas[akshara_idx]
    if current[0] not in plosives:
        return PHONO_NONE
        
    if virama in current:
        if akshara_idx + 1 < len(aksharas) and aksharas[akshara_idx + 1][0] == current[0]:
            return PHONO_GEM
            
    if akshara_idx > 0:
        prev = aksharas[akshara_idx - 1]
        if virama in prev and prev[0] == current[0]:
            return PHONO_GEM
        if virama in prev and prev[0] in nasals:
            return PHONO_NASAL
            
    if virama in current:
        return PHONO_DEF
    if akshara_idx == 0:
        return PHONO_INIT
    if akshara_idx > 0:
        prev = aksharas[akshara_idx - 1]
        if virama not in prev:
            return PHONO_INTER
            
    return PHONO_DEF

def generate_char_level_phonology_labels(word: str, lang: str = "tam") -> List[int]:
    aksharas = segment_aksharas(word, lang=lang)
    labels = []
    for idx, ak in enumerate(aksharas):
        phono_label = get_phonotactic_label(word, idx, aksharas, lang=lang)
        labels.append(phono_label)
        for _ in range(len(ak) - 1):
            labels.append(PHONO_NONE)
    return labels

# =====================================================================
# 3. METRICS ENGINE
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
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )
    return dp[m][n]

def calculate_partitioned_metrics(predictions: List[str], references: List[str], 
                                  indic_sources: List[str], partition_tags: List[str],
                                  lang: str) -> Dict[str, Any]:
    plosives = TAMIL_PLOSIVES if lang == "tam" else MALAYALAM_PLOSIVES
    
    def _compute_subset(preds, refs, indics):
        if not preds:
            return {"exact_match_accuracy": 0.0, "character_error_rate": 0.0, "stop_voicing_accuracy": 0.0, "count": 0}
        exact = 0
        edit_dist = 0
        ref_chars = 0
        sva_corr = 0
        sva_tot = 0
        
        for p, r, ind in zip(preds, refs, indics):
            p_clean = p.strip().lower()
            r_clean = r.strip().lower()
            is_match = (p_clean == r_clean)
            if is_match:
                exact += 1
            edit_dist += levenshtein_distance(p_clean, r_clean)
            ref_chars += max(len(r_clean), 1)
            
            if any(c in ind for c in plosives):
                sva_tot += 1
                if is_match:
                    sva_corr += 1
                    
        return {
            "exact_match_accuracy": (exact / len(preds)) * 100.0,
            "character_error_rate": (edit_dist / ref_chars) * 100.0,
            "stop_voicing_accuracy": (sva_corr / max(sva_tot, 1)) * 100.0,
            "count": len(preds),
            "plosive_count": sva_tot
        }
        
    native_preds, native_refs, native_indics = [], [], []
    ne_preds, ne_refs, ne_indics = [], [], []
    
    for p, r, ind, tag in zip(predictions, references, indic_sources, partition_tags):
        if tag in {"Dakshina", "AK-Freq", "native"}:
            native_preds.append(p)
            native_refs.append(r)
            native_indics.append(ind)
        else:
            ne_preds.append(p)
            ne_refs.append(r)
            ne_indics.append(ind)
            
    combined_metrics = _compute_subset(predictions, references, indic_sources)
    native_metrics = _compute_subset(native_preds, native_refs, native_indics)
    ne_metrics = _compute_subset(ne_preds, ne_refs, ne_indics)
    
    return {
        "combined": combined_metrics,
        "native_words": native_metrics,
        "named_entities": ne_metrics
    }

# =====================================================================
# 4. UNIGRAM LANGUAGE MODEL RESCORER
# =====================================================================

class UnigramLMRescorer:
    def __init__(self, target_words: List[str], lm_weight: float = 0.5):
        self.lm_weight = lm_weight
        self.counts = Counter(target_words)
        self.total_tokens = max(len(target_words), 1)
        self.vocab_size = len(self.counts)
        
    def rescore(self, beam_hypotheses: List[Tuple[str, float]]) -> str:
        if not beam_hypotheses:
            return ""
        if self.lm_weight == 0.0 or not self.counts:
            return beam_hypotheses[0][0]
            
        best_candidate = beam_hypotheses[0][0]
        best_score = float("-inf")
        
        for cand, model_log_prob in beam_hypotheses:
            cand_clean = cand.strip()
            count = self.counts.get(cand_clean, 0)
            lm_prob = (count + 1.0) / (self.total_tokens + self.vocab_size)
            lm_log_prob = math.log(lm_prob)
            
            combined_score = model_log_prob + (self.lm_weight * lm_log_prob)
            if combined_score > best_score:
                best_score = combined_score
                best_candidate = cand
                
        return best_candidate

# =====================================================================
# 5. DATASET & VOCABULARY
# =====================================================================

def extract_from_dict(item: dict) -> Tuple[Optional[str], Optional[str], str]:
    if not isinstance(item, dict):
        return None, None, "native"
    clean_item = {str(k).lower().strip(): v for k, v in item.items() if v is not None}
    native = clean_item.get("native word") or clean_item.get("native_word") or clean_item.get("native") or clean_item.get("indic")
    roman = clean_item.get("english word") or clean_item.get("english_word") or clean_item.get("english") or clean_item.get("roman") or clean_item.get("target")
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

def get_official_dataset_splits(lang: str, max_train_samples: Optional[int] = None, 
                                max_val_samples: Optional[int] = None, 
                                max_test_samples: Optional[int] = None) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str, str]], List[Tuple[str, str, str]]]:
    extracted_dir = os.path.join(DATA_DIR, f"extracted_{lang}")
    if not os.path.exists(extracted_dir):
        zip_candidates = [
            os.path.join(WORKSPACE_DIR, "data", f"{lang}.zip"),
            os.path.join(DATA_DIR, f"{lang}.zip")
        ]
        found_zip = None
        for zc in zip_candidates:
            if os.path.exists(zc):
                found_zip = zc
                break
        if found_zip:
            print(f" -> Extracting '{found_zip}' to '{extracted_dir}'...", flush=True)
            shutil.unpack_archive(found_zip, extracted_dir)
        else:
            print(f" ❌ Dataset archive for {lang} not found.", flush=True)
            sys.exit(1)
            
    train_file = os.path.join(extracted_dir, f"{lang}_train.json")
    valid_file = os.path.join(extracted_dir, f"{lang}_valid.json")
    test_file = os.path.join(extracted_dir, f"{lang}_test.json")
    
    print(f" -> Loading {lang.upper()} Official Splits:", flush=True)
    train_pairs = load_split_file(train_file, max_samples=max_train_samples)
    valid_pairs = load_split_file(valid_file, max_samples=max_val_samples)
    test_pairs = load_split_file(test_file, max_samples=max_test_samples)
    print(f"    Train pairs: {len(train_pairs):,} | Valid pairs: {len(valid_pairs):,} | Test pairs: {len(test_pairs):,}", flush=True)
    return train_pairs, valid_pairs, test_pairs

class CharVocabulary:
    PAD_TOKEN = "<PAD>"
    SOS_TOKEN = "<SOS>"
    EOS_TOKEN = "<EOS>"
    UNK_TOKEN = "<UNK>"
    
    def __init__(self):
        self.char2idx = {self.PAD_TOKEN: 0, self.SOS_TOKEN: 1, self.EOS_TOKEN: 2, self.UNK_TOKEN: 3}
        self.idx2char = {0: self.PAD_TOKEN, 1: self.SOS_TOKEN, 2: self.EOS_TOKEN, 3: self.UNK_TOKEN}
        self.num_chars = 4
        
    def add_token(self, token: str):
        if token not in self.char2idx:
            self.char2idx[token] = self.num_chars
            self.idx2char[self.num_chars] = token
            self.num_chars += 1
            
    def encode(self, text: str, is_source: bool = False) -> List[int]:
        tokens = []
        tokens.append(self.char2idx[self.SOS_TOKEN])
        
        # Handle language prefix tokens like __ta__ and __ml__
        i = 0
        while i < len(text):
            if text[i:i+6] in {"__ta__", "__ml__"}:
                tokens.append(self.char2idx.get(text[i:i+6], self.char2idx[self.UNK_TOKEN]))
                i += 6
            else:
                tokens.append(self.char2idx.get(text[i], self.char2idx[self.UNK_TOKEN]))
                i += 1
                
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

class BilingualTransliterationDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str, str]], 
                 vocab_src: CharVocabulary, vocab_tgt: CharVocabulary):
        """
        samples: List of (src_with_tag, tgt_str, lang, source_tag)
        """
        self.data = []
        for src_str, tgt_str, lang, source_tag in samples:
            phono_labels = generate_char_level_phonology_labels(tgt_str, lang=lang)
            tgt_encoded = vocab_tgt.encode(tgt_str)
            phono_encoded = [0] + phono_labels + [0]
            
            self.data.append((
                vocab_src.encode(src_str, is_source=True),
                tgt_encoded,
                phono_encoded,
                tgt_str,
                src_str,
                lang,
                source_tag
            ))
            
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx]

def bilingual_pad_collate_fn(batch):
    src_list, tgt_list, phono_list, raw_indic_list, raw_roman_list, lang_list, source_tag_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)
    
    src_tensor = torch.zeros(len(src_list), max_src, dtype=torch.long)
    tgt_tensor = torch.zeros(len(tgt_list), max_tgt, dtype=torch.long)
    phono_tensor = torch.zeros(len(phono_list), max_tgt, dtype=torch.long)
    
    for i, (s, t, p) in enumerate(zip(src_list, tgt_list, phono_list)):
        src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        phono_tensor[i, :len(p)] = torch.tensor(p, dtype=torch.long)
        
    return src_tensor, tgt_tensor, phono_tensor, raw_indic_list, raw_roman_list, lang_list, source_tag_list

# =====================================================================
# 6. MODEL ARCHITECTURE
# =====================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 128):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class MultiTaskTransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, num_phono_classes: int = 6,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        self.aux_phono_proj = nn.Linear(d_model, num_phono_classes)
        
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
        
        char_logits = self.out_proj(out)
        phono_logits = self.aux_phono_proj(out)
        
        batch_size, seq_len, vocab_size = char_logits.size()
        padded_char = torch.zeros(batch_size, seq_len + 1, vocab_size, device=device)
        padded_char[:, 1:] = char_logits
        
        _, _, phono_size = phono_logits.size()
        padded_phono = torch.zeros(batch_size, seq_len + 1, phono_size, device=device)
        padded_phono[:, 1:] = phono_logits
        
        return padded_char, padded_phono

    def greedy_decode(self, src: torch.Tensor, max_len: int = 40, sos_idx: int = 1, eos_idx: int = 2) -> List[List[int]]:
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        with torch.no_grad():
            src_mask = (src == 0)
            src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
            memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
            
            tgt_indices = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=device)
            decoded_batch = [[] for _ in range(batch_size)]
            finished = [False] * batch_size
            
            for _ in range(max_len):
                tgt_emb = self.pos_encoder(self.tgt_embed(tgt_indices) * math.sqrt(self.d_model))
                causal_mask = self.generate_square_subsequent_mask(tgt_indices.size(1), device)
                
                out = self.decoder(
                    tgt_emb, memory,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=src_mask
                )
                
                next_token_logits = self.out_proj(out[:, -1])
                next_tokens = next_token_logits.argmax(dim=-1)
                
                for b in range(batch_size):
                    if not finished[b]:
                        tok = next_tokens[b].item()
                        if tok == eos_idx:
                            finished[b] = True
                        else:
                            decoded_batch[b].append(tok)
                            
                if all(finished):
                    break
                    
                tgt_indices = torch.cat((tgt_indices, next_tokens.unsqueeze(1)), dim=1)
                
        return decoded_batch

# =====================================================================
# 7. BILINGUAL TRAINING & EVALUATION
# =====================================================================

def evaluate_bilingual_on_split(model: MultiTaskTransformerSeq2Seq, data_loader: DataLoader, 
                                vocab_tgt: CharVocabulary, lang: str,
                                rescorer: Optional[UnigramLMRescorer] = None) -> Tuple[float, Dict[str, Any]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = next(model.parameters()).device
    
    total_loss = 0.0
    predictions = []
    references = []
    indic_sources = []
    partition_tags = []
    
    with torch.no_grad():
        for src_tensor, tgt_tensor, _, raw_indic, raw_roman, _, source_tags in data_loader:
            src_tensor = src_tensor.to(device)
            tgt_tensor = tgt_tensor.to(device)
            
            char_out, _ = model(src_tensor, tgt_tensor)
            loss = criterion(char_out[:, 1:].reshape(-1, vocab_tgt.num_chars), tgt_tensor[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            decoded_indices = model.greedy_decode(src_tensor)
            for b in range(len(decoded_indices)):
                pred_str = vocab_tgt.decode(decoded_indices[b])
                target_str = raw_indic[b]
                predictions.append(pred_str)
                references.append(target_str)
                indic_sources.append(raw_indic[b])
                partition_tags.append(source_tags[b])
                
    avg_loss = total_loss / max(len(data_loader), 1)
    partitioned_metrics = calculate_partitioned_metrics(predictions, references, indic_sources, partition_tags, lang)
    return avg_loss, partitioned_metrics

def run_bilingual_experiment(arm: str = "Bilingual-A1-MT", epochs: int = 8, 
                             batch_size: int = 256, lr: float = 5e-4, aux_lambda: float = 0.3,
                             max_samples_per_lang: int = 500000) -> Dict[str, Any]:
    check_disk_usage()
    print("\n" + "=" * 80, flush=True)
    print(f" PROJECT VALIMELI — JOINT BILINGUAL DRAVIDIAN BENCHMARK (TAM + MAL)", flush=True)
    print(f" Arm: {arm} | Epochs: {epochs} | Pairs/Lang: {max_samples_per_lang:,} (Total: {max_samples_per_lang*2:,})", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f" -> Compute Device: {device}", flush=True)
    
    run_dir = os.path.join(RUNS_DIR, f"bilingual_{arm}")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "checkpoint_best.pt")
    
    # 1. Load Tamil and Malayalam data
    tam_train, tam_val, tam_test = get_official_dataset_splits("tam", max_train_samples=max_samples_per_lang)
    mal_train, mal_val, mal_test = get_official_dataset_splits("mal", max_train_samples=max_samples_per_lang)
    
    # 2. Build Unified Vocabularies with language prefix tags
    vocab_src = CharVocabulary()
    vocab_tgt = CharVocabulary()
    
    vocab_src.add_token("__ta__")
    vocab_src.add_token("__ml__")
    
    tam_target_words = []
    mal_target_words = []
    
    joint_train_samples = []
    for n, r, s in tam_train:
        src_tag = f"__ta__{r}"
        joint_train_samples.append((src_tag, n, "tam", s))
        tam_target_words.append(n)
        for char in r:
            vocab_src.add_token(char)
        for char in n:
            vocab_tgt.add_token(char)
            
    for n, r, s in mal_train:
        src_tag = f"__ml__{r}"
        joint_train_samples.append((src_tag, n, "mal", s))
        mal_target_words.append(n)
        for char in r:
            vocab_src.add_token(char)
        for char in n:
            vocab_tgt.add_token(char)
            
    random.shuffle(joint_train_samples)
    print(f"\n -> Unified Vocabulary: Source Tokens = {vocab_src.num_chars} | Target Tokens = {vocab_tgt.num_chars}", flush=True)
    print(f" -> Joint Training Pairs: {len(joint_train_samples):,} (50% Tamil, 50% Malayalam)", flush=True)
    
    tam_rescorer = UnigramLMRescorer(tam_target_words, lm_weight=0.5)
    mal_rescorer = UnigramLMRescorer(mal_target_words, lm_weight=0.5)
    
    joint_val_samples = [(f"__ta__{r}", n, "tam", s) for n, r, s in tam_val] + [(f"__ml__{r}", n, "mal", s) for n, r, s in mal_val]
    tam_test_samples = [(f"__ta__{r}", n, "tam", s) for n, r, s in tam_test]
    mal_test_samples = [(f"__ml__{r}", n, "mal", s) for n, r, s in mal_test]
    
    train_ds = BilingualTransliterationDataset(joint_train_samples, vocab_src, vocab_tgt)
    val_ds = BilingualTransliterationDataset(joint_val_samples, vocab_src, vocab_tgt)
    tam_test_ds = BilingualTransliterationDataset(tam_test_samples, vocab_src, vocab_tgt)
    mal_test_ds = BilingualTransliterationDataset(mal_test_samples, vocab_src, vocab_tgt)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=bilingual_pad_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=bilingual_pad_collate_fn)
    tam_test_loader = DataLoader(tam_test_ds, batch_size=batch_size, shuffle=False, collate_fn=bilingual_pad_collate_fn)
    mal_test_loader = DataLoader(mal_test_ds, batch_size=batch_size, shuffle=False, collate_fn=bilingual_pad_collate_fn)
    
    model = MultiTaskTransformerSeq2Seq(
        vocab_src.num_chars, vocab_tgt.num_chars, num_phono_classes=6,
        d_model=256, nhead=4, num_layers=6, dim_feedforward=1024
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Trainable Parameters: {num_params:,} ({num_params / 1e6:.2f}M)", flush=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    char_criterion = nn.CrossEntropyLoss(ignore_index=0)
    phono_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_loss = float("inf")
    
    print(f"\n -> Initiating Joint Bilingual Training ({len(train_loader)} batches/epoch)...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_char_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (src_batch, tgt_batch, phono_batch, _, _, _, _) in enumerate(train_loader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            phono_batch = phono_batch.to(device)
            
            optimizer.zero_grad()
            char_logits, phono_logits = model(src_batch, tgt_batch)
            
            loss_char = char_criterion(char_logits[:, 1:].reshape(-1, vocab_tgt.num_chars), tgt_batch[:, 1:].reshape(-1))
            loss_phono = phono_criterion(phono_logits[:, 1:].reshape(-1, 6), phono_batch[:, 1:].reshape(-1))
            
            total_loss = loss_char + (aux_lambda * loss_phono) if "A1" in arm else loss_char
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_char_loss += loss_char.item()
            
            if (batch_idx + 1) % 500 == 0 or (batch_idx + 1) == len(train_loader):
                pct = ((batch_idx + 1) / len(train_loader)) * 100
                print(f"  [Epoch {epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] ({pct:.1f}%) | CharLoss: {loss_char.item():.4f}", flush=True)
                
        avg_train_loss = total_char_loss / max(len(train_loader), 1)
        val_loss, _ = evaluate_bilingual_on_split(model, val_loader, vocab_tgt, "tam")
        epoch_time = time.time() - start_time
        
        print(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ({epoch_time:.1f}s) ===", flush=True)
        print(f"    Train CharLoss: {avg_train_loss:.4f} | Joint Val Loss: {val_loss:.4f}", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "vocab_src": vocab_src,
                "vocab_tgt": vocab_tgt,
                "best_val_loss": best_val_loss
            }, checkpoint_path)
            print(f"    ⭐ New best bilingual checkpoint saved with Val Loss: {best_val_loss:.4f}", flush=True)
            
    print(f"\n -> Evaluating Best Checkpoint on TAMIL Holdout Test Set...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    _, tam_metrics = evaluate_bilingual_on_split(model, tam_test_loader, vocab_tgt, "tam", rescorer=tam_rescorer)
    _, mal_metrics = evaluate_bilingual_on_split(model, mal_test_loader, vocab_tgt, "mal", rescorer=mal_rescorer)
    
    print("\n" + "*" * 80, flush=True)
    print(f" BILINGUAL MODEL RESULTS ({arm}):", flush=True)
    print(f"   ► TAMIL NATIVE WORDS:      {tam_metrics['native_words']['exact_match_accuracy']:.2f}% EM | CER: {tam_metrics['native_words']['character_error_rate']:.2f}% | SVA: {tam_metrics['native_words']['stop_voicing_accuracy']:.2f}%", flush=True)
    print(f"   ► TAMIL NAMED ENTITIES:    {tam_metrics['named_entities']['exact_match_accuracy']:.2f}% EM | CER: {tam_metrics['named_entities']['character_error_rate']:.2f}%", flush=True)
    print(f"   ► TAMIL OVERALL COMBINED:  {tam_metrics['combined']['exact_match_accuracy']:.2f}% EM | CER: {tam_metrics['combined']['character_error_rate']:.2f}%", flush=True)
    print("-" * 80, flush=True)
    print(f"   ► MALAYALAM NATIVE WORDS:  {mal_metrics['native_words']['exact_match_accuracy']:.2f}% EM | CER: {mal_metrics['native_words']['character_error_rate']:.2f}% | SVA: {mal_metrics['native_words']['stop_voicing_accuracy']:.2f}%", flush=True)
    print(f"   ► MALAYALAM NAMED ENTITIES:{mal_metrics['named_entities']['exact_match_accuracy']:.2f}% EM | CER: {mal_metrics['named_entities']['character_error_rate']:.2f}%", flush=True)
    print(f"   ► MALAYALAM OVERALL:       {mal_metrics['combined']['exact_match_accuracy']:.2f}% EM | CER: {mal_metrics['combined']['character_error_rate']:.2f}%", flush=True)
    print("*" * 80 + "\n", flush=True)
    
    report = {
        "experiment_metadata": {
            "model_type": "bilingual_transformer",
            "arm": arm,
            "samples_per_lang": max_samples_per_lang,
            "total_train_samples": len(joint_train_samples),
            "device": str(device)
        },
        "metrics": {
            "joint_val_loss": best_val_loss,
            "tamil": tam_metrics,
            "malayalam": mal_metrics
        }
    }
    
    artifact_report_path = os.path.join(ARTIFACTS_DIR, f"{arm}_results.json")
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project ValiMeli Joint Bilingual & Monolingual Benchmarks")
    parser.add_argument("--mode", type=str, default="bilingual", choices=["bilingual", "monolingual"])
    parser.add_argument("--arm", type=str, default="Bilingual-A1-MT", choices=["Bilingual-A0", "Bilingual-A1-MT"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--aux_lambda", type=float, default=0.3)
    parser.add_argument("--max_samples_per_lang", type=int, default=500000)
    args = parser.parse_args()
    
    if args.mode == "bilingual":
        run_bilingual_experiment(
            arm=args.arm,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            aux_lambda=args.aux_lambda,
            max_samples_per_lang=args.max_samples_per_lang
        )
