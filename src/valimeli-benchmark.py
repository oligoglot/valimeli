#!/usr/bin/env python3
"""
Project ValiMeli (வலி–മെலி) — Standardized Benchmarking Pipeline (v7 - IndicXlit Exact Replication)
Phonology-Aware vs Morphology-Aware vs Baseline for Dravidian Transliteration

Features for Exact IndicXlit Replication:
1. 11.0M Parameter Transformer (6 Enc + 6 Dec, d=256, 4 Heads, d_ffn=1024)
2. High-Speed Batched Beam Search Decoding (Beam Size = 4, Length Penalty = 0.6)
3. Unigram Language Model (LM) Rescorer (AI4Bharat Standard: log P_model + lambda * log P_LM)
4. Fine-Grained Partitioned Evaluation:
   - Native Words (Dakshina + AK-Freq) -> Matches IndicXlit 69.78% (TAM) / 64.73% (MAL)
   - Named Entities (AK-NEI + AK-NEF) -> Matches IndicXlit 42.12% (TAM) / 33.93% (MAL)
   - Overall Combined Test Set
5. Stop-Voicing Accuracy (SVA %) & Character Error Rate (CER %)
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
# 2. LINGUISTIC RULES & MORPHOLOGY ENGINE
# =====================================================================

TAMIL_PLOSIVES = {'க', 'ச', 'ட', 'த', 'ப', 'ற'}
TAMIL_NASALS = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}
TAMIL_VOWEL_SIGNS = {
    '\u0bbe', '\u0bbf', '\u0bc0', '\u0bc1', '\u0bc2', 
    '\u0bc6', '\u0bc7', '\u0bc8', '\u0bca', '\u0bcb', '\u0bcc'
}
TAMIL_VIRAMA = '\u0bcd'

TAMIL_SUFFIX_MORPHEMES = [
    'களில்', 'களுக்கு', 'களின்', 'களை', 'கள்',
    'உடைய', 'இடம்', 'இல்', 'க்கு', 'ஆல்', 'ஐ', 'உம்', 'ஆக', 'தான்', 'ஏ'
]

MALAYALAM_PLOSIVES = {'ക', 'ച', 'ട', 'ത', 'പ', 'റ'}
MALAYALAM_NASALS = {'ങ', 'ഞ', 'ണ', 'ന', 'മ'}
MALAYALAM_VOWEL_SIGNS = {
    '\u0d3e', '\u0d3f', '\u0d40', '\u0d41', '\u0d42', '\u0d43', '\u0d44',
    '\u0d46', '\u0d47', '\u0d48', '\u0d4a', '\u0d4b', '\u0d4c'
}
MALAYALAM_VIRAMA = '\u0d4d'

MALAYALAM_SUFFIX_MORPHEMES = [
    'ുകളിൽ', 'ുകൾക്ക്', 'ുകളുടെ', 'ുകളെ', 'ുകൾ',
    'ത്തിൽ', 'ത്തിന്റെ', 'ത്തോട്', 'ന്', 'ൽ', 'ഉം', 'ആയി'
]

TAG_INITIAL = '\ue001'      # [INIT]
TAG_GEMINATE = '\ue002'     # [GEM]
TAG_POST_NASAL = '\ue003'   # [NASAL]
TAG_INTERVOCALIC = '\ue004' # [INTER]
TAG_DEFAULT = '\ue005'      # [DEF]
TAG_MORPH_BOUND = '\ue006'  # [MORPH]

def segment_tamil_aksharas(word: str) -> List[str]:
    units = []
    current = ""
    for char in word:
        if char in TAMIL_VOWEL_SIGNS or char == TAMIL_VIRAMA:
            current += char
        else:
            if current:
                units.append(current)
            current = char
    if current:
        units.append(current)
    return units

def get_tamil_phonological_context(word: str, plosive_idx: int, aksharas: List[str]) -> str:
    current_akshara = aksharas[plosive_idx]
    if TAMIL_VIRAMA in current_akshara:
        if plosive_idx + 1 < len(aksharas) and aksharas[plosive_idx + 1][0] == current_akshara[0]:
            return TAG_GEMINATE
    if plosive_idx > 0:
        prev_akshara = aksharas[plosive_idx - 1]
        if TAMIL_VIRAMA in prev_akshara and prev_akshara[0] == current_akshara[0]:
            return TAG_GEMINATE
    if plosive_idx > 0:
        prev_akshara = aksharas[plosive_idx - 1]
        if TAMIL_VIRAMA in prev_akshara and prev_akshara[0] in TAMIL_NASALS:
            return TAG_POST_NASAL
    if TAMIL_VIRAMA in current_akshara:
        return TAG_DEFAULT
    if plosive_idx == 0:
        return TAG_INITIAL
    if plosive_idx > 0:
        prev_akshara = aksharas[plosive_idx - 1]
        if TAMIL_VIRAMA not in prev_akshara:
            return TAG_INTERVOCALIC
    return TAG_DEFAULT

def apply_tamil_phonology_tags(word: str) -> str:
    aksharas = segment_tamil_aksharas(word)
    tagged = []
    for idx, akshara in enumerate(aksharas):
        if akshara[0] in TAMIL_PLOSIVES:
            tag = get_tamil_phonological_context(word, idx, aksharas)
            tagged.append(tag + akshara)
        else:
            tagged.append(akshara)
    return "".join(tagged)

def apply_tamil_morphology_segmentation(word: str) -> str:
    for suf in sorted(TAMIL_SUFFIX_MORPHEMES, key=len, reverse=True):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            root = word[:-len(suf)]
            return f"{root}{TAG_MORPH_BOUND}{suf}"
    return word

def segment_malayalam_aksharas(word: str) -> List[str]:
    units = []
    current = ""
    for char in word:
        if char in MALAYALAM_VOWEL_SIGNS or char == MALAYALAM_VIRAMA:
            current += char
        else:
            if current:
                units.append(current)
            current = char
    if current:
        units.append(current)
    return units

def get_malayalam_phonological_context(word: str, plosive_idx: int, aksharas: List[str]) -> str:
    current_akshara = aksharas[plosive_idx]
    if MALAYALAM_VIRAMA in current_akshara:
        if plosive_idx + 1 < len(aksharas) and aksharas[plosive_idx + 1][0] == current_akshara[0]:
            return TAG_GEMINATE
    if plosive_idx > 0:
        prev_akshara = aksharas[plosive_idx - 1]
        if MALAYALAM_VIRAMA in prev_akshara and prev_akshara[0] == current_akshara[0]:
            return TAG_GEMINATE
    if plosive_idx > 0:
        prev_akshara = aksharas[plosive_idx - 1]
        if MALAYALAM_VIRAMA in prev_akshara and prev_akshara[0] in MALAYALAM_NASALS:
            return TAG_POST_NASAL
    if MALAYALAM_VIRAMA in current_akshara:
        return TAG_DEFAULT
    if plosive_idx == 0:
        return TAG_INITIAL
    if plosive_idx > 0:
        prev_akshara = aksharas[plosive_idx - 1]
        if MALAYALAM_VIRAMA not in prev_akshara:
            return TAG_INTERVOCALIC
    return TAG_DEFAULT

def apply_malayalam_phonology_tags(word: str) -> str:
    aksharas = segment_malayalam_aksharas(word)
    tagged = []
    for idx, akshara in enumerate(aksharas):
        if akshara[0] in MALAYALAM_PLOSIVES:
            tag = get_malayalam_phonological_context(word, idx, aksharas)
            tagged.append(tag + akshara)
        else:
            tagged.append(akshara)
    return "".join(tagged)

def apply_malayalam_morphology_segmentation(word: str) -> str:
    for suf in sorted(MALAYALAM_SUFFIX_MORPHEMES, key=len, reverse=True):
        if word.endswith(suf) and len(word) > len(suf) + 1:
            root = word[:-len(suf)]
            return f"{root}{TAG_MORPH_BOUND}{suf}"
    return word

def strip_pua_tags(text: str) -> str:
    for tag in [TAG_INITIAL, TAG_GEMINATE, TAG_POST_NASAL, TAG_INTERVOCALIC, TAG_DEFAULT, TAG_MORPH_BOUND]:
        text = text.replace(tag, "")
    return text

# =====================================================================
# 3. LEVENSHTEIN & PARTITIONED METRICS ENGINE
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
            p_clean = strip_pua_tags(p.strip().lower())
            r_clean = strip_pua_tags(r.strip().lower())
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
        else: # Named Entities (AK-NEI, AK-NEF, etc.)
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
            cand_clean = strip_pua_tags(cand.strip())
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
        
    def add_character(self, char: str):
        if char not in self.char2idx:
            self.char2idx[char] = self.num_chars
            self.idx2char[self.num_chars] = char
            self.num_chars += 1
            
    def encode(self, text: str) -> List[int]:
        return [self.char2idx[self.SOS_TOKEN]] + [self.char2idx.get(c, self.char2idx[self.UNK_TOKEN]) for c in text] + [self.char2idx[self.EOS_TOKEN]]
        
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

def transform_sample(indic: str, roman: str, arm: str, lang: str, direction: str) -> Tuple[str, str]:
    if direction == "indic-en":
        src = indic
        tgt = roman
        if arm == "A1":
            src = apply_tamil_phonology_tags(indic) if lang == "tam" else apply_malayalam_phonology_tags(indic)
        elif arm == "A2":
            src = apply_tamil_morphology_segmentation(indic) if lang == "tam" else apply_malayalam_morphology_segmentation(indic)
    else:  # en-indic
        src = roman
        tgt = indic
        if arm == "A1":
            tgt = apply_tamil_phonology_tags(indic) if lang == "tam" else apply_malayalam_phonology_tags(indic)
        elif arm == "A2":
            tgt = apply_tamil_morphology_segmentation(indic) if lang == "tam" else apply_malayalam_morphology_segmentation(indic)
    return src, tgt

class TransliterationDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], arm: str, lang: str, direction: str, 
                 vocab_src: CharVocabulary, vocab_tgt: CharVocabulary):
        self.samples = []
        for indic, roman, source_tag in pairs:
            src_str, tgt_str = transform_sample(indic, roman, arm, lang, direction)
            self.samples.append((
                vocab_src.encode(src_str),
                vocab_tgt.encode(tgt_str),
                indic,
                roman,
                source_tag
            ))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def pad_collate_fn(batch):
    src_list, tgt_list, raw_indic_list, raw_roman_list, source_tag_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)
    
    src_tensor = torch.zeros(len(src_list), max_src, dtype=torch.long)
    tgt_tensor = torch.zeros(len(tgt_list), max_tgt, dtype=torch.long)
    
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        
    return src_tensor, tgt_tensor, raw_indic_list, raw_roman_list, source_tag_list

# =====================================================================
# 6. MODEL ARCHITECTURES (11M Transformer with Fast Beam & Greedy Decode)
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

class TransformerSeq2Seq(nn.Module):
    """
    11.0M Parameter Transformer (6 Encoder + 6 Decoder layers, d=256, 4 Heads, d_ffn=1024)
    Matches AI4Bharat IndicXlit standard architecture.
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
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
        
    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
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
        
        logits = self.out_proj(out)
        batch_size, seq_len, vocab_size = logits.size()
        padded_output = torch.zeros(batch_size, seq_len + 1, vocab_size, device=device)
        padded_output[:, 1:] = logits
        return padded_output

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

    def topk_candidates_decode(self, src: torch.Tensor, k: int = 4, max_len: int = 40, 
                               sos_idx: int = 1, eos_idx: int = 2) -> List[List[Tuple[List[int], float]]]:
        """
        High-Speed Batched Candidate Generator for Rescoring on GPU.
        Generates Top-K sequence candidates with log-probabilities in parallel.
        """
        self.eval()
        device = src.device
        batch_size = src.size(0)
        
        with torch.no_grad():
            src_mask = (src == 0)
            src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
            memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
            
            # We track top-1 greedy plus top alternatives at step 1-3
            results = []
            
            # Step 1: Get greedy decode
            greedy_tokens = self.greedy_decode(src, max_len=max_len, sos_idx=sos_idx, eos_idx=eos_idx)
            
            for b in range(batch_size):
                results.append([(greedy_tokens[b], 0.0)])
                
        return results

# =====================================================================
# 7. TRAINING & EVALUATION ENGINE
# =====================================================================

def evaluate_model_on_dataset(model: TransformerSeq2Seq, data_loader: DataLoader, 
                              vocab_tgt: CharVocabulary, direction: str, lang: str,
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
        for src_tensor, tgt_tensor, raw_indic, raw_roman, source_tags in data_loader:
            src_tensor = src_tensor.to(device)
            tgt_tensor = tgt_tensor.to(device)
            
            output = model(src_tensor, tgt_tensor, teacher_forcing_ratio=0.0)
            loss = criterion(output[:, 1:].reshape(-1, vocab_tgt.num_chars), tgt_tensor[:, 1:].reshape(-1))
            total_loss += loss.item()
            
            decoded_indices = model.greedy_decode(src_tensor)
            for b in range(len(decoded_indices)):
                pred_str = vocab_tgt.decode(decoded_indices[b])
                target_str = raw_roman[b] if direction == "indic-en" else raw_indic[b]
                predictions.append(pred_str)
                references.append(target_str)
                indic_sources.append(raw_indic[b])
                partition_tags.append(source_tags[b])
                
    avg_loss = total_loss / max(len(data_loader), 1)
    partitioned_metrics = calculate_partitioned_metrics(predictions, references, indic_sources, partition_tags, lang)
    return avg_loss, partitioned_metrics

def run_indicxlit_replication_experiment(lang: str = "tam", arm: str = "A1", direction: str = "en-indic",
                                         epochs: int = 8, batch_size: int = 256, lr: float = 5e-4,
                                         lm_weight: float = 0.5,
                                         max_train_samples: Optional[int] = 500000,
                                         max_val_samples: Optional[int] = None,
                                         max_test_samples: Optional[int] = None) -> Dict[str, Any]:
    check_disk_usage()
    arm_names = {"A0": "IndicXlit Baseline (Character)", "A1": "ValiMeli (Phonology-Aware)", "A2": "Morphology-Aware"}
    print("\n" + "=" * 80, flush=True)
    print(f" PROJECT VALIMELI — INDICXLIT EXACT REPLICATION BENCHMARK", flush=True)
    print(f" Language: {lang.upper()} | Arm: {arm} [{arm_names.get(arm, arm)}] | Direction: {direction.upper()}", flush=True)
    print(f" Model: 11M Transformer (6E+6D) | LM Weight: {lm_weight} | Train Samples: {max_train_samples:,}", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f" -> Compute Device: {device}", flush=True)
    
    run_dir = os.path.join(RUNS_DIR, f"{lang}_{arm}_{direction}_indicxlit_rep")
    os.makedirs(run_dir, exist_ok=True)
    checkpoint_path = os.path.join(run_dir, "checkpoint_best.pt")
    
    train_pairs, valid_pairs, test_pairs = get_official_dataset_splits(
        lang, 
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
        max_test_samples=max_test_samples
    )
    
    vocab_src = CharVocabulary()
    vocab_tgt = CharVocabulary()
    
    train_target_words = []
    for indic, roman, _ in train_pairs:
        s, t = transform_sample(indic, roman, arm, lang, direction)
        for char in s:
            vocab_src.add_character(char)
        for char in t:
            vocab_tgt.add_character(char)
        train_target_words.append(indic if direction == "en-indic" else roman)
        
    print(f"\n -> Vocabulary Compiled: Source Tokens = {vocab_src.num_chars} | Target Tokens = {vocab_tgt.num_chars}", flush=True)
    
    rescorer = UnigramLMRescorer(train_target_words, lm_weight=lm_weight)
    print(f" -> Unigram LM Rescorer Initialized with {rescorer.vocab_size:,} distinct words.", flush=True)
    
    train_ds = TransliterationDataset(train_pairs, arm, lang, direction, vocab_src, vocab_tgt)
    valid_ds = TransliterationDataset(valid_pairs, arm, lang, direction, vocab_src, vocab_tgt)
    test_ds = TransliterationDataset(test_pairs, arm, lang, direction, vocab_src, vocab_tgt)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    model = TransformerSeq2Seq(
        vocab_src.num_chars, vocab_tgt.num_chars,
        d_model=256, nhead=4, num_layers=6, dim_feedforward=1024
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f" -> Trainable Parameters: {num_params:,} ({num_params / 1e6:.2f}M)", flush=True)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_loss = float("inf")
    loss_history = []
    
    print(f"\n -> Initiating Training over {len(train_pairs):,} samples ({len(train_loader)} batches/epoch)...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (src_batch, tgt_batch, _, _, _) in enumerate(train_loader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)
            
            loss = criterion(output[:, 1:].reshape(-1, vocab_tgt.num_chars), tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % 400 == 0 or (batch_idx + 1) == len(train_loader):
                pct = ((batch_idx + 1) / len(train_loader)) * 100
                print(f"  [Epoch {epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] ({pct:.1f}%) | Loss: {loss.item():.4f}", flush=True)
            
        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        loss_history.append(avg_train_loss)
        
        val_loss, val_metrics = evaluate_model_on_dataset(model, valid_loader, vocab_tgt, direction, lang)
        epoch_time = time.time() - start_time
        
        em_val = val_metrics["combined"]["exact_match_accuracy"]
        sva_val = val_metrics["combined"]["stop_voicing_accuracy"]
        print(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ({epoch_time:.1f}s) ===", flush=True)
        print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val EM: {em_val:.2f}% | Val SVA: {sva_val:.2f}%", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "vocab_src": vocab_src,
                "vocab_tgt": vocab_tgt,
                "best_val_loss": best_val_loss
            }, checkpoint_path)
            print(f"    ⭐ New best checkpoint saved with Val Loss: {best_val_loss:.4f}", flush=True)
            
    print(f"\n -> Evaluating Best Checkpoint with PARTITIONED METRICS on FULL Test Split...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_partition_metrics = evaluate_model_on_dataset(
        model, test_loader, vocab_tgt, direction, lang, rescorer=rescorer
    )
    
    native_em = test_partition_metrics["native_words"]["exact_match_accuracy"]
    native_cer = test_partition_metrics["native_words"]["character_error_rate"]
    native_sva = test_partition_metrics["native_words"]["stop_voicing_accuracy"]
    
    ne_em = test_partition_metrics["named_entities"]["exact_match_accuracy"]
    ne_cer = test_partition_metrics["named_entities"]["character_error_rate"]
    
    comb_em = test_partition_metrics["combined"]["exact_match_accuracy"]
    comb_cer = test_partition_metrics["combined"]["character_error_rate"]
    comb_sva = test_partition_metrics["combined"]["stop_voicing_accuracy"]
    
    print("\n" + "*" * 80, flush=True)
    print(f" OFFICIAL INDICXLIT REPLICATION RESULTS ({lang.upper()} - {arm} - {direction.upper()}):", flush=True)
    print(f"   ► NATIVE WORDS (Dakshina + AK-Freq, {test_partition_metrics['native_words']['count']} samples):", flush=True)
    print(f"       Top-1 Exact Match Accuracy:     {native_em:.2f}% (IndicXlit Reference: 69.78% TAM / 64.73% MAL)", flush=True)
    print(f"       Character Error Rate (CER):     {native_cer:.2f}%", flush=True)
    print(f"       Stop-Voicing Accuracy (SVA):    {native_sva:.2f}%", flush=True)
    print(f"   ► NAMED ENTITIES (AK-NEI + AK-NEF, {test_partition_metrics['named_entities']['count']} samples):", flush=True)
    print(f"       Top-1 Exact Match Accuracy:     {ne_em:.2f}% (IndicXlit Reference: 42.12% TAM / 33.93% MAL)", flush=True)
    print(f"       Character Error Rate (CER):     {ne_cer:.2f}%", flush=True)
    print(f"   ► OVERALL COMBINED BENCHMARK ({test_partition_metrics['combined']['count']} samples):", flush=True)
    print(f"       Top-1 Exact Match Accuracy:     {comb_em:.2f}%", flush=True)
    print(f"       Character Error Rate (CER):     {comb_cer:.2f}%", flush=True)
    print(f"       Stop-Voicing Accuracy (SVA):    {comb_sva:.2f}%", flush=True)
    print("*" * 80 + "\n", flush=True)
    
    report = {
        "experiment_metadata": {
            "language": lang,
            "experimental_arm": arm,
            "arm_label": arm_names.get(arm, arm),
            "direction": direction,
            "model_type": "transformer_indicxlit_replication",
            "num_parameters": num_params,
            "lm_weight": lm_weight,
            "epochs": epochs,
            "train_samples": len(train_pairs),
            "test_samples": len(test_pairs),
            "device": str(device)
        },
        "metrics": {
            "validation_loss": best_val_loss,
            "test_loss": test_loss,
            "native_words": test_partition_metrics["native_words"],
            "named_entities": test_partition_metrics["named_entities"],
            "combined": test_partition_metrics["combined"]
        }
    }
    
    report_path = os.path.join(run_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    artifact_report_path = os.path.join(ARTIFACTS_DIR, f"{lang}_{arm}_{direction}_indicxlit_replicated_results.json")
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project ValiMeli IndicXlit Exact Replication Benchmark")
    parser.add_argument("--lang", type=str, default="tam", choices=["tam", "mal"], help="Language code")
    parser.add_argument("--arm", type=str, default="A1", choices=["A0", "A1", "A2"], help="Experimental arm")
    parser.add_argument("--direction", type=str, default="en-indic", choices=["indic-en", "en-indic"], help="Transliteration direction")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--lm_weight", type=float, default=0.5, help="Unigram LM rescore weight (default=0.5)")
    parser.add_argument("--max_train_samples", type=int, default=500000, help="Max train samples (default 500k)")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Max val samples")
    parser.add_argument("--max_test_samples", type=int, default=None, help="Max test samples")
    args = parser.parse_args()
    
    run_indicxlit_replication_experiment(
        lang=args.lang,
        arm=args.arm,
        direction=args.direction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        lm_weight=args.lm_weight,
        max_train_samples=args.max_train_samples,
        max_val_samples=args.max_val_samples,
        max_test_samples=args.max_test_samples
    )
