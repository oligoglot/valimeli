#!/usr/bin/env python3
"""
Project ValiMeli (வலி–മെലി) — Standardized Benchmarking Pipeline (v5 - Full 3-Arm Suite)
Phonology-Aware vs Morphology-Aware vs Baseline for Dravidian Transliteration

Experimental Arms:
- Arm A0: Standard Character-Level Baseline (IndicXlit / Aksharantar standard)
- Arm A1: ValiMeli Phonology-Aware Tokenization (Phonotactic context tagging: [INIT], [GEM], [NASAL], [INTER])
- Arm A2: Morphology-Aware Tokenization (Subword Morpheme Segmentation inspired by arXiv:2508.08424)

Linguistic & Evaluation Metrics:
1. Top-1 Word Exact Match (EM %)
2. Levenshtein Character Error Rate (CER %)
3. Stop-Voicing Accuracy (SVA %): Precision/Recall on plosive voicing realization.
4. Full Aksharantar Holdout Test Set Evaluation.
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
from typing import List, Tuple, Dict, Generator, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
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
TAMIL_STANDALONE_VOWELS = {
    'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ'
}

# Common Tamil Inflectional/Derivational Suffix Morphemes
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
MALAYALAM_STANDALONE_VOWELS = {
    'അ', 'ആ', 'ഇ', 'ഈ', 'ഉ', 'ഊ', 'ഋ', 'ൠ', 'എ', 'ഏ', 'ഐ', 'ഒ', 'ഓ', 'ഔ'
}

# Common Malayalam Inflectional Suffix Morphemes
MALAYALAM_SUFFIX_MORPHEMES = [
    'ുകളിൽ', 'ുകൾക്ക്', 'ുകളുടെ', 'ുകളെ', 'ുകൾ',
    'ത്തിൽ', 'ത്തിന്റെ', 'ത്തോട്', 'ന്', 'ൽ', 'ഉം', 'ആയി'
]

TAG_INITIAL = '\ue001'      # [INIT]
TAG_GEMINATE = '\ue002'     # [GEM]
TAG_POST_NASAL = '\ue003'   # [NASAL]
TAG_INTERVOCALIC = '\ue004' # [INTER]
TAG_DEFAULT = '\ue005'      # [DEF]
TAG_MORPH_BOUND = '\ue006'  # [MORPH] Morpheme boundary delimiter for Arm A2

TAG_MAP = {
    TAG_INITIAL: "[INIT]",
    TAG_GEMINATE: "[GEM]",
    TAG_POST_NASAL: "[NASAL]",
    TAG_INTERVOCALIC: "[INTER]",
    TAG_DEFAULT: "[DEF]",
    TAG_MORPH_BOUND: "@@"
}

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
    """
    Arm A2: Segments Tamil words into root + grammatical suffixes (arXiv:2508.08424 benchmark).
    Example: 'பக்கத்தில்' -> 'பக்க' + '@@' + 'த்தில்'
    """
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
# 3. LEVENSHTEIN & STOP-VOICING ACCURACY (SVA)
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

def calculate_fine_grained_metrics(predictions: List[str], references: List[str], indic_sources: List[str], lang: str) -> Dict[str, float]:
    """
    Computes Exact Match, Character Error Rate, and Stop-Voicing Accuracy (SVA %).
    SVA isolates words containing intervocalic/post-nasal/geminate plosives to measure
    whether the model resolved the voicing distinction.
    """
    total_samples = len(predictions)
    if total_samples == 0:
        return {"exact_match_accuracy": 0.0, "character_error_rate": 0.0, "stop_voicing_accuracy": 0.0, "total_samples": 0}
        
    exact_matches = 0
    total_edit_distance = 0
    total_ref_chars = 0
    
    sva_correct = 0
    sva_total = 0
    
    plosives = TAMIL_PLOSIVES if lang == "tam" else MALAYALAM_PLOSIVES
    
    for pred, ref, indic in zip(predictions, references, indic_sources):
        pred_clean = strip_pua_tags(pred.strip().lower())
        ref_clean = strip_pua_tags(ref.strip().lower())
        
        is_exact = (pred_clean == ref_clean)
        if is_exact:
            exact_matches += 1
            
        dist = levenshtein_distance(pred_clean, ref_clean)
        total_edit_distance += dist
        total_ref_chars += max(len(ref_clean), 1)
        
        # Check if word contains a conditional plosive
        has_plosive = any(p in indic for p in plosives)
        if has_plosive:
            sva_total += 1
            if is_exact:
                sva_correct += 1
                
    em_acc = (exact_matches / total_samples) * 100.0
    cer = (total_edit_distance / total_ref_chars) * 100.0
    sva = (sva_correct / max(sva_total, 1)) * 100.0
    
    return {
        "exact_match_accuracy": em_acc,
        "character_error_rate": cer,
        "stop_voicing_accuracy": sva,
        "total_samples": total_samples,
        "plosive_word_samples": sva_total
    }

# =====================================================================
# 4. DATASET & VOCABULARY
# =====================================================================

def extract_from_dict(item: dict) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(item, dict):
        return None, None
    clean_item = {str(k).lower().strip(): v for k, v in item.items() if v is not None}
    native = clean_item.get("native word") or clean_item.get("native_word") or clean_item.get("native") or clean_item.get("indic")
    roman = clean_item.get("english word") or clean_item.get("english_word") or clean_item.get("english") or clean_item.get("roman") or clean_item.get("target")
    if native and roman:
        return str(native).strip(), str(roman).strip()
    return None, None

def load_split_file(file_path: str, max_samples: Optional[int] = None) -> List[Tuple[str, str]]:
    pairs = []
    if not os.path.exists(file_path):
        return pairs
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                n, r = extract_from_dict(obj)
                if n and r:
                    pairs.append((n, r))
                    if max_samples and len(pairs) >= max_samples:
                        break
            except Exception:
                continue
    return pairs

def get_official_dataset_splits(lang: str, max_train_samples: Optional[int] = None, 
                                max_val_samples: Optional[int] = None, 
                                max_test_samples: Optional[int] = None) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]], List[Tuple[str, str]]]:
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
    def __init__(self, pairs: List[Tuple[str, str]], arm: str, lang: str, direction: str, 
                 vocab_src: CharVocabulary, vocab_tgt: CharVocabulary):
        self.samples = []
        for indic, roman in pairs:
            src_str, tgt_str = transform_sample(indic, roman, arm, lang, direction)
            self.samples.append((
                vocab_src.encode(src_str),
                vocab_tgt.encode(tgt_str),
                indic,
                roman
            ))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def pad_collate_fn(batch):
    src_list, tgt_list, raw_indic_list, raw_roman_list = zip(*batch)
    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)
    
    src_tensor = torch.zeros(len(src_list), max_src, dtype=torch.long)
    tgt_tensor = torch.zeros(len(tgt_list), max_tgt, dtype=torch.long)
    
    for i, (s, t) in enumerate(zip(src_list, tgt_list)):
        src_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        tgt_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        
    return src_tensor, tgt_tensor, raw_indic_list, raw_roman_list

# =====================================================================
# 5. MODEL ARCHITECTURE (Seq2Seq with Bahdanau Attention)
# =====================================================================

class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super(Seq2SeqAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.enc_hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.attn = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
        self.decoder = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, tgt_vocab_size)
        
    def forward(self, src: torch.Tensor, tgt: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.size(0)
        max_tgt_len = tgt.size(1)
        tgt_vocab_size = self.out_proj.out_features
        
        src_emb = self.src_embed(src)
        enc_outputs, enc_hidden = self.encoder(src_emb)
        
        combined_h = torch.cat((enc_hidden[0], enc_hidden[1]), dim=-1)
        dec_hidden = torch.tanh(self.enc_hidden_proj(combined_h)).unsqueeze(0)
        
        outputs = torch.zeros(batch_size, max_tgt_len, tgt_vocab_size, device=src.device)
        dec_input = tgt[:, 0].unsqueeze(1)
        
        src_len = enc_outputs.size(1)
        
        for t in range(1, max_tgt_len):
            dec_emb = self.tgt_embed(dec_input)
            dec_h_exp = dec_hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
            attn_energy = torch.tanh(self.attn(torch.cat((dec_h_exp, enc_outputs), dim=-1)))
            attn_scores = self.v(attn_energy).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(1)
            
            context = torch.bmm(attn_weights, enc_outputs)
            dec_out, dec_hidden = self.decoder(torch.cat((dec_emb, context), dim=-1), dec_hidden)
            
            logits = self.out_proj(dec_out.squeeze(1))
            outputs[:, t] = logits
            
            teacher_force = (random.random() < teacher_forcing_ratio) and self.training
            top1 = logits.argmax(dim=-1).unsqueeze(1)
            dec_input = tgt[:, t].unsqueeze(1) if teacher_force else top1
            
        return outputs

    def greedy_decode(self, src: torch.Tensor, max_len: int = 40, sos_idx: int = 1, eos_idx: int = 2) -> List[List[int]]:
        self.eval()
        with torch.no_grad():
            batch_size = src.size(0)
            src_emb = self.src_embed(src)
            enc_outputs, enc_hidden = self.encoder(src_emb)
            
            combined_h = torch.cat((enc_hidden[0], enc_hidden[1]), dim=-1)
            dec_hidden = torch.tanh(self.enc_hidden_proj(combined_h)).unsqueeze(0)
            
            src_len = enc_outputs.size(1)
            dec_input = torch.full((batch_size, 1), sos_idx, dtype=torch.long, device=src.device)
            
            decoded_batch = [[] for _ in range(batch_size)]
            finished = [False] * batch_size
            
            for _ in range(max_len):
                dec_emb = self.tgt_embed(dec_input)
                dec_h_exp = dec_hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
                attn_energy = torch.tanh(self.attn(torch.cat((dec_h_exp, enc_outputs), dim=-1)))
                attn_scores = self.v(attn_energy).squeeze(-1)
                attn_weights = torch.softmax(attn_scores, dim=-1).unsqueeze(1)
                
                context = torch.bmm(attn_weights, enc_outputs)
                dec_out, dec_hidden = self.decoder(torch.cat((dec_emb, context), dim=-1), dec_hidden)
                
                logits = self.out_proj(dec_out.squeeze(1))
                top1 = logits.argmax(dim=-1)
                
                for b in range(batch_size):
                    if not finished[b]:
                        idx = top1[b].item()
                        if idx == eos_idx:
                            finished[b] = True
                        else:
                            decoded_batch[b].append(idx)
                            
                if all(finished):
                    break
                dec_input = top1.unsqueeze(1)
                
        return decoded_batch

# =====================================================================
# 6. TRAINING & EVALUATION ENGINE
# =====================================================================

def evaluate_model_on_dataset(model: Seq2SeqAttention, data_loader: DataLoader, 
                              vocab_tgt: CharVocabulary, direction: str, lang: str) -> Tuple[float, Dict[str, float]]:
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = next(model.parameters()).device
    
    total_loss = 0.0
    predictions = []
    references = []
    indic_sources = []
    
    with torch.no_grad():
        for src_tensor, tgt_tensor, raw_indic, raw_roman in data_loader:
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
                
    avg_loss = total_loss / max(len(data_loader), 1)
    metric_results = calculate_fine_grained_metrics(predictions, references, indic_sources, lang)
    return avg_loss, metric_results

def run_valimeli_experiment(lang: str = "tam", arm: str = "A1", direction: str = "indic-en",
                            epochs: int = 8, batch_size: int = 256, lr: float = 1e-3,
                            max_train_samples: Optional[int] = 200000,
                            max_val_samples: Optional[int] = None,
                            max_test_samples: Optional[int] = None) -> Dict[str, Any]:
    check_disk_usage()
    arm_names = {"A0": "Baseline (Character)", "A1": "ValiMeli (Phonology-Aware)", "A2": "Morphology-Aware (arXiv:2508.08424)"}
    print("\n" + "=" * 80, flush=True)
    print(f" PROJECT VALIMELI — 3-ARM COMPARATIVE BENCHMARK RUN", flush=True)
    print(f" Language: {lang.upper()} | Arm: {arm} [{arm_names.get(arm, arm)}] | Direction: {direction.upper()}", flush=True)
    print(f" Epochs: {epochs} | Batch: {batch_size} | Time: {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print("=" * 80, flush=True)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f" -> Compute Device: {device}", flush=True)
    
    run_dir = os.path.join(RUNS_DIR, f"{lang}_{arm}_{direction}")
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
    
    for indic, roman in train_pairs:
        s, t = transform_sample(indic, roman, arm, lang, direction)
        for char in s:
            vocab_src.add_character(char)
        for char in t:
            vocab_tgt.add_character(char)
            
    print(f"\n -> Vocabulary Compiled: Source Tokens = {vocab_src.num_chars} | Target Tokens = {vocab_tgt.num_chars}", flush=True)
    
    train_ds = TransliterationDataset(train_pairs, arm, lang, direction, vocab_src, vocab_tgt)
    valid_ds = TransliterationDataset(valid_pairs, arm, lang, direction, vocab_src, vocab_tgt)
    test_ds = TransliterationDataset(test_pairs, arm, lang, direction, vocab_src, vocab_tgt)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=pad_collate_fn)
    
    model = Seq2SeqAttention(vocab_src.num_chars, vocab_tgt.num_chars, embed_dim=128, hidden_dim=256).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    best_val_loss = float("inf")
    loss_history = []
    
    print(f"\n -> Initiating Training over {len(train_pairs):,} samples ({len(train_loader)} batches/epoch)...", flush=True)
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (src_batch, tgt_batch, _, _) in enumerate(train_loader):
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)
            
            optimizer.zero_grad()
            output = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)
            
            loss = criterion(output[:, 1:].reshape(-1, vocab_tgt.num_chars), tgt_batch[:, 1:].reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if (batch_idx + 1) % 200 == 0 or (batch_idx + 1) == len(train_loader):
                pct = ((batch_idx + 1) / len(train_loader)) * 100
                print(f"  [Epoch {epoch+1}/{epochs}] Batch [{batch_idx+1}/{len(train_loader)}] ({pct:.1f}%) | Loss: {loss.item():.4f}", flush=True)
            
        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        loss_history.append(avg_train_loss)
        
        val_loss, val_metrics = evaluate_model_on_dataset(model, valid_loader, vocab_tgt, direction, lang)
        epoch_time = time.time() - start_time
        
        print(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ({epoch_time:.1f}s) ===", flush=True)
        print(f"    Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val EM: {val_metrics['exact_match_accuracy']:.2f}% | Val CER: {val_metrics['character_error_rate']:.2f}% | Val SVA: {val_metrics['stop_voicing_accuracy']:.2f}%", flush=True)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss_history": loss_history,
                "vocab_src": vocab_src,
                "vocab_tgt": vocab_tgt,
                "best_val_loss": best_val_loss,
                "val_em": val_metrics["exact_match_accuracy"],
                "val_cer": val_metrics["character_error_rate"],
                "val_sva": val_metrics["stop_voicing_accuracy"]
            }, checkpoint_path)
            print(f"    ⭐ New best checkpoint saved with Val Loss: {best_val_loss:.4f}", flush=True)
            
    print("\n -> Evaluating Best Checkpoint on FULL Official Holdout Test Split...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    test_loss, test_metrics = evaluate_model_on_dataset(model, test_loader, vocab_tgt, direction, lang)
    
    print("\n" + "*" * 70, flush=True)
    print(f" OFFICIAL HOLDOUT TEST RESULTS ({lang.upper()} - {arm} - {direction.upper()}):", flush=True)
    print(f"   Exact Match Accuracy (Top-1 EM):   {test_metrics['exact_match_accuracy']:.2f}%", flush=True)
    print(f"   Character Error Rate (CER):         {test_metrics['character_error_rate']:.2f}%", flush=True)
    print(f"   Stop-Voicing Accuracy (SVA):        {test_metrics['stop_voicing_accuracy']:.2f}%", flush=True)
    print(f"   Test Cross-Entropy Loss:            {test_loss:.4f}", flush=True)
    print("*" * 70 + "\n", flush=True)
    
    report = {
        "experiment_metadata": {
            "language": lang,
            "experimental_arm": arm,
            "arm_label": arm_names.get(arm, arm),
            "direction": direction,
            "epochs": epochs,
            "train_samples": len(train_pairs),
            "test_samples": len(test_pairs),
            "device": str(device)
        },
        "metrics": {
            "validation_loss": best_val_loss,
            "test_loss": test_loss,
            "exact_match_accuracy": test_metrics["exact_match_accuracy"],
            "character_error_rate": test_metrics["character_error_rate"],
            "stop_voicing_accuracy": test_metrics["stop_voicing_accuracy"],
            "direction": direction
        }
    }
    
    report_path = os.path.join(run_dir, "eval_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    artifact_report_path = os.path.join(ARTIFACTS_DIR, f"{lang}_{arm}_{direction}_results.json")
    with open(artifact_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
        
    try:
        plot_script = os.path.join(WORKSPACE_DIR, "src", "plot_results.py")
        if os.path.exists(plot_script):
            import subprocess
            subprocess.run([sys.executable, plot_script], check=False)
    except Exception:
        pass
        
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project ValiMeli Transliteration Benchmark")
    parser.add_argument("--lang", type=str, default="tam", choices=["tam", "mal"], help="Language code")
    parser.add_argument("--arm", type=str, default="A1", choices=["A0", "A1", "A2"], help="Experimental arm (A0, A1, A2)")
    parser.add_argument("--direction", type=str, default="indic-en", choices=["indic-en", "en-indic"], help="Transliteration direction")
    parser.add_argument("--epochs", type=int, default=8, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--max_train_samples", type=int, default=200000, help="Max train samples")
    args = parser.parse_args()
    
    run_valimeli_experiment(
        lang=args.lang,
        arm=args.arm,
        direction=args.direction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_train_samples=args.max_train_samples
    )
