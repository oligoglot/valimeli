#!/usr/bin/env python3
"""
Project ValiMeli — Large-Scale Vowel-Collapse Multi-Task Training (A2-MT)
Trains the next-generation ValiMeli-A2-MT model with:
  1. Multi-Task Stop-Voicing Supervision (ValiMeli phonotactic head)
  2. Stochastic Training-Time Vowel-Collapse Augmentation (Tanglish invariance)
  3. Warm-starting from ValiMeli-A1-MT checkpoint (preserving A1-MT intact)

Output Checkpoint:
  valimeli/scratch/valimeli/runs/multilingual_multitask_a2_best.pt
  pulli/models/multilingual_multitask_a2_best.pt

Copyright (c) 2026 BalaSundaraRaman Lakshmanan (oligoglot). All Rights Reserved.
"""

import os
import sys
import json
import math
import time
import random
import re
import shutil
import argparse
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
PULLI_MODELS_DIR = os.path.join(os.path.dirname(WORKSPACE_DIR), "pulli", "models")

os.makedirs(SCRATCH_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

A1_CKPT_PATH = os.path.join(RUNS_DIR, "multilingual_multitask_scaling_best.pt")
A2_CKPT_PATH = os.path.join(RUNS_DIR, "multilingual_multitask_a2_best.pt")

LANG_MAP = {
    "tam": "__ta__",
    "mal": "__ml__",
    "tel": "__te__",
    "kan": "__kn__",
    "hin": "__hi__",
    "ben": "__bn__",
    "guj": "__gu__",
    "mar": "__mr__"
}

# =====================================================================
# 1. TANGLISH VOWEL-COLLAPSE AUGMENTATION
# =====================================================================

def augment_tanglish_vowels(roman_str: str, prob: float = 0.6) -> str:
    """
    Stochastically collapses doubled vowels, diphthongs, and colloquial endings
    to simulate in-the-wild conversational Tanglish and Romanized typing.
    """
    if random.random() > prob:
        return roman_str

    s = roman_str

    # 1. Doubled long vowels -> single vowels within the SAME phonemic quality
    # aa -> a (/aː/ -> /a/, e.g., polaamaa -> polama, kaadhal -> kadhal)
    # ii / ee -> i (/iː/ -> /i/, e.g., veettukku -> vitukku, theedhum -> thidhum)
    # uu / oo -> u (/uː/ -> /u/, e.g., kuudai -> kudai, koottam -> kuttam)
    # oo -> o (/oː/ -> /o/, e.g., poolaamaa -> polama, poodum -> podum)
    vowel_collapses = [
        (r'aa', 'a'),  # /aː/ -> /a/ (polaamaa -> polama, kaadhal -> kadhal)
        (r'ii', 'i'),  # /iː/ -> /i/ (viidu -> vidu, siir -> sir)
        (r'ee', 'i'),  # /iː/ -> /i/ (veedu -> vidu, neenga -> ninga) -- PREVENTS collision with /e/ (vettu)
        (r'uu', 'u'),  # /uː/ -> /u/ (kuudai -> kudai)
        (r'oo', 'o'),  # /oː/ -> /o/ (poolaamaa -> polama, poodum -> podum)
    ]
    for pattern, rep in vowel_collapses:
        if random.random() < 0.75:
            s = re.sub(pattern, rep, s)

    # 2. Diphthong reductions (kadaikku -> kadekku / kadaykku)
    if 'ai' in s and random.random() < 0.35:
        s = s.replace('ai', 'ay' if random.random() < 0.5 else 'e')

    return s

# =====================================================================
# 2. PHONOTACTIC SUPERVISION LABELS
# =====================================================================

PHONO_NONE = 0
PHONO_INIT = 1
PHONO_INTER = 2
PHONO_POST_NASAL = 3
PHONO_GEM = 4
PHONO_DEFAULT = 5
NUM_PHONO_CLASSES = 7

TAMIL_PLOSIVES = {'க', 'ச', 'ட', 'த', 'ப', 'ற'}
TAMIL_NASALS = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}
MALAYALAM_BASE_PLOSIVES = {'ക', 'ച', 'ട', 'ത', 'പ', 'റ'}
MALAYALAM_NASALS = {'ങ', 'ഞ', 'ണ', 'ന', 'മ', 'ഩ'}

def get_phonology_labels_for_word(indic_word: str, roman_word: str, lang_code: str) -> List[int]:
    labels = [PHONO_NONE] * len(roman_word)
    if lang_code not in {"tam", "mal"}:
        return labels

    plosives = TAMIL_PLOSIVES if lang_code == "tam" else MALAYALAM_BASE_PLOSIVES
    nasals = TAMIL_NASALS if lang_code == "tam" else MALAYALAM_NASALS

    chars = list(indic_word)
    is_initial = True
    prev_was_nasal = False

    for idx, c in enumerate(chars):
        if c in plosives:
            is_initial = False
            prev_was_nasal = False
        elif c in nasals:
            prev_was_nasal = True
            is_initial = False
        elif c not in {'்'}:
            prev_was_nasal = False
            is_initial = False

    for i, r_char in enumerate(roman_word):
        if r_char.lower() in {'k', 'c', 't', 'p', 'r', 'g', 'j', 'd', 'b', 's', 'h'}:
            if i == 0:
                labels[i] = PHONO_INIT
            elif i > 0 and roman_word[i-1].lower() in {'n', 'm'}:
                labels[i] = PHONO_POST_NASAL
            elif i > 0 and roman_word[i-1].lower() == r_char.lower():
                labels[i] = PHONO_GEM
            elif i > 0 and i < len(roman_word) - 1 and roman_word[i-1].lower() in "aeiou" and roman_word[i+1].lower() in "aeiou":
                labels[i] = PHONO_INTER
            else:
                labels[i] = PHONO_DEFAULT
        else:
            labels[i] = PHONO_NONE

    return labels

# =====================================================================
# 3. METRICS & VOCABULARY
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

class MultilingualVocab:
    PAD = "<PAD>"
    SOS = "<SOS>"
    EOS = "<EOS>"
    UNK = "<UNK>"
    def __init__(self):
        self.t2i = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.i2t = {0: self.PAD, 1: self.SOS, 2: self.EOS, 3: self.UNK}
        self.n = 4
    def add(self, token: str):
        if token not in self.t2i:
            self.t2i[token] = self.n
            self.i2t[self.n] = token
            self.n += 1
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

# =====================================================================
# 4. DATASET WITH VOWEL-COLLAPSE AUGMENTATION
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
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
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

def get_language_splits(lang: str, max_train: int = 400000):
    extracted_dir = os.path.join(DATA_DIR, f"extracted_{lang}")
    train_file = os.path.join(extracted_dir, f"{lang}_train.json")
    valid_file = os.path.join(extracted_dir, f"{lang}_valid.json")
    test_file = os.path.join(extracted_dir, f"{lang}_test.json")
    return (
        load_split_file(train_file, max_samples=max_train),
        load_split_file(valid_file),
        load_split_file(test_file)
    )

class MultilingualAugmentedMultiTaskDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, str, str, str]], src_vocab: MultilingualVocab, tgt_vocab: MultilingualVocab, augment: bool = True):
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.augment = augment
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        full_src, raw_indic, lang_code, source_tag = self.samples[idx]
        lang_tag = LANG_MAP[lang_code]
        roman_str = full_src[len(lang_tag):][:40]
        indic_str = raw_indic[:40]

        # Apply stochastic training-time Tanglish vowel collapse
        if self.augment and lang_code in {"tam", "mal", "tel", "kan"}:
            roman_str = augment_tanglish_vowels(roman_str, prob=0.6)

        src_tokens = [lang_tag] + list(roman_str)
        tgt_tokens = list(indic_str)

        # Multi-task phonology targets
        raw_phono = get_phonology_labels_for_word(indic_str, roman_str, lang_code)
        enc_phono = [PHONO_NONE, PHONO_NONE] + raw_phono + [PHONO_NONE]

        return (
            self.src_vocab.encode(src_tokens),
            self.tgt_vocab.encode(tgt_tokens),
            enc_phono,
            raw_indic,
            lang_code,
            source_tag
        )

def multitask_pad_collate(batch):
    src_list, tgt_list, phono_list, raw_indic_list, lang_list, source_tag_list = zip(*batch)
    max_s = max(len(s) for s in src_list)
    max_t = max(len(t) for t in tgt_list)

    s_tensor = torch.zeros(len(src_list), max_s, dtype=torch.long)
    t_tensor = torch.zeros(len(tgt_list), max_t, dtype=torch.long)
    p_tensor = torch.zeros(len(phono_list), max_s, dtype=torch.long)

    for i, (s, t, p) in enumerate(zip(src_list, tgt_list, phono_list)):
        s_tensor[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        t_tensor[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        p_len = min(len(p), max_s)
        p_tensor[i, :p_len] = torch.tensor(p[:p_len], dtype=torch.long)

    return s_tensor, t_tensor, p_tensor, raw_indic_list, lang_list, source_tag_list

# =====================================================================
# 5. MODEL ARCHITECTURE
# =====================================================================

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

class MultilingualMultiTaskTransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024, num_phono_classes: int = 6):
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
        phono_logits = self.aux_phono_proj(memory)

        out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, tgt_key_padding_mask=tgt_mask, memory_key_padding_mask=src_mask)
        logits = self.out_proj(out)

        padded = torch.zeros(logits.size(0), logits.size(1) + 1, logits.size(2), device=device)
        padded[:, 1:] = logits
        return padded, phono_logits

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
# 6. TRAINING & EVALUATION PIPELINE
# =====================================================================

def evaluate_test_loader(model: MultilingualMultiTaskTransformer, loader: DataLoader, tgt_vocab: MultilingualVocab):
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    device = next(model.parameters()).device
    total_loss = 0.0
    preds, refs, tags = [], [], []
    with torch.no_grad():
        for s_t, t_t, p_t, raw_indic, l_list, tag_list in loader:
            s_t, t_t, p_t = s_t.to(device), t_t.to(device), p_t.to(device)
            out, _ = model(s_t, t_t)
            loss = criterion(out[:, 1:].reshape(-1, tgt_vocab.n), t_t[:, 1:].reshape(-1))
            total_loss += loss.item()
            decoded = model.greedy_decode(s_t)
            for b in range(len(decoded)):
                pred_str = tgt_vocab.decode(decoded[b])
                preds.append(pred_str)
                refs.append(raw_indic[b])
                tags.append(tag_list[b])
    avg_loss = total_loss / max(len(loader), 1)
    metrics = calculate_partitioned_metrics(preds, refs, tags)
    return avg_loss, metrics

def evaluate_tanglish_suite(model: MultilingualMultiTaskTransformer, src_vocab: MultilingualVocab, tgt_vocab: MultilingualVocab, device: torch.device):
    tanglish_cases = [
        ("__ta__polama", "போலாமா"),
        ("__ta__kadaikku polama", "கடைக்கு போலாமா"),
        ("__ta__theedhum", "தீதும்"),
        ("__ta__nandrum", "நன்றும்"),
        ("__ta__naarkaali", "நாற்காலி"),
        ("__ta__kadhal", "காதல்"),
        ("__ta__veetuku varen", "வீட்டுக்கு வரேன்"),
        ("__ta__enaku theriyadhu", "எனக்கு தெரியாது"),
        ("__ta__podum", "போதும்")
    ]
    print("\n--- TANGLISH REAL-WORLD TEST BATTERY ---", flush=True)
    for src_raw, expected in tanglish_cases:
        pfx = src_raw[:6]
        word = src_raw[6:]
        encoded = src_vocab.encode([pfx] + list(word[:35]))
        tensor = torch.tensor([encoded], dtype=torch.long, device=device)
        dec = model.greedy_decode(tensor)[0]
        out_str = tgt_vocab.decode(dec)
        match_sym = "✅" if out_str == expected else "⚠️"
        print(f" {match_sym} Input: '{word:22}' -> Prediction: '{out_str:18}' (Target: '{expected}')", flush=True)
    print("----------------------------------------\n", flush=True)

def train_valimeli_a2(epochs: int = 5, batch_size: int = 512, lr: float = 2e-4, phono_weight: float = 0.3, max_train_per_lang: int = 150000):
    print("\n" + "=" * 80, flush=True)
    print(" PROJECT VALIMELI — VOWEL-COLLAPSE MULTI-TASK TRAINING (VALIMELI-A2-MT)", flush=True)
    print(" Preserving ValiMeli-A1-MT Checkpoint Intact • Fine-Tuning Tanglish Invariance", flush=True)
    print("=" * 80, flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f" -> Compute Device: {device}", flush=True)

    # Warm-start vocabularies and weights from ValiMeli-A1-MT if available
    a1_ckpt = None
    if os.path.exists(A1_CKPT_PATH):
        print(f" -> 🔄 Loading base vocabularies from ValiMeli-A1-MT ({A1_CKPT_PATH})...", flush=True)
        a1_ckpt = torch.load(A1_CKPT_PATH, map_location=device, weights_only=False)
        src_vocab = a1_ckpt["src_vocab"]
        tgt_vocab = a1_ckpt["tgt_vocab"]
    else:
        src_vocab = MultilingualVocab()
        tgt_vocab = MultilingualVocab()
        for tag in LANG_MAP.values():
            src_vocab.add(tag)
        for c in "abcdefghijklmnopqrstuvwxyz' -":
            src_vocab.add(c)

    all_train_samples = []
    all_val_samples = []
    test_samples_by_lang = {}

    for lang, tag in LANG_MAP.items():
        train_p, val_p, test_p = get_language_splits(lang, max_train=max_train_per_lang)
        print(f" -> [{lang.upper()}] Loaded: Train={len(train_p):,} | Val={len(val_p):,} | Test={len(test_p):,}", flush=True)
        for n, _, _ in train_p + val_p:
            for c in n:
                tgt_vocab.add(c)
        for n, r, s in train_p:
            all_train_samples.append((f"{tag}{r}", n, lang, s))
        for n, r, s in val_p:
            all_val_samples.append((f"{tag}{r}", n, lang, s))
        test_samples_by_lang[lang] = [(f"{tag}{r}", n, lang, s) for n, r, s in test_p]

    random.shuffle(all_train_samples)
    print(f"\n -> Total Training Corpus: {len(all_train_samples):,} pairs with Vowel-Collapse Augmentation", flush=True)
    print(f" -> Unified Multi-Script Target Vocab: {tgt_vocab.n} unique Unicode characters", flush=True)

    train_loader = DataLoader(
        MultilingualAugmentedMultiTaskDataset(all_train_samples, src_vocab, tgt_vocab, augment=True),
        batch_size=batch_size, shuffle=True, collate_fn=multitask_pad_collate
    )
    val_loader = DataLoader(
        MultilingualAugmentedMultiTaskDataset(all_val_samples, src_vocab, tgt_vocab, augment=False),
        batch_size=batch_size, shuffle=False, collate_fn=multitask_pad_collate
    )

    test_loaders = {}
    for lang, s_list in test_samples_by_lang.items():
        test_loaders[lang] = DataLoader(
            MultilingualAugmentedMultiTaskDataset(s_list, src_vocab, tgt_vocab, augment=False),
            batch_size=batch_size, shuffle=False, collate_fn=multitask_pad_collate
        )

    model = MultilingualMultiTaskTransformer(src_vocab.n, tgt_vocab.n, d_model=256, nhead=4, num_layers=6, dim_feedforward=1024, num_phono_classes=NUM_PHONO_CLASSES).to(device)

    # Warm-start weights
    if a1_ckpt is not None:
        model.load_state_dict(a1_ckpt["model_state_dict"], strict=True)
        print("    ✅ Loaded pretrained weights from ValiMeli-A1-MT base model successfully.", flush=True)

    evaluate_tanglish_suite(model, src_vocab, tgt_vocab, device)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion_translit = nn.CrossEntropyLoss(ignore_index=0)
    criterion_phono = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float("inf")

    print("\n -> Starting ValiMeli-A2-MT Augmented Fine-Tuning...", flush=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        start_time = time.time()
        for idx, (s_b, t_b, p_b, _, _, _) in enumerate(train_loader):
            s_b, t_b, p_b = s_b.to(device), t_b.to(device), p_b.to(device)
            optimizer.zero_grad()
            out, phono_logits = model(s_b, t_b)

            l_translit = criterion_translit(out[:, 1:].reshape(-1, tgt_vocab.n), t_b[:, 1:].reshape(-1))
            l_phono = criterion_phono(phono_logits.reshape(-1, NUM_PHONO_CLASSES), p_b.reshape(-1))
            loss = l_translit + (phono_weight * l_phono)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

            if (idx + 1) % 500 == 0 or (idx + 1) == len(train_loader):
                print(f"  [Epoch {epoch+1}/{epochs}] Batch [{idx+1}/{len(train_loader)}] ({(idx+1)/len(train_loader)*100:.1f}%) | Total Loss: {loss.item():.4f} (Trans: {l_translit.item():.4f}, Phono: {l_phono.item():.4f})", flush=True)

        val_loss, _ = evaluate_test_loader(model, val_loader, tgt_vocab)
        dur = time.time() - start_time
        print(f"=== EPOCH {epoch+1}/{epochs} SUMMARY ({dur:.1f}s) | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} ===", flush=True)
        evaluate_tanglish_suite(model, src_vocab, tgt_vocab, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save A2-MT separately
            torch.save({
                "epoch": epoch + 1,
                "model_name": "ValiMeli-A2-MT",
                "model_state_dict": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "val_loss": val_loss
            }, A2_CKPT_PATH)
            print(f"    ⭐ Saved new best ValiMeli-A2-MT checkpoint: {A2_CKPT_PATH}", flush=True)

            # Copy to Pulli models directory
            if os.path.exists(PULLI_MODELS_DIR):
                target_dest = os.path.join(PULLI_MODELS_DIR, "multilingual_multitask_a2_best.pt")
                shutil.copy2(A2_CKPT_PATH, target_dest)
                print(f"    ⭐ Synced checkpoint to Pulli: {target_dest}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(" VALIMELI-A2-MT TRAINING COMPLETE!", flush=True)
    print("=" * 80 + "\n", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max_train", type=int, default=150000)
    args = parser.parse_args()

    train_valimeli_a2(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, max_train_per_lang=args.max_train)
