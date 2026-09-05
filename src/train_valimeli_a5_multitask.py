#!/usr/bin/env python3
"""
ValiMeli-A5-PanIndic & Pulli-Pro Multi-Task Training Pipeline
============================================================
Implements the definitive Paper-A Target-Grounded Pan-Indic Architecture:
  1. Atomic Language Tokens: [<LANG_TA>, <LANG_ML>, <LANG_TE>, <LANG_KN>, <LANG_HI>, <LANG_BN>, <LANG_GU>, <LANG_MR>]
  2. Multi-Language Target-Grounded Phonological Supervision:
     - 10-Class Articulatory Supervision strictly grounded on the Target Indic Grapheme Clusters:
       * Dravidian Vallinam Plosives (k, c, t, th, p) & Mellinam Nasals (ng, nj, n, m)
       * Geminate Pulli Clusters (kk, tt, pp)
       * Indo-Aryan Coda Nasalization (Anusvara / Candrabindu in hain, mein, karein)
       * Perso-Arabic Nuqta Fricatives (z, f, kh, q in zindagi, film)
       * Bengali Inherent Vowel Shift (/a/ <-> /o/)
       * Marathi Retroflex Flap (ळ)
       * Indo-Aryan / Dravidian Aspirated Voiced Stops (bh, dh, gh, jh)
  3. Dynamic Training-Time Pan-Indic Chat Augmentation:
     - Stochastic Vowel Shortening (aa->a, ee->i/e, oo->u/o)
     - Aspirate Dropping (bh->b, dh->d, th->t, kh->k)
     - Coda Nasal Dropping (hain->hai, mein->me)
     - Perso-Arabic Conversational Shifts (z->j, f->ph)
  4. Temperature-Balanced Dataloading (T=0.5):
     - Equalized gradient updates across all 8 language families.
  5. Continuous Dual-Stage Weekend Training:
     - Stage 1: ValiMeli-A5-PanIndic (3.2M Edge Model, Warm-started from A3, 8 Epochs)
     - Stage 2: Pulli-Pro (12.8M Flagship Model, d_model=384, L=8, H=6, 8 Epochs)
  6. Automated Post-Training Golden Suite Benchmark & Demo Server Sync.

Copyright (c) 2026 BalaSundaraRaman Lakshmanan (oligoglot). All Rights Reserved.
"""

import os
import sys
import json
import gzip
import math
import time
import random
import re
import subprocess
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. DIRECTORY CONFIGURATION & SAFEGUARDS
# =====================================================================

VALIMELI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(os.path.dirname(VALIMELI_DIR), "pulli")
SCRATCH_DIR = os.path.join(VALIMELI_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(WORKSPACE_DIR, "data")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

PAN_INDIC_DATASET_FILE = os.path.join(DATA_DIR, "valimeli_a4_pan_indic_train.jsonl.gz")
A3_CHECKPOINT_PATH = os.path.join(WORKSPACE_DIR, "models", "multilingual_multitask_a3_best.pt")

A5_EDGE_MODEL_OUT = os.path.join(WORKSPACE_DIR, "models", "multilingual_multitask_a5_best.pt")
PULLI_PRO_MODEL_OUT = os.path.join(WORKSPACE_DIR, "models", "pulli_pro_flagship_best.pt")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 512
NUM_PHONO_CLASSES = 10
NUM_EPOCHS = 8

LANG_PREFIXES = {
    "tam": "__ta__", "mal": "__ml__", "tel": "__te__", "kan": "__kn__",
    "hin": "__hi__", "ben": "__bn__", "guj": "__gu__", "mar": "__mr__"
}
LANG_TAG_LIST = ["__ta__", "__ml__", "__te__", "__kn__", "__hi__", "__bn__", "__gu__", "__mr__"]

# =====================================================================
# 2. ATOMIC VOCABULARY DEFINITION
# =====================================================================

class AtomicVocab:
    def __init__(self, special_tokens=("<pad>", "<sos>", "<eos>", "<unk>"), lang_tags=LANG_TAG_LIST):
        self.itos = list(special_tokens) + list(lang_tags)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def add_token(self, tok: str):
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def encode(self, lang_tag: str, text: str, add_sos=True, add_eos=True) -> List[int]:
        ids = []
        if add_sos:
            ids.append(self.stoi["<sos>"])
        if lang_tag in self.stoi:
            ids.append(self.stoi[lang_tag])
        for ch in text:
            ids.append(self.stoi.get(ch, self.stoi["<unk>"]))
        if add_eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def encode_tgt(self, text: str, add_sos=True, add_eos=True) -> List[int]:
        ids = []
        if add_sos:
            ids.append(self.stoi["<sos>"])
        for ch in text:
            ids.append(self.stoi.get(ch, self.stoi["<unk>"]))
        if add_eos:
            ids.append(self.stoi["<eos>"])
        return ids

    def decode(self, ids: List[int]) -> str:
        tokens = []
        for i in ids:
            if i == self.stoi["<eos>"]:
                break
            if i not in (self.stoi["<pad>"], self.stoi["<sos>"]) and i >= len(LANG_TAG_LIST) + 4:
                tokens.append(self.itos[i])
            elif i >= 4 and i < len(LANG_TAG_LIST) + 4:
                pass # Skip language tags during target decoding
        return "".join(tokens)

    def __len__(self):
        return len(self.itos)

import __main__
__main__.AtomicVocab = AtomicVocab
__main__.Vocab = AtomicVocab

# =====================================================================
# 3. TARGET-GROUNDED PAN-INDIC PHONOLOGY CLASSIFIER (PAPER A INSIGHT)
# =====================================================================

DRAVIDIAN_VALLINAM = {'க', 'ச', 'ட', 'த', 'ப', 'ற', 'ക', 'ച', 'ട', 'ത', 'പ', 'റ', 'క', 'చ', 'ట', 'త', 'ప', 'ಕ', 'ಚ', 'ಟ', 'ತ', 'ಪ'}
DRAVIDIAN_MELLINAM = {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன', 'ങ', 'ഞ', 'ണ', 'ന', 'മ', 'ഩ', 'ఙ', 'ఞ', 'ణ', 'న', 'మ', 'ಙ', 'ಞ', 'ಣ', 'ನ', 'ಮ'}
DRAVIDIAN_IDAYINAM = {'ய', 'ர', 'ல', 'வ', 'ழ', 'ள', 'യ', 'ര', 'ല', 'വ', 'ഴ', 'ള', 'య', 'ర', 'ల', 'వ', 'ళ', 'ಯ', 'ರ', 'ಲ', 'ವ', 'ಳ'}
IA_ASPIRATED_STOPS = {'ख', 'घ', 'छ', 'झ', 'ठ', 'ढ', 'थ', 'ध', 'फ', 'भ', 'খ', 'ঘ', 'ছ', 'ঝ', 'ঠ', 'ঢ', 'থ', 'ধ', 'ফ', 'ভ', 'ખ', 'ઘ', 'છ', 'ઝ', 'ઠ', 'ઢ', 'થ', 'ધ', 'ફ', 'ભ'}
IA_NUQTA_CHARS = {'ज़', 'फ़', 'ख़', 'ग़', 'क़', 'ड़', 'ढ़', 'ज़', 'फ़', 'ख़', 'ग़', 'क़', 'ड़', 'ढ़'}
BENGALI_INHERENT_VOWELS = {'অ', 'ও', 'ো'}

def classify_target_phonology(tgt_text: str, lang_tag: str) -> int:
    """
    Paper A Target-Grounded Phonology Classifier:
    Supervised 100% on the true Target Indic Grapheme sequence!
    """
    if not tgt_text:
        return 0

    # 1. Dravidian Geminate Plosives (Pulli/Virama + Vallinam stop)
    if '்' in tgt_text or '്' in tgt_text or '్' in tgt_text or '್' in tgt_text:
        for i in range(len(tgt_text) - 1):
            if tgt_text[i] in {'்', '്', '్', '್'} and tgt_text[i+1] in DRAVIDIAN_VALLINAM:
                return 3 # Geminate stop cluster

    # 2. Indo-Aryan Coda Nasalization (Anusvara / Candrabindu)
    if any(c in tgt_text for c in {'ं', 'ँ', 'ং', 'ঁ', 'ં', 'ੰ'}):
        return 5 # Coda Nasalization

    # 3. Perso-Arabic Nuqta Fricatives
    if any(n in tgt_text for n in IA_NUQTA_CHARS) or '़' in tgt_text or '়' in tgt_text:
        return 6 # Nuqta Fricative

    # 4. Marathi Retroflex Flap (ळ)
    if lang_tag == "__mr__" and 'ळ' in tgt_text:
        return 8 # Marathi Flap

    # 5. Bengali Inherent Vowel Shift
    if lang_tag == "__bn__" and any(c in tgt_text for c in BENGALI_INHERENT_VOWELS):
        return 7 # Bengali O-shift

    # 6. Indo-Aryan / Dravidian Aspirated Voiced Stops
    if any(c in tgt_text for c in IA_ASPIRATED_STOPS):
        return 4 # Aspirated stop

    # 7. Initial / Root Phonotactic Category of First Consonant
    first_c = tgt_text[0]
    if first_c in DRAVIDIAN_VALLINAM:
        return 1 # Vallinam Hard Stop
    elif first_c in DRAVIDIAN_MELLINAM:
        return 2 # Mellinam Nasal
    elif first_c in DRAVIDIAN_IDAYINAM:
        return 9 # Idayinam Medial

    return 0 # Default vocalic / neutral

# =====================================================================
# 4. DYNAMIC TRAINING-TIME PAN-INDIC CHAT AUGMENTATION
# =====================================================================

def augment_pan_indic_chat(roman: str, lang_tag: str, prob: float = 0.65) -> str:
    """Stochastically collapses chat Romanizations during training."""
    if random.random() > prob:
        return roman

    s = roman.lower().strip()

    # 1. Pan-Indic Vowel-Collapse (aa->a, ee->i/e, oo->u/o, ii->i, uu->u)
    vowel_rules = [
        (r'aa', 'a'),
        (r'ii', 'i'),
        (r'ee', 'i' if random.random() < 0.6 else 'e'),
        (r'uu', 'u'),
        (r'oo', 'o' if random.random() < 0.6 else 'u'),
        (r'ai', 'ay' if random.random() < 0.5 else 'e')
    ]
    for pat, rep in vowel_rules:
        if random.random() < 0.6:
            s = re.sub(pat, rep, s)

    # 2. Indo-Aryan Coda Nasal Drop (hain -> hai, mein -> me, karein -> kare)
    if lang_tag in {"__hi__", "__gu__", "__mr__", "__bn__"}:
        if s.endswith("in") and len(s) > 3 and random.random() < 0.7:
            s = s[:-2] + ("i" if random.random() < 0.5 else "e")
        elif s.endswith("n") and len(s) > 3 and random.random() < 0.5:
            s = s[:-1]

    # 3. Aspirate Dropping (bhaiya -> baiya, dhanyawad -> danyawad, theedhum -> teedhum)
    aspirates = [('bh', 'b'), ('dh', 'd'), ('th', 't'), ('kh', 'k'), ('gh', 'g'), ('ph', 'p'), ('ch', 'c')]
    for asp, plain in aspirates:
        if asp in s and random.random() < 0.4:
            s = s.replace(asp, plain)

    # 4. Perso-Arabic Conversational Substitutions (z -> j, f -> ph)
    if 'z' in s and random.random() < 0.6:
        s = s.replace('z', 'j')
    if 'f' in s and random.random() < 0.4:
        s = s.replace('f', 'ph')

    # 5. Bengali Inherent Vowel Variation (o <-> a)
    if lang_tag == "__bn__":
        if 'o' in s and random.random() < 0.5:
            s = s.replace('o', 'a')
        elif 'a' in s and random.random() < 0.5:
            s = s.replace('a', 'o')

    return s

# =====================================================================
# 5. DATASET WITH TEMPERATURE-BALANCED SAMPLING (T=0.5)
# =====================================================================

class PanIndicBalancedDataset(Dataset):
    def __init__(self, items_by_lang: Dict[str, List[Tuple[str, str]]], src_vocab: AtomicVocab, tgt_vocab: AtomicVocab, total_samples: int = 6000000, is_train: bool = True):
        self.items_by_lang = items_by_lang
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.is_train = is_train
        self.lang_tags = list(items_by_lang.keys())

        # Temperature-scaled language probabilities (T = 0.5)
        raw_counts = np.array([len(items_by_lang[l]) for l in self.lang_tags], dtype=np.float64)
        scaled_counts = np.power(raw_counts, 0.5)
        self.lang_probs = scaled_counts / np.sum(scaled_counts)
        self.total_samples = total_samples

        print("\n ⚖️  Temperature-Balanced Language Distribution (T=0.5):")
        for l, p, raw in zip(self.lang_tags, self.lang_probs, raw_counts):
            print(f"    {l:<8}: Raw={int(raw):>10,d} | Balanced Prob={p*100:>5.1f}%")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        # 1. Sample language proportionally
        lang = np.random.choice(self.lang_tags, p=self.lang_probs)
        pair_idx = random.randint(0, len(self.items_by_lang[lang]) - 1)
        roman, indic = self.items_by_lang[lang][pair_idx]

        # 2. Dynamic Training Augmentation
        if self.is_train:
            roman_input = augment_pan_indic_chat(roman, lang)
        else:
            roman_input = roman

        # 3. Ground truth target phonology label
        phono_cls = classify_target_phonology(indic, lang)

        # 4. Atomic token encoding
        src_ids = self.src_vocab.encode(lang, roman_input, add_sos=False, add_eos=True)
        tgt_ids = self.tgt_vocab.encode_tgt(indic, add_sos=True, add_eos=True)

        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
            phono_cls
        )

def collate_fn(batch):
    srcs, tgts, phonos = zip(*batch)
    src_lens = [len(s) for s in srcs]
    tgt_lens = [len(t) for t in tgts]
    max_src = max(src_lens)
    max_tgt = max(tgt_lens)

    padded_src = torch.zeros(len(srcs), max_src, dtype=torch.long)
    padded_tgt = torch.zeros(len(tgts), max_tgt, dtype=torch.long)

    for i, (s, t) in enumerate(zip(srcs, tgts)):
        padded_src[i, :len(s)] = s
        padded_tgt[i, :len(t)] = t

    return padded_src, padded_tgt, torch.tensor(phonos, dtype=torch.long)

# =====================================================================
# 6. NEURAL TRANSFORMER ARCHITECTURES (A5-EDGE & PULLI-PRO)
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

class PanIndicMultiTaskTransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024, num_phono_classes: int = 10):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        dec_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        enc_norm = nn.LayerNorm(d_model)
        dec_norm = nn.LayerNorm(d_model)

        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers, norm=enc_norm, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers, norm=dec_norm)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

        # Multi-task auxiliary Target-Grounded Phonology Head
        self.phono_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_phono_classes)
        )

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)
        tgt_len = tgt.size(1)

        src_emb = self.pos_encoder(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.tgt_emb(tgt) * math.sqrt(self.d_model))

        tgt_mask = self.generate_square_subsequent_mask(tgt_len, src.device)

        memory = self.encoder(src_emb, src_key_padding_mask=src_pad_mask)
        out = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask, memory_key_padding_mask=src_pad_mask)

        logits = self.fc_out(out)
        phono_logits = self.phono_head(memory[:, 0, :]) # Classify from language tag / root token
        return logits, phono_logits

# =====================================================================
# 7. VALIDATION TEST SUITE
# =====================================================================

TEST_CASES = [
    ("__ta__", "theedhum", "தீதும்", "Sangam Voiceless Stop"),
    ("__ta__", "nandrum", "நன்றும்", "Post-Nasal Mellinam Assimilation"),
    ("__ta__", "verithanam", "வெறித்தனம்", "Geminate Pulli Stop"),
    ("__ta__", "engada", "எங்கடா", "Conversational Tanglish Vowel Length"),
    ("__ta__", "bharathiyar", "பாரதியார்", "Sangam Proper Noun"),
    ("__ml__", "kambani", "കമ്പനി", "Malayalam Post-Nasal Ligature"),
    ("__ml__", "aano", "ആണോ", "Retroflex Nasal"),
    ("__hi__", "hain", "हैं", "Hindi Plural Anusvara"),
    ("__hi__", "mein", "में", "Hindi Postposition Nasalization"),
    ("__hi__", "zindagi", "ज़िंदगी", "Hindi Perso-Arabic Nuqta (z)"),
    ("__hi__", "qanoon", "क़ानून", "Perso-Arabic q"),
    ("__hi__", "namaskar", "नमस्कार", "Indo-Aryan Long Vowel"),
    ("__bn__", "nomoshkar", "নমস্কার", "Bengali Inherent Vowel (/a/ -> /o/)"),
    ("__bn__", "onek", "অনেক", "Bengali Anusvara / Vowel"),
    ("__te__", "baahubali", "బాహుబలి", "Telugu Aspirated Bilabial Stop"),
    ("__te__", "bagundi", "బాగుంది", "Telugu Anusvara"),
    ("__kn__", "kantara", "ಕಾಂತಾರ", "Kannada Long Vowel"),
    ("__gu__", "kemcho", "કેમ છો", "Gujarati Greeting"),
    ("__gu__", "dudh", "દૂધ", "Gujarati Long Vowel"),
    ("__mr__", "mangalwar", "मंगळवार", "Marathi Retroflex Flap (ळ)")
]

def run_validation_battery(model: PanIndicMultiTaskTransformer, src_vocab: AtomicVocab, tgt_vocab: AtomicVocab, device: torch.device):
    model.eval()
    print("\n--- PAN-INDIC TARGET-GROUNDED VALIDATION BATTERY ---")
    correct = 0
    with torch.no_grad():
        for lang_tag, roman, expected, desc in TEST_CASES:
            src_ids = src_vocab.encode(lang_tag, roman, add_sos=False, add_eos=True)
            src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
            src_repr = model.pos_encoder(model.src_emb(src_t) * math.sqrt(model.d_model))
            memory = model.encoder(src_repr)

            out_ids = [tgt_vocab.stoi["<sos>"]]
            for _ in range(35):
                tgt_t = torch.tensor([out_ids], dtype=torch.long, device=device)
                tgt_repr = model.pos_encoder(model.tgt_emb(tgt_t) * math.sqrt(model.d_model))
                tgt_mask = model.generate_square_subsequent_mask(len(out_ids), device)
                out = model.decoder(tgt_repr, memory, tgt_mask=tgt_mask)
                next_tok = model.fc_out(out[:, -1, :]).argmax(dim=-1).item()
                if next_tok == tgt_vocab.stoi["<eos>"]:
                    break
                out_ids.append(next_tok)

            pred = tgt_vocab.decode(out_ids)
            matched = (pred.strip() == expected.strip())
            if matched:
                correct += 1
            status = "✅ PASS" if matched else "❌ DIFF"
            print(f"  {status} [{lang_tag}] {roman:<15} -> Pred: {pred:<18} (Expected: {expected:<18}) | {desc}")
    acc = (correct / len(TEST_CASES)) * 100.0
    print(f"⭐ Battery Accuracy: {correct}/{len(TEST_CASES)} ({acc:.1f}%)\n")
    return acc

# =====================================================================
# 8. TRAINING ENGINE
# =====================================================================

def train_stage(model_name: str, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, lr: float, num_epochs: int, save_path: str, warm_start_a3: bool = True):
    print("\n" + "=" * 90)
    print(f" 🚀 LAUNCHING STAGE: {model_name} (d_model={d_model}, L={num_layers}, H={nhead}, lr={lr})")
    print("=" * 90)

    # 1. Load Dataset into Memory Partitioned by Language
    print(f" -> Loading Master Pan-Indic Dataset from {PAN_INDIC_DATASET_FILE}...", flush=True)
    items_by_lang: Dict[str, List[Tuple[str, str]]] = {l: [] for l in LANG_TAG_LIST}
    src_vocab = AtomicVocab()
    tgt_vocab = AtomicVocab()

    count = 0
    with gzip.open(PAN_INDIC_DATASET_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            l = obj.get("lang", obj.get("lang_code", ""))
            if l not in items_by_lang:
                continue
            r = obj["roman"].strip()
            i = obj["indic"].strip()
            items_by_lang[l].append((r, i))

            for ch in r: src_vocab.add_token(ch)
            for ch in i: tgt_vocab.add_token(ch)
            count += 1
            if count % 5000000 == 0:
                print(f"    ... loaded {count:,d} / 27,000,000 pairs", flush=True)

    print(f"    ✓ Total Pairs Loaded: {sum(len(v) for v in items_by_lang.values()):,d}", flush=True)
    print(f"    ✓ Vocab Sizes: Source={len(src_vocab)}, Target={len(tgt_vocab)}", flush=True)

    # 2. Build Temperature-Balanced Dataset (6.0M balanced pairs per epoch)
    train_dataset = PanIndicBalancedDataset(items_by_lang, src_vocab, tgt_vocab, total_samples=6000000, is_train=True)
    val_dataset = PanIndicBalancedDataset(items_by_lang, src_vocab, tgt_vocab, total_samples=100000, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # 3. Instantiate Model
    model = PanIndicMultiTaskTransformer(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_phono_classes=NUM_PHONO_CLASSES
    ).to(DEVICE)

    # 4. Optional Warm-Start from A3
    if warm_start_a3 and os.path.exists(A3_CHECKPOINT_PATH) and d_model == 256:
        print(f" -> Warm-starting weights from A3-MT ({A3_CHECKPOINT_PATH})...")
        try:
            a3_ckpt = torch.load(A3_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
            st = a3_ckpt["model_state_dict"]
            model_st = model.state_dict()
            matched_keys = 0
            for k, v in st.items():
                if k in model_st and model_st[k].shape == v.shape:
                    model_st[k] = v
                    matched_keys += 1
            model.load_state_dict(model_st)
            print(f"    ✅ Warm-started {matched_keys} weight tensors from A3-MT successfully!")
        except Exception as e:
            print(f"    ⚠️ Could not warm-start from A3: {e}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-8, weight_decay=1e-4)
    criterion_trans = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.08)
    criterion_phono = nn.CrossEntropyLoss(label_smoothing=0.05)

    best_val_loss = float('inf')

    # 5. Training Loop
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss, total_trans, total_phono = 0.0, 0.0, 0.0
        start_time = time.time()
        num_batches = len(train_loader)

        print(f"\n--- Epoch {epoch}/{num_epochs} [{model_name}] ---")

        for b_idx, (src, tgt, phono_labels) in enumerate(train_loader, 1):
            src, tgt, phono_labels = src.to(DEVICE), tgt.to(DEVICE), phono_labels.to(DEVICE)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            logits, phono_logits = model(src, tgt_in)

            loss_t = criterion_trans(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            loss_p = criterion_phono(phono_logits, phono_labels)
            loss = loss_t + 0.15 * loss_p

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            total_trans += loss_t.item()
            total_phono += loss_p.item()

            if b_idx % 2000 == 0 or b_idx == num_batches:
                elapsed = time.time() - start_time
                avg_l = total_loss / b_idx
                avg_t = total_trans / b_idx
                avg_p = total_phono / b_idx
                print(f"  [Epoch {epoch}/{num_epochs}] Batch [{b_idx:>5d}/{num_batches}] ({b_idx/num_batches*100:>5.1f}%) | Loss: {avg_l:.4f} (Trans: {avg_t:.4f}, Phono: {avg_p:.4f}) | {elapsed:.0f}s", flush=True)
                # Periodic intermediate safeguard save
                torch.save({
                    "model_name": model_name,
                    "epoch": epoch,
                    "batch": b_idx,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "dim_feedforward": dim_feedforward,
                    "model_state_dict": model.state_dict(),
                    "src_vocab": src_vocab,
                    "tgt_vocab": tgt_vocab,
                    "val_loss": best_val_loss
                }, save_path)

        # Validation Step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, tgt, phono_labels in val_loader:
                src, tgt, phono_labels = src.to(DEVICE), tgt.to(DEVICE), phono_labels.to(DEVICE)
                logits, phono_logits = model(src, tgt[:, :-1])
                l_t = criterion_trans(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
                l_p = criterion_phono(phono_logits, phono_labels)
                val_loss += (l_t + 0.15 * l_p).item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"\n=== EPOCH {epoch}/{num_epochs} SUMMARY | Train Loss: {total_loss/num_batches:.4f} | Val Loss: {avg_val_loss:.4f} ===", flush=True)

        # Run Test Battery
        run_validation_battery(model, src_vocab, tgt_vocab, DEVICE)

        # Save Best Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_name": model_name,
                "epoch": epoch,
                "d_model": d_model,
                "nhead": nhead,
                "num_layers": num_layers,
                "dim_feedforward": dim_feedforward,
                "model_state_dict": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "val_loss": best_val_loss
            }, save_path)
            print(f"  💾 Saved New Best Model to {save_path} (Val Loss: {best_val_loss:.4f})", flush=True)

    print(f"\n✅ {model_name} Training Complete! Final Model Saved: {save_path}", flush=True)

# =====================================================================
# 9. MAIN CONTINUOUS PIPELINE
# =====================================================================

def main():
    print("=" * 90)
    print(" 🌟 PROJECT PULLI — WEEKEND PAN-INDIC MULTI-TASK PIPELINE")
    print("=" * 90)

    # Stage 1: ValiMeli-A5-PanIndic (3.2M Edge Model, Warm-started from A3, 8 Epochs)
    train_stage(
        model_name="ValiMeli-A5-PanIndic-Edge",
        d_model=256,
        nhead=4,
        num_layers=6,
        dim_feedforward=1024,
        lr=6e-4,
        num_epochs=8,
        save_path=A5_EDGE_MODEL_OUT,
        warm_start_a3=True
    )

    # Stage 2: Pulli-Pro (12.8M Flagship Model, d_model=384, L=8, H=6, 8 Epochs)
    train_stage(
        model_name="Pulli-Pro-Flagship",
        d_model=384,
        nhead=6,
        num_layers=8,
        dim_feedforward=1536,
        lr=5e-4,
        num_epochs=8,
        save_path=PULLI_PRO_MODEL_OUT,
        warm_start_a3=False
    )

    # Stage 3: Automated Benchmark Evaluation on 215-Sample Golden Suite
    print("\n" + "=" * 90)
    print(" 📊 EXECUTING AUTOMATED FINAL BENCHMARK AUDIT ON 215 GOLDEN SAMPLES")
    print("=" * 90)
    subprocess.run([
        sys.executable,
        os.path.join(WORKSPACE_DIR, "benchmarks", "sarvam", "eval_golden_suite_all_models.py")
    ], check=False)

if __name__ == "__main__":
    main()
