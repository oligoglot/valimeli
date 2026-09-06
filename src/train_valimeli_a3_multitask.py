#!/usr/bin/env python3
"""
ValiMeli-A3-MT: 8-Language Multi-Task Multilingual Transliteration Model
Trained on:
  - Super-Curated Grounded Dataset (DEDR Verb Paradigms + Theedhum Nandrum Tanglish + Aksharantar)
  - Collision-Free 4-Rule Alignment Matrix (Zero over-lengthening regressions)
  - Tolkappiyam Multi-Task Auxiliary Stop-Allophony Head (7 phonotactic classes)

Copyright (c) 2026 BalaSundaraRaman Lakshmanan (oligoglot). All Rights Reserved.
"""

import os
import sys
import json
import gzip
import math
import time
import random
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. DIRECTORY CONFIGURATION
# =====================================================================

VALIMELI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(os.path.dirname(VALIMELI_DIR), "pulli")
SCRATCH_DIR = os.path.join(VALIMELI_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

GROUNDED_DATASET_FILE = os.path.join(WORKSPACE_DIR, "data", "valimeli_a3_grounded_train.jsonl.gz")
MODEL_OUT_PATH = os.path.join(RUNS_DIR, "multilingual_multitask_a3_best.pt")
PULLI_SYNC_PATH = os.path.join(WORKSPACE_DIR, "models", "multilingual_multitask_a3_best.pt")

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 512
D_MODEL = 256
NHEAD = 4
NUM_LAYERS = 6
DIM_FEEDFORWARD = 1024
NUM_PHONO_CLASSES = 7
NUM_EPOCHS = 8
LR = 7e-4
WEIGHT_DECAY = 1e-4

LANG_CODES = ["tam", "mal", "tel", "kan", "hin", "ben", "guj", "mar"]
LANG_PREFIXES = {
    "tam": "__ta__", "mal": "__ml__", "tel": "__te__", "kan": "__kn__",
    "hin": "__hi__", "ben": "__bn__", "guj": "__gu__", "mar": "__mr__"
}

# =====================================================================
# 2. VOCABULARIES & DATASET
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

def get_phonetic_class(char: str) -> int:
    """Classifies Tamil grapheme into 7 phonotactic categories."""
    if char in {'க', 'ச', 'ட', 'த', 'ப', 'ற'}:
        return 1  # Vallinam (Hard stops)
    elif char in {'ங', 'ஞ', 'ண', 'ந', 'ம', 'ன'}:
        return 2  # Mellinam (Nasals)
    elif char in {'ய', 'ர', 'ல', 'வ', 'ழ', 'ள'}:
        return 3  # Idayinam (Medials)
    elif char in {'அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ'}:
        return 4  # Independent Vowels
    elif char in {'ா', 'ி', 'ீ', 'ு', 'ூ', 'ெ', 'ே', 'ை', 'ொ', 'ோ', 'ௌ'}:
        return 5  # Matras / Vowel signs
    elif char == '்':
        return 6  # Halant / Pulli
    return 0      # Neutral / Non-Tamil

class GroundedA3Dataset(Dataset):
    def __init__(self, items: List[Dict[str, Any]], src_vocab: MultilingualVocab, tgt_vocab: MultilingualVocab):
        self.items = items
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        lang = item["lang"]
        roman = item["roman"]
        indic = item["indic"]
        
        src_tokens = [lang] + list(roman.lower().strip()[:35])
        tgt_tokens = list(indic.strip()[:35])
        
        src_enc = self.src_vocab.encode(src_tokens)
        tgt_enc = self.tgt_vocab.encode(tgt_tokens)
        
        # Auxiliary phonotactic label (first consonant classification)
        first_indic = indic[0] if indic else ''
        phono_label = get_phonetic_class(first_indic) if lang == "__ta__" else 0
        
        return (
            torch.tensor(src_enc, dtype=torch.long),
            torch.tensor(tgt_enc, dtype=torch.long),
            torch.tensor(phono_label, dtype=torch.long)
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
        
    return padded_src, padded_tgt, torch.stack(phonos)

# =====================================================================
# 3. TRANSFORMER MODEL ARCHITECTURE
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

class ValiMeliMultiTaskTransformer(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024, num_phono_classes: int = 7):
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
        src_mask = (src == 0)
        tgt_mask = (tgt == 0)
        causal_mask = self.generate_square_subsequent_mask(tgt.size(1), src.device)
        
        src_emb = self.pos_encoder(self.src_embed(src) * math.sqrt(self.d_model))
        memory = self.encoder(src_emb, src_key_padding_mask=src_mask)
        
        phono_logits = self.aux_phono_proj(memory[:, 0, :])
        
        tgt_emb = self.pos_encoder(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        logits = self.out_proj(out)
        return logits, phono_logits

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
# 4. TRAINING & EVALUATION LOOP
# =====================================================================

def evaluate_validation(model: nn.Module, val_loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for src, tgt, phono in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            
            logits, phono_logits = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
            count += 1
            if count >= 100:  # Fast validation estimate
                break
    return total_loss / max(count, 1)

def run_tanglish_test_battery(model: nn.Module, src_vocab: MultilingualVocab, tgt_vocab: MultilingualVocab, device: torch.device):
    test_cases = [
        ("polama", "போலாமா"),
        ("theedhum", "தீதும்"),
        ("nandrum", "நன்றும்"),
        ("naarkaali", "நாற்காலி"),
        ("pirarthara", "பிறர்தர"),
        ("vaathiyaar", "வாத்தியார்"),
        ("vathiyar", "வாத்தியார்"),
        ("puriyadhu", "புரியாது"),
        ("veetuku", "வீட்டுக்கு"),
        ("kadhal", "காதல்"),
        ("thamizhagatthai", "தமிழகத்தை")
    ]
    print("\n--- VALIMELI-A3 REAL-WORLD TEST BATTERY ---")
    for rom, target in test_cases:
        enc = src_vocab.encode(["__ta__"] + list(rom[:35]))
        t = torch.tensor([enc], dtype=torch.long, device=device)
        d = model.greedy_decode(t)[0]
        pred = tgt_vocab.decode(d)
        is_exact = (pred.strip() == target.strip())
        status = "✅" if is_exact else "⚠️"
        print(f" {status} Input: '{rom:18s}' -> Pred: '{pred:18s}' (Target: '{target}')")
    print("-------------------------------------------\n")

import subprocess

def get_git_commit(repo_path: str) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_path).decode().strip()
    except Exception:
        return "unknown"

VALIMELI_COMMIT = get_git_commit(VALIMELI_DIR)
PULLI_COMMIT = get_git_commit(WORKSPACE_DIR)

def train_valimeli_a3():
    print("=" * 80)
    print(" VALIMELI-A3-MT 8-LANGUAGE MULTI-TASK GROUNDED TRAINING PIPELINE")
    print(f" Device: {DEVICE} | Batch Size: {BATCH_SIZE} | Epochs: {NUM_EPOCHS}")
    print(f" Git Commit ID (ValiMeli Repo): {VALIMELI_COMMIT}")
    print(f" Git Commit ID (Pulli Repo):    {PULLI_COMMIT}")
    print("=" * 80, flush=True)

    if not os.path.exists(GROUNDED_DATASET_FILE):
        print(f"❌ Error: Grounded dataset not found at {GROUNDED_DATASET_FILE}")
        return

    # 1. Load Grounded Dataset
    print(f" -> Loading grounded training dataset from {GROUNDED_DATASET_FILE}...")
    train_items = []
    src_vocab = MultilingualVocab()
    tgt_vocab = MultilingualVocab()
    
    # Add language prefix tags
    for tag in LANG_PREFIXES.values():
        src_vocab.add(tag)

    with gzip.open(GROUNDED_DATASET_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                train_items.append(obj)
                for c in obj["roman"].lower().strip():
                    src_vocab.add(c)
                for c in obj["indic"].strip():
                    tgt_vocab.add(c)
            except Exception:
                continue

    print(f"    Loaded {len(train_items):,} grounded training samples")
    print(f"    Vocabularies: Source Vocab = {src_vocab.n:,} | Target Vocab = {tgt_vocab.n:,}")

    # 2. Load Validation Set
    val_items = []
    for iso, tag in LANG_PREFIXES.items():
        v_path = os.path.join(DATA_DIR, f"extracted_{iso}", f"{iso}_valid.json")
        if os.path.exists(v_path):
            with open(v_path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        obj = json.loads(line)
                        ind = (obj.get("native word") or obj.get("indic") or "").strip()
                        rom = (obj.get("english word") or obj.get("roman") or "").strip().lower()
                        if ind and rom:
                            val_items.append({"roman": rom, "indic": ind, "lang": tag})
                    except Exception:
                        continue

    print(f"    Loaded {len(val_items):,} multilingual validation samples")

    # 3. Create DataLoaders
    train_dataset = GroundedA3Dataset(train_items, src_vocab, tgt_vocab)
    val_dataset = GroundedA3Dataset(val_items, src_vocab, tgt_vocab)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 4. Initialize Model
    model = ValiMeliMultiTaskTransformer(
        src_vocab_size=src_vocab.n,
        tgt_vocab_size=tgt_vocab.n,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_phono_classes=NUM_PHONO_CLASSES
    ).to(DEVICE)

    criterion_trans = nn.CrossEntropyLoss(ignore_index=0)
    criterion_phono = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    best_val_loss = float("inf")

    # 5. Training Loop
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        running_trans = 0.0
        running_phono = 0.0
        total_batches = len(train_loader)

        for b_idx, (src, tgt, phono) in enumerate(train_loader):
            src, tgt, phono = src.to(DEVICE), tgt.to(DEVICE), phono.to(DEVICE)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            optimizer.zero_grad()
            logits, phono_logits = model(src, tgt_in)
            
            l_trans = criterion_trans(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            l_phono = criterion_phono(phono_logits, phono)
            loss = l_trans + 0.05 * l_phono

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_loss += loss.item()
            running_trans += l_trans.item()
            running_phono += l_phono.item()

            if (b_idx + 1) % 500 == 0 or (b_idx + 1) == total_batches:
                avg_l = running_loss / (b_idx + 1)
                avg_t = running_trans / (b_idx + 1)
                avg_p = running_phono / (b_idx + 1)
                pct = ((b_idx + 1) / total_batches) * 100.0
                print(f"  [Epoch {epoch}/{NUM_EPOCHS}] Batch [{b_idx+1}/{total_batches}] ({pct:5.1f}%) | Loss: {avg_l:.4f} (Trans: {avg_t:.4f}, Phono: {avg_p:.4f})", flush=True)

        scheduler.step()
        epoch_time = time.time() - t0
        train_loss = running_loss / max(total_batches, 1)
        val_loss = evaluate_validation(model, val_loader, criterion_trans, DEVICE)

        print(f"\n=== EPOCH {epoch}/{NUM_EPOCHS} SUMMARY ({epoch_time:.1f}s) | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} ===")
        run_tanglish_test_battery(model, src_vocab, tgt_vocab, DEVICE)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = {
                "model_state_dict": model.state_dict(),
                "src_vocab": src_vocab,
                "tgt_vocab": tgt_vocab,
                "epoch": epoch,
                "val_loss": val_loss,
                "git_commit_valimeli": VALIMELI_COMMIT,
                "git_commit_pulli": PULLI_COMMIT,
                "config": {
                    "d_model": D_MODEL,
                    "nhead": NHEAD,
                    "num_layers": NUM_LAYERS,
                    "dim_feedforward": DIM_FEEDFORWARD,
                    "num_phono_classes": NUM_PHONO_CLASSES
                }
            }
            torch.save(ckpt, MODEL_OUT_PATH)
            torch.save(ckpt, PULLI_SYNC_PATH)
            print(f"    ⭐ Saved new best ValiMeli-A3-MT checkpoint: {MODEL_OUT_PATH}")
            print(f"    ⭐ Synced checkpoint to Pulli: {PULLI_SYNC_PATH}\n", flush=True)

    print("=" * 80)
    print(" VALIMELI-A3-MT TRAINING COMPLETE!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    train_valimeli_a3()
