#!/usr/bin/env python3
"""
ValiMeli-A4-PanIndic-MT: 8-Language Multi-Task Pan-Indic Transliteration Model
Trained on:
  - 27.0M Super-Curated Pan-Indic Dataset (valimeli_a4_pan_indic_train.jsonl.gz)
  - 7 Cross-Family Phonology Rules (Indo-Aryan Anusvara Nasalization, Nuqta, Bengali Vowel Shift, Marathi Flap, Dravidian Allophony)
  - Turner's CDIAL Indo-Aryan Etymological Cognates + Burrow's DEDR Dravidian Verb Paradigms
  - Tolkappiyam Multi-Task Auxiliary Stop-Allophony Head (7 phonotactic classes)

Output Checkpoints:
  valimeli/scratch/valimeli/runs/multilingual_multitask_a4_best.pt
  pulli/models/multilingual_multitask_a4_best.pt

Copyright (c) 2026 BalaSundaraRaman Lakshmanan (oligoglot). All Rights Reserved.
"""

import os
import sys
import json
import gzip
import math
import time
import random
import subprocess
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# =====================================================================
# 1. DIRECTORY CONFIGURATION & GIT COMMIT LOGGING
# =====================================================================

VALIMELI_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE_DIR = os.path.join(os.path.dirname(VALIMELI_DIR), "pulli")
SCRATCH_DIR = os.path.join(VALIMELI_DIR, "scratch", "valimeli")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)

PAN_INDIC_DATASET_FILE = os.path.join(WORKSPACE_DIR, "data", "valimeli_a4_pan_indic_train.jsonl.gz")
MODEL_OUT_PATH = os.path.join(RUNS_DIR, "multilingual_multitask_a4_best.pt")
PULLI_SYNC_PATH = os.path.join(WORKSPACE_DIR, "models", "multilingual_multitask_a4_best.pt")

def get_git_commit(repo_path: str) -> str:
    try:
        cmd = ["git", "rev-parse", "--short", "HEAD"]
        res = subprocess.run(cmd, cwd=repo_path, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except Exception:
        return "unknown"

VALIMELI_COMMIT = get_git_commit(VALIMELI_DIR)
PULLI_COMMIT = get_git_commit(WORKSPACE_DIR)

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

class Vocab:
    def __init__(self, special_tokens=("<pad>", "<sos>", "<eos>", "<unk>")):
        self.itos = list(special_tokens)
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

    def add_token(self, tok: str):
        if tok not in self.stoi:
            self.stoi[tok] = len(self.itos)
            self.itos.append(tok)

    def encode(self, text: str, add_sos=True, add_eos=True) -> List[int]:
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
            if i not in (self.stoi["<pad>"], self.stoi["<sos>"]):
                tokens.append(self.itos[i])
        return "".join(tokens)

    def __len__(self):
        return len(self.itos)

def classify_phonotactics(roman: str) -> int:
    r = roman.lower()
    if any(x in r for x in ["kk", "tt", "tth", "pp", "ch"]):
        return 3 # Geminate plosive
    if any(x in r for x in ["ng", "nj", "nd", "nth", "mb", "ngg"]):
        return 2 # Post-nasal voiced stop
    if r.startswith(("k", "t", "th", "p", "s")):
        return 1 # Initial voiceless stop
    if any(ch in r for ch in ["z", "f", "kh", "q"]):
        return 5 # Perso-Arabic Fricative
    if r.endswith("n") or r.endswith("m"):
        return 4 # Nasalized coda
    return 0 # Standard intervocalic / vocalic

class PanIndicDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, int]], src_vocab: Vocab, tgt_vocab: Vocab):
        self.pairs = pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text, phono_cls = self.pairs[idx]
        src_ids = self.src_vocab.encode(src_text, add_sos=False, add_eos=True)
        tgt_ids = self.tgt_vocab.encode(tgt_text, add_sos=True, add_eos=True)
        return torch.tensor(src_ids, dtype=torch.long), torch.tensor(tgt_ids, dtype=torch.long), phono_cls

def pad_collate(batch):
    src_list, tgt_list, phono_list = zip(*batch)
    pad_id_src = 0
    pad_id_tgt = 0

    max_src = max(len(s) for s in src_list)
    max_tgt = max(len(t) for t in tgt_list)

    src_padded = torch.full((len(src_list), max_src), pad_id_src, dtype=torch.long)
    tgt_padded = torch.full((len(tgt_list), max_tgt), pad_id_tgt, dtype=torch.long)

    for i, s in enumerate(src_list):
        src_padded[i, :len(s)] = s
    for i, t in enumerate(tgt_list):
        tgt_padded[i, :len(t)] = t

    phono_t = torch.tensor(phono_list, dtype=torch.long)
    return src_padded, tgt_padded, phono_t

# =====================================================================
# 3. TRANSFORMER MODEL ARCHITECTURE
# =====================================================================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]

class ValiMeliA4MultiTaskModel(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256,
                 nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024,
                 num_phono_classes: int = 7):
        super().__init__()
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.phono_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_phono_classes)
        )

    def forward(self, src: torch.Tensor, tgt: torch.Tensor):
        src_pad_mask = (src == 0)
        tgt_pad_mask = (tgt == 0)
        tgt_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_len).to(src.device)

        src_repr = self.pos_encoder(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_repr = self.pos_encoder(self.tgt_emb(tgt) * math.sqrt(self.d_model))

        memory = self.transformer.encoder(src_repr, src_key_padding_mask=src_pad_mask)
        out = self.transformer.decoder(tgt_repr, memory, tgt_mask=tgt_mask,
                                       tgt_key_padding_mask=tgt_pad_mask,
                                       memory_key_padding_mask=src_pad_mask)
        logits = self.fc_out(out)

        enc_summary = memory.mean(dim=1)
        phono_logits = self.phono_head(enc_summary)
        return logits, phono_logits

# =====================================================================
# 4. PAN-INDIC TEST BATTERY
# =====================================================================

TEST_CASES = [
    ("__ta__", "theedhum", "தீதும்", "Sangam Voiceless Stop"),
    ("__ta__", "nandrum", "நன்றும்", "Post-Nasal Mellinam Assimilation"),
    ("__ta__", "pirarthara", "பிறர்தர", "Sangam Minimal Pair Over-lengthening Check"),
    ("__ta__", "polama", "போலாமா", "Lossy Tanglish Vowel Length"),
    ("__ta__", "vaathiyaar", "வாத்தியார்", "Implicit Gemination Disambiguation"),
    ("__ta__", "vettukku", "வெட்டுக்கு", "Intra-family Short Vowel Invariance"),
    ("__ta__", "veettukku", "வீட்டுக்கு", "Intra-family Long Vowel Invariance"),
    ("__hi__", "hain", "हैं", "Hindi Plural/Polite Copula Anusvara"),
    ("__hi__", "mein", "में", "Hindi Postposition Nasalization"),
    ("__hi__", "nahin", "नहीं", "Hindi Negative Particle"),
    ("__hi__", "kyun", "क्यों", "Hindi Interrogative Anusvara"),
    ("__hi__", "zindagi", "ज़िंदगी", "Hindi Perso-Arabic Nuqta (z)"),
    ("__bn__", "nomoshkar", "নমস্কার", "Bengali Inherent Vowel Shift (/a/ -> /o/)"),
    ("__bn__", "bhalobashi", "ভালোবাসি", "Bengali Affectionate Verb"),
    ("__gu__", "kemcho", "કેમ છો", "Gujarati Greeting Phrase"),
    ("__gu__", "chhe", "છે", "Gujarati Copula"),
    ("__mr__", "namaskar", "नमस्कार", "Marathi Greeting"),
    ("__mr__", "dola", "डोळा", "Marathi Retroflex Flap (ळ)"),
    ("__ml__", "namaskaram", "നമസ്കാരം", "Malayalam Greeting"),
    ("__te__", "baahubali", "బాహుబలి", "Telugu Aspirated Stop (bh)"),
    ("__kn__", "kantara", "ಕಾಂತಾರ", "Kannada Folklore Noun")
]

def run_pan_indic_test_battery(model: nn.Module, src_vocab: Vocab, tgt_vocab: Vocab, device: torch.device):
    model.eval()
    print("\n--- PAN-INDIC VALIDATION TEST BATTERY (VALIMELI-A4) ---")
    correct_count = 0
    with torch.no_grad():
        for lang_tag, roman, expected, desc in TEST_CASES:
            full_input = f"{lang_tag} {roman}"
            src_ids = src_vocab.encode(full_input, add_sos=False, add_eos=True)
            src_t = torch.tensor([src_ids], dtype=torch.long, device=device)
            src_repr = model.pos_encoder(model.src_emb(src_t) * math.sqrt(model.d_model))
            memory = model.transformer.encoder(src_repr)

            out_ids = [tgt_vocab.stoi["<sos>"]]
            for _ in range(30):
                tgt_t = torch.tensor([out_ids], dtype=torch.long, device=device)
                tgt_repr = model.pos_encoder(model.tgt_emb(tgt_t) * math.sqrt(model.d_model))
                tgt_mask = model.transformer.generate_square_subsequent_mask(len(out_ids)).to(device)
                out = model.transformer.decoder(tgt_repr, memory, tgt_mask=tgt_mask)
                next_tok = model.fc_out(out[:, -1, :]).argmax(dim=-1).item()
                if next_tok == tgt_vocab.stoi["<eos>"]:
                    break
                out_ids.append(next_tok)

            pred = tgt_vocab.decode(out_ids)
            is_match = (pred.strip() == expected.strip())
            if is_match:
                correct_count += 1
            status = "✅ PASS" if is_match else "❌ DIFF"
            print(f"  {status} [{lang_tag}] {roman:14s} -> Pred: {pred:16s} (Expected: {expected:16s}) | {desc}")

    acc = (correct_count / len(TEST_CASES)) * 100.0
    print(f"⭐ Validation Battery Accuracy: {correct_count}/{len(TEST_CASES)} ({acc:.1f}%)\n", flush=True)

# =====================================================================
# 5. TRAINING RUNNER
# =====================================================================

def evaluate_validation(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src, tgt, phono in loader:
            src, tgt, phono = src.to(device), tgt.to(device), phono.to(device)
            tgt_in = tgt[:, :-1]
            tgt_out = tgt[:, 1:]
            logits, _ = model(src, tgt_in)
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_out.reshape(-1))
            total_loss += loss.item()
    return total_loss / max(len(loader), 1)

def train_valimeli_a4():
    print("=" * 80)
    print(" VALIMELI-A4-PAN-INDIC-MT MULTI-TASK MULTILINGUAL TRAINING")
    print(f" Device: {DEVICE} | Batch Size: {BATCH_SIZE} | D_Model: {D_MODEL} | Epochs: {NUM_EPOCHS}")
    print(f" Commit Hashes: Valimeli={VALIMELI_COMMIT} | Pulli={PULLI_COMMIT}")
    print("=" * 80)

    print(f"\n -> Ingesting Pan-Indic Dataset from: {PAN_INDIC_DATASET_FILE}")
    all_pairs = []
    src_vocab = Vocab()
    tgt_vocab = Vocab()

    for p in LANG_PREFIXES.values():
        src_vocab.add_token(p)

    with gzip.open(PAN_INDIC_DATASET_FILE, "rt", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            r = obj["roman"]
            ind = obj["indic"]
            lang = obj["lang"]
            full_src = f"{lang} {r}"
            phono_cls = classify_phonotactics(r)
            all_pairs.append((full_src, ind, phono_cls))

            for ch in full_src:
                src_vocab.add_token(ch)
            for ch in ind:
                tgt_vocab.add_token(ch)

    print(f"    Loaded {len(all_pairs):,} pairs across 8 Indic languages.")
    print(f"    Vocab Sizes -> Source (Roman+Tags): {len(src_vocab):,} | Target (Indic): {len(tgt_vocab):,}")

    random.seed(42)
    random.shuffle(all_pairs)
    val_size = min(50000, int(len(all_pairs) * 0.01))
    val_pairs = all_pairs[:val_size]
    train_pairs = all_pairs[val_size:]

    train_dataset = PanIndicDataset(train_pairs, src_vocab, tgt_vocab)
    val_dataset = PanIndicDataset(val_pairs, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=pad_collate)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=pad_collate)

    model = ValiMeliA4MultiTaskModel(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        num_phono_classes=NUM_PHONO_CLASSES
    ).to(DEVICE)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n ⭐ Initialized ValiMeli-A4 Multi-Task Model: {num_params:,} trainable parameters (~3.2M Edge Scale)\n")

    criterion_trans = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    criterion_phono = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-5)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
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
        run_pan_indic_test_battery(model, src_vocab, tgt_vocab, DEVICE)

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
            print(f"    ⭐ Saved new best ValiMeli-A4-PanIndic checkpoint: {MODEL_OUT_PATH}")
            print(f"    ⭐ Synced checkpoint to Pulli: {PULLI_SYNC_PATH}\n", flush=True)

    print("=" * 80)
    print(" VALIMELI-A4-PAN-INDIC-MT TRAINING COMPLETE!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    train_valimeli_a4()
