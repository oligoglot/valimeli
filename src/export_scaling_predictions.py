#!/usr/bin/env python3
"""
Project ValiMeli — Export Scaling Predictions & Paired McNemar Contingencies
==========================================================================
Re-decodes all surviving scaling checkpoints on the full test sets:
  - 250k BiGRU (Tamil & Malayalam)
  - 250k Transformer (Tamil & Malayalam)
  - 500k Transformer (Tamil & Malayalam)
  - 1.0M Bilingual Transformer (Tamil & Malayalam)

Outputs:
  - 16 JSONL instance prediction files in artifacts/predictions/
  - artifacts/scaling_mcnemar_contingencies.json
"""

import os
import sys
import json
import math
import time
from typing import List, Tuple, Dict, Any

import torch
import torch.nn as nn
from scipy.stats import binomtest

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
DATA_DIR = os.path.join(SCRATCH_DIR, "data")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
PRED_DIR = os.path.join(ARTIFACTS_DIR, "predictions")

os.makedirs(PRED_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# Vocabulary stub for unpickling
# ---------------------------------------------------------------------
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
        tokens = [self.char2idx[self.SOS_TOKEN]]
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
            if char in {self.PAD_TOKEN, self.SOS_TOKEN}: continue
            if char == self.EOS_TOKEN: break
            chars.append(char)
        return "".join(chars)

import __main__
__main__.CharVocabulary = CharVocabulary

# ---------------------------------------------------------------------
# Architecture 1: BiGRU Seq2Seq with Attention (250k BiGRU)
# ---------------------------------------------------------------------
class Seq2SeqAttention(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.src_embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.enc_hidden_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn = nn.Linear(hidden_dim + hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        self.decoder = nn.GRU(embed_dim + hidden_dim * 2, hidden_dim, batch_first=True)
        self.out_proj = nn.Linear(hidden_dim, tgt_vocab_size)

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
                        if idx == eos_idx: finished[b] = True
                        else: decoded_batch[b].append(idx)
                if all(finished): break
                dec_input = top1.unsqueeze(1)
        return decoded_batch

# ---------------------------------------------------------------------
# Architecture 2: 11M Transformer Seq2Seq (250k & 500k Transformer)
# ---------------------------------------------------------------------
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
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

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
                out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_mask)
                next_tokens = self.out_proj(out[:, -1]).argmax(dim=-1)
                for b in range(batch_size):
                    if not finished[b]:
                        tok = next_tokens[b].item()
                        if tok == eos_idx: finished[b] = True
                        else: decoded_batch[b].append(tok)
                if all(finished): break
                tgt_indices = torch.cat((tgt_indices, next_tokens.unsqueeze(1)), dim=1)
        return decoded_batch

# ---------------------------------------------------------------------
# Architecture 3: MultiTask Transformer Seq2Seq (1.0M Bilingual)
# ---------------------------------------------------------------------
class MultiTaskTransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, num_phono_classes: int = 6,
                 d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model, max_len=128)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)
        self.aux_phono_proj = nn.Linear(d_model, num_phono_classes)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

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
                out = self.decoder(tgt_emb, memory, tgt_mask=causal_mask, memory_key_padding_mask=src_mask)
                next_tokens = self.out_proj(out[:, -1]).argmax(dim=-1)
                for b in range(batch_size):
                    if not finished[b]:
                        tok = next_tokens[b].item()
                        if tok == eos_idx: finished[b] = True
                        else: decoded_batch[b].append(tok)
                if all(finished): break
                tgt_indices = torch.cat((tgt_indices, next_tokens.unsqueeze(1)), dim=1)
        return decoded_batch

# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
TAGS = ['\ue001', '\ue002', '\ue003', '\ue004', '\ue005', '\ue006']
def strip_pua(s: str) -> str:
    for t in TAGS: s = s.replace(t, '')
    return s.strip().lower()

def load_test_split(file_path: str) -> List[Dict[str, str]]:
    items = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def decode_model(model, vocab_src, vocab_tgt, test_items, device, prefix="") -> List[str]:
    batch_size = 256
    preds = []
    for i in range(0, len(test_items), batch_size):
        chunk = test_items[i:i+batch_size]
        enc_list = []
        for item in chunk:
            r = prefix + item.get("english word", "").strip().lower()
            if hasattr(vocab_src, "encode"):
                enc = vocab_src.encode(r, is_source=True)
            else:
                enc = [vocab_src.char2idx[vocab_src.SOS_TOKEN]] + [vocab_src.char2idx.get(c, vocab_src.char2idx[vocab_src.UNK_TOKEN]) for c in r] + [vocab_src.char2idx[vocab_src.EOS_TOKEN]]
            enc_list.append(enc)
            
        max_src = max(len(enc) for enc in enc_list)
        src_t = torch.zeros(len(chunk), max_src, dtype=torch.long)
        for j, enc in enumerate(enc_list):
            src_t[j, :len(enc)] = torch.tensor(enc, dtype=torch.long)
            
        src_t = src_t.to(device)
        dec = model.greedy_decode(src_t)
        for j in range(len(dec)):
            pred_raw = vocab_tgt.decode(dec[j])
            preds.append(strip_pua(pred_raw))
    return preds

def save_prediction_jsonl(filepath: str, test_items: List[Dict[str, str]], preds: List[str]):
    with open(filepath, "w", encoding="utf-8") as f:
        for idx, (item, pred) in enumerate(zip(test_items, preds)):
            gold = strip_pua(item.get("native word", "").strip())
            roman = item.get("english word", "").strip().lower()
            rec = {
                "id": idx,
                "input": roman,
                "gold": gold,
                "target": gold,
                "pred": pred,
                "correct": (pred == gold),
                "source_tag": item.get("source", "unknown")
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

# ---------------------------------------------------------------------
# Main Execution Harness
# ---------------------------------------------------------------------
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Project ValiMeli: Exporting Scaling Predictions on {device}...")
    
    tam_test_file = os.path.join(DATA_DIR, "extracted_tam", "tam_test.json")
    mal_test_file = os.path.join(DATA_DIR, "extracted_mal", "mal_test.json")
    
    tam_items = load_test_split(tam_test_file)
    mal_items = load_test_split(mal_test_file)
    print(f"Loaded {len(tam_items)} Tamil items, {len(mal_items)} Malayalam items.")
    
    jobs = [
        # (Tier Name, Lang, TestItems, Prefix, ArchClass, A0_path, A1_path, A0_out, A1_out)
        (
            "250k_bigru", "tam", tam_items, "",
            Seq2SeqAttention,
            os.path.join(RUNS_DIR, "tam_A0_en-indic", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "tam_A1_en-indic", "checkpoint_best.pt"),
            "predictions_tam_A0_250k_bigru.jsonl", "predictions_tam_A1_250k_bigru.jsonl"
        ),
        (
            "250k_bigru", "mal", mal_items, "",
            Seq2SeqAttention,
            os.path.join(RUNS_DIR, "mal_A0_en-indic", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "mal_A1_en-indic", "checkpoint_best.pt"),
            "predictions_mal_A0_250k_bigru.jsonl", "predictions_mal_A1_250k_bigru.jsonl"
        ),
        (
            "250k_transformer", "tam", tam_items, "",
            TransformerSeq2Seq,
            os.path.join(RUNS_DIR, "tam_A0_en-indic_transformer", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "tam_A1_en-indic_transformer", "checkpoint_best.pt"),
            "predictions_tam_A0_250k_transformer.jsonl", "predictions_tam_A1_250k_transformer.jsonl"
        ),
        (
            "250k_transformer", "mal", mal_items, "",
            TransformerSeq2Seq,
            os.path.join(RUNS_DIR, "mal_A0_en-indic_transformer", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "mal_A1_en-indic_transformer", "checkpoint_best.pt"),
            "predictions_mal_A0_250k_transformer.jsonl", "predictions_mal_A1_250k_transformer.jsonl"
        ),
        (
            "500k_transformer", "tam", tam_items, "",
            TransformerSeq2Seq,
            os.path.join(RUNS_DIR, "tam_A0_en-indic_indicxlit_rep", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "tam_A1_en-indic_indicxlit_rep", "checkpoint_best.pt"),
            "predictions_tam_A0_500k_transformer.jsonl", "predictions_tam_A1_500k_transformer.jsonl"
        ),
        (
            "500k_transformer", "mal", mal_items, "",
            TransformerSeq2Seq,
            os.path.join(RUNS_DIR, "mal_A0_en-indic_indicxlit_rep", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "mal_A1_en-indic_indicxlit_rep", "checkpoint_best.pt"),
            "predictions_mal_A0_500k_transformer.jsonl", "predictions_mal_A1_500k_transformer.jsonl"
        ),
        (
            "1.0M_bilingual", "tam", tam_items, "__ta__",
            MultiTaskTransformerSeq2Seq,
            os.path.join(RUNS_DIR, "bilingual_Bilingual-A0", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "bilingual_Bilingual-A1-MT", "checkpoint_best.pt"),
            "predictions_tam_A0_1M_bilingual.jsonl", "predictions_tam_A1-MT_1M_bilingual.jsonl"
        ),
        (
            "1.0M_bilingual", "mal", mal_items, "__ml__",
            MultiTaskTransformerSeq2Seq,
            os.path.join(RUNS_DIR, "bilingual_Bilingual-A0", "checkpoint_best.pt"),
            os.path.join(RUNS_DIR, "bilingual_Bilingual-A1-MT", "checkpoint_best.pt"),
            "predictions_mal_A0_1M_bilingual.jsonl", "predictions_mal_A1-MT_1M_bilingual.jsonl"
        ),
    ]
    
    contingency_summary = {}
    
    for tier, lang, items, prefix, ArchClass, a0_path, a1_path, a0_out, a1_out in jobs:
        t0 = time.time()
        print(f"\n--- Processing {tier} ({lang.upper()}) ---")
        
        out0_path = os.path.join(PRED_DIR, a0_out)
        out1_path = os.path.join(PRED_DIR, a1_out)
        
        if os.path.exists(out0_path) and os.path.exists(out1_path) and "1.0M" not in tier:
            print(f"  Reusing existing predictions from {a0_out} and {a1_out}...")
            with open(out0_path, "r", encoding="utf-8") as f:
                preds0 = [json.loads(line)["pred"] for line in f if line.strip()]
            with open(out1_path, "r", encoding="utf-8") as f:
                preds1 = [json.loads(line)["pred"] for line in f if line.strip()]
        else:
            # 1. Load A0
            ckpt0 = torch.load(a0_path, map_location=device, weights_only=False)
            m0 = ArchClass(ckpt0["vocab_src"].num_chars, ckpt0["vocab_tgt"].num_chars).to(device)
            m0.load_state_dict(ckpt0["model_state_dict"])
            m0.eval()
            preds0 = decode_model(m0, ckpt0["vocab_src"], ckpt0["vocab_tgt"], items, device, prefix=prefix)
            del m0, ckpt0
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
            
            # 2. Load A1
            ckpt1 = torch.load(a1_path, map_location=device, weights_only=False)
            m1 = ArchClass(ckpt1["vocab_src"].num_chars, ckpt1["vocab_tgt"].num_chars).to(device)
            m1.load_state_dict(ckpt1["model_state_dict"])
            m1.eval()
            preds1 = decode_model(m1, ckpt1["vocab_src"], ckpt1["vocab_tgt"], items, device, prefix=prefix)
            del m1, ckpt1
            torch.mps.empty_cache() if torch.backends.mps.is_available() else None
            
            # 3. Save JSONLs
            save_prediction_jsonl(out0_path, items, preds0)
            save_prediction_jsonl(out1_path, items, preds1)
        
        # 4. Compute exact match & McNemar
        golds = [strip_pua(it.get("native word", "").strip()) for it in items]
        m_a0 = [p == g for p, g in zip(preds0, golds)]
        m_a1 = [p == g for p, g in zip(preds1, golds)]
        
        em0 = (sum(m_a0) / len(m_a0)) * 100.0
        em1 = (sum(m_a1) / len(m_a1)) * 100.0
        delta = em1 - em0
        
        n00 = sum(not a0 and not a1 for a0, a1 in zip(m_a0, m_a1))
        n01 = sum(not a0 and a1 for a0, a1 in zip(m_a0, m_a1)) # b (gain)
        n10 = sum(a0 and not a1 for a0, a1 in zip(m_a0, m_a1)) # c (loss)
        n11 = sum(a0 and a1 for a0, a1 in zip(m_a0, m_a1))
        
        p_val = binomtest(min(n01, n10), n01 + n10, 0.5, alternative="two-sided").pvalue
        elapsed = time.time() - t0
        
        print(f"  A0 EM: {em0:.2f}% | A1 EM: {em1:.2f}% | Delta: {delta:+.2f} pp")
        print(f"  Contingency: n00={n00}, b={n01}, c={n10}, n11={n11} | McNemar p={p_val:.4e} ({elapsed:.1f}s)")
        
        key = f"{tier}_{lang}"
        contingency_summary[key] = {
            "tier": tier,
            "lang": lang,
            "total_items": len(items),
            "em_base": em0,
            "em_phon": em1,
            "delta_pp": delta,
            "n00": n00,
            "b_gain": n01,
            "c_loss": n10,
            "n11": n11,
            "mcnemar_p_value": p_val,
            "file_base": a0_out,
            "file_phon": a1_out
        }
        
    summary_path = os.path.join(ARTIFACTS_DIR, "scaling_mcnemar_contingencies.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(contingency_summary, f, indent=2)
    print(f"\nSaved all 16 JSONL files to {PRED_DIR} and summary to {summary_path}")

if __name__ == "__main__":
    main()
