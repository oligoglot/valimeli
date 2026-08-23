#!/usr/bin/env python3
"""
Project ValiMeli — Standalone Multilingual Inference Engine
Easily transliterate words and sentences across 8 Indic languages using the pre-trained 3.2M model.
Ready for downstream NLP, Theedhum Nandrum, and the Pulli epigraphy project.

Usage:
    from src.valimeli_transliterate import ValiMeliEngine
    engine = ValiMeliEngine()
    print(engine.transliterate("thambi", lang="tam"))      # -> 'தம்பி'
    print(engine.transliterate("thampi", lang="mal"))      # -> 'തമ്പി'
    print(engine.transliterate("namaste", lang="hin"))     # -> 'नमस्ते'
"""

import os
import re
import math
from typing import List, Optional
import torch
import torch.nn as nn

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(WORKSPACE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "valimeli_multilingual_3.2m.pt")

LANG_PREFIXES = {
    "tam": "__ta__", "tamil": "__ta__", "ta": "__ta__",
    "mal": "__ml__", "malayalam": "__ml__", "ml": "__ml__",
    "tel": "__te__", "telugu": "__te__", "te": "__te__",
    "kan": "__kn__", "kannada": "__kn__", "kn": "__kn__",
    "hin": "__hi__", "hindi": "__hi__", "hi": "__hi__",
    "ben": "__bn__", "bengali": "__bn__", "bn": "__bn__",
    "guj": "__gu__", "gujarati": "__gu__", "gu": "__gu__",
    "mar": "__mr__", "marathi": "__mr__", "mr": "__mr__",
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

class MultilingualTransformerSeq2SeqInference(nn.Module):
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, d_model: int = 256, nhead: int = 4, num_layers: int = 6, dim_feedforward: int = 1024):
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

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)

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

class ValiMeliEngine:
    def __init__(self, model_path: Optional[str] = None, device: Optional[torch.device] = None):
        if model_path is None:
            # Fallback to scratch if models/ copy not created yet
            if os.path.exists(MODEL_PATH):
                model_path = MODEL_PATH
            else:
                model_path = os.path.join(WORKSPACE_DIR, "scratch", "valimeli", "runs", "multilingual_scaling_best.pt")
                
        if device is None:
            device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            
        self.device = device
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
        self.src_vocab = ckpt["src_vocab"]
        self.tgt_vocab = ckpt["tgt_vocab"]
        
        self.model = MultilingualTransformerSeq2SeqInference(
            self.src_vocab.n, self.tgt_vocab.n,
            d_model=256, nhead=4, num_layers=6, dim_feedforward=1024
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    def transliterate(self, word: str, lang: str = "tam") -> str:
        lang_tag = LANG_PREFIXES.get(lang.lower(), "__ta__")
        clean_w = word.strip().lower()[:35]
        tokens = [lang_tag] + list(clean_w)
        encoded = self.src_vocab.encode(tokens)
        tensor = torch.tensor([encoded], dtype=torch.long, device=self.device)
        
        decoded_indices = self.model.greedy_decode(tensor)
        return self.tgt_vocab.decode(decoded_indices[0])

    def transliterate_sentence(self, sentence: str, lang: str = "tam") -> str:
        words = re.findall(r"[a-zA-Z]+|[^\s\w]+", sentence)
        out = []
        for w in words:
            if re.match(r"^[a-zA-Z]+$", w):
                out.append(self.transliterate(w, lang=lang))
            else:
                out.append(w)
        return " ".join(out)

if __name__ == "__main__":
    engine = ValiMeliEngine()
    print("ValiMeli 3.2M Multilingual Engine Loaded Successfully!")
    for l, word in [("tam", "thambi"), ("mal", "thampi"), ("tel", "thammudu"), ("kan", "tamma"), ("hin", "namaste")]:
        print(f"[{l.upper()}] '{word}' -> '{engine.transliterate(word, lang=l)}'")
