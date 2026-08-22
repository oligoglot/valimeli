#!/usr/bin/env python3
"""
Project ValiMeli — Joint Acoustic-Orthographic Contrastive Alignment Engine (Phase 3)
Aligns audio spectrogram features with ValiMeli phonology embeddings via InfoNCE loss.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict

class AudioSpectrogramEncoder(nn.Module):
    """
    Lightweight 2D-CNN + Conformer/Transformer Encoder for 80-channel Log-Mel Spectrograms.
    Input: (Batch, 80, Time_Frames) -> Output: (Batch, d_model=256)
    """
    def __init__(self, in_channels: int = 80, d_model: int = 256, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        # 2D-CNN subsampling stem (stride=2, reduces time & freq dimension by 4x)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()
        )
        
        # Linear projection to d_model
        # After 2 strides of 2 on 80 mels: 80 -> 40 -> 20; 64 * 20 = 1280
        self.proj = nn.Linear(64 * 20, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.projection_head = nn.Linear(d_model, d_model)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        spec: (Batch, 80, Time)
        """
        x = spec.unsqueeze(1)  # (Batch, 1, 80, Time)
        x = self.conv_stem(x)  # (Batch, 64, 20, Time // 4)
        
        batch, c, f, t = x.size()
        x = x.permute(0, 3, 1, 2).reshape(batch, t, c * f) # (Batch, Time // 4, 1280)
        x = self.proj(x)       # (Batch, Time // 4, d_model)
        
        feat = self.transformer(x) # (Batch, Time // 4, d_model)
        pooled = feat.mean(dim=1)  # Global temporal average pooling
        out = self.out_norm(pooled)
        z_a = F.normalize(self.projection_head(out), dim=-1)
        return z_a

class InfoNCEContrastiveLoss(nn.Module):
    """
    Symmetric InfoNCE loss aligning Acoustic representations (z_a) with Text representations (z_t).
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_audio: torch.Tensor, z_text: torch.Tensor) -> torch.Tensor:
        # z_audio: (B, D), z_text: (B, D)
        sim = torch.matmul(z_audio, z_text.T) / self.temperature # (B, B)
        labels = torch.arange(z_audio.size(0), device=z_audio.device)
        
        loss_a2t = F.cross_entropy(sim, labels)
        loss_t2a = F.cross_entropy(sim.T, labels)
        return (loss_a2t + loss_t2a) / 2.0

class AcousticAllophonyProbe(nn.Module):
    """
    Diagnostic probing classifier: Predicts stop allophone class ([INIT], [GEM], [NASAL], [INTER])
    directly from the acoustic embedding to verify phonetic separability.
    """
    def __init__(self, d_model: int = 256, num_classes: int = 4):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, z_a: torch.Tensor) -> torch.Tensor:
        return self.classifier(z_a)

if __name__ == "__main__":
    audio_encoder = AudioSpectrogramEncoder()
    loss_fn = InfoNCEContrastiveLoss()
    
    dummy_spec = torch.randn(8, 80, 100) # 8 samples, 80 mels, 100 frames (~1 sec)
    dummy_text_emb = F.normalize(torch.randn(8, 256), dim=-1)
    
    z_a = audio_encoder(dummy_spec)
    loss = loss_fn(z_a, dummy_text_emb)
    
    probe = AcousticAllophonyProbe()
    probe_logits = probe(z_a)
    
    print(f" -> Acoustic Encoder Output: z_a shape = {z_a.shape}")
    print(f" -> InfoNCE Contrastive Loss = {loss.item():.4f}")
    print(f" -> Acoustic Allophony Probing Logits = {probe_logits.shape} (8 samples x 4 classes)")
