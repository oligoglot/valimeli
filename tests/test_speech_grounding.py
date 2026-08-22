#!/usr/bin/env python3
"""
Test Suite for Project ValiMeli Speech-Augmented Transliteration (Phase 1-3)
Validates spectrogram extraction, acoustic encoder, contrastive loss, and allophony probing.
"""

import os
import sys
import unittest
import torch
import torch.nn.functional as F

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(WORKSPACE_DIR, "src"))

from speech.acoustic_features import LogMelSpectrogramExtractor, SyntheticTamilAudioAcoustics
from speech.speech_contrastive_model import AudioSpectrogramEncoder, InfoNCEContrastiveLoss, AcousticAllophonyProbe
from speech.fetch_wikimedia_speech import extract_tamil_word_from_filename

def test_extract_tamil_word_from_filename():
    assert extract_tamil_word_from_filename("File:Ta-தம்பி.ogg") == "தம்பி"
    assert extract_tamil_word_from_filename("File:Ta-படம்.ogg") == "படம்"
    assert extract_tamil_word_from_filename("File:Ta-பக்கம்.ogg") == "பக்கம்"
    assert extract_tamil_word_from_filename("File:LL-Q5885 (tam)-User-பந்து.wav") == "பந்து"
    assert extract_tamil_word_from_filename("File:English-word.ogg") is None

def test_log_mel_spectrogram_extractor():
    extractor = LogMelSpectrogramExtractor(sample_rate=16000, n_mels=80)
    waveform = torch.randn(2, 16000) # 2 audio clips of 1 second each
    spec = extractor(waveform)
    assert spec.dim() == 3
    assert spec.size(0) == 2
    assert spec.size(1) == 80 # 80 mel channels
    assert spec.size(2) > 90  # ~100 time frames for 1 sec at 10ms hop

def test_synthetic_tamil_audio_generation():
    audio_init = SyntheticTamilAudioAcoustics.synthesize_utterance("initial", duration_sec=0.5)
    audio_gem = SyntheticTamilAudioAcoustics.synthesize_utterance("geminate", duration_sec=0.5)
    audio_nasal = SyntheticTamilAudioAcoustics.synthesize_utterance("post_nasal", duration_sec=0.5)
    audio_inter = SyntheticTamilAudioAcoustics.synthesize_utterance("intervocalic", duration_sec=0.5)
    
    assert audio_init.size(-1) == 8000
    assert audio_gem.size(-1) == 8000
    assert audio_nasal.size(-1) == 8000
    assert audio_inter.size(-1) == 8000

def test_acoustic_encoder_and_contrastive_alignment():
    extractor = LogMelSpectrogramExtractor()
    encoder = AudioSpectrogramEncoder(in_channels=80, d_model=256)
    loss_fn = InfoNCEContrastiveLoss(temperature=0.07)
    
    # 4 distinct synthetic utterances
    contexts = ["initial", "geminate", "post_nasal", "intervocalic"]
    waveforms = torch.cat([SyntheticTamilAudioAcoustics.synthesize_utterance(c, duration_sec=0.5) for c in contexts], dim=0)
    
    specs = extractor(waveforms)
    z_audio = encoder(specs)
    
    assert z_audio.shape == (4, 256)
    assert torch.allclose(torch.norm(z_audio, dim=-1), torch.ones(4), atol=1e-3)
    
    z_text = F.normalize(torch.randn(4, 256), dim=-1)
    loss = loss_fn(z_audio, z_text)
    assert loss.item() > 0.0

def test_acoustic_allophony_probing():
    probe = AcousticAllophonyProbe(d_model=256, num_classes=4)
    z_audio = torch.randn(4, 256)
    logits = probe(z_audio)
    assert logits.shape == (4, 4)
    probs = F.softmax(logits, dim=-1)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(4), atol=1e-4)

if __name__ == "__main__":
    pytest.main([__file__])
