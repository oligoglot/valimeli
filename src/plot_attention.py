#!/usr/bin/env python3
"""
Project ValiMeli — Attention Alignment Heatmap Generator (v3.4)
Extracts Bahdanau attention weights from trained Seq2Seq models to visualize
how the attention mechanism aligns with phonological context tags ([INIT], [GEM], [NASAL], [INTER]).
Features canonical, phonologically unambiguous Tamil examples: தம்பி, பந்து, பக்கம், படம்.
"""

import os
import sys
import torch
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")

os.environ["MPLCONFIGDIR"] = os.path.join(SCRATCH_DIR, "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'AppleGothic', 'Noto Sans Tamil', 'Noto Sans Malayalam', 'DejaVu Sans', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.join(WORKSPACE_DIR, "src"))
import importlib.util
spec = importlib.util.spec_from_file_location("valimeli_benchmark", os.path.join(os.path.dirname(__file__), "valimeli-benchmark.py"))
vb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vb)

import __main__
__main__.CharVocabulary = vb.CharVocabulary

TAG_DISPLAY = {
    vb.TAG_INITIAL: "[INIT]",
    vb.TAG_GEMINATE: "[GEM]",
    vb.TAG_POST_NASAL: "[NASAL]",
    vb.TAG_INTERVOCALIC: "[INTER]",
    vb.TAG_DEFAULT: "[DEF]"
}

def format_tokens(token_list):
    return [TAG_DISPLAY.get(t, t) for t in token_list]

def render_attention_heatmap(word: str, lang: str = "tam", direction: str = "indic-en", output_name: str = "attention_heatmap.png"):
    device = torch.device("cpu")
    run_dir = os.path.join(RUNS_DIR, f"{lang}_A1_{direction}")
    checkpoint_path = os.path.join(run_dir, "checkpoint_best.pt")
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ Checkpoint not found at: {checkpoint_path}. Train Arm A1 first.")
        return None
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_src = checkpoint["vocab_src"]
    vocab_tgt = checkpoint["vocab_tgt"]
    
    model = vb.Seq2SeqAttention(vocab_src.num_chars, vocab_tgt.num_chars, embed_dim=128, hidden_dim=256)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    if lang == "tam":
        tagged_src = vb.apply_tamil_phonology_tags(word) if direction == "indic-en" else word
    else:
        tagged_src = vb.apply_malayalam_phonology_tags(word) if direction == "indic-en" else word
        
    src_tokens = list(tagged_src)
    src_encoded = vocab_src.encode(tagged_src)
    src_tensor = torch.tensor(src_encoded, dtype=torch.long).unsqueeze(0)
    src_len = src_tensor.size(1)
    
    with torch.no_grad():
        src_emb = model.src_embed(src_tensor)
        enc_outputs, enc_hidden = model.encoder(src_emb)
        
        combined_h = torch.cat((enc_hidden[0], enc_hidden[1]), dim=-1)
        dec_hidden = torch.tanh(model.enc_hidden_proj(combined_h)).unsqueeze(0)
        
        dec_input = torch.tensor([[vocab_tgt.char2idx[vocab_tgt.SOS_TOKEN]]], dtype=torch.long)
        
        predicted_chars = []
        attn_matrix = []
        
        for _ in range(25):
            dec_emb = model.tgt_embed(dec_input)
            dec_h_exp = dec_hidden.squeeze(0).unsqueeze(1).repeat(1, src_len, 1)
            attn_energy = torch.tanh(model.attn(torch.cat((dec_h_exp, enc_outputs), dim=-1)))
            attn_scores = model.v(attn_energy).squeeze(-1)
            attn_weights = torch.softmax(attn_scores, dim=-1)
            
            attn_matrix.append(attn_weights.squeeze(0).numpy())
            
            context = torch.bmm(attn_weights.unsqueeze(1), enc_outputs)
            dec_out, dec_hidden = model.decoder(torch.cat((dec_emb, context), dim=-1), dec_hidden)
            
            logits = model.out_proj(dec_out.squeeze(1))
            top1 = logits.argmax(dim=-1)
            char = vocab_tgt.idx2char.get(top1.item(), "")
            
            if char == vocab_tgt.EOS_TOKEN:
                break
            predicted_chars.append(char)
            dec_input = top1.unsqueeze(1)
            
    if not predicted_chars:
        predicted_chars = ["?"]
        attn_matrix = [np.zeros(src_len)]
        
    attn_matrix = np.array(attn_matrix)
    x_labels = format_tokens(["<SOS>"] + src_tokens + ["<EOS>"])
    y_labels = predicted_chars
    
    fig, ax = plt.subplots(figsize=(max(len(x_labels) * 0.75 + 2, 8), max(len(y_labels) * 0.5 + 2, 4)))
    sns.heatmap(
        attn_matrix,
        xticklabels=x_labels,
        yticklabels=y_labels,
        annot=True,
        fmt=".2f",
        cmap="Purples",
        cbar=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={'label': 'Attention Weight (α)'}
    )
    
    pred_word = "".join(predicted_chars)
    ax.set_title(f"ValiMeli Attention Alignment: Input '{word}' → Predicted '{pred_word}'", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Phonology-Tagged Input Tokens", fontsize=11, labelpad=8)
    ax.set_ylabel("Predicted Transliteration Output", fontsize=11, labelpad=8)
    
    plt.xticks(rotation=45, ha='right', fontsize=10, fontweight='bold')
    plt.yticks(rotation=0, fontsize=10, fontweight='bold')
    
    for i, lab in enumerate(x_labels):
        if any(tag in lab for tag in TAG_DISPLAY.values()):
            ax.get_xticklabels()[i].set_color("#7B1FA2")
            
    plt.tight_layout()
    out_path = os.path.join(SCRATCH_DIR, output_name)
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f" -> Rendered attention heatmap: {out_path}")
    return out_path

if __name__ == "__main__":
    render_attention_heatmap("தம்பி", lang="tam", direction="indic-en", output_name="attn_tamil_thambi.png")
    render_attention_heatmap("பந்து", lang="tam", direction="indic-en", output_name="attn_tamil_pandhu.png")
    render_attention_heatmap("பக்கம்", lang="tam", direction="indic-en", output_name="attn_tamil_pakkam.png")
    render_attention_heatmap("படம்", lang="tam", direction="indic-en", output_name="attn_tamil_padam.png")
