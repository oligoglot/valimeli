#!/usr/bin/env python3
"""
Project ValiMeli — Publication-Quality Plotting Engine (v5 - 3-Arm Comparative Edition)
Generates high-resolution comparative figures comparing:
- Baseline (A0: Character)
- ValiMeli (A1: Phonology-Aware)
- Morphology-Aware (A2: arXiv:2508.08424)

Figures:
1. Exact Match Word Accuracy (Top-1 EM %)
2. Character Error Rate (CER %)
3. Stop-Voicing Accuracy (SVA %)
4. Training Convergence Curves
"""

import os
import sys
import json
import torch
import numpy as np

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")
DOCS_DIR = os.path.join(WORKSPACE_DIR, "docs")

os.environ["MPLCONFIGDIR"] = os.path.join(SCRATCH_DIR, "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', font='DejaVu Sans')
CHART_DPI = 300

def load_all_results():
    experiments = [
        ('Tamil', 'tam', 'A0', 'Baseline', 'indic-en'),
        ('Tamil', 'tam', 'A1', 'Phonology-Aware', 'indic-en'),
        ('Tamil', 'tam', 'A2', 'Morphology-Aware', 'indic-en'),
        ('Tamil', 'tam', 'A0', 'Baseline', 'en-indic'),
        ('Tamil', 'tam', 'A1', 'Phonology-Aware', 'en-indic'),
        ('Tamil', 'tam', 'A2', 'Morphology-Aware', 'en-indic'),
        ('Malayalam', 'mal', 'A0', 'Baseline', 'indic-en'),
        ('Malayalam', 'mal', 'A1', 'Phonology-Aware', 'indic-en'),
        ('Malayalam', 'mal', 'A2', 'Morphology-Aware', 'indic-en'),
        ('Malayalam', 'mal', 'A0', 'Baseline', 'en-indic'),
        ('Malayalam', 'mal', 'A1', 'Phonology-Aware', 'en-indic'),
        ('Malayalam', 'mal', 'A2', 'Morphology-Aware', 'en-indic')
    ]
    
    results = []
    for lang_name, lang_code, arm, arm_label, direction in experiments:
        run_folder = os.path.join(RUNS_DIR, f"{lang_code}_{arm}_{direction}")
        report_file = os.path.join(run_folder, "eval_report.json")
        checkpoint_file = os.path.join(run_folder, "checkpoint_best.pt")
        
        em_acc, cer, sva, test_loss = 0.0, 0.0, 0.0, 0.0
        loss_history = []
        status = "Pending"
        
        if os.path.exists(report_file):
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    em_acc = data["metrics"].get("exact_match_accuracy", 0.0)
                    cer = data["metrics"].get("character_error_rate", 0.0)
                    sva = data["metrics"].get("stop_voicing_accuracy", 0.0)
                    test_loss = data["metrics"].get("test_loss", 0.0)
                    status = "Ready"
            except Exception as e:
                print(f"  ⚠️ Error reading report {report_file}: {e}")
                
        if os.path.exists(checkpoint_file):
            try:
                chk = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
                loss_history = chk.get("loss_history", [])
            except Exception:
                pass
                
        results.append({
            "Language": lang_name,
            "LangCode": lang_code,
            "Arm": arm,
            "ArmLabel": arm_label,
            "Direction": direction,
            "ExactMatch": em_acc,
            "CER": cer,
            "SVA": sva,
            "TestLoss": test_loss,
            "LossHistory": loss_history,
            "Status": status
        })
    return results

def render_comparative_figures():
    data = load_all_results()
    ready_runs = [d for d in data if d["Status"] == "Ready"]
    
    if not ready_runs:
        print("⚠️ No completed runs found to plot yet.")
        return
        
    color_a0 = '#607D8B'   # Gray-blue (Baseline)
    color_a1 = '#7B1FA2'   # Purple (ValiMeli Phonology)
    color_a2 = '#00897B'   # Teal (Morphology)
    
    for direction in ["indic-en", "en-indic"]:
        dir_data = [d for d in data if d["Direction"] == direction]
        if not any(d["Status"] == "Ready" for d in dir_data):
            continue
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        languages = ['Tamil', 'Malayalam']
        x = np.arange(len(languages))
        width = 0.26
        
        a0_em = [next((d['ExactMatch'] for d in dir_data if d['Language'] == l and d['Arm'] == 'A0'), 0.0) for l in languages]
        a1_em = [next((d['ExactMatch'] for d in dir_data if d['Language'] == l and d['Arm'] == 'A1'), 0.0) for l in languages]
        a2_em = [next((d['ExactMatch'] for d in dir_data if d['Language'] == l and d['Arm'] == 'A2'), 0.0) for l in languages]
        
        a0_cer = [next((d['CER'] for d in dir_data if d['Language'] == l and d['Arm'] == 'A0'), 0.0) for l in languages]
        a1_cer = [next((d['CER'] for d in dir_data if d['Language'] == l and d['Arm'] == 'A1'), 0.0) for l in languages]
        a2_cer = [next((d['CER'] for d in dir_data if d['Language'] == l and d['Arm'] == 'A2'), 0.0) for l in languages]
        
        # Exact Match
        r1 = ax1.bar(x - width, a0_em, width, label='A0: Baseline (Char)', color=color_a0, edgecolor='white')
        r2 = ax1.bar(x, a1_em, width, label='A1: ValiMeli (Phonology)', color=color_a1, edgecolor='white')
        r3 = ax1.bar(x + width, a2_em, width, label='A2: Morphology (2508.08424)', color=color_a2, edgecolor='white')
        
        ax1.set_title(f'Word Exact Match (Top-1 EM %) — {direction.upper()}', fontsize=12, fontweight='bold', pad=10)
        ax1.set_xticks(x)
        ax1.set_xticklabels(languages, fontsize=11, fontweight='bold')
        ax1.set_ylabel('Exact Match Accuracy (%)', fontsize=11)
        ax1.set_ylim(0, max(max(a0_em + a1_em + a2_em) * 1.25, 20.0))
        
        for r in list(r1) + list(r2) + list(r3):
            h = r.get_height()
            if h > 0:
                ax1.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
        ax1.legend(loc='upper left', frameon=True)
        
        # CER
        r4 = ax2.bar(x - width, a0_cer, width, label='A0: Baseline (Char)', color=color_a0, edgecolor='white')
        r5 = ax2.bar(x, a1_cer, width, label='A1: ValiMeli (Phonology)', color=color_a1, edgecolor='white')
        r6 = ax2.bar(x + width, a2_cer, width, label='A2: Morphology (2508.08424)', color=color_a2, edgecolor='white')
        
        ax2.set_title(f'Character Error Rate (CER %) — {direction.upper()} (Lower is Better)', fontsize=12, fontweight='bold', pad=10)
        ax2.set_xticks(x)
        ax2.set_xticklabels(languages, fontsize=11, fontweight='bold')
        ax2.set_ylabel('CER (%)', fontsize=11)
        ax2.set_ylim(0, max(max(a0_cer + a1_cer + a2_cer) * 1.25, 20.0))
        
        for r in list(r4) + list(r5) + list(r6):
            h = r.get_height()
            if h > 0:
                ax2.annotate(f'{h:.1f}%', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 4), textcoords="offset points", ha='center', fontsize=8, fontweight='bold')
        ax2.legend(loc='upper right', frameon=True)
        
        sns.despine()
        fig.suptitle(f"Project ValiMeli: 3-Arm Comparative Transliteration Matrix ({direction.upper()})", fontsize=14, fontweight='bold', y=0.98)
        
        out_file = os.path.join(SCRATCH_DIR, f"valimeli_metrics_{direction}.png")
        plt.tight_layout(pad=1.8)
        plt.savefig(out_file, dpi=CHART_DPI, bbox_inches='tight')
        plt.close()
        print(f" -> Generated 3-arm metrics chart: {out_file}")

if __name__ == "__main__":
    render_comparative_figures()
