#!/usr/bin/env python3
"""
Project ValiMeli — Real-Time Live Benchmark Monitor
Displays active run progress, memory consumption, loss curves, and empirical metrics.
"""

import os
import sys
import json
import time

WORKSPACE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRATCH_DIR = os.path.join(WORKSPACE_DIR, "scratch", "valimeli")
RUNS_DIR = os.path.join(SCRATCH_DIR, "runs")
ARTIFACTS_DIR = os.path.join(WORKSPACE_DIR, "artifacts")

experiments = [
    ('Tamil', 'tam', 'A0', 'Baseline (Character)', 'indic-en'),
    ('Tamil', 'tam', 'A1', 'ValiMeli (Phonology)', 'indic-en'),
    ('Tamil', 'tam', 'A0', 'Baseline (Character)', 'en-indic'),
    ('Tamil', 'tam', 'A1', 'ValiMeli (Phonology)', 'en-indic'),
    ('Malayalam', 'mal', 'A0', 'Baseline (Character)', 'indic-en'),
    ('Malayalam', 'mal', 'A1', 'ValiMeli (Phonology)', 'indic-en'),
    ('Malayalam', 'mal', 'A0', 'Baseline (Character)', 'en-indic'),
    ('Malayalam', 'mal', 'A1', 'ValiMeli (Phonology)', 'en-indic'),
]

def display_live_status():
    print("=" * 80)
    print(f" PROJECT VALIMELI — LIVE BENCHMARK MONITOR [{time.strftime('%Y-%m-%d %H:%M:%S')}]")
    print("=" * 80)
    
    print("\n| Language | Experimental Arm | Direction | Status | Best Val Loss | Holdout EM (%) | Holdout CER (%) |")
    print("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |")
    
    total_completed = 0
    for lang, code, arm, name, direction in experiments:
        run_folder = os.path.join(RUNS_DIR, f"{code}_{arm}_{direction}")
        report_path = os.path.join(run_folder, "eval_report.json")
        checkpoint_path = os.path.join(run_folder, "checkpoint_best.pt")
        
        status = "⏳ Pending"
        val_loss = "—"
        em = "—"
        cer = "—"
        
        if os.path.exists(report_path):
            try:
                with open(report_path) as f:
                    d = json.load(f)
                    val_loss = f"{d['metrics'].get('validation_loss', 0.0):.4f}"
                    em = f"{d['metrics'].get('exact_match_accuracy', 0.0):.2f}%"
                    cer = f"{d['metrics'].get('character_error_rate', 0.0):.2f}%"
                    status = "✅ Completed"
                    total_completed += 1
            except Exception:
                status = "⚠️ Error"
        elif os.path.exists(checkpoint_path):
            status = "🏃 Running"
            
        print(f"| {lang:9s} | {name:20s} ({arm}) | {direction:8s} | {status:11s} | {val_loss:13s} | {em:14s} | {cer:15s} |")
        
    print("\n" + "-" * 80)
    print(f" Total Grid Completion: [{total_completed}/8] cells finished.")
    print("=" * 80)

if __name__ == "__main__":
    display_live_status()
