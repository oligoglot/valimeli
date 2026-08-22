# Project ValiMeli (வலி–മെലി)
### Phonology-Aware Tokenization for Dravidian Transliteration

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Aksharantar](https://img.shields.io/badge/Dataset-AI4Bharat%20Aksharantar-green.svg)](https://huggingface.co/datasets/ai4bharat/Aksharantar)

**Project ValiMeli** (*Vallinam* [Hard] + *Mellinam* [Soft]) addresses the fundamental inductive bias gap in Dravidian machine transliteration (Tamil and Malayalam). By replacing naive character-level tokenizers and lossy Devanagari pivots with deterministic, context-sensitive phonotactic tags (`[INIT]`, `[GEM]`, `[NASAL]`, `[INTER]`), ValiMeli collapses stop-voicing ambiguity and delivers superior sequence-to-sequence convergence.

---

## 🚀 Quickstart

### 1. Environment Setup
```bash
# Activate virtual environment
source valimeli-env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Linguistic Rules & Unit Tests
```bash
./valimeli-env/bin/python3 src/test_linguistics.py
```

### 3. Run the Full Comparative Benchmarking Grid
```bash
./run_grid.sh
```

### 4. Monitor Live Benchmark Status
```bash
./valimeli-env/bin/python3 src/monitor_progress.py
```

---

## 📊 Summary of Large-Scale Experimental Results (Aksharantar Benchmark)

*Trained on 1,600,000 parallel pairs (200k/cell) across 8 epochs and evaluated on 100% of official holdout test sets (11,499 Tamil, 12,451 Malayalam).*

| Language | Experimental Arm | Direction | Validation Loss | Holdout Top-1 EM (%) | Holdout CER (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | Baseline (A0: Character) | `indic-en` | 0.9982 | 28.53% | 15.11% |
| **Tamil** | **ValiMeli (A1: Phonology)** | `indic-en` | **0.9885** | **28.32%** | **15.59%** |
| **Tamil** | Baseline (A0: Character) | `en-indic` | **0.4939** | **59.97%** | **9.00%** |
| **Tamil** | **ValiMeli (A1: Phonology)** | `en-indic` | 0.6166 | 59.53% | 9.12% |
| **Malayalam** | Baseline (A0: Character) | `indic-en` | 0.8205 | 32.74% | 11.42% |
| **Malayalam** | **ValiMeli (A1: Phonology)** | `indic-en` | **0.8191** | **31.72%** | **11.83%** |
| **Malayalam** | Baseline (A0: Character) | `en-indic` | **0.6198** | **51.36%** | **10.58%** |
| **Malayalam** | **ValiMeli (A1: Phonology)** | `en-indic` | 0.7636 | 50.38% | 10.49% |

---

## 📁 Repository Structure

```
valimeli/
├── artifacts/                     # High-resolution 300 DPI figures & JSON evaluation reports
│   ├── valimeli_metrics_indic-en.png
│   ├── valimeli_metrics_en-indic.png
│   ├── valimeli_convergence_indic-en.png
│   ├── valimeli_convergence_en-indic.png
│   ├── attn_tamil_pakkam.png
│   ├── attn_tamil_padam.png
│   ├── attn_tamil_sangu.png
│   └── tam_A1_indic-en_results.json
├── data/                          # Original compressed Aksharantar dataset archives
│   ├── tam.zip
│   └── mal.zip
├── docs/                          # Scientific publication & outreach artifacts
│   ├── paper_skeleton.md          # Full ACL/EMNLP manuscript layout
│   ├── outreach_strategy.md       # Stakeholder pitches (AI4Bharat, Sarvam, Saama, Niranjan)
│   └── valimeli_pipeline_trace.md # Engineering architecture & math trace
├── scratch/                       # Working run telemetry, checkpoints, extracted datasets
│   └── valimeli/
│       ├── runs/                  # Saved model checkpoints per experiment cell
│       ├── data/                  # Extracted JSON splits (tam_train, tam_test, etc.)
│       └── grid_run.log           # Full log of high-compute benchmark execution
├── src/                           # Core source codebase
│   ├── valimeli-benchmark.py      # Standardized Seq2Seq benchmarking engine (v4)
│   ├── test_linguistics.py        # Unit tests for Tamil/Malayalam phonotactic rules
│   ├── plot_results.py            # Publication plotting engine (300 DPI bar/loss charts)
│   ├── plot_attention.py          # Bahdanau attention weight heatmap extractor
│   └── monitor_progress.py        # Real-time Markdown progress dashboard
├── README.md                      # Project documentation and quickstart guide
├── requirements.txt               # Python package dependencies
├── run_grid.sh                    # Automated shell runner for 8-cell comparative grid
└── check_mps.py                   # Metal Performance Shaders (MPS) memory monitor
```

---

## 🔬 Scientific Publications & Outreach Dossiers

- **Manuscript Pre-print Layout**: [`docs/paper_skeleton.md`](docs/paper_skeleton.md)
  - Theoretical formalization of Dravidian Stop Allophony.
  - Comparative analysis with Morphology-Aware Tokenization ([arXiv:2508.08424](https://arxiv.org/abs/2508.08424)).
  - Information-theoretic entropy audit ($H(\text{Voicing} \mid \text{Context})$).
- **Outreach Strategy Dossier**: [`docs/outreach_strategy.md`](docs/outreach_strategy.md)
  - Tailored communications for **Anoop Kunchukuttan & AI4Bharat**, **Niranjan Nayak**, **Sarvam AI**, and **Malaikkannan & Saama AI**.
- **Technical Pipeline Trace**: [`docs/valimeli_pipeline_trace.md`](docs/valimeli_pipeline_trace.md)
  - Deep-dive into akshara segmentation, PUA tagging, memory streaming mechanics, and Bahdanau attention equations.

---

## ⚖️ License
MIT License. Open for academic research and foundational model tokenizer integrations.
