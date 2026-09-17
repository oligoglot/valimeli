# Project ValiMeli (வலி–மெலி)
### Phonology-Aware Tokenisation for Dravidian Transliteration

[![Preprint: arXiv:submit/8042292](https://img.shields.io/badge/Preprint-arXiv%3Asubmit%2F8042292%20[cs.CL]-b31b1b.svg)](https://github.com/oligoglot/valimeli/releases/download/v1.0-preprint/parsimonious_code_preprint.pdf)
[![GitHub Release](https://img.shields.io/badge/Release-v1.0--preprint-blue.svg)](https://github.com/oligoglot/valimeli/releases/tag/v1.0-preprint)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Dataset: Aksharantar](https://img.shields.io/badge/Dataset-AI4Bharat%20Aksharantar-green.svg)](https://huggingface.co/datasets/ai4bharat/Aksharantar)

> **Preprint:** [Download PDF (arXiv:submit/8042292)](https://github.com/oligoglot/valimeli/releases/download/v1.0-preprint/parsimonious_code_preprint.pdf) *(arXiv approval pending)*.  
> **Paper Title:** *A Parsimonious Code: Allophonic Voicing and the Limits of Phonological Supervision in Romanised Tamil and Malayalam Transliteration*  
> **Direct Repository Mirror:** [`docs/parsimonious_code_preprint.pdf`](docs/parsimonious_code_preprint.pdf)

**Project ValiMeli** (*Vallinam* [Hard] + *Mellinam* [Soft]) addresses the fundamental inductive bias gap in Dravidian machine transliteration (Tamil and Malayalam). By replacing naive character-level tokenisers and lossy Devanagari pivots with deterministic, context-sensitive phonotactic tags (`[INIT]`, `[GEM]`, `[NASAL]`, `[INTER]`), ValiMeli collapses stop-voicing ambiguity and delivers superior sequence-to-sequence convergence.

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
│   ├── fig_effects_corrected.png  # Forest plot of 10 matched scaling comparisons
│   ├── acoustic_voicing_spectrograms.png # STFT spectrograms of native Tamil speech
│   └── multiseed_rigorous_results.json # Multi-seed 25k paired McNemar evaluation logs
├── data/                          # Original compressed Aksharantar dataset archives
│   ├── tam.zip
│   └── mal.zip
├── docs/                          # Scientific publication & preprint artifacts
│   ├── parsimonious_code_preprint.pdf # Final compiled submission preprint PDF
│   ├── revised2/                  # Camera-ready LaTeX manuscript and table suite
│   │   ├── main.tex
│   │   ├── references.bib
│   │   └── table_*.tex
│   └── valimeli_pipeline_trace.md # Engineering architecture & math trace
├── scratch/                       # Working run telemetry, checkpoints, extracted datasets
├── src/                           # Core source codebase
│   ├── valimeli-benchmark.py      # Standardised Seq2Seq benchmarking engine (v4)
│   ├── test_linguistics.py        # Unit tests for Tamil/Malayalam phonotactic rules
│   ├── compute_voicing_entropy.py # Dynamic-programming phonetic aligner & entropy audit
│   ├── run_multiseed_rigorous_benchmark.py # Multi-seed paired McNemar runner
│   ├── speech/                    # Wikimedia Commons audio harvesting & acoustic features
│   └── monitor_progress.py        # Real-time Markdown progress dashboard
├── README.md                      # Project documentation and quickstart guide
├── requirements.txt               # Python package dependencies
└── run_grid.sh                    # Automated shell runner for comparative grid
```

---

## 🔬 Scientific Publication

- **Preprint Manuscript**: [`docs/parsimonious_code_preprint.pdf`](docs/parsimonious_code_preprint.pdf) | [Release Download](https://github.com/oligoglot/valimeli/releases/download/v1.0-preprint/parsimonious_code_preprint.pdf)
  - **Title**: *A Parsimonious Code: Allophonic Voicing and the Limits of Phonological Supervision in Romanised Tamil and Malayalam Transliteration*
  - **Author**: BalaSundaraRaman Lakshmanan
  - **Submission Identifier**: `arXiv:submit/8042292 [cs.CL]` (Submitted 6 September 2026; moderation approval pending)
  - **LaTeX Source & Tables**: Complete camera-ready manuscript sources located in [`docs/revised2/`](docs/revised2/)
- **Core Findings**:
  - **Argmax Invariance under Skewed Allophony (Proposition 1)**: In crowdsourced benchmarks, voiceless spellings remain the plurality across all phonotactic contexts ($P(\text{voiced}) < 0.50$). While context provides mutual information ($I(V; C) > 0$), context-conditioned rules reduce argmax decision error by $0.00\%$.
  - **Acoustic Realisation in Native Speech**: In spoken speech acoustics (VALIMELI-SPEECH), phonotactic context decisively flips voicing in post-nasal ($85.14\%$ voiced) and intervocalic ($56.77\%$ voiced) positions, achieving a $+32.92\%$ relative error reduction.
  - **Scaling & Multi-Seed Dynamics**: Across random seeds 42, 43, 44, auxiliary phonological supervision yields overlapping performance at 25k pairs ($\Delta \le 0.15$ points, $p \ge 0.58$) and reverses to significant degradation at 1.0M bilingual pairs ($-2.33\%$, $p = 0.0003$).
  - **Open Speech Index**: Releases the VALIMELI-SPEECH index (5,325 phonotactically annotated native spoken Tamil audio recordings across 12,435 plosive slots) under CC BY-SA 4.0.
- **Technical Pipeline Trace**: [`docs/valimeli_pipeline_trace.md`](docs/valimeli_pipeline_trace.md)
  - Deep-dive into akshara segmentation, PUA tagging, memory streaming mechanics, and Bahdanau attention equations.

---

## ⚖️ License
MIT License. Open for academic research and foundational model tokeniser integrations. Spoken audio recordings curated from Wikimedia Commons / Lingua Libre under CC BY-SA 4.0.
