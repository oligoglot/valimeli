---
name: rigorous-nlp-paper-authoring
description: Comprehensive authoring guide for high-impact NLP and computational linguistics papers. Enforces Commonwealth English spelling standards, 0-error Overleaf LaTeX compilation, ISO 15919 transliteration standards, single-author voice options, and transparent AI assistance disclosures.
---

# Rigorous NLP Paper Authoring Guide

This skill governs the structure, typography, linguistic representation, and styling standards for NLP and computational linguistics manuscripts (ACL, EMNLP, TACL, NAACL, COLING, IEEE).

---

## 1. Commonwealth English Spelling Standard

All manuscripts must strictly adhere to standard Commonwealth (British) English spelling conventions:

| Commonwealth English (Required) | American Spelling (Avoid) |
| :--- | :--- |
| **analysed / analysing** | analyzed / analyzing |
| **behaviour / behavioural** | behavior / behavioral |
| **categorised / categorising** | categorized / categorizing |
| **initialisation** | initialization |
| **modelling / modelled** | modeling / modeled |
| **optimised / optimising** | optimized / optimizing |
| **parsimonious** | parsimonious |
| **prioritise / prioritising** | prioritize / prioritizing |
| **programme** (system/plan) | program |
| **realisation / realisations** | realization / realizations |
| **regulariser / normaliser** | regularizer / normalizer |
| **spirantised** | spirantized |
| **utilising / utilised** | utilizing / utilized |

---

## 2. Authorial Voice: Single-Author Paper Guidelines

When writing a single-author paper, choose one of two standard academic styles and maintain internal consistency:

### Option A: The Authorial "We" (*Universally Accepted*)
* Standard in CS, mathematics, and computational linguistics.
* Understood as **inclusive** (*"the author and the reader examining the evidence together"*).
* *Example*: *"We evaluate this hypothesis across data scales from 25k to 1.0M pairs."*

### Option B: Paper-Centric Active Voice (*Graceful Third-Person*)
* Avoids the plural pronoun without resorting to heavy passive voice (*"It was observed that..."*).
* Uses the paper, analysis, or data as the active grammatical subject:
  * *"This paper evaluates the orthographic deficit hypothesis..."*
  * *"Empirical measurements demonstrate that $P(\text{voiced}\mid C) < 0.50$..."*
  * *"Section~\ref{sec:scale} examines scaling dynamics across architectures..."*
  * *"The \textsc{ValiMeli-Speech} corpus is open-sourced under CC BY-SA 4.0..."*

---

## 3. 0-Error Overleaf LaTeX Architecture

To ensure seamless compilation on standard **Overleaf pdfLaTeX**, follow these structural rules:

### A. Preamble & Font Fallbacks
Include `\usepackage{natbib}`, `\usepackage{iftex}`, and `\usepackage{newunicodechar}` with explicit ISO 15919 macro fallbacks:

```latex
\documentclass[11pt]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{times,latexsym,url,graphicx,booktabs,amsmath,amssymb,microtype}
\usepackage{natbib}
\usepackage[hidelinks]{hyperref}
\usepackage[margin=1in]{geometry}
\usepackage{iftex}

\ifPDFTeX
  \usepackage{newunicodechar}
  % ISO 15919 & Indic Unicode fallbacks for standard pdfLaTeX
  \newunicodechar{ṅ}{\.{n}}
  \newunicodechar{ŋ}{\ng}
  \newunicodechar{ḻ}{\underline{l}}
  \newunicodechar{ṭ}{\d{t}}
  \newunicodechar{ṇ}{\d{n}}
  \newunicodechar{ṟ}{\b{r}}
  \newunicodechar{ḍ}{\d{d}}
  \newunicodechar{ா}{\={a}}
  \newunicodechar{ோ}{\={o}}
  \newunicodechar{ங}{\.{n}a}
  \newunicodechar{ழ}{\underline{l}a}
  \newunicodechar{ள}{\d{l}a}
\fi
```

### B. Modular Table Structure
* Keep large tables in standalone files (`table_scale.tex`, `table_entropy.tex`) and include them with `\input{table_name.tex}`.
* Every number in a table must map 1-to-1 with a frozen JSON artifact.

---

## 4. Linguistic Transliteration Standards (ISO 15919)

When introducing native words, always provide the quadruple representation:
1. **Native Script**: படம், பக்கம், தம்பி, குரங்கு
2. **ISO 15919 Romanisation**: *paṭam*, *pakkam*, *tampi*, *kuraṅku* (use LaTeX macros `\textit{pa\d{t}am}`, `\textit{kura\.{n}ku}`)
3. **Colloquial / Benchmark Romanisation**: (*padam*), (*pakkam*), (*thambi*), (*kurangu*)
4. **Phonetic IPA**: $[pɐɖɐm]$, $[pɐkːɐm]$, $[t̪ɐmbi]$, $[kʊɾɐŋɡɯ]$

**Critical Constraint**: Never use raw Unicode-decomposed strings like `pat.ama` or `[ma][.]` for pure coda consonants (*meyyeḻuttu* ம் is $/m/$, not $/ma/ + \text{virāma}$).

---

## 5. Transparent AI Assistance Disclosure

Include a dedicated unnumbered section before the bibliography detailing all AI systems by their specific research contributions:

```latex
\section*{Use of AI-Assisted Technologies}
During the preparation and execution of this research, the author utilised AI-assisted systems for specialised research workflows:
\begin{itemize}
    \item \textbf{Anthropic Claude}: Utilised during the conceptualisation and planning phases for initial data availability assessments and experimental design ideation, and subsequently served as an automated peer reviewer providing critical interpretation checks, structural critique, and prose feedback on working drafts.
    \item \textbf{Google Antigravity / Gemini}: Utilised as an agentic pair-programming assistant for implementing data processing scripts, orchestrating multi-seed PyTorch benchmarking pipelines, harvesting Wikimedia Commons speech corpora, and formatting \LaTeX\ typography and bibliography assets.
\end{itemize}
The author conceived the research questions, directed the experimental methodology, derived the theoretical proofs and information-theoretic formulations, verified all empirical data artifacts, authored the final prose, and takes full scientific and legal responsibility for the integrity and conclusions of this publication.
```
