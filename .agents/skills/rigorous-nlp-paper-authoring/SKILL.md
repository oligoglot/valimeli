---
name: rigorous-nlp-paper-authoring
description: Comprehensive paper authoring, typesetting, and typography guide for computational linguistics, NLP, and speech science manuscripts (ACL, EMNLP, TACL, NAACL, COLING, NeurIPS, IEEE, LREC). Enforces Commonwealth English spelling standards, 0-error Overleaf LaTeX preambles, international scientific transliteration standards, single-author voice options, and transparent multi-tool AI disclosures. For the complete cross-subfield linguistics authoring guide, see the companion skill linguistics-paper-authoring.
---

# Universal Computational Linguistics Paper Authoring Guide

This skill governs the structure, typography, transliteration, styling, and ethical disclosure standards for manuscripts submitted to computational linguistics, NLP, and speech science venues (ACL, EMNLP, TACL, NAACL, COLING, NeurIPS, IEEE/ACM TASLP, Interspeech, LREC).

> [!NOTE]
> For broader manuscripts encompassing formal syntax/semantics, psycholinguistics, field documentation, or historical linguistics, use the comprehensive companion skill **`linguistics-paper-authoring`**.

---

## 1. Commonwealth English Spelling Standard

All manuscripts must strictly adhere to standard Commonwealth (British) English orthography. Maintain strict internal consistency across all sections, captions, tables, and figures:

| Commonwealth English (Required) | American Spelling (Avoid) |
| :--- | :--- |
| **analysed / analysing / analysis** | analyzed / analyzing |
| **behaviour / behavioural** | behavior / behavioral |
| **categorised / categorising** | categorized / categorizing |
| **centre / centred** | center / centered |
| **colour / coloured** | color / colored |
| **initialisation** | initialization |
| **modelling / modelled** | modeling / modeled |
| **optimised / optimising** | optimized / optimizing |
| **parsimonious** | parsimonious |
| **prioritise / prioritising** | prioritize / prioritizing |
| **programme** (system, software, schedule) | program (except computer code context if preferred) |
| **realisation / realisations** | realization / realizations |
| **regulariser / normaliser** | regularizer / normalizer |
| **spirantised / lenited** | spirantized |
| **summarised / summarising** | summarized / summarizing |
| **utilising / utilised / utilisation** | utilizing / utilized / utilization |

---

## 2. Authorial Voice: Single-Author Paper Conventions

For single-author manuscripts, choose one of two standard academic styles and maintain internal consistency throughout the manuscript:

### Option A: The Authorial "We" (*Universally Standard in CS & NLP*)
* The standard convention in mathematics, computer science, and computational linguistics.
* Defined as **inclusive authorial voice** (*"the author and the reader examining the evidence together"*).
* *Example*: *"We evaluate this hypothesis across data scales from 25k to 1.0M pairs."*

### Option B: Paper-Centric Active Voice (*Graceful Third-Person Alternative*)
* Eliminates the plural pronoun without resorting to heavy passive voice (*"It was observed that..."*).
* Uses the paper, the empirical analysis, or the data as the active subject:
  * *"This paper evaluates the orthographic deficit hypothesis..."*
  * *"Empirical measurements demonstrate that $P(y \mid X) < 0.50$..."*
  * *"Section~\ref{sec:scale} examines scaling dynamics across model families..."*
  * *"The curated dataset is open-sourced under CC BY-SA 4.0..."*

---

## 3. 0-Error Overleaf LaTeX Architecture

To ensure seamless compilation on standard **Overleaf pdfLaTeX** as well as **XeLaTeX/LuaLaTeX**, use a defensive preamble with explicit Unicode macro fallbacks:

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
  % Universal Unicode & Diacritic Fallbacks for standard pdfLaTeX engines
  \newunicodechar{ṅ}{\.{n}}
  \newunicodechar{ŋ}{\ng}
  \newunicodechar{ḻ}{\underline{l}}
  \newunicodechar{ṭ}{\d{t}}
  \newunicodechar{ṇ}{\d{n}}
  \newunicodechar{ṟ}{\b{r}}
  \newunicodechar{ḍ}{\d{d}}
  \newunicodechar{ḷ}{\d{l}}
  \newunicodechar{ṣ}{\d{s}}
  \newunicodechar{ś}{\'{s}}
  \newunicodechar{ā}{\={a}}
  \newunicodechar{ō}{\={o}}
  \newunicodechar{ī}{\={i}}
  \newunicodechar{ū}{\={u}}
  \newunicodechar{ē}{\={e}}
  \newunicodechar{š}{\v{s}}
  \newunicodechar{č}{\v{c}}
  \newunicodechar{ž}{\v{z}}
\fi

\bibliographystyle{plainnat}
```

### Modular Structure
* Place large tables and complex appendices in dedicated sub-files (`table_results.tex`, `table_ablation.tex`) and include them via `\input{...}`.
* Ground every numerical value directly in verified, code-generated JSON/CSV artifacts.

---

## 4. International Scientific Transliteration Standards

Whenever introducing examples from non-Latin scripts (Indic, Semitic, Cyrillic, Sinitic, Japanese, Greek), provide the **quadruple representation**:

1. **Native Script / Orthography**: Primary writing system (e.g., Arabic, Devanagari, Tamil, Cyrillic, Kanji).
2. **Standard Scientific Transliteration**: Established international standard (e.g., ISO 15919 for Indic, DIN 31635 for Arabic, ALA-LC / ISO 9 for Cyrillic, Hepburn for Japanese, Pinyin for Mandarin).
3. **Phonetic / Phonemic IPA**: Enclosed in `/.../` (phonemic) or `[...]` (narrow phonetic).
4. **English Gloss / Translation**: Enclosed in quotation marks or parentheses.

*Example Format*:
$$\text{Native: \textbf{படம்}} \quad \longrightarrow \quad \text{ISO 15919: \textit{pa\d{t}am}} \quad \longrightarrow \quad \text{IPA: } [pɐɖɐm] \quad \longrightarrow \quad \text{Gloss: ``picture/movie''}$$

---

## 5. Transparent Multi-Tool AI Assistance Disclosure

When using AI systems during the research lifecycle, include an explicit, unnumbered section before the bibliography detailing the distinct functional roles of each tool:

```latex
\section*{Use of AI-Assisted Technologies}
During the preparation and execution of this research, the author utilised AI-assisted systems for specialised research workflows:
\begin{itemize}
    \item \textbf{[Ideation / Review Model, e.g., Anthropic Claude]}: Utilised during the conceptualisation and planning phases for initial data availability assessments and experimental design ideation, and subsequently served as an automated peer reviewer providing critical interpretation checks, structural critique, and prose feedback on working drafts.
    \item \textbf{[Agentic Coding Model, e.g., Google Antigravity / Gemini]}: Utilised as an agentic pair-programming assistant for implementing data processing scripts, orchestrating multi-seed PyTorch benchmarking pipelines, harvesting speech corpora, and formatting \LaTeX\ typography and bibliography assets.
\end{itemize}
The author conceived the research questions, directed the experimental methodology, derived the theoretical formulations, verified all empirical data artifacts, authored the final prose, and takes full scientific and legal responsibility for the integrity and conclusions of this publication.
```

---

## 6. Research Integrity: Research as a Truth-Seeking Exercise

> [!CAUTION]
> **RESEARCH IS ESSENTIALLY A TRUTH-SEEKING EXERCISE. FABRICATING ANYTHING IS COMPLETELY UNACCEPTABLE.**
> Scientific publication exists solely to report genuine, verified truth. Fabricating or hallucinating ANY empirical number, baseline result, statistical metric, architectural setup, or academic citation is a catastrophic breach of research ethics.

### Mandatory Directives for Manuscripts and Reports:
1. **Zero-Fabrication of Numbers & Empirical Results**:
   - Every single number, percentage, parameter count, $p$-value, and standard deviation in the text, tables, and figures MUST be programmatically generated by audited code from real data and saved in frozen artifact files.
   - Never insert estimated or "plausible" placeholder numbers into a draft.
2. **Never Cite from Memory or Speculation**:
   - Every citation key, paper title, author list, venue, and DOI must be verified against primary indexing databases (ACL Anthology, Crossref, arXiv API, DBLP, PubMed).
   - DOIs must resolve directly to the published article. Historical or book sources must be verified against library catalogs or scanned front matter.
3. **If Unresolvable, STOP AND ASK THE USER**:
   - If you cannot find or verify a citation, DOI, paper title, or published baseline figure: **NEVER create a plausible placeholder**. 
   - **STOP IMMEDIATELY AND ASK THE USER** for the citation, paper PDF, or BibTeX record.
4. **No Cross-Attribution of Internal Numbers**:
   - NEVER attribute figures from internal experiments or replication baselines to an external published paper.
   - Internal baselines must always be introduced as *this work's replication baseline* (with forward pointers to the study's results table and frozen JSON artifacts).


