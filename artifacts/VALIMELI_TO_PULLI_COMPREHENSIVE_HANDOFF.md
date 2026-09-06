# Project ValiMeli to Pulli: Comprehensive Engineering & Scientific Handoff

**Author / Maintainer**: S. Lakshmanan  
**Repository**: `oligoglot/valimeli`  
**Active Branch**: `feat/multilingual-pretraining-26m`  
**Target Projects**: ValiMeli Paper Camera-Ready / Pulli IME & Normalisation Engine  
**Date**: September 2026  

---

## 1. Executive Summary & Paper State

The research paper ***"Underspecified but Not Undecidable: Argmax Invariance Bounds the Value of Phonological Supervision in Dravidian Transliteration"*** is complete, verified, and frozen for camera-ready submission.

### Key Artifact Locations
* **Main LaTeX Manuscript**: [`docs/revised2/main.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/main.tex)
* **LaTeX Table Set**:
  * [`docs/revised2/table_acoustic_alignment.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_acoustic_alignment.tex) (Table 1: Acoustic & Phonotactic Alignment)
  * [`docs/revised2/table_entropy.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_entropy.tex) (Table 2: Information-Theoretic Voicing Entropy)
  * [`docs/revised2/table_context.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_context.tex) (Table 3: Phonotactic Context Voicing Probabilities)
  * [`docs/revised2/table_scale.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_scale.tex) (Table 4: Multi-Seed & Large-Scale Transliteration Benchmarks)
  * [`docs/revised2/table_disagree.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_disagree.tex) (Table 5: Annotator Duality in Dakshina vs. YouTube Tanglish)
  * [`docs/revised2/table_downstream.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_downstream.tex) (Table 6: Downstream Sentiment Classification on Theedhum Nandrum)
* **Bibliography**: [`docs/revised2/references.bib`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/references.bib) (Fully verified citations including Kunchukuttan et al. TACL 2021, Madhani et al. EMNLP 2023, Niklas 1988, Lakshmanan et al. 2020).
* **Spectrogram Figure**: [`docs/revised2/acoustic_voicing_spectrograms.png`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/acoustic_voicing_spectrograms.png)

---

## 2. Core Scientific & Empirical Findings (Crucial for Pulli)

### A. The Principle of Argmax Invariance
1. **The Myth**: Prior computational literature attributed lower transliteration accuracy in Tamil and Malayalam to "orthographic underspecification" (a single graphemic stop series spanning voiced and voiceless allophones).
2. **The Information-Theoretic Reality**: 
   - Unconditional voicing entropy $H(V) = 0.8066$ bits (Tamil) and $0.5509$ bits (Malayalam).
   - Conditioning on phonotactic context reduces entropy to $H(V \mid C) = 0.6705$ bits (Tamil) and $0.3922$ bits (Malayalam), yielding mutual information $I(V; C) = 0.1361$ and $0.1587$ bits.
3. **Why Auxiliary Loss Heads Fail**:
   - In crowdsourced benchmark target references (Aksharantar / Dakshina), $P(\text{voiced} \mid C) \le 0.4901$ across **all** phonotactic environments (even post-nasally, where speech mandates voicing).
   - Because voiceless Roman spellings remain the plurality outcome in every context, conditioning on context alters the greedy/beam argmax decision boundary by exactly **0.00%**.
   - **Takeaway for Pulli**: Do not rely on source-side loss heads to resolve phonetic ambiguity. Pulli must implement **input-side multi-modal candidate generation and acoustic/citation typing toggles**.

### B. Multi-Seed Benchmark Results (25k Low-Resource Scale)
Conducted across Seeds 42, 43, 44 on canonical holdout test sets ($N = 11{,}499$ for Tamil, $N = 12{,}451$ for Malayalam):
* **Tamil (25k)**: Baseline ($A_0$) $25.21 \pm 1.15\%$ vs. Multi-Task ($A_1\text{-MT}$) $25.07 \pm 1.08\%$ ($\Delta = -0.14\%$, Paired McNemar $\chi^2 = 0.298$, $p = 0.585$).
* **Malayalam (25k)**: Baseline ($A_0$) $18.86 \pm 0.96\%$ vs. Multi-Task ($A_1\text{-MT}$) $19.00 \pm 1.17\%$ ($\Delta = +0.14\%$, Paired McNemar $\chi^2 = 0.015$, $p = 0.901$).
* Over $1{,}620$ discordant prediction pairs per language prove that small low-resource shifts are random churn rather than systematic phonological learning.

### C. Large-Scale Scaling Dynamics (250k, 500k, 1.0M)
* **250k BiGRU (IndicXlit Architecture)**: Tamil $58.19\% \to 59.72\%$ ($+1.53\%$), Malayalam $51.08\% \to 52.16\%$ ($+1.08\%$).
* **250k Transformer**: Tamil $59.18\% \to 59.94\%$ ($+0.76\%$, $p = 0.24$).
* **500k Transformer**: Tamil $60.71\% \to 60.49\%$ ($-0.22\%$, $p = 0.74$).
* **1.0M Bilingual Transformer**: Tamil $62.42\% \to 60.09\%$ ($\mathbf{-2.33\%}$, $p = 0.0003$), Malayalam $55.28\% \to 53.91\%$ ($\mathbf{-1.37\%}$, $p = 0.0295$).
* **Conclusion**: Auxiliary phonological loss degrades transformer capacity at scale.

---

## 3. Linguistic & Orthographic Nuances (Must Follow in Pulli)

### A. The Unicode/ISCII Storage Artifact vs. Native Phonology
* **Trap**: In the Brahmic ISCII/Unicode digital model, vowel-bearing syllables (*uyirmey*, e.g., `ம` `U+0BAE`) are base codepoints, requiring a combining virāma/puḷḷi (`்` `U+0BCD`) to strip the vowel to produce pure consonants (`ம்`). Naive segmenters incorrectly tokenise `படம்` into `[pa][ṭa][ma][.]` or `pat.ama`.
* **Linguistic Truth**: In *Tolkāppiyam*, pure consonants (*meyyeḻuttu*) are the primary phonemic units ($/m/$). The word **படம்** (*paṭam*) consists of exactly 3 *eḻuttukkaḷ*:
  $$\text{ப} \ ([pa]) \quad + \quad \text{ட} \ ([\text{ɖ}a]) \quad + \quad \text{ம்} \ ([m]) \implies \mathbf{pa\text{-}\d{t}a\text{-}m \ (pa\d{t}am)}$$
* **Rule for Pulli**: Never decompose coda consonants into sub-syllabic base + virāma strings.

### B. Palatal Sibilant Lenition vs. Plosive Voicing
* In Modern Spoken Tamil, word-initial **ச** undergoes variable spirantization/lenition to a sibilant ($[s]$ or $[ɕ]$ or $[h]$), causing divergent spellings (*sangu* vs. *chanku* vs. *cangu*).
* For pure plosive voicing demonstrations, we use words with unspirantized velar/bilabial stops like **குரங்கு** (*kuraṅku* / *kurangu* $[kʊɾɐŋɡɯ]$) or **படம்** (*paṭam* / *padam*).
* **Rule for Pulli**: Implement distinct IME phonetic mapping rules for palatal affricates (`c`/`s`/`ch`) vs. standard plosive voicing (`k`/`g`, `t`/`d`, `p`/`b`).

### C. Standard Quadruple Representation
All linguistic examples must follow the format:
1. Native Script: **படம்**, **பக்கம்**, **தம்பி**, **குரங்கு**
2. Standard ISO 15919: **paṭam**, **pakkam**, **tampi**, **kuraṅku** (`\textit{pa\d{t}am}`, `\textit{kura\.{n}ku}`)
3. Colloquial / Benchmark Romanisation: (*padam*), (*pakkam*), (*thambi*), (*kurangu*)
4. Narrow Phonetic IPA: $[pɐɖɐm]$, $[pɐkːɐm]$, $[t̪ɐmbi]$, $[kʊɾɐŋɡɯ]$

---

## 4. Open-Source Speech Datasets Released

### A. The ValiMeli-Speech Corpus
* **Artifact**: [`artifacts/tamil_speech_full_corpus_manifest.json`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/artifacts/tamil_speech_full_corpus_manifest.json) ($5.63$\,MB, $138{,}007$ lines).
* **Contents**: $5{,}324$ native spoken Tamil audio recordings from Lingua Libre / Wikimedia Commons, annotated at the *eḻuttu* level into $14{,}718$ plosive slots:
  * Word-Initial Plosives: $2{,}158$ slots
  * Geminate Fortis: $2{,}375$ slots
  * Post-Nasal Voiced: $1{,}918$ slots
  * Intervocalic Lenis: $8{,}267$ slots
* **License**: Creative Commons Attribution-ShareAlike 4.0 International (**CC BY-SA 4.0**).
* **Attribution**: Documented in [`artifacts/AUDIO_ATTRIBUTION.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/artifacts/AUDIO_ATTRIBUTION.md) attributing native contributors (`User:Sriveenkat`, `User:Manimaran96`, `User:StarlitNocturne`).

---

## 5. LaTeX & Paper Hygiene

1. **Commonwealth English**: Full British English spelling standard strictly enforced (*categorised*, *utilising*, *initialisation*, *modelling*, *optimised*, *parsimonious*, *realisation*, *analysed*, *behaviour*).
2. **0-Error Overleaf Compilation**: Standardised on `\usepackage{natbib}`, `\bibliographystyle{plainnat}`, and `\newunicodechar` fallbacks for `ṅ` (`\.{n}`), `ŋ` (`\ng`), `ḻ` (`\underline{l}`), `ṭ` (`\d{t}`), `ṇ` (`\d{n}`), `ṟ` (`\b{r}`), and `ḍ` (`\d{d}`).
3. **AI Assistance Disclosure**: Explicit statement added before the bibliography detailing the distinct roles of **Anthropic Claude** (ideation, planning, and peer review) and **Google Antigravity / Gemini** (agentic coding, multi-seed benchmarking, audio manifests, and LaTeX typography).

---

## 6. Antigravity Custom Skills Created

Two permanent skills are installed in the workspace ([`.agents/skills/`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/.agents/skills/)) and globally (`~/.gemini/config/skills/`):

1. **`empirical-nlp-research-methodology`**:
   - Theoretical bounding before modelling ($H(Y), H(Y\mid X), I(X;Y)$, Argmax Invariance).
   - Multi-seed low-resource protocols ($\mu \pm \sigma$, Seeds 42, 43, 44).
   - Paired McNemar significance testing on instance prediction logs.
   - Dual-domain (benchmark vs. in-the-wild) distribution auditing.
   - Indigenous linguistic integrity (*meyyeḻuttu* vs. ISCII traps).
2. **`rigorous-nlp-paper-authoring`**:
   - Commonwealth English spelling standard.
   - Single-author voice alternatives (authorial "we" vs. paper-centric active voice).
   - 0-error Overleaf LaTeX architecture and font fallbacks.
   - ISO 15919 quadruple transliteration standards.
   - Multi-system AI disclosure templates.

---

## 7. Actionable Directives for the Pulli Agent

When transitioning to **Pulli** (the Roman $\leftrightarrow$ Tamil typing, IME, and text normalisation system):

1. **Dual Input Mode Architecture**:
   - **Citation Mode**: Translates letter-name spelling (*thambi* $\to$ தம்பி, *kondaadi* $\to$ கொண்டாடி).
   - **Phonetic / Naturalistic Mode**: Translates conversational acoustic spelling (*thambi* $\to$ தம்பி, *dambi* $\to$ தம்பி, *padam* $\to$ படம்).
2. **Leverage ValiMeli-Speech for IME Evaluation**:
   - Use the $5{,}324$-word manifest in [`artifacts/tamil_speech_full_corpus_manifest.json`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/artifacts/tamil_speech_full_corpus_manifest.json) as a direct gold-standard benchmark for testing Pulli's G2P and phonetic mapping rules.
3. **Strict Linguistic Parsing**:
   - Ensure Pulli uses `segment_eluttu` from [`src/valimeli-benchmark.py`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/src/valimeli-benchmark.py) to preserve atomic consonant clusters (`ட்டி`, `க்க`, `ம்ப`, `ந்த`, `ங்க`) without sub-syllabic Unicode virāma bugs.
4. **Follow the Skills**:
   - Enforce the `empirical-nlp-research-methodology` and `rigorous-nlp-paper-authoring` skills for all Pulli code and documentation.
