---
name: empirical-nlp-research-methodology
description: Comprehensive methodology guide for empirical NLP and computational linguistics research. Covers theoretical bounding (information-theoretic entropy & argmax invariance), multi-seed statistical hygiene, indigenous linguistic integrity, copyleft speech data curation, and reproducible artifact pipelines.
---

# Empirical NLP & Computational Linguistics Research Methodology

This skill provides an end-to-end framework for conducting rigorous, scientifically defensible NLP research, grounded in the methodologies, statistical hygiene, and linguistic practices established in Project ValiMeli.

---

## 1. Core Principles: Information-Theoretic Bounding First

Before proposing complex neural architectures or auxiliary loss objectives, establish the mathematical bounds of the problem:

1. **Quantify Residual Entropy**:
   - Compute unconditional target entropy $H(Y)$ and context-conditional entropy $H(Y \mid X)$.
   - Measure Mutual Information $I(X; Y) = H(Y) - H(Y \mid X)$.
2. **Test for Argmax Invariance**:
   - Check if context-conditioned probabilities $P(y \mid x)$ cross the decision boundary ($0.50$ in binary choice).
   - If $P(y^* \mid x) < 0.50$ across all contexts $x$, an optimal context-conditioned classifier produces zero argmax gain over an unconditional majority baseline ($0.00\%$ error reduction).
   - **Takeaway**: Mutual information below the decision threshold cannot improve standard greedy/beam argmax decoding.

---

## 2. Experimental Hygiene & Statistical Protocols

### A. Multi-Seed Low-Resource Evaluation
* **The Low-Resource Bias**: In regimes with limited training data ($\le 25\text{k}$ pairs), random weight initialisation causes high accuracy variance ($\pm 1.5\%$). Single-seed reports are scientifically unreliable.
* **Mandatory Protocol**:
  - Run all low-resource experiments across at least three distinct random seeds (e.g., Seeds 42, 43, 44).
  - Report exact sample means and standard deviations ($\mu \pm \sigma$).
  - Never claim an intervention advantage unless the delta exceeds inter-seed variance ($\Delta > 2\sigma$) and achieves statistical significance.

### B. Paired Instance-Level Prediction Logs & McNemar Testing
* Save full instance-level JSONL prediction logs (`{"id": i, "input": x, "gold": y, "pred": y_hat, "correct": bool}`).
* Compute **Paired McNemar's Test** on the $2 \times 2$ contingency matrix:
  $$n_{01} = (\text{Base incorrect, Intervention correct}), \quad n_{10} = (\text{Base correct, Intervention incorrect})$$
  $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$
* Report both the discordant pair count ($n_{01} + n_{10}$) and exact $p$-value. Large discordant counts with small net gain demonstrate random churn rather than systematic phonological learning.

### C. Target-Side Duality & Annotator Variance
* When benchmark performance saturates or neural heads fail to transfer:
  - Measure pairwise inter-annotator disagreement on identical inputs carrying multi-reference targets.
  - Audit against in-the-wild, naturalistic corpora (e.g., YouTube comments, social media) to determine whether ambiguity is an inherent property of the spoken language or a citation-spelling artifact of formal crowdsourced benchmarking.

---

## 3. Linguistic & Orthographic Integrity

### A. Distinguish Encoding Artifacts from Orthography
* **The Unicode/ISCII Trap**: Digital encoding standards (like Unicode) often treat vowel-bearing syllables (*uyirmey* / *akṣara*) as base codepoints and pure consonants as base + combining virāma/puḷḷi.
* **Linguistic Reality**: In indigenous Dravidian orthography (*Tolkāppiyam*), pure consonants (*meyyeḻuttu*) are the primary phonemic units. 
* **Rule**: Ensure tokenizers and aligners never create spurious sub-syllabic decomposition artifacts (e.g., mapping coda ம் to $[ma][.]$ or `pat.ama`).

### B. Dual-Terminology Grounding
* Employ indigenous grammatical terms alongside international linguistic equivalents:
  - *Eḻuttu* (syllabic graphemic unit / *akṣara*)
  - *Puḷḷi* (virāma / consonant-vowelless dot)
  - *Vallinam* (hard plosives / stops)
  - *Mellinam* (soft nasals)
  - *Puṇarcci* (morphophonemic *sandhi* assimilation)

---

## 4. Acoustic Grounding & Copyleft Data Curation

### A. Physical Signal Corroboration
* Corroborate abstract phonological rules using raw acoustic waveform signal processing:
  - **STFT Spectrograms**: Compute $0\text{--}5000$\,Hz time-frequency representations.
  - **Voicing Bar**: Detect continuous low-frequency periodic fundamental energy ($F_0 < 300$\,Hz) during nasal-stop junctures.
  - **Closure Gaps**: Measure silent closure duration ($\approx 170$\,ms in geminates) vs. burst duration in singletons.

### B. Copyleft Licensing & Attribution (CC BY-SA 4.0 / CC0)
* When curating open-source speech datasets from Wikimedia Commons / Lingua Libre:
  - Maintain a structured manifest with speaker usernames, direct Wikimedia Commons URLs, and license tags.
  - Include an explicit `AUDIO_ATTRIBUTION.md` detailing contributor profiles and license terms.
  - Respect API rate limits (avoid aggressive burst scraping on binary media endpoints).

---

## 5. Artifact-Driven Reproducibility Pipeline

1. **Code-to-JSON**: All scripts write deterministic metrics to `artifacts/<experiment>_results.json`.
2. **JSON-to-LaTeX**: LaTeX tables (`table_*.tex`) must pull numbers directly from verified JSON artifacts. Zero hand-transcribed metrics.
3. **Paired Logs**: Retain all raw prediction logs in `artifacts/predictions/` for post-hoc significance testing.
