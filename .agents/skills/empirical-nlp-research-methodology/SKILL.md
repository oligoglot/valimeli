---
name: empirical-nlp-research-methodology
description: Comprehensive methodology guide for empirical computational linguistics and NLP research across any language family, typology, or task domain. Covers theoretical & information-theoretic bounding, multi-seed statistical hygiene (McNemar/bootstrap tests), typological & orthographic integrity, multi-modal signal grounding, and reproducible artifact pipelines. For the complete cross-subfield linguistics framework, see the companion skill linguistics-research-methodology.
---

# Universal Empirical NLP & Computational Linguistics Methodology

This skill provides a domain-general, scientifically rigorous framework for conducting empirical computational linguistics, natural language processing (NLP), and speech research across any language family, typology, or subfield.

> [!NOTE]
> For broader research encompassing formal syntax/semantics, psycholinguistics, field documentation, or historical linguistics, use the comprehensive companion skill **`linguistics-research-methodology`**.

---

## 0. Foundational Axiom: Research as a Truth-Seeking Exercise

> [!CAUTION]
> **RESEARCH IS ESSENTIALLY A TRUTH-SEEKING EXERCISE. FABRICATING ANYTHING IS COMPLETELY UNACCEPTABLE.**
> The foundation of scientific inquiry is absolute epistemic honesty. Fabricating, inventing, or synthesizing ANY empirical data, benchmark metric, sample count, model output, statistical test, linguistic phenomenon, theoretical proof step, or academic citation is a catastrophic breach of research integrity.

1. **Zero-Fabrication Across All Dimensions**:
   - **Empirical Numbers & Metrics**: NEVER guess, approximate, or fabricate a number, percentage, count, variance, or $p$-value. Every number must originate from frozen, traceable computational artifacts produced by real code executed on real data.
   - **Citations & Literature**: NEVER fabricate or guess paper titles, author lists, venues, DOIs, URLs, or external findings. Every citation must be resolved against authoritative primary registries (ACL Anthology, Crossref, arXiv API, DBLP, PubMed).
   - **Linguistic Claims**: Ground all phonetic, phonological, orthographic, or grammatical claims in verified primary corpora, acoustic signals, or authoritative treatises.
   - **Mathematical Rigor**: Every theoretical derivation must be mathematically sound with all boundary conditions and assumptions explicitly declared.
2. **Mandatory User Escalation Protocol ("Ask Me If You Can't Get Something")**:
   - If any citation, DOI, baseline number, dataset, or empirical metric cannot be found, accessed, or verified: **STOP AND ASK THE USER IMMEDIATELY**.
   - Transparently state what is missing and ask the user to provide the reference, PDF, or data. Never supply a plausible-sounding proxy or speculative answer.
3. **Strict Attribution & Provenance Separation**:
   - Never attribute figures from internal experiments or replications to an external publication.
   - Internal baselines must be explicitly presented as *this work's replication baseline* (pointing directly to the project's own artifacts and result tables).
   - External benchmark results must be quoted directly from the verified primary publication with exact page/table citation.

---

## 1. Theoretical & Information-Theoretic Bounding Before Modeling

Before proposing neural architectural modifications, prompt pipelines, or auxiliary loss objectives, establish the mathematical and information-theoretic limits of the task:

1. **Quantify Intrinsic Ambiguity & Entropy**:
   - Compute unconditional target entropy $H(Y)$ and context-conditional entropy $H(Y \mid X)$ over empirical corpora.
   - Measure Mutual Information:
     $$I(X; Y) = H(Y) - H(Y \mid X)$$
2. **Evaluate Decision Boundaries & Argmax Invariance**:
   - When evaluating task-conditioned models under standard argmax decoding (greedy search, beam search, Viterbi), check if the context-conditioned posterior shifts the plurality choice:
     $$\hat{y} = \arg\max_{y \in \mathcal{Y}} P(y \mid X)$$
   - If the majority class or token remains dominant across all contexts (i.e. $P(y^* \mid X) < 0.50$ in binary choices, or the plurality candidate never flips), an optimal context-conditioned model achieves **0.00% argmax decision error reduction** over an unconditional majority baseline.
   - **Methodological Rule**: Information gain ($I(X; Y) > 0$) residing below the decision threshold cannot improve standard argmax sequence generation; tasks with this property require multi-modal candidate generation, input-side augmentation, or controlled decoding rather than source-side loss heads.

---

## 2. Experimental Hygiene & Statistical Protocols

### A. Multi-Seed Protocols for Low-Resource & Few-Shot Regimes
* **The Variance Trap**: In low-data regimes ($\le 25\text{k}$ training instances, few-shot prompts, or fine-tuning with limited samples), random parameter initialisation, data shuffling, and batch sampling produce high variance ($\pm 1.0\text{--}2.5\%$). Single-seed reports are scientifically invalid.
* **Protocol**:
  - Run all low-resource experiments across at least **three distinct random seeds** (e.g., Seeds 42, 43, 44).
  - Report exact sample means and standard deviations ($\mu \pm \sigma$).
  - Only claim an empirical advantage if the mean delta $\Delta$ exceeds inter-seed variance ($\Delta > 2\sigma$) and passes statistical significance testing.

### B. Instance-Level Paired Testing & Prediction Logging
* Store full instance-level test predictions in structured JSONL files:
  ```json
  {"id": 0, "input": "...", "gold": "...", "pred": "...", "correct": true, "metadata": {...}}
  ```
* **Categorical / Exact-Match Outputs**: Compute **Paired McNemar's Test** on the $2 \times 2$ contingency matrix:
  $$n_{01} = (\text{Base incorrect, Intervention correct}), \quad n_{10} = (\text{Base correct, Intervention incorrect})$$
  $$\chi^2 = \frac{(|n_{01} - n_{10}| - 1)^2}{n_{01} + n_{10}}$$
  Always report discordant pair counts ($n_{01} + n_{10}$) alongside the exact $p$-value. High discordant counts with near-zero net gain indicate stochastic label churn rather than systematic feature learning.
* **Continuous Metrics (BLEU, chrF, WER, Macro-F1)**: Use Paired Bootstrap Resampling ($B \ge 1{,}000$ iterations) or Wilcoxon Signed-Rank tests.

### C. Isolating Annotator Bias via Dual-Domain Audits
* When benchmark performance plateaus or supervised models fail to transfer to downstream tasks:
  1. **Multi-Reference Variance**: Measure pairwise inter-annotator disagreement on identical inputs carrying multi-reference annotations.
  2. **In-The-Wild Cross-Auditing**: Contrast formal benchmark distributions against naturalistic, conversational, or in-the-wild corpora to verify whether variance is an inherent linguistic property or an artifact of formal citation priors in crowdsourced guidelines.

---

## 3. Typological, Morphological & Orthographic Integrity

### A. Distinguish Encoding Artifacts from Linguistic Units
* **The Digital Storage Artifact**: Digital character encodings (Unicode, ISCII, UTF-8 byte sequences) frequently optimize for storage or rendering rather than linguistic grammar (e.g., decomposing pure consonants into base + combining virāma, splitting Semitic root consonants across vowel templates, separating tone marks from base vowels).
* **Scientific Rule**: Ensure tokenizers, aligners, and evaluation metrics operate over true **linguistic units** (morphemes, phonemes, syllables, orthographic grapheme clusters) rather than raw codepoints.

### B. Dual-Framework Grounding
* Respect indigenous and historical grammatical traditions of the language under study while bridging to standard international typological frameworks:
  - Provide native grammatical terms alongside standard international linguistic equivalents (e.g., *sandhi* / morphophonemic assimilation; *akṣara* / syllabic grapheme unit; *root-and-pattern* / non-concatenative morphology).
  - Adhere to the **Leipzig Glossing Rules** for interlinear morphological glossing.

---

## 4. Multi-Modal Grounding & Acoustic Signal Verification

* When studying phonetics, phonotactics, or speech-text interfaces:
  1. **Physical Signal Corroboration**: Validate theoretical or script-based rules against acoustic waveforms (Short-Time Fourier Transform spectrograms, fundamental frequency $F_0$ pitch tracking, duration of silent closure gaps, voice onset time [VOT]).
  2. **Open-Access Licensing (CC BY-SA / CC0)**: Ensure all harvested speech and text corpora maintain full provenance tracking, contributor attribution manifests, and license compliance.

---

## 5. Deterministic Artifact Pipeline

1. **Code-to-Artifact**: All computational scripts must output metrics to structured JSON/CSV files.
2. **Artifact-to-Manuscript**: Tables in LaTeX manuscripts must draw directly from frozen artifact files.
3. **Reproducibility Guarantee**: The entire paper's tables, figures, and statistical claims must be reproducible via a single non-interactive command line.
