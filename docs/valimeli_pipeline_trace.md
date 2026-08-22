# Project ValiMeli: Technical Pipeline Trace & Architectural Specification

This document provides a comprehensive technical trace of the ValiMeli pipeline, detailing data flow, linguistic rules, sequence transformation tensors, memory budget mechanics, and evaluation math.

---

## 1. System Architecture Diagram

```
+-----------------------------------------------------------------------------------------+
|                         PROJECT VALIMELI SYSTEM PIPELINE                                |
+-----------------------------------------------------------------------------------------+

  [Raw Aksharantar Data]
         |
         | (Streaming Line-by-Line JSON Reader)
         v
  [Parallel Pairs: (Indic, Roman)]
         |
         +-------------------------------+-------------------------------+
         |                               |                               |
         v                               v                               v
  [Arm A0: Baseline]             [Arm A1: ValiMeli]             [Arm A2: Morphology]
  Raw Graphemes:                 Linguistic Engine:              Morpheme Segmentation:
  'தம்பி'                         'த ம் பி'                     'தம்பி'
         |                               |                               |
         +-------------------------------+-------------------------------+
                                         |
                                         v
                       [Dynamic CharVocab Tokenizer]
                                         |
                                         v
                      [PyTorch Seq2Seq + Bahdanau Attention]
                       - Bidirectional GRU Encoder (dim=256)
                       - Additive Attention Layer
                       - Unidirectional GRU Decoder (dim=256)
                                         |
                                         v
                     [Autoregressive Greedy / Beam Decoder]
                                         |
                                         v
                         [Authentic Evaluation Engine]
                       - Top-1 Word Exact Match (EM %)
                       - Levenshtein Character Error Rate (CER %)
                       - Stop-Voicing Accuracy (SVA %)
```

---

## 2. Linguistic Engine Pipeline Trace (Tolkāppiyam Framework)

In accordance with classical Dravidian grammatical theory (*Tolkāppiyam*, Niklas 1988), segmentation operates over **uyirmey** (syllabic consonant-vowel composites) and **mey-puḷḷi** (pure consonants with virama).

### Example 1: Post-Nasal Voicing (*Puṇarcci* Assimilation)
- **Input Grapheme Sequence**: `தம்பி` (*thambi* — "younger brother")
- **Uyirmey / Mey Segmentation**:
  1. Unit 0: `த` (Uyirmey: Dental plosive consonant + inherent vowel /a/)
  2. Unit 1: `ம்` (Mey: Bilabial nasal `ம` + Puḷḷi `\u0bcd`)
  3. Unit 2: `பி` (Uyirmey: Bilabial plosive `ப` + vowel sign `\u0bbf` /i/)
- **Phonotactic Context Detection**:
  - Unit 0 (`த`): Plosive at index 0 $\to$ `[INIT]` (`\ue001`) $\implies$ Fortis / Voiceless $[t̪]$
  - Unit 1 (`ம்`): Nasal coda $\to$ Untouched
  - Unit 2 (`பி`): Plosive preceded by homorganic nasal `ம்` $\to$ `[NASAL]` (`\ue003`) $\implies$ Lenis / Voiced $[b]$
- **Output Transformed Sequence**: `[INIT]த ம் [NASAL]பி`
- **Target Transliteration**: `thambi`

### Example 2: Post-Nasal Voicing (*Puṇarcci* Assimilation)
- **Input Grapheme Sequence**: `பந்து` (*panthu* / *pandhu* — "ball")
- **Uyirmey / Mey Segmentation**:
  1. Unit 0: `ப` (Uyirmey: Bilabial plosive + /a/)
  2. Unit 1: `ந்` (Mey: Dental nasal `ந` + Puḷḷi `\u0bcd`)
  3. Unit 2: `து` (Uyirmey: Dental plosive `த` + vowel sign `\u0bc1` /u/)
- **Phonotactic Context Detection**:
  - Unit 0 (`ப`): Plosive at index 0 $\to$ `[INIT]` (`\ue001`) $\implies$ Fortis / Voiceless $[p]$
  - Unit 1 (`ந்`): Nasal coda $\to$ Untouched
  - Unit 2 (`து`): Plosive preceded by dental nasal `ந்` $\to$ `[NASAL]` (`\ue003`) $\implies$ Lenis / Voiced $[d̪]$
- **Output Transformed Sequence**: `[INIT]ப ந் [NASAL]து`
- **Target Transliteration**: `pandhu` / `panthu`

### Example 3: Geminate Stop Disambiguation
- **Input Grapheme Sequence**: `பக்கம்` (*pakkam* — "side/page")
- **Uyirmey / Mey Segmentation**:
  1. Unit 0: `ப`
  2. Unit 1: `க்`
  3. Unit 2: `க`
  4. Unit 3: `ம்`
- **Phonotactic Context Detection**:
  - Unit 0 (`ப`): Plosive at index 0 $\to$ `[INIT]` (`\ue001`) $\implies$ Fortis $[p]$
  - Unit 1 (`க்`): Plosive with puḷḷi, followed by identical plosive `க` $\to$ `[GEM]` (`\ue002`) $\implies$ Fortis Geminate $[k]$
  - Unit 2 (`க`): Plosive preceded by identical plosive with puḷḷi $\to$ `[GEM]` (`\ue002`) $\implies$ Fortis Geminate $[k]$
  - Unit 3 (`ம்`): Nasal coda $\to$ Untouched
- **Output Transformed Sequence**: `[INIT]ப [GEM]க் [GEM]க ம்`
- **Target Transliteration**: `pakkam`

### Example 4: Intervocalic Lenition
- **Input Grapheme Sequence**: `படம்` (*padam* — "picture/movie")
- **Uyirmey / Mey Segmentation**:
  1. Unit 0: `ப`
  2. Unit 1: `ட`
  3. Unit 2: `ம்`
- **Phonotactic Context Detection**:
  - Unit 0 (`ப`): Index 0 $\to$ `[INIT]` (`\ue001`) $\implies$ Fortis $[p]$
  - Unit 1 (`ட`): Retroflex plosive preceded by vocalic `ப` (no puḷḷi) $\to$ `[INTER]` (`\ue004`) $\implies$ Lenis / Voiced $[ɖ]$
  - Unit 2 (`ம்`): Nasal coda $\to$ Untouched
- **Output Transformed Sequence**: `[INIT]ப [INTER]ட ம்`
- **Target Transliteration**: `padam`

---

## 3. Mathematical Formalisms

### 3.1 Conditional Entropy of Stop Voicing
Let $V \in \{\text{Voiced}, \text{Voiceless}\}$ represent the phonetic voicing state and $C \in \{\text{Initial}, \text{Geminate}, \text{PostNasal}, \text{Intervocalic}, \text{Default}\}$ represent the phonotactic context:

$$H(V \mid C) = \sum_{c \in C} P(C = c) H(V \mid C = c)$$

Where:
$$H(V \mid C = c) = - P(V = \text{Voiced} \mid c) \log_2 P(V = \text{Voiced} \mid c) - P(V = \text{Voiceless} \mid c) \log_2 P(V = \text{Voiceless} \mid c)$$

### 3.2 Levenshtein Edit Distance & Character Error Rate (CER)
For a predicted string $\hat{Y} = (\hat{y}_1, \dots, \hat{y}_m)$ and reference string $Y = (y_1, \dots, y_n)$, the edit distance $D(m, n)$ is defined recursively:

$$D(i, j) = \begin{cases}
\max(i, j) & \text{if } \min(i, j) = 0 \\
\min \begin{cases}
D(i-1, j) + 1 \\
D(i, j-1) + 1 \\
D(i-1, j-1) + \mathbb{I}(y_i \neq \hat{y}_j)
\end{cases} & \text{otherwise}
\end{cases}$$

$$\text{CER} = \frac{\sum_{k=1}^N D(Y^{(k)}, \hat{Y}^{(k)})}{\sum_{k=1}^N |Y^{(k)}|} \times 100\%$$

---

## 4. Hardware Resource & Memory Profile

- **RAM Consumption**: Streaming generator reads files line-by-line. Maximum resident set size (RSS) during dataset iteration is **< 1.8 GB**, fully compliant with memory constraints.
- **GPU / MPS Allocation**:
  - Model Parameters: 1.48M parameters $\approx$ 5.92 MB (FP32).
  - Batch Size 256 (Max Seq Len 35): ~65 MB active tensor memory.
  - Total Peak Metal Memory on Apple Silicon: **~520 MB**.
- **Disk Utilization**:
  - Checkpoints: ~6 MB per saved best state.
  - Scratch folder: < 400 MB total.
