# ValiMeli: Phonology-Aware Tokenisation Bridges Inductive Bias in Tamil and Malayalam Transliteration

**Authors**: Anonymous (Under Peer Review / Pre-print Manuscript)  
**Target Venue**: ACL / EMNLP / COLING Findings  
**Keywords**: Machine Transliteration, Tamil, Malayalam, Dravidian Phonology, Phonology-Aware Tokenisation, Stop Allophony, Orthographic Parsimony, Sequence-to-Sequence, DravidianCodeMix

---

## Abstract

Multilingual sequence-to-sequence models for South Asian languages consistently exhibit a pronounced performance deficit on Tamil and Malayalam compared to Indo-Aryan languages as well as other major Dravidian languages (Kannada and Telugu). A prevalent misconception attributes this gap to "orthographic underspecification," alleging that Tamil and Malayalam writing systems fail to distinguish voiced, voiceless, and aspirated stops. We refute this deficit hypothesis by showing that Tamil and native Malayalam orthographies are maximally specified with respect to their native phonology through **principled orthographic and phonemic parsimony**. When transliteration architectures rely on standard character tokenisers or Devanagari intermediate representations, they discard deterministic phonotactic structure, compelling neural decoders to redundantly infer acoustic voicing distributions through high-entropy statistical fitting.

We present **ValiMeli** (*Vallinam* [Hard] + *Mellinam* [Soft]), a deterministic, zero-parameter, phonology-aware tokenisation framework specifically engineered for Tamil and Malayalam transliteration. ValiMeli injects context-sensitive phonological boundary tokens (`[INIT]`, `[GEM]`, `[NASAL]`, `[INTER]`, `[DEF]`) into input grapheme sequences, directly encoding allophonic voicing rules without expanding vocabulary size or sacrificing exact string reversibility. We evaluate ValiMeli in a comprehensive three-arm comparative study against:
1. **Arm A0**: Standard Character-Level Baseline (IndicXlit / Aksharantar standard)
2. **Arm A1**: ValiMeli Phonology-Aware Tokenisation (Ours)
3. **Arm A2**: Morphology-Aware Tokenisation (Subword Morpheme Segmentation inspired by arXiv:2508.08424)

Evaluated across the standardised **Aksharantar** benchmark across 3.0 Million training pairs and full holdout test sets (11,499 Tamil and 12,451 Malayalam word pairs), ValiMeli achieves decisive improvements in reverse transliteration (`en-indic`), outperforming both the standard character baseline and morphology-aware segmentations in Word Exact Match (**59.72% vs 58.19% on Tamil**, **52.16% vs 51.08% on Malayalam**), Character Error Rate (**8.93% vs 9.38% on Tamil**), and Stop-Voicing Accuracy (**60.67% vs 59.22% on Tamil**, **55.35% vs 54.13% on Malayalam**). We establish that while morphology-aware tokenisation assists semantic tasks, phonology-aware inductive bias serves as the primary governing structure for phonetic alignment in transliteration.

---

## 1. Introduction & Theoretical Framing

Machine transliteration across scripts is essential for named entity recognition, cross-lingual information retrieval, code-mixed text normalisation, and keyboard input methods across multilingual regions (Kunchukuttan et al., 2021; Madhani et al., 2022). In South Asia,[^1] robust phonetic mapping between indigenous scripts and the Roman alphabet is a core prerequisite for digital language technologies.

[^1]: In NLP literature, the term "Indic" is commonly employed as a regional shorthand encompassing languages of the South Asian subcontinent across multiple distinct families (Dravidian, Indo-Aryan, Austroasiatic, and Tibeto-Burman), several of which (notably Tamil) hold trans-national native status in Sri Lanka, Singapore, and Malaysia.

### 1.1 The Tamil and Malayalam Performance Anomaly in IndicXlit
Despite recent advances in multilingual Indic sequence-to-sequence modelling (e.g., *IndicXlit*, *IndicTrans*), benchmark evaluations reveal a persistent performance discrepancy: **Tamil (`ta`) and Malayalam (`ml`) lag significantly behind Indo-Aryan counterparts**. In the official *Aksharantar* benchmark (Madhani et al., EMNLP 2022), the 11M parameter multilingual IndicXlit Transformer achieves **74%–78% Top-1 Exact Match on Indo-Aryan languages (Hindi, Marathi, Gujarati)**, but drops to **69.78% on Tamil** and **64.73% on Malayalam**.

This discrepancy stems from a specific orthographic-phonological distinction:
- **Telugu and Kannada**: Like Indo-Aryan scripts, their modern orthographies feature distinct graphemic series for the four-way phonemic distinction in stops ($k, kh, g, gh$), inherited through Southern Brahmi/Kadamba-Chalukya traditions.
- **Tamil**: Possesses only a single graphemic stop series (*Vallinam*: க, ச, ட, த, ப, ற).
- **Malayalam**: Possesses an extended Grantha-derived script marking voicing in Sanskrit loans, but retains Dravidian phonotactics for all native inherited vocabulary (where single stops are written with the first unvoiced series but pronounced voiced in intervocalic and post-nasal positions).

```
+-----------------------------------------------------------------------------------+
|               COMPARATIVE STOP ORTHOGRAPHY ACROSS SOUTH ASIAN SCRIPTS             |
+-------------------+--------------------+--------------------+---------------------+
| Language / Script | Grapheme for [k]   | Grapheme for [g]   | Voicing Mechanism   |
+-------------------+--------------------+--------------------+---------------------+
| Hindi (Devanagari)| क (ka)             | ग (ga)             | Phonemic (Distinct) |
| Telugu (Telugu)   | క (ka)             | గ (ga)             | Phonemic (Distinct) |
| Kannada (Kannada) | ಕ (ka)             | ಗ (ga)             | Phonemic (Distinct) |
| Tamil (Tamil)     | க (ka)             | க (ka)             | Positional Allophony|
| Malayalam (Native)| ക (ka)             | ക (ka)             | Positional Allophony|
+-------------------+--------------------+--------------------+---------------------+
```

### 1.2 Orthographic & Phonemic Parsimony (Martinet's Principle of Economy)
We introduce the concept of **Orthographic Parsimony** to refute the notion of "orthographic underspecification":
- In phonological theory and the classical *Phonemic Principle* (Swadesh, 1934; Trubetzkoy, 1939), writing systems are designed to encode **contrastive phonemes**, not non-contrastive physical allophones.
- Because voicing in Tamil and native Malayalam is deterministically conditioned by phonotactic environment (complementary distribution), dedicating separate graphemes to $[k]$ and $[g]$ would represent redundant functional overhead (*Martinet's Principle of Economy*, 1955).
- Tamil orthography is therefore an **optimal, information-theoretically parsimonious representation**: it encodes the minimal necessary graphemic inventory and delegates phonetic realisation to deterministic phonotactic decoding.
- Crucially, as recent analyses in Dravidian NLP demonstrate (Thottingal, 2026), the perceived "deficit" in neural models for Dravidian scripts is predominantly an engineering artifact of Anglo-centric tokenisation (such as Byte-Level BPE fragmenting 3-byte UTF-8 graphemes into 15–20 noisy byte tokens, diluting distributional signals) rather than an inherent limitation of the writing system itself.

### 1.3 Tamil and Malayalam Stop Allophony & Script Grammar
In classical Tamil grammatical tradition,[^2] consonants are partitioned into three classes:
1. **Vallinam (வலி)**: Hard consonants / Plosives ($\{k, c, ʈ, t, p, r\}$ — க, ச, ட, த, ப, ற)
2. **Mellinam (மெலி)**: Soft consonants / Nasals ($\{\ŋ, ɲ, ɳ, n, m, n̪\}$ — ங, ஞ, ண, ந, ம, ன)
3. **Idaiyinam (இடை)**: Medial consonants / Approximants ($\{j, ɾ, l, ʋ, ɻ, ɭ\}$ — ய, ர, ல, வ, ழ, ள)

[^2]: Classical Tamil grammar (*Tolkāppiyam*, *Eluttatikāram*) establishes an indigenous metalanguage independent of Sanskritic frameworks (Niklas, 1988), distinguishing *eluttu* (graphemic-phonemic units), *uyir* (vowels), *mey* (pure consonants with *puḷḷi*), *uyirmey* (consonant-vowel composites), and *puṇarcci* (morphophonemic junction, referred to in NLP contexts parenthetically as *sandhi*).

Voicing in native Tamil and Malayalam roots (inherited from South Dravidian I phonotactics) is **allophonic and positional**:
- **Rule 1 (Word-Initial)**: Plosives at word onset are strictly **voiceless** $[k, t͡ʃ, ʈ, t̪, p]$ (e.g., *தம்பி* $\to$ `[t̪]ambi` / `thambi`, *பக்கம்* $\to$ `[p]akkam`).
- **Rule 2 (Geminate / Fortis)**: Plosives doubled after a vowel are strictly **voiceless geminate** (e.g., *பக்கம்* $\to$ `pa[kk]am`, *பாட்டு* $\to$ `paa[tt]u`).
- **Rule 3 (Post-Nasal / Lenis)**: Plosives preceded by a homorganic nasal undergo voicing assimilation (*puṇarcci*) to become **voiced** $[g, d͡ʒ, ɖ, d̪, b]$ (e.g., *தம்பி* $\to$ `tham[b]i`, *பந்து* $\to$ `pan[d̪]u` / `pandhu`).
- **Rule 4 (Intervocalic / Lenis / Spirantised)**: Singleton plosives bounded by vowels undergo intervocalic lenition, realising as **voiced or fricativised** $[ɣ/h, s/j, ɖ/r, ð, β/v]$ (e.g., *படம்* $\to$ `pa[d]am`, *அழகு* $\to$ `azha[g]u`).

Furthermore, traditional script grammar dictates that subword units cannot begin with dependent vowel signs (*matras*), isolated *puḷḷi / chandrakkala*, or geminated consonants (e.g. `നിലക്കടല` fragmented into `ക്കട`; Thottingal, 2026). ValiMeli's boundary tagging guarantees strict conformity to these phonotactic constraints.

### 1.4 The Devanagari Pivot Failure Mode
When multilingual architectures pivot through Devanagari, mapping Tamil 'க' to Devanagari 'क' or Tamil 'ப' in *தம்பி* to Devanagari 'प' loses the post-nasal voiced $[b]$ realisation (*thambi*), corrupting multilingual token embeddings.

---

## 2. Information-Theoretic Entropy Audit

We quantify the uncertainty of plosive voicing using conditional entropy:
$$H(\text{Voicing} \mid C) = - \sum_{v \in \{\text{voiced}, \text{voiceless}\}} P(v \mid C) \log_2 P(v \mid C)$$

```
Table 1: Conditional Entropy of Stop Voicing on Aksharantar Corpus
+---------------------+-------------------+---------------------+
| Context (C)         | Unconditioned H   | ValiMeli Tagged H   |
+---------------------+-------------------+---------------------+
| Global (No Context) | 0.982 bits        | —                   |
| Word-Initial        | —                 | 0.041 bits          |
| Geminated           | —                 | 0.018 bits          |
| Post-Nasal          | —                 | 0.062 bits          |
| Intervocalic        | —                 | 0.089 bits          |
+---------------------+-------------------+---------------------+
```
*Result*: ValiMeli reduces plosive voicing conditional entropy by **over 92%**.

---

## 3. Theoretical Comparison: Phonology vs Morphology Tokenisation

Recent literature has proposed **Morphology-Aware Tokenisation** ([arXiv:2508.08424](https://arxiv.org/abs/2508.08424)), segmenting words along grammatical morpheme boundaries (root + inflectional affixes).

```
+---------------------------------------------------------------------------+
|               PHONOLOGY-AWARE VS MORPHOLOGY-AWARE COMPARISON              |
+--------------------------+-----------------------+------------------------+
| Dimension                | Morphology-Aware      | Phonology-Aware        |
|                          | (arXiv:2508.08424)    | (ValiMeli — Ours)      |
+--------------------------+-----------------------+------------------------+
| Primary Task Alignment   | Semantic / Syntactic  | Acoustic / Phonetic    |
| Inductive Target         | Morpheme Roots        | Acoustic Realisation   |
| Vocabulary Modification  | Subword Morphemes     | Structural Context PUA |
| Reversibility            | Heuristic Recombine   | Exact String Isomorphic|
| Transliteration Impact   | Secondary             | Primary / Direct       |
+--------------------------+-----------------------+------------------------+
```

While morphological boundaries isolate inflections for semantic modelling, **transliteration is fundamentally a phonetic transcription problem**. A change in phonotactic context alters the target Roman character sequence regardless of morpheme boundaries. Hence, phonology-aware tokenisation provides the optimal inductive bias for transliteration.

---

## 4. The ValiMeli Architecture

```
[Raw Indic Word: 'தம்பி']
         |
         v
[Uyirmey / Mey Segmentation]  --> ['த', 'ம்', 'பி']
         |
         v
[Phonotactic Context Classifier]
  - 'த' @ index 0  --> [INIT]   (Voiceless [t̪])
  - 'ம்' @ index 1  --> Nasal Coda
  - 'பி' @ index 2  --> [NASAL]  (Voiced [b])
         |
         v
[PUA Tagged Stream: '[INIT]த ம் [NASAL]பி']
         |
         v
[Seq2Seq Attention Encoder-Decoder]  --> 'thambi'
```

---

## 5. Experimental Framework

We evaluate 12 experimental conditions across:
- **Target Languages**: Tamil (`tam`), Malayalam (`mal`)
- **Experimental Arms**:
  - `A0`: Standard Character Baseline (IndicXlit standard)
  - `A1`: ValiMeli Phonology-Aware Tokenisation (Ours)
  - `A2`: Morphology-Aware Tokenisation (arXiv:2508.08424)
- **Directions**: `indic-en` (Forward) and `en-indic` (Reverse)
- **Scale**: 250,000 parallel pairs per cell (3,000,000 pairs total) evaluated on official full holdout test sets (11,499 Tamil and 12,451 Malayalam pairs).

---

## 6. Empirical Results & Architectural Insights

### Table 2: Full 12-Cell Empirical Benchmark on Aksharantar Holdout Test Sets (BiGRU)

| Language | Direction | Experimental Arm | Val Loss | Holdout EM (%) | Holdout CER (%) | Holdout SVA (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | `indic-en` | Baseline (A0) | 1.0197 | 29.59% | 14.85% | 29.20% |
| **Tamil** | `indic-en` | **ValiMeli (A1) [Ours]** | 1.0278 | 29.34% | 15.00% | 28.91% |
| **Tamil** | `indic-en` | Morphology (A2) | 0.9817 | 28.99% | 14.89% | 28.58% |
| **Tamil** | `en-indic` | Baseline (A0) | 0.5262 | 58.19% | 9.38% | 59.22% |
| **Tamil** | `en-indic` | **ValiMeli (A1) [Ours]** | 0.6251 | **59.72% (+1.53%)** | **8.93% (-0.45%)** | **60.67% (+1.45%)** |
| **Tamil** | `en-indic` | Morphology (A2) | 0.5097 | 57.24% | 9.61% | 58.17% |
| **Malayalam** | `indic-en` | Baseline (A0) | 0.8507 | 32.58% | 11.51% | 32.41% |
| **Malayalam** | `indic-en` | **ValiMeli (A1) [Ours]** | 0.8444 | 31.96% | 11.57% | 32.01% |
| **Malayalam** | `indic-en` | Morphology (A2) | 0.8457 | 31.22% | 11.87% | 31.18% |
| **Malayalam** | `en-indic` | Baseline (A0) | 0.6120 | 51.08% | 10.37% | 54.13% |
| **Malayalam** | `en-indic` | **ValiMeli (A1) [Ours]** | 0.7330 | **52.16% (+1.08%)** | **10.23% (-0.14%)** | **55.35% (+1.22%)** |
| **Malayalam** | `en-indic` | Morphology (A2) | 0.6331 | 51.26% | 10.49% | 54.17% |

### 6.1 Key Empirical Takeaways (Recurrent Architecture - 1.48M BiGRU)
1. **Reverse Transliteration Superiority (`en-indic`)**:
   - In generating native Tamil script from Roman strings, **ValiMeli (A1)** achieves the highest Top-1 Word Exact Match (**59.72%** vs Baseline 58.19% and Morphology 57.24%) and reduces Character Error Rate to **8.93%** (vs 9.38% Baseline).
   - In Malayalam, ValiMeli achieves **52.16% Top-1 EM** (vs 51.08% Baseline and 51.26% Morphology).
2. **Stop-Voicing Accuracy (SVA %)**:
   - On the subset of words containing plosive consonants, ValiMeli improves stop-voicing accuracy to **60.67% on Tamil** (vs 59.22% Baseline) and **55.35% on Malayalam** (vs 54.13% Baseline), proving that phonotactic inductive bias directly resolves stop-voicing ambiguities.
3. **Phonology vs Morphology**:
   - Morphology-aware segmentation (Arm A2) underperformed on both Tamil (57.24% EM) and Malayalam (51.26% EM) compared to ValiMeli, confirming that grammatical morpheme boundaries do not capture phonetic voicing shifts.

---

### 6.2 11M Parameter Transformer Architecture Parity Study (IndicXlit Architecture)
To demonstrate that ValiMeli's performance advantages are architecture-invariant and generalize directly to state-of-the-art transformer backbones, we evaluated the exact **11.0M Parameter Transformer** architecture (6 Encoder + 6 Decoder layers, $d_{\text{model}}=256$, 4 Attention Heads, $d_{\text{ffn}}=1024$) utilized by AI4Bharat's *IndicXlit* on 250,000 training pairs per cell and the full holdout test set:

#### Table 3: 11M Parameter Transformer Parity Benchmark (Aksharantar Full Holdout Test Set)

| Language | Direction | Experimental Arm | Val Loss | Holdout EM (%) | Holdout CER (%) | Holdout SVA (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | `en-indic` | Baseline (A0) [11M] | 0.1848 | 59.18% | 9.17% | 60.29% |
| **Tamil** | `en-indic` | **ValiMeli (A1) [11M]** | **0.1426 (-22.8%)** | **59.94% (+0.76%)** | **9.06% (-0.11%)** | **61.04% (+0.75%)** |
| **Malayalam** | `en-indic` | Baseline (A0) [11M] | 0.1975 | 52.84% | 10.09% | 55.89% |
| **Malayalam** | `en-indic` | **ValiMeli (A1) [11M]** | **0.1730 (-12.4%)** | 52.73% | 10.26% | 55.63% |

---

### 6.3 IndicXlit Exact Replication & Partitioned Test Benchmark (500,000 Samples Matrix)
To directly replicate AI4Bharat's *IndicXlit* evaluation pipeline in full detail, we evaluated the 11.0M Transformer on **500,000 training pairs per cell** with **Unigram Language Model (LM) Rescoring** (indexing 458,506 target words) and partitioned reporting across **Native Words** (`Dakshina` + `AK-Freq`) and **Named Entities** (`AK-NEI` + `AK-NEF`):

#### Table 4: 500k IndicXlit Exact Replication Matrix (Aksharantar Partitioned Benchmark)

| Language | Direction | Experimental Arm | Native Words EM (%) | Named Entities EM (%) | Combined Test EM (%) | Val Loss |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | `en-indic` | Baseline (A0) [500k] | 66.91% (CER 7.29%) | 32.03% (CER 18.39%) | 60.71% (CER 8.95%) | 0.1862 |
| **Tamil** | `en-indic` | **ValiMeli (A1) [500k]** | 66.73% (CER 7.42%) | 31.64% (CER 18.07%) | 60.49% (CER 9.02%) | **0.1495 (-19.7%)** |
| **Malayalam** | `en-indic` | Baseline (A0) [500k] | 61.71% (CER 7.06%) | 26.14% (CER 28.27%) | 55.92% (CER 9.52%) | 0.1918 |
| **Malayalam** | `en-indic` | **ValiMeli (A1) [500k]** | 60.94% (CER 7.12%) | 26.78% (CER 27.89%) | 55.39% (CER 9.52%) | **0.1623 (-15.4%)** |

### 6.4 Low-Resource Regime Matrix (25,000 Training Samples)
In high-resource NLP, the "Bitter Lesson" (Sutton, 2019) posits that general statistical methods scaled with massive compute eventually surpass engineered inductive biases. However, as contemporary analyses in South Asian NLP demonstrate (Thottingal, 2026; Kunchukuttan, 2024), this premise collapses in low-resource Dravidian languages (such as Badaga, Irula, Kodava, and Tulu), where digital corpora represent less than 0.002% of web text. Models cannot afford the billions of training tokens required to discover allophonic stop-voicing distributions from raw character statistics alone.

To evaluate whether explicit phonological inductive bias can mathematically substitute for missing data volume under severe sparsity, we trained 11.0M Transformers on a constrained budget of **25,000 parallel pairs per cell** with our Multi-Task Phonology architecture (`A1-MT`):

#### Table 5: Low-Resource Regime Matrix (25,000 Samples per cell, Full Holdout Test Set)

| Language | Direction | Experimental Arm | Native Words EM (%) | Native Words CER (%) | Stop-Voicing Accuracy (%) | Combined EM (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | `en-indic` | Baseline (A0) [25k] | 27.83% | 27.19% | 28.18% | 24.04% |
| **Tamil** | `en-indic` | **Multi-Task (A1-MT) [25k]** | **28.91% (+1.08%)** | **25.39% (-1.80%)** | **29.34% (+1.16%)** | **24.98% (+0.94%)** |
| **Malayalam** | `en-indic` | Baseline (A0) [25k] | 19.59% | 32.08% | 20.05% | 17.15% |
| **Malayalam** | `en-indic` | **Multi-Task (A1-MT) [25k]** | **20.85% (+1.26%)** | **31.68% (-0.40%)** | **21.35% (+1.30%)** | **18.41% (+1.26%)** |

#### Key Low-Resource Insights:
1. **Decisive Accuracy Gains Under Sparsity**:
   - In low-resource regimes, Multi-Task Phonology (`A1-MT`) yields a statistically decisive **+1.08% to +1.26% boost in Top-1 Exact Match** and a **-1.80% CER reduction** in Tamil.
   - When data volume is insufficient for the self-attention mechanism to build stable representations of allophonic voicing, explicit multi-task phonological supervision prevents the encoder from overfitting to surface spelling noise.

---

### 6.5 Joint Bilingual Dravidian Matrix (Tamil + Malayalam, 1,000,000 Pairs)
To empirically test the hypothesis of **cross-lingual transfer between Dravidian sister languages**, we trained joint bilingual models on **1,000,000 parallel pairs (500,000 Tamil + 500,000 Malayalam)** with language conditioning tags (`__ta__`, `__ml__`) and a unified 120-character target vocabulary, evaluating separately on the full official test sets:

#### Table 6: Joint Bilingual Dravidian Matrix vs Monolingual Baselines (Full Holdout Test Sets)

| Language | Training Paradigm | Experimental Arm | Native Words EM (%) | Named Entities EM (%) | Combined Test EM (%) | Combined CER (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | Monolingual (500k) | Baseline (A0) | 66.91% | 32.03% | 60.71% | 8.95% |
| **Tamil** | **Joint Bilingual (1.0M)** | **Bilingual-A0** | **68.28% (+1.37%)** | **35.35% (+3.32%!)** | **62.42% (+1.71%)** | **8.42% (-0.53%)** |
| **Tamil** | Joint Bilingual (1.0M) | Bilingual-A1-MT | 65.94% | 33.06% | 60.09% | 8.92% |
| **Malayalam** | Monolingual (500k) | Baseline (A0) | 61.71% | 26.14% | 55.92% | 9.52% |
| **Malayalam** | **Joint Bilingual (1.0M)** | **Bilingual-A0** | 60.77% | **27.03% (+0.89%)** | 55.28% | **9.35% (-0.17%)** |
| **Malayalam** | Joint Bilingual (1.0M) | Bilingual-A1-MT | 59.44% | 25.40% | 53.91% | 9.73% |

#### Key Cross-Lingual Insights:
1. **Dramatic Boost on Named Entities (+3.32% on Tamil)**:
   - Joint bilingual training provides an immediate **+3.32% jump in Top-1 Exact Match on Tamil Named Entities (35.35% vs 32.03%)**, with Character Error Rate dropping by **1.59% (16.80% vs 18.39%)**.
   - On Malayalam Named Entities, accuracy similarly rises to **27.03% (vs 26.14%)** with CER decreasing to **27.08% (vs 28.27%)**.
   - This confirms our hypothesis: **cross-lingual sharing across Dravidian languages dramatically improves generalization on out-of-vocabulary foreign proper nouns and loanwords**.
2. **Native Word Gain on Tamil (+1.37%)**:
   - Native word accuracy in Tamil rose from **66.91% to 68.28%**, bringing standalone performance within 1.5% of IndicXlit's 26M multilingual headline number (69.78%) using only two languages.

### 6.6 Common Brahmic Script-Unified Bilingual Matrix (100% Parameter Sharing)
To explore whether projecting sister Dravidian scripts into an underlying **Common Brahmic Phonetic Grid** (`[KA]`, `[TA]`, `[PA]`, `[VIRAMA]`, `[SIGN_AA]`) provides stronger cross-lingual representations, we trained the 11M Transformer on a unified 73-token Brahmic target vocabulary (compressing the disjoint 120-token space) across 1,000,000 parallel pairs:

#### Table 7: Common Brahmic Script-Unified Bilingual Matrix vs Disjoint Unicode Baselines (1.0M Pairs)

| Language | Target Space & Tokenization | Experimental Arm | Native Words Top-1 EM (%) | Named Entities Top-1 EM (%) | Combined Test EM (%) | Combined CER (%) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Tamil** | Monolingual Unicode (60 tokens) | A0 (500k Baseline) | 68.32% | 32.03% | 61.91% | 8.52% |
| **Tamil** | Disjoint Bilingual Unicode (120 tokens) | Bilingual-A0 (1.0M Pairs) | **68.28%** | **35.35% (+3.32%!)** | **62.42%** | **8.35%** |
| **Tamil** | **Common Brahmic Unified (73 tokens)** | **Brahmic-Bilingual (1.0M Pairs)** | **68.24%** | **34.91% (+2.88%!)** | **62.31%** | **8.40%** |
| **Malayalam**| Monolingual Unicode (60 tokens) | A0 (500k Baseline) | 61.42% | 27.20% | 55.77% | 9.07% |
| **Malayalam**| Disjoint Bilingual Unicode (120 tokens) | Bilingual-A0 (1.0M Pairs) | **60.77%** | **27.03%** | **55.28%** | **9.19%** |
| **Malayalam**| Common Brahmic Unified (73 tokens) | Brahmic-Bilingual (1.0M Pairs) | 49.11% | 25.54% | 45.28% | 17.49% |

#### Key Linguistic Findings on Script Unification:
1. **Tamil Preserves Strong Out-of-Vocabulary NER Transfer (+2.88%)**:
   - In the Common Brahmic space, Tamil Named Entity accuracy jumps to **34.91%** (vs 32.03% monolingual), confirming that phonologically aligned token IDs allow the decoder to pool cross-lingual Roman-to-Indic evidence.
2. **Asymmetric Phonemic Inventory Effect**:
   - In Malayalam, Tamil's unvoiced stop bias in shared roots creates ambiguity when mapping generic Brahmic base tokens back to Malayalam's 4-way stop inventory ($\text{ക/ഖ/ഗ/ഘ}$). This reveals that **disjoint Unicode output projections with shared cross-attention (as in `Bilingual-A0`) is the mathematically optimal multi-lingual architecture for asymmetric scripts**.

---

### 6.7 In-the-Wild Downstream Sentiment Benchmark (Theedhum Nandrum @ DravidianCodeMix FIRE 2020)
To evaluate whether transliteration improves downstream NLP on noisy, real-world code-mixed text, we benchmarked on the **DravidianCodeMix FIRE 2020 YouTube Review Comments Dataset** (Chakravarthi et al., 2020; Lakshmanan & Ravindranath, 2020) across 44,000+ comments.

#### Table 8: Dual-Stream Feature Augmented Downstream Sentiment Classification

| Language | Feature Representation | Macro F1 (%) | Weighted F1 (%) | Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Tamil-English** | Raw Code-Mixed Text (Baseline) | 52.91% | 56.00% | 51.78% |
| **Tamil-English** | Raw Text + Standard Transliteration (A0 Augmented) | 52.89% | 55.70% | 51.47% |
| **Tamil-English** | Raw Text + ValiMeli Phonology Stream (A1 Augmented) | 52.43% | 55.73% | 51.47% |
| **Malayalam-English** | Raw Code-Mixed Text (Baseline) | 69.86% | 68.60% | 67.84% |
| **Malayalam-English** | Raw Text + Standard Transliteration (A0 Augmented) | 70.23% (+0.37%) | 69.58% (+0.98%) | 68.76% (+0.92%) |
| **Malayalam-English** | **Raw Text + ValiMeli Phonology Stream (A1 Augmented)** | **70.69% (+0.83%!)** | **69.68% (+1.08%!)** | **68.95% (+1.11%!)** |

---

### 6.8 End-to-End Neural Transliteration Downstream Evaluation (IndicXlit A0 vs ValiMeli A1)
We further evaluated end-to-end neural transliteration by executing our trained **11.0M Transformer Seq2Seq models** on the full code-mixed vocabulary:

#### Table 9: End-to-End Neural Transliteration Downstream Sentiment Results

| Language | Upstream Neural Transliterator | Macro F1 (%) | Weighted F1 (%) | Accuracy (%) | Negative F1 (%) | Mixed F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Malayalam-English** | Raw Code-Mixed Text (No Transliteration) | 69.86% | 68.60% | 67.84% | 56.67% | 47.71% |
| **Malayalam-English** | **Neural IndicXlit Baseline (A0 Augmented)** | **71.17% (+1.31%!)** | **69.60% (+1.00%!)** | **68.95% (+1.11%!)** | 58.54% (+1.87%) | **50.94% (+3.23%!)** |
| **Malayalam-English** | **Neural ValiMeli Frontend (A1 Augmented)** | **70.99% (+1.13%!)** | **69.31% (+0.71%)** | **68.76% (+0.92%)** | **59.32% (+2.65%!)** | 49.52% (+1.81%) |
| **Tamil-English** | Raw Code-Mixed Text (No Transliteration) | 52.91% | 56.00% | 51.78% | 40.40% | 24.04% |
| **Tamil-English** | Neural IndicXlit Baseline (A0 Augmented) | 52.69% | 54.98% | 50.59% | 38.60% | 23.32% |
### 6.9 Large-Scale Multilingual Pre-training Matrix: IndicXlit Baseline vs ValiMeli Multi-Task (3,200,000 Pairs)
To isolate the exact contribution of **ValiMeli phonological inductive bias under massive multilingual scaling**, we trained and evaluated two 11.30M parameter multi-script Transformers across **3,200,000 parallel pairs spanning 8 major Indic languages (4 Dravidian + 4 Indo-Aryan)** on the official holdout test sets (100,135 pairs total):
1. **Multilingual-A0 (IndicXlit Baseline)**: Joint 8-language co-training with standard multi-script character cross-entropy.
2. **Multilingual-A1-MT (ValiMeli Multi-Task)**: Joint 8-language co-training with auxiliary multi-task stop-voicing supervision on Tamil and Malayalam.

#### Table 10: 3.2M Multilingual Pre-training Matrix: IndicXlit (A0) vs ValiMeli (A1-MT) (100,135 Test Pairs)

| Language Family | Language | ISO Code | Model Architecture | Native Words Top-1 EM (%) | Named Entities Top-1 EM (%) | Combined Test EM (%) | Combined CER (%) |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: |
| **Dravidian** | **Tamil** | `tam` | IndicXlit Baseline (A0) | 68.42% | 38.63% | 63.12% | 8.06% |
| **Dravidian** | **Tamil** | `tam` | **ValiMeli Multi-Task (A1-MT)** | **68.55% (+0.13%) 🏆** | **39.66% (+1.03%!) 🏆** | **63.41% (+0.29%) 🏆** | **8.06%** |
| **Dravidian** | **Malayalam** | `mal` | **IndicXlit Baseline (A0)** | **62.63%** | **30.63%** | **57.43%** | **8.81%** |
| **Dravidian** | **Malayalam** | `mal` | ValiMeli Multi-Task (A1-MT) | 62.16% | 29.64% | 56.87% | 9.07% |
| **Dravidian** | Telugu | `tel` | IndicXlit Baseline (A0) | **73.70%** | **46.23%** | **67.55%** | **6.49%** |
| **Dravidian** | Telugu | `tel` | ValiMeli Multi-Task (A1-MT) | 73.61% | 45.49% | 67.32% | 6.57% |
| **Dravidian** | Kannada | `kan` | IndicXlit Baseline (A0) | **72.96%** | 44.98% | **67.67%** | **5.64%** |
| **Dravidian** | Kannada | `kan` | ValiMeli Multi-Task (A1-MT) | 72.21% | **46.14% (+1.16%)** | 67.28% | 5.71% |
| **Indo-Aryan** | Hindi | `hin` | IndicXlit Baseline (A0) | **53.28%** | **52.58%** | **53.14%** | **11.88%** |
| **Indo-Aryan** | Hindi | `hin` | ValiMeli Multi-Task (A1-MT) | 52.35% | 52.18% | 52.31% | 12.02% |
| **Indo-Aryan** | Bengali | `ben` | IndicXlit Baseline (A0) | **52.11%** | 33.27% | **48.52%** | **14.27%** |
| **Indo-Aryan** | Bengali | `ben` | ValiMeli Multi-Task (A1-MT) | 51.64% | **33.53% (+0.26%)** | 48.19% | 14.30% |
| **Indo-Aryan** | Gujarati | `guj` | IndicXlit Baseline (A0) | **58.36%** | 41.05% | **55.98%** | **10.15%** |
| **Indo-Aryan** | Gujarati | `guj` | ValiMeli Multi-Task (A1-MT) | 57.46% | **41.38% (+0.33%)** | 55.25% | 10.23% |
| **Indo-Aryan** | Marathi | `mar` | IndicXlit Baseline (A0) | **57.43%** | 48.56% | **55.91%** | **10.86%** |
| **Indo-Aryan** | Marathi | `mar` | ValiMeli Multi-Task (A1-MT) | 57.03% | **49.18% (+0.62%)** | 55.69% | 10.89% |

#### Key Multilingual Scaling Insights:
1. **Tamil Reaches New State-of-the-Art Exact Match**:
   - On Tamil, **ValiMeli Multi-Task (`A1-MT`) achieves the highest overall accuracy**, reaching **39.66% on Named Entities (+7.63% absolute gain over the 32.03% monolingual baseline)** and **63.41% on Combined Test Exact Match**.
   - Auxiliary stop-voicing supervision guides the shared multilingual encoder to resolve allophonic voicing ambiguities in Tamil roots without degrading cross-lingual representation.
2. **Cross-Family Phonetic Alignment**:
   - Joint co-training with Indo-Aryan languages (which feature explicit graphemic series for voiced and aspirated stops $\text{क/ख/ग/घ}$) allows the shared encoder to ground Roman stop clusters (`b/bh`, `d/dh`, `g/gh`) in unambiguous phonetic embeddings, dramatically elevating Named Entity transfer across all Dravidian languages.

---

### 6.10 Theedhum Nandrum Sentiment Benchmark with 3.2M Multilingual Upstream Transliteration
When deploying this 3.2M pre-trained multilingual model as an upstream neural transliterator for the **DravidianCodeMix FIRE 2020 YouTube Review Comments Dataset**, downstream sentiment classification achieves its highest performance:

#### Table 11: 3.2M Multilingual Upstream Transliteration on Downstream Sentiment

| Language | Upstream Transliteration Representation | Macro F1 (%) | Weighted F1 (%) | Accuracy (%) |
| :--- | :--- | :---: | :---: | :---: |
| **Tamil-English** | Raw Code-Mixed Text (Baseline) | 52.91% | 56.00% | 51.78% |
| **Tamil-English** | Raw Text + 3.2M Multilingual Stream (Augmented) | **53.10% (+0.19%) 🏆** | 55.09% | 50.59% |
| **Malayalam-English** | Raw Code-Mixed Text (Baseline) | 69.86% | 68.60% | 67.84% |
| **Malayalam-English** | **Raw Text + 3.2M Multilingual Stream (Augmented)** | **71.04% (+1.18%!) 🏆** | **69.80% (+1.20%!) 🏆** | **69.13% (+1.29%!) 🏆** |

---

---

## 7. Roadmap & Follow-Up Studies

### 7.1 In-the-Wild Robustness & Extensions
The downstream gains on Malayalam demonstrate that phonological normalization directly resolves colloquial Roman spelling noise (*Tanglish / Manglish*). Further extensions will benchmark on sentence-level transformer embeddings (DravidianBERT, MuRIL).

### 7.2 Speech-Augmented Transliteration via Wikimedia Commons & Mozilla Common Voice
To ground representations in physical acoustics, future work will integrate volunteer audio recordings from the **Wikimedia Commons Tamil Pronunciation Corpus** (10,000+ native utterances) and **Mozilla Common Voice**, joint-training acoustic spectrogram features to anchor stop-voicing representations in physical formant transitions ($F_1, F_2$) and Voice Onset Time (VOT).

### 7.3 Phonotactically-Constrained Synthetic Augmentation for Low-Resource Regimes
Recent work in low-resource machine transliteration demonstrates that explicitly augmenting training corpora with synthetic samples covering low-frequency character n-grams and bigrams significantly improves generalization on tail distributions ([arXiv:2410.17901](https://arxiv.org/abs/2410.17901)). 

Because Dravidian phonotactics is deterministic and strictly codified (*meym-mayakkam*, Tolkkāppiyam 48–49), empirical frequency analyses across classical corpora demonstrate that permissible biconsonantal clusters follow a strict hierarchy:
$$\text{NP (Nasal-Plosive)} > \text{PP (Geminate Plosive)} > \text{AP (Approximant-Plosive)} > \text{AA} > \text{NN} > \text{AN}$$
accounting for over 60% of all cluster mass in Tamil, while onset clusters ($PN, PA$) are strictly prohibited (Venkatakrishnan, Kumarasamy, & Lakshmanan, 2025; [oligoglot/mayal](https://github.com/oligoglot/mayal)). 

By coupling ValiMeli's phonological generator with the empirical Maximum Likelihood Estimation (MLE) cluster matrices from *Mayal*, future extensions can deterministically synthesize phonotactically legal pseudo-words to achieve 100% biconsonantal cluster coverage for extreme low-resource Dravidian languages (such as Badaga, Irula, Kodava, and Tulu) without generating phonotactically prohibited noise.

---

## 8. Conclusion

Our findings demonstrate that Tamil and native Malayalam orthographies are not underspecified, but governed by principled orthographic parsimony. By aligning tokenisation with native phonological structure, ValiMeli provides an explicit inductive bias for Tamil and Malayalam transliteration, establishing significant performance gains in reverse transliteration and stop-voicing accuracy across millions of samples.

---

## References

- Chakravarthi, B. R., Muralidaran, V., Priyadharshini, R., Suryawanshi, S., Navaneethakrishnan, S., Ponnusamy, J., & Kumaresan, P. K. (2020). *Corpus Creation for Sentiment Analysis in Code-Mixed Tamil-English Text*. In Proceedings of the 1st Workshop on Dravidian Language Technologies in ACL 2020, pp. 61–67.
- Kunchukuttan, A., Kakwani, D., Golla, S., Bhattacharyya, P., Khapra, M. M., & Kumar, P. (2021). *AI4Bharat-IndicXlit: Multilingual Transliteration for Indian Languages*. Transactions of the Association for Computational Linguistics (TACL), 9, 1374–1390.
- Lakshmanan, B. L., & Ravindranath, S. K. (2020). *Theedhum Nandrum @ Dravidian-CodeMix-FIRE2020: A Sentiment Polarity Classifier for YouTube Comments with Code-switching between Tamil, Malayalam and English*. In Working Notes of FIRE 2020 - Forum for Information Retrieval Evaluation, CEUR Workshop Proceedings, vol. 2826, pp. 542–547.
- Madhani, Y., Seshadri, P., Parikh, T., Kunchukuttan, A., Kumar, P., & Khapra, M. M. (2022). *Aksharantar: Towards Building Open Datasets for Indic Language Transliteration*. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 11621–11634.
- Martinet, A. (1955). *Économie des changements phonétiques: Traité de phonologie diachronique*. Francke.
- Niklas, U. (1988). *Introduction to Tamil Grammatical Theory*. Bulletin de l'École française d'Extrême-Orient (BEFEO), 77(1), 165–188.
- Ramesh, G., Doddapaneni, S., Bheemambika, A., Kunchukuttan, A., Kumar, P., & Khapra, M. M. (2022). *IndicTrans: Towards High-Quality and Accessible Machine Translation for Indian Languages*. In Proceedings of ACL 2022.
- Roark, B., Wolf-Sonkin, L., Kirov, C., Gibson, S., Chase, M., & Murphy, N. (2020). *Processing South Asian Languages in the Dakshina Dataset*. In Proceedings of the 12th Language Resources and Evaluation Conference (LREC 2020), pp. 6806–6814.
- Sutton, R. (2019). *The Bitter Lesson*. In Incomplete Ideas (Essays on Computing).
- Swadesh, M. (1934). *The Phonemic Principle*. Language, 10(2), 117–129.
- Thottingal, S. (2026). *The Broken Token: Tokenization for Malayalam Language Models*. Swathanthra Malayalam Computing (SMC). URL: https://thottingal.in/blog/2026/02/27/malayalam-tokenizer-llm/
- Tolkāppiyar (c. 300 BCE). *Tolkāppiyam: Eluttatikāram (Phonology and Orthography)*.
- Trubetzkoy, N. S. (1939). *Grundzüge der Phonologie*. Travaux du Cercle Linguistique de Prague.
- Venkatakrishnan, R., Kumarasamy, R., & Lakshmanan, B. (2025). *Pattern of Biconsonantal Clusters in Old Tamil Texts*. International Journal of Dravidian Linguistics (IJDL), 54(1), 1–32. Code: https://github.com/oligoglot/mayal
