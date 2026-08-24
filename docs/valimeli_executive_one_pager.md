# Project ValiMeli (வலி–மெலி)
### *Phonology-Aware Inductive Bias for Tamil and Malayalam in Multilingual Sequence-to-Sequence Models*

**Author**: BalaSundaraRaman (Sundar) Lakshmanan ([`oligoglot/valimeli`](https://github.com/oligoglot/valimeli))  
**Target Context**: Architectural Integration for IndicXlit / IndicTrans / Foundation Tokenisers  
**Language Style**: Commonwealth English  

---

## 1. Executive Summary & The Problem Statement

In South Asian multilingual sequence-to-sequence benchmarks—including **Aksharantar** (*EMNLP 2022*) and **IndicXlit** (*TACL 2021*)—Tamil (`ta`) and Malayalam (`ml`) consistently exhibit a pronounced performance deficit compared to Indo-Aryan languages (Hindi, Gujarati, Marathi: 74%–78% Exact Match) and fellow Dravidian languages (Telugu, Kannada: 67%–73% Exact Match).

A prevalent misconception attributes this gap to *"orthographic underspecification"*, alleging that Tamil and Malayalam scripts fail to mark stop voicing. We refute this deficit hypothesis:
- **Orthographic Parsimony (*Martinet’s Principle of Economy*)**: Tamil and native Malayalam orthographies are maximally specified for their native phonology. Positional stop voicing (*Vallinam* $\{k, c, \tau, t, p, \underline{r}\}$ realising as voiced $[g, j, d, b]$ post-nasally and intervocalically) is deterministic in native roots.
- **The Information Bottleneck**: When standard character-level Seq2Seq models or Devanagari intermediate pivots process single-stop scripts, they discard deterministic phonotactic structure, forcing the encoder to redundantly infer acoustic voicing distributions through high-entropy statistical fitting.

---

## 2. The ValiMeli Framework & Multi-Task Objective (`A1-MT`)

We introduce **ValiMeli** (Hard *Vallinam* + Soft *Mellinam*), a zero-overhead phonological framework:
1. **Context-Aware Tagging (`A1`)**: Deterministically tags plosives by phonotactic environment (`[INIT]`, `[INTER]`, `[POST_NASAL]`, `[GEMINATE]`), reducing conditional voicing entropy $H(\text{Voicing} \mid \text{Context})$ from **0.982 bits to <0.089 bits (>92% reduction)**.
2. **Multi-Task Auxiliary Supervision (`A1-MT`)**: Employs an auxiliary classification loss over encoder representations without modifying inference-time string lengths:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seq2seq}}(\theta) + \lambda \cdot \mathcal{L}_{\text{phono}}(\theta_{\text{enc}}, \phi)$$

---

## 3. Empirical Results on the Official Aksharantar Benchmark (100,135 Holdout Pairs)

| Pre-training Regime & Scale | Tamil Native EM (%) | Tamil Named Entities EM (%) | Tamil Combined Test EM (%) |
| :--- | :---: | :---: | :---: |
| **Monolingual Baseline (500k)** | 66.91% | 32.03% | 60.71% (CER 8.95%) |
| **Joint Bilingual (Tamil-Mal, 1.0M)** | 68.28% | 35.35% (+3.32% boost!) | 62.42% (CER 8.42%) |
| **3.2M Multilingual Standard (A0)** | 68.42% | 38.63% (+6.60% boost!) | 63.12% (CER 8.06%) |
| **3.2M Multilingual Multi-Task (A1-MT)** | **68.55% 🏆** | **39.66% (+7.63% over Base!) 🏆** | **63.41% (Highest Overall) 🏆** |

### Key Findings Across Regimes:
1. **Low-Resource Regime (25k Samples)**: Under severe data sparsity, `A1-MT` delivers an immediate **+1.26% Top-1 EM boost** and **-1.80% CER reduction** over standard baselines.
2. **Massive Multilingual Scaling (3.2M Pairs Across 8 Languages)**: `ValiMeli-A1-MT` achieves an all-time peak of **39.66% on Tamil Named Entities**, outperforming standard multilingual co-training by **+1.03%** and the monolingual baseline by **+7.63% absolute**.
3. **Downstream NLP Generalisation (*Theedhum Nandrum* on DravidianCodeMix FIRE 2020)**: Deploying our 3.2M models as upstream neural normalisers achieves state-of-the-art sentiment classification:
   - **Tamil**: Macro F1 surges to **53.89% (+0.98% over raw text)**.
   - **Malayalam**: Macro F1 reaches **70.80%–71.17% (+1.31% over raw text)**.

---

## 4. Why Scale Alone Does Not Solve Single-Stop Orthographies

In response to the hypothesis that *"an unconstrained neural model will autonomously learn stop voicing if scaled with enough data"*:
- As training data scales from 500k to 3.2M pairs, models encounter thousands of English loanwords and North Indian names with conflicting Roman spellings (`b`, `p`, `d`, `t`, `g`, `k` mapping to the same Tamil letters).
- Without inductive bias, this causes **gradient interference** across stop tokens.
- **ValiMeli (`A1-MT`) acts as an invariant regulariser**: It forces the encoder to ground stop tokens in their structural phonological context, allowing native roots and foreign loanwords to coexist without mutual degradation.

---

## 5. Proposed Points for Collaboration & Integration

1. **Integration into IndicXlit / IndicTrans v3**: Incorporate the lightweight auxiliary multi-task stop-voicing head ($\lambda=0.08$) into the shared multilingual encoder for South Dravidian scripts.
2. **Subword Tokeniser Alignment**: Standardise syllable and *uyirmey* cluster boundary constraints in Indic BPE/Unigram tokenisers to eliminate cross-script glyph leakage and dangling viramas.
3. **Open Access**: Complete PyTorch source code, training logs, checkpoints, and evaluation harnesses are openly accessible in the repository: [`https://github.com/oligoglot/valimeli`](https://github.com/oligoglot/valimeli).
