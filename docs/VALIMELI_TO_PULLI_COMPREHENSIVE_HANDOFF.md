# Project ValiMeli to Pulli: Comprehensive Engineering & Scientific Handoff

**Author / Maintainer**: BalaSundaraRaman Lakshmanan  
**Repository**: `oligoglot/valimeli`  
**Active Branch**: `feat/multilingual-pretraining-26m`  
**Target Projects**: ValiMeli Paper Camera-Ready / Pulli IME & Normalisation Engine  
**Date**: September 2026 (Updated with Revisions 3 & 4 Audit Findings)  

---

## 0. Foundational Directive: Research as a Truth-Seeking Exercise

> [!CAUTION]
> **"Research is essentially a truth-seeking exercise. Fabricating anything is completely unacceptable."**  
> This axiom is permanently codified in [`AGENTS.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/AGENTS.md) and [`.agents/rules/research_integrity.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/.agents/rules/research_integrity.md). Every empirical number, standard deviation, percentage, sample count, acoustic metric, and citation must be programmatically derived from real data and verified against primary sources. If anything cannot be found or verified, **STOP AND ASK THE USER IMMEDIATELY**.

---

## 1. Executive Summary & Paper State

The research manuscript ***"A Parsimonious Code: Allophonic Voicing and the Limits of Phonological Supervision in Romanised Tamil and Malayalam Transliteration"*** is complete, verified, and frozen for camera-ready submission in [`docs/revised2/`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2) and [`docs/revised2.zip`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2.zip).

### Key Manuscript & Artifact Locations
* **Main LaTeX Manuscript**: [`docs/revised2/main.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/main.tex) (384 lines, Paper-Centric Active Voice, Single-Author)
* **LaTeX Table Suite**:
  * [`docs/revised2/table_entropy.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_entropy.tex) (Table 1: Voicing Entropy across Text vs. Spoken Speech Acoustics)
  * [`docs/revised2/table_context.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_context.tex) (Table 2: Phonotactic Context Voicing Probabilities & Detector Validation)
  * [`docs/revised2/table_scale.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_scale.tex) (Table 3: Multi-Seed & Scaling Transliteration Benchmarks: 25k to 1.0M pairs)
  * [`docs/revised2/table_disagree.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_disagree.tex) (Table 4: Annotator Duality in Dakshina vs. In-the-Wild YouTube Tanglish)
  * [`docs/revised2/table_downstream.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_downstream.tex) (Table 5: Downstream Sentiment Transfer on Theedhum Nandrum)
  * [`docs/revised2/table_acoustic_alignment.tex`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/table_acoustic_alignment.tex) (Table A / Appendix: Empirical Phonetic & Acoustic Alignment)
* **Verified Bibliography**: [`docs/revised2/references.bib`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/references.bib) (14 verified entries; excised all unverified references; verified DOIs for Keane 2004 `10.1017/S0025100304001549`, Lisker 1958 *Indian Linguistics*, Madhani et al. EMNLP 2023, Ramesh et al. TACL 2022, Niklas 1988, Swadesh 1934, Vemula et al. 2025, Lakshmanan et al. 2020, 2025).
* **Figures**:
  * [`docs/revised2/fig_effects_corrected.png`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/fig_effects_corrected.png) (Figure 1: Forest plot of 10 matched scaling comparisons with multi-seed error bars)
  * [`docs/revised2/acoustic_voicing_spectrograms.png`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/docs/revised2/acoustic_voicing_spectrograms.png) (Figure 2: STFT spectrograms rendered with Unicode typography)

---

## 2. Core Scientific & Empirical Findings (Crucial for Pulli)

### A. Proposition 1: Argmax Invariance under Skewed Allophony
1. **The Myth**: Prior computational literature attributed lower transliteration accuracy in Tamil and Malayalam to "orthographic underspecification" (a single graphemic stop series spanning voiced and voiceless allophones).
2. **The Formal Proof**: Formally proven in Section 3 of `main.tex`:
   $$\exists v^* \in \mathcal{V} \text{ s.t. } \forall c \in \mathcal{C}, \; v^* \in \arg\max_{v \in \mathcal{V}} P(V=v \mid C=c) \implies v^* \in \arg\max_{v \in \mathcal{V}} P(V=v) \text{ and } R(\hat{v}_C) = R(\hat{v}_\emptyset)$$
   Because the voiceless Latin spelling remains the plurality outcome across **all** phonotactic environments in crowdsourced benchmark data ($P(\text{voiced} \mid C) < 0.50$ everywhere; word-initial $11.45\%$, post-nasal $47.16\%$), conditioning on context alters the greedy/beam argmax decision boundary by exactly **0.00%** (an exact mathematical identity rather than rounding), yielding zero Bayes risk reduction under 0-1 loss. This holds irrespective of mutual information, including when $I(V; C) > 0$ ($0.1361$ bits in Tamil, $0.1587$ in Malayalam). The formulation strictly handles ties via set membership ($v^* \in \arg\max$) and proves unconditional plurality from conditional plurality.
3. **Scope Boundary (Slot-Level Risk vs. Sequence-Level Exact Match)**:
   While Proposition 1 establishes the mathematical bound on per-slot classification risk under 0-1 loss, full sequence-to-sequence models optimize sequence likelihood and are evaluated on exact-match string accuracy. The proposition proves why auxiliary phonological loss heads have no local decision-boundary advantage; the empirical scaling experiments (\S5, Table 3) establish that this lack of utility carries over to full sequence generation across model scales.
4. **Takeaway for Pulli**: Auxiliary source-side loss heads cannot overcome target-side reference variance. Pulli must implement **input-side multi-modal candidate generation, style codes, and acoustic/citation typing toggles**.

### B. The Spoken Speech Contrast (Full-Corpus Population Audit)
Unlike crowdsourced text benchmarks where argmax decisions never shift ($\Delta R = 0.00\%$), physical speech acoustics exhibits an active decision shift:
* **Corpus Scale**: $5{,}325$ native spoken Tamil audio recordings ($12{,}435$ evaluated plosive slots across $11{,}590$ core syllable positions and $845$ coda slots; [`artifacts/tamil_speech_full_corpus_manifest.json`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/artifacts/tamil_speech_full_corpus_manifest.json) and [`artifacts/acoustic_voicing_entropy_results.json`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/artifacts/acoustic_voicing_entropy_results.json)).
* **Phonotactic Environment Breakdown**:
  * **Word-Initial (`#_`)**: $2{,}131$ slots, **$99.72\%$ voiceless** ($H = 0.0279$ bits, mean closure $44.0\text{ ms}$).
  * **Post-Nasal (`N_`)**: $1{,}507$ slots, **$85.14\%$ voiced** ($78.8\%$ voice-bar ratio, $H = 0.6064$ bits, mean closure $4.7\text{ ms}$). Flips the argmax decisively ($12.1\%$ of tokens).
  * **Intervocalic (`V_V`)**: $5{,}566$ slots, **$56.77\%$ lenis voiced** ($H = 0.9867$ bits, mean closure $18.8\text{ ms}$). Flips the argmax marginally ($44.8\%$ of tokens).
  * **Geminate (`C_C`)**: $2{,}386$ slots, silent occlusion gap averaging **$50.5\text{ ms}$** (roughly $11\times$ / $10.7\times$ longer than post-nasal). While automated frame-level STFT energy registers a $30.22\%$ low-$F_0$ ratio due to vowel formant transition bleed into early closure, the extended occlusion confirms physical fortis voicelessness.
  * **Post-consonantal / Coda**: $845$ slots, $40.00\%$ voiced ($H = 0.9707$ bits).
* **Spoken Bayes Rule Accuracy**: **$70.29\%$** rule accuracy, representing a **$+32.92\%$ relative error reduction** over the unconditional majority baseline (from $44.29\%$ error down to $29.71\%$).
* **Contributor Disclosure**: 7 contributors, with **96.0% (5,113 recordings)** contributed by primary native speaker `User:Sriveenkat` reading citation lemma forms via Lingua Libre.

### C. Automated Voicing Detection Error Bounding & Limitations
Voicing labels in \textsc{ValiMeli-Speech} are assigned by an automated detector over spectral energy ratios (voice-bar ratio $\ge 0.50$ or low-$F_0$ energy $\ge 0.25$, with silent closure $< 35\text{ ms}$), not by human annotation:
* **Geminate Error Bounding**: In an environment that Tamil phonology holds to be strictly voiceless, the detector reports $30.22\%$ voiced, consistent with vowel formant energy bleeding into early closure ($15$--$20\text{ ms}$).
* **Robustness of Error Reduction**: Because post-vocalic bleed applies to intervocalic and post-nasal slots as well, sensitivity analysis demonstrates that the relative error reduction reported in Table 1 is robust: it does not fall below $+24\%$ for any uniform correction of up to $30$ points applied to both post-sonorant contexts.
* **Provisional Intervocalic Status**: While the post-nasal flip ($85.14\%$ voiced) survives any plausible bleed correction, the intervocalic argmax sits only $6.77$ points above the $50\%$ decision boundary ($56.77\%$), so the marginal intervocalic flip should be treated as provisional pending future human-annotated validation.

### D. Multi-Seed Benchmark Results (25k Scale)
Evaluated across Seeds 42, 43, 44 on canonical holdout test sets ($N = 11{,}499$ Tamil, $N = 12{,}451$ Malayalam):
* **Tamil (25k)**: Baseline ($A_0$) $25.21 \pm 1.41\%$ vs. Multi-Task ($A_1\text{-MT}$) $25.07 \pm 1.33\%$ ($\Delta = -0.14\%$, Paired McNemar $\chi^2 = 0.298$, $p = 0.585$).
* **Malayalam (25k)**: Baseline ($A_0$) $18.86 \pm 1.17\%$ vs. Multi-Task ($A_1\text{-MT}$) $19.00 \pm 1.44\%$ ($\Delta = +0.15\%$, Paired McNemar $\chi^2 = 0.015$, $p = 0.901$).
* Over $1{,}620$ discordant prediction pairs per language confirm that low-resource shifts are random parameter churn rather than systematic phonological learning.

### E. Large-Scale Scaling Dynamics & Multiple Testing
* **250k BiGRU (IndicXlit Architecture Replications)**: Tamil $58.19\% \to 59.72\%$ ($+1.53\%$, nominal $p = 0.018$), Malayalam $51.08\% \to 52.16\%$ ($+1.08\%$, $p = 0.089$).
* **250k Transformer**: Tamil $59.18\% \to 59.94\%$ ($+0.77\%$, $p = 0.237$), Malayalam $52.84\% \to 52.73\%$ ($-0.10\%$, $p = 0.869$).
* **500k Transformer**: Tamil $60.71\% \to 60.49\%$ ($-0.22\%$, $p = 0.736$), Malayalam $55.92\% \to 55.39\%$ ($-0.54\%$, $p = 0.393$).
* **1.0M Bilingual Transformer**: Tamil $62.42\% \to 60.09\%$ ($\mathbf{-2.33\%}$, $p = 0.0003$), Malayalam $55.28\% \to 53.91\%$ ($\mathbf{-1.37\%}$, $p = 0.0295$).
* **Multiple Testing Correction**: Under Bonferroni correction for $m=10$ matched comparisons ($\alpha_{\text{Bonf}} = 0.005$), the single nominal gain at 250k ($p=0.018 > 0.005$) **does not survive**, confirming the global null hypothesis, whereas degradation at 1.0M ($p = 0.0003 < 0.005$) is highly robust.
* **Morphology vs. Phonology**: Phonology-aware string tags strictly outperform morphology-aware tokenisation across all four partitions ($+0.35$ to $+2.48$ EM points), while morphology-aware tokenisation drops below standard character baselines in 3 of 4 partitions ($-1.36$ in Malayalam $\text{indic}\to\text{en}$, $p = 0.022$).

### F. Target-Side Disagreement & In-The-Wild Voicing
* **Dakshina Multi-Reference Duality**: Independent human annotators disagree on **$69.26\%$** of Tamil post-nasal plosive slots on identical words ($40.83\%$ in Malayalam; $44.50\%$ pair-level disagreement).
* **In-The-Wild YouTube Tanglish (Theedhum Nandrum)**: Across $13{,}512$ aligned plosives from YouTube comments, post-nasal stops flip to **$64.00\%$ voiced** ($24.67\%$ overall wild voicing rate).
* **Insight**: Formal benchmark annotators follow a schoolroom citation prior (typing by letter name: *t/ta* $\to$ `th/tha`), while naturalistic conversational typists type by phonetic realisation (*t/ta* $\to$ `d/dh/da/dha`).

### G. Malayalam Alignment Coverage Audit
* **Aksharantar Audit (150,000 pairs; `artifacts/malayalam_alignment_audit.json`)**:
  * Aligned: $65.31\%$ ($97{,}965$ pairs); Unaligned: $34.69\%$ ($52{,}035$ pairs).
  * Unaligned pairs are **$7.30\times$ more likely** to contain Sanskrit/Grantha loan consonants ($84.49\%$ vs. $42.73\%$, $\chi^2 = 24{,}203.8$, $p < 10^{-300}$).
  * Unaligned pairs actually contain slightly *fewer* complex conjunct ligatures ($46.17\%$ vs. $51.01\%$), refuting the hypothesis that Dravidian consonant conjuncts cause alignment failure.

---

## 3. Linguistic & Orthographic Nuances for Pulli

### A. The Unicode/ISCII Storage Artifact vs. Native Phonology
* **Trap**: In the Brahmic ISCII/Unicode digital model, vowel-bearing syllables (*uyirmey*, e.g., `ம` `U+0BAE`) are base codepoints, requiring a combining virāma/puḷḷi (`்` `U+0BCD`) to strip the vowel to produce pure consonants (`ம்`). Naive segmenters incorrectly tokenize `படம்` into `[pa][ṭa][ma][.]` or `pat.ama`.
* **Linguistic Truth**: In *Tolkāppiyam*, pure consonants (*meyyeḻuttu*) are the primary phonemic units ($/m/$). The word **படம்** (*paṭam*) consists of exactly 3 *eḻuttukkaḷ*:
  $$\text{ப} \ ([pa]) \quad + \quad \text{ட} \ ([\text{ɖ}a]) \quad + \quad \text{ம்} \ ([m]) \implies \mathbf{pa\text{-}\d{t}a\text{-}m \ (pa\d{t}am)}$$
* **Rule for Pulli**: Never decompose coda consonants into sub-syllabic base + virāma strings. Use atomic *eḻuttu* segmentation.

### B. Palatal Sibilant Lenition vs. Plosive Voicing
* In Modern Spoken Tamil, word-initial **ச** undergoes variable spirantization/lenition to a sibilant ($[s]$ or $[ɕ]$ or $[h]$), causing divergent spellings (*sangu* vs. *chanku* vs. *cangu*).
* For pure plosive voicing demonstrations, use unspirantized velar/bilabial stops like **குரங்கு** (*kuraṅku* / *kurangu* $[kʊɾɐŋɡɯ]$) or **படம்** (*paṭam* / *padam*).
* **Rule for Pulli**: Implement distinct IME phonetic mapping rules for palatal affricates (`c`/`s`/`ch`) vs. standard plosive voicing (`k`/`g`, `t`/`d`, `p`/`b`).

### C. Standard Quadruple Representation
All linguistic examples in documentation and test suites must follow:
1. Native Script: **படம்**, **பக்கம்**, **தம்பி**, **குரங்கு**
2. Standard ISO 15919: **paṭam**, **pakkam**, **tampi**, **kuraṅku** (`\textit{pa\d{t}am}`, `\textit{kura\.{n}ku}`)
3. Colloquial / Benchmark Romanisation: (*padam*), (*pakkam*), (*thambi*), (*kurangu*)
4. Narrow Phonetic IPA: $[pɐɖɐm]$, $[pɐkːɐm]$, $[t̪ɐmbi]$, $[kʊɾɐŋɡɯ]$

---

## 4. Architectural Directives for Pulli & Dravidian ASR

The findings from Section 7 of `main.tex` dictate four direct engineering blueprints for Pulli and related speech tools:

1. **Direct Phonemic Targets**:
   * End-to-end typing / ASR models must decode directly into native syllabic graphemes (*eḻuttu* / aksharas) rather than intermediate Latin/IPA tokens. Because native orthography collapses allophonic variants ($[k, g, \gamma] \to \text{\textit{ka}}$), direct graphemic decoding eliminates the representational burden of predicting surface allophones.
2. **Word-Initial Voicing as Loanword Prior**:
   * Because native vocabulary exhibits $99.72\%$ voicelessness initially ($H = 0.0279$ bits), the presence of an initial low-$F_0$ voicing bar ($<300\text{ Hz}$) or initial voiced letter (`g`, `d`, `b`) provides a decisive prior to dynamically bias language models toward English/Sanskrit loanword lexicons.
3. **Atomic Nasal-Plosive Acoustic Units**:
   * Given that post-nasal stops exhibit $85.14\%$ continuous glottal voicing with negligible closure silence ($4.7\text{ ms}$), defining nasal+plosive conjuncts (\textit{\.{n}ka}, \textit{\~{n}ca}, \textit{\d{n}\d{t}a}, \textit{nta}, \textit{mpa}, \textit{\b{n}\b{r}a}) as atomic modelling units prevents boundary insertion errors.
4. **Dual-Mode Typing Engine (IME)**:
   * **Citation Mode**: Maps letter-name spellings (*thambi* $\to$ தம்பி, *kondaadi* $\to$ கொண்டாடி).
   * **Phonetic / Naturalistic Mode**: Maps acoustic conversational variants (*thambi* $\to$ தம்பி, *dambi* $\to$ தம்பி, *padam* $\to$ படம், *vandu* $\to$ வந்து).
   * **Input-Side Augmentation**: Permute Latin plosive clusters ($nth \leftrightarrow ndh$, $mp \leftrightarrow mb$) during IME model training to enforce input invariance on rare roots.

---

## 5. Skills & Governance Framework

Four skills and two workspace-level governance rules are installed and active:

1. **`empirical-nlp-research-methodology`** ([`.agents/skills/empirical-nlp-research-methodology/SKILL.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/.agents/skills/empirical-nlp-research-methodology/SKILL.md)):
   - Section 0: Foundational Axiom: Research as a Truth-Seeking Exercise.
   - Theoretical bounding ($H(Y), H(Y\mid X), I(X;Y)$, Argmax Invariance).
   - Multi-seed low-resource protocols ($\mu \pm \sigma$, Seeds 42, 43, 44).
   - Paired McNemar significance testing on instance prediction logs.
   - Dual-domain (benchmark vs. in-the-wild) distribution auditing.
2. **`rigorous-nlp-paper-authoring`** ([`.agents/skills/rigorous-nlp-paper-authoring/SKILL.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/.agents/skills/rigorous-nlp-paper-authoring/SKILL.md)):
   - Section 6: Research Integrity: Research as a Truth-Seeking Exercise.
   - Commonwealth English spelling standard.
   - Single-author voice alternatives (authorial "we" vs. paper-centric active voice).
   - 0-error Overleaf LaTeX architecture and font fallbacks.
   - ISO 15919 quadruple transliteration standards.
3. **`linguistics-research-methodology`** ([`.agents/skills/linguistics-research-methodology/SKILL.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/.agents/skills/linguistics-research-methodology/SKILL.md)):
   - Cross-subfield linguistic foundations (syntax, semantics, phonology, psycholinguistics).
4. **`linguistics-paper-authoring`** ([`.agents/skills/linguistics-paper-authoring/SKILL.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/.agents/skills/linguistics-paper-authoring/SKILL.md)):
   - Interlinear glossing (`gb4e`), OT tableaux, IPA phonetics, and Unified Style Sheet.
5. **Project Directives & Rules**:
   - [`AGENTS.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/AGENTS.md): Root directives mandating zero-fabrication and immediate user escalation.
   - [`.agents/rules/research_integrity.md`](file:///Users/slakshmanan/Playspace/Gemini/valimeli/.agents/rules/research_integrity.md): Customization rule enforcing truth-seeking across all project tasks.
