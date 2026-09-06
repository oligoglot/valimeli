---
name: linguistics-research-methodology
description: Universal methodology guide for research across all subfields of linguistics—including theoretical/formal linguistics (syntax, semantics, phonology, morphology), experimental psycholinguistics (Latin squares, eye-tracking, ERPs, mixed-effects models), field documentation and language description (CARE/FAIR ethics, ELAN/FLEx), corpus and sociolinguistics (collocations, variable rules), historical-comparative linguistics and typology, and computational linguistics / NLP (information-theoretic bounding, multi-seed statistical hygiene, speech signal grounding).
---

# Universal Linguistics & Computational Linguistics Research Methodology

This skill provides a comprehensive, scientifically rigorous framework for conducting research across all major paradigms in linguistics: theoretical, descriptive/field, experimental/psycholinguistic, corpus/sociolinguistic, historical/typological, and computational/NLP.

---

## 0. Foundational Axiom: Research as a Truth-Seeking Exercise

> [!CAUTION]
> **RESEARCH IS ESSENTIALLY A TRUTH-SEEKING EXERCISE. FABRICATING ANYTHING IS COMPLETELY UNACCEPTABLE.**
> Scientific research in linguistics and cognitive science exists solely to seek and document truth. Fabricating, inventing, or synthesizing ANY empirical data, consultant attribution, audio metric, phonetic value, statistical test, theoretical derivation, or scholarly citation is an intolerable violation of research integrity.

1. **Zero-Fabrication Across All Linguistic Domains**:
   - **Primary Data & Consultant Attribution**: NEVER invent language consultant names, native speaker judgments, field recordings, or sociolinguistic demographic data. Adhere strictly to verified field elicitation and CARE/FAIR ethics.
   - **Empirical Statistics & Acoustics**: NEVER invent, approximate, or fabricate formant values ($F_1, F_2$), VOT, closure durations, pitch tracks, counts, or $p$-values. Every figure must derive programmatically from audited, reproducible code and frozen artifacts.
   - **Citations & Treatises**: NEVER fabricate historical grammarian rules, book titles, author lists, venues, DOIs, or volume/page numbers. Every reference must be verified against primary registries (LSA, Crossref, ACL, library catalogs).
   - **Formal Proofs & Grammatical Diagnostics**: Every structural diagnostic (clefting, binding, constituency) and semantic/phonological derivation must be logically and mathematically sound.
2. **Mandatory User Escalation Protocol ("Ask Me If You Can't Get Something")**:
   - If any citation, DOI, historical source, baseline number, dataset, or empirical metric cannot be found, accessed, or verified: **STOP AND ASK THE USER IMMEDIATELY**.
   - Transparently state what is missing and request primary material or guidance. Never generate a speculative placeholder.
3. **Strict Attribution & Provenance Separation**:
   - Never attribute figures from internal experiments or replications to an external publication.
   - Internal baselines must be explicitly presented as *this work's replication baseline* (referencing internal artifacts).

---

## 1. Scientific Foundations & Epistemology in Linguistics

1. **Levels of Adequacy (Chomsky, 1965)**:
   - **Observational Adequacy**: Accurately presenting primary linguistic data (corpora, field recordings, judgment distributions).
   - **Descriptive Adequacy**: Formulating structural rules or model representations that account for the native speaker's internalized grammatical competence.
   - **Explanatory Adequacy**: Grounding descriptive generalizations in universal cognitive, evolutionary, or mathematical principles (Universal Grammar, information theory, processing constraints, communicative efficiency).
2. **Ecological Validity vs. Experimental Control**:
   - Balance naturalistic corpus/discourse observation with tightly controlled experimental or formal minimal-pair paradigms.
   - Every empirical study must declare its methodological stance: *competence vs. performance*, *production vs. comprehension*, *synchronic vs. diachronic*.

---

## 2. Theoretical & Formal Linguistics Methodology

### A. Phonology & Phonetics-Phonology Interface
* **Feature Representation**: Ground all segmental analyses in established distinctive feature theories (Chomsky & Halle SPE, Clements & Hume Feature Geometry, or Element Theory).
* **Rule-Based Derivations**: Formulate standard phonological rewrite rules:
  $$A \longrightarrow B \;/\; C \;\underline{\quad}\; D$$
  Specify rule ordering (feeding, bleeding, counterfeeding, counterbleeding) explicitly when evaluating derivational interactions.
* **Optimality Theory (OT) & Harmonic Grammar**:
  - Define Candidate Generation ($\text{Gen}$), Universal Constraints ($\text{Con}$), and Evaluation ($\text{Eval}$).
  - Establish strict constraint rankings through explicit ranking arguments: show that Candidate $A$ defeats Candidate $B$ only if Constraint $C_1 \gg C_2$.
  - Clearly identify harmonic bounding (candidates that can never win under any ranking).

### B. Morphology & Morphosyntax
* **Architectural Typology**: Distinguish concatenative (affixation, compounding) from non-concatenative morphology (Semitic root-and-pattern, ablaut, reduplication, tone inflections).
* **Framework Selection**: Explicitly state the analytical paradigm:
  - *Item-and-Arrangement (IA)*: Morpheme-based concourse.
  - *Item-and-Process (IP)*: Rule-based item mutation.
  - *Word-and-Paradigm (W&P)*: Realizational, paradigm-cell-based inflectional systems.
* **Morphome Analysis**: Identify purely morphological entities (Aronoff's morphomes) that lack systematic phonological or semantic conditioning.

### C. Syntax & Structural Diagnostics
* **Constituency Diagnostics**: Never assert phrase boundaries without at least two independent structural diagnostic tests:
  1. *Clefting & Pseudoclefting*: *It was [XP] that...*
  2. *Topicalization & Fronting*: *[XP], they had never seen.*
  3. *Pro-form Substitution*: Pronouns, *do so*, *there*, *then*.
  4. *Coordination*: Coordinating only like categories ($XP \text{ and } XP$).
  5. *Fragment Answers*: Standalone responses to wh-questions.
* **Binding & Locality Diagnostics**:
  - Test Principle A (anaphors bound in local domain), Principle B (pronominals free in local domain), Principle C (R-expressions free everywhere).
  - Test Island Constraints: Wh-islands, Complex NP islands, Adjunct islands, Coordinate Structure Constraint, Left Branch Condition.
* **Minimal Pairs**: Maintain strictly controlled minimal pairs differing by exactly one syntactic/lexical feature to eliminate semantic and length confounds.

### D. Formal Semantics & Pragmatics
* **Type Theory & Functional Application**: Ground compositional semantic derivations in standard typed $\lambda$-calculus ($e, t, \langle e, t \rangle, \langle \langle e, t \rangle, t \rangle$).
* **Diagnostic Tests for Meaning Components**:
  - *Entailment*: If $P \models Q$, then $P \land \neg Q$ is a logical contradiction.
  - *Presupposition*: Invariant under the **Family-of-Sentences** tests (negation, interrogation, conditional antecedents, epistemic modals: *It is possible that P*).
  - *Conversational Implicature (Gricean)*: Must pass **Cancellability** (*P, but not Q*) and **Reinforceability** (*P, and in fact Q* without redundancy).

---

## 3. Experimental Linguistics & Psycholinguistics

### A. Stimulus Design & Psycholinguistic Controls
* **Factorial Designs & Latin Square Counterbalancing**:
  - Group items into experimental conditions using a balanced Latin Square design such that each participant sees each item in only one condition, with equal exposure across conditions.
  - Maintain a minimum 1:1 (ideally 1:2 or 1:3) **filler-to-target ratio** to obscure experimental manipulation and prevent participants from developing task-specific response strategies.
* **Matching Confounding Lexical Variables**:
  - Control or match target stimuli across conditions using standardized psycholinguistic databases (e.g., SUBTLEX, CELEX, WordNet):
    - *Word/Lemma Frequency* (log-transformed Zipf scale)
    - *Length* (character count, syllable count, morpheme count)
    - *Orthographic & Phonological Neighborhood Density* (OLD20, PLD20)
    - *Bigram / Trigram Transitional Probabilities*
    - *Semantic Concreteness & Imageability*

### B. Experimental Paradigms
* **Acceptability & Grammaticality Judgments**:
  - *Standardized Likert Scales* (5-point, 7-point), *Magnitude Estimation* (unbounded ratio scaling), or *Forced-Choice Binary Selection*.
  - Standardize scores ($z$-score transformation per participant) to eliminate individual scale-use biases.
* **Online Processing (Reaction Time & Eye-Tracking)**:
  - *Self-Paced Reading (Moving Window)*: Measure reading times at the critical target region and subsequent spillover regions ($N+1, N+2$).
  - *Eye-Tracking During Reading*: Distinguish early vs. late measures:
    - *First Fixation Duration (FFD)*: Initial lexical access.
    - *First Pass / Gaze Duration (GD)*: Early syntactic & morphological processing.
    - *Regression Path / Go-Past Duration*: Syntactic reanalysis and repair.
    - *Total Reading Time (TRT)*: Overall integration effort.
* **Electrophysiology & Neuroimaging (EEG/ERP)**:
  - *N400*: Negative deflection at $\sim 400\text{ms}$ (centro-parietal) indexical of semantic retrieval effort and lexical predictability (cloze probability).
  - *P600*: Positive deflection at $\sim 600\text{ms}$ (posterior) indexical of syntactic reanalysis, structural violations, and garden-path recovery.
  - *LAN / ELAN*: (Early) Left Anterior Negativity reflecting early phrase-structure or morphosyntactic agreement violations.
  - *Mismatch Negativity (MMN)*: Frontal negative deflection reflecting pre-attentive auditory/phonetic discrimination.

### C. Statistical Analysis: Mixed-Effects Models
* **Linear Mixed-Effects Models (`lme4::lmer` in R, `brms` for Bayesian, `statsmodels` in Python)**:
  - Analyze continuous dependent variables (reading times, reaction times, acoustic formants) with subject and item random intercepts and slopes.
  - **Maximal Random-Effects Structure** (Barr et al., 2013):
    $$\text{RT} \sim \text{Condition} + (1 + \text{Condition} \mid \text{Subject}) + (1 + \text{Condition} \mid \text{Item})$$
  - Log-transform or apply Box-Cox transformation to reaction time data to correct for right-skewness.
  - For binary/categorical outcomes (accuracy, choice), use Generalized Linear Mixed Models (`glmer(family = binomial)`).
  - Report: Fixed-effect estimates ($\beta$), standard errors ($SE$), $t$/$z$ statistics, degrees of freedom (Satterthwaite or Kenward-Roger approximation), and exact $p$-values or Bayes Factors ($BF_{10}$).

---

## 4. Field Linguistics, Language Documentation & Description

### A. Ethics & Indigenous Data Sovereignty
* **CARE Principles**: Complement FAIR data principles (Findable, Accessible, Interoperable, Reusable) with the **CARE Principles for Indigenous Data Governance**:
  - **C**ollective Benefit
  - **A**uthority to Control
  - **R**esponsibility
  - **E**thics
* **Free, Prior, and Informed Consent (FPIC)**: Document formal consent for audio, video, text, and downstream computational reuse.
* **Speaker & Consultant Attribution**: Explicitly credit language consultants and communities as intellectual collaborators, adhering to community-specified attribution preferences.

### B. Elicitation Methodologies
* Combine three complementary elicitation streams:
  1. *Targeted Structural Elicitation*: Paradigm filling, controlled minimal pairs, translation tasks.
  2. *Stimulus-Based Semi-Directed Elicitation*: Storyboards (e.g., Totem Field Storyboards), video clips (e.g., Staged Event clips, Pear Stories), spatial direction tasks.
  3. *Naturalistic Spontaneous Discourse*: Narratives, procedural descriptions, conversations, oral histories.

### C. Audio/Video Quality & Archival Standards
* **Audio Capture**: Lossless, uncompressed broadcast WAV format ($24\text{-bit}, 48\text{kHz}$ or $96\text{kHz}$). Use directional head-mounted or lavalier condenser microphones to eliminate room reverberation and distance variations.
* **Multi-Tier Time-Aligned Annotation (ELAN / FLEx)**:
  - Maintain structured hierarchical tiers in ELAN (`.eaf`):
    - `orthography` (base transcription)
    - `phonetic` (narrow IPA)
    - `morpheme-break` (segmented morphemes)
    - `morpheme-gloss` (Leipzig Glossing Rules)
    - `pos` (part-of-speech tags)
    - `translation` (free translation into reference language)
    - `notes` (pragmatic, sociolinguistic, or cultural commentary)
* **Open Archival Repositories**: Deposit corpus packages with open metadata (IMDI / OLAC) in established language archives (e.g., ELAR, PARADISEC, TLA, Kaipuleohone).

---

## 5. Corpus Linguistics & Quantitative Sociolinguistics

### A. Corpus Design & Representativeness
* Define sampling frames, genre stratification, and temporal/demographic balance before data collection.
* Distinguish reference corpora (e.g., BNC, COCA) from specialized domain corpora.

### B. Quantitative Collocation & Keyness Metrics
* **Collocation Strength**: Measure lexical association beyond raw co-occurrence:
  - *Pointwise Mutual Information (PMI)*: Measures informativeness, but biases toward low-frequency pairs.
  - *Log-Dice*: Scale-invariant metric directly comparing observed overlap against harmonic mean of marginals:
    $$\text{Log-Dice} = 14 + \log_2 \left( \frac{2 f_{xy}}{f_x + f_y} \right)$$
  - *$t$-score*: Measures certainty of association (favours high-frequency grammatical collocations).
* **Keyness & Keyword Analysis**: Compute Log-Likelihood Ratio ($G^2$) or Fisher's Exact Test to identify overrepresented words in target vs. reference corpora. Always report effect size measures (%DIFF or Relative Risk).
* **Dispersion Metrics**: Report Juilland's $D$ or Gries' Deviation of Proportions ($DP$) to ensure extracted patterns are evenly distributed across corpus files rather than driven by a single outlier document.

### C. Variationist Sociolinguistics
* **Variable Rule Analysis**: Model sociolinguistic variation using mixed-effects logistic regression (Rbrul / `glmer`):
  - Model the dependent linguistic variable (e.g., $(t, d)$-deletion, rhoticity, morphosyntactic variants) as a function of internal linguistic constraints (following phonological environment, stress, morphological status) and external social factors (age, gender, social network density, socioeconomic status).
* **Apparent-Time Construct**: Infer historical language change in progress by comparing synchronic age cohorts, validated against real-time trend or panel studies.

---

## 6. Historical, Comparative & Typological Linguistics

### A. The Comparative Method & Reconstruction
1. **Establish Regular Sound Correspondences**: Compile rigorous cognate sets across related languages; establish a complete matrix of regular phonetic correspondences.
2. **Reconstruct Proto-Forms ($*$)**: Reconstruct proto-phonemes that minimize the total number of unconditioned sound changes and adhere to natural phonological pathways (lenition, assimilation, palatalization).
3. **Isolate Borrowing & Substrate Influence**: Distinguish shared retentions (symplesiomorphies), shared innovations (synapomorphies—the only valid diagnostic for sub-grouping), and areal contact borrowings (isoglosses crossing genealogical boundaries).

### B. Dialectometry & Geolinguistics
* Compute aggregated linguistic distances between dialect varieties using Levenshtein distance on phonetic/phonemic transcriptions.
* Perform multidimensional scaling (MDS), cluster analysis, and spatial autocorrelation (Moran's $I$) to identify dialect boundaries, transition zones, and focal areas.

### C. Quantitative & Computational Typology
* **Sampling Bias Mitigation**: When testing universal linguistic correlations (e.g., Word Order vs. Morphological Type across WALS/Grambank/Autotyp), control for genealogical relatedness and areal diffusion (Galton's Problem) using **Phylogenetic Comparative Methods** (e.g., phylogenetic generalized least squares or mixed models with language family random effects).

---

## 7. Computational Linguistics, NLP & Speech Technology

### A. Theoretical & Information-Theoretic Bounding Before Modeling
* Quantify intrinsic target entropy $H(Y)$ and context-conditional entropy $H(Y \mid X)$.
* Measure Mutual Information:
  $$I(X; Y) = H(Y) - H(Y \mid X)$$
* **Argmax Invariance / Decision Boundary Property**:
  - When evaluating sequence-to-sequence or classification models under standard argmax decoding ($\hat{y} = \arg\max P(y \mid X)$), check whether context-conditioned posteriors flip the majority decision boundary.
  - If information gain $I(X; Y) > 0$ resides strictly below the decision threshold (e.g. $P(y^* \mid X) < 0.50$ in binary choices), optimal models achieve **0.00% argmax decision error reduction** over majority baselines without structured decoding or multi-candidate generation.

### B. Experimental Hygiene & Statistical Testing
* **Multi-Seed Protocols**: In low-resource or fine-tuning regimes ($\le 25\text{k}$ instances, few-shot prompting), run across $\ge 3$ random seeds. Report exact sample means and standard deviations ($\mu \pm \sigma$).
* **Instance-Level Paired Testing**:
  - *Categorical / Exact-Match*: Compute **Paired McNemar's Test** on the $2 \times 2$ contingency matrix ($n_{01}, n_{10}$ discordant pairs) with exact $p$-value.
  - *Continuous Metrics (BLEU, chrF, WER, Macro-F1)*: Compute Paired Bootstrap Resampling ($B \ge 1{,}000$ iterations) with $95\%$ confidence intervals.
* **Linguistic Unit Integrity**: Verify that tokenizers, aligners, and evaluation metrics operate over genuine linguistic units (phonemes, morphemes, grapheme clusters) rather than digital encoding artifacts (decomposed virāmas, uncombined diacritics).

### C. Multi-Modal & Acoustic Signal Verification
* When modeling phonetics, speech synthesis, or ASR:
  - Validate symbolic rules against physical acoustic signals: spectrograms, formant trajectories ($F_1, F_2, F_3$), voice onset time (VOT), duration of silent closure gaps, and fundamental frequency ($F_0$).
  - Ensure all harvested audio/text corpora maintain complete provenance tracking, contributor attribution manifests, and copyleft/open-access compliance.

### D. Deterministic Artifact Pipeline
1. Computational scripts output metrics to structured, machine-readable JSON/CSV files.
2. Tables and figures in manuscripts draw directly from frozen artifact files.
3. Every numerical result and statistical claim is reproducible via a single automated script.
