---
name: linguistics-paper-authoring
description: Universal paper authoring, typesetting, and typography guide for manuscripts across all subfields of linguistics (formal syntax/semantics, phonetics/phonology, psycholinguistics, field documentation, historical linguistics, sociolinguistics, NLP/computational linguistics). Covers Leipzig interlinear glossing, linguistic example numbering (gb4e/linguex/expex), syntactic trees (forest), OT tableaux, IPA phonetics (tipa/unicode), citation standards (LSA Unified Style Sheet, APA, ACL), orthography (Commonwealth & American English), statistical reporting, ethics/CARE/consultant attribution, and multi-tool AI disclosures.
---

# Universal Linguistics & Computational Linguistics Paper Authoring Guide

This skill governs the structure, typography, interlinear glossing, transliteration, statistical reporting, and ethical disclosure standards for manuscripts across all subfields of linguistics and computational linguistics (e.g., *Language*, *Linguistic Inquiry*, *Glossa*, *Journal of Memory and Language*, *Natural Language & Linguistic Theory*, *Phonology*, *Journal of Phonetics*, *Computational Linguistics*, *Transactions of the ACL*, *EMNLP*, *Interspeech*).

---

## 1. Orthographic Standards: Commonwealth vs. American English

The manuscript must strictly adhere to one consistent orthographic standard across all sections, captions, tables, and figures.

### Standard A: Commonwealth (British) English

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
| **travelling / dialled** | traveling / dialed |
| **utilising / utilised / utilisation** | utilizing / utilized / utilization |

* **Punctuation Rules in Commonwealth English**:
  - Place punctuation **outside** quotation marks unless the punctuation is an intrinsic part of the quoted material: *He referred to the morpheme as ‘bound’, not ‘free’.*
  - Prefer single quotation marks (`‘...’`) for initial quotes/terms and double (`“...”`) for nested quotes.

### Standard B: American English
* If American English is selected, use `-ize` verbal endings (*formalize, analyze*), `-or` noun endings (*behavior, color*), single consonants in inflections (*modeling, traveling*), and place commas/periods inside quotation marks (*"...bound," not "...free."*).

---

## 2. Authorial Voice & Scholarly Register

### A. Single-Author Manuscripts
Choose one of three standard academic styles and maintain consistency:
1. **The Inclusive Authorial "We" (*Universally standard in CS, NLP, and Math*)**:
   - Treats author and reader as exploring the empirical data together.
   - *Example*: *"We evaluate whether the classifier maintains sensitivity across low-resource splits."*
2. **Paper-Centric Active Voice (*Graceful Third-Person Alternative*)**:
   - Eliminates first-person pronouns without heavy passive voice constructions.
   - *Example*: *"This study evaluates the phonological transparency hypothesis..."*, *"Section 3 presents the acoustic formant analysis..."*
3. **The Personal "I" (*Standard in Descriptive, Field, and Sociolinguistics*)**:
   - Standard and preferred when describing primary fieldwork, elicitation sessions, and personal consultant interactions.
   - *Example*: *"During fieldwork in Madurai in 2024, I conducted semi-directed elicitation sessions with six native speakers."*

### B. Multi-Author Manuscripts
* Use standard plural *"we"*.
* Include a **CRediT Authorship Contribution Statement** (Conceptualization, Data Curation, Formal Analysis, Funding Acquisition, Investigation, Methodology, Software, Validation, Visualization, Writing - Original Draft, Writing - Review & Editing).

---

## 3. Typesetting Linguistic Examples & Leipzig Interlinear Glossing

### A. Standard LaTeX Example Environments (`gb4e` or `linguex`)
Every linguistic example must be numbered consecutively with sub-examples labeled `(a)`, `(b)`:

```latex
\usepackage{gb4e} % Put \noautomulticitation or place \usepackage{gb4e} as the LAST package before \begin{document}

\begin{exe}
\ex\label{ex:tamil-trans}
\begin{xlist}
\ex[]{
  \gll avaṉ vīṭṭ-ukku vanta-āṉ \\
       3\textsc{sg.m} house-\textsc{dat} come.\textsc{pst}-3\textsc{sg.m} \\
  \trans `He came to the house.'}
\ex[*]{
  \gll avaṉ vīṭṭ-ukku vanta-āḷ \\
       3\textsc{sg.m} house-\textsc{dat} come.\textsc{pst}-3\textsc{sg.f} \\
  \trans `*He came to the house (subject-verb agreement mismatch).'}
\end{xlist}
\end{exe}
```

### B. Leipzig Glossing Rules Reference
All interlinear morpheme glosses must strictly adhere to the **Leipzig Glossing Rules**:
1. **Morpheme Boundaries**: Use hyphens (`-`) for segmented morpheme boundaries in both source text and gloss.
2. **Clitic Boundaries**: Use equals signs (`=`) for clitic boundaries.
3. **Fused / Multi-Word Glosses**: Use periods (`.`) when a single morpheme expresses multiple grammatical categories (e.g., `come.PST`, `3SG.M.NOM`).
4. **Reduplication**: Use tildes (`~`) for reduplicated elements (`run~REDUP`).
5. **Infixes**: Use angle brackets (`<...>`) or backslashes (`\`).
6. **Small Capitals for Grammatical Categories**: Typeset all category labels in small capitals (`\textsc{...}`).

#### Standard Grammatical Category Abbreviations:
* **Person/Number**: `1`, `2`, `3`, `SG` (singular), `DU` (dual), `PL` (plural).
* **Case**: `NOM` (nominative), `ACC` (accusative), `DAT` (dative), `GEN` (genitive), `ERG` (ergative), `ABS` (absolutive), `LOC` (locative), `ABL` (ablative), `INS` (instrumental), `COM` (comitative), `VOC` (vocative).
* **Tense/Aspect/Mood**: `PST` (past), `PRS` (present), `FUT` (future), `IPFV` (imperfective), `PRF` (perfect), `PROG` (progressive), `HAB` (habitual), `SBJV` (subjunctive), `OPT` (optative), `COND` (conditional), `IMP` (imperative).
* **Voice/Valency**: `CAUS` (causative), `PASS` (passive), `APPL` (applicative), `REFL` (reflexive), `RECIP` (reciprocal), `MID` (middle).
* **Information Structure & Other**: `TOP` (topic), `FOC` (focus), `NEG` (negation), `Q` (question particle), `COMP` (complementizer), `REL` (relative), `CLF` (classifier), `NMLZ` (nominalizer).

### C. Acceptability & Grammaticality Judgment Prefixes
* `*`: Ungrammatical / phonotactically illegal.
* `?`: Marginally acceptable / slightly degraded.
* `??`: Highly degraded / questionable.
* `?*`: Almost ungrammatical.
* `#`: Semantically anomalous / pragmatically inappropriate in context.
* `%`: Dialectally variable / accepted only in specific regional varieties.
* `!`: Pragmatically infelicitous.

---

## 4. Phonetic, Phonological & Structural Typesetting

### A. Phonetic IPA Transcription
* Distinguish phonemic representations in slashes `/.../` from narrow phonetic transcriptions in brackets `[...]`.
* Use Unicode IPA directly or LaTeX `tipa`:
  ```latex
  \usepackage{tipa}
  \textipa{["k_h\&t]} \quad \text{or Unicode: } [kʰæt]
  ```

### B. Syntactic & Prosodic Trees (`forest`)
Use the modern `forest` package for clean, publication-ready tree structures:

```latex
\usepackage[linguistics]{forest}

\begin{forest}
[TP
  [DP [D [the]] [NP [N [linguist]]]]
  [T$'$
    [T [$\emptyset$]]
    [VP
      [V [analysed]]
      [DP [D [the]] [NP [N [data]]]]
    ]
  ]
]
\end{forest}
```

### C. Optimality Theory (OT) Tableaux
Format constraint ranking tableaux using standard `tabular` environments with shading:

```latex
\usepackage{colortbl}
\usepackage{pifont} % for \ding{43} candidate pointer

\begin{table}[h]
\centering
\begin{tabular}{|rcc||c|c|c|}
\hline
\multicolumn{3}{|c||}{/Input/} & \textsc{Constraint-1} & \textsc{Constraint-2} & \textsc{Constraint-3} \\
\hline\hline
\ding{43} a. & Candidate A & $[x.y]$ & & & * \\
\hline
          b. & Candidate B & $[x.z]$ & *! & & \\
\hline
          c. & Candidate C & $[y.z]$ & & *! \cellcolor{gray!20} & * \cellcolor{gray!20} \\
\hline
\end{tabular}
\caption{Optimality Theory tableau illustrating the ranking $\textsc{Con-1} \gg \textsc{Con-2} \gg \textsc{Con-3}$.}
\end{table}
```

### D. Attribute-Value Matrices (AVMs)
For HPSG, LFG, or Construction Grammar feature structures:
```latex
\usepackage{avm}
\avmfont{\sc}
\begin{avm}
\[ \textsc{syn} & \[ \textsc{head} & \textsc{noun} \\
                     \textsc{agr}  & \[ \textsc{num} & \textsc{sg} \\
                                        \textsc{pers} & 3 \] \] \\
   \textsc{sem} & \[ \textsc{index} & $i$ \] \]
\end{avm}
```

---

## 5. Transliteration Systems & Multi-Script Integrity

Whenever introducing non-Latin scripts, use the **Standard Quadruple Representation**:

1. **Native Script / Orthography**: Primary writing system.
2. **Standard Scientific Transliteration**:
   - *Indic*: ISO 15919 or IAST
   - *Arabic / Semitic*: DIN 31635 or ALA-LC
   - *Cyrillic*: ISO 9 or ALA-LC
   - *Mandarin Sinitic*: Hanyu Pinyin with tone marks
   - *Japanese*: Modified Hepburn
   - *Ancient Greek*: ISO 843
3. **Phonetic / Phonemic IPA**: Enclosed in `/.../` or `[...]`.
4. **English Gloss**: Enclosed in quotation marks.

$$\text{Native: \textbf{படம்}} \quad \longrightarrow \quad \text{ISO 15919: \textit{pa\d{t}am}} \quad \longrightarrow \quad \text{IPA: } [pɐɖɐm] \quad \longrightarrow \quad \text{Gloss: ``picture/movie''}$$

---

## 6. Statistical Reporting Standards Across Paradigms

### A. Experimental & Psycholinguistics (Mixed-Effects Models)
* Report fixed-effect coefficients ($\beta$), standard errors ($SE$), test statistics ($t$ or $z$), and $p$-values:
  * *"Linear mixed-effects modeling revealed a significant main effect of predictability ($\beta = -34.2\,\text{ms}$, $SE = 8.1$, $t = -4.22$, $p < 0.001$)."*
* Always specify the random-effects structure and whether it was maximal or reduced due to singular fit.

### B. Corpus & Sociolinguistics
* Report frequency counts, normalized frequencies (per million words), association scores ($\text{Log-Dice}$, $G^2$, PMI), and dispersion measures ($DP$):
  * *"The collocation exhibited strong association ($\text{Log-Dice} = 11.4$, $G^2 = 842.1$, $p < 0.0001$) and even distribution across genres ($DP = 0.18$)."*

### C. Computational Linguistics / NLP
* Report exact means and sample standard deviations over $\ge 3$ random seeds: $\mu \pm \sigma$.
* For categorical exact-match accuracy, report **Paired McNemar's Test** with discordant pair counts ($n_{01}, n_{10}$) and $\chi^2$.
* For continuous generation metrics (BLEU, chrF, WER), report **Paired Bootstrap Resampling** with 95% confidence intervals.
* For annotation reliability, report **Cohen's $\kappa$** (two raters) or **Krippendorff's $\alpha$** (multiple raters/continuous scales).

---

## 7. Universal 0-Error Overleaf LaTeX Architecture

Use a robust, cross-engine preamble compatible with **pdfLaTeX**, **XeLaTeX**, and **LuaLaTeX**:

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
  % Universal Diacritic & Phonetic Fallbacks for pdfLaTeX
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
  \newunicodechar{ə}{\textschwa}
  \newunicodechar{ɪ}{\textsci}
  \newunicodechar{ʊ}{\textupsilon}
  \newunicodechar{ʌ}{\textturnv}
  \newunicodechar{ɔ}{\textopeno}
  \newunicodechar{ɛ}{\textepsilon}
  \newunicodechar{ɲ}{\textltailn}
  \newunicodechar{ʃ}{\textesh}
  \newunicodechar{ʒ}{\textezh}
  \newunicodechar{ʔ}{\textglotstop}
\fi

\bibliographystyle{plainnat}
```

---

## 8. Citation Standards & Bibliography

### A. Unified Style Sheet for Linguistics (LSA / *Language* / *Glossa* / *Linguistic Inquiry*)
* **In-Text Citations**:
  - Narrative: *Chomsky (1965: 25) argued that...*
  - Parenthetical: *(Bresnan et al., 2001: 110)*
  - Multiple works: *(Hale, 1983; Jelinek, 1984; Baker, 1996)*
* **Bibliography Format**:
  - Book: Chomsky, Noam. 1965. *Aspects of the Theory of Syntax*. Cambridge, MA: MIT Press.
  - Journal Article: Bresnan, Joan, Shipra Dingare & Christopher D. Manning. 2001. Soft constraints mirror hard constraints: Voice and person in English and Lummi. *Syntax* 4(1). 1–25.
  - Edited Volume Chapter: Kiparsky, Paul. 1982. From cyclic phonology to lexical phonology. In Harry van der Hulst & Norval Smith (eds.), *The Structure of Phonological Representations (Part I)*, 131–175. Dordrecht: Foris.

### B. Computational Linguistics (ACL Anthology / IEEE)
* Use standard ACL BibTeX templates with `\citet{...}` and `\citep{...}`.

---

## 9. Ethics, Fieldwork Consent & Multi-Tool AI Assistance Disclosures

### A. Language Consultant & Fieldwork Ethics Statement
Include a dedicated statement for studies involving field data, human participants, or indigenous languages:
```latex
\section*{Ethics and Language Consultant Attribution}
All field elicitation and experimental procedures were conducted under [Institutional Review Board / Ethics Committee Protocol \#XXXXX]. Free, Prior, and Informed Consent (FPIC) was obtained from all language consultants. In accordance with the CARE Principles for Indigenous Data Governance, the contributing speakers [and community representatives] retain sovereignty over their cultural and linguistic materials. We gratefully acknowledge the intellectual collaboration of our consultants [Names / Community attribution].
```

### B. Transparent Multi-Tool AI-Assisted Technologies Disclosure
When AI tools are used during research, provide an explicit, unnumbered section before the bibliography specifying functional roles:

```latex
\section*{Use of AI-Assisted Technologies}
During the preparation of this manuscript, the author(s) utilised AI-assisted tools for specific research workflows:
\begin{itemize}
    \item \textbf{[Conceptualisation / Review Model, e.g., Anthropic Claude]}: Utilised during early research planning for literature review mapping and experimental design critique, and subsequently provided structural and editorial feedback on working drafts.
    \item \textbf{[Agentic Coding Model, e.g., Google Antigravity / Gemini]}: Utilised as an agentic pair-programming assistant for implementing statistical processing scripts, orchestrating reproducible computational benchmarking, and formatting \LaTeX\ typography and bibliography assets.
\end{itemize}
The author(s) conceived the research questions, formulated the theoretical models, directed the methodology, verified all empirical and statistical results, authored the final text, and take full scientific responsibility for the integrity and conclusions of this work.
```
