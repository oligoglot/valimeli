# Replacement manuscript — Project ValiMeli

Built from verified artifacts in `Gemini/valimeli/` only. Every number in every table is
emitted programmatically from a named `artifacts/*.json` file; none was typed by hand.

## Contents
- `main.tex` — full manuscript with Paper-Centric Active Voice (Option B)
- `table_entropy.tex`   Table 1, from `artifacts/voicing_entropy_results.json` (text) and `artifacts/acoustic_voicing_entropy_results.json` (speech)
- `table_context.tex`   Table 2, same sources, per-context breakdown
- `table_scale.tex`     Table 3, from `multiseed_rigorous_results.json` (25k) and scaling `*_results.json` files
- `table_disagree.tex`  Table 4, computed with `src/compute_dakshina_disagreement.py`; output saved as
                        `valimeli_disagreement_slots.json` (slots) and `valimeli_disagreement_pairs.json` (pairs)
- `table_downstream.tex` Table 5, from BOTH downstream artifacts
- `table_acoustic_alignment.tex` Table A (Appendix / Section 1), from `artifacts/acoustic_wav_analysis_summary.json`
- `references.bib`      verified bibliography (14 entries, including foundational phonetics and phonology literature; all verified against primary records).
- `fig_effects_corrected.png` Figure 1 (forest plot of all 10 matched comparisons with multi-seed error bars)
- `acoustic_voicing_spectrograms.png` Figure 2 (spectrograms rendered with Arial Unicode MS)

## Deliberate departures from `docs/paper_skeleton.md`
1. The scaling result is reported as two mechanism-specific comparisons rather than one pooled curve, because the 25k/1.0M arms (auxiliary loss) and the 250k/500k arms (string tags) are different interventions.
2. Multi-seed 25k results (seeds 42, 43, 44) are reported as sample standard deviation ($s$, $N=3$) and show overlapping performance within seed variance ($\Delta = -0.14\%$ for Tamil, $+0.15\%$ for Malayalam), with paired McNemar tests showing non-significant differences ($p = 0.585$ and $p = 0.901$).
3. Table 4 reports slot-level disagreement ($69.26\%$ Tamil post-nasal split slots) and pair-level disagreement ($44.50\%$), with Wild Tanglish post-nasal voicing ($64.00\%$) and overall wild rate ($24.67\%$).
4. Table 5 reports both downstream experiments, which disagree in sign on Malayalam.
5. All acoustic claims are strictly grounded in the 5,325-recording open speech index (`artifacts/tamil_speech_full_corpus_manifest.json`), with slot-level acoustic feature extractions across 11,590 plosive slots (`artifacts/tamil_speech_acoustic_features_5k.json`) and full-population acoustic voicing entropy evaluation over 12,435 plosive slots (`artifacts/acoustic_voicing_entropy_results.json`).
6. Table 1 (`table_entropy.tex`) and Table 2 (`table_context.tex`) directly contrast written orthography (Dakshina text) with spoken speech acoustics (Lingua Libre), demonstrating that the argmax barrier ($0.00\%$ gain) is an artifact of Latin text crowdsourcing, whereas spoken acoustics exhibits $+32.92\%$ relative error reduction ($70.29\%$ rule accuracy) with $99.72\%$ initial voicelessness and $85.14\%$ post-nasal voicing.
7. Geminate closure duration ($50.5\text{ ms}$) is contrasted with post-nasal transition ($4.7\text{ ms}$), explaining the low-$F_0$ formant bleed in automated frame extraction and confirming physical fortis voicelessness.
8. The BiGRU parameter count is omitted (unrecorded in artifacts) and $\lambda = 0.3$ is explicitly stated.
9. Citations for `lisker1958tamil` (*Indian Linguistics* 1958, Turner Jubilee Vol. I), `keane2004dravidian` (DOI `10.1017/S0025100304001549`), `lakshmanan2025biconsonantal`, `vemula-etal-2025-rethinking`, `lakshmanan2020theedhum`, `ramesh-etal-2022-samanantar`, and `christdas1988phonology` are verified against primary records.
10. Formal Proposition 1 mathematically proves Argmax Invariance under Skewed Allophony, demonstrating zero Bayes risk reduction under 0-1 loss despite strictly positive mutual information.
11. An empirical audit of the 34.69% unaligned Malayalam pairs confirms that unaligned pairs are strongly enriched in Sanskrit/Grantha loanword markers (`artifacts/malayalam_alignment_audit.json`; odds ratio $7.30\times$, $\chi^2 = 24{,}203.8$, $p < 10^{-300}$), while unaligned pairs exhibit fewer complex conjunct ligatures ($46.17\%$ vs.\ $51.01\%$).
12. Section 7 provides four actionable architectural principles for Dravidian Automatic Speech Recognition (ASR) derived from the acoustic findings.

## To build
    pdflatex main && bibtex main && pdflatex main && pdflatex main
