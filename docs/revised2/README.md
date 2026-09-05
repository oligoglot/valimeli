# Replacement manuscript — Project ValiMeli

Built from verified artifacts in `Gemini/valimeli/` only. Every number in every table is
emitted programmatically from a named `artifacts/*.json` file; none was typed by hand.

## Contents
- `main.tex` — full manuscript with Paper-Centric Active Voice (Option B)
- `table_entropy.tex`   Table 1, from `artifacts/voicing_entropy_results.json`
- `table_context.tex`   Table 2, same source, per-context breakdown
- `table_scale.tex`     Table 3, from `multiseed_rigorous_results.json` (25k) and scaling `*_results.json` files
- `table_disagree.tex`  Table 4, computed with `src/compute_dakshina_disagreement.py`; output saved as
                        `valimeli_disagreement_slots.json` (slots) and `valimeli_disagreement_pairs.json` (pairs)
- `table_downstream.tex` Table 5, from BOTH downstream artifacts
- `table_acoustic_alignment.tex` Table A (Appendix / Section 1), from `artifacts/acoustic_wav_analysis_summary.json`
- `references.bib`      verified bibliography (12 entries).
- `fig_effects_corrected.png` Figure 1 (forest plot of all 10 matched comparisons with multi-seed error bars)
- `acoustic_voicing_spectrograms.png` Figure 2 (spectrograms rendered with Arial Unicode MS)

## Deliberate departures from `docs/paper_skeleton.md`
1. The scaling result is reported as two mechanism-specific comparisons rather than one pooled curve, because the 25k/1.0M arms (auxiliary loss) and the 250k/500k arms (string tags) are different interventions.
2. Multi-seed 25k results (seeds 42, 43, 44) are reported as sample standard deviation ($s$, $N=3$) and show overlapping performance within seed variance ($\Delta = -0.14\%$ for Tamil, $+0.15\%$ for Malayalam), with paired McNemar tests showing non-significant differences ($p = 0.585$ and $p = 0.901$).
3. Table 4 reports slot-level disagreement ($69.26\%$ Tamil post-nasal split slots) and pair-level disagreement ($44.50\%$), with Wild Tanglish post-nasal voicing ($64.00\%$) and overall wild rate ($24.67\%$).
4. Table 5 reports both downstream experiments, which disagree in sign on Malayalam.
5. All acoustic claims are strictly grounded in the 11 verified Wikimedia Commons recordings (`artifacts/acoustic_wav_analysis_summary.json`) and the 5,325-recording open speech index (`artifacts/tamil_speech_full_corpus_manifest.json`).
6. The BiGRU parameter count is omitted (unrecorded in artifacts) and $\lambda = 0.3$ is explicitly stated.
7. Citations for `lakshmanan2025biconsonantal`, `vemula-etal-2025-rethinking`, `lakshmanan2020theedhum`, and `ramesh-etal-2022-samanantar` are fully verified against original primary sources.

## To build
    pdflatex main && bibtex main && pdflatex main && pdflatex main
