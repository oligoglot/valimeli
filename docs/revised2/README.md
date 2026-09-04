# Replacement manuscript — Project ValiMeli

Built from verified artifacts in `Gemini/valimeli/` only. Every number in every table is
emitted programmatically from a named `artifacts/*.json` file; none was typed by hand.

## Contents
- `main.tex` — full manuscript
- `table_entropy.tex`   Table 1, from `artifacts/voicing_entropy_results.json`
- `table_context.tex`   Table 2, same source, per-context breakdown
- `table_scale.tex`     Table 3, from the eight per-arm `*_results.json` files
- `table_disagree.tex`  Table 4, recomputed here with the repo's own DP aligner
                        (`src/compute_voicing_entropy.py`); output saved as
                        `valimeli_disagreement_recomputed.json`
- `table_downstream.tex` Table 5, from BOTH downstream artifacts
- `table_control.tex`   Appendix B, from `multilingual_*_scaling_results.json`
- `references.bib`      verified bibliography. The IJDL biconsonantal-clusters entry is now
                        verified against the journal front matter (Vol. LIV No. 1, Jan 2025,
                        pp. 140-171, three authors) -- the Pulli draft had it single-authored
                        on pp. 45-68. The FIRE 2020 Theedhum Nandrum entry is
                        verified against the paper's first page and cites the CEUR volume and
                        paper URL rather than a page range, which that volume does not use.
- `fig_scaling_corrected.png` Figure 1

## Deliberate departures from `docs/paper_skeleton.md`
1. The 3.2M-pair run is demoted from headline result to negative control. The auxiliary
   labels in that run are derived from the encoder's own input and are constant for six of
   eight languages, so the treatment was never administered.
2. The scaling result is reported as two mechanism-specific comparisons rather than one
   pooled curve, because the 25k/1.0M arms and the 250k/500k arms are different interventions.
3. Table 5 reports both downstream experiments, which disagree in sign on Malayalam.
4. The disagreement statistics are recomputed rather than reused: the skeleton's 64.02%
   figure appears in no artifact and in no script. The recomputation broadly confirms it
   (65.38% Tamil post-nasal) but the exact figure could not be reproduced.
5. The abstract's illustrative pair is replaced with கொண்டாடி, whose four romanisations are
   attested single-word variants that split on the post-nasal contrast.
6. The BiGRU parameter count is dropped (recorded in no artifact) and lambda = 0.3 is stated.

## To build
    pdflatex main && bibtex main && pdflatex main && pdflatex main

7. The `lakshmanan2025biconsonantal` citation is corrected from the journal's own volume front
   matter rather than from an index: three authors (Ramprashanth V., Kumarasamy R.,
   BalaSundaraRaman L.), pages 140-171. Neither Crossref nor OpenAlex indexes IJDL.

8. The `lakshmanan2020theedhum` citation is verified from the paper's first page: two authors
   (BalaSundaraRaman Lakshmanan, Sanjeeth Kumar Ravindranath), CEUR-WS Vol-2826 paper T4-23,
   no page range. The three page ranges previously in circulation are all spurious.
