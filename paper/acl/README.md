# paper/acl — ACL/BioNLP submission

`acl_latex.tex` is the paper in the official \*ACL template, currently in
`[review]` mode (anonymous, line numbers). Switch `review` → `final` for the
camera-ready, or → `preprint` for a non-anonymous version with page numbers.

## Build (local)

    pdflatex acl_latex && bibtex acl_latex && pdflatex acl_latex && pdflatex acl_latex

or simply `latexmk -pdf acl_latex`.

`acl.sty` and `acl_natbib.bst` are now in this folder (fetched from
<https://github.com/acl-org/acl-style-files>, master). If they ever go missing:

    curl -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl.sty
    curl -O https://raw.githubusercontent.com/acl-org/acl-style-files/master/acl_natbib.bst

Figures are resolved through `\graphicspath` from `figures/final/` in the repo
root, so nothing needs copying. If you move this folder, adjust that one line.

## Build (Overleaf)

`NEL_paper_overleaf.zip` in this folder is the self-contained project:
**New Project → Upload Project → pick the ZIP**. Main file `main.tex`,
compiler pdfLaTeX. It contains the `.tex`, both style files, `custom.bib` and
the six figure PDFs flattened into the root — the `{./}` entry in
`\graphicspath` is what makes the bare `\includegraphics{fig1_value.pdf}`
calls resolve there.

Pasting only the `.tex` into a blank Overleaf project cannot work: `acl.sty`
is not part of Overleaf's TeX Live, and the figures and the `.bib` are not
either.

Regenerate the ZIP after editing:

    cd paper/acl && rm -f NEL_paper_overleaf.zip && \
      cp acl_latex.tex main.tex && \
      zip -j NEL_paper_overleaf.zip main.tex acl.sty acl_natbib.bst custom.bib \
        ../../figures/final/fig{1_value,3_dangerzone,4_deference,5_churn,6_dose,7_prompt}.pdf && \
      rm main.tex

## Status

* Compiles clean: no errors, no undefined references or citations, no BibTeX
  warnings, no overfull box above 1.2 pt (that one is a prose line in §5.4 and
  is invisible at print size).
* `acl_latex_preview.pdf` is the current build, **13 pages**: content pp. 1–10,
  Limitations from the middle of p10, references pp. 11–12, appendices A/B on
  pp. 12–13.
* **Content is ~1.7 pages over the 8-page \*ACL limit.** Not an issue for the
  lab report; it is the first thing to fix before a workshop submission. The
  obvious candidates are §5.4 (prompt ablations, could become an appendix) and
  the four-corpus example table in §3.
* Table 4 (`tab:ft`) reports the seed range as a sub/superscript stack rather
  than in parentheses — the parenthesised form was 23 pt wider than
  `\columnwidth` and overhung the gutter.
* Author block still has the template placeholders. `[review]` hides them, so
  fill them in before the camera-ready, not before submission.

## Every number is recomputed, not copied

Nothing was carried over from the old `paper/main.tex`. Every figure in
`figures/final/` and every number in the text and the tables is recomputed from
the `preds_*.jsonl` dumps in the repo root by the scripts in
`figures/final/`. Numbers that differ from the old draft are corrections, most
importantly:

* The **frame** (retriever strength → LLM gain) is *not* significant when pooled
  across corpora (ρ = −0.50, p = 0.069). It is significant *within* a corpus
  (12/14 ordered pairs, sign test p = 0.013; dataset-demeaned ρ = −0.72,
  p = 0.004). The paper now states both.
* The **gate** does not uniformly beat the full LLM. Over 20 document splits it
  wins on BC5CDR/BioSyn (+1.21 ± 0.27) and BC5CDR/pipeline (+0.55 ± 0.27), is
  neutral on SapBERT and MedMentions, and **loses on BioRED (−1.11 ± 0.92)**.
  The old single-split claim was optimistic.
* **Three fine-tuning seeds replaced the single run** and corrected two claims.
  Pooled precision in the truly-uncertain band `[0,2)` *falls* under
  fine-tuning (96.1 % → 91.6 %, range 90.6–92.5); the single run said it was
  flat. The BioRED band repair is 39 % (range 38–41), not the 69 % a single run
  suggested, and that band stays net-harmful. Deference held and sharpened:
  seen 68.7 → 90.5 % (88.2–94.9) against unseen 72.2 → 69.8 % (66.7–73.0),
  non-overlapping ranges.
* **fig4_memorization** and **fig2_substitution** (the old figures) are gone;
  `fig4_deference.pdf` and `fig5_churn.pdf` replace them and are built from the
  current dumps by `make_fig4_deference.py` / `make_fig5_churn.py`.
