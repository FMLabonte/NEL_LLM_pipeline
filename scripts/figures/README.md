# scripts/figures — figure scripts for the paper and the lab report

Every figure is recomputed from the per-mention `preds_*.jsonl` dumps in the
repository root. No model and no cluster are needed. Requires numpy, scipy,
pandas and matplotlib. Run the scripts from the repository root; all output
(PDF and PNG) is written to `figures/`, which is not tracked.

    python3 scripts/figures/make_fig_frame.py      # -> figures/fig_frame.{pdf,png}

`aclfig.py` holds the shared loaders, band definitions, colours and the ACL
figure geometry. Every other script imports it.

| Script | Output | Used in |
|---|---|---|
| `make_fig_frame.py` | `fig_frame` | report, Fig. 1 (gain vs. retriever strength, 14 runs) |
| `make_fig_danger.py` | `fig_danger` | report, Fig. 2 (confidence bands, BC5CDR) |
| `make_fig_finetune.py` | `fig_finetune` | report, Fig. 3 (LoRA fine-tuning per corpus) |
| `make_fig_data.py` | `fig_data` | report, appendix (corpus composition); needs `corpus_stats.py` first |
| `corpus_stats.py` | `corpus_stats.csv`, `corpus_types.json` | input for `make_fig_data.py`; reads `Data/` |
| `fig_pipeline.tex` | `fig_pipeline.pdf` | report, appendix (pipeline overview); `pdflatex -output-directory figures scripts/figures/fig_pipeline.tex` |
| `make_fig6_dose.py` | `fig6_dose` | paper and report appendix (controlled degradation) |
| `make_fig0_teaser.py`, `make_fig1_value.py`, `make_fig2_budget.py`, `make_fig4_deference.py`, `make_fig5_churn.py`, `make_fig7_prompt.py` | `fig0`–`fig7` | ACL paper draft |

Type is TeX Gyre Termes (Times-metric), so the figures match the body font.
If that face is missing locally, matplotlib falls back to Times New Roman,
Liberation Serif and DejaVu Serif; install `tex-gyre` for an exact match.

## What each figure claims

**Figure 1 — `fig1_value.pdf`.** Where along the retriever's own confidence the
LLM stage pays off. Panel (b) is the quantitative version of what the old
heatmap only showed qualitatively: net Δ Accuracy@1 on the confident 80 % falls
by 0.17 points per point of Recall@1 (Spearman ρ = −0.90, p = 0.002) and crosses
zero at Recall@1 ≈ 80.

**Figure 2 — `fig2_budget.pdf`.** Accuracy@1 against the share of mentions routed
to the LLM. Panel (a) is the controlled comparison: identical mentions, only the
retriever changes. Replaces both the old `fig1_curve` heatmap and
`f1b_gate_pareto` — and in absolute accuracy rather than "% of the full gain",
which is what produced the misleading 168 % spike in the old Pareto plot.

## Decisions baked into the code

* **Cost axis = share of LLM calls**, not wall-clock. The LLM call count
  dominates runtime, and the share needs no assumption about hardware. Real
  wall-times are recoverable from the cluster logs (the `[NNNNs, ETA …]`
  progress lines) if a second axis is ever wanted, but those are shared-cluster
  times and are not comparable across configurations.
* **Rank axis, not score axis.** Retriever confidence is binned by percentile
  *within each run*, because BM25, cosine and rule scores are on different
  scales. This is why `preds_zs_bc5cdr_bm25.jsonl`'s `999.0` sentinel (10.9 % of
  rows: no second candidate, hence maximal confidence) can be kept rather than
  dropped — a rank axis is insensitive to its magnitude. `aclfig.by_confidence`
  documents this.
* **Common subset for Figure 2(a).** The BM25 dump covers 9,391 of the 9,661
  mentions (the abbreviation path drops 270). `common_subset` restricts all four
  runs to those 9,391 so "identical mentions" is literally true — which is why
  the R@1 values in that panel sit slightly above the full-set numbers in the
  results table (e.g. SapBERT 77.1 vs 75.9). Set `COMMON = False` in
  `make_fig2_budget.py` to plot each run on its own full set instead.
* **Palette.** Categorical colours (per dataset) are validated for deuteranopia,
  protanopia and tritanopia. Retriever strength is a magnitude, so it gets a
  sequential ramp with a colourbar, never a categorical hue.

## Single column instead

If a figure needs to fit one column, change `WIDE` to `COL` (3.15 in) in the
`plt.subplots` call and switch `subplots(1, 2, ...)` to `subplots(2, 1, ...)`;
the panels stack and everything else scales.
