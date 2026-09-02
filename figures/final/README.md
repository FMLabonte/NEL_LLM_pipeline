# figures/final — Figure 1 and Figure 2

Built for the ACL two-column template: both are **6.3 in wide** (`\textwidth`),
so both go in a `figure*`. Ready-to-paste `figure*` environments with captions
are in `captions.tex`.

    cd figures/final
    python3 make_fig1_value.py    # -> fig1_value.{pdf,png}
    python3 make_fig2_budget.py   # -> fig2_budget.{pdf,png}

No model and no cluster: everything is recomputed from the `preds_*.jsonl` dumps
in the repository root. Requires numpy, scipy, matplotlib.

Type is TeX Gyre Termes (Times-metric), so the figures match the ACL body font.
If that face is missing locally, matplotlib falls back to Times New Roman →
Liberation Serif → DejaVu Serif; install `tex-gyre` for an exact match.

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
  wall-times are recoverable from `bender_logs/*.out` (the `[NNNNs, ETA …]`
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

## Still open

`figures/make_fig1_heatmap.py` has the same BM25 sentinel issue and does **not**
handle it — its Q5 column for the BM25 row is a block of 999.0 values. Fix or
retire it.
