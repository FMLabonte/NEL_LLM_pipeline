#!/usr/bin/env python3
"""Teaser for page 1 - the paper's thesis in one single-column panel.

This reproduces panel (a) of Figure~\\ref{fig:value} at \\columnwidth: BC5CDR,
four retrievers, restricted to the 9,391 mentions all four cover.

Why BC5CDR only and not all fourteen runs. The paper's frame is identified
*within* a corpus and is confounded across corpora (pooled rho = -0.50, p =
0.069, against a dataset-demeaned rho = -0.72, p = 0.004). A teaser drawing
curves from five different corpora would therefore illustrate the weaker version
of the claim on the paper's own account. Holding the corpus and the mention set
fixed and varying only the retriever is the comparison the paper actually
defends, so it is the one that belongs on page 1.

    python3 figures/final/make_fig0_teaser.py
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aclfig import (COL, INK, MUTED, NEG, load, common_subset, value_profile,
                    recall_at_1, use_acl, style, save)

# ordered weakest -> strongest; the ramp is read off that order, not hard-coded,
# so re-running with a different retriever set cannot mislabel the colours.
RETRIEVERS = [("BM25",          "preds_zs_bc5cdr_bm25.jsonl"),
              ("SapBERT",       "preds_zs_bc5cdr_sapbert.jsonl"),
              ("rule pipeline", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl"),
              ("BioSyn",        "preds_zs_bc5cdr_biosyn.jsonl")]

rows = common_subset([load(f) for _, f in RETRIEVERS])
labs = [lab for lab, _ in RETRIEVERS]
R1 = [recall_at_1(r) for r in rows]
order = sorted(range(len(rows)), key=lambda i: R1[i])
blues = mpl.colormaps["Blues"]
COLOR = {i: blues(0.42 + 0.19 * k) for k, i in enumerate(order)}

use_acl()
fig, ax = plt.subplots(figsize=(COL, 2.15), dpi=400, layout="constrained")
fig.get_layout_engine().set(w_pad=.02, h_pad=.02)

style(ax)
ax.axhspan(-5.5, 0, color=NEG, alpha=.07, lw=0, zorder=0)
ax.axhline(0, color="#2a2a2a", lw=.8, zorder=2)
for i in order:
    x, y = value_profile(rows[i], window=.25, step=.005)
    ax.plot(x, y, color=COLOR[i], lw=1.25, zorder=3)

ax.set_xlim(10, 90)
ax.set_ylim(-5.5, 21.5)
ax.set_yticks([-5, 0, 5, 10, 15, 20])
ax.set_xticks([10, 30, 50, 70, 90])
ax.set_xlabel("retriever confidence percentile", labelpad=2)
ax.set_ylabel("local net $\\Delta$ Accuracy@1\nof the LLM stage (points)", labelpad=2)

ax.text(12.5, 16.4, "BC5CDR, 9,391 shared mentions", fontsize=6.2, color=MUTED,
        ha="left", va="center")
ax.text(88.5, -5.0, "LLM net-harmful", ha="right", va="bottom", fontsize=6.4,
        color=NEG, style="italic")

h = [Line2D([], [], color=COLOR[i], lw=1.4, label=f"{labs[i]}  (R@1 {R1[i]:.0f})")
     for i in order]
ax.legend(handles=h, loc="upper right", frameon=False, fontsize=6.3,
          handlelength=1.3, handletextpad=.5, labelspacing=.30,
          borderpad=.1, borderaxespad=.2)

save(fig, "fig0_teaser")
print("  R@1: " + ", ".join(f"{labs[i]} {R1[i]:.1f}" for i in order))
