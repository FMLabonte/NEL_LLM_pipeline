#!/usr/bin/env python3
"""Figure 2 - Accuracy@1 against the LLM budget.

The gate routes the least-confident mentions to the LLM first, so the x axis is
'call the LLM on the b % of mentions with the smallest top1-top2 gap'.  Behind a
strong retriever the curve turns over: the best operating point uses a small
budget and beats calling the LLM on everything.

(a) BC5CDR, four retrievers, identical mentions - the controlled comparison.
(b) five datasets under dense retrieval - the same shape across corpora.

    python3 figures/final/make_fig2_budget.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from aclfig import (WIDE, DSCOL, INK, MUTED, load, common_subset, budget_curve,
                    dev_tuned_gate, use_acl, style, save)

# Restrict the four BC5CDR runs to the 9,391 mentions all of them cover, so
# panel (a) really is 'same mentions, only the retriever changes'.  Set to False
# to plot each run on its own full set (BM25 then covers 9,391 of 9,661).
COMMON = True

MARKS = np.array([0, 5, 10, 20, 30, 50, 75, 100.0])

BC5 = [("BM25",          "preds_zs_bc5cdr_bm25.jsonl",           "s"),
       ("SapBERT",       "preds_zs_bc5cdr_sapbert.jsonl",        "o"),
       ("rule pipeline", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl",  "D"),
       ("BioSyn",        "preds_zs_bc5cdr_biosyn.jsonl",         "^")]

PER_DATASET = [("MedMentions", "preds_zs_medmentions_emb.jsonl"),
               ("NCBI",        "preds_zs_ncbi_sapbert.jsonl"),
               ("BioRED",      "preds_zs_biored_sapbert.jsonl"),
               ("NLM-Chem",    "preds_zs_nlmchem.jsonl"),
               ("BC5CDR",      "preds_zs_bc5cdr_sapbert.jsonl")]

use_acl()
fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDE, 2.5), dpi=400, layout="constrained",
                               gridspec_kw={"width_ratios": [1, 1]})
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.07, hspace=0)

# ------------------------------------------------ (a) one dataset, four retrievers
style(axL)
runs = [load(f) for _, f, _ in BC5]
if COMMON:
    runs = common_subset(runs)
n_common = len(runs[0])
summary = []
BLUE = DSCOL["BC5CDR"]
shades = ["#6aabd6", "#2f88bf", "#0f6199", "#08375a"]
for (name, _, mk), rows, col in zip(BC5, runs, shades):
    frac, acc = budget_curve(rows)
    axL.plot(frac, acc, color=col, lw=1.4, zorder=3)
    mx, my = budget_curve(rows, MARKS)
    axL.plot(mx, my, ls="none", marker=mk, ms=3.9, mfc="white", mec=col, mew=1.05, zorder=4)
    _, calls, _ = dev_tuned_gate(rows)
    axL.plot(calls, np.interp(calls, frac, acc), marker="*", ms=8.5, color=col,
             mec="white", mew=.5, zorder=6)
    axL.annotate(f"{name}", (100, acc[-1]), xytext=(4, 0), textcoords="offset points",
                 va="center", fontsize=7, color=col)
    axL.annotate(f"{acc[0]:.1f}", (0, acc[0]), xytext=(-3.5, 0), textcoords="offset points",
                 va="center", ha="right", fontsize=6.4, color=MUTED)
    summary.append((name, acc[0], acc[-1], acc.max(), frac[int(np.argmax(acc))]))
axL.set_xlim(-13, 126)
axL.set_xticks([0, 25, 50, 75, 100])
axL.set_ylim(72.6, 88.2)
axL.set_xlabel("LLM budget: mentions sent to the LLM (%)", labelpad=2)
axL.set_ylabel("Accuracy@1 (%)", labelpad=2)
axL.text(.985, .955, "(a)", transform=axL.transAxes, ha="right", va="top",
         fontsize=8.5, fontweight="bold")
best = summary[-1]
axL.annotate(f"gate beats the\nfull LLM by {best[3]-best[2]:.1f}",
             xy=(best[4], best[3] + .12), xytext=(37, 87.8),
             fontsize=6.6, color=shades[3], va="center", linespacing=1.15,
             arrowprops=dict(arrowstyle="-", lw=.6, color=shades[3],
                             connectionstyle="arc3,rad=-.25"))
h = [Line2D([], [], ls="none", marker="*", ms=8, color=MUTED,
            label="budget tuned on dev documents")]
axL.legend(handles=h, loc="lower right", frameon=False, handletextpad=.25,
           borderpad=.1, borderaxespad=.3)

# ------------------------------------------------ (b) five datasets
style(axR)
axR.axhline(0, color="#2a2a2a", lw=.8, ls=(0, (3.5, 2.5)), zorder=1)
for ds, f in PER_DATASET:
    rows = load(f)
    frac, acc = budget_curve(rows)
    g = acc - acc[0]
    axR.plot(frac, g, color=DSCOL[ds], lw=1.4, zorder=3)
    mx, my = budget_curve(rows, MARKS)
    axR.plot(mx, my - acc[0], ls="none", marker="o", ms=3.6, mfc="white",
             mec=DSCOL[ds], mew=1.05, zorder=4)
    axR.plot(100, g[-1], marker="o", ms=3.6, color=DSCOL[ds], zorder=5)
    axR.annotate(f"{ds}", (100, g[-1]), xytext=(4, 0), textcoords="offset points",
                 va="center", fontsize=7, color=DSCOL[ds])
axR.set_xlim(-3, 152)
axR.set_xticks([0, 25, 50, 75, 100])
axR.set_ylim(-.8, 8.1)
axR.set_xlabel("LLM budget: mentions sent to the LLM (%)", labelpad=2)
axR.set_ylabel("Accuracy@1 gain over the\nretriever alone (points)", labelpad=2)
axR.text(.985, .955, "(b)", transform=axR.transAxes, ha="right", va="top",
         fontsize=8.5, fontweight="bold")

save(fig, "fig2_budget")
print(f"  panel (a) on {n_common} mentions common to all four retrievers"
      if COMMON else "  panel (a) on each run's own full set")
print(f"  {'retriever':16}{'R@1':>7}{'full LLM':>10}{'best':>8}{'at budget':>11}{'best-full':>11}")
for name, base, full, peak, at in summary:
    print(f"  {name:16}{base:7.1f}{full:10.1f}{peak:8.1f}{at:10.0f}%{peak-full:+11.2f}")
