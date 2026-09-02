#!/usr/bin/env python3
"""Figure 1 - where along the retriever's own confidence the LLM stage pays off.

(a) local net Delta Accuracy@1 (P4-P3) in a sliding window over the
    retriever-confidence percentile, one curve per retriever run, coloured by
    Candidate Recall@1.
(b) the same effect summarised on the 80 % of mentions the retriever is most
    confident about, against retriever strength, with document-level bootstrap
    95 % CIs and an OLS fit.

    python3 figures/final/make_fig1_value.py
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from scipy import stats

from aclfig import (RUNS, WIDE, STRENGTH, SNORM, INK, MUTED, NEG,
                    load, value_profile, confident_delta, recall_at_1,
                    use_acl, style, save)

use_acl()
rows = {lab: load(f) for lab, ds, ret, f, mk in RUNS}
R1 = {lab: recall_at_1(rows[lab]) for lab in rows}
MK = {lab: mk for lab, ds, ret, f, mk in RUNS}
RET = {lab: ret for lab, ds, ret, f, mk in RUNS}
order = sorted(rows, key=R1.get)

fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDE, 2.45), dpi=400, layout="constrained",
                               gridspec_kw={"width_ratios": [1.16, 1]})
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.10, hspace=0)

# ------------------------------------------------ (a) value profile
style(axL)
axL.axhspan(-5, 0, color=NEG, alpha=.06, lw=0, zorder=0)
axL.axhline(0, color="#2a2a2a", lw=.8, zorder=2)
for lab in order:
    x, y = value_profile(rows[lab], window=.25, step=.005)
    axL.plot(x, y, color=STRENGTH(SNORM(R1[lab])), lw=1.35, zorder=3)
axL.set_xlim(10, 90)
axL.set_ylim(-5, 23.5)
axL.set_yticks([-4, 0, 4, 8, 12, 16, 20])
axL.set_xticks([10, 30, 50, 70, 90])
axL.set_xlabel("retriever confidence percentile", labelpad=2)
axL.set_ylabel("net $\\Delta$ Accuracy@1 of the\nLLM stage (points)", labelpad=2)
axL.text(.985, .955, "(a)", transform=axL.transAxes, ha="right", va="top",
         fontsize=8.5, fontweight="bold")
axL.text(88.5, -4.3, "LLM net-harmful", ha="right", va="bottom", fontsize=6.6,
         color=NEG, style="italic")
axL.annotate("weak retriever", xy=(78, 2.9), xytext=(52, 8.6), fontsize=6.8,
             color=STRENGTH(SNORM(56)),
             arrowprops=dict(arrowstyle="-", lw=.6, color=STRENGTH(SNORM(56)),
                             connectionstyle="arc3,rad=.25"))
axL.annotate("strong retriever", xy=(33, -2.7), xytext=(13.5, -4.4), fontsize=6.8,
             color=STRENGTH(SNORM(85)),
             arrowprops=dict(arrowstyle="-", lw=.6, color=STRENGTH(SNORM(85)),
                             connectionstyle="arc3,rad=-.25"))
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=SNORM, cmap=STRENGTH), ax=axL,
                  fraction=.032, pad=.03)
cb.set_label("Candidate Recall@1 (%)", fontsize=6.8, labelpad=2)
cb.ax.tick_params(labelsize=6.5, colors=MUTED, length=2, width=.5)
cb.outline.set_linewidth(.4)

# ------------------------------------------------ (b) confident-region summary
style(axR)
axR.axhspan(-3.2, 0, color=NEG, alpha=.06, lw=0, zorder=0)
axR.axhline(0, color="#2a2a2a", lw=.8, zorder=2)
xs, ys = [], []
for lab in order:
    d, lo, hi = confident_delta(rows[lab], lo=20)
    c = STRENGTH(SNORM(R1[lab]))
    axR.plot([R1[lab]] * 2, [lo, hi], color=c, lw=.9, alpha=.85, zorder=3)
    axR.plot(R1[lab], d, marker=MK[lab], ms=4.6, mfc="white", mec=c, mew=1.25, zorder=4)
    xs.append(R1[lab]); ys.append(d)
xs, ys = np.array(xs), np.array(ys)
sl, ic, r, p, _ = stats.linregress(xs, ys)
rho, prho = stats.spearmanr(xs, ys)
gx = np.linspace(53, 87, 40)
axR.plot(gx, sl * gx + ic, color="#7a7a7a", lw=.9, ls=(0, (4, 2.6)), zorder=1)
axR.set_xlim(52.5, 88.5)
axR.set_ylim(-3.2, 7.4)
axR.set_xlabel("retriever strength: Candidate Recall@1 (%)", labelpad=2)
axR.set_ylabel("net $\\Delta$ Accuracy@1 on the\nconfident 80 % (points)", labelpad=2)
axR.text(.985, .955, "(b)", transform=axR.transAxes, ha="right", va="top",
         fontsize=8.5, fontweight="bold")
axR.text(.03, .955, f"Spearman $\\rho$ = {rho:+.2f}   ($p$ = {prho:.3f})",
         transform=axR.transAxes, ha="left", va="top", fontsize=6.9, color=INK)
h = [Line2D([], [], ls="none", marker=m, ms=4.6, mfc="white", mec=MUTED, mew=1.2, label=t)
     for m, t in [("o", "SapBERT / dense"), ("s", "BM25"),
                  ("D", "rule pipeline"), ("^", "BioSyn")]]
axR.legend(handles=h, loc="lower left", frameon=False, handletextpad=.4,
           labelspacing=.28, borderpad=.1, borderaxespad=.3, ncol=2, columnspacing=.9)

save(fig, "fig1_value")
print(f"  OLS slope {sl:+.3f} pts per Recall@1 point, r={r:+.3f}, p={p:.4f}, "
      f"zero crossing at Recall@1 = {-ic/sl:.1f}")
