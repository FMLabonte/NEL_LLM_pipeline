#!/usr/bin/env python3
"""Figure 6 - graded retriever degradation on a fixed candidate pool.

The frame in Section 4 is observational: swapping retrievers changes the
candidate pool, the score scale and top-1 accuracy at once. Here only one thing
changes. Starting from BioSyn's candidate lists for BC5CDR, the concept at rank 1
is swapped with a uniformly drawn rank in [2,10] for a fraction p of mentions;
the score vector stays in place, so the list is still sorted, the top1-top2 gap
distribution is unchanged and Recall@10 is 90.5 at every level by construction.
Only which concept sits on top changes.

    python3 figures/final/make_fig6_dose.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

from aclfig import (WIDE, DSCOL, INK, MUTED, NEG, load, acc, recall_at_1,
                    confident_delta, use_acl, style, save)

RUNS = [(0.0, "preds_zs_bc5cdr_biosyn.jsonl"), (0.1, "preds_dose_p0.1.jsonl"),
        (0.2, "preds_dose_p0.2.jsonl"), (0.3, "preds_dose_p0.3.jsonl"),
        (0.4, "preds_dose_p0.4.jsonl"), (0.5, "preds_dose_p0.5.jsonl")]

use_acl()
BLUE, DARK = DSCOL["BC5CDR"], "#08375a"

rows = {p: load(f) for p, f in RUNS}
r1 = np.array([recall_at_1(rows[p]) for p, _ in RUNS])
p4 = np.array([acc(rows[p], "p4_correct") for p, _ in RUNS])
cd = np.array([confident_delta(rows[p], lo=20) for p, _ in RUNS])   # value, lo, hi
ps = np.array([p for p, _ in RUNS])

fig, (axL, axR) = plt.subplots(1, 2, figsize=(WIDE, 2.12), dpi=400, layout="constrained",
                               gridspec_kw={"width_ratios": [1, 1]})
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.07)

# ---------------------------------------------- (a) the stage absorbs the damage
style(axL)
lo, hi = 38, 92
axL.plot([lo, hi], [lo, hi], color="#9a9a9a", lw=1.0, ls=(0, (5, 4)), zorder=1)
axL.annotate("retriever alone", (hi - 1, hi - 1), xytext=(-2, -4),
             textcoords="offset points", ha="right", va="top", fontsize=6.3,
             color="#8a8a8a", rotation=38, rotation_mode="anchor")
axL.fill_between(r1, r1, p4, color=BLUE, alpha=.13, lw=0, zorder=2)
axL.plot(r1, p4, color=DARK, lw=1.5, marker="o", ms=4.4, mfc="white", mew=1.2, zorder=4)
for x, y, p in zip(r1, p4, ps):
    axL.annotate(f"{p:g}", (x, y), xytext=(0, 6), textcoords="offset points",
                 ha="center", fontsize=6.0, color=MUTED)
axL.annotate("after the LLM stage", (r1[3], p4[3]), xytext=(-4, -16),
             textcoords="offset points", ha="left", fontsize=6.6, color=DARK)
axL.set_xlim(lo, hi); axL.set_ylim(lo, hi)
axL.set_xticks([40, 50, 60, 70, 80, 90]); axL.set_yticks([40, 50, 60, 70, 80, 90])
axL.set_xlabel("Candidate \\rone{} after degradation (%)".replace("\\rone{}", "Recall@1"), labelpad=2)
axL.set_ylabel("Accuracy@1 (%)", labelpad=2)
axL.text(.012, .955, "(a)", transform=axL.transAxes, ha="left", va="top",
         fontsize=8.5, fontweight="bold")
axL.text(.985, .06, f"$-{r1[0]-r1[-1]:.0f}$ points of \\rone{{}}, $-{p4[0]-p4[-1]:.1f}$ of accuracy"
         .replace("\\rone{}", "Recall@1"),
         transform=axL.transAxes, ha="right", va="bottom", fontsize=6.6, color=INK)

# ---------------------------------------------- (b) confident region crosses zero
style(axR)
axR.axhspan(-6, 0, color=NEG, alpha=.06, lw=0, zorder=0)
axR.axhline(0, color="#2a2a2a", lw=.8, zorder=2)
for x, (v, l, h) in zip(r1, cd):
    axR.plot([x, x], [l, h], color=BLUE, lw=.9, alpha=.7, zorder=3)
axR.plot(r1, cd[:, 0], color=DARK, lw=1.5, marker="o", ms=4.4, mfc="white", mew=1.2, zorder=4)
sl, ic, rr, pp, _ = stats.linregress(r1, cd[:, 0])
gx = np.linspace(38, 92, 40)
axR.plot(gx, sl * gx + ic, color="#8a8a8a", lw=.9, ls=(0, (4, 2.6)), zorder=1)
cross = -ic / sl
axR.plot([cross], [0], marker="v", ms=5, color=NEG, zorder=5, clip_on=False)
axR.annotate(f"sign flips at \\rone{{}} {cross:.0f}".replace("\\rone{}", "Recall@1"),
             (cross, 0), xytext=(-6, 9), textcoords="offset points", ha="right",
             fontsize=6.5, color=NEG)
axR.set_xlim(38, 92); axR.set_ylim(-6, 46)
axR.set_xticks([40, 50, 60, 70, 80, 90])
axR.set_xlabel("Candidate Recall@1 after degradation (%)", labelpad=2)
axR.set_ylabel("net $\\Delta$ Accuracy@1 on the\nconfident 80 % (points)", labelpad=2)
axR.text(.012, .955, "(b)", transform=axR.transAxes, ha="left", va="top",
         fontsize=8.5, fontweight="bold")
axR.text(.985, .955, f"slope ${sl:+.2f}$,  $r={rr:+.4f}$", transform=axR.transAxes,
         ha="right", va="top", fontsize=6.6, color=INK)

save(fig, "fig6_dose")
s1, i1, r1_, p1, _ = stats.linregress(r1, p4 - r1)
print(f"  gain vs R@1: slope {s1:+.3f}, r={r1_:+.4f}, p={p1:.3g}")
print(f"  confD vs R@1: slope {sl:+.3f}, r={rr:+.4f}, p={pp:.3g}, zero at R@1={cross:.1f}")
print(f"  R@1 {r1[0]:.1f} -> {r1[-1]:.1f} ({r1[0]-r1[-1]:.1f} points), "
      f"P4 {p4[0]:.1f} -> {p4[-1]:.1f} ({p4[0]-p4[-1]:.1f} points) "
      f"=> the stage absorbs {100*(1-(p4[0]-p4[-1])/(r1[0]-r1[-1])):.0f}%")
