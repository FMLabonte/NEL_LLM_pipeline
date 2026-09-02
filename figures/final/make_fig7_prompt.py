#!/usr/bin/env python3
"""Figure 7 - the danger zone survives every prompt intervention we tried.

v8 already instructs the model to trust the ranking AND shows it every
candidate's match score. These three ablations remove the instruction, remove the
scores, and state the top1-top2 margin explicitly as a number together with a
margin-conditioned rule. Everything else -- retriever, candidate lists, decoder,
mention set -- is identical.

    python3 figures/final/make_fig7_prompt.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aclfig import (WIDE, BLAB, band_of, INK, MUTED, NEG, load, acc, precision,
                    use_acl, style, save)

VARIANTS = [("v8", "#08375a"), ("v8-notrust", "#2f88bf"),
            ("v8-noscore", "#B0472A"), ("v8-gap", "#B08000")]
PANELS = [("BC5CDR", {"v8": "preds_qwen3-4b-zeroshot_bc5cdr.jsonl",
                      "v8-notrust": "preds_zs_bc5cdr_v8-notrust.jsonl",
                      "v8-noscore": "preds_zs_bc5cdr_v8-noscore.jsonl",
                      "v8-gap": "preds_zs_bc5cdr_v8-gap.jsonl"}),
          ("BioRED", {"v8": "preds_zs_biored.jsonl",
                      "v8-notrust": "preds_zs_biored_v8-notrust.jsonl",
                      "v8-noscore": "preds_zs_biored_v8-noscore.jsonl",
                      "v8-gap": "preds_zs_biored_v8-gap.jsonl"})]

use_acl()
fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.08), dpi=400, layout="constrained")
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.06)

w = .2
for ax, (name, files) in zip(axes, PANELS):
    style(ax, grid_axis="y")
    ax.axhspan(-7.5, 0, color=NEG, alpha=.055, lw=0, zorder=0)
    ax.axhline(0, color="#2a2a2a", lw=.8, zorder=2)
    rates = {}
    for k, (vlab, col) in enumerate(VARIANTS):
        rows = load(files[vlab])
        rates[vlab] = acc(rows, "llm_changed")
        xs, ys = [], []
        for i, bl in enumerate(BLAB):
            sub = [r for r in rows if band_of(r["p3_score_gap"]) == i]
            if not sub:
                continue
            xs.append(i + (k - 1.5) * w)
            ys.append(acc(sub, "p4_correct") - acc(sub, "p3_correct"))
        ax.bar(xs, ys, width=w * .88, color=col, lw=0, zorder=3, label=vlab)
    ax.set_xticks(range(len(BLAB)))
    ax.set_xticklabels(BLAB, fontsize=6.6)
    ax.set_ylim(-7.5, 31)
    ax.set_title(name, fontsize=8.5, pad=3)
    ax.set_xlabel("retriever confidence band (top1$-$top2 score gap)", labelpad=2)
    rate_txt = "  ".join(f"{v}: {rates[v]:.1f}%" for v, _ in VARIANTS)
    ax.text(.5, .965, "intervention rate — " + rate_txt, transform=ax.transAxes,
            ha="center", va="top", fontsize=5.8, color=MUTED)

axes[0].set_ylabel("net $\\Delta$ Accuracy@1 (points)", labelpad=2)
axes[0].annotate("the danger zone stays\nnegative under every variant",
                 xy=(2.95, -2.6), xytext=(2.05, 15.5), fontsize=6.4, color=NEG,
                 ha="center", va="center", linespacing=1.25,
                 arrowprops=dict(arrowstyle="-", lw=.6, color=NEG,
                                 connectionstyle="arc3,rad=-.2"))

h = [Patch(facecolor=c, label=v) for v, c in VARIANTS]
fig.legend(handles=h, loc="outside upper center", ncol=4, frameon=False,
           handlelength=1.0, handleheight=.7, handletextpad=.45,
           columnspacing=1.8, fontsize=7)
save(fig, "fig7_prompt")
