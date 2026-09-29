#!/usr/bin/env python3
"""Figure 6 (appendix) - every change the LLM makes, decomposed.

The headline "46 % of changes are lateral" reads as a model that swaps one wrong
concept for another half the time. Most of those moves had no correct option in
the list at all: the gold concept was never retrieved, so any choice was wrong.
Splitting the lateral bucket on `gold_in_list` separates genuine confusion --
the only part that measures disambiguation skill -- from moves that were lost
before the LLM saw them.

Single column (\\columnwidth in the appendix).

    python3 figures/final/make_fig5_churn.py
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aclfig import COL, INK, MUTED, load, seed_files, use_acl, style, save

FIX, BREAK, CONF, LOST = "#0072B2", "#B0472A", "#B08000", "#c9ced4"
CATS = [("fixes", FIX), ("breaks", BREAK),
        ("genuine confusion", CONF), ("gold never retrieved", LOST)]


def decompose(rows):
    ch = [r for r in rows if r.get("llm_changed")]
    fx = [r for r in ch if r["p4_correct"] and not r["p3_correct"]]
    bk = [r for r in ch if r["p3_correct"] and not r["p4_correct"]]
    lat = [r for r in ch if not r["p3_correct"] and not r["p4_correct"]]
    conf = [r for r in lat if r.get("gold_in_list")]
    lost = [r for r in lat if not r.get("gold_in_list")]
    n = len(ch)
    return [len(fx), len(bk), len(conf), len(lost)], n


zs = load("preds_qwen3-4b-zeroshot_bc5cdr.jsonl")
seeds = seed_files("bc5cdr")
rows_zs, n_zs = decompose(zs)
per_seed = [decompose(r)[0] for _, r in seeds]
n_ft = [decompose(r)[1] for _, r in seeds]
rows_ft = [sum(c[i] for c in per_seed) / len(per_seed) for i in range(4)]
n_ft_mean = sum(n_ft) / len(n_ft)

use_acl()
fig, ax = plt.subplots(figsize=(COL, 1.62), dpi=400, layout="constrained")
fig.get_layout_engine().set(w_pad=.02, h_pad=.02)
style(ax, grid_axis="x")

bars = [(f"fine-tuned\n{n_ft_mean:.0f} changes", rows_ft, n_ft_mean),
        (f"zero-shot\n{n_zs} changes", rows_zs, n_zs)]
for y, (lab, counts, tot) in enumerate(bars):
    left = 0.0
    for v, (cname, col) in zip(counts, CATS):
        share = v / tot * 100
        ax.barh(y, share, left=left, height=.52, color=col, lw=0, zorder=3)
        if share > 7:
            ax.text(left + share / 2, y, f"{share:.0f}%", ha="center", va="center",
                    fontsize=6.4, color="white" if col != LOST else INK, zorder=5)
        left += share

ax.set_yticks(range(len(bars)))
ax.set_yticklabels([b[0] for b in bars], fontsize=7, linespacing=1.15)
ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xlabel("share of all changes to the retriever's top-1 (%)", labelpad=2)
ax.text(1, 1.5, "BC5CDR, rule pipeline", fontsize=6.2, color=MUTED, va="center")

h = [Patch(facecolor=c, label=n) for n, c in CATS]
fig.legend(handles=h, loc="outside lower center", ncol=2, frameon=False,
           handlelength=1.0, handleheight=.7, handletextpad=.45,
           columnspacing=1.4, fontsize=6.4)
save(fig, "fig5_churn")
