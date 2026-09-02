#!/usr/bin/env python3
"""Figure 5 - what LoRA fine-tuning changes: deference, not competence.

Both panels are means over the THREE fine-tuning seeds, with the observed range
as the error bar. A single run overstated two of these numbers (it put the BioRED
band repair at 69 %, which no seed reproduces), so the seeds are not decoration
here -- they are the reason the panel is drawn this way.

    (a) Intervention precision per confidence band, pooled over the four MeSH
        corpora. The truly-uncertain band is the competence test: the answer has
        to come from context, and restraint cannot help. Nothing happens there.
    (b) Intervention precision on the three corpora the model never trained on,
        split by whether the gold concept was in the fine-tuning data.

    python3 figures/final/make_fig4_deference.py
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from aclfig import (WIDE, BLAB, INK, MUTED, in_band, load, precision, seed_files,
                    mean_range, train_concepts, is_seen, use_acl, style, save)

ZS, FT = "#7c8894", "#0072B2"          # zero-shot vs fine-tuned, used in both panels

# (label, zero-shot dump, tag of the seed dumps)
CORPORA = [("BC5CDR", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl", "bc5cdr"),
           ("BioRED", "preds_zs_biored.jsonl", "biored"),
           ("NCBI", "preds_zs_ncbi.jsonl", "ncbi"),
           ("NLM-Chem", "preds_zs_nlmchem.jsonl", "nlmchem")]
HELD_OUT = CORPORA[1:]                 # BC5CDR is in-domain and excluded from (b)

zs_runs = {tag: load(f) for _, f, tag in CORPORA}
ft_runs = {tag: seed_files(tag) for _, _, tag in CORPORA}
n_seeds = min(len(v) for v in ft_runs.values())

use_acl()
fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.15), dpi=400, layout="constrained")
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.07)

# ---------------------------------------------------------------- (a) by band
ax = axes[0]
style(ax, grid_axis="y")
w = .36
for i, _ in enumerate(BLAB):
    pooled_zs = [r for tag in zs_runs for r in in_band(zs_runs[tag], i)]
    p, _, _ = precision(pooled_zs)
    ax.bar(i - w / 2, p or 0, width=w * .9, color=ZS, lw=0, zorder=3)
    per_seed = []
    for k in range(n_seeds):
        pooled = [r for tag in ft_runs for r in in_band(ft_runs[tag][k][1], i)]
        per_seed.append(precision(pooled)[0])
    m, lo, hi = mean_range(per_seed)
    ax.bar(i + w / 2, m, width=w * .9, color=FT, lw=0, zorder=3)
    ax.errorbar(i + w / 2, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK,
                elinewidth=.7, capsize=1.8, capthick=.7, zorder=4)

ax.set_xticks(range(len(BLAB)))
ax.set_xticklabels(BLAB, fontsize=6.6)
# headroom above the tallest bars so the annotations never sit on the data
ax.set_ylim(0, 132)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("intervention precision (%)", labelpad=2)
ax.set_xlabel("retriever confidence band (top1$-$top2 score gap)", labelpad=2)
ax.set_title("(a) pooled over the four MeSH corpora", fontsize=8, pad=3)
ax.annotate("no gain where the answer\nmust come from context",
            xy=(0.0, 101), xytext=(1.05, 122), fontsize=6.2, color=MUTED,
            ha="center", va="center", linespacing=1.25,
            arrowprops=dict(arrowstyle="-", lw=.6, color=MUTED,
                            connectionstyle="arc3,rad=.18"))
ax.annotate("and the repair lands here", xy=(3.18, 70), xytext=(3.5, 118),
            fontsize=6.2, color=MUTED, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=.6, color=MUTED,
                            connectionstyle="arc3,rad=-.18"))

# ------------------------------------------------------------ (b) seen/unseen
ax = axes[1]
style(ax, grid_axis="y")
concepts = train_concepts()
groups = [("seen in\nfine-tuning data", True), ("unseen", False)]
w = .3
for j, (lab, grp) in enumerate(groups):
    pool_zs = [r for _, f, tag in HELD_OUT for r in zs_runs[tag]
               if is_seen(r, concepts) == grp]
    p, _, bk = precision(pool_zs)
    ax.bar(j - w / 2, p or 0, width=w * .9, color=ZS, lw=0, zorder=3)
    ax.text(j - w / 2, 3, f"{bk}", ha="center", va="bottom", fontsize=6,
            color="white", zorder=5)
    vals, breaks = [], []
    for k in range(n_seeds):
        pool = [r for _, _, tag in HELD_OUT for r in ft_runs[tag][k][1]
                if is_seen(r, concepts) == grp]
        q, _, b2 = precision(pool)
        vals.append(q); breaks.append(b2)
    m, lo, hi = mean_range(vals)
    ax.bar(j + w / 2, m, width=w * .9, color=FT, lw=0, zorder=3)
    ax.errorbar(j + w / 2, m, yerr=[[m - lo], [hi - m]], fmt="none", ecolor=INK,
                elinewidth=.7, capsize=1.8, capthick=.7, zorder=4)
    ax.text(j + w / 2, 3, f"{min(breaks)}–{max(breaks)}", ha="center", va="bottom",
            fontsize=6, color="white", zorder=5)

ax.set_xticks(range(len(groups)))
ax.set_xticklabels([g[0] for g in groups], fontsize=7, linespacing=1.2)
ax.set_xlim(-.58, 1.58)
ax.set_ylim(0, 132)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_title("(b) three held-out corpora, by concept familiarity", fontsize=8, pad=3)
# "breaks inside the bars" is stated in the LaTeX caption; repeating it inside the
# axes only collides with the numbers it describes.
ax.annotate("+22 points", xy=(.15, 91), xytext=(.15, 120), fontsize=6.4,
            color=INK, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=.6, color=MUTED))
ax.annotate("the reliability does not transfer", xy=(1.15, 71), xytext=(1.15, 120),
            fontsize=6.2, color=MUTED, ha="center", va="center",
            arrowprops=dict(arrowstyle="-", lw=.6, color=MUTED))

h = [Patch(facecolor=ZS, label="zero-shot"),
     Patch(facecolor=FT, label=f"fine-tuned (mean of {n_seeds} seeds, bars = range)")]
fig.legend(handles=h, loc="outside upper center", ncol=2, frameon=False,
           handlelength=1.0, handleheight=.7, handletextpad=.45,
           columnspacing=1.8, fontsize=7)
save(fig, "fig4_deference")
