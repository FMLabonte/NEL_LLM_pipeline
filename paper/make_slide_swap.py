#!/usr/bin/env python3
"""Slide figure: what the graded degradation actually does to a candidate list.

One real BioSyn candidate list from BC5CDR (pmid 8701013, mention "Famotidine"),
shown before and after a rank-1 <-> rank-4 concept swap. The point the picture
has to make is the one that is easy to get backwards in words: the CONCEPTS move,
the SCORES do not. Hence the score column is identical on both sides, the list is
still sorted, the top1-top2 gap is unchanged, and the gold concept is still in the
list -- only which concept sits on top has changed.

Sans-serif and large type: this is for a projector, not for the paper.

    python3 make_slide_swap.py          -> slides_fig_swap.{pdf,png}
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# The project's CVD-validated palette, reused so the slide matches the paper.
GOLD = "#009E73"     # the gold concept
WRONG = "#D55E00"    # the concept that is now (wrongly) on top
INK = "#1b1815"
MUTED = "#7a7167"
RULE = "#d8d2c9"

CANDS = [("D015738", "famotidine", 219.58),
         ("C023990", "tiotidine", 178.06),
         ("C081969", "faman", 171.26),
         ("D011899", "ranitidine", 169.75),
         ("C041784", "ramixotidine", 169.63),
         ("C520153", "cimicifugadine", 169.28),
         ("C043465", "icotidine", 166.91),
         ("D0000775", "famciclovir", 165.46),
         ("C113351", "alpha fampaamidine", 165.06),
         ("C116092", "cimicifugic acid d", 163.19)]
SWAP_TO = 3          # 0-based: rank 4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ROW_H = 0.050
TOP = 0.715


def panel(fig, x0, title, order, note, note_color):
    """One candidate list. `order` maps display rank -> index into CANDS;
    the score column always comes from the display rank, never from the concept,
    which is the whole point of the figure."""
    fig.text(x0, 0.845, title, ha="left", va="baseline", fontsize=15,
             fontweight="bold", color=INK)
    fig.text(x0, 0.805, note, ha="left", va="baseline", fontsize=11.5,
             color=note_color, fontweight="bold")
    fig.text(x0 + 0.012, TOP + 0.040, "rank", fontsize=9.5, color=MUTED, ha="center")
    fig.text(x0 + 0.048, TOP + 0.040, "concept", fontsize=9.5, color=MUTED, ha="left")
    fig.text(x0 + 0.345, TOP + 0.040, "score", fontsize=9.5, color=MUTED, ha="right")
    ys = []
    for r, idx in enumerate(order):
        cui, name, _ = CANDS[idx]
        score = CANDS[r][2]                      # score belongs to the ROW
        y = TOP - r * ROW_H
        ys.append(y)
        gold = (idx == 0)
        moved = idx in (0, SWAP_TO)
        col = GOLD if gold else (WRONG if idx == SWAP_TO else INK)
        weight = "bold" if moved else "normal"
        if moved:
            fig.patches.append(FancyBboxPatch(
                (x0 - 0.004, y - 0.017), 0.358, 0.036,
                boxstyle="round,pad=0.004,rounding_size=0.006",
                linewidth=0, facecolor=col, alpha=0.11,
                transform=fig.transFigure, zorder=0))
        fig.text(x0 + 0.012, y, f"{r+1}", fontsize=12, color=MUTED, ha="center", va="center")
        fig.text(x0 + 0.048, y, f"{cui}", fontsize=11.5, color=col, va="center",
                 fontweight=weight, family="monospace")
        fig.text(x0 + 0.128, y, name, fontsize=12, color=col, va="center",
                 fontweight=weight)
        fig.text(x0 + 0.345, y, f"{score:.2f}", fontsize=12, color=INK, va="center",
                 ha="right", fontweight="bold" if r < 2 else "normal")
    return ys


def main():
    fig = plt.figure(figsize=(13.33, 7.0))
    fig.patch.set_facecolor("white")

    fig.text(0.5, 0.945, 'One real candidate list: mention "Famotidine" (BC5CDR, pmid 8701013)',
             ha="center", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.5, 0.905, "The concepts swap places. The scores do not.",
             ha="center", fontsize=13.5, color=MUTED)

    left, right = 0.045, 0.575
    order_a = list(range(10))
    order_b = list(range(10))
    order_b[0], order_b[SWAP_TO] = order_b[SWAP_TO], order_b[0]

    ys = panel(fig, left, "BEFORE  (p = 0)", order_a,
               "retriever is right:  gold on rank 1", GOLD)
    panel(fig, right, f"AFTER  (this mention degraded)", order_b,
          "retriever is now wrong:  gold pushed to rank 4", WRONG)

    # swap arrows
    for a, b, col in ((0, SWAP_TO, WRONG), (SWAP_TO, 0, GOLD)):
        fig.patches.append(FancyArrowPatch(
            (left + 0.395, ys[a]), (right - 0.022, ys[b]),
            transform=fig.transFigure, arrowstyle="-|>", mutation_scale=17,
            linewidth=1.8, color=col, alpha=0.85,
            connectionstyle=f"arc3,rad={0.22 if a == 0 else -0.22}", zorder=3))

    # what this buys
    box_y = 0.042
    fig.patches.append(FancyBboxPatch(
        (0.045, box_y), 0.91, 0.115,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=1.1, edgecolor=RULE, facecolor="#faf8f5",
        transform=fig.transFigure, zorder=0))
    items = [("same 10 concepts", "gold still in the list", "R@10 = 90.5 unchanged"),
             ("same score vector", "top1 - top2 = 41.52 either way",
              "the LLM sees the same confidence signal"),
             ("one thing moves", "which concept is on top", "R@1 falls")]
    for i, (a, b, c) in enumerate(items):
        x = 0.075 + i * 0.302
        fig.text(x, box_y + 0.082, a, fontsize=12.5, fontweight="bold", color=INK)
        fig.text(x, box_y + 0.049, b, fontsize=11.5, color=MUTED)
        fig.text(x, box_y + 0.018, c, fontsize=11.5, color=INK)

    fig.text(0.5, 0.208,
             "p is the fraction of mentions this is done to:  p = 0 leaves everything "
             "as it is,  p = 0.5 degrades half of them.",
             ha="center", fontsize=12.5, color=INK)

    out = Path(__file__).resolve().parent
    for ext in ("pdf", "png"):
        fig.savefig(out / f"slides_fig_swap.{ext}", dpi=200,
                    facecolor="white")
    print("wrote slides_fig_swap.pdf / .png")


if __name__ == "__main__":
    main()
