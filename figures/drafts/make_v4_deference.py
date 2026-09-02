import numpy as np, matplotlib.pyplot as plt
from common import *
from matplotlib.lines import Line2D

MARKS = np.array([0, 5, 10, 20, 30, 50, 75, 100.0])
PAIR = [("zero-shot Qwen3-4B", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl", "#0072B2", "o"),
        ("LoRA fine-tuned",    "preds_ft_bc5cdr.jsonl",                "#B0472A", "D")]

fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0), dpi=200)

ax = axes[0]; style(ax)
for lab, f, col, mk in PAIR:
    rows = load(f); frac, acc = budget_curve(rows)
    ax.plot(frac, acc, color=col, lw=2.2, zorder=3, solid_capstyle="round")
    mx, my = budget_curve(rows, MARKS)
    ax.plot(mx, my, ls="none", marker=mk, ms=7, mfc="white", mec=col, mew=1.8, zorder=4)
    ax.annotate(lab, (100, acc[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=col, fontweight="bold")
    ip = int(np.argmax(acc))
    ax.plot(frac[ip], acc[ip], marker="*", ms=15, color=col, mec="white", mew=.9, zorder=6)
ax.axhline(82.75, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
ax.text(2, 82.45, "retriever alone  (R@1 82.7)", fontsize=8.8, color=MUTED, va="top")
ax.set_xlim(-3, 142); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_ylim(82.2, 87.2)
ax.set_xlabel("LLM budget — share of mentions sent to the LLM (%)", fontsize=10.5)
ax.set_ylabel("Accuracy@1 (%)", fontsize=10.5)
ax.set_title("(a)  the gate is only needed before fine-tuning",
             fontsize=11.5, fontweight="bold", loc="left", pad=9)

ax = axes[1]; style(ax)
ax.axhline(0, color="#333", lw=1.1, zorder=2)
ax.axhspan(-4.5, 0, color="#B0472A", alpha=.055, zorder=0)
for lab, f, col, mk in PAIR:
    rows = load(f); xs, ys = value_profile(rows, w_frac=.25, step_frac=.005)
    ax.plot(xs, ys, color=col, lw=2.4, zorder=3, solid_capstyle="round")
    ax.annotate(lab, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                va="center", fontsize=10, color=col, fontweight="bold")
ax.set_xlim(0, 138); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_ylim(-4.5, 11)
ax.set_xlabel("retriever confidence percentile", fontsize=10.5)
ax.set_ylabel("local net Δ Accuracy@1 (points)", fontsize=10.5)
ax.set_title("(b)  fine-tuning fills in the danger zone, not the hard cases",
             fontsize=11.5, fontweight="bold", loc="left", pad=9)

fig.suptitle("BC5CDR, identical retriever and identical mentions — fine-tuning internalises the gate",
             fontsize=12.6, fontweight="bold", y=1.015)
fig.tight_layout(rect=[0, 0, 1, .985])
fig.savefig(str(Path(__file__).resolve().parent / "V4_finetune.png"), dpi=200, bbox_inches="tight")
fig.savefig(str(Path(__file__).resolve().parent / "V4_finetune.pdf"), bbox_inches="tight")
print("ok")
