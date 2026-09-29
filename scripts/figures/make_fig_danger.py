"""Figure for Results 6.2: the mid-confidence band on BC5CDR (rule pipeline,
zero-shot Qwen3-4B). Report-type version of presentation/figs/danger_3.
(a) intervention rate, (b) intervention precision, (c) net effect of each band
on the whole corpus, (fixes - breaks) / all mentions.
    python3 paper/report/make_fig_danger.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aclfig as A
from aclfig import load, in_band, precision, BLAB

rows = load("preds_qwen3-4b-zeroshot_bc5cdr.jsonl")
N = len(rows)
rate, prec, net, ns, fx, bk = [], [], [], [], [], []
for i in range(len(BLAB)):
    sub = in_band(rows, i)
    ns.append(len(sub))
    rate.append(100 * sum(1 for r in sub if r.get("llm_changed")) / len(sub))
    p, f, b = precision(sub)
    prec.append(p or 0); fx.append(f); bk.append(b)
    net.append(100 * (f - b) / N)
    print(f"{BLAB[i]:8s} n={len(sub):5d} rate={rate[-1]:5.1f} prec={prec[-1]:5.1f} "
          f"fixes={f} breaks={b} net={net[-1]:+.2f}")
print(f"total net {sum(net):+.2f}")

GREY, BLUE, RED = "#8a95a0", "#0072B2", A.NEG
worst = int(np.argmin(net))
x = np.arange(len(BLAB))

A.use_acl()
fig, axs = plt.subplots(1, 3, figsize=(6.69, 2.05), dpi=400, layout="constrained")
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.06)

def lab(ax, i, v, txt, col=A.INK, below=False):
    ax.text(i, v, txt, ha="center", va="top" if below else "bottom",
            fontsize=6.6, color=col)

ax = axs[0]; A.style(ax, "y")
ax.bar(x, rate, color=GREY, width=.62)
for i, v in enumerate(rate):
    lab(ax, i, v + 1, f"{v:.0f}%")
ax.set_ylim(0, 65)
ax.set_ylabel("intervention rate (%)", labelpad=2)
ax.set_title("(a) how often the LLM changes the answer", loc="left", fontsize=7.6)

ax = axs[1]; A.style(ax, "y")
ax.bar(x, prec, color=[RED if i == worst else BLUE for i in x], width=.62)
for i, v in enumerate(prec):
    lab(ax, i, v + 1.5, f"{v:.0f}%", RED if i == worst else A.INK)
ax.text(worst, prec[worst] + 13, f"{fx[worst]} fixes\n{bk[worst]} breaks",
        ha="center", va="bottom", fontsize=6.4, color=RED, linespacing=1.1)
ax.set_ylim(0, 112); ax.set_yticks([0, 25, 50, 75, 100])
ax.set_ylabel("intervention precision (%)", labelpad=2)
ax.set_title("(b) how often a change is a fix", loc="left", fontsize=7.6)

ax = axs[2]; A.style(ax, "y")
ax.axhline(0, color="#8f8f8f", lw=.6)
ax.bar(x, net, color=["#1B8A5A" if v >= 0 else RED for v in net], width=.62)
for i, v in enumerate(net):
    if v >= 0:
        lab(ax, i, v + .03, f"{v:+.1f}")
    else:
        lab(ax, i, v - .03, f"{v:+.1f}", RED, below=True)
ax.text(4.4, .85, f"total {sum(net):+.1f}", ha="right", va="center",
        fontsize=6.6, color=A.MUTED)
ax.set_ylim(-1.0, 1.05)
ax.set_ylabel("net effect on corpus (points)", labelpad=2)
ax.set_title("(c) net effect on the whole corpus", loc="left", fontsize=7.6)

for ax in axs:
    ax.set_xticks(x); ax.set_xticklabels(BLAB, fontsize=6.6)
fig.supxlabel("retriever confidence band (score gap between top-1 and top-2 candidate)",
              fontsize=7.4, color=A.INK)

A.FIGDIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(A.FIGDIR / f"fig_danger.{ext}", dpi=400)
print("wrote fig_danger.pdf / .png")
