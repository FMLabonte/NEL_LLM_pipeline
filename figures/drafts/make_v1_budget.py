import numpy as np, matplotlib.pyplot as plt
from common import *
from matplotlib.lines import Line2D

MARKS = np.array([0, 5, 10, 20, 30, 50, 75, 100.0])
BC5 = [r for r in RUNS if r[1] == "BC5CDR"]
MAIN = [r for r in RUNS if r[1] != "BC5CDR"] + [r for r in RUNS if r[0] == "BC5CDR · SapBERT"]

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.3), dpi=200)

# ================= (a) BC5CDR: same mentions, four retrievers =================
ax = axes[0]; style(ax)
lbl_y = {}
for lab, ds, ret, f, mk in BC5:
    rows = load(f)
    frac, acc = budget_curve(rows)
    base, full = acc[0], acc[-1]
    col = DS[ds]
    ax.plot(frac, acc, color=col, lw=2.0, zorder=3, solid_capstyle="round")
    mx, my = budget_curve(rows, MARKS)
    ax.plot(mx, my, ls="none", marker=mk, ms=7, mfc="white", mec=col, mew=1.8, zorder=4)
    T, calls, hacc = dev_tuned_point(rows)
    ystar = np.interp(calls, frac, acc)
    ax.plot(calls, ystar, marker="*", ms=16, color=col, mec="white", mew=.9, zorder=6)
    ax.annotate(f"{ret}", (100, full), xytext=(7, 0), textcoords="offset points",
                va="center", fontsize=10, color=col, fontweight="bold")
    ax.annotate(f"R@1 {base:.1f}", (0, base), xytext=(-7, 0), textcoords="offset points",
                va="center", ha="right", fontsize=8.6, color=MUTED)
ax.set_xlim(-16, 128); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_ylim(72.6, 87.6)
ax.set_xlabel("LLM budget  —  share of mentions sent to the LLM (%)", fontsize=10.5)
ax.set_ylabel("Accuracy@1 (%)", fontsize=10.5)
ax.set_title("(a)  BC5CDR: the same 9,661 mentions, four retrievers",
             fontsize=11.5, fontweight="bold", loc="left", pad=9)
ax.annotate("gate at ~15 % of calls\nbeats the full LLM by +1.4",
            xy=(15, 86.8), xytext=(38, 87.1), fontsize=9, color="#0072B2",
            ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#0072B2", lw=.9,
                            connectionstyle="arc3,rad=-.2"))

# ================= (b) one run per dataset =================
ax = axes[1]; style(ax)
ax.axhline(0, color="#444", lw=1.0, ls=(0, (4, 3)), zorder=1)
for lab, ds, ret, f, mk in MAIN:
    rows = load(f)
    frac, acc = budget_curve(rows)
    g = acc - acc[0]; col = DS[ds]
    ax.plot(frac, g, color=col, lw=2.0, zorder=3, solid_capstyle="round")
    mx, my = budget_curve(rows, MARKS)
    ax.plot(mx, my - acc[0], ls="none", marker="o", ms=6.5, mfc="white", mec=col, mew=1.7, zorder=4)
    ax.plot(100, g[-1], marker="o", ms=6.5, color=col, zorder=5)
    ax.annotate(f"{ds}   R@1 {acc[0]:.0f}", (100, g[-1]), xytext=(7, 0),
                textcoords="offset points", va="center", fontsize=9.2,
                color=col, fontweight="bold")
ax.set_xlim(-3, 152); ax.set_xticks([0, 20, 40, 60, 80, 100]); ax.set_ylim(-.7, 8)
ax.set_xlabel("LLM budget  —  share of mentions sent to the LLM (%)", fontsize=10.5)
ax.set_ylabel("Accuracy@1 gain over the retriever alone (points)", fontsize=10.5)
ax.set_title("(b)  five datasets, dense retrieval throughout",
             fontsize=11.5, fontweight="bold", loc="left", pad=9)

h = [Line2D([], [], ls="none", marker=m, ms=7.5, mfc="white", mec=MUTED, mew=1.7, label=r)
     for m, r in [("s", "BM25"), ("o", "SapBERT"), ("D", "rule pipeline"), ("^", "BioSyn")]]
h += [Line2D([], [], ls="none", marker="*", ms=14, color=MUTED,
             label="budget tuned on held-out dev documents")]
axes[0].legend(handles=h, loc="lower right", frameon=False, fontsize=8.8,
               handletextpad=.5, borderpad=.2)

fig.suptitle("Mentions are cheap to route: behind a strong retriever, accuracy peaks at a small LLM budget "
             "and then falls",
             fontsize=13, fontweight="bold", x=.5, y=1.015)
fig.tight_layout(rect=[0, 0, 1, .985])
fig.savefig(str(Path(__file__).resolve().parent / "V1_budget.png"), dpi=200, bbox_inches="tight")
fig.savefig(str(Path(__file__).resolve().parent / "V1_budget.pdf"), bbox_inches="tight")
print("ok")
