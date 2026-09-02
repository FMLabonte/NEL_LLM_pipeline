import numpy as np, matplotlib.pyplot as plt, matplotlib as mpl
from common import *
from matplotlib.lines import Line2D

rows_by = {lab: load(f) for lab, ds, ret, f, mk in RUNS}
R1  = {l: np.mean([r["p3_correct"] for r in rows_by[l]]) * 100 for l in rows_by}
MK  = {lab: mk for lab, ds, ret, f, mk in RUNS}
order = sorted(rows_by, key=lambda l: R1[l])

fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), dpi=200,
                         gridspec_kw={"width_ratios": [1.25, 1]})
cmap = mpl.colormaps["plasma"]
norm = mpl.colors.Normalize(vmin=52, vmax=90)

# ---------------- (a) continuous value profile ----------------
ax = axes[0]; style(ax)
ax.axhspan(-6, 0, color="#D55E00", alpha=.055, zorder=0)
ax.axhline(0, color="#333", lw=1.1, zorder=2)
for l in order:
    xs, ys = value_profile(rows_by[l], w_frac=.28, step_frac=.004)
    ax.plot(xs, ys, color=cmap(norm(R1[l])), lw=2.5, zorder=3, solid_capstyle="round")
ax.set_xlim(0, 100); ax.set_ylim(-6, 23)
ax.set_xlabel("retriever confidence percentile   (least → most confident)", fontsize=10.5)
ax.set_ylabel("local net Δ Accuracy@1 of the LLM stage (points)", fontsize=10.5)
ax.set_title("(a)  the LLM's value profile along the retriever's own confidence",
             fontsize=11.5, fontweight="bold", loc="left", pad=9)
ax.text(52, -5.2, "LLM is net-harmful", fontsize=9.5, color="#A84600", style="italic")
ax.annotate("weak retriever:\npositive everywhere", xy=(70, 2.6), xytext=(58, 12),
            fontsize=9, color=cmap(norm(56)), fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=cmap(norm(56)), lw=.9,
                            connectionstyle="arc3,rad=.25"))
ax.annotate("strong retriever:\nmid-confidence danger zone", xy=(38, -3.2), xytext=(18, -5.4),
            fontsize=9, color=cmap(norm(85)), fontweight="bold", ha="left",
            arrowprops=dict(arrowstyle="-", color=cmap(norm(85)), lw=.9,
                            connectionstyle="arc3,rad=-.25"))
sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
cb = fig.colorbar(sm, ax=ax, fraction=.033, pad=.015)
cb.set_label("retriever strength — Candidate Recall@1 (%)", fontsize=9.5)
cb.ax.tick_params(labelsize=8.5, colors=MUTED)

# ---------------- (b) how deep the danger zone gets ----------------
ax = axes[1]; style(ax)
ax.axhline(0, color="#333", lw=1.1, zorder=2)
ax.axhspan(-6, 0, color="#D55E00", alpha=.055, zorder=0)
xs_, ys_ = [], []
for l in order:
    x, y = value_profile(rows_by[l], w_frac=.28, step_frac=.004)
    m = x > 20                       # ignore the low-confidence spike
    worst = y[m].min()
    c = cmap(norm(R1[l]))
    ax.plot(R1[l], worst, marker=MK[l], ms=10, mfc="white", mec=c, mew=2.2, zorder=4)
    ax.annotate(l.replace(" · ", "\n"), (R1[l], worst), xytext=(0, -13),
                textcoords="offset points", ha="center", va="top",
                fontsize=7.8, color=MUTED, linespacing=1.25)
    xs_.append(R1[l]); ys_.append(worst)
z = np.polyfit(xs_, ys_, 1)
xx = np.linspace(52, 88, 50)
ax.plot(xx, np.polyval(z, xx), color="#888", lw=1.3, ls=(0, (5, 4)), zorder=1)
ax.set_xlim(51, 89); ax.set_ylim(-6, 2.4)
ax.set_xlabel("retriever strength — Candidate Recall@1 (%)", fontsize=10.5)
ax.set_ylabel("deepest net Δ Accuracy@1 above the 20th confidence percentile", fontsize=10.5)
ax.set_title("(b)  the danger zone deepens with the retriever",
             fontsize=11.5, fontweight="bold", loc="left", pad=9)

fig.suptitle("The LLM helps where the retriever is unsure — and starts to hurt in the middle "
             "as soon as the retriever is good", fontsize=13, fontweight="bold", y=1.015)
fig.tight_layout(rect=[0, 0, 1, .985])
fig.savefig(str(Path(__file__).resolve().parent / "V2_profile.png"), dpi=200, bbox_inches="tight")
fig.savefig(str(Path(__file__).resolve().parent / "V2_profile.pdf"), bbox_inches="tight")
print("ok")
