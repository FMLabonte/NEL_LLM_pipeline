"""Figure for Results 6.1: gain of the LLM stage against retriever strength,
all fourteen retriever configurations of Table 2 (tab:runs).
Same numbers as the slide figure presentation/make_slide_figs.py (frame_scatter_2),
set in report type. Colour = corpus, marker = retriever family,
lines connect the configurations of one corpus.
    python3 paper/report/make_fig_frame.py
"""
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aclfig as A

# corpus, retriever, R@1, Delta = P4 - R@1  (Table tab:runs)
RUNS = [("BC5CDR", "BM25", 73.4, 2.0), ("BC5CDR", "SapBERT", 75.9, 4.6),
        ("BC5CDR", "rule pipeline", 82.7, 0.9), ("BC5CDR", "BioSyn", 84.8, 0.6),
        ("BioRED", "SapBERT", 70.8, 7.1), ("BioRED", "lexical", 74.2, 7.9),
        ("BioRED", "rule pipeline", 79.5, 4.0),
        ("NCBI", "SapBERT", 60.7, 3.5), ("NCBI", "rule pipeline", 70.1, 2.4),
        ("NCBI", "lexical", 71.3, 1.5),
        ("NLM-Chem", "lexical", 65.9, 10.1), ("NLM-Chem", "dense", 72.3, 5.7),
        ("MedMentions", "lexical", 49.6, 8.9), ("MedMentions", "SapBERT", 55.3, 4.3)]
MARK = {"SapBERT": "o", "dense": "o", "BM25": "s", "lexical": "s",
        "rule pipeline": "D", "BioSyn": "^"}
NAME = {"NCBI": "NCBI-Disease"}
ORDER = ["BC5CDR", "BioRED", "NCBI", "NLM-Chem", "MedMentions"]

A.use_acl()
fig, ax = plt.subplots(figsize=(4.68, 2.75), dpi=400, layout="constrained")
fig.get_layout_engine().set(w_pad=.02, h_pad=.02)
A.style(ax)

for corpus in ORDER:
    pts = sorted((x, y, r) for c, r, x, y in RUNS if c == corpus)
    col = A.DSCOL[corpus]
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.0,
            alpha=.55, zorder=2)
    for x, y, r in pts:
        ax.plot(x, y, marker=MARK[r], ms=5.2, mfc=col, mec="white", mew=.7,
                ls="none", zorder=3)

for lab, x, y, dx, dy, ha in [("SapBERT +4.6", 75.9, 4.6, -5, 4, "right"),
                              ("BioSyn +0.6", 84.8, 0.6, 0, 11, "center")]:
    ax.annotate(lab, (x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va="bottom", fontsize=7, color=A.DSCOL["BC5CDR"], zorder=5)

ax.set_xlim(46, 88)
ax.set_ylim(0, 11)
ax.set_xticks([50, 60, 70, 80])
ax.set_yticks([0, 2, 4, 6, 8, 10])
ax.set_xlabel("retriever strength: Recall@1 (%)", labelpad=2)
ax.set_ylabel(r"gain of the LLM stage, $\Delta$ (points)", labelpad=2)

h1 = [Line2D([], [], color=A.DSCOL[c], lw=1.0, marker="o", ms=4.2, mec="white",
             mew=.6, label=NAME.get(c, c)) for c in ORDER]
h2 = [Line2D([], [], ls="none", marker=m, ms=4.2, mfc=A.MUTED, mec="white",
             mew=.6, label=t) for m, t in [("o", "SapBERT / dense"),
                                           ("s", "BM25 / lexical"),
                                           ("D", "rule pipeline"),
                                           ("^", "BioSyn")]]
leg1 = ax.legend(handles=h1, loc="upper right", frameon=False, handlelength=1.6,
                 handletextpad=.4, labelspacing=.25, borderaxespad=.2)
ax.add_artist(leg1)
ax.legend(handles=h2, loc="lower left", frameon=False,
          handletextpad=.3, labelspacing=.25, borderaxespad=.2)

A.FIGDIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(A.FIGDIR / f"fig_frame.{ext}", dpi=400)
print("wrote fig_frame.pdf / .png")
