"""Figure for the Data Set section: (a) entity-type composition of the evaluated
mentions, (b) two properties that matter for the analysis later."""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aclfig as A

A.use_acl()
df = pd.read_csv(A.FIGDIR / "corpus_stats.csv").set_index("corpus")
types = json.load(open(A.FIGDIR / "corpus_types.json"))
order = ["BC5CDR", "BioRED", "NCBI", "NLM-Chem", "MedMentions"]

# harmonise labels
CHEM = {"Chemical", "ChemicalEntity", "T103"}
DIS = {"Disease", "DiseaseOrPhenotypicFeature", "SpecificDisease", "DiseaseClass",
       "Modifier", "CompositeMention", "T047"}
UMLS = {"T038": "Biologic function", "T058": "Health-care activity",
        "T017": "Anatomical structure", "T033": "Finding"}
cats = ["Chemical", "Disease", "Biologic function", "Health-care activity",
        "Anatomical structure", "Finding", "Other"]
colors = ["#0072B2", "#D55E00", "#009E73", "#7B4FA0", "#B08000", "#56B4E9", "#bdbdbd"]

comp = {}
for c in order:
    d = {k: 0 for k in cats}
    for t, n in types[c].items():
        if t in CHEM: d["Chemical"] += n
        elif t in DIS: d["Disease"] += n
        elif t in UMLS: d[UMLS[t]] += n
        else: d["Other"] += n
    tot = sum(d.values()); comp[c] = {k: 100 * v / tot for k, v in d.items()}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(A.WIDE, 2.05), constrained_layout=True,
                               gridspec_kw={"width_ratios": [1.25, 1]})
y = np.arange(len(order))[::-1]
left = np.zeros(len(order))
for k, col in zip(cats, colors):
    vals = np.array([comp[c][k] for c in order])
    ax1.barh(y, vals, left=left, color=col, height=0.62, label=k, lw=0)
    left += vals
ax1.set_yticks(y, order); ax1.set_xlim(0, 100); ax1.set_xlabel("share of evaluated mentions (%)")
ax1.legend(ncol=4, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.32),
           handlelength=0.9, columnspacing=0.9, fontsize=6.4)
A.style(ax1, grid_axis="x"); ax1.set_title("(a) entity types", loc="left")

w = 0.36
x = np.arange(len(order))
ax2.bar(x - w/2, df.loc[order, "dup_share"], w, color=A.INK, label="repeated mention–concept pair")
ax2.bar(x + w/2, df.loc[order, "short_share"], w, color="#9a9a9a", label="mention $\\leq$ 5 characters")
ax2.set_xticks(x, ["BC5CDR", "BioRED", "NCBI", "NLM-\nChem", "Med-\nMentions"], fontsize=7)
ax2.set_ylabel("share of mentions (%)"); ax2.set_ylim(0, 118); ax2.set_yticks([0,25,50,75,100])
ax2.legend(frameon=False, loc="upper left", fontsize=6.4, handlelength=0.9, ncol=1, borderaxespad=0.1)
A.style(ax2, grid_axis="y"); ax2.set_title("(b) repetition and short forms", loc="left")

fig.savefig(A.FIGDIR / "fig_data.pdf"); fig.savefig(A.FIGDIR / "fig_data.png", dpi=200)
print("ok")
