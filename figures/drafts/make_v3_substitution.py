import numpy as np, matplotlib.pyplot as plt
from common import *
from matplotlib.lines import Line2D

ALL = [
    ("MedMentions", "lexical",       "preds_zs_medmentions.jsonl",          "v"),
    ("MedMentions", "SapBERT",       "preds_zs_medmentions_emb.jsonl",      "o"),
    ("NCBI",        "lexical",       "preds_zs_ncbi_lex.jsonl",             "v"),
    ("NCBI",        "SapBERT",       "preds_zs_ncbi_sapbert.jsonl",         "o"),
    ("BioRED",      "lexical",       "preds_zs_biored_lex.jsonl",           "v"),
    ("BioRED",      "SapBERT",       "preds_zs_biored_sapbert.jsonl",       "o"),
    ("NLM-Chem",    "lexical",       "preds_zs_nlmchem_lex.jsonl",          "v"),
    ("NLM-Chem",    "dense",         "preds_zs_nlmchem.jsonl",              "o"),
    ("BC5CDR",      "BM25",          "preds_zs_bc5cdr_bm25.jsonl",          "s"),
    ("BC5CDR",      "SapBERT",       "preds_zs_bc5cdr_sapbert.jsonl",       "o"),
    ("BC5CDR",      "rule pipeline", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl","D"),
    ("BC5CDR",      "BioSyn",        "preds_zs_bc5cdr_biosyn.jsonl",        "^"),
]
P = {}
for ds, ret, f, mk in ALL:
    rows = load(f)
    P[(ds, ret)] = (np.mean([r["p3_correct"] for r in rows]) * 100,
                    np.mean([r["p4_correct"] for r in rows]) * 100,
                    np.mean([bool(r.get("gold_in_list")) for r in rows]) * 100, mk)

fig, ax = plt.subplots(figsize=(8.4, 6.0), dpi=200); style(ax)
lo, hi = 46, 96
ax.plot([lo, hi], [lo, hi], color="#9a9a9a", lw=1.2, ls=(0, (5, 4)), zorder=1)
ax.text(93.5, 92.6, "retriever alone\n(no LLM)", color="#8a8a8a", fontsize=8.6,
        ha="right", va="top", rotation=34, rotation_mode="anchor", linespacing=1.3)

for ds in ["MedMentions", "NCBI", "BioRED", "NLM-Chem", "BC5CDR"]:
    pts = sorted([(k[1], v) for k, v in P.items() if k[0] == ds], key=lambda t: t[1][0])
    ax.plot([v[0] for _, v in pts], [v[1] for _, v in pts],
            color=DS[ds], lw=1.1, alpha=.5, zorder=2)
    for ret, (p3, p4, c, mk) in pts:
        ax.plot([p3, p3], [p3, c],  color=DS[ds], lw=7, alpha=.15, zorder=2, solid_capstyle="butt")
        ax.plot([p3, p3], [p3, p4], color=DS[ds], lw=7, alpha=.70, zorder=3, solid_capstyle="butt")
        ax.plot(p3, c,  marker="_", ms=10, color=DS[ds], mew=2.0, zorder=4)
        ax.plot(p3, p4, marker=mk,  ms=8.5, mfc="white", mec=DS[ds], mew=1.9, zorder=5)
    p3, p4, c, mk = pts[-1][1]
    ax.annotate(f" {ds}", (p3, p4), xytext=(9, -1), textcoords="offset points",
                fontsize=9.6, color=DS[ds], fontweight="bold", va="center")

ax.set_xlim(lo, hi); ax.set_ylim(lo, 96)
ax.set_xlabel("retriever strength  —  Candidate Recall@1 (%)", fontsize=11)
ax.set_ylabel("Accuracy@1 after the LLM stage (%)", fontsize=11)
ax.set_title("Retriever and LLM are partial substitutes\n"
             "the LLM's slice shrinks faster than the headroom it could still recover",
             fontsize=12.4, fontweight="bold", loc="left", pad=10)
h = [Line2D([], [], color="#666", lw=7, alpha=.70, label="gain contributed by the LLM stage"),
     Line2D([], [], color="#666", lw=7, alpha=.15, label="headroom left (gold in top-10, not picked)"),
     Line2D([], [], color="#666", marker="_", ls="none", ms=10, mew=2, label="ceiling = Candidate Recall@10"),
     Line2D([], [], color="#9a9a9a", lw=1.1, alpha=.6, label="same dataset, different retriever")]
ax.legend(handles=h, loc="upper left", frameon=False, fontsize=8.9, borderpad=.2)
fig.tight_layout()
fig.savefig(str(Path(__file__).resolve().parent / "V3_substitution.png"), dpi=200, bbox_inches="tight")
fig.savefig(str(Path(__file__).resolve().parent / "V3_substitution.pdf"), bbox_inches="tight")
print("ok")
