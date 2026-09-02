#!/usr/bin/env python3
"""
Figure 1 (heatmap): Net Delta Accuracy@1 of the LLM stage per retriever-confidence
quintile, across retriever runs sorted by Candidate Recall@1. Blue = LLM helps,
red = LLM hurts. The point: the mid/high-confidence "danger zone" (red) emerges as
the retriever strengthens (top -> bottom), while the low-confidence column stays
uniformly positive. The rightmost column is the overall gain per run.

Confidence is each retriever's own top1-top2 score gap (P3), binned into 5 EQUAL-COUNT
quintiles PER RUN, so columns are comparable across retrievers despite different score
scales (BM25 vs cosine vs rule scores).

Run:  python3 figures/make_fig1_heatmap.py
Reads preds_*.jsonl from the repo root; writes figures/fig1_curve.{pdf,png}.
"""
import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NQ = 5
CLIP = 6.0  # symmetric colour clip so danger-zone reds stay vivid; big Q1 blues saturate

# (label, preds file) — ordered later by measured Recall@1
RUNS = [
    ("MedMentions · SapBERT",      "preds_zs_medmentions_emb.jsonl"),
    ("NCBI · SapBERT",             "preds_zs_ncbi_sapbert.jsonl"),
    ("BioRED · SapBERT",           "preds_zs_biored_sapbert.jsonl"),
    ("NLM-Chem · dense",           "preds_zs_nlmchem.jsonl"),
    ("BC5CDR · BM25",              "preds_zs_bc5cdr_bm25.jsonl"),
    ("BC5CDR · SapBERT",           "preds_zs_bc5cdr_sapbert.jsonl"),
    ("BC5CDR · full pipeline",     "preds_qwen3-4b-zeroshot_bc5cdr.jsonl"),
    ("BC5CDR · BioSyn",            "preds_zs_bc5cdr_biosyn.jsonl"),
]


def load(path):
    rows = []
    for line in open(ROOT / path):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("p3_score_gap") is None:
            continue
        rows.append((float(r["p3_score_gap"]),
                     1 if r["p3_correct"] else 0,
                     1 if r["p4_correct"] else 0))
    return rows


def build():
    data = []
    for lab, f in RUNS:
        rows = load(f)
        gaps = np.array([r[0] for r in rows]); p3 = np.array([r[1] for r in rows])
        p4 = np.array([r[2] for r in rows])
        R = p3.mean() * 100; gain = (p4.mean() - p3.mean()) * 100
        order = np.argsort(gaps); binsz = len(rows) // NQ
        cells = []
        for q in range(NQ):
            idx = order[q * binsz:(q + 1) * binsz] if q < NQ - 1 else order[q * binsz:]
            cells.append((p4[idx].mean() - p3[idx].mean()) * 100)
        data.append((lab, R, gain, cells))
    data.sort(key=lambda t: t[1])  # ascending Recall@1
    return data


def main():
    data = build()
    labels = [f"{d[0]}   (R@1 {d[1]:.0f})" for d in data]
    M = np.array([d[3] for d in data])
    gains = [d[2] for d in data]

    fig, ax = plt.subplots(figsize=(9.6, 5.9), dpi=150)
    norm = TwoSlopeNorm(vmin=-CLIP, vcenter=0, vmax=CLIP)
    im = ax.imshow(np.clip(M, -CLIP, CLIP), cmap="RdBu", norm=norm, aspect="auto")

    ax.set_xticks(range(NQ))
    ax.set_xticklabels(["Q1\nleast\nconfident", "Q2", "Q3", "Q4", "Q5\nmost\nconfident"])
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_ylabel("retriever strength increases  ↓\n(rows sorted by Candidate Recall@1)",
                  fontsize=10)
    for i in range(len(labels)):
        for j in range(NQ):
            v = M[i, j]
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                    color="white" if abs(v) > 3 else "black", fontsize=9.5, fontweight="bold")

    # right-hand overall-gain column (outside the grid)
    xg = NQ - 0.5 + 0.55
    ax.text(xg, -0.72, "overall\ngain", ha="center", va="bottom", fontsize=9, fontweight="bold")
    for i, g in enumerate(gains):
        ax.text(xg, i, f"{g:+.1f}", ha="center", va="center", fontsize=9.5,
                fontweight="bold", color="#222")
    ax.set_xlim(-0.5, NQ - 0.5 + 1.2)

    ax.set_title("The LLM helps where the retriever is unsure (blue), and a mid-confidence\n"
                 "danger zone (red) emerges as the retriever strengthens",
                 fontsize=12.5, fontweight="bold", pad=26)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.10, extend="both")
    cb.set_label("Net Δ Accuracy@1 of the LLM stage (P4 − P3), points")

    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / "figures" / f"fig1_curve.{ext}", dpi=150, bbox_inches="tight")
    print("wrote figures/fig1_curve.{pdf,png}")
    for l, g in zip(labels, gains):
        print(f"  {l}  gain {g:+.1f}")


if __name__ == "__main__":
    main()
