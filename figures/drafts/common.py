import json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import random

# repo root: figures/drafts/common.py -> ../..
U = Path(__file__).resolve().parent.parent.parent

# validated colourblind-safe categorical palette (dataset identity)
DS = {
    "BC5CDR":      "#0072B2",
    "MedMentions": "#D55E00",
    "BioRED":      "#009E73",
    "NCBI":        "#7B4FA0",
    "NLM-Chem":    "#B08000",
}
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#d9d9d9"

RUNS = [
    # label,           dataset,       retriever,      file,                                marker
    ("MedMentions · SapBERT",  "MedMentions", "SapBERT",       "preds_zs_medmentions_emb.jsonl",     "o"),
    ("NCBI · SapBERT",         "NCBI",        "SapBERT",       "preds_zs_ncbi_sapbert.jsonl",        "o"),
    ("BioRED · SapBERT",       "BioRED",      "SapBERT",       "preds_zs_biored_sapbert.jsonl",      "o"),
    ("NLM-Chem · dense",       "NLM-Chem",    "SapBERT",       "preds_zs_nlmchem.jsonl",             "o"),
    ("BC5CDR · BM25",          "BC5CDR",      "BM25",          "preds_zs_bc5cdr_bm25.jsonl",         "s"),
    ("BC5CDR · SapBERT",       "BC5CDR",      "SapBERT",       "preds_zs_bc5cdr_sapbert.jsonl",      "o"),
    ("BC5CDR · rule pipeline", "BC5CDR",      "rule pipeline", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl","D"),
    ("BC5CDR · BioSyn",        "BC5CDR",      "BioSyn",        "preds_zs_bc5cdr_biosyn.jsonl",       "^"),
]

def load(f):
    rows = [json.loads(l) for l in open(U / f) if l.strip()]
    return [r for r in rows if r.get("p3_score_gap") is not None]

def budget_curve(rows, grid=None):
    """Accuracy@1 as a function of the share of mentions sent to the LLM
    (lowest retriever-confidence first)."""
    g  = np.array([float(r["p3_score_gap"]) for r in rows])
    p3 = np.array([1.0 if r["p3_correct"] else 0.0 for r in rows])
    p4 = np.array([1.0 if r["p4_correct"] else 0.0 for r in rows])
    o  = np.argsort(g, kind="stable")
    a3, a4 = p3[o], p4[o]
    n = len(a3)
    c4 = np.concatenate([[0], np.cumsum(a4)])
    c3 = np.concatenate([[0], np.cumsum(a3)])
    tot3 = c3[-1]
    acc = (c4 + (tot3 - c3)) / n * 100.0     # index k = send k lowest-gap mentions
    frac = np.arange(n + 1) / n * 100.0
    if grid is None:
        return frac, acc
    return grid, np.interp(grid, frac, acc)

def dev_tuned_point(rows, seed=0):
    """Threshold tuned on half the DOCUMENTS, reported on the other half."""
    by = defaultdict(list)
    for r in rows: by[r.get("pmid")].append(r)
    docs = list(by); random.Random(seed).shuffle(docs)
    h = len(docs) // 2
    dev  = [r for d in docs[:h]  for r in by[d]]
    test = [r for d in docs[h:] for r in by[d]]
    thr = sorted({round(float(r["p3_score_gap"]), 2) for r in dev})
    thr = thr[:: max(1, len(thr) // 200)] + [1e9]
    best_T, best = None, -1
    for T in thr:
        a = np.mean([(r["p4_correct"] if r["p3_score_gap"] < T else r["p3_correct"]) for r in dev])
        if a > best: best, best_T = a, T
    calls = np.mean([r["p3_score_gap"] < best_T for r in test]) * 100
    acc   = np.mean([(r["p4_correct"] if r["p3_score_gap"] < best_T else r["p3_correct"]) for r in test]) * 100
    return best_T, calls, acc

def value_profile(rows, w_frac=0.15, step_frac=0.01):
    """Local net Delta Accuracy@1 in a sliding window over the retriever-confidence
    percentile axis."""
    g  = np.array([float(r["p3_score_gap"]) for r in rows])
    o  = np.argsort(g, kind="stable")
    d  = np.array([(1 if r["p4_correct"] else 0) - (1 if r["p3_correct"] else 0) for r in rows])[o]
    n  = len(d); w = max(60, int(w_frac * n)); st = max(1, int(step_frac * n))
    cs = np.concatenate([[0], np.cumsum(d)])
    xs, ys = [], []
    for s in range(0, n - w + 1, st):
        xs.append((s + w / 2) / n * 100); ys.append((cs[s + w] - cs[s]) / w * 100)
    return np.array(xs), np.array(ys)

def style(ax):
    ax.set_facecolor("white")
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#9a9a9a"); ax.spines[s].set_linewidth(0.8)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=3, width=0.8)
    for t in ax.get_xticklabels() + ax.get_yticklabels(): t.set_color(INK)

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white",
    "savefig.facecolor": "white", "axes.labelcolor": INK,
    "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
})
