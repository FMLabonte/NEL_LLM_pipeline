"""Shared plumbing for the paper figures (ACL two-column geometry).

Everything is computed from the preds_*.jsonl dumps in the repository root:
no model, no cluster, only numpy + matplotlib.

    COL   = 3.15 in   single column  (\\columnwidth)
    WIDE  = 6.30 in   both columns   (figure*, \\textwidth)
"""
import json, random
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

# repo root:  <root>/figures/final/aclfig.py  ->  ../..
ROOT = Path(__file__).resolve().parent.parent.parent   # repo root (scripts/figures/ -> ..)
FIGDIR = ROOT / "figures"                              # all figure output, gitignored

COL, WIDE = 3.15, 6.30

# ---------------------------------------------------------------- runs
# label, dataset, retriever, preds file, marker
RUNS = [
    ("MedMentions · SapBERT",  "MedMentions", "SapBERT",       "preds_zs_medmentions_emb.jsonl",      "o"),
    ("NCBI · SapBERT",         "NCBI",        "SapBERT",       "preds_zs_ncbi_sapbert.jsonl",         "o"),
    ("BioRED · SapBERT",       "BioRED",      "SapBERT",       "preds_zs_biored_sapbert.jsonl",       "o"),
    ("NLM-Chem · dense",       "NLM-Chem",    "SapBERT",       "preds_zs_nlmchem.jsonl",              "o"),
    ("BC5CDR · BM25",          "BC5CDR",      "BM25",          "preds_zs_bc5cdr_bm25.jsonl",          "s"),
    ("BC5CDR · SapBERT",       "BC5CDR",      "SapBERT",       "preds_zs_bc5cdr_sapbert.jsonl",       "o"),
    ("BC5CDR · rule pipeline", "BC5CDR",      "rule pipeline", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl", "D"),
    ("BC5CDR · BioSyn",        "BC5CDR",      "BioSyn",        "preds_zs_bc5cdr_biosyn.jsonl",        "^"),
]

# colour-blind-validated categorical palette (dataset identity)
DSCOL = {"BC5CDR": "#0072B2", "MedMentions": "#D55E00", "BioRED": "#009E73",
         "NCBI": "#7B4FA0", "NLM-Chem": "#B08000"}
INK, MUTED, GRID = "#1a1a1a", "#5a5a5a", "#dcdcdc"
NEG = "#B0472A"          # "the LLM hurts here" wash

# sequential ramp for retriever strength (truncated plasma: keeps contrast at both ends)
_p = mpl.colormaps["plasma"]
STRENGTH = mpl.colors.LinearSegmentedColormap.from_list(
    "strength", _p(np.linspace(0.04, 0.70, 256)))
SNORM = mpl.colors.Normalize(vmin=54, vmax=86)


# ---------------------------------------------------------------- data
def load(fname):
    """One dict per mention. Rows without a retriever confidence are dropped."""
    rows = [json.loads(l) for l in open(ROOT / fname) if l.strip()]
    return [r for r in rows if r.get("p3_score_gap") is not None]


def by_confidence(rows):
    """Rows sorted by the retriever's own top1-top2 gap, least confident first.

    NOTE  preds_zs_bc5cdr_bm25.jsonl encodes 'no second candidate' as 999.0.
    Those mentions are genuinely the most confident ones (no competitor at all),
    so they are kept: the axis is a *rank* axis, which is insensitive to the
    magnitude of the sentinel.  They occupy the top ~11 % of that run.
    """
    g = np.array([float(r["p3_score_gap"]) for r in rows])
    o = np.argsort(g, kind="stable")
    p3 = np.array([1.0 if rows[i]["p3_correct"] else 0.0 for i in o])
    p4 = np.array([1.0 if rows[i]["p4_correct"] else 0.0 for i in o])
    return o, p3, p4


def common_subset(runs):
    """runs: list of row-lists in identical emission order, the shortest a subset."""
    key = lambda r: (r["pmid"], r["mention"], r["gold_id"])
    short = min(runs, key=len)
    out = []
    for rows in runs:
        if len(rows) == len(short):
            out.append(rows); continue
        keep, j = [], 0
        for r in rows:
            if j < len(short) and key(r) == key(short[j]):
                keep.append(r); j += 1
        if j != len(short):
            raise ValueError("not an ordered subset")
        out.append(keep)
    return out


def budget_curve(rows, grid=None):
    """Accuracy@1 as a function of the share of mentions routed to the LLM.

    The gate sends the least-confident mentions first, so budget b means
    'call the LLM on the b % of mentions with the smallest top1-top2 gap'.
    """
    _, p3, p4 = by_confidence(rows)
    n = len(p3)
    c3, c4 = np.concatenate([[0], np.cumsum(p3)]), np.concatenate([[0], np.cumsum(p4)])
    acc = (c4 + (c3[-1] - c3)) / n * 100.0
    frac = np.arange(n + 1) / n * 100.0
    if grid is None:
        grid = np.linspace(0, 100, 401)          # smooth 0.25 % raster
    return grid, np.interp(grid, frac, acc)


def dev_tuned_gate(rows, seed=0):
    """Threshold chosen on half the DOCUMENTS, reported on the other half.

    Returns (threshold, share of calls on test, accuracy on test).
    """
    by = defaultdict(list)
    for r in rows:
        by[r.get("pmid")].append(r)
    docs = list(by)
    random.Random(seed).shuffle(docs)
    h = len(docs) // 2
    dev = [r for d in docs[:h] for r in by[d]]
    test = [r for d in docs[h:] for r in by[d]]
    cand = sorted({round(float(r["p3_score_gap"]), 2) for r in dev})
    cand = cand[:: max(1, len(cand) // 200)] + [1e9]
    best_T, best = None, -1.0
    for T in cand:
        a = np.mean([(r["p4_correct"] if r["p3_score_gap"] < T else r["p3_correct"]) for r in dev])
        if a > best:
            best, best_T = a, T
    calls = np.mean([r["p3_score_gap"] < best_T for r in test]) * 100
    acc = np.mean([(r["p4_correct"] if r["p3_score_gap"] < best_T else r["p3_correct"])
                   for r in test]) * 100
    return best_T, calls, acc


def value_profile(rows, window=0.25, step=0.005):
    """Local net Delta Accuracy@1 (P4 - P3) in a sliding window over the
    retriever-confidence percentile axis."""
    _, p3, p4 = by_confidence(rows)
    d = p4 - p3
    n = len(d)
    w, st = max(80, int(window * n)), max(1, int(step * n))
    cs = np.concatenate([[0], np.cumsum(d)])
    xs = np.arange(0, n - w + 1, st)
    return (xs + w / 2) / n * 100, (cs[xs + w] - cs[xs]) / w * 100


def confident_delta(rows, lo=20, seed=0, n_boot=800):
    """Net Delta Accuracy@1 on the (100-lo) % of mentions the retriever is most
    confident about, with a document-level bootstrap 95 % CI."""
    o, p3, p4 = by_confidence(rows)
    d = p4 - p3
    n = len(d)
    k = int(lo / 100 * n)
    pm = [rows[i].get("pmid") for i in o]
    idx = defaultdict(list)
    for i in range(k, n):
        idx[pm[i]].append(i)
    docs = list(idx)
    rng = np.random.default_rng(seed)
    vals = [d[np.concatenate([idx[docs[j]] for j in rng.choice(len(docs), len(docs))])].mean() * 100
            for _ in range(n_boot)]
    return d[k:].mean() * 100, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def recall_at_1(rows):
    return float(np.mean([r["p3_correct"] for r in rows]) * 100)


# ------------------------------------------------------- confidence bands
# Absolute gap edges. These are only comparable WITHIN one retriever, because
# score scales differ per retriever -- never bin across runs with them.
BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
BLAB = ["[0,2)", "[2,5)", "[5,10)", "[10,20)", "20+"]


def band_of(gap):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= gap < hi:
            return i
    return None


def in_band(rows, i):
    return [r for r in rows if band_of(r["p3_score_gap"]) == i]


def acc(rows, field):
    return float(np.mean([bool(r.get(field)) for r in rows]) * 100) if rows else float("nan")


def precision(rows):
    """Intervention precision: of the changes that altered correctness, how many
    helped. Returns (precision in %, fixes, breaks); precision is None when the
    model never intervened decisively, which is not the same as 0 %."""
    fx = sum(1 for r in rows if r.get("p4_correct") and not r.get("p3_correct"))
    bk = sum(1 for r in rows if r.get("p3_correct") and not r.get("p4_correct"))
    return (fx / (fx + bk) * 100 if fx + bk else None), fx, bk


def band_delta(rows, i):
    sub = in_band(rows, i)
    return acc(sub, "p4_correct") - acc(sub, "p3_correct") if sub else float("nan")


# ------------------------------------------------- fine-tuning: seeds, seen/unseen
SEEDS = (1, 2, 3)


def seed_files(tag, template="preds_ft_{tag}_s{seed}.jsonl"):
    """The seed runs that exist, in seed order. Missing seeds are skipped rather
    than faked, so a figure built before all runs land is visibly thinner."""
    out = []
    for s in SEEDS:
        p = ROOT / template.format(tag=tag, seed=s)
        if p.exists():
            out.append((s, load(p.name)))
    return out


def mean_range(vals):
    """Mean and the observed range. With three seeds a standard deviation is
    noise dressed as a statistic; the range is what we report in the paper."""
    v = [x for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(v)), float(min(v)), float(max(v))


def train_concepts(fname="train_gold_ids.txt"):
    p = ROOT / fname
    return set(open(p).read().split()) if p.exists() else set()


def is_seen(row, concepts):
    """A mention counts as 'seen' if its gold concept appeared in the fine-tuning
    data. Gold ids can be pipe-joined (composite mentions); any match counts."""
    return any(g in concepts for g in str(row.get("gold_id", "")).split("|"))


# ---------------------------------------------------------------- style
def use_acl():
    """Times-metric type at ACL figure sizes."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["TeX Gyre Termes", "Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7, "figure.facecolor": "white",
        "savefig.facecolor": "white", "axes.labelcolor": INK, "text.color": INK,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "lines.solid_capstyle": "round",
        # STIX is Times-compatible, so $\Delta$ and $\rho$ match the body face
        "mathtext.fontset": "stix",
    })


def style(ax, grid_axis="both"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#8f8f8f")
        ax.spines[s].set_linewidth(0.6)
    ax.grid(True, axis=grid_axis, color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=2.5, width=0.6, pad=2)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(INK)


def save(fig, stem):
    """Save at the declared figsize.

    No bbox_inches='tight': the figures are built at exactly ACL \textwidth, and
    cropping would make LaTeX scale them up, silently enlarging every label.
    Constrained layout is what keeps the content inside the frame instead.
    """
    out = FIGDIR
    out.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"{stem}.{ext}", dpi=400)
    w, h = fig.get_size_inches()
    print(f"wrote {stem}.pdf / .png  ({w:.2f} x {h:.2f} in)")
