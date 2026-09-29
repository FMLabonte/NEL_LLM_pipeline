#!/usr/bin/env python3
"""Dose-response analysis for the graded retriever-degradation sweep.

Reads the preds produced by cluster/bender_retriever_dose.sbatch and answers, on one
fixed candidate pool and one fixed mention set, the question the paper's frame
currently answers only observationally: as the retriever's top-1 accuracy is
lowered and nothing else changes, what happens to the LLM stage's gain, and to
its effect on the mentions the retriever is confident about?

    python3 src/analyze_dose.py
    python3 src/analyze_dose.py --figure figures/fig6_dose.pdf

Prints a table and, with --figure, writes the dose-response plot.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

# rate -> preds file. p=0 reuses the BioSyn run the paper already reports.
DEFAULT_RUNS = [
    (0.0, "preds_zs_bc5cdr_biosyn.jsonl"),
    (0.1, "preds_dose_p0.1.jsonl"),
    (0.2, "preds_dose_p0.2.jsonl"),
    (0.3, "preds_dose_p0.3.jsonl"),
    (0.4, "preds_dose_p0.4.jsonl"),
    (0.5, "preds_dose_p0.5.jsonl"),
]


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return [r for r in rows if r.get("p3_score_gap") is not None]


def acc(rows, field):
    return float(np.mean([bool(r.get(field)) for r in rows]) * 100) if rows else float("nan")


def confident_delta(rows, lo=20, n_boot=600, seed=0):
    """Net delta on the (100-lo)% of mentions the retriever is most confident
    about, with a document-level bootstrap 95% CI."""
    g = np.array([float(r["p3_score_gap"]) for r in rows])
    o = np.argsort(g, kind="stable")
    d = np.array([(1 if rows[i]["p4_correct"] else 0) - (1 if rows[i]["p3_correct"] else 0)
                  for i in o], float)
    k = int(lo / 100 * len(d))
    idx = defaultdict(list)
    for i in range(k, len(d)):
        idx[rows[o[i]].get("pmid")].append(i)
    docs = list(idx)
    rng = np.random.default_rng(seed)
    vals = [d[np.concatenate([idx[docs[j]] for j in rng.choice(len(docs), len(docs))])].mean() * 100
            for _ in range(n_boot)]
    return d[k:].mean() * 100, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figure", default=None, help="write the dose-response plot here")
    a = ap.parse_args()

    rows_by, missing = {}, []
    for rate, fname in DEFAULT_RUNS:
        p = ROOT / fname
        if p.exists():
            rows_by[rate] = load(p)
        else:
            missing.append(fname)
    if missing:
        print("missing (job not finished yet?):", ", ".join(missing))
    if len(rows_by) < 2:
        print("need at least two levels to say anything. Stopping.")
        return

    print(f"{'rate':>6}{'n':>7}{'R@1':>8}{'R@10':>8}{'P4':>8}{'gain':>8}"
          f"{'rate%':>8}{'confD':>8}{'CI':>16}")
    rates, r1s, gains, confs, los, his = [], [], [], [], [], []
    for rate in sorted(rows_by):
        r = rows_by[rate]
        r1, p4 = acc(r, "p3_correct"), acc(r, "p4_correct")
        cd, lo, hi = confident_delta(r)
        print(f"{rate:>6g}{len(r):>7d}{r1:>8.1f}{acc(r,'gold_in_list'):>8.1f}{p4:>8.1f}"
              f"{p4-r1:>+8.1f}{acc(r,'llm_changed'):>8.1f}{cd:>+8.2f}"
              f"{f'[{lo:+.2f},{hi:+.2f}]':>16}")
        rates.append(rate); r1s.append(r1); gains.append(p4 - r1)
        confs.append(cd); los.append(lo); his.append(hi)

    if len(rates) >= 3:
        from scipy import stats
        sl, ic, rr, pp, _ = stats.linregress(r1s, gains)
        print(f"\n  gain vs Recall@1 on a fixed candidate pool: "
              f"slope {sl:+.3f} points per point, r={rr:+.3f}, p={pp:.4g}")
        sl2, ic2, rr2, pp2, _ = stats.linregress(r1s, confs)
        print(f"  confident-region delta vs Recall@1:          "
              f"slope {sl2:+.3f}, r={rr2:+.3f}, p={pp2:.4g}, "
              f"zero crossing at Recall@1 = {-ic2/sl2:.1f}")

    if a.figure:
        import sys
        sys.path.insert(0, str(ROOT / "scripts" / "figures"))
        from aclfig import COL, DSCOL, INK, MUTED, NEG, use_acl, style
        import matplotlib.pyplot as plt
        use_acl()
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(6.3, 2.3), dpi=400,
                                       layout="constrained")
        fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.07)
        blue = DSCOL["BC5CDR"]
        style(axL)
        axL.plot(r1s, gains, color=blue, lw=1.4, marker="o", ms=4.4,
                 mfc="white", mew=1.2, zorder=3)
        for x, y, p in zip(r1s, gains, rates):
            axL.annotate(f"p={p:g}", (x, y), xytext=(0, 6), textcoords="offset points",
                         ha="center", fontsize=6.0, color=MUTED)
        axL.set_xlabel("Candidate Recall@1 (%), degraded", labelpad=2)
        axL.set_ylabel("gain of the LLM stage (points)", labelpad=2)
        axL.text(.012, .955, "(a)", transform=axL.transAxes, ha="left", va="top",
                 fontsize=8.5, fontweight="bold")
        style(axR)
        axR.axhspan(min(los) - 1, 0, color=NEG, alpha=.06, lw=0, zorder=0)
        axR.axhline(0, color="#2a2a2a", lw=.8, zorder=2)
        for x, lo, hi in zip(r1s, los, his):
            axR.plot([x, x], [lo, hi], color=blue, lw=.9, alpha=.7, zorder=3)
        axR.plot(r1s, confs, color=blue, lw=1.4, marker="o", ms=4.4,
                 mfc="white", mew=1.2, zorder=4)
        axR.set_xlabel("Candidate Recall@1 (%), degraded", labelpad=2)
        axR.set_ylabel("net $\\Delta$ on the confident 80 %", labelpad=2)
        axR.text(.012, .955, "(b)", transform=axR.transAxes, ha="left", va="top",
                 fontsize=8.5, fontweight="bold")
        out = Path(a.figure)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        fig.savefig(out.with_suffix(".png"))
        print(f"\n  wrote {out} / {out.with_suffix('.png').name}")


if __name__ == "__main__":
    main()
