"""
Decomposition: how much — and WHERE — does the LLM disambiguator contribute?

The field reports a single accuracy number, which hides the key fact (shown by
LLM4BioEL): on easy corpora the retriever does almost everything and the LLM
changes almost nothing. This script opens that up. For one model's run it shows:

  * The retriever->LLM delta overall, and the LLM's *change rate* (how many
    predictions it flips) split into fixes vs. breaks.
  * That contribution stratified by:
      - retriever confidence   (Phase-3 top1-vs-top2 score gap)
      - mention difficulty     (surface similarity of mention to the top-1 label)
      - entity type            (Chemical vs Disease ...)
      - whether gold was retrieved at all
      - seen vs unseen concept (needs --train)

The story we expect (and want to quantify): the disambiguator earns its cost
only on LOW-confidence, LOW-surface-similarity, hard mentions — and is dead
weight on the easy majority. That precise claim is the paper.

The stratification fields come from the enriched --dump-predictions. Older
(sparse) prediction files still work — the script runs the parts it can and
tells you what needs a re-run.

Usage:
    python3 src/analyze_decomposition.py preds_4b_finetuned.jsonl
    python3 src/analyze_decomposition.py preds_4b_finetuned.jsonl --train Data/finetune/bc5cdr_train_sft.jsonl
"""

import argparse
import json


def load(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def train_golds(path):
    out = set()
    for line in open(path):
        try:
            g = str(json.loads(line)["meta"]["gold_id"])
            out.update(g.split("|"))
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def pct(n, d):
    return f"{n/d*100:5.1f}%" if d else "   -  "


def acc(rows, field="p4_correct"):
    return sum(bool(r.get(field)) for r in rows) / len(rows) * 100 if rows else 0.0


def stratum_line(name, rows):
    """One row: n, Phase-3 acc, Phase-4 acc, delta, fixes/breaks."""
    if not rows:
        return f"  {name:22s}  n=0"
    p3 = acc(rows, "p3_correct")
    p4 = acc(rows, "p4_correct")
    fixes = sum(1 for r in rows if r.get("p4_correct") and not r.get("p3_correct"))
    breaks = sum(1 for r in rows if r.get("p3_correct") and not r.get("p4_correct"))
    return (f"  {name:22s}  n={len(rows):5d}  "
            f"P3 {p3:5.1f}%  P4 {p4:5.1f}%  Δ {p4-p3:+5.1f}  "
            f"(+{fixes}/-{breaks})")


def show_bins(rows, key, edges, labels):
    have = [r for r in rows if r.get(key) is not None]
    if not have:
        print(f"    (field '{key}' missing — re-run with enriched --dump-predictions)")
        return
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        sub = [r for r in have if lo <= r[key] < hi]
        print(stratum_line(lab, sub))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds")
    ap.add_argument("--train", default=None, help="train JSONL for seen/unseen")
    args = ap.parse_args()

    rows = load(args.preds)
    n = len(rows)
    has_rich = "llm_changed" in rows[0] if rows else False

    print("=" * 68)
    print(f"DECOMPOSITION — {args.preds}   (n={n})")
    print("=" * 68)

    # ── Overall + change rate ──
    p3, p4 = acc(rows, "p3_correct"), acc(rows, "p4_correct")
    print(f"\nOverall:  Phase 3 {p3:.1f}%  →  Phase 4 {p4:.1f}%   (Δ {p4-p3:+.1f})")

    if has_rich:
        changed = [r for r in rows if r.get("llm_changed")]
        fixes = sum(1 for r in changed if r["p4_correct"] and not r["p3_correct"])
        breaks = sum(1 for r in changed if r["p3_correct"] and not r["p4_correct"])
        print(f"LLM changed the prediction on {len(changed)}/{n} = "
              f"{len(changed)/n*100:.1f}% of mentions.")
        print(f"  of those: {fixes} fixes, {breaks} breaks, "
              f"{len(changed)-fixes-breaks} lateral (wrong→wrong / right→right-other)")
        print(f"  -> The LLM does NOTHING on {(n-len(changed))/n*100:.1f}% of mentions "
              f"(retriever already decided).")
    else:
        flips = sum(1 for r in rows if r.get("p3_correct") != r.get("p4_correct"))
        print(f"Outcome flips (P3≠P4 correctness): {flips}/{n} = {flips/n*100:.1f}%")
        print("(sparse file: install enriched --dump-predictions for the full change rate)")

    # ── By entity type ──
    print(f"\n── By entity type ──")
    types = sorted({r.get("entity_type") for r in rows}, key=lambda x: str(x))
    for t in types:
        print(stratum_line(str(t), [r for r in rows if r.get("entity_type") == t]))

    # ── By retriever confidence (the key figure) ──
    print(f"\n── By retriever confidence (Phase-3 top1−top2 score gap) ──")
    print("   low gap = retriever unsure -> LLM should help most here")
    show_bins(rows, "p3_score_gap",
              [0, 2, 5, 10, 20, 1e9],
              ["gap [0,2)", "gap [2,5)", "gap [5,10)", "gap [10,20)", "gap 20+"])

    # ── By mention difficulty (surface similarity) ──
    print(f"\n── By mention difficulty (surface sim: mention vs top-1 label) ──")
    print("   low sim = surface form doesn't decide it -> context/LLM matters")
    show_bins(rows, "surface_sim",
              [0, 40, 60, 80, 95, 100.01],
              ["sim [0,40)", "sim [40,60)", "sim [60,80)", "sim [80,95)", "sim [95,100]"])

    # ── Was gold even retrievable ──
    if has_rich:
        print(f"\n── By retrievability ──")
        print(stratum_line("gold in list", [r for r in rows if r.get("gold_in_list")]))
        print(stratum_line("gold NOT in list", [r for r in rows if not r.get("gold_in_list")]))

    # ── Seen vs unseen ──
    if args.train:
        tg = train_golds(args.train)
        seen = [r for r in rows if any(g in tg for g in str(r["gold_id"]).split("|"))]
        unseen = [r for r in rows if not any(g in tg for g in str(r["gold_id"]).split("|"))]
        print(f"\n── Seen vs unseen concept (train has {len(tg)} concepts) ──")
        print(stratum_line("seen", seen))
        print(stratum_line("unseen", unseen))

    print()


if __name__ == "__main__":
    main()
