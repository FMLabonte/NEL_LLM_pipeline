"""
Retriever-strength curve: LLM disambiguation gain as a function of retriever R@1.

Takes several prediction files (same LLM, different retriever config) and prints
one row per retriever: R@1 (=Phase-3 top-1), final acc (Phase 4), the LLM gain,
how often the LLM intervened, and its yield (fixes / winnable cases).

The core causal claim: the LLM gain shrinks as the retriever gets stronger.

Usage:
    python3 src/analyze_retriever_curve.py \
        preds_zs_weak.jsonl preds_zs_mid.jsonl preds_qwen3-4b-zeroshot_bc5cdr.jsonl \
        --labels weak,mid,full
"""

import argparse
import json


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def acc(rows, f):
    return sum(bool(r.get(f)) for r in rows) / len(rows) * 100 if rows else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--labels", default=None, help="comma-separated names, one per file")
    args = ap.parse_args()

    labels = args.labels.split(",") if args.labels else args.preds
    print(f"{'retriever':14s} {'R@1 (P3)':>9} {'final (P4)':>11} "
          f"{'LLM gain':>9} {'changed':>8} {'yield':>7}")
    print("-" * 62)
    for path, lab in zip(args.preds, labels):
        rows = load(path)
        n = len(rows)
        p3, p4 = acc(rows, "p3_correct"), acc(rows, "p4_correct")
        changed = sum(1 for r in rows if r.get("llm_changed"))
        winnable = sum(1 for r in rows
                       if r.get("gold_in_list") and not r.get("p3_correct"))
        fixes = sum(1 for r in rows
                    if r.get("p4_correct") and not r.get("p3_correct"))
        yld = fixes / winnable * 100 if winnable else 0.0
        print(f"{lab:14s} {p3:8.1f}% {p4:10.1f}% {p4-p3:+8.1f} "
              f"{changed/n*100:6.1f}% {yld:6.1f}%")
    print("-" * 62)
    print("Erwartung: je höher R@1, desto kleiner der LLM-Gewinn (die Kurve).")


if __name__ == "__main__":
    main()
