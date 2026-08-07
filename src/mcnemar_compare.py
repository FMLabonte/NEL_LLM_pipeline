"""
Paired significance test (McNemar) between two evaluation runs.

Comparing marginal accuracies (e.g. 74.5% vs 75.5%) does NOT tell you whether a
difference is real — on 400 mentions, 1% is 4 mentions, well inside the noise.
Because both runs are evaluated on the SAME mentions, the correct test is
PAIRED: look only at the mentions where the two configs DISAGREE.

McNemar's test uses exactly those discordant pairs:
    b = A correct, B wrong
    c = A wrong,   B correct
and asks whether b and c differ more than chance would produce.

Usage:
    # produce prediction files with:  --dump-predictions run_X.jsonl
    python3 src/mcnemar_compare.py run_baseline.jsonl run_2sentences.jsonl
    python3 src/mcnemar_compare.py a.jsonl b.jsonl --phase p3
"""

import argparse
import json
from pathlib import Path


def load(path):
    """Map (pmid, mention, gold_id) -> correctness dict."""
    out = {}
    for line in open(path):
        r = json.loads(line)
        key = (r["pmid"], r["mention"], r["gold_id"])
        out[key] = r
    return out


def mcnemar_p(b, c):
    """Two-sided p-value. Exact binomial for small n, normal approx otherwise."""
    n = b + c
    if n == 0:
        return 1.0
    if n < 25:
        from math import comb
        k = min(b, c)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        return min(1.0, 2 * tail)
    # continuity-corrected chi-square -> p via survival function
    from math import erfc, sqrt
    chi2 = (abs(b - c) - 1) ** 2 / n
    return erfc(sqrt(chi2 / 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--phase", choices=["p3", "p4"], default="p4",
                    help="Which stage's correctness to compare (default: p4).")
    args = ap.parse_args()

    A, B = load(args.file_a), load(args.file_b)
    keys = sorted(set(A) & set(B))
    only_a, only_b = len(set(A) - set(B)), len(set(B) - set(A))
    if only_a or only_b:
        print(f"  warn: {only_a} mentions only in A, {only_b} only in B "
              f"(compared on the {len(keys)} shared)")

    field = f"{args.phase}_correct"
    a_corr = sum(A[k][field] for k in keys)
    b_corr = sum(B[k][field] for k in keys)
    both = sum(A[k][field] and B[k][field] for k in keys)
    b_only = sum(A[k][field] and not B[k][field] for k in keys)   # A right, B wrong
    c_only = sum((not A[k][field]) and B[k][field] for k in keys)  # A wrong, B right
    neither = sum((not A[k][field]) and (not B[k][field]) for k in keys)
    n = len(keys)

    p = mcnemar_p(b_only, c_only)

    print(f"Phase {args.phase} | n = {n} shared mentions")
    print(f"  A ({Path(args.file_a).name}): {a_corr}/{n} = {a_corr/n*100:.2f}%")
    print(f"  B ({Path(args.file_b).name}): {b_corr}/{n} = {b_corr/n*100:.2f}%")
    print(f"  marginal difference: {(b_corr - a_corr)/n*100:+.2f} pp")
    print()
    print(f"  Discordant pairs (the only ones that matter):")
    print(f"    A right / B wrong (b): {b_only}")
    print(f"    A wrong / B right (c): {c_only}")
    print(f"    both right: {both} | both wrong: {neither}")
    print()
    print(f"  McNemar two-sided p = {p:.4f}")
    if p < 0.05:
        winner = args.file_b if c_only > b_only else args.file_a
        print(f"  => SIGNIFICANT (p<0.05). Better: {Path(winner).name}")
    else:
        print(f"  => NOT significant. The difference is within noise — do not "
              f"switch configs (or fine-tune) based on it.")


if __name__ == "__main__":
    main()
