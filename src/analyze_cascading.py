"""
Find the optimal --phase4-threshold from a completed evaluation log.

Cascading skips the LLM when Phase 3's top-1/top-2 score gap is >= threshold.
The idea is to only call the LLM on ambiguous cases (small gap) and trust
Phase 3 on confident ones (large gap) — which also protects the confident
cases from being talked out of a correct answer.

This script reads the p3_score_gap that evaluate_pipeline.py now logs for
every changed case, and simulates every threshold: for each candidate value
it counts how many Phase 4 improvements would be KEPT (gap < threshold, LLM
still runs) and how many degradations would be AVOIDED (gap >= threshold, LLM
skipped). It reports the threshold with the best net effect.

Run a normal Phase 4 evaluation with --phase4-threshold 0 first (LLM on all,
so every case is logged with its gap), then:

    python3 src/analyze_cascading.py
    python3 src/analyze_cascading.py path/to/pipeline_evaluation_log.json
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    log_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1
        else PROJECT_ROOT / "src" / "pipeline_evaluation_log.json"
    )
    d = json.load(open(log_path))
    changes = d.get("changes", [])

    # Only cases the LLM actually touched (Phase 3 != Phase 4 outcome) carry a
    # meaningful signal for cascading. A case where the LLM was called and
    # agreed with Phase 3 is unaffected by skipping it.
    improvements = [
        c for c in changes
        if c.get("p4_correct") and not c.get("p3_correct")
        and c.get("p3_score_gap") is not None
    ]
    degradations = [
        c for c in changes
        if c.get("p3_correct") and not c.get("p4_correct")
        and c.get("p3_score_gap") is not None
    ]

    if not improvements and not degradations:
        print("No usable cases. Re-run evaluation with --phase4-threshold 0 "
              "so every changed case is logged with its p3_score_gap.")
        if changes and "p3_score_gap" not in changes[0]:
            print("(This log predates score-gap logging — re-run needed.)")
        return

    imp_gaps = sorted(c["p3_score_gap"] for c in improvements)
    deg_gaps = sorted(c["p3_score_gap"] for c in degradations)

    def summarize(name, gaps):
        if not gaps:
            print(f"  {name}: none")
            return
        n = len(gaps)
        med = gaps[n // 2]
        print(f"  {name}: n={n}  min={gaps[0]:.1f}  median={med:.1f}  "
              f"max={gaps[-1]:.1f}")

    print(f"Log: {log_path}")
    print(f"Phase 4 changed cases: "
          f"{len(improvements)} improvements, {len(degradations)} degradations")
    print()
    print("Score-gap distribution (top1 - top2 at Phase 3):")
    summarize("improvements (want LLM to RUN: small gap)", imp_gaps)
    summarize("degradations (want LLM to SKIP: large gap)", deg_gaps)
    print()

    # Simulate every threshold. LLM runs iff gap < threshold.
    #   kept improvement   = improvement with gap <  threshold
    #   avoided degradation = degradation with gap >= threshold
    candidates = sorted({round(g, 1) for g in imp_gaps + deg_gaps} | {0.0})
    # extend a bit beyond the largest gap so "skip almost everything" is tried
    candidates.append((deg_gaps + imp_gaps and max(deg_gaps + imp_gaps) or 0) + 5)

    best = None
    rows = []
    base_net = len(improvements) - len(degradations)  # threshold=0 → call all
    for t in candidates:
        kept_imp = sum(1 for g in imp_gaps if g < t) if t > 0 else len(imp_gaps)
        lost_imp = len(imp_gaps) - kept_imp
        avoided_deg = sum(1 for g in deg_gaps if g >= t) if t > 0 else 0
        net = kept_imp - (len(deg_gaps) - avoided_deg)
        rows.append((t, kept_imp, lost_imp, avoided_deg, net))
        if best is None or net > best[4] or (net == best[4] and t < best[0]):
            best = (t, kept_imp, lost_imp, avoided_deg, net)

    print(f"{'thresh':>7} {'kept+':>6} {'lost+':>6} {'saved-':>7} {'net':>5}")
    print("-" * 36)
    for t, ki, li, ad, net in rows:
        mark = "  <== best" if (t, ki, li, ad, net) == best else ""
        print(f"{t:7.1f} {ki:6d} {li:6d} {ad:7d} {net:+5d}{mark}")
    print("-" * 36)
    print(f"threshold 0 (call LLM on everything): net {base_net:+d}")
    print(f"best threshold {best[0]:.1f}: net {best[4]:+d} "
          f"(keeps {best[1]}/{len(imp_gaps)} improvements, "
          f"avoids {best[3]}/{len(deg_gaps)} degradations)")
    print()
    print(f"=> try:  --phase4-threshold {best[0]:.1f}")
    print("   Also cuts LLM calls (and cost) for every case with a gap >= that.")
    print("   NOTE: tune on a subset, then confirm on the full run — the gap")
    print("   distribution shifts a little between the head slice and the whole set.")


if __name__ == "__main__":
    main()
