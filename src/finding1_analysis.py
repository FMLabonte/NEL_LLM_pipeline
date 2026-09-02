"""
Finding 1 — rigorous: the mid-confidence "danger zone" + the confidence gate.

Runs on ENRICHED prediction dumps (need fields: p3_score_gap, p3_correct,
p4_correct, llm_changed, pmid). For each dataset it prints:

  * per confidence band:  n, intervention RATE, precision (+ document-level
    bootstrap 95% CI), net delta  -> the mechanism (rate monotone, precision dips)
  * the gate Pareto curve: accuracy vs fraction of LLM calls (threshold sweep)
  * an honest dev/test gate: tune the threshold on half the DOCUMENTS, report on
    the other half (so the threshold is never chosen on the reported set)

It also writes a JSON with every number so the figures can be built from it.

Usage:
  python3 src/finding1_analysis.py \
     preds_qwen3-4b-zeroshot_bc5cdr.jsonl preds_zs_biored.jsonl preds_zs_medmentions.jsonl \
     --labels BC5CDR-zs,BioRED-zs,MedMentions-zs --out finding1.json
"""
import json, argparse, random
from collections import defaultdict

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
BLAB = ["[0,2)", "[2,5)", "[5,10)", "[10,20)", "20+"]


def load(path):
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def band_of(gap):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= gap < hi:
            return i
    return None


def precision(rows):
    fx = sum(1 for r in rows if r.get("p4_correct") and not r.get("p3_correct"))
    bk = sum(1 for r in rows if r.get("p3_correct") and not r.get("p4_correct"))
    return (fx / (fx + bk) if (fx + bk) else None), fx, bk


def acc(rows, f):
    return sum(1 for r in rows if r.get(f)) / len(rows) if rows else 0.0


def doc_bootstrap_ci(rows, stat, n_boot=1000, seed=0):
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r.get("pmid")].append(r)
    docs = list(by_doc)
    if not docs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        samp = []
        for _ in range(len(docs)):
            samp.extend(by_doc[rng.choice(docs)])
        v = stat(samp)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def gate_curve(rows, thresholds):
    base = acc(rows, "p3_correct")
    full = acc(rows, "p4_correct")
    out = []
    for T in thresholds:
        calls = sum(1 for r in rows if r["p3_score_gap"] < T)
        correct = sum(1 for r in rows
                      if (r["p4_correct"] if r["p3_score_gap"] < T else r["p3_correct"]))
        out.append({"T": T, "calls_frac": calls / len(rows),
                    "accuracy": correct / len(rows)})
    return out, base, full


def dev_test_gate(rows, thresholds, seed=0):
    """Tune threshold on half the DOCUMENTS (maximise accuracy), report on the other half."""
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[r.get("pmid")].append(r)
    docs = list(by_doc)
    random.Random(seed).shuffle(docs)
    half = len(docs) // 2
    dev = [r for d in docs[:half] for r in by_doc[d]]
    test = [r for d in docs[half:] for r in by_doc[d]]
    # best threshold on dev
    best_T, best_acc = None, -1
    for T in thresholds:
        a = sum(1 for r in dev if (r["p4_correct"] if r["p3_score_gap"] < T else r["p3_correct"])) / len(dev)
        if a > best_acc:
            best_acc, best_T = a, T
    # report on test with that threshold
    calls = sum(1 for r in test if r["p3_score_gap"] < best_T)
    test_acc = sum(1 for r in test if (r["p4_correct"] if r["p3_score_gap"] < best_T else r["p3_correct"])) / len(test)
    return {"dev_best_T": best_T, "test_calls_frac": calls / len(test),
            "test_accuracy": test_acc, "test_full_accuracy": acc(test, "p4_correct"),
            "test_base_accuracy": acc(test, "p3_correct")}


def analyze(path, label):
    rows = [r for r in load(path) if r.get("p3_score_gap") is not None]
    print("=" * 74)
    print(f"{label}   (n={len(rows)})")
    print("=" * 74)
    print(f"{'band':10}{'n':>7}{'rate':>9}{'precision':>11}{'95% CI':>13}{'netΔ':>8}")
    bands_out = []
    for i, lab in enumerate(BLAB):
        sub = [r for r in rows if band_of(r["p3_score_gap"]) == i]
        if not sub:
            continue
        rate = sum(1 for r in sub if r.get("llm_changed")) / len(sub)
        p, fx, bk = precision(sub)
        ci = doc_bootstrap_ci(sub, lambda s: precision(s)[0])
        d = (acc(sub, "p4_correct") - acc(sub, "p3_correct")) * 100
        ps = f"{p*100:.0f}%" if p is not None else "-"
        cis = f"[{ci[0]*100:.0f},{ci[1]*100:.0f}]" if ci else "-"
        print(f"{lab:10}{len(sub):7d}{rate*100:8.1f}%{ps:>11}{cis:>13}{d:+8.1f}")
        bands_out.append({"band": lab, "n": len(sub), "rate": rate,
                          "precision": p, "ci": ci, "fixes": fx, "breaks": bk, "delta": d})
    thr = [1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 1e9]
    curve, base, full = gate_curve(rows, thr)
    print(f"\n  gate Pareto  (base P3={base*100:.1f}%, full P4={full*100:.1f}%):")
    print(f"  {'gap<T':>8}{'calls%':>9}{'accuracy':>10}")
    for c in curve:
        lab = f"<{c['T']:g}" if c["T"] < 1e8 else "all"
        print(f"  {lab:>8}{c['calls_frac']*100:8.1f}%{c['accuracy']*100:9.1f}%")
    dt = dev_test_gate(rows, thr)
    print(f"\n  honest dev/test gate: threshold tuned on dev docs = gap<{dt['dev_best_T']:g}")
    print(f"    -> on held-out test docs: {dt['test_calls_frac']*100:.0f}% of LLM calls, "
          f"acc {dt['test_accuracy']*100:.1f}%  (full {dt['test_full_accuracy']*100:.1f}%, "
          f"base {dt['test_base_accuracy']*100:.1f}%)")
    return {"label": label, "n": len(rows), "bands": bands_out,
            "pareto": curve, "base": base, "full": full, "dev_test": dt}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels = a.labels.split(",") if a.labels else a.preds
    results = [analyze(p, l) for p, l in zip(a.preds, labels)]
    if a.out:
        json.dump(results, open(a.out, "w"), indent=2, default=lambda x: None)
        print(f"\nwrote {a.out}")
