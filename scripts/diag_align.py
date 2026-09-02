#!/usr/bin/env python3
"""Why do two preds dumps of the same corpus not cover the same mentions?

    python3 scripts/diag_align.py preds_zs_nlmchem.jsonl preds_zs_nlmchem_qwen3-8b.jsonl

Answers one question: when the shorter run is *supposed* to be a subset of the
longer one (e.g. produced with --limit 1500 --limit-random from the same
corpus), which of its rows have no counterpart, and why. Prints examples, so
the cause is visible rather than inferred.

Reads prediction dumps only. No model, no GPU, stdlib only.
"""
import json
import sys
from collections import Counter, defaultdict


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def key(r):
    return (r["pmid"], r["mention"], r.get("gold_id"))


def main():
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    a_path, b_path = sys.argv[1], sys.argv[2]
    A, B = load(a_path), load(b_path)
    if len(A) > len(B):
        A, B, a_path, b_path = B, A, b_path, a_path
    print(f"kurz : {a_path}  {len(A)} Zeilen")
    print(f"lang : {b_path}  {len(B)} Zeilen")

    for lab, R in (("kurz", A), ("lang", B)):
        n_null = sum(1 for r in R if r.get("p3_score_gap") is None)
        print(f"  {lab}: {len(set(r['pmid'] for r in R))} Dokumente, "
              f"{len(set(key(r) for r in R))} eindeutige Keys, "
              f"{n_null} Zeilen ohne p3_score_gap (die verwirft analyze_robustness)")

    da, db = set(r["pmid"] for r in A), set(r["pmid"] for r in B)
    print(f"\nDokumente: kurz\\lang {len(da - db)}, lang\\kurz {len(db - da)}, "
          f"gemeinsam {len(da & db)}")
    if da - db:
        print(f"  nur im kurzen Lauf, Beispiele: {sorted(da - db)[:5]}")

    ca, cb = Counter(key(r) for r in A), Counter(key(r) for r in B)
    missing = {k: n for k, n in ca.items() if k not in cb}
    short = {k: (n, cb[k]) for k, n in ca.items() if k in cb and cb[k] < n}
    matched = sum(min(n, cb.get(k, 0)) for k, n in ca.items())
    print(f"\nZeilen des kurzen Laufs, die zugeordnet werden koennen: {matched} von {len(A)}")
    print(f"  Keys, die im langen Lauf GAR NICHT vorkommen: {len(missing)} "
          f"({sum(missing.values())} Zeilen)")
    print(f"  Keys, die im langen Lauf SELTENER vorkommen:  {len(short)} "
          f"({sum(n - m for n, m in short.values())} Zeilen)")

    # Wo genau weichen sie ab? Mention allein, oder gold_id?
    by_mention = defaultdict(set)
    for k in cb:
        by_mention[(k[0], k[1])].add(k[2])
    same_mention_other_gold = [
        (k, sorted(by_mention[(k[0], k[1])])) for k in missing
        if (k[0], k[1]) in by_mention]
    print(f"\n  davon mit gleicher (pmid, mention), aber anderem gold_id: "
          f"{len(same_mention_other_gold)}")
    for k, alts in same_mention_other_gold[:5]:
        print(f"    pmid {k[0]}  \"{k[1]}\"   kurz: {k[2]!r}   lang: {alts!r}")
    only_pmid_mention = [k for k in missing if (k[0], k[1]) not in by_mention]
    print(f"  davon mit (pmid, mention), die im langen Lauf fehlt: {len(only_pmid_mention)}")
    for k in only_pmid_mention[:5]:
        print(f"    pmid {k[0]}  \"{k[1]}\"  gold {k[2]!r}")
    for k, (n, m) in list(short.items())[:5]:
        print(f"  seltener: pmid {k[0]}  \"{k[1]}\"  kurz {n}x, lang {m}x")

    # Stimmt P3 dort ueberein, wo eine Zuordnung moeglich ist?
    idx = defaultdict(list)
    for r in B:
        idx[key(r)].append(r)
    bad, seen = 0, Counter()
    for r in A:
        k = key(r)
        if seen[k] < len(idx[k]):
            o = idx[k][seen[k]]
            seen[k] += 1
            if r.get("p3_top1") != o.get("p3_top1"):
                bad += 1
    print(f"\nZuordenbare Zeilen mit abweichendem p3_top1: {bad} von {matched}")
    print("  0 = gleiche Retriever-Konfiguration. Alles andere heisst: die Laeufe")
    print("  duerfen nicht verglichen werden, egal wie man sie ausrichtet.")


if __name__ == "__main__":
    main()
