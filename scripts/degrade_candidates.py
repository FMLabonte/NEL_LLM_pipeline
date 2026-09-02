#!/usr/bin/env python3
"""Graded retriever degradation on a fixed candidate pool.

The paper's frame ("a stronger retriever leaves the LLM less to do") is measured
by swapping whole retrievers, which changes the candidate pool, the score scale
and top-1 accuracy at once. This script isolates one factor: it keeps the
candidate SET and the score vector of every mention exactly as they are and only
changes WHICH concept sits on top.

With probability p, the concept at rank 1 is swapped with the concept at a
uniformly drawn rank in [2, max_rank]; the two scores stay where they are, so
the score list is still sorted descending and the top1-top2 gap distribution is
unchanged. Only the retriever's accuracy falls. Sweeping p therefore gives a
dose-response curve for retriever strength with everything else held fixed.

The draw is nested across levels: a mention degraded at p is also degraded at
every p' > p, and it always swaps with the same rank. Levels are therefore
directly comparable and differences between them are not sampling noise.

Usage
-----
    python3 scripts/degrade_candidates.py cands_biosyn_bc5cdr.jsonl \
        --rates 0 0.1 0.2 0.3 0.4 0.5 \
        --out-prefix cands_dose_biosyn --gold mentions_bc5cdr_test.jsonl

Writes one JSONL per rate, named <out-prefix>_p<rate>.jsonl, in the format
`evaluate_pipeline.py --candidates-from` expects. With --gold it also prints the
resulting Recall@1 and Recall@10 per level, so you can check the dose landed
before spending GPU time.
"""
import argparse
import hashlib
import json
from pathlib import Path


def _unit(key: str, salt: str) -> float:
    """A stable uniform(0,1) draw for this mention. Same key -> same value, in
    this run and in any later one, so the levels nest and the sweep is
    reproducible without carrying a random state around."""
    h = hashlib.sha256(f"{salt}|{key}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


def degrade(rec: dict, rate: float, max_rank: int, salt: str) -> dict:
    cands = rec.get("candidates") or []
    if rate <= 0 or len(cands) < 2:
        return rec
    key = f'{rec.get("pmid")}|{rec.get("start")}|{rec.get("mention")}'
    if _unit(key, salt + "/select") >= rate:
        return rec                                   # untouched at this level
    hi = min(max_rank, len(cands))
    j = 1 + int(_unit(key, salt + "/rank") * (hi - 1))   # rank index 1..hi-1
    j = max(1, min(j, len(cands) - 1))
    out = dict(rec)
    new = [dict(c) for c in cands]
    # swap the concepts, leave the scores in place: the list stays sorted and
    # the top1-top2 gap distribution is untouched.
    s0, sj = new[0].get("score"), new[j].get("score")
    new[0], new[j] = new[j], new[0]
    new[0]["score"], new[j]["score"] = s0, sj
    out["candidates"] = new
    out["_degraded"] = j
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("candidates", help="input candidates JSONL (--candidates-from format)")
    ap.add_argument("--rates", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    ap.add_argument("--max-rank", type=int, default=10,
                    help="swap the top-1 with a rank drawn from [2, MAX_RANK]")
    ap.add_argument("--out-prefix", default="cands_dose")
    ap.add_argument("--salt", default="dose-v1",
                    help="change this for an independent replication of the sweep")
    ap.add_argument("--gold", default=None,
                    help="mentions JSONL with a 'gold' field; prints Recall@1/@10 per level")
    a = ap.parse_args()

    records = [json.loads(l) for l in open(a.candidates) if l.strip()]
    gold = {}
    if a.gold:
        for line in open(a.gold):
            if line.strip():
                r = json.loads(line)
                gold[(str(r.get("pmid")), int(r.get("start", -1)))] = str(r.get("gold", ""))

    if gold:
        print(f"{'rate':>6}{'n':>8}{'R@1':>8}{'R@10':>8}{'swapped':>9}")
    for rate in a.rates:
        out = [degrade(r, rate, a.max_rank, a.salt) for r in records]
        path = Path(f"{a.out_prefix}_p{rate:g}.jsonl")
        with open(path, "w") as fh:
            for r in out:
                fh.write(json.dumps({k: v for k, v in r.items()
                                     if k != "_degraded"}) + "\n")
        if gold:
            hit1 = hit10 = seen = 0
            for r in out:
                g = gold.get((str(r.get("pmid")), int(r.get("start", -1))))
                if g is None:
                    continue
                ids = [str(c.get("cui") or c.get("mesh_id") or "") for c in r.get("candidates", [])]
                seen += 1
                hit1 += bool(ids and ids[0] == g)
                hit10 += g in ids
            swapped = sum(1 for r in out if "_degraded" in r)
            print(f"{rate:>6g}{seen:>8d}{hit1/seen*100:>8.1f}{hit10/seen*100:>8.1f}"
                  f"{swapped/len(out)*100:>8.1f}%")
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
