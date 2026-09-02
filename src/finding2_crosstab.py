"""
Finding 2 — the deference cross-tabulation.

Tests whether fine-tuning changes WHEN the LLM intervenes (deference) rather than
HOW WELL it decides (competence), and whether the learned restraint is bound to the
training distribution (seen vs unseen concepts).

For a zero-shot and a fine-tuned enriched prediction file (same test set), it reports
intervention precision (fixes/(fixes+breaks)) broken down by:
      confidence band  ×  seen/unseen concept  ×  {zero-shot, fine-tuned}
with document-level bootstrap 95% CIs. "seen" = the gold concept appears in the
fine-tuning training set (--train, an SFT jsonl with meta.gold_id).

Prediction of the deference thesis:
  * on the truly-hard band [0,2), fine-tuning barely moves precision (competence flat);
  * fine-tuning's precision gains sit in the mid/high bands (learned restraint);
  * that gain is realised on SEEN concepts and fails (or reverses) on UNSEEN ones.

Usage:
  python3 src/finding2_crosstab.py preds_zs_biored.jsonl preds_ft_biored.jsonl \
        --train Data/finetune/bc5cdr_train_sft.jsonl --out finding2.json
"""
import json, argparse, random
from collections import defaultdict

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
BLAB = ["[0,2)", "[2,5)", "[5,10)", "[10,20)", "20+"]


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def train_golds(p):
    s = set()
    for l in open(p):
        try:
            s.update(str(json.loads(l)["meta"]["gold_id"]).split("|"))
        except (json.JSONDecodeError, KeyError):
            pass
    return s


def band_of(g):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= g < hi:
            return i
    return None


def seen(r, tg):
    return any(g in tg for g in str(r.get("gold_id", "")).split("|"))


def precision(rows):
    fx = sum(1 for r in rows if r.get("p4_correct") and not r.get("p3_correct"))
    bk = sum(1 for r in rows if r.get("p3_correct") and not r.get("p4_correct"))
    return (fx / (fx + bk) if (fx + bk) else None), fx, bk


def boot_ci(rows, seed=0, nb=1000):
    bd = defaultdict(list)
    for r in rows:
        bd[r.get("pmid")].append(r)
    docs = list(bd)
    if not docs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(nb):
        s = []
        for _ in range(len(docs)):
            s.extend(bd[rng.choice(docs)])
        p = precision(s)[0]
        if p is not None:
            vals.append(p)
    if not vals:
        return None
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def cell(rows):
    p, fx, bk = precision(rows)
    ci = boot_ci(rows) if rows else None
    rate = sum(1 for r in rows if r.get("llm_changed")) / len(rows) if rows else 0
    d = ((sum(bool(r.get("p4_correct")) for r in rows) -
          sum(bool(r.get("p3_correct")) for r in rows)) / len(rows) * 100) if rows else 0
    return {"n": len(rows), "precision": p, "ci": ci, "rate": rate,
            "fixes": fx, "breaks": bk, "delta": d}


def fmt(c):
    if c["precision"] is None:
        return f"n={c['n']:4d}  prec   -"
    ci = f"[{c['ci'][0]*100:.0f},{c['ci'][1]*100:.0f}]" if c["ci"] else "-"
    return f"n={c['n']:4d}  prec {c['precision']*100:3.0f}% {ci:>9}  (+{c['fixes']}/-{c['breaks']})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("zs")
    ap.add_argument("ft")
    ap.add_argument("--train", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tg = train_golds(a.train)
    ZS = [r for r in load(a.zs) if r.get("p3_score_gap") is not None]
    FT = [r for r in load(a.ft) if r.get("p3_score_gap") is not None]
    print(f"train concepts: {len(tg)}  |  zs n={len(ZS)}  ft n={len(FT)}\n")

    out = {}
    for grp in ["seen", "unseen"]:
        print("=" * 78)
        print(f"{grp.upper()} concepts")
        print("=" * 78)
        print(f"{'band':9}{'  zero-shot':38}{'  fine-tuned':38}")
        out[grp] = []
        for i, lab in enumerate(BLAB):
            zc = cell([r for r in ZS if band_of(r["p3_score_gap"]) == i and (seen(r, tg) == (grp == "seen"))])
            fc = cell([r for r in FT if band_of(r["p3_score_gap"]) == i and (seen(r, tg) == (grp == "seen"))])
            print(f"{lab:9}  {fmt(zc):36}  {fmt(fc):36}")
            out[grp].append({"band": lab, "zs": zc, "ft": fc})
        # aggregate
        zc = cell([r for r in ZS if seen(r, tg) == (grp == "seen")])
        fc = cell([r for r in FT if seen(r, tg) == (grp == "seen")])
        print(f"{'ALL':9}  {fmt(zc):36}  {fmt(fc):36}")
        out[grp + "_all"] = {"zs": zc, "ft": fc}
        print()

    # the deference test, stated plainly
    print("=" * 78)
    print("DEFERENCE TEST  (fine-tuning precision change, ft − zs, by band)")
    print("=" * 78)
    print(f"{'band':9}{'seen Δprec':>14}{'unseen Δprec':>16}")
    for i, lab in enumerate(BLAB):
        s = out["seen"][i]; u = out["unseen"][i]
        sd = (s["ft"]["precision"] - s["zs"]["precision"]) * 100 if s["ft"]["precision"] is not None and s["zs"]["precision"] is not None else None
        ud = (u["ft"]["precision"] - u["zs"]["precision"]) * 100 if u["ft"]["precision"] is not None and u["zs"]["precision"] is not None else None
        ss = f"{sd:+.0f}pp" if sd is not None else "-"
        us = f"{ud:+.0f}pp" if ud is not None else "-"
        print(f"{lab:9}{ss:>14}{us:>16}")

    if a.out:
        json.dump(out, open(a.out, "w"), indent=2, default=lambda x: None)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
