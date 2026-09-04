#!/usr/bin/env python3
"""Does GRPO have a gradient on this task? Measured before writing any RL code.

GRPO normalises the reward *within* a group of rollouts:

    A_i = (r_i - mean(r)) / std(r)

If every rollout in a group earns the same reward, std is zero, every advantage
is zero, and the group contributes nothing to the update. That is not a tuning
problem, it is the objective having no signal, and it is a real risk here:

  * constrained decoding pins the answer to one of ten candidate ids, so the
    action space is tiny;
  * the retriever is already right on ~83 % of BC5CDR mentions, so most groups
    should agree on the easy answer;
  * where the gold concept was never retrieved, *no* rollout can be correct,
    so the group is unanimously wrong and equally useless.

This script measures the collapse rate directly, with no RL involved: sample G
completions per mention at temperature T through the same endpoint, the same v8
prompt and the same enum-constrained decoder the paper uses, then count how
often a group is non-degenerate.

Four outcomes per group, and only the last one trains anything:

    all-correct           the model already knows this one
    all-wrong, gold in    hard, but learnable -- one lucky rollout would help
    all-wrong, no gold    unlearnable, must be filtered out of the RL set
    mixed                 non-degenerate: this is the effective training set

The per-band breakdown is the interesting part. If the mixed groups concentrate
in the mid-confidence band, the paper's danger zone is exactly where GRPO would
put its gradient, and the two halves of the project connect.

    python3 scripts/grpo_pilot.py --cands cands_p3_bc5cdr.jsonl \
        --ref preds_qwen3-4b-zeroshot_bc5cdr.jsonl \
        --model "$SERVED" --base-url http://localhost:8000/v1 \
        --group-size 8 --temperatures 0.7 1.0 --limit 800
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "llm-disambiguation"))

from prompts import get_prompt_config, build_user_prompt   # noqa: E402

BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
BLAB = ["[0,2)", "[2,5)", "[5,10)", "[10,20)", "20+"]


def band_of(gap):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= gap < hi:
            return i
    return len(BANDS) - 1


class Cand:
    """The attribute surface build_user_prompt() reads, nothing more.

    Rebuilt from the dump rather than from the MeSH index on purpose: the dump
    is what the cross-encoder scores too, so both baselines and this pilot are
    provably looking at the same candidate records.
    """
    __slots__ = ("mesh_id", "preferred_label", "definition", "synonyms", "score")

    def __init__(self, d):
        self.mesh_id = d.get("cui", "")
        self.preferred_label = d.get("name", "") or ""
        self.definition = d.get("definition", "") or ""
        self.synonyms = list(d.get("synonyms") or [])
        self.score = float(d.get("score", 0.0))


def id_schema(cands):
    """The same per-request enum the pipeline builds, so the model physically
    cannot answer outside the candidate list."""
    ids = list(dict.fromkeys(c.mesh_id for c in cands))
    return {"type": "json_schema",
            "json_schema": {"name": "entity_link", "strict": True,
                            "schema": {"type": "object",
                                       "properties": {"mesh_id": {"type": "string",
                                                                  "enum": ids}},
                                       "required": ["mesh_id"],
                                       "additionalProperties": False}}}


def classify(rewards, gold_in_list):
    if len(set(rewards)) > 1:
        return "mixed"
    if rewards[0] == 1:
        return "all_correct"
    return "all_wrong_gold_in" if gold_in_list else "all_wrong_no_gold"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", required=True)
    ap.add_argument("--docs", default=None,
                    help="companion <cands>.docs.jsonl written by --dump-candidates. "
                         "Without it the prompt uses the sentence instead of the "
                         "64-word window v8 actually sees, and the pilot no longer "
                         "measures the deployed prompt.")
    ap.add_argument("--ref", required=True,
                    help="preds dump of the zero-shot run: gold, p3 fields and the "
                         "confidence band come from here")
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default="dummy")
    ap.add_argument("--prompt-version", default="v8")
    ap.add_argument("--group-size", type=int, default=8, help="G rollouts per mention")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[0.7, 1.0])
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--llm-top-k", type=int, default=10)
    ap.add_argument("--limit", type=int, default=800,
                    help="mentions to sample; a pilot does not need the full corpus")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    cfg = get_prompt_config(args.prompt_version)

    ref = {}
    for line in open(args.ref):
        if line.strip():
            r = json.loads(line)
            if r.get("p3_score_gap") is not None:
                ref[(str(r["pmid"]), (r.get("mention") or "").lower().strip())] = r

    docs_path = args.docs or (str(args.cands).replace(".jsonl", "") + ".docs.jsonl")
    docs = {}
    if Path(docs_path).exists():
        for line in open(docs_path):
            if line.strip():
                d = json.loads(line)
                docs[str(d.get("pmid", ""))] = d
        print(f"  document texts: {len(docs):,} from {docs_path}")
    else:
        print(f"  !! {docs_path} not found -- falling back to the stored sentence.\n"
              f"     v8 windows the full document (context_words=64), so this is a "
              f"DIFFERENT prompt from the deployed one. Re-run the dump with the "
              f"current evaluate_pipeline.py before trusting these numbers.")

    items = []
    for line in open(args.cands):
        if not line.strip():
            continue
        c = json.loads(line)
        key = (str(c.get("pmid", "")), (c.get("mention") or "").lower().strip())
        r = ref.get(key)
        if r is None or not c.get("candidates"):
            continue
        items.append((c, r))

    # Sample by DOCUMENT so the pilot keeps whole abstracts, the same unit the
    # bootstrap resamples elsewhere in the paper.
    by_doc = defaultdict(list)
    for it in items:
        by_doc[str(it[0].get("pmid", ""))].append(it)
    doc_ids = sorted(by_doc)          # NOT `docs` -- that holds the texts
    random.Random(args.seed).shuffle(doc_ids)
    chosen, n = [], 0
    for d in doc_ids:
        if args.limit and n >= args.limit:
            break
        chosen.extend(by_doc[d]); n += len(by_doc[d])
    print(f"  {len(items):,} matched mentions -> pilot on {len(chosen):,} "
          f"from {len({str(c.get('pmid','')) for c, _ in chosen})} documents")
    if not chosen:
        sys.exit("no mentions matched -- are --cands and --ref from the same run?")

    report = {"model": args.model, "group_size": args.group_size,
              "prompt_version": args.prompt_version, "n_mentions": len(chosen),
              "temperatures": {}}

    for T in args.temperatures:
        print(f"\n=== temperature {T}, G={args.group_size} ===", flush=True)
        cnt = Counter()
        band_cnt = defaultdict(Counter)
        uniq_hist = Counter()
        mixed_gap, errors = [], 0

        for idx, (c, r) in enumerate(chosen, 1):
            cands = [Cand(d) for d in c["candidates"][:args.llm_top_k]]
            doc = docs.get(str(c.get("pmid", "")))
            if doc:
                ctx, title, start = doc["full_text"], doc.get("title", ""), c.get("start", -1)
            else:
                ctx, title, start = c.get("sentence", ""), c.get("title", ""), -1
            prompt = build_user_prompt(cfg, c.get("mention", ""), cands,
                                       ctx, title, mention_start=start)
            msgs = [{"role": "system", "content": cfg.system_prompt},
                    {"role": "user", "content": prompt}]
            kw = dict(model=args.model, messages=msgs, temperature=T,
                      max_tokens=args.max_tokens, response_format=id_schema(cands))
            # vLLM refuses n>1 under greedy sampling ("n must be 1 when using
            # greedy sampling"). The T=0 control is still worth running -- it is
            # what catches a server that samples when it was told not to -- so
            # issue G separate calls instead of one batched one.
            try:
                if T <= 0.0:
                    resp = [client.chat.completions.create(**kw)
                            for _ in range(args.group_size)]
                    choices = [rr.choices[0] for rr in resp]
                else:
                    choices = client.chat.completions.create(
                        **kw, n=args.group_size).choices
            except Exception as e:
                errors += 1
                if errors <= 3:
                    print(f"    call failed ({type(e).__name__}): {str(e)[:120]}")
                continue

            gold = set(str(r.get("gold_id", "")).split("|"))
            picks = []
            for ch in choices:
                try:
                    picks.append(str(json.loads(ch.message.content).get("mesh_id", "")))
                except Exception:
                    picks.append("")
            rewards = [1 if p in gold else 0 for p in picks]
            kind = classify(rewards, bool(r.get("gold_in_list")))
            b = band_of(float(r["p3_score_gap"]))
            cnt[kind] += 1
            band_cnt[b][kind] += 1
            uniq_hist[len(set(picks))] += 1
            if kind == "mixed":
                mixed_gap.append(float(r["p3_score_gap"]))
            if idx % 100 == 0:
                m = cnt["mixed"] / max(1, sum(cnt.values())) * 100
                print(f"    {idx}/{len(chosen)}  mixed so far {m:.1f} %", flush=True)

        tot = sum(cnt.values())
        if not tot:
            print("  every call failed -- is the endpoint up?"); continue

        print(f"\n  {tot} groups" + (f"  ({errors} failed calls)" if errors else ""))
        for k, lab in [("mixed", "mixed  -> TRAINS"),
                       ("all_correct", "all correct"),
                       ("all_wrong_gold_in", "all wrong, gold in list"),
                       ("all_wrong_no_gold", "all wrong, gold never retrieved")]:
            print(f"    {lab:34} {cnt[k]:6}  {cnt[k]/tot*100:5.1f} %")

        learnable = tot - cnt["all_wrong_no_gold"]
        print(f"\n    effective gradient coverage: {cnt['mixed']}/{tot} = "
              f"{cnt['mixed']/tot*100:.1f} % of all mentions, "
              f"{cnt['mixed']/max(1,learnable)*100:.1f} % of the learnable ones")

        print(f"\n    distinct answers per group of {args.group_size}:")
        for u in sorted(uniq_hist):
            print(f"      {u:2}  {uniq_hist[u]:6}  {uniq_hist[u]/tot*100:5.1f} %")

        print(f"\n    mixed-group rate per retriever-confidence band:")
        print("      " + "".join(f"{l:>10}" for l in BLAB))
        row_n = [sum(band_cnt[i].values()) for i in range(len(BLAB))]
        row_m = [band_cnt[i]["mixed"] for i in range(len(BLAB))]
        print("      " + "".join(
            f"{(m/n*100 if n else 0):9.1f}%" for m, n in zip(row_m, row_n)))
        print("      " + "".join(f"{('n=' + str(n)):>10}" for n in row_n))

        report["temperatures"][str(T)] = {
            "counts": dict(cnt), "n": tot, "failed_calls": errors,
            "mixed_share": cnt["mixed"] / tot,
            "mixed_share_of_learnable": cnt["mixed"] / max(1, learnable),
            "unique_answers_hist": dict(uniq_hist),
            "per_band": {BLAB[i]: {"n": row_n[i], "mixed": row_m[i]}
                         for i in range(len(BLAB))},
        }

    out = args.out or f"grpo_pilot_{Path(args.cands).stem}.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\n  wrote {out}")
    print("\n  Reading it: a mixed share in the low single digits means GRPO spends "
          "\n  almost every rollout on groups with zero advantage. Raising the "
          "\n  temperature buys diversity but also noise; the per-band row says "
          "\n  whether the diversity lands where the errors are.")


if __name__ == "__main__":
    main()
