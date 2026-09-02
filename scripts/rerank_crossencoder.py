#!/usr/bin/env python3
"""A non-LLM re-ranker on the identical candidate lists — the missing baseline.

The paper claims the LLM disambiguation stage buys little behind a strong
retriever. The obvious objection is that we never checked what a *cheap*
re-ranker would have bought on the same lists. This script is that check.

It is deliberately NOT wired into evaluate_pipeline.py. It reads the post-P3
candidate dump (`--dump-candidates`) and the LLM run's own prediction dump, and
writes a prediction dump in the identical schema. Consequences:

  * every mention, every candidate list and every P3 field is byte-identical to
    the LLM run, so the document bootstrap in analyze_robustness.py is exactly
    paired — no re-alignment, no subset, no excuse;
  * the cross-encoder sees the same top-k, the same names and the same
    definitions the LLM prompt contains, so a loss is a loss on the merits;
  * it costs one forward pass per (mention, candidate) pair, no vLLM server.

`p4_top1` becomes the cross-encoder's pick and `llm_changed` its intervention
flag. The field names stay as they are so that every downstream metric —
gain, intervention rate, precision, aggressiveness, selection skill — is
computed by the same code that computes them for the LLMs. Read "llm_changed"
as "the stage changed the retriever's answer".

Usage
-----
    # zero-shot re-ranker
    python3 scripts/rerank_crossencoder.py \
        --cands cands_p3_bc5cdr.jsonl \
        --ref   preds_qwen3-4b-zeroshot_bc5cdr.jsonl \
        --out   preds_zs_bc5cdr_medcpt.jsonl

    # a checkpoint trained by scripts/train_crossencoder.py
    python3 scripts/rerank_crossencoder.py ... \
        --model models/ce_bc5cdr --out preds_zs_bc5cdr_medcpt-ft.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

# MedCPT's cross-encoder is a biomedical re-ranker (query, passage) -> score,
# trained on PubMed search logs. Public, ungated, 109M parameters: about 1/40th
# of the 4B disambiguator, which is the point of the comparison.
DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def build_pair(rec, cand, mode, def_chars):
    """(query, passage) for one candidate.

    query   = the mention in its sentence, which is the evidence the LLM gets
    passage = the concept's preferred label plus its definition
    """
    mention = rec.get("mention", "")
    if mode == "sentence" and rec.get("sentence"):
        query = f"{mention} [SEP] {rec['sentence']}"
    elif mode == "title" and rec.get("title"):
        query = f"{mention} [SEP] {rec['title']}"
    else:
        query = mention
    name = cand.get("name") or cand.get("cui", "")
    definition = (cand.get("definition") or "")[:def_chars]
    passage = f"{name}. {definition}".strip() if definition else name
    return query, passage


@torch.inference_mode()
def score_all(pairs, tok, model, device, batch_size, max_len):
    """Relevance logit per pair, in input order."""
    out = []
    for i in range(0, len(pairs), batch_size):
        chunk = pairs[i:i + batch_size]
        enc = tok([p[0] for p in chunk], [p[1] for p in chunk],
                  truncation=True, padding=True, max_length=max_len,
                  return_tensors="pt").to(device)
        logits = model(**enc).logits
        # num_labels == 1 for a ranking head; a 2-way head is read as P(relevant)
        col = logits[:, 0] if logits.shape[-1] == 1 else logits[:, 1] - logits[:, 0]
        out.extend(col.float().cpu().tolist())
        if (i // batch_size) % 50 == 0:
            print(f"    scored {min(i + batch_size, len(pairs)):>7}/{len(pairs)}",
                  flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", required=True,
                    help="post-P3 candidate dump (evaluate_pipeline.py --dump-candidates)")
    ap.add_argument("--ref", required=True,
                    help="the LLM run's preds dump: supplies gold and every P3 field, "
                         "and fixes the mention set so the comparison is paired")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--context", choices=["sentence", "title", "none"], default="sentence")
    ap.add_argument("--top-k", type=int, default=10,
                    help="candidates to score; MUST equal the run's --llm-top-k (default 10)")
    ap.add_argument("--definition-chars", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=512,
                    help="MedCPT-Cross-Encoder is BERT-base: 512 is its hard limit. Do not\n"
                         "lower it to save time -- truncating the definition is exactly how\n"
                         "a baseline gets handicapped without anyone noticing.")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  model   {args.model}")
    print(f"  device  {dev}   context={args.context}   top-k={args.top_k}")

    ref = read_jsonl(args.ref)
    cands = read_jsonl(args.cands)
    print(f"  ref     {len(ref):,} mentions   cands {len(cands):,} mentions")

    # Index the candidate dump by offset, falling back to lowercased text, the
    # same two-key scheme evaluate_pipeline.py uses for --candidates-from.
    by_off, by_txt = {}, {}
    for c in cands:
        pmid = str(c.get("pmid", ""))
        try:
            st = int(c.get("start", -1))
        except (TypeError, ValueError):
            st = -1
        if st >= 0:
            by_off[(pmid, st)] = c
        by_txt.setdefault((pmid, (c.get("mention") or "").lower().strip()), c)

    # One row per REFERENCE mention, in the reference's order. Anything the
    # candidate dump is missing is reported, never silently dropped: a shorter
    # output would quietly change the evaluated subset.
    rows, pairs, spans, missing, p3_mismatch = [], [], [], 0, 0
    for r in ref:
        pmid = str(r.get("pmid", ""))
        rec = by_txt.get((pmid, (r.get("mention") or "").lower().strip()))
        if rec is None:
            missing += 1
            continue
        cl = (rec.get("candidates") or [])[:args.top_k]
        if not cl:
            missing += 1
            continue
        if cl[0].get("cui") != r.get("p3_top1"):
            # The dump and the reference run did not come from the same
            # retriever configuration. Counting these is the only honest guard.
            p3_mismatch += 1
        start = len(pairs)
        pairs.extend(build_pair(rec, c, args.context, args.definition_chars) for c in cl)
        spans.append((start, len(pairs)))
        rows.append((r, cl))

    print(f"  matched {len(rows):,} mentions -> {len(pairs):,} pairs to score")
    if missing:
        print(f"  !! {missing} reference mentions had no candidate list and are skipped")
    if p3_mismatch:
        print(f"  !! {p3_mismatch} mentions where the dump's rank-1 differs from the "
              f"reference's p3_top1 -- the two runs are NOT the same retriever "
              f"configuration. Fix that before using the output.", file=sys.stderr)
    if not rows:
        sys.exit("nothing to score")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model).to(dev).eval()
    scores = score_all(pairs, tok, model, dev, args.batch_size, args.max_len)

    n_changed = n_fix = n_break = 0
    with open(args.out, "w") as fh:
        for (r, cl), (lo, hi) in zip(rows, spans):
            best = max(range(lo, hi), key=lambda j: scores[j])
            pick = cl[best - lo]["cui"]
            gold = str(r.get("gold_id", ""))
            # Composite gold ids are pipe-joined; any match counts, exactly as
            # in the pipeline's expanded_gold_ids test.
            gold_set = set(gold.split("|")) if gold else set()
            correct = pick in gold_set
            changed = pick != r.get("p3_top1")
            n_changed += changed
            n_fix += changed and correct and not r.get("p3_correct")
            n_break += changed and r.get("p3_correct") and not correct
            out = dict(r)                      # keep every P3 field verbatim
            out["p4_top1"] = pick
            out["p4_correct"] = bool(correct)
            out["llm_changed"] = bool(changed)
            out["reranker"] = Path(args.model).name
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")

    n = len(rows)
    p3 = sum(bool(r.get("p3_correct")) for r, _ in rows) / n * 100
    p4 = p3 + (n_fix - n_break) / n * 100
    prec = n_fix / (n_fix + n_break) * 100 if (n_fix + n_break) else float("nan")
    print(f"\n  wrote {args.out}  ({n:,} rows)")
    print(f"  P3 {p3:5.1f}   P4 {p4:5.1f}   gain {p4 - p3:+.2f}")
    print(f"  interventions {n_changed} ({n_changed / n * 100:.1f} %)   "
          f"fixes {n_fix}  breaks {n_break}  precision {prec:.0f} %")


if __name__ == "__main__":
    main()
