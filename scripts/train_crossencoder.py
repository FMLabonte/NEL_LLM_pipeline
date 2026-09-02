#!/usr/bin/env python3
"""Train the cross-encoder baseline on the same data the LoRA model saw.

A zero-shot re-ranker is the weak form of the baseline: if it loses, a reviewer
answers "you did not train it". This trains one on the BC5CDR training split
with the *same objective the LLM is given* — pick one concept out of the k
candidates the retriever produced — so the comparison in §6 is fine-tuned LLM
against fine-tuned re-ranker, not against a handicap.

Objective: listwise softmax cross-entropy over each candidate list. Mentions
whose gold is not in the list carry no learning signal (nothing correct to pick)
and are dropped, which is the same rule the SFT dump uses.

The split is by DOCUMENT, never by mention: two mentions of the same concept in
one abstract share context, and splitting them across train/val leaks.

    python3 scripts/train_crossencoder.py \
        --cands cands_p3_bc5cdr_train.jsonl \
        --out models/ce_bc5cdr --epochs 2 --seed 1
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

DEFAULT_MODEL = "ncbi/MedCPT-Cross-Encoder"


def read_jsonl(path):
    with open(path) as fh:
        return [json.loads(l) for l in fh if l.strip()]


def make_groups(recs, top_k, def_chars, context):
    """One training group per mention: k (query, passage) pairs + the gold index."""
    groups = []
    for r in recs:
        cl = (r.get("candidates") or [])[:top_k]
        if len(cl) < 2:
            continue
        gold = set(str(r.get("gold", "")).split("|")) if r.get("gold") else set()
        idx = next((i for i, c in enumerate(cl) if c.get("cui") in gold), None)
        if idx is None:
            continue                      # nothing correct to pick: no signal
        mention = r.get("mention", "")
        if context == "sentence" and r.get("sentence"):
            query = f"{mention} [SEP] {r['sentence']}"
        elif context == "title" and r.get("title"):
            query = f"{mention} [SEP] {r['title']}"
        else:
            query = mention
        passages = []
        for c in cl:
            name = c.get("name") or c.get("cui", "")
            d = (c.get("definition") or "")[:def_chars]
            passages.append(f"{name}. {d}".strip() if d else name)
        groups.append({"pmid": str(r.get("pmid", "")), "query": query,
                       "passages": passages, "label": idx})
    return groups


def split_by_document(groups, val_frac, seed):
    docs = sorted({g["pmid"] for g in groups})
    random.Random(seed).shuffle(docs)
    n_val = max(1, int(len(docs) * val_frac))
    val_docs = set(docs[:n_val])
    tr = [g for g in groups if g["pmid"] not in val_docs]
    va = [g for g in groups if g["pmid"] in val_docs]
    return tr, va


def collate(batch, tok, max_len):
    q, p, sizes = [], [], []
    for g in batch:
        q.extend([g["query"]] * len(g["passages"]))
        p.extend(g["passages"])
        sizes.append(len(g["passages"]))
    enc = tok(q, p, truncation=True, padding=True, max_length=max_len, return_tensors="pt")
    return enc, torch.tensor(sizes), torch.tensor([g["label"] for g in batch])


def listwise_loss(logits, sizes, labels):
    """Softmax over each candidate list, cross-entropy against the gold slot.

    Lists have different lengths, so they are sliced rather than reshaped;
    padding them to a rectangle would put probability mass on slots that do
    not exist.
    """
    out, off = [], 0
    for i, k in enumerate(sizes.tolist()):
        out.append(F.cross_entropy(logits[off:off + k].unsqueeze(0), labels[i:i + 1]))
        off += k
    return torch.stack(out).mean()


@torch.inference_mode()
def evaluate(model, loader, dev):
    hit = tot = 0
    for enc, sizes, labels in loader:
        enc = {k: v.to(dev) for k, v in enc.items()}
        lg = model(**enc).logits
        col = lg[:, 0] if lg.shape[-1] == 1 else lg[:, 1] - lg[:, 0]
        off = 0
        for i, k in enumerate(sizes.tolist()):
            hit += int(col[off:off + k].argmax().item() == labels[i].item())
            off += k
            tot += 1
    return hit / tot * 100 if tot else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cands", required=True, help="candidate dump of the TRAIN split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--context", choices=["sentence", "title", "none"], default="sentence")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--definition-chars", type=int, default=400)
    ap.add_argument("--max-len", type=int, default=512,
                    help="MedCPT-Cross-Encoder is BERT-base: 512 is its hard limit. Do not\n"
                         "lower it to save time -- truncating the definition is exactly how\n"
                         "a baseline gets handicapped without anyone noticing.")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--group-batch", type=int, default=8, help="candidate LISTS per step")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    recs = read_jsonl(args.cands)
    groups = make_groups(recs, args.top_k, args.definition_chars, args.context)
    tr, va = split_by_document(groups, args.val_frac, args.seed)
    print(f"  {len(recs):,} mentions -> {len(groups):,} trainable lists "
          f"({len(groups)/max(1,len(recs))*100:.0f} % have the gold in the list)")
    print(f"  train {len(tr):,} lists / val {len(va):,} lists   device {dev}   seed {args.seed}")

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=1).to(dev)

    cl = lambda b: collate(b, tok, args.max_len)
    dl_tr = DataLoader(tr, batch_size=args.group_batch, shuffle=True, collate_fn=cl)
    dl_va = DataLoader(va, batch_size=args.group_batch, shuffle=False, collate_fn=cl)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(dl_tr) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)
    scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

    print(f"  val accuracy before training: {evaluate(model, dl_va, dev):.1f} %")
    best, best_ep = -1.0, -1
    for ep in range(1, args.epochs + 1):
        model.train()
        run = 0.0
        for step, (enc, sizes, labels) in enumerate(dl_tr, 1):
            enc = {k: v.to(dev) for k, v in enc.items()}
            labels = labels.to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=(dev == "cuda")):
                lg = model(**enc).logits
                col = lg[:, 0] if lg.shape[-1] == 1 else lg[:, 1] - lg[:, 0]
                loss = listwise_loss(col.float(), sizes, labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            run += loss.item()
            if step % 100 == 0:
                print(f"    epoch {ep}  step {step}/{len(dl_tr)}  loss {run/step:.4f}",
                      flush=True)
        model.eval()
        acc = evaluate(model, dl_va, dev)
        print(f"  epoch {ep}: train loss {run/max(1,len(dl_tr)):.4f}   val accuracy {acc:.1f} %")
        if acc > best:
            best, best_ep = acc, ep
            Path(args.out).mkdir(parents=True, exist_ok=True)
            model.save_pretrained(args.out); tok.save_pretrained(args.out)
            print(f"    saved -> {args.out}")

    print(f"\n  best val accuracy {best:.1f} % (epoch {best_ep}) -> {args.out}")


if __name__ == "__main__":
    main()
