"""
Seen-vs-unseen breakdown — is a fine-tuned model generalizing or memorizing?

The worry with fine-tuning on BC5CDR: train and test share many frequent
mentions, so high test accuracy could just be a memorized mention->gold lookup
rather than learned disambiguation. This splits the test predictions by whether
the GOLD CONCEPT was present in the fine-tuning data:

  * seen   = gold_id appears in the training set  -> memorization possible
  * unseen = gold_id never in training            -> only generalization helps

If the fine-tuned model improves on UNSEEN concepts too, the gain is real
(the standard zero-shot-entity definition, as in BioLinkerAI / ACL 2025).

Usage:
    python3 src/analyze_seen_unseen.py \
        --train Data/finetune/bc5cdr_train_sft.jsonl \
        preds_0.6b_zeroshot.jsonl preds_0.6b_finetuned.jsonl
"""

import argparse
import json


def train_golds(path):
    ids = set()
    for line in open(path):
        try:
            ids.add(str(json.loads(line)["meta"]["gold_id"]))
        except (json.JSONDecodeError, KeyError):
            continue
    # split pipe-composite gold ids too
    out = set()
    for g in ids:
        out.update(g.split("|"))
    return out


def load_preds(path):
    rows = []
    for line in open(path):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="Data/finetune/bc5cdr_train_sft.jsonl")
    ap.add_argument("preds", nargs="+", help="one or more prediction files")
    ap.add_argument("--phase", choices=["p3", "p4"], default="p4")
    args = ap.parse_args()

    tg = train_golds(args.train)
    print(f"Training set: {len(tg)} distinct gold concepts\n")

    field = f"{args.phase}_correct"
    for path in args.preds:
        rows = load_preds(path)
        seen = [r for r in rows
                if any(g in tg for g in str(r["gold_id"]).split("|"))]
        unseen = [r for r in rows
                  if not any(g in tg for g in str(r["gold_id"]).split("|"))]

        def acc(rs):
            return (sum(bool(r[field]) for r in rs) / len(rs) * 100) if rs else 0.0

        print(f"{path}")
        print(f"  overall: {acc(rows):5.1f}%  (n={len(rows)})")
        print(f"  seen   : {acc(seen):5.1f}%  (n={len(seen)}, {len(seen)/len(rows)*100:.0f}%)")
        print(f"  unseen : {acc(unseen):5.1f}%  (n={len(unseen)}, {len(unseen)/len(rows)*100:.0f}%)")
        print()


if __name__ == "__main__":
    main()
