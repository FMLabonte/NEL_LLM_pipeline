"""
LoRA / QLoRA SFT for the Phase-4 disambiguator.
===============================================

Fine-tunes a small open model on the chat-format data produced by
`evaluate_pipeline.py --dump-finetune`. Trains ONLY on the assistant answer
(completion-only loss) — the prompt is long (10 candidates + definitions) and
the target is one short JSON line, so training on the prompt would waste almost
all the signal.

Staged plan (Frederik): start with Qwen3-0.6B as a "does it even learn the
convention?" probe, then 4B, then larger. For 0.6B plain LoRA in bf16 is enough;
for 4B+ add --load-in-4bit (QLoRA) to fit on one GPU.

Version note: TRL's API moves fast. Written for the pinned versions in
bender_finetune.sbatch (transformers>=4.46, trl>=0.12, peft>=0.13). If your TRL
differs, the 3 lines flagged with [TRL] may need adjusting.

Run (local smoke test):
    python3 src/finetune_lora.py --model Qwen/Qwen3-0.6B --epochs 1 --limit 200

Run (real):
    python3 src/finetune_lora.py --model Qwen/Qwen3-0.6B \
        --train Data/finetune/bc5cdr_train_sft.jsonl \
        --eval  Data/finetune/bc5cdr_dev_sft.jsonl \
        --out out/qwen3-0.6b-nel-lora
"""

import argparse
import os

# Reduce CUDA fragmentation. Set before torch initializes the allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
    EarlyStoppingCallback, set_seed,
)
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B",
                    help="Base model. Staged: Qwen3-0.6B -> Qwen3-4B -> larger.")
    ap.add_argument("--train", default="Data/finetune/bc5cdr_train_sft.jsonl")
    ap.add_argument("--eval", default="Data/finetune/bc5cdr_dev_sft.jsonl")
    ap.add_argument("--out", default="out/qwen3-0.6b-nel-lora")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    # Small per-device batch: Qwen's ~152k-token vocab makes the loss logits
    # tensor (batch × seq × vocab) the memory bottleneck, not the 0.6B weights.
    # batch=8 × seq=4096 OOMs even a 48GB A40; batch=2 leaves plenty of headroom.
    # grad-accum keeps the effective batch at 2×16=32.
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--max-seq-len", type=int, default=4096,
                    help="Raise if many examples truncate (see warning at start), "
                         "or rebuild data with --max-definition-len 300.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--patience", type=int, default=3,
                    help="Early-stopping patience on eval_loss.")
    ap.add_argument("--load-in-4bit", action="store_true",
                    help="QLoRA (4-bit). Unnecessary for 0.6B; use for 4B+.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Use only the first N train examples (smoke test).")
    ap.add_argument("--eval-limit", type=int, default=500,
                    help="Use only the first N dev examples for during-training "
                         "eval / early stopping (default 500). The full dev set "
                         "(~5k) makes each eval take ~8 min — a representative "
                         "subset is enough for the early-stopping signal. Final, "
                         "proper evaluation is done separately via evaluate_pipeline.")
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed for weight init, data order and shuffling. Vary it "
                         "to measure how much of a fine-tuning effect is run-to-run "
                         "noise; the paper reports three seeds.")
    ap.add_argument("--merge", action="store_true",
                    help="After training, also save a merged full model for "
                         "serving (only without --load-in-4bit).")
    args = ap.parse_args()
    # Seeds python, numpy and torch before anything touches a generator, so a
    # run is reproducible from the command line alone.
    set_seed(args.seed)
    print(f"seed = {args.seed}")

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_kwargs = dict(torch_dtype=torch.bfloat16, device_map="auto")
    if args.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    model.config.use_cache = False

    # ── Data: chat JSONL -> plain text via the model's chat template ──
    ds = load_dataset("json", data_files={"train": args.train, "eval": args.eval})
    if args.limit:
        ds["train"] = ds["train"].select(range(min(args.limit, len(ds["train"]))))
    if args.eval_limit:
        ds["eval"] = ds["eval"].select(range(min(args.eval_limit, len(ds["eval"]))))

    def to_text(ex):
        return {"text": tok.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False)}
    ds = ds.map(to_text, remove_columns=ds["train"].column_names)

    # Sanity: how many examples exceed max_seq_len (would drop the answer)?
    over = sum(len(tok(t["text"])["input_ids"]) > args.max_seq_len
               for t in ds["train"].select(range(min(500, len(ds["train"])))))
    if over:
        print(f"  WARNING: ~{over}/500 sampled train examples exceed "
              f"--max-seq-len {args.max_seq_len}. Raise it or shorten definitions.")

    # ── Completion-only loss: only the assistant turn contributes. ──
    # Qwen uses ChatML; the assistant answer starts after this marker.
    collator = DataCollatorForCompletionOnlyLM(
        response_template="<|im_start|>assistant\n", tokenizer=tok,   # [TRL]
    )

    peft_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=20,
        eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False,
        max_seq_length=args.max_seq_len,        # [TRL] older TRL: max_length
        dataset_text_field="text",
        packing=False, report_to="none",
        seed=args.seed, data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds["train"], eval_dataset=ds["eval"],
        data_collator=collator, peft_config=peft_cfg,
        processing_class=tok,                    # [TRL] older TRL: tokenizer=tok
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f"\nLoRA adapter saved to {args.out}")

    if args.merge:
        merged_dir = args.out.rstrip("/") + "-merged"
        if args.load_in_4bit:
            # Can't merge into a 4-bit base in place: reload the base in fp16,
            # apply the just-saved adapter, then merge.
            del model, trainer
            torch.cuda.empty_cache()
            base = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.float16, device_map="cpu",
            )
            merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
        else:
            merged = trainer.model.merge_and_unload()
        merged.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"Merged full model saved to {merged_dir} (ready to serve)")


if __name__ == "__main__":
    main()
