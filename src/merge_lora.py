"""
Merge a (Q)LoRA adapter into its base model -> a standalone HF model.

Needed for QLoRA runs: you cannot merge an adapter into a 4-bit base in place,
so we reload the base in fp16, apply the adapter, merge, and save. Also works
for plain (non-quantized) LoRA adapters.

Usage (on Bender, base model already cached):
    module load Python
    ~/NEL_LLM_pipeline/ft_env/bin/python3 src/merge_lora.py \
        --base Qwen/Qwen3-4B \
        --adapter out/qwen3-4b-nel-lora \
        --out out/qwen3-4b-nel-lora-merged
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="Base model id, e.g. Qwen/Qwen3-4B")
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (out/...-lora)")
    ap.add_argument("--out", required=True, help="Output dir for the merged model")
    args = ap.parse_args()

    print(f"Loading base {args.base} in fp16 (CPU)...")
    base = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.float16, device_map="cpu",
    )
    print(f"Applying + merging adapter {args.adapter}...")
    model = PeftModel.from_pretrained(base, args.adapter)
    model = model.merge_and_unload()

    model.save_pretrained(args.out)
    AutoTokenizer.from_pretrained(args.adapter).save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")


if __name__ == "__main__":
    main()
