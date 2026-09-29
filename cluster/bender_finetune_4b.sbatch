#!/bin/bash
# ==========================================================================
# Stufe 2: Qwen3-4B mit QLoRA (4-bit) auf Bender.
# Vorher einmal auf dem Login-Node das Modell cachen:
#     huggingface-cli download Qwen/Qwen3-4B
# Einreichen:  sbatch cluster/bender_finetune_4b.sbatch
# ==========================================================================
#SBATCH --job-name=nel-lora-4b
#SBATCH --partition=A40short       # 48GB reicht fuer 4B-QLoRA; bei OOM: A100short
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/nel-lora-4b-%j.out

set -e
mkdir -p logs out
module load Python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

PY=~/NEL_LLM_pipeline/ft_env/bin/python3
cd ~/NEL_LLM_pipeline

$PY src/finetune_lora.py \
  --model Qwen/Qwen3-4B \
  --train Data/finetune/bc5cdr_train_sft.jsonl \
  --eval  Data/finetune/bc5cdr_dev_sft.jsonl \
  --out   out/qwen3-4b-nel-lora \
  --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-4 \
  --load-in-4bit --merge

echo "Fertig. Merge in out/qwen3-4b-nel-lora-merged."
