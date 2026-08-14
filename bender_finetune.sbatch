#!/bin/bash
# ==========================================================================
# LoRA-SFT des Phase-4-Disambiguators auf Bender (Uni Bonn GPU-Cluster).
#
# Einreichen:  sbatch bender_finetune.sbatch
# Status:      squeue --me      | Log:  tail -f logs/nel-lora-<jobid>.out
# Effizienz:   seff <jobid>     (nach Ende — hat es die GPU wirklich genutzt?)
# ==========================================================================
#SBATCH --job-name=nel-lora
#SBATCH --partition=A40short       # A40, 8h Limit; A40devel (1h) zum Testen
#SBATCH --gpus=1                   # 1 GPU (CPUs + RAM kommen anteilig automatisch)
#SBATCH --time=04:00:00            # Puffer; 0.6B ueber den vollen Satz ~1.5-2.5h
#SBATCH --output=logs/nel-lora-%j.out

set -e
mkdir -p logs out

# ── Umgebung ──────────────────────────────────────────────────────────────
module load Python                 # Basis-Python fuer die venv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# venv-Python DIREKT ansprechen (umgeht die flakige 'source activate'-PATH-Sache)
PY=~/NEL_LLM_pipeline/ft_env/bin/python3
cd ~/NEL_LLM_pipeline

# ── Stufe 1: Qwen3-0.6B (die "lernt es ueberhaupt?"-Probe) ─────────────────
$PY src/finetune_lora.py \
  --model Qwen/Qwen3-0.6B \
  --train Data/finetune/bc5cdr_train_sft.jsonl \
  --eval  Data/finetune/bc5cdr_dev_sft.jsonl \
  --out   out/qwen3-0.6b-nel-lora \
  --epochs 3 --batch-size 2 --grad-accum 16 --lr 2e-4 \
  --merge

# ── Stufe 2 (spaeter, wenn 0.6B zeigt "es lernt"): Qwen3-4B mit QLoRA ───────
# A100 nehmen (80GB) fuer mehr Luft: --partition=A100short oben setzen.
# $PY src/finetune_lora.py \
#   --model Qwen/Qwen3-4B \
#   --train Data/finetune/bc5cdr_train_sft.jsonl \
#   --eval  Data/finetune/bc5cdr_dev_sft.jsonl \
#   --out   out/qwen3-4b-nel-lora \
#   --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-4 \
#   --load-in-4bit --merge

echo "Fertig. Adapter in out/. Serviere out/*-merged und evaluiere mit evaluate_pipeline.py."
