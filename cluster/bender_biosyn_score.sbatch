#!/bin/bash
# ==========================================================================
# BioSyn scoring (GPU) -> candidates JSONL for our --candidates-from adapter.
# Runs the FULL hybrid BioSyn (sparse+dense) with the authors' pretrained
# BC5CDR checkpoints on OUR exact eval mentions (disease + chemical routed to
# their own models). Runs in eval_env via the compat shims in run_biosyn.py
# (no separate BioSyn env). Prereqs: cluster/bender_setup_biosyn.sh done + mentions
# file produced in eval_env (see RETRIEVER_CURVE_STEPS.md).
# ==========================================================================
#SBATCH --job-name=biosyn-score
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/biosyn-%j.out

set -e
mkdir -p ~/NEL_LLM_pipeline/logs
module load Python
module load CUDA || true
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/BioSyn
export TOKENIZERS_PARALLELISM=false

# OUR mentions file (produced by evaluate_pipeline --dump-mentions in eval_env)
MENTIONS=~/NEL_LLM_pipeline/mentions_bc5cdr_test.jsonl

python run_biosyn.py \
    --mentions "$MENTIONS" \
    --disease-model  models/biosyn-sapbert-bc5cdr-disease \
    --disease-dict   datasets/bc5cdr-disease/test_dictionary.txt \
    --chemical-model models/biosyn-sapbert-bc5cdr-chemical \
    --chemical-dict  datasets/bc5cdr-chemical/test_dictionary.txt \
    --topk-dict 40 --topk-out 10 --use-cuda \
    --out ~/NEL_LLM_pipeline/cands_biosyn_bc5cdr.jsonl

echo "Done -> ~/NEL_LLM_pipeline/cands_biosyn_bc5cdr.jsonl"
