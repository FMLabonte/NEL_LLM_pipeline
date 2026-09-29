#!/bin/bash
# ==========================================================================
# Named-retriever curve points (Figure 1), zero-shot Qwen3-4B + our LLM stage:
#   (1) BM25-only        -> native Okapi BM25 over MeSH labels (--bm25-only)
#   (2) BioSyn           -> our LLM stage on BioSyn candidates (--candidates-from)
# Both write per-mention predictions so finding1_analysis.py can compute the
# (Recall@1, LLM-gain) point and the danger-zone bands on each retriever.
# BC5CDR test, same flags as the SapBERT-only run (fair comparison).
# ==========================================================================
#SBATCH --job-name=nel-curve
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/curve-%j.out

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3-4B"; SERVED="qwen3-4b-zeroshot"; PORT=8000
vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
    > logs/vllm-curve-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done

COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy \
        --llm-top-k 10 --prompt-version v8 --temperature 0 --max-tokens 256 \
        --structured-output --keep-duplicates --dataset bc5cdr --split test"

echo "=== (1) BM25-only ==="
python3 src/evaluate_pipeline.py $COMMON --bm25-only \
        --dump-predictions preds_zs_bc5cdr_bm25.jsonl

if [ -f cands_biosyn_bc5cdr.jsonl ]; then
    echo "=== (2) BioSyn (candidates-from) ==="
    python3 src/evaluate_pipeline.py $COMMON \
            --candidates-from cands_biosyn_bc5cdr.jsonl \
            --dump-predictions preds_zs_bc5cdr_biosyn.jsonl
else
    echo "=== (2) BioSyn SKIPPED: cands_biosyn_bc5cdr.jsonl not found."
    echo "    Run cluster/bender_setup_biosyn.sh + dump mentions + cluster/bender_biosyn_score.sbatch first,"
    echo "    then re-submit this job to add the BioSyn point."
fi

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_zs_bc5cdr_{bm25,biosyn}.jsonl"
