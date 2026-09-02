#!/bin/bash
# ==========================================================================
# MedMentions +EMBEDDING (SapBERT/FAISS ueber UMLS) auf DEMSELBEN 5000er-Sample
# wie der string-only-Lauf (--limit-seed 42). Gibt den zweiten, STAERKEREN
# Retriever-Punkt fuer den Within-Dataset-Sweep:
#   string-only  -> preds_zs_medmentions.jsonl       (schon vorhanden)
#   +embedding   -> preds_zs_medmentions_emb.jsonl   (dieser Lauf)
# Voraussetzung: bender_build_umls_faiss.sbatch lief einmal durch.
# ==========================================================================
#SBATCH --job-name=nel-medm-emb
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/medm-emb-%j.out

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3-4B"
SERVED="qwen3-4b-zeroshot"
PORT=8000

vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
    > logs/vllm-medm-emb-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done

# identisch zum string-only-Lauf, NUR + --embedding
python3 src/evaluate_pipeline.py --model "$SERVED" \
    --base-url "http://localhost:$PORT/v1" --api-key dummy \
    --umls-index --embedding --no-expansion \
    --llm-top-k 10 --prompt-version v8 --dataset medmentions --split test \
    --temperature 0 --max-tokens 256 --structured-output --keep-duplicates \
    --limit 5000 --limit-random --limit-seed 42 \
    --dump-predictions preds_zs_medmentions_emb.jsonl

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_zs_medmentions_emb.jsonl"
