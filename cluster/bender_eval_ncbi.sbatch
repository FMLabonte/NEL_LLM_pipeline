#!/bin/bash
# ==========================================================================
# NCBI-Disease: HELD-OUT MeSH-Datensatz fuer die Cross-Domain-Deferenz.
# Fährt BEIDE Modelle (zero-shot + BC5CDR-finetuned) auf NCBI-Test, damit wir
# die Kreuztabelle (Precision x Band x seen/unseen x zs/ft) out-of-domain testen.
#   zero-shot -> preds_zs_ncbi.jsonl
#   finetuned -> preds_ft_ncbi.jsonl
# Voraussetzung: Data/NCBI/NCBItestset_corpus.txt liegt auf Bender.
# ==========================================================================
#SBATCH --job-name=nel-ncbi
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/ncbi-%j.out

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false
PORT=8000

run_eval () {
    local MODEL="$1" SERVED="$2" OUT="$3"
    echo "==================== $SERVED ===================="
    vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
        --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
        > "logs/vllm-ncbi-${SERVED}-${SLURM_JOB_ID}.out" 2>&1 &
    local PID=$!
    echo "Warte auf vLLM ($SERVED)..."
    for i in $(seq 1 120); do
        curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "up."; break; }
        sleep 10
    done
    python3 src/evaluate_pipeline.py --model "$SERVED" \
        --base-url "http://localhost:$PORT/v1" --api-key dummy \
        --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
        --llm-top-k 10 --prompt-version v8 --dataset ncbi --split test --temperature 0 \
        --max-tokens 256 --structured-output --keep-duplicates \
        --dump-predictions "$OUT"
    kill $PID 2>/dev/null || true
    sleep 20
}

run_eval "Qwen/Qwen3-4B"                 "qwen3-4b-zeroshot" "preds_zs_ncbi.jsonl"
run_eval "out/qwen3-4b-nel-lora-merged"  "nel-4b"            "preds_ft_ncbi.jsonl"

echo "Fertig. preds_zs_ncbi.jsonl + preds_ft_ncbi.jsonl"
