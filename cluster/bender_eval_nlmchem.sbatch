#!/bin/bash
# ==========================================================================
# NLM-Chem: HELD-OUT MeSH-Chemie-Datensatz (Entitätstyp-Vielfalt fuer Deferenz).
# Voll held-out (Modell nie darauf trainiert). Beide Modelle, gecappt auf 4000
# zufaellige Mentions (reicht fuer die Kreuztabelle, haelt die Laufzeit im Rahmen).
#   zero-shot -> preds_zs_nlmchem.jsonl
#   finetuned -> preds_ft_nlmchem.jsonl
# Voraussetzung: Data/NLM-Chem/nlm_chem_test.pubtator (via bioc_to_pubtator.py).
# ==========================================================================
#SBATCH --job-name=nel-nlmchem
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=20:00:00
#SBATCH --output=logs/nlmchem-%j.out

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
        > "logs/vllm-nlmchem-${SERVED}-${SLURM_JOB_ID}.out" 2>&1 &
    local PID=$!
    echo "Warte auf vLLM ($SERVED)..."
    for i in $(seq 1 120); do
        curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "up."; break; }
        sleep 10
    done
    python3 src/evaluate_pipeline.py --model "$SERVED" \
        --base-url "http://localhost:$PORT/v1" --api-key dummy \
        --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
        --llm-top-k 10 --prompt-version v8 --dataset nlm_chem --split test --temperature 0 \
        --max-tokens 256 --structured-output --keep-duplicates \
        --limit 1500 --limit-random --limit-seed 42 \
        --dump-predictions "$OUT"
    kill $PID 2>/dev/null || true
    sleep 20
}

run_eval "Qwen/Qwen3-4B"                 "qwen3-4b-zeroshot" "preds_zs_nlmchem.jsonl"
run_eval "out/qwen3-4b-nel-lora-merged"  "nel-4b"            "preds_ft_nlmchem.jsonl"

echo "Fertig. preds_zs_nlmchem.jsonl + preds_ft_nlmchem.jsonl"
