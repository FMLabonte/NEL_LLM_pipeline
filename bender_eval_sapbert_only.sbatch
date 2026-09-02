#!/bin/bash
# ==========================================================================
# Reviewer-Killer (Option B): unsere Zerlegung auf einem RETRIEVER, den wir
# NICHT gebaut haben -- reiner oeffentlicher SapBERT-Bi-Encoder (--embedding-only,
# keine Regeln/Expansion/Abkuerzungen/Topic-Scoring). Genau der Retriever, den
# LLM4BioEL und Ye & Mitchell nutzen. Zeigt: taucht die Danger Zone auch hier auf?
# Zero-shot Qwen3-4B auf NCBI + BioRED + BC5CDR.
# ==========================================================================
#SBATCH --job-name=nel-sapbert
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/sapbert-%j.out

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
    > logs/vllm-sapbert-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done

# --embedding-only erzwingt Embedding an + schaltet alle handgebauten Teile aus.
# KEIN --wikidata/--dbpedia/--umls (reiner SapBERT).
COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy \
        --embedding-only --llm-top-k 10 --prompt-version v8 --temperature 0 \
        --max-tokens 256 --structured-output --keep-duplicates"

echo "=== NCBI (SapBERT-only) ===";   python3 src/evaluate_pipeline.py $COMMON --dataset ncbi --split test --dump-predictions preds_zs_ncbi_sapbert.jsonl
echo "=== BioRED (SapBERT-only) ==="; python3 src/evaluate_pipeline.py $COMMON --dataset biored             --dump-predictions preds_zs_biored_sapbert.jsonl
echo "=== BC5CDR (SapBERT-only) ==="; python3 src/evaluate_pipeline.py $COMMON --dataset bc5cdr --split test --dump-predictions preds_zs_bc5cdr_sapbert.jsonl

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_zs_{ncbi,biored,bc5cdr}_sapbert.jsonl"
