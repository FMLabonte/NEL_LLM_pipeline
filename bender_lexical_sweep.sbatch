#!/bin/bash
# ==========================================================================
# Overnight: LEXICAL (ohne Embedding) zero-shot auf NCBI, BioRED, NLM-Chem.
# Wir haben die VOLLEN (+Embedding) zero-shot-Preds schon -> das hier ist der
# schwache Punkt jedes Datensatzes => je ein Within-Dataset-Retriever-PFEIL
# fuer Figur 1 (wie MedMentions lexical->+SapBERT). Nur EIN Knopf (Embedding).
# Reihenfolge: schnelle zuerst (NCBI, BioRED), langsame Volltext zuletzt.
# ==========================================================================
#SBATCH --job-name=nel-lexsweep
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=14:00:00
#SBATCH --output=logs/lexsweep-%j.out

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
    > logs/vllm-lexsweep-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done

COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy \
        --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF \
        --llm-top-k 10 --prompt-version v8 --temperature 0 --max-tokens 256 \
        --structured-output --keep-duplicates"
# HINWEIS: KEIN --embedding  => das ist der lexikalische (schwache) Retriever.

echo "=== NCBI lexical ===";    python3 src/evaluate_pipeline.py $COMMON --dataset ncbi --split test     --dump-predictions preds_zs_ncbi_lex.jsonl
echo "=== BioRED lexical ===";  python3 src/evaluate_pipeline.py $COMMON --dataset biored               --dump-predictions preds_zs_biored_lex.jsonl
echo "=== NLM-Chem lexical ==="; python3 src/evaluate_pipeline.py $COMMON --dataset nlm_chem --split test \
        --limit 1500 --limit-random --limit-seed 42 --dump-predictions preds_zs_nlmchem_lex.jsonl

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_zs_{ncbi,biored,nlmchem}_lex.jsonl"
