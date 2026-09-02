#!/bin/bash
# ==========================================================================
# BioRED-Sweep, schwacher Punkt: identisch zum vollen BioRED-Lauf, NUR ohne
# --embedding (ein Knopf) -> sauberer Within-Dataset-Pfeil wie bei MedMentions.
#   mit Embedding  -> preds_zs_biored.jsonl        (schon vorhanden, R@1 79,5)
#   ohne Embedding -> preds_zs_biored_lex.jsonl    (dieser Lauf, niedrigeres R@1)
# ==========================================================================
#SBATCH --job-name=nel-biored-lex
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=03:00:00
#SBATCH --output=logs/biored-lex-%j.out

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
    > logs/vllm-biored-lex-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done

# identisch zum vollen BioRED-Lauf, NUR OHNE --embedding
python3 src/evaluate_pipeline.py --model "$SERVED" \
    --base-url "http://localhost:$PORT/v1" --api-key dummy \
    --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF \
    --llm-top-k 10 --prompt-version v8 --dataset biored --temperature 0 \
    --max-tokens 256 --structured-output --keep-duplicates \
    --dump-predictions preds_zs_biored_lex.jsonl

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_zs_biored_lex.jsonl"
