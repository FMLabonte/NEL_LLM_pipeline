#!/bin/bash
# ==========================================================================
# BC5CDR mit dem FEINABGESTIMMTEN Modell, ENRICHED dump.
# Gleiche Config wie der Zero-Shot-BC5CDR-Lauf (preds_qwen3-4b-zeroshot_bc5cdr),
# nur mit dem gemergten LoRA-Modell -> preds_ft_bc5cdr.jsonl.
# Liefert: finetuned Churn (Finding 3), finetuned Gate (Finding 1 zs vs ft),
#          und die enriched BC5CDR-ft-Daten, die bisher fehlten.
# ==========================================================================
#SBATCH --job-name=nel-bc5-ft
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/bc5-ft-%j.out

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

MODEL="out/qwen3-4b-nel-lora-merged"
SERVED="nel-4b"
PORT=8000

vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
    > logs/vllm-bc5-ft-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done

# identische Config wie der Zero-Shot-BC5CDR-"full"-Lauf, NUR anderes Modell
python3 src/evaluate_pipeline.py --model "$SERVED" \
    --base-url "http://localhost:$PORT/v1" --api-key dummy \
    --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
    --llm-top-k 10 --prompt-version v8 --dataset bc5cdr --temperature 0 \
    --max-tokens 256 --structured-output --keep-duplicates \
    --dump-predictions preds_ft_bc5cdr.jsonl

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_ft_bc5cdr.jsonl"
