#!/bin/bash
# ==========================================================================
# Retriever-Stärke-Sweep: LLM konstant (Zero-Shot 4B), NUR den Retriever variieren.
# Liefert die kausale Kurve "LLM-Nutzen = f(Retriever-R@1)".
#
# Voraussetzung: Base-Modell gecached (huggingface-cli download Qwen/Qwen3-4B),
#                MRREL.RRF getoucht (Bridge), Caches vollständig (wie beim Eval).
# Einreichen:  sbatch cluster/bender_sweep.sbatch
# ==========================================================================
#SBATCH --job-name=nel-sweep
#SBATCH --partition=A40medium      # 1-Tag-Limit; 2 Evals ~8h. A40 = Intel (wie env)
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/sweep-%j.out

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

# LLM KONSTANT (Zero-Shot) — wir variieren nur den Retriever darunter.
MODEL="Qwen/Qwen3-4B"
SERVED="qwen3-4b-zeroshot"
PORT=8000

vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
    > logs/vllm-sweep-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done

COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --llm-top-k 10 --prompt-version v8 --dataset bc5cdr --temperature 0 --max-tokens 256 --structured-output --keep-duplicates"

# ── SCHWACH: nur rapidfuzz-String (kein Embedding, keine Expansion) ──
echo "=== Retriever WEAK ==="
python3 src/evaluate_pipeline.py $COMMON --no-expansion \
    --dump-predictions preds_zs_weak.jsonl

# ── MITTEL: + SapBERT-Embedding (keine Expansion) ──
echo "=== Retriever MID ==="
python3 src/evaluate_pipeline.py $COMMON --embedding --no-expansion \
    --dump-predictions preds_zs_mid.jsonl

# FULL = bestehendes preds_qwen3-4b-zeroshot_bc5cdr.jsonl (Embedding + Expansion)

kill $VLLM_PID 2>/dev/null || true
echo "Sweep fertig. Punkte: preds_zs_weak, preds_zs_mid, + preds_qwen3-4b-zeroshot (full)."
