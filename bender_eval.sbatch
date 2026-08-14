#!/bin/bash
# ==========================================================================
# Eval auf dem Cluster: vLLM serviert das Modell (OpenAI-kompatibel), und
# evaluate_pipeline.py fragt es lokal ab. Kein GGUF, keine Quantisierung,
# kein LMStudio, kein Tokenizer-Fix — vLLM lädt das HF-Modell direkt.
#
# Vorher: siehe CLUSTER_EVAL.md (Datentransfer + eval_env einmalig aufsetzen).
# Modell/Datensatz unten anpassen, dann:  sbatch bender_eval.sbatch
# ==========================================================================
#SBATCH --job-name=nel-eval
#SBATCH --partition=A100short      # 80GB: vLLM + SapBERT teilen sich die GPU bequem
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/eval-%j.out

set -e
mkdir -p logs

# ── HIER anpassen ──────────────────────────────────────────────────────────
MODEL="out/qwen3-4b-nel-lora-merged"   # feinabgestimmt: Pfad | Zero-Shot: HF-ID z.B. Qwen/Qwen3-4B
SERVED_NAME="nel-4b"                    # frei wählbarer Name, muss nur unten übereinstimmen
DATASET="bc5cdr"                       # bc5cdr | medmentions | biored
OUT="preds_${SERVED_NAME}_${DATASET}.jsonl"
PORT=8000
# ───────────────────────────────────────────────────────────────────────────

module load Python
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

# ── vLLM starten (im Hintergrund), OpenAI-kompatibel ──
# gpu-memory-utilization 0.5 lässt Platz für SapBERT/FAISS der Eval-Pipeline.
vllm serve "$MODEL" --served-model-name "$SERVED_NAME" \
    --port $PORT --dtype bfloat16 --max-model-len 8192 \
    --gpu-memory-utilization 0.5 > logs/vllm-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!

# ── warten, bis der Server antwortet ──
echo "Warte auf vLLM..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1; then
        echo "vLLM ist oben (nach ${i}0s)."; break
    fi
    sleep 10
done

# ── Eval gegen den lokalen vLLM-Server ──
# Erster Lauf baut den FAISS-Index (SapBERT über 1.1M Labels, ~15 Min auf GPU)
# und cached ihn — spätere Läufe sind schnell.
python3 src/evaluate_pipeline.py --model "$SERVED_NAME" \
    --base-url "http://localhost:$PORT/v1" --api-key dummy \
    --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
    --llm-top-k 10 --prompt-version v8 --dataset "$DATASET" --temperature 0 \
    --structured-output --keep-duplicates --dump-predictions "$OUT"

# ── vLLM beenden ──
kill $VLLM_PID 2>/dev/null || true
echo "Fertig. Predictions in $OUT"
