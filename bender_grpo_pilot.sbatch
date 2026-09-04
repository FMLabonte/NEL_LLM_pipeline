#!/bin/bash
#SBATCH --job-name=nel-grpopilot
#SBATCH --partition=A40short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=node-02
#SBATCH --output=bender_logs/grpopilot-%j.out
#SBATCH --error=bender_logs/grpopilot-%j.err
#
# GRPO-Vorstudie: hat der Algorithmus auf dieser Aufgabe ueberhaupt einen
# Gradienten? Kein RL, keine Trainingsschleife -- nur G Rollouts pro Mention bei
# mehreren Temperaturen, gezaehlt wird, wie oft eine Gruppe uneinig ist.
#
# GRPO normiert den Reward INNERHALB der Gruppe. Sind alle Rollouts gleich gut,
# ist die Standardabweichung null, jeder Advantage null und die Gruppe traegt
# nichts zum Update bei. Bei enum-beschraenktem Decoding ueber zehn Kandidaten
# und einem Retriever, der auf BC5CDR schon 82.7 % richtig liegt, ist das die
# realistische Ausgangslage -- und die eine Annahme, auf der der ganze RL-Teil
# steht. Sie kostet hier zwei GPU-Stunden statt zwei Wochen.
#
# Voraussetzung: bender_dump_candidates.sbatch ist durch (liefert die
# Kandidatenlisten UND die Dokumenttexte, ohne die der v8-Prompt nicht
# rekonstruierbar ist -- v8 fenstert das volle Dokument, context_words=64).
#
#   sbatch bender_grpo_pilot.sbatch
set -euo pipefail

module load Python
module load CUDA
source eval_env/bin/activate
mkdir -p bender_logs
export TOKENIZERS_PARALLELISM=false

MODEL="Qwen/Qwen3-4B"
SERVED="qwen3-4b-zeroshot"
CANDS="cands_p3_bc5cdr.jsonl"
REF="preds_qwen3-4b-zeroshot_bc5cdr.jsonl"
GROUP=8
LIMIT=800

for f in "$CANDS" "${CANDS%.jsonl}.docs.jsonl" "$REF"; do
    [ -s "$f" ] || { echo "FEHLT: $f"; echo "Erst bender_dump_candidates.sbatch laufen lassen."; exit 1; }
done

# --- ECC-Wache: node-02 zaehlt 12.676 nicht korrigierbare Fehler ------------
if command -v nvidia-smi >/dev/null 2>&1; then
    ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader 2>/dev/null | head -1)
    if [[ "$ECC" =~ ^[0-9]+$ ]] && [ "$ECC" -gt 0 ]; then
        echo "GPU meldet $ECC unkorrigierbare ECC-Fehler auf $(hostname). Abbruch."
        echo "Neu starten mit: sbatch --exclude=node-02,$(hostname) $0"
        exit 1
    fi
fi

PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
echo "vLLM auf Port $PORT"
vllm serve "$MODEL" --served-model-name "$SERVED" --port "$PORT" \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.60 \
    > "bender_logs/vllm-grpopilot-${SLURM_JOB_ID}.out" 2>&1 &
VLLM_PID=$!
trap 'kill $VLLM_PID 2>/dev/null || true' EXIT

for i in $(seq 1 120); do
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; break; }
    sleep 10
done
curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 || { echo "vLLM kam nicht hoch."; exit 1; }

python3 scripts/grpo_pilot.py \
    --cands "$CANDS" --ref "$REF" \
    --model "$SERVED" --base-url "http://localhost:$PORT/v1" \
    --group-size "$GROUP" --temperatures 0.0 0.7 1.0 1.3 \
    --limit "$LIMIT" --out grpo_pilot_bc5cdr.json

echo
echo "Fertig -> grpo_pilot_bc5cdr.json"
echo "T=0.0 ist die Kontrolle: dort MUSS der mixed-Anteil 0 sein. Ist er das"
echo "nicht, sampelt der Server trotz temperature 0, und alle anderen Zahlen"
echo "messen etwas anderes als gedacht."
