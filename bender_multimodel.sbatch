#!/bin/bash
# ==========================================================================
# Zweiter und dritter Disambiguator -- der letzte offene Reviewer-Einwand.
#
# Alles im Paper ruht bisher auf Qwen3-4B. Dieser Job wiederholt die
# Kernmessung mit zwei weiteren Modellen auf denselben Mentions, denselben
# Kandidatenlisten, demselben Prompt und demselben Decoder:
#
#   Qwen/Qwen3-8B                    gleiche Familie, doppelte Groesse
#                                    -> ist der Effekt ein Artefakt kleiner Modelle?
#   ibm-granite/granite-3.1-8b-instruct  andere Familie, gleiche Groesse
#                                    -> ist er ein Artefakt von Qwen?
#
# Als SLURM-Array: ein Modell pro Task, laeuft parallel wenn Slots frei sind.
#     sbatch bender_multimodel.sbatch
#
# Granite ist Apache-2.0 und NICHT gated -- kein HF-Login noetig. Beide Modelle
# haben ein Kontextfenster >= 8192, also laeuft alles mit exakt derselben
# --max-model-len wie die Referenzlaeufe.
#
# Wenn ihr ein anderes Modell eintragt: erst max_position_embeddings in dessen
# config.json pruefen. Es muss >= MAXLEN sein, sonst lehnt vLLM die laengsten
# Prompts ab und der Lauf hat weniger Zeilen als die Referenz -- die Danger-Zone-
# Baender waeren dann auf einer anderen Teilmenge gerechnet und nicht vergleichbar.
# Konkret ungeeignet: allenai/OLMo-2-1124-*-Instruct (nur 4096).
# Geeignete ungated Alternativen: microsoft/phi-4 (MIT, 16k, 14B -> GPU_UTIL 0.85),
# tiiuae/Falcon3-7B-Instruct (32k).
#
# Scheitert ein Modell beim Laden, bricht nur dessen Task ab, nicht der andere.
# ==========================================================================
#SBATCH --job-name=nel-multimodel
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --array=1-2
#SBATCH --exclude=node-02
#SBATCH --output=logs/multimodel-%A_%a.out

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

# --- ein Modell pro Array-Task -------------------------------------------
MODELS=("Qwen/Qwen3-8B" "ibm-granite/granite-3.1-8b-instruct")
TAGS=("qwen3-8b" "granite31-8b")
IDX=$(( SLURM_ARRAY_TASK_ID - 1 ))
MODEL="${MODELS[$IDX]}"
TAG="${TAGS[$IDX]}"
GPU_UTIL=0.60          # 8B in bfloat16 braucht ~16 GB Gewichte + KV-Cache
MAXLEN=8192            # identisch zu den Referenzlaeufen -- nicht senken
echo "Task $SLURM_ARRAY_TASK_ID: $MODEL  ->  Suffix _$TAG"

# --- GPU gesund? ----------------------------------------------------------
echo "Knoten: ${SLURMD_NODENAME:-$(hostname)}"
nvidia-smi --query-gpu=index,name,memory.total,ecc.errors.uncorrected.volatile.total \
           --format=csv,noheader 2>/dev/null || echo "nvidia-smi nicht verfuegbar"
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader 2>/dev/null | head -1 | tr -d ' []' || true)
case "$ECC" in
    ""|"N/A"|"NotSupported"|0) ;;
    *) echo ">>> ABBRUCH: GPU auf ${SLURMD_NODENAME:-$(hostname)} meldet $ECC ECC-Fehler."
       echo ">>>     sbatch --exclude=${SLURMD_NODENAME:-$(hostname)} bender_multimodel.sbatch"
       exit 1 ;;
esac

# --- Kontextfenster gross genug? -----------------------------------------
# Ein Modell mit weniger als MAXLEN Positionen laesst vLLM zwar starten, lehnt
# aber die laengsten Prompts ab. Das faellt sonst erst am Zeilenzaehler auf,
# nach Stunden. Lieber hier abbrechen.
python3 - "$MODEL" "$MAXLEN" <<'PY' || exit 1
import sys
try:
    from transformers import AutoConfig
    n = getattr(AutoConfig.from_pretrained(sys.argv[1]), "max_position_embeddings", None)
except Exception as e:                       # kein Netz, kein Cache: nicht blockieren
    print(f"HINWEIS: Kontextlaenge nicht pruefbar ({e.__class__.__name__}).")
    sys.exit(0)
if n is None:
    print("HINWEIS: config.json nennt kein max_position_embeddings.")
elif n < int(sys.argv[2]):
    sys.exit(f">>> ABBRUCH: {sys.argv[1]} hat nur {n} Positionen, gebraucht werden "
             f"{sys.argv[2]}. Ein kuerzeres Fenster verwirft die laengsten Prompts, "
             f"dann rechnet der Lauf auf einer anderen Teilmenge als die Referenz.")
else:
    print(f"Kontextfenster {n} >= {sys.argv[2]}, ok.")
PY

# --- freier Port, damit parallele Array-Tasks sich nicht blockieren -------
PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
SERVED="disamb-$TAG"
VLLM_LOG="logs/vllm-multimodel-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "vLLM auf Port $PORT"
vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len $MAXLEN --gpu-memory-utilization $GPU_UTIL \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!

UP=0
for i in $(seq 1 180); do          # grosszuegiger: das Modell muss erst geladen werden
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "FEHLER: vLLM ist beim Start gestorben. Letzte Zeilen aus $VLLM_LOG:"
        tail -n 25 "$VLLM_LOG"
        if grep -qi "gated\|401\|403\|restricted" "$VLLM_LOG" 2>/dev/null; then
            echo ""
            echo ">>> $MODEL ist auf dem HF-Hub gated. Entweder 'hf auth login',"
            echo ">>> oder in MODELS durch ein ungated Modell ersetzen (siehe Kopf der Datei)."
        fi
        if grep -qi "not supported\|Unsupported architecture\|no module named" "$VLLM_LOG" 2>/dev/null; then
            echo ""
            echo ">>> Diese vLLM-Version kennt die Architektur von $MODEL nicht."
            echo ">>> Anderes Modell eintragen -- nicht vLLM upgraden: die Referenzlaeufe"
            echo ">>> stammen aus 0.27.1, ein Versionswechsel macht sie unvergleichbar."
        fi
        exit 1
    fi
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; UP=1; break; }
    sleep 10
done
[ "$UP" = 1 ] || { echo "FEHLER: vLLM antwortet nach 30 min nicht."; kill $VLLM_PID; exit 1; }

# --- identisch zu den v8-Referenzlaeufen, nur --model unterscheidet sich ---
COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy \
        --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
        --llm-top-k 10 --prompt-version v8 --temperature 0 --max-tokens 256 \
        --structured-output --keep-duplicates --split test"

# Zeilenzahl gegen den Qwen3-4B-Referenzlauf desselben Korpus pruefen, damit ein
# an der Wandzeit abgeschnittener Lauf beim Neueinreichen nicht uebersprungen wird.
run () {                       # run <dataset> <outfile> <referenz> [soll-zeilen]
    local want have
    want=${4:-$(wc -l < "$3")}
    if [ -f "$2" ]; then
        have=$(wc -l < "$2")
        if [ "$have" -eq "$want" ]; then
            echo "=== $1 / $TAG -- vollstaendig ($have Zeilen), uebersprungen ==="; return
        fi
        echo "=== $1 / $TAG -- unvollstaendig ($have von $want), wird neu erzeugt ==="
        rm -f "$2"
    fi
    echo "=== $1 / $TAG ==="
    python3 src/evaluate_pipeline.py $COMMON --dataset "$1" --dump-predictions "$2"
    have=$(wc -l < "$2")
    [ "$have" -eq "$want" ] || echo "WARNUNG: $2 hat $have statt $want Zeilen."
}

# kleine Korpora zuerst: bei Wandzeit-Abbruch hat man drei komplette statt keines
run biored   "preds_zs_biored_${TAG}.jsonl"   "preds_zs_biored.jsonl"
run ncbi     "preds_zs_ncbi_${TAG}.jsonl"     "preds_zs_ncbi.jsonl"

# NLM-Chem: die Referenz preds_zs_nlmchem.jsonl ist NICHT das volle Korpus --
# bender_eval_nlmchem.sbatch hat sie mit "--limit 1500 --limit-random
# --limit-seed 42" erzeugt, also aus einer zufaelligen, gemischten Stichprobe.
# Wir fahren hier das volle Korpus (11729 Mentions, mehr statistische Power fuer
# die Baender) und lassen analyze_robustness.py auf die gemeinsamen Mentions
# beschneiden. Die Soll-Zeilenzahl muss deshalb explizit stehen, sonst haelt die
# run()-Pruefung jeden Lauf fuer unvollstaendig und erzeugt ihn bei jedem
# Neueinreichen neu.
run nlm_chem "preds_zs_nlmchem_${TAG}.jsonl"  "preds_zs_nlmchem.jsonl" 11729

run bc5cdr   "preds_zs_bc5cdr_${TAG}.jsonl"   "preds_qwen3-4b-zeroshot_bc5cdr.jsonl"

kill $VLLM_PID 2>/dev/null || true
echo "Fertig: preds_zs_{biored,ncbi,nlmchem,bc5cdr}_${TAG}.jsonl"
