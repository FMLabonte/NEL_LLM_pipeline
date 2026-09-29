#!/bin/bash
# ==========================================================================
# Prompt ablation: what in the v8 prompt keeps the danger zone in check?
#
# v8 ALREADY tells the model "TRUST THE RANKING ... override only with clear
# evidence" and ALREADY shows it every candidate's match score -- and the
# mid-confidence danger zone appears anyway. This job measures what each of
# those two ingredients actually does, and whether stating the margin
# explicitly removes the zone:
#
#   v8-notrust  -> v8 minus the conservatism rule      (does the rule do anything?)
#   v8-noscore  -> v8 minus the per-candidate scores   (does seeing scores help?)
#   v8-gap      -> v8 plus an explicit top1-top2 margin + a margin-conditioned rule
#
# The flags below are byte-identical to cluster/bender_eval.sbatch and
# cluster/bender_eval_biored.sbatch, which produced the v8 reference runs
# (preds_qwen3-4b-zeroshot_bc5cdr.jsonl, preds_zs_biored.jsonl). Only
# --prompt-version differs, so each pair is a clean single-factor comparison.
#
# Needs the patched src/llm-disambiguation/prompts.py (three new registry
# entries). The job checks for them before starting vLLM.
#
# Resumable: a run whose output file already exists is skipped, so if the job
# hits the wall clock you can simply re-submit it.
# ==========================================================================
#SBATCH --job-name=nel-promptabl
# Ein BC5CDR-Lauf mit voller Retriever-Konfiguration braucht ~3,2 h, drei davon
# plus drei BioRED-Laeufe also ~11 h. 8 h reichen nicht. Vor dem Einreichen die
# Partitionsgrenzen pruefen:  sinfo -o "%P %l"
# Falls A40short bei 8 h deckelt, hier die laengere Partition eintragen.
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/promptabl-%j.out
# node-02 meldete am 23.08.2026 12.676 nicht korrigierbare ECC-Fehler auf der
# zugeteilten GPU und bringt vLLM beim ersten CUDA-Aufruf um. Zeile entfernen,
# sobald die HPC-Betreuung den Knoten repariert oder ersetzt hat.
#SBATCH --exclude=node-02

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

# ── fail fast: are the new prompt variants actually installed? ────────────
python3 - <<'PYCHECK'
import sys
sys.path.insert(0, "src/llm-disambiguation")
from prompts import PROMPT_REGISTRY
need = {"v8-notrust", "v8-noscore", "v8-gap"}
missing = need - set(PROMPT_REGISTRY)
if missing:
    sys.exit(f"prompts.py is missing {sorted(missing)} -- rsync the patched "
             "src/llm-disambiguation/prompts.py before submitting.")
print("prompt variants present:", sorted(need))
PYCHECK

# Welche GPU haben wir bekommen, und ist sie gesund? Eine A40 mit nicht
# korrigierbaren ECC-Fehlern bringt vLLM beim ersten CUDA-Aufruf zum Absturz
# (node-02 meldete 12.676 davon), also lieber hier abbrechen und den
# Slot freigeben, als acht Stunden Wandzeit an eine defekte Karte zu verlieren.
echo "Knoten: ${SLURMD_NODENAME:-$(hostname)}"
nvidia-smi --query-gpu=index,name,memory.total,ecc.errors.uncorrected.volatile.total \
           --format=csv,noheader 2>/dev/null || echo "nvidia-smi nicht verfuegbar"

ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader 2>/dev/null | head -1 | tr -d ' []' || true)
case "$ECC" in
    ""|"N/A"|"NotSupported"|0) ;;                 # leer, abgeschaltet oder sauber
    *)
        echo ""
        echo ">>> ABBRUCH: die GPU auf ${SLURMD_NODENAME:-$(hostname)} meldet $ECC nicht korrigierbare"
        echo ">>> ECC-Fehler. Auf einer gesunden Karte steht dort 0. vLLM wuerde beim"
        echo ">>> ersten CUDA-Aufruf sterben. Knoten ausschliessen und neu einreichen:"
        echo ">>>     sbatch --exclude=${SLURMD_NODENAME:-$(hostname)} cluster/bender_prompt_ablation.sbatch"
        echo ">>> Und den Knoten der HPC-Betreuung melden."
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-4B"; SERVED="qwen3-4b-zeroshot"; PORT=8000
vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
    > logs/vllm-promptabl-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!

# Warten, aber nicht blind: wenn vLLM stirbt, sofort abbrechen statt 20 Minuten
# gegen einen toten Server zu warten und die Eval anschliessend ins Leere laufen
# zu lassen.
VLLM_LOG="logs/vllm-promptabl-${SLURM_JOB_ID}.out"
echo "Warte auf vLLM..."
UP=0
for i in $(seq 1 120); do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo ""
        echo "FEHLER: vLLM ist beim Start gestorben. Letzte Zeilen aus $VLLM_LOG:"
        tail -n 25 "$VLLM_LOG"
        if grep -q "uncorrectable ECC error" "$VLLM_LOG" 2>/dev/null; then
            echo ""
            echo ">>> Das ist ein HARDWARE-Defekt der GPU auf ${SLURMD_NODENAME:-$(hostname)}, kein Fehler"
            echo ">>> dieses Skripts. Diesen Knoten ausschliessen und neu einreichen:"
            echo ">>>     sbatch --exclude=${SLURMD_NODENAME:-$(hostname)} cluster/bender_prompt_ablation.sbatch"
            echo ">>> Und den Knoten der HPC-Betreuung melden."
        fi
        exit 1
    fi
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; UP=1; break; }
    sleep 10
done
if [ "$UP" != 1 ]; then
    echo "FEHLER: vLLM laeuft, antwortet aber nach 20 Minuten nicht auf Port $PORT."
    tail -n 25 "$VLLM_LOG"
    kill $VLLM_PID 2>/dev/null || true
    exit 1
fi

# identical to the v8 reference runs except for --prompt-version
COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy \
        --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
        --llm-top-k 10 --temperature 0 --max-tokens 256 \
        --structured-output --keep-duplicates --split test"

# --dump-predictions schreibt waehrend des Laufs Zeile fuer Zeile. Ein Abbruch
# an der Wandzeit hinterlaesst also eine ABGESCHNITTENE Datei. "existiert" ist
# deshalb kein brauchbares Kriterium -- wir vergleichen die Zeilenzahl mit dem
# v8-Referenzlauf desselben Korpus und erzeugen Unvollstaendiges neu.
run () {                       # run <dataset> <prompt-version> <outfile> <referenz>
    local want have
    if [ ! -f "$4" ]; then
        echo "FEHLER: Referenzdatei $4 fehlt, Vollstaendigkeit nicht pruefbar."
        exit 1
    fi
    want=$(wc -l < "$4")
    if [ -f "$3" ]; then
        have=$(wc -l < "$3")
        if [ "$have" -eq "$want" ]; then
            echo "=== $1 / $2  -- $3 vollstaendig ($have Zeilen), uebersprungen ==="
            return
        fi
        echo "=== $1 / $2  -- $3 unvollstaendig ($have von $want Zeilen), wird neu erzeugt ==="
        rm -f "$3"
    fi
    echo "=== $1 / $2 ==="
    python3 src/evaluate_pipeline.py $COMMON --dataset "$1" --prompt-version "$2" \
            --dump-predictions "$3"
    have=$(wc -l < "$3")
    if [ "$have" -ne "$want" ]; then
        echo "WARNUNG: $3 hat $have statt $want Zeilen -- Lauf unvollstaendig."
    fi
}

# BioRED first: 1,661 mentions, so if the wall clock runs out you still have a
# complete second corpus rather than three half-finished BC5CDR runs.
for PV in v8-notrust v8-noscore v8-gap; do
    run biored "$PV" "preds_zs_biored_${PV}.jsonl" "preds_zs_biored.jsonl"
done
for PV in v8-notrust v8-noscore v8-gap; do
    run bc5cdr "$PV" "preds_zs_bc5cdr_${PV}.jsonl" "preds_qwen3-4b-zeroshot_bc5cdr.jsonl"
done

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_zs_{biored,bc5cdr}_v8-{notrust,noscore,gap}.jsonl"
