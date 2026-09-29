#!/bin/bash
# ==========================================================================
# Graded retriever degradation -- the controlled version of the paper's frame.
#
# The frame currently rests on swapping whole retrievers, which changes the
# candidate pool, the score scale and top-1 accuracy at once. Here the
# candidate SET and the score vector stay byte-identical and only the concept
# sitting at rank 1 changes, at rates 0 .. 0.5. Recall@1 therefore falls from
# 84.8 to 42.4 while Recall@10 stays at 90.5 and the top1-top2 gap distribution
# is unchanged -- a dose-response for retriever strength with everything else
# held fixed.
#
# Prerequisite (login node, no GPU):
#   python3 scripts/degrade_candidates.py cands_biosyn_bc5cdr.jsonl \
#       --gold mentions_bc5cdr_test.jsonl --out-prefix cands_dose_biosyn
#
# p=0 is skipped: it is byte-identical to the BioSyn run the paper already
# reports (preds_zs_bc5cdr_biosyn.jsonl). Drop the `continue` below to re-run it
# as a sanity check.
#
# The flags match cluster/bender_eval_curve_retrievers.sbatch, which produced that
# reference run. --candidates-from replaces phases 2 and 3 wholesale, so the
# retriever-enrichment flags are deliberately absent here.
#
# Resumable: a level whose output already exists is skipped.
# ==========================================================================
#SBATCH --job-name=nel-dose
# Fuenf Laeufe auf BioSyn-Kandidaten (kein Retriever-Aufbau, daher schneller
# als die Ablation), aber 8 h sind knapp. Partitionsgrenzen:  sinfo -o "%P %l"
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=16:00:00
#SBATCH --output=logs/dose-%j.out
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
        echo ">>>     sbatch --exclude=${SLURMD_NODENAME:-$(hostname)} cluster/bender_retriever_dose.sbatch"
        echo ">>> Und den Knoten der HPC-Betreuung melden."
        exit 1
        ;;
esac

MODEL="Qwen/Qwen3-4B"; SERVED="qwen3-4b-zeroshot"; PORT=8000
vllm serve "$MODEL" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
    > logs/vllm-dose-${SLURM_JOB_ID}.out 2>&1 &
VLLM_PID=$!

# Warten, aber nicht blind: wenn vLLM stirbt, sofort abbrechen statt 20 Minuten
# gegen einen toten Server zu warten und die Eval anschliessend ins Leere laufen
# zu lassen.
VLLM_LOG="logs/vllm-dose-${SLURM_JOB_ID}.out"
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
            echo ">>>     sbatch --exclude=${SLURMD_NODENAME:-$(hostname)} cluster/bender_retriever_dose.sbatch"
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

COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy \
        --llm-top-k 10 --prompt-version v8 --temperature 0 --max-tokens 256 \
        --structured-output --keep-duplicates --dataset bc5cdr --split test"

for P in 0 0.1 0.2 0.3 0.4 0.5; do
    if [ "$P" = "0" ]; then
        echo "=== p=0 skipped (== preds_zs_bc5cdr_biosyn.jsonl) ==="
        continue
    fi
    CANDS="cands_dose_biosyn_p${P}.jsonl"
    if [ ! -f "$CANDS" ]; then
        echo "MISSING $CANDS -- run scripts/degrade_candidates.py first."
        exit 1
    fi
    # dump-predictions schreibt waehrend des Laufs, ein Wandzeit-Abbruch
    # hinterlaesst also eine abgeschnittene Datei -- Zeilenzahl gegen den
    # BioSyn-Referenzlauf pruefen statt nur auf Existenz.
    WANT=$(wc -l < preds_zs_bc5cdr_biosyn.jsonl)
    if [ -f "preds_dose_p${P}.jsonl" ]; then
        HAVE=$(wc -l < "preds_dose_p${P}.jsonl")
        if [ "$HAVE" -eq "$WANT" ]; then
            echo "=== p=$P vollstaendig ($HAVE Zeilen), uebersprungen ==="
            continue
        fi
        echo "=== p=$P unvollstaendig ($HAVE von $WANT), wird neu erzeugt ==="
        rm -f "preds_dose_p${P}.jsonl"
    fi
    echo "=== degradation rate p=$P ==="
    python3 src/evaluate_pipeline.py $COMMON --candidates-from "$CANDS" \
            --dump-predictions preds_dose_p${P}.jsonl
done

kill $VLLM_PID 2>/dev/null || true
echo "Fertig. preds_dose_p{0.1..0.5}.jsonl  (+ preds_zs_bc5cdr_biosyn.jsonl as p=0)"
echo "Auswerten:  python3 src/analyze_dose.py"
