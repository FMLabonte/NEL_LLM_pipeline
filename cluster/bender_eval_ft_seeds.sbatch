#!/bin/bash
# ==========================================================================
# Die drei fine-getunten Seeds evaluieren -- ein Seed pro Array-Task.
#
# Identische Flags wie die Zero-shot-Referenzlaeufe, damit jeder Seed direkt
# gegen preds_qwen3-4b-zeroshot_bc5cdr.jsonl bzw. preds_zs_<korpus>.jsonl
# vergleichbar ist. Auswertung danach:
#     python3 src/analyze_robustness.py --seeds
#
#     sbatch cluster/bender_eval_ft_seeds.sbatch
# ==========================================================================
#SBATCH --job-name=nel-ft-seeds
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --array=1-3
#SBATCH --exclude=node-02
#SBATCH --output=logs/ftseed-%A_%a.out

# --mem=32G: der Partitionsstandard sind 6000M, gemessen wurden 10,3 GB fuer
# einen vergleichbaren Lauf. Zu wenig angeforderter Speicher hat Job 245631
# gekillt.

set -e
mkdir -p logs

# Schreibtest, bevor irgendetwas laeuft. Ein volles Home-Quota toetet den Lauf
# sonst erst nach Stunden, und zwar lautlos: die Logzeilen brechen mitten im
# Wort ab, die Ausgabedatei bleibt leer, und sacct meldet nur "FAILED 1:0".
if ! dd if=/dev/zero of=.quota_probe_$SLURM_ARRAY_TASK_ID bs=1M count=200 status=none 2>/dev/null; then
    rm -f .quota_probe_$SLURM_ARRAY_TASK_ID
    echo ">>> ABBRUCH: kein Schreibplatz im Home (200 MB Test fehlgeschlagen)."
    echo ">>>     quota -s"
    echo ">>>     du -sh ~/.cache/huggingface/hub/* | sort -h | tail"
    exit 1
fi
rm -f .quota_probe_$SLURM_ARRAY_TASK_ID
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

SEED=$SLURM_ARRAY_TASK_ID
MERGED=out/qwen3-4b-nel-lora-s${SEED}-merged

# Nicht nur auf Existenz pruefen: ein beim Quota-Limit abgeschnittener Merge
# hinterlaesst ein Verzeichnis mit 0-Byte tokenizer.json, das vLLM erst nach
# dem Modell-Laden mit einem JSONDecodeError quittiert -- zwei verlorene Minuten
# Jobzeit und eine unverstaendliche Fehlermeldung.
merged_ok () {                      # merged_ok <dir>
    local d="$1" f bytes
    [ -d "$d" ] || return 1
    for f in config.json tokenizer.json tokenizer_config.json; do
        [ -s "$d/$f" ] || return 1
    done
    bytes=$(du -sb "$d" 2>/dev/null | cut -f1)
    [ "${bytes:-0}" -gt 6000000000 ]        # 4B in fp16 sind rund 8 GB
}

if ! merged_ok "$MERGED"; then
    echo "FEHLER: $MERGED fehlt oder ist unvollstaendig."
    [ -d "$MERGED" ] && du -sh "$MERGED" && ls -l "$MERGED"/*.json 2>/dev/null
    echo ">>> Adapter ist vermutlich heil, nur der Merge kaputt:"
    echo ">>>     sbatch --array=${SEED} cluster/bender_merge_seed.sbatch"
    echo ">>> Erst wenn auch der Adapter fehlt:"
    echo ">>>     sbatch --array=${SEED} cluster/bender_finetune_seeds.sbatch"
    exit 1
fi

echo "Knoten: ${SLURMD_NODENAME:-$(hostname)}"
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader 2>/dev/null | head -1 | tr -d ' []' || true)
case "$ECC" in
    ""|"N/A"|"NotSupported"|0) ;;
    *) echo ">>> ABBRUCH: GPU meldet $ECC ECC-Fehler."; exit 1 ;;
esac

PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
SERVED="nel-4b-s${SEED}"
VLLM_LOG="logs/vllm-ftseed-${SLURM_ARRAY_JOB_ID}_${SEED}.out"
vllm serve "$MERGED" --served-model-name "$SERVED" --port $PORT \
    --dtype bfloat16 --max-model-len 8192 --gpu-memory-utilization 0.5 \
    > "$VLLM_LOG" 2>&1 &
VLLM_PID=$!
UP=0
for i in $(seq 1 120); do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "FEHLER: vLLM gestorben. Letzte Zeilen aus $VLLM_LOG:"; tail -n 25 "$VLLM_LOG"; exit 1
    fi
    curl -s "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && { echo "vLLM up."; UP=1; break; }
    sleep 10
done
[ "$UP" = 1 ] || { echo "FEHLER: vLLM antwortet nicht."; kill $VLLM_PID; exit 1; }

COMMON="--model $SERVED --base-url http://localhost:$PORT/v1 --api-key dummy \
        --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
        --llm-top-k 10 --prompt-version v8 --temperature 0 --max-tokens 256 \
        --structured-output --keep-duplicates --split test"

run () {                       # run <dataset> <outfile> <referenz> [extra-flags]
    local want have
    want=$(wc -l < "$3")
    if [ -f "$2" ]; then
        have=$(wc -l < "$2")
        if [ "$have" -eq "$want" ]; then
            echo "=== $1 / Seed $SEED -- vollstaendig, uebersprungen ==="; return
        fi
        echo "=== $1 / Seed $SEED -- unvollstaendig ($have von $want), neu ==="; rm -f "$2"
    fi
    echo "=== $1 / Seed $SEED ==="
    python3 src/evaluate_pipeline.py $COMMON --dataset "$1" ${4:-} --dump-predictions "$2"
    have=$(wc -l < "$2")
    [ "$have" -eq "$want" ] || echo "WARNUNG: $2 hat $have statt $want Zeilen."
}

run biored   "preds_ft_biored_s${SEED}.jsonl"   "preds_zs_biored.jsonl"
run ncbi     "preds_ft_ncbi_s${SEED}.jsonl"     "preds_zs_ncbi.jsonl"

# NLM-Chem exakt so einschraenken wie cluster/bender_eval_nlmchem.sbatch, aus dem
# preds_zs_nlmchem.jsonl und preds_ft_nlmchem.jsonl stammen. Gleiche Zahl,
# gleicher Seed, gleiche Stichprobe -- damit ist der Seed-Vergleich Zeile fuer
# Zeile derselbe Satz Mentions, und es spart pro Seed ueber 10k Mentions.
run nlm_chem "preds_ft_nlmchem_s${SEED}.jsonl"  "preds_zs_nlmchem.jsonl" \
             "--limit 1500 --limit-random --limit-seed 42"

run bc5cdr   "preds_ft_bc5cdr_s${SEED}.jsonl"   "preds_qwen3-4b-zeroshot_bc5cdr.jsonl"

kill $VLLM_PID 2>/dev/null || true
echo "Fertig: preds_ft_*_s${SEED}.jsonl"
