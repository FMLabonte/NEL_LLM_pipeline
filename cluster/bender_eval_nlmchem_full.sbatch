#!/bin/bash
# ==========================================================================
# Qwen3-4B zero-shot auf dem VOLLEN NLM-Chem-Testset.
#
# Warum: preds_zs_nlmchem.jsonl deckt nur 1500 von 11729 Mentions ab
# (cluster/bender_eval_nlmchem.sbatch, "--limit 1500 --limit-random --limit-seed 42").
# Im Band [10,20) bleiben davon ~300 Mentions uebrig, und das
# Bootstrap-Intervall ist entsprechend breit: "kein Effekt nachweisbar" laesst
# sich dort nicht von "zu wenig Daten" unterscheiden. Qwen3-8B und Granite
# laufen ohnehin schon auf dem vollen Korpus -- fehlt nur die Basislinie.
#
# Erwartung: Band-n von ~300 auf ~4000, Intervall von etwa +-4,5 auf +-1,7.
# Damit entscheidet sich, ob NLM-Chem wirklich null ist oder nur zu klein war.
# Beide Ausgaenge sind publizierbar; unentscheidbar ist der schlechteste.
#
#     sbatch cluster/bender_eval_nlmchem_full.sbatch
#
# Ergebnis: preds_zs_nlmchem_full.jsonl
# analyze_robustness.py nimmt diese Datei automatisch als NLM-Chem-Referenz,
# sobald sie existiert. Die Seed-Auswertung bleibt bewusst auf der 1500er
# Stichprobe, weil die Fine-Tuning-Laeufe dagegen gemessen werden.
# ==========================================================================
#SBATCH --job-name=nel-nlmchem-full
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --exclude=node-02
#SBATCH --output=logs/nlmchem-full-%j.out

# --mem=32G: der Partitionsstandard sind 6000M, gemessen wurden 10,3 GB
# (MaxRSS von Job 245631). Ueber das volle Korpus haelt die Pipeline die
# Kandidatenlisten aller 11729 Mentions plus MRCONSO im Speicher.

set -e
mkdir -p logs

# Schreibtest, bevor irgendetwas laeuft. Ein volles Home-Quota toetet den Lauf
# sonst erst nach Stunden, und zwar lautlos: die Logzeilen brechen mitten im
# Wort ab, die Ausgabedatei bleibt leer, und sacct meldet nur "FAILED 1:0".
if ! dd if=/dev/zero of=.quota_probe bs=1M count=200 status=none 2>/dev/null; then
    rm -f .quota_probe
    echo ">>> ABBRUCH: kein Schreibplatz im Home (200 MB Test fehlgeschlagen)."
    echo ">>>     quota -s"
    echo ">>>     du -sh ~/.cache/huggingface/hub/* | sort -h | tail"
    exit 1
fi
rm -f .quota_probe
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

OUT=preds_zs_nlmchem_full.jsonl
# Sollgroesse aus einem vorhandenen vollen Lauf desselben Korpus ableiten.
WANT=11729
for f in preds_zs_nlmchem_qwen3-8b.jsonl preds_zs_nlmchem_granite31-8b.jsonl; do
    [ -f "$f" ] && [ "$(wc -l < "$f")" -gt "$WANT" ] && WANT=$(wc -l < "$f")
done
if [ -f "$OUT" ] && [ "$(wc -l < "$OUT")" -eq "$WANT" ]; then
    echo "$OUT ist bereits vollstaendig ($WANT Zeilen)."; exit 0
fi
[ -f "$OUT" ] && { echo "$OUT unvollstaendig, wird neu erzeugt."; rm -f "$OUT"; }

echo "Knoten: ${SLURMD_NODENAME:-$(hostname)}"
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader 2>/dev/null | head -1 | tr -d ' []' || true)
case "$ECC" in
    ""|"N/A"|"NotSupported"|0) ;;
    *) echo ">>> ABBRUCH: GPU auf ${SLURMD_NODENAME:-$(hostname)} meldet $ECC ECC-Fehler."
       echo ">>>     sbatch --exclude=${SLURMD_NODENAME:-$(hostname)} cluster/bender_eval_nlmchem_full.sbatch"
       exit 1 ;;
esac

PORT=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
SERVED="qwen3-4b-zeroshot"
VLLM_LOG="logs/vllm-nlmchem-full-${SLURM_JOB_ID}.out"
vllm serve "Qwen/Qwen3-4B" --served-model-name "$SERVED" --port $PORT \
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

# Byte-identisch zu cluster/bender_eval_nlmchem.sbatch -- nur ohne die drei --limit-Flags.
echo "=== nlm_chem, volles Testset, Qwen3-4B zero-shot ==="
python3 src/evaluate_pipeline.py --model "$SERVED" \
    --base-url "http://localhost:$PORT/v1" --api-key dummy \
    --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
    --llm-top-k 10 --prompt-version v8 --dataset nlm_chem --split test \
    --temperature 0 --max-tokens 256 --structured-output --keep-duplicates \
    --dump-predictions "$OUT"

kill $VLLM_PID 2>/dev/null || true
HAVE=$(wc -l < "$OUT")
[ "$HAVE" -eq "$WANT" ] || echo "WARNUNG: $OUT hat $HAVE statt $WANT Zeilen."
echo "Fertig: $OUT ($HAVE Zeilen)"
echo "Danach: python3 src/analyze_robustness.py --models"
