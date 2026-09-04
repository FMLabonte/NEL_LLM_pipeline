#!/bin/bash
#SBATCH --job-name=nel-cands
#SBATCH --partition=A40medium
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --exclude=node-02
#SBATCH --output=bender_logs/cands-%j.out
#SBATCH --error=bender_logs/cands-%j.err
#
# Stage A der Cross-Encoder-Baseline UND Voraussetzung der GRPO-Vorstudie:
# die Kandidatenlisten nach P3, plus die Dokumenttexte.
#
# Kein LLM (--no-phase4), also kein vLLM und kein KV-Cache. Die GPU wird nur
# fuer SapBERT/FAISS gebraucht, weil die Referenzlaeufe mit --embedding liefen
# und die Listen sonst nicht vergleichbar waeren.
#
# ZEITBUDGET: der erste Anlauf (Job 248348) lief in ein 3h-Limit, das ich zu
# knapp gesetzt hatte -- A40short erlaubt 8h, A40medium einen Tag. Gemessen:
# BC5CDR-Test rund 2h50 (9.661 Mentions, ~1 s/Mention ohne LLM), BioRED 9 min.
# Hochgerechnet liegt der ganze Satz bei 8-11h, deshalb A40medium/16h.
#
# REIHENFOLGE: nach Wert, nicht nach Groesse. BC5CDR-Test zuerst, weil es die
# GRPO-Vorstudie UND den Cross-Encoder freischaltet; NLM-Chem zuletzt, weil es
# das teuerste und am wenigsten dringende ist. Laeuft die Zeit doch aus,
# existieren die wichtigen Dumps trotzdem und ein Resubmit macht nur den Rest.
#
#   sbatch bender_dump_candidates.sbatch
set -euo pipefail

module load Python
module load CUDA
source eval_env/bin/activate
mkdir -p bender_logs
export TOKENIZERS_PARALLELISM=false

if ! dd if=/dev/zero of=.quota_probe bs=1M count=500 status=none 2>/dev/null; then
    rm -f .quota_probe
    echo "FEHLER: keine 500 MB frei im Home. Erst aufraeumen (quota -s)."; exit 1
fi
rm -f .quota_probe

RET="--wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding"
COMMON="$RET --llm-top-k 10 --keep-duplicates --no-phase4"

# Ein Dump zaehlt nur als fertig, wenn er AUCH die Begleitdatei hat und die
# Kandidaten Synonyme tragen. Die erste Fassung dieses Jobs schrieb beides
# nicht, und ein blosses "Datei existiert" wuerde genau die unbrauchbaren
# Dumps ueberspringen -- derselbe Fehler wie damals beim abgeschnittenen
# LoRA-Merge, wo [ -d "$MERGED" ] einen kaputten Ordner durchgewinkt hat.
dump_ok () {
    local out="$1" docs="${1%.jsonl}.docs.jsonl"
    [ -s "$out" ] && [ -s "$docs" ] || return 1
    head -1 "$out" | python3 -c '
import json,sys
try:
    c = json.loads(sys.stdin.read())["candidates"][0]
except Exception:
    sys.exit(1)
sys.exit(0 if "synonyms" in c else 1)' 2>/dev/null
}

dump () {                       # dump <dataset> <split> <outfile> [extra flags]
    local ds="$1" sp="$2" out="$3"; shift 3
    if dump_ok "$out"; then
        echo "=== $out vollstaendig ($(wc -l < "$out") Zeilen + docs), uebersprungen ==="
        return
    fi
    if [ -s "$out" ]; then
        echo "=== $out ist eine alte Fassung ohne docs/synonyms -- wird neu erzeugt ==="
        rm -f "$out" "${out%.jsonl}.docs.jsonl"
    fi
    echo "=== $ds/$sp -> $out   (Start $(date +%H:%M:%S)) ==="
    local t0=$SECONDS
    python3 src/evaluate_pipeline.py $COMMON \
        --dataset "$ds" --split "$sp" "$@" \
        --dump-candidates "$out"
    echo "    $(wc -l < "$out") Zeilen, $(( (SECONDS-t0)/60 )) min"
}

dump bc5cdr   test  cands_p3_bc5cdr.jsonl          # GRPO-Pilot + Cross-Encoder
dump biored   test  cands_p3_biored.jsonl
dump ncbi     test  cands_p3_ncbi.jsonl
dump bc5cdr   train cands_p3_bc5cdr_train.jsonl    # Training des Cross-Encoders
# NLM-Chem zuletzt: teuerster Lauf. Voller Test-Satz, weil die Modelltabelle
# gegen preds_zs_nlmchem_full.jsonl vergleicht, sobald diese Datei existiert.
dump nlm_chem test  cands_p3_nlmchem.jsonl

echo
echo "Fertig nach $(( SECONDS/60 )) min. Weiter mit:"
echo "  sbatch bender_crossencoder.sbatch     # Baseline"
echo "  sbatch bender_grpo_pilot.sbatch       # braucht nur cands_p3_bc5cdr*"
