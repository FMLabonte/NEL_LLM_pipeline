#!/bin/bash
#SBATCH --job-name=nel-cands
#SBATCH --partition=A40short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=node-02
#SBATCH --output=bender_logs/cands-%j.out
#SBATCH --error=bender_logs/cands-%j.err
#
# Stage A of the cross-encoder baseline: dump the post-P3 candidate lists.
#
# No LLM is involved (--no-phase4), so no vLLM server and no 8 GB of KV cache --
# but the GPU is still needed because the reference runs use --embedding
# (SapBERT + FAISS), and the candidate lists have to be produced by the SAME
# retriever configuration or the baseline is not comparable.
#
# Flags below are copied verbatim from the reference eval jobs. If you change a
# retriever flag in one place, change it in both, or analyze_robustness.py will
# report a P3 mismatch (which is exactly what it is there for).
#
#   sbatch bender_dump_candidates.sbatch
set -euo pipefail

module load Python
source eval_env/bin/activate
mkdir -p bender_logs

# --- quota guard: a truncated dump looks like a finished one ----------------
if ! dd if=/dev/zero of=.quota_probe bs=1M count=200 status=none 2>/dev/null; then
    rm -f .quota_probe
    echo "FEHLER: keine 200 MB frei im Home. Erst aufraeumen (quota -s)."; exit 1
fi
rm -f .quota_probe

RET="--wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding"
COMMON="$RET --llm-top-k 10 --keep-duplicates --no-phase4"

dump () {                       # dump <dataset> <split> <outfile> [extra flags]
    local ds="$1" sp="$2" out="$3"; shift 3
    if [ -s "$out" ]; then
        echo "=== $out existiert ($(wc -l < "$out") Zeilen), uebersprungen ==="
        return
    fi
    echo "=== $ds/$sp -> $out ==="
    python3 src/evaluate_pipeline.py $COMMON \
        --dataset "$ds" --split "$sp" "$@" \
        --dump-candidates "$out"
    echo "    $(wc -l < "$out") Zeilen"
}

# --- test splits: one per corpus in the paper's model table -----------------
dump bc5cdr   test cands_p3_bc5cdr.jsonl
dump biored   test cands_p3_biored.jsonl
dump ncbi     test cands_p3_ncbi.jsonl
# NLM-Chem: the FULL test set, because analyze_robustness.py compares the model
# table against preds_zs_nlmchem_full.jsonl whenever that file exists.
dump nlm_chem test cands_p3_nlmchem.jsonl

# --- training split: only BC5CDR, the split the LoRA model was tuned on -----
dump bc5cdr   train cands_p3_bc5cdr_train.jsonl

echo
echo "Fertig. Weiter mit:  sbatch bender_crossencoder.sbatch"
