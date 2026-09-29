#!/bin/bash
#SBATCH --job-name=nel-ce
#SBATCH --partition=A40short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=node-02
#SBATCH --output=bender_logs/ce-%j.out
#SBATCH --error=bender_logs/ce-%j.err
#
# Stage B: the non-LLM re-ranker baseline on the identical candidate lists.
#
# Two variants, because a reviewer will ask for both:
#   medcpt      zero-shot MedCPT cross-encoder (109M params, no training)
#   medcpt-ft   the same model trained on the BC5CDR train split with the same
#               listwise objective the LLM is given -- the fair counterpart to
#               the LoRA run in section 6
#
# Both write preds_zs_<corpus>_<tag>.jsonl, which is the name pattern
# analyze_robustness.py --models already looks for. Nothing else to wire up.
#
#   sbatch cluster/bender_crossencoder.sbatch
set -euo pipefail

module load Python
source eval_env/bin/activate
mkdir -p bender_logs models

if ! dd if=/dev/zero of=.quota_probe bs=1M count=500 status=none 2>/dev/null; then
    rm -f .quota_probe
    echo "FEHLER: keine 500 MB frei im Home (Checkpoint braucht ~450 MB)."; exit 1
fi
rm -f .quota_probe

# corpus tag | candidate dump | reference preds run
CORPORA=(
  "bc5cdr   cands_p3_bc5cdr.jsonl   preds_qwen3-4b-zeroshot_bc5cdr.jsonl"
  "biored   cands_p3_biored.jsonl   preds_zs_biored.jsonl"
  "ncbi     cands_p3_ncbi.jsonl     preds_zs_ncbi.jsonl"
  "nlmchem  cands_p3_nlmchem.jsonl  preds_zs_nlmchem_full.jsonl"
)

for row in "${CORPORA[@]}"; do
    set -- $row
    [ -s "$2" ] || { echo "FEHLT: $2 -- erst cluster/bender_dump_candidates.sbatch"; exit 1; }
    [ -s "$3" ] || { echo "FEHLT: $3 -- Referenzlauf nicht vorhanden"; exit 1; }
done

# ---------------------------------------------------------------- zero-shot
for row in "${CORPORA[@]}"; do
    set -- $row
    OUT="preds_zs_${1}_medcpt.jsonl"
    if [ -s "$OUT" ] && [ "$(wc -l < "$OUT")" -eq "$(wc -l < "$3")" ]; then
        echo "=== $OUT vollstaendig, uebersprungen ==="; continue
    fi
    echo "=== zero-shot cross-encoder: $1 ==="
    python3 scripts/rerank_crossencoder.py --cands "$2" --ref "$3" --out "$OUT"
done

# ---------------------------------------------------------------- fine-tuned
CKPT=models/ce_bc5cdr
if [ ! -s "$CKPT/config.json" ]; then
    if [ ! -s cands_p3_bc5cdr_train.jsonl ]; then
        echo "cands_p3_bc5cdr_train.jsonl fehlt -- ueberspringe die trainierte Variante."
        echo "Fertig (nur zero-shot)."; exit 0
    fi
    echo "=== training the cross-encoder on the BC5CDR train split ==="
    python3 scripts/train_crossencoder.py \
        --cands cands_p3_bc5cdr_train.jsonl --out "$CKPT" --epochs 2 --seed 1
fi

for row in "${CORPORA[@]}"; do
    set -- $row
    OUT="preds_zs_${1}_medcpt-ft.jsonl"
    if [ -s "$OUT" ] && [ "$(wc -l < "$OUT")" -eq "$(wc -l < "$3")" ]; then
        echo "=== $OUT vollstaendig, uebersprungen ==="; continue
    fi
    echo "=== trained cross-encoder: $1 ==="
    python3 scripts/rerank_crossencoder.py --cands "$2" --ref "$3" \
        --model "$CKPT" --out "$OUT"
done

echo
echo "Fertig. Auswerten:  python3 src/analyze_robustness.py --models"
