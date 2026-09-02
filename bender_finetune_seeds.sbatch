#!/bin/bash
# ==========================================================================
# Drei Fine-Tuning-Seeds -- damit "deference" keine Einzelbeobachtung bleibt.
#
# Das Paper stuetzt Finding 2 bisher auf genau einen LoRA-Lauf. Ohne
# Seed-Varianz laesst sich nicht sagen, wie viel der Praezisionssprung
# (18 % -> 64 % im Band [10,20)) Effekt und wie viel Rauschen ist.
#
# SLURM-Array: ein Seed pro Task, laeuft parallel wenn Slots frei sind.
#     sbatch bender_finetune_seeds.sbatch
#
# Ergebnis: out/qwen3-4b-nel-lora-s{1,2,3}-merged
# Danach:   sbatch bender_eval_ft_seeds.sbatch
#
# Braucht das gepatchte src/finetune_lora.py mit --seed; der Job prueft das.
# ==========================================================================
#SBATCH --job-name=nel-lora-seeds
#SBATCH --partition=A40medium
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --array=1-3
#SBATCH --exclude=node-02
#SBATCH --output=logs/lora-seed-%A_%a.out

set -e
mkdir -p logs out
module load Python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

PY=~/NEL_LLM_pipeline/ft_env/bin/python3
cd ~/NEL_LLM_pipeline
SEED=$SLURM_ARRAY_TASK_ID
OUT=out/qwen3-4b-nel-lora-s${SEED}

$PY -c "
import sys, subprocess
h = subprocess.run([sys.executable,'src/finetune_lora.py','--help'],
                   capture_output=True, text=True).stdout
sys.exit(0 if '--seed' in h else
         'src/finetune_lora.py kennt --seed nicht -- gepatchte Fassung rsyncen.')
"

echo "Knoten: ${SLURMD_NODENAME:-$(hostname)}"
ECC=$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total \
      --format=csv,noheader 2>/dev/null | head -1 | tr -d ' []' || true)
case "$ECC" in
    ""|"N/A"|"NotSupported"|0) ;;
    *) echo ">>> ABBRUCH: GPU meldet $ECC ECC-Fehler."; exit 1 ;;
esac

# "Verzeichnis existiert" reicht als Pruefung NICHT. Laeuft das Quota beim
# Schreiben des Merges voll, meldet der Job COMPLETED und hinterlaesst ein
# abgeschnittenes Verzeichnis -- typisch eine 0-Byte tokenizer.json, die erst
# Stunden spaeter beim Servieren als JSONDecodeError auffaellt.
merged_ok () {                      # merged_ok <dir>
    local d="$1" f bytes
    [ -d "$d" ] || return 1
    for f in config.json tokenizer.json tokenizer_config.json; do
        [ -s "$d/$f" ] || return 1
    done
    bytes=$(du -sb "$d" 2>/dev/null | cut -f1)
    [ "${bytes:-0}" -gt 6000000000 ]        # 4B in fp16 sind rund 8 GB
}

if merged_ok "${OUT}-merged"; then
    echo "${OUT}-merged ist vollstaendig -- Seed $SEED uebersprungen."
    exit 0
fi
if [ -d "${OUT}-merged" ]; then
    echo "${OUT}-merged ist unvollstaendig und wird verworfen."
    rm -rf "${OUT}-merged"
fi

echo "=== LoRA, Seed $SEED ==="
$PY src/finetune_lora.py \
  --model Qwen/Qwen3-4B \
  --train Data/finetune/bc5cdr_train_sft.jsonl \
  --eval  Data/finetune/bc5cdr_dev_sft.jsonl \
  --out   "$OUT" \
  --seed  "$SEED" \
  --epochs 3 --batch-size 1 --grad-accum 32 --lr 2e-4 \
  --load-in-4bit --merge

echo "Fertig: ${OUT}-merged"
