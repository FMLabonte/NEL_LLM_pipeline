#!/bin/bash
# ==========================================================================
# Einen LoRA-Adapter nachtraeglich mergen -- ohne neu zu trainieren.
#
# Noetig, wenn cluster/bender_finetune_seeds.sbatch beim Schreiben des gemergten
# Modells ins Quota gelaufen ist: der Job meldet COMPLETED, hinterlaesst aber
# ein abgeschnittenes Verzeichnis (typisch: 0-Byte tokenizer.json), das erst
# Stunden spaeter beim Servieren als JSONDecodeError auffaellt.
#
#     sbatch --array=3 cluster/bender_merge_seed.sbatch      # nur Seed 3
#     sbatch --array=1-3 cluster/bender_merge_seed.sbatch    # alle pruefen/neu bauen
#
# Braucht KEINE GPU -- nur RAM. Danach:
#     sbatch --array=3 cluster/bender_eval_ft_seeds.sbatch
# ==========================================================================
#SBATCH --job-name=nel-merge
#SBATCH --partition=A40devel
#SBATCH --time=00:30:00
#SBATCH --mem=48G
#SBATCH --array=1-3
#SBATCH --output=logs/merge-%A_%a.out

set -e
mkdir -p logs
module load Python
cd ~/NEL_LLM_pipeline
PY=~/NEL_LLM_pipeline/ft_env/bin/python3
SEED=$SLURM_ARRAY_TASK_ID

# Ein 4B-Modell in fp16 braucht rund 8 GB auf der Platte. Vorher testen, sonst
# entsteht wieder genau die abgeschnittene Kopie, die wir hier reparieren.
if ! dd if=/dev/zero of=.merge_probe_$SEED bs=1M count=9000 status=none 2>/dev/null; then
    rm -f .merge_probe_$SEED
    echo ">>> ABBRUCH: weniger als 9 GB frei. 'quota -s' pruefen und aufraeumen."
    exit 1
fi
rm -f .merge_probe_$SEED

$PY scripts/merge_lora.py \
    --adapter "out/qwen3-4b-nel-lora-s${SEED}" \
    --base Qwen/Qwen3-4B \
    --force

echo "Seed $SEED gemergt."
