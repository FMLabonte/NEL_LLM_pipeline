#!/bin/bash
# ==========================================================================
# EINMALIGER Build des UMLS-SapBERT/FAISS-Index (fuer MedMentions +embedding).
# Kein vLLM noetig. ~20-30 Min. Danach laedt jeder --embedding-Lauf aus Cache.
# ==========================================================================
#SBATCH --job-name=umls-faiss
#SBATCH --partition=A40short
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/faiss-build-%j.out

set -e
mkdir -p logs
module load Python
module load CUDA
source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
export TOKENIZERS_PARALLELISM=false

python3 src/build_umls_faiss.py --batch-size 512

echo "Fertig. Index unter src/improvements/cache/faiss_umls/"
