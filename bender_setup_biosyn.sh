#!/bin/bash
# ==========================================================================
# ONE-TIME BioSyn setup on Bender (run on the LOGIN node, not sbatch).
# We DO NOT build BioSyn's ancient env (its transformers==4.11.3 pin needs a
# Rust compiler to build old `tokenizers` under Python 3.12). Instead we run
# BioSyn's INFERENCE on your existing modern eval_env via compat shims baked
# into run_biosyn.py. This script:
#   1) clones BioSyn (for the src.biosyn package)
#   2) adds scikit-learn + gdown to eval_env (BioSyn's only extra deps)
#   3) downloads the BC5CDR disease+chemical checkpoints FULLY to local dirs
#      (so sparse_encoder.pk + sparse_weight.pt are present -> no HF download)
#   4) downloads the BC5CDR dictionaries
#   5) drops our faithful runner into the repo
# ==========================================================================
set -e
module load Python
cd ~

echo "=== 1) clone BioSyn (for the src.biosyn package) ==="
[ -d BioSyn ] || git clone https://github.com/dmis-lab/BioSyn.git

echo "=== 2) extra deps into eval_env (does NOT touch torch/transformers) ==="
source ~/NEL_LLM_pipeline/eval_env/bin/activate
pip install scikit-learn gdown

cd ~/BioSyn
mkdir -p models datasets

echo "=== 3) download BioSyn checkpoints FULLY (incl. sparse files) ==="
# `huggingface-cli` is removed in huggingface_hub >=1.0; use `hf download`.
hf download dmis-lab/biosyn-sapbert-bc5cdr-disease \
    --local-dir models/biosyn-sapbert-bc5cdr-disease
hf download dmis-lab/biosyn-sapbert-bc5cdr-chemical \
    --local-dir models/biosyn-sapbert-bc5cdr-chemical
echo "sparse files present?"
ls -1 models/biosyn-sapbert-bc5cdr-disease/sparse_encoder.pk \
       models/biosyn-sapbert-bc5cdr-disease/sparse_weight.pt \
       models/biosyn-sapbert-bc5cdr-chemical/sparse_encoder.pk \
       models/biosyn-sapbert-bc5cdr-chemical/sparse_weight.pt

echo "=== 4) download BC5CDR dictionaries (disease + chemical) ==="
# NOTE: these Drive files are gzip TARBALLS despite the .zip name -> use tar.
mkdir -p datasets
gdown 1moAqukbrdpAPseJc3UELEY6NLcNk22AA -O bc5cdr-disease.tar.gz
tar xzf bc5cdr-disease.tar.gz -C datasets/
gdown 1mgQhjAjpqWLCkoxIreLnNBYcvjdsSoGi -O bc5cdr-chemical.tar.gz
tar xzf bc5cdr-chemical.tar.gz -C datasets/
ls -1 datasets/bc5cdr-disease/test_dictionary.txt datasets/bc5cdr-chemical/test_dictionary.txt

echo "=== 5) copy our faithful runner into the repo ==="
cp ~/NEL_LLM_pipeline/scripts/run_biosyn.py ~/BioSyn/run_biosyn.py

echo "Setup done. Next: dump mentions (eval_env), then sbatch bender_biosyn_score.sbatch"
