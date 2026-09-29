# Named-retriever curve points — exact steps (BM25, BioSyn; ArboEL deferred)

Goal: add **BM25** and **BioSyn** as real points on Figure 1 (Recall@1 vs LLM-gain),
each run through *our* identical zero-shot Qwen3-4B stage. Every `[MAC]` line runs on
your Mac; every `[BENDER]` line runs on the cluster. Replace `bender:` with your SSH
host/alias and adjust `module load` names if Bender differs.

ArboEL is **left out for now**: no released checkpoints, no BC5CDR support (MedMentions/
ZeShEL only), py3.7/torch1.4/cuda10.1 + Cython — a multi-day train-from-scratch on 2 big
GPUs. Not this weekend.

---

## PART A — push the updated code to Bender  `[MAC]`
```bash
cd "/Users/moritz/Desktop/NLP Lab/NEL_LLM_pipeline"
B=s24mgilg@bender.hpc.uni-bonn.de
rsync -avP src/evaluate_pipeline.py            $B:NEL_LLM_pipeline/src/
rsync -avP src/improvements/bm25_retriever.py  $B:NEL_LLM_pipeline/src/improvements/
rsync -avP scripts/run_biosyn.py               $B:NEL_LLM_pipeline/scripts/
rsync -avP cluster/bender_setup_biosyn.sh cluster/bender_biosyn_score.sbatch \
           cluster/bender_eval_curve_retrievers.sbatch  $B:NEL_LLM_pipeline/
```

---

## PART B — BM25 point (no dependency, do this first)  `[BENDER]`
```bash
cd ~/NEL_LLM_pipeline
source eval_env/bin/activate
pip install rank_bm25            # one-time
sbatch cluster/bender_eval_curve_retrievers.sbatch
```
This job serves vLLM, runs `--bm25-only`, and (if the BioSyn candidates file is not
there yet) prints a SKIP for BioSyn. Output: `preds_zs_bc5cdr_bm25.jsonl`.
First BM25 run builds+caches the index (`src/improvements/cache/bm25/`).

---

## PART C — BioSyn point

### C1. one-time setup  `[BENDER, login node]`
```bash
cd ~/NEL_LLM_pipeline
bash cluster/bender_setup_biosyn.sh
```
(clones BioSyn, makes `biosyn_env`, gdown-loads the BC5CDR disease+chemical
dictionaries, copies `run_biosyn.py` into `~/BioSyn`.)
If gdown is rate-limited, download the two zips from the BioSyn README links by hand
and unzip into `~/BioSyn/datasets/`.

### C2. dump OUR exact eval mentions  `[BENDER, login node]`
```bash
cd ~/NEL_LLM_pipeline
source eval_env/bin/activate
python3 src/evaluate_pipeline.py --dataset bc5cdr --split test \
        --keep-duplicates --no-phase4 \
        --dump-mentions mentions_bc5cdr_test.jsonl
```
Writes `mentions_bc5cdr_test.jsonl` (byte-identical to what the curve eval will look up).

### C3. score with full hybrid BioSyn  `[BENDER, GPU]`
```bash
cd ~/NEL_LLM_pipeline
sbatch cluster/bender_biosyn_score.sbatch
```
Output: `cands_biosyn_bc5cdr.jsonl` (top-10 unique-CUI candidates per mention,
disease/chemical routed to their own checkpoints).

### C4. run our LLM stage on BioSyn candidates  `[BENDER, GPU]`
```bash
cd ~/NEL_LLM_pipeline
sbatch cluster/bender_eval_curve_retrievers.sbatch     # re-submit: now BioSyn step runs too
```
Output: `preds_zs_bc5cdr_biosyn.jsonl` (BM25 will just re-run harmlessly).

---

## PART D — pull predictions back to the Mac  `[MAC]`
```bash
cd "/Users/moritz/Desktop/NLP Lab/NEL_LLM_pipeline"
rsync -avP s24mgilg@bender.hpc.uni-bonn.de:'NEL_LLM_pipeline/preds_zs_bc5cdr_bm25.jsonl NEL_LLM_pipeline/preds_zs_bc5cdr_biosyn.jsonl' .
```

---

## PART E — compute each curve point + danger-zone bands  `[MAC]`
```bash
cd "/Users/moritz/Desktop/NLP Lab/NEL_LLM_pipeline"
# each preds file carries P3 (retriever top-1) and P4 (final). In the printout:
#   x-axis  Recall@1 = "base P3"   |   y-axis  gain = full P4 - base P3
python3 src/finding1_analysis.py \
    preds_zs_bc5cdr_bm25.jsonl preds_zs_bc5cdr_biosyn.jsonl \
    --labels BM25,BioSyn --out points_bm25_biosyn.json
```
Send me `points_bm25_biosyn.json` (or paste the two printed summaries) and I'll drop the
named points onto Figure 1 and update the Frame text.

---

### Sanity checks before trusting a point
- BioSyn CUI coverage: the runner emits MeSH ids (`D…/C…`); if many mentions get 0
  candidates, the dictionary path is wrong. `wc -l cands_biosyn_bc5cdr.jsonl` should equal
  `wc -l mentions_bc5cdr_test.jsonl`.
- BM25 Recall@1 on BC5CDR should land clearly *below* SapBERT (~lexical territory); if it's
  ~0, `rank_bm25` didn't install or the MeSH index is empty.
- Both preds files must have the same mention count as the SapBERT-only run for a fair
  overlay.
