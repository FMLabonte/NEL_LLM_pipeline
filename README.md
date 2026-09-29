# NEL_LLM_pipeline

When does an LLM help in biomedical entity linking? This repository contains an
open-source re-implementation of the BioLinkerAI pipeline
([Sakor et al., 2024](https://link.springer.com/chapter/10.1007/978-981-96-0573-6_19))
and the experiments we ran with it in the NLP Lab (University of Bonn, summer
semester 2026).

A retriever proposes a ranked list of ten candidate concepts for each mention,
and a disambiguator (an LLM or a cross-encoder) selects one of them. We use the
pipeline as a measuring instrument: the mentions, the candidate lists and the
prompt stay fixed, and one component changes at a time. The experiments cover
fourteen retriever configurations on five corpora, a controlled degradation of
the retriever, three LLMs, LoRA fine-tuning and a MedCPT cross-encoder.

## Repository structure

```
Data/                 corpora (included) and knowledge bases (MeSH, UMLS: download yourself)
src/                  the pipeline
  candidate-generation/   Phase 2: MeSH/UMLS index, string and embedding retrieval, expansion
  domain-rules/           Phase 3: rule-based re-ranking
  llm-disambiguation/     Phase 4: prompts and the LLM disambiguator
  linguistic-rules/       Phase 1: mention detection (not used with gold mentions)
  improvements/           extensions (abbreviations, few-shot retrieval, ...)
  evaluate_pipeline.py    main entry point: runs Phases 2-4 and writes per-mention predictions
  finetune_lora.py        LoRA fine-tuning of the disambiguator
  analyze_*.py            analyses on the prediction dumps
scripts/              stand-alone tools (BioSyn, cross-encoder, LoRA merge, GRPO pilot, release)
  figures/                all figure scripts for the paper and the lab report (see its README)
cluster/              SLURM jobs for the Bender cluster (submit from the repository root)
docs/                 working notes: cluster runs, fine-tuning, next steps
gui/                  small Flask web interface (see "Web interface" below)
pubtator_parser.py    loader for PubTator files (used by the whole pipeline)
```

Not tracked, but created locally: `figures/` (figure output), `preds_*.jsonl`
(per-mention prediction dumps), `models/` and `out/` (fine-tuned weights) and
the retrieval caches in `src/candidate-generation/cache/`.

## Setup

Python 3.11 is tested.

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt        # lite run and figures
pip install -r requirements-gpu.txt    # embeddings, BM25, fine-tuning, cross-encoder, vLLM
```

### Data

The five test corpora are included in `Data/` (BC5CDR, BioRED, NCBI-Disease,
NLM-Chem, MedMentions ST21pv). The knowledge bases are not, and have to be
downloaded separately.

**MeSH (required, free).** Download the 2026 descriptor and supplementary
concept files in XML format from the
[NLM MeSH download page](https://www.nlm.nih.gov/databases/download/mesh.html)
and place them as

```
Data/MeSH/desc2026.xml
Data/MeSH/supp2026.xml
```

**UMLS (optional, license required).** The UMLS Metathesaurus is only
available with a free UMLS license, which has to be requested from the NLM
with a [UTS account](https://uts.nlm.nih.gov/uts/signup-login). Approval can
take a few days. After approval, download the Metathesaurus files from the
[UMLS Knowledge Sources page](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html)
and place the RRF files in `Data/UMLS/`:

```
Data/UMLS/MRCONSO.RRF    synonyms (--umls) and the UMLS index for MedMentions (--umls-index)
Data/UMLS/MRREL.RRF      relation-based candidate expansion
Data/UMLS/MRDEF.RRF      definitions (--mrdef, optional)
Data/UMLS/MRSTY.RRF      semantic types (optional)
```

UMLS data and everything derived from it (for example the caches in
`src/candidate-generation/cache/`) must not be shared or committed. The
`.gitignore` already excludes them.

Without UMLS, the pipeline still runs on the four MeSH corpora (see the lite
run below). It then skips the UMLS synonyms and the UMLS-based candidate
expansion, so accuracy is somewhat lower than in our reported runs.
MedMentions needs UMLS.

## Quick start (lite run: no UMLS, no GPU)

Retriever only (Phases 2 and 3), 100 BioRED mentions. This needs only MeSH and
takes less than a minute on a laptop.

```bash
python3 src/evaluate_pipeline.py --dataset biored --no-phase4 \
    --limit 100 --keep-duplicates
```

The output ends with `FINAL Accuracy@1 (no LLM, Phase 3)` and a breakdown of
the errors into missing candidates and wrong ranking.

With an LLM as disambiguator, any OpenAI-compatible endpoint works. With
[LM Studio](https://lmstudio.ai), load a model (we use Qwen3-4B), start the
local server on port 1234 and pass the model identifier shown in LM Studio:

```bash
python3 src/evaluate_pipeline.py --dataset biored --limit 100 --keep-duplicates \
    --model qwen3-4b --base-url http://localhost:1234/v1 \
    --llm-top-k 10 --prompt-version v8 --temperature 0 --max-tokens 256 \
    --structured-output --dump-predictions preds_test.jsonl
```

`--dump-predictions` writes one line per mention with the retriever's answer,
the final answer and the retriever's confidence. All analyses and figures are
computed from such files.

Available datasets: `bc5cdr`, `biored`, `ncbi`, `nlm_chem`, `medmentions`.

## Web interface

`gui/` contains a small Flask app for trying the pipeline on a single abstract
or PDF without the command line.

```bash
pip install flask PyMuPDF      # PyMuPDF is only needed for PDF upload
python3 gui/app.py
```

Then open http://localhost:5555 in the browser.

1. Choose the pipeline components in the left panel and click "Initialize
   Pipeline". This builds the MeSH index and takes about a minute. If you only
   installed `requirements.txt`, switch off "SapBERT Embeddings" first, since
   it needs the packages from `requirements-gpu.txt`.
2. In the tab "Entity Linking", paste a text or upload a PDF, enter the
   mentions to link (one per line or comma-separated) and click "Link
   Entities". "Load Example" fills in an example abstract. The result lists
   the ranked MeSH candidates with their scores for each mention.
3. In the tab "Evaluation", run the retriever (Phases 2 and 3, without the
   LLM) on BC5CDR, BioRED or MedMentions, optionally on a limited number of
   mentions.

The switch "LLM Disambiguation" sends the candidates to an LM Studio server at
`localhost:1234` with the model `qwen3.5-9b`. To use another model or
endpoint, change `llm_model` and `llm_base_url` in `init_pipeline()` in
`gui/app.py`. The web interface uses the interactive pipeline in
`src/pipeline.py` and is meant for exploration. The reported numbers come from
`src/evaluate_pipeline.py` with the settings above.

## Full runs

The reported runs used all enrichments, SapBERT embeddings and Qwen3-4B served
with vLLM on a single A40 GPU:

```bash
vllm serve Qwen/Qwen3-4B --served-model-name qwen3-4b --port 8000 \
    --dtype bfloat16 --max-model-len 8192

python3 src/evaluate_pipeline.py --dataset biored \
    --model qwen3-4b --base-url http://localhost:8000/v1 --api-key dummy \
    --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
    --llm-top-k 10 --prompt-version v8 --temperature 0 --max-tokens 256 \
    --structured-output --keep-duplicates --dump-predictions preds_zs_biored.jsonl
```

`--wikidata` and `--dbpedia` query the public SPARQL endpoints on the first run
and cache the results. On the Bender cluster, the jobs in `cluster/` run these
commands. Submit them from the repository root, for example
`sbatch cluster/bender_eval_biored.sbatch`.

| Experiment | Jobs in `cluster/` |
|---|---|
| Zero-shot runs per corpus | `bender_eval_*.sbatch`, `bender_medm_emb.sbatch` |
| Other retrievers (SapBERT, BM25, lexical, BioSyn) | `bender_eval_curve_retrievers.sbatch`, `bender_eval_sapbert_only.sbatch`, `bender_biored_lex.sbatch`, `bender_setup_biosyn.sh`, `bender_biosyn_score.sbatch` |
| Controlled degradation | `bender_retriever_dose.sbatch` |
| Prompt variants | `bender_prompt_ablation.sbatch` |
| Qwen3-8B and Granite-3.1-8B | `bender_multimodel.sbatch` |
| LoRA fine-tuning, three seeds | `bender_finetune_seeds.sbatch`, `bender_merge_seed.sbatch`, `bender_eval_ft_seeds.sbatch` |
| MedCPT cross-encoder | `bender_dump_candidates.sbatch`, `bender_crossencoder.sbatch` |
| GRPO pilot | `bender_grpo_pilot.sbatch` |

`docs/CLUSTER_EVAL.md` and `docs/FINETUNING.md` describe the cluster setup in
more detail.

## Figures

All figures of the report are computed from the `preds_*.jsonl` dumps, without
running a model:

```bash
python3 scripts/figures/make_fig_frame.py     # -> figures/fig_frame.pdf
```

`scripts/figures/README.md` lists which script produces which figure. The
dumps themselves are not in the repository because of their size; they can be
reproduced with the runs above.

## Team

Moritz Gilges, Snehpreet Dhinsa and Mohamed Abouelella. NLP Lab (CAISA Lab),
University of Bonn. Supervisor: Frederik Labonté.
