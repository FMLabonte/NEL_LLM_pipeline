# Phase 4: LLM Disambiguation

> **Input:** Ranked candidates from Phase 2 + document context  
> **Output:** Final MeSH entity ID for each mention  
> **Paper reference:** Section 3, "LLM Disambiguation"

## What does this do?

Takes the top-k candidates from Phase 2 (candidate generation) and asks an LLM to select the best match, using the paper context (title + abstract) to make an informed decision.

This is where the "neuro" part of BioLinkerAI's neuro-symbolic approach comes in: the LLM can understand that "scleroderma renal crisis" refers to a kidney disease even when there's no string overlap with "Kidney Diseases" in the MeSH vocabulary.

```
Phase 2 Candidates + Paper Context
   │
   ▼
┌──────────────────────────────────┐
│  Prompt Builder                  │
│                                  │
│  - System: biomedical expert     │
│  - Context: title + abstract     │
│  - Mention: highlighted in text  │
│  - Candidates: numbered list     │
│    with labels, definitions,     │
│    synonyms, match scores        │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  LLM API Call                    │
│                                  │
│  OpenAI-compatible API           │
│  (LMStudio on localhost:1234)    │
│  Model: qwen3-4b-2507           │
│  Temperature: 0                  │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Response Parser                 │
│                                  │
│  Extract candidate number from   │
│  LLM response. Fallback to      │
│  top-1 if parse fails.          │
└──────────────┬───────────────────┘
               │
               ▼
       Final MeSH Entity ID
```

## Files

| File | What it does |
|---|---|
| `llm_disambiguator.py` | `LLMDisambiguator` class — builds prompts, calls the LLM, parses responses. Works with any OpenAI-compatible API. |
| `evaluate.py` | End-to-end evaluation script: loads BC5CDR, runs Phase 2 + Phase 4, computes Accuracy@1, logs improvements and degradations. |
| `Implementierungsplan.md` | Implementation plan (German). |

## Prerequisites

### 1. Install the openai package

```bash
pip install openai
```

### 2. Start LMStudio

1. Open LMStudio
2. Load model: `qwen/qwen3-4b-2507` (or any other model)
3. Go to the **Developer** tab
4. Click **Start Server** — it will run on `http://localhost:1234`
5. Verify it works: `curl http://localhost:1234/v1/models`

## CLI usage

```bash
# Quick test (50 mentions)
python3 evaluate.py --model qwen3-4b-2507 --limit 50

# Full evaluation (all 2625 mentions — takes ~30-60 min with 4B model)
python3 evaluate.py --model qwen3-4b-2507

# With Phase 2 enrichment (recommended for best results)
python3 evaluate.py --model qwen3-4b-2507 --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF

# Adjust how many candidates the LLM sees
python3 evaluate.py --model qwen3-4b-2507 --top-k 10 --llm-top-k 5
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `--model` | `qwen3-4b-2507` | Model name as shown in LMStudio |
| `--base-url` | `http://localhost:1234/v1` | LLM API endpoint |
| `--temperature` | `0.0` | Sampling temperature (0 = deterministic) |
| `--top-k` | `10` | Phase 2: how many candidates to retrieve |
| `--llm-top-k` | `5` | Phase 4: how many candidates to show the LLM |
| `--limit` | (all) | Limit evaluation to N mentions (for testing) |
| `--wikidata` | off | Enrich Phase 2 with Wikidata synonyms |
| `--dbpedia` | off | Enrich Phase 2 with DBpedia labels |
| `--umls` | off | Path to MRCONSO.RRF for UMLS enrichment |

## Quick usage (Python)

```python
from llm_disambiguator import LLMDisambiguator

# Connect to LMStudio
disambiguator = LLMDisambiguator(
    model="qwen3-4b-2507",
    base_url="http://localhost:1234/v1",
)

# Disambiguate a single mention
result = disambiguator.disambiguate(
    mention="seizures",
    candidates=candidates,  # from Phase 2
    context="The patient experienced recurrent seizures after drug administration.",
    title="Adverse effects of famotidine",
)

print(f"Chosen: [{result.mesh_id}] {result.preferred_label}")
print(f"Rank: {result.chosen_rank}, Confidence: {result.confidence}")
```

## Output

The evaluation prints a summary like:

```
RESULTS
  Total mentions:          2625
  Phase 2 Accuracy@1:      72.8%
  Phase 4 Accuracy@1:      78.5%  (example — actual results will vary)
  Improvement (P4 vs P2):  +5.7%

  LLM valid choices:       2510 (95.6%)
  Fallback to top-1:       115 (4.4%)
  P4 improved over P2:     210 mentions
  P4 degraded vs P2:       60 mentions
  Net change:              +150 mentions
```

Detailed change logs are saved to `evaluation_log.json` for analysis.

## Data dependencies

Same as Phase 2 — see `src/candidate-generation/README.md` for setup instructions.
