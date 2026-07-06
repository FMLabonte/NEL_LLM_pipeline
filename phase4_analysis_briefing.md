# Phase 4 (LLM Disambiguation) Analysis & Briefing

## Project Context

We are an NLP Lab team (University of Bonn) reproducing and extending **BioLinkerAI** (https://link.springer.com/chapter/10.1007/978-981-96-0573-6_19), a neuro-symbolic pipeline for biomedical Named Entity Linking (NEL). The goal: given a mention like "CF" in a PubMed abstract, link it to the correct MeSH/UMLS concept (e.g., D003550 = Cystic Fibrosis, not D003550 = Carbon Fiber).

## Pipeline Architecture

The pipeline has 4 phases. Entities are already annotated (gold entities from the dataset) — we only do the **linking** step.

- **Phase 2 (Candidate Generation):** Given a mention, retrieve top-k candidate MeSH entities using rapidfuzz string similarity + synonym enrichment (UMLS, Wikidata, DBpedia) + SapBERT embedding retrieval (FAISS). Output: ranked list of ~10-30 candidates with match scores.
- **Phase 2b (Candidate Expansion):** Expand candidate list using UMLS relation graphs, multi-word decomposition, and parent concept injection. Brings additional candidates into the pool.
- **Phase 3 (Domain Rules):** Re-rank candidates using hard-coded rules: abbreviation expansion (rule-based + LLM fallback), definition-based scoring, semantic type filtering, confidence from multiple knowledge graphs. Output: re-ranked candidates with updated scores.
- **Phase 4 (LLM Disambiguation):** Take the top-k re-ranked candidates + document context → send to an LLM → LLM selects the best candidate. This is the phase we're trying to improve.

## The Problem: Phase 4 Adds Very Little

BioLinkerAI's paper reports that their LLM (GPT-4) adds a huge boost:

| Dataset | Without LLM | With LLM (GPT-4) | LLM Gain |
|---------|-------------|-------------------|----------|
| BC5CDR | 80.4% | 93.3% | **+12.9%** |
| MedMentions | 65.5% | 81.3% | **+15.8%** |

Our pipeline's LLM gain is much smaller:

| Dataset | Phase 3 (no LLM) | Phase 4 (with LLM) | LLM Gain |
|---------|-------------------|---------------------|----------|
| BC5CDR (GPT-4o-mini, v3, full 2625) | 79.3% | 81.9% | **+2.6%** |
| BC5CDR (GPT-5-mini, v2, 400) | 74.5% | 76.0% | **+1.5%** |
| BC5CDR (GPT-5-mini, v3, 400) | 74.5% | 73.8% | **-0.8%** (worse!) |

So our LLM adds +1.5% to +2.6% instead of BioLinkerAI's +12.9%. The LLM is barely helping.

## What We've Tried (Prompt Versions)

We have a prompt registry (`src/llm-disambiguation/prompts.py`) with 4 versions:

### V1 — Original
- System prompt asks for step-by-step reasoning + candidate number
- Sentence-level context (not full abstract)
- Match scores shown, no few-shot examples
- NONE option allowed
- Response format: number

### V2 — Concise
- System prompt: concise, no reasoning, just pick a number
- Sentence-level context
- Match scores shown, few-shot examples, categories
- Response format: number
- **Best performer so far**

### V3 — JSON + Full Abstract (T.A. Suggestion)
- System prompt asks for JSON output: `{"entity_name": "...", "mesh_id": "..."}`
- **Full abstract** instead of sentence snippet
- Match scores **hidden** (to avoid position bias)
- Few-shot examples, categories
- Response format: json_entity
- **Performed worse than v2** — the LLM actually degraded accuracy below Phase 3 baseline with GPT-5-mini

### V4 — Best-of-Both (just created, untested)
- JSON output format (cleaner parsing, 0 parse fails)
- Match scores **included** (critical ranking signal from Phase 3)
- **Sentence-level context** (less noise than full abstract)
- Few-shot examples, categories
- Response format: json_entity

## Key Findings from Experiments

### 1. The LLM degrades as many mentions as it fixes

On BC5CDR (full 2625 mentions, GPT-4o-mini, v3):
- **157 improvements** (Phase 3 wrong → Phase 4 correct)
- **89 degradations** (Phase 3 correct → Phase 4 wrong)
- Net: only +68 out of 2625

On BC5CDR (400 mentions, GPT-5-mini, v3):
- 26 improvements, **29 degradations** — net **negative**!

### 2. Common degradation patterns

The LLM frequently confuses semantically close concepts:
- "depressive disorder" (D003866) → flips to "Depressive Disorder, Major" (D003865)
- "anxiety" (D001008) → flips to "Anxiety Disorders" (D001007)
- "corticosteroid" (D000305 = Adrenal Cortex Hormones) → flips to "Glucocorticoids" (D005938)
- "aspartate" (D001224) → flips to "Aspartic Acid" (D001220)

These are all cases where the LLM "reasons" itself from the correct general concept to a more specific (wrong) subconcept.

### 3. Hiding match scores hurts

V3 hides match scores to avoid "position bias." But the scores are a valuable signal: Phase 3's rule-based ranking is often correct, and the LLM should trust it unless it has strong contextual evidence to override. Without scores, the LLM treats all candidates equally and makes more wrong switches.

### 4. Full abstract context doesn't help (and may hurt)

V3 passes the full abstract. V2 passes just a ~200-char sentence snippet. Result: full abstract = more tokens, more cost, and worse accuracy. The extra context introduces noise and gives the LLM spurious associations.

### 5. Thinking mode can hurt

GPT-5-mini (with thinking/chain-of-thought) performed worse than GPT-4o-mini (without thinking) on the same 400 mentions. More reasoning ≠ better answers for this task. The model overthinks and talks itself out of correct answers.

### 6. Candidate recall ceiling limits everything

| k | Phase 2 | Phase 2+2b | Phase 2b+3 |
|---|---------|------------|------------|
| 1 | 77.8% | 77.7% | 79.3% |
| 5 | 88.0% | 88.0% | 89.6% |
| 10 | 90.1% | 90.2% | 91.5% |
| 20 | 91.2% | 92.2% | 92.7% |

Even a perfect LLM selecting from the top-10 candidates can only reach ~91.5%. BioLinkerAI achieves 93.3%, which means they either have better candidate generation or their LLM can somehow compensate.

## Current Prompt Structure

The user prompt sent to the LLM looks like this (v2 example):

```
## Biomedical Text
**Title:** Effects of corticosteroid therapy on cardiac function
**Text:** Patients treated with corticosteroid showed improvement in symptoms.

## Mention to Link: "corticosteroid"
**Context window:** Patients treated with **corticosteroid** showed improvement in symptoms.

## Candidates
1. **Adrenal Cortex Hormones** [D000305]
   Category: Chemicals and Drugs
   Definition: Hormones secreted by the ADRENAL CORTEX that affect numerous physiological processes.
   Synonyms: Corticosteroids, Adrenocorticosteroids, Corticoids
   Match score: 95.0

2. **Glucocorticoids** [D005938]
   Category: Chemicals and Drugs
   Definition: A group of CORTICOSTEROIDS that affect carbohydrate metabolism.
   Synonyms: Glucocorticoid, Glucocorticoid Effect
   Match score: 88.0

Which candidate best matches the mention in the given context? Reply with ONLY the number.

Examples of correct linking:
- "hypertension" in "patients with hypertension and diabetes" → Hypertensive disease [D006973], not Ocular Hypertension
- "AD" in "a mouse model of AD" → Alzheimer Disease [D000544], not Autonomic Dysreflexia
- "lithium" in "lithium treatment for bipolar disorder" → Lithium [D008094] (the element/drug), not Lithium Compounds
```

## Code Structure

- `src/llm-disambiguation/prompts.py` — Prompt registry, builder, response parsers
- `src/llm-disambiguation/llm_disambiguator.py` — API calls, retry logic, token tracking
- `src/evaluate_pipeline.py` — Full evaluation pipeline (Phase 2→2b→3→4)
- `src/candidate-generation/mesh_index.py` — MeSH knowledge base search
- `src/domain-rules/domain_rules.py` — Phase 3 re-ranking rules
- `src/improvements/abbreviation_expander.py` — Rule-based abbreviation expansion
- `src/improvements/llm_abbreviation_expander.py` — LLM-based abbreviation expansion
- `src/improvements/embedding_retriever.py` — SapBERT/FAISS embedding retrieval

## API Setup

We use a shared LiteLLM proxy (university server) that routes to commercial API models:
- URL: `http://131.220.150.238:8080`
- Models tested: `openai/gpt-4o-mini`, `openai/gpt-5-mini`, `openai/gpt-4.1-mini`
- Budget: ~$20 total, ~$8-10 remaining
- Local models via LMStudio: `qwen3-4b`, `qwen3.5-9b`

## What We Need Help With

1. **Why does our LLM add so little compared to BioLinkerAI's +12.9%?** Is it a prompt issue, a candidate quality issue, or something else entirely?

2. **How can we reduce degradations?** The LLM fixes ~6% of mentions but also breaks ~3.4%. How can we make it more conservative — only overriding Phase 3 when it's truly confident?

3. **Are there fundamentally different prompting strategies** we should try? For example:
   - Pairwise comparison instead of list selection?
   - Asking the LLM to verify Phase 3's top-1 rather than re-rank?
   - Using the LLM only for hard cases (high ambiguity)?
   - Chain-of-thought that specifically addresses the parent/child hierarchy confusion?

4. **Should we focus on candidate generation instead?** Our @10 recall ceiling is 91.5% — is that the real bottleneck?

## Constraints

- Budget: ~$8-10 remaining on the API proxy
- Timeline: University NLP lab project, need to show results to supervisor
- We want to reproduce BioLinkerAI first, then extend it
- Open-source models (Qwen3) are also available via local LMStudio but were slower and less accurate in early tests
