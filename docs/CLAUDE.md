# NEL_LLM_pipeline

## Project Overview
Open-source re-implementation of BioLinkerAI (https://link.springer.com/chapter/10.1007/978-981-96-0573-6_19), a neuro-symbolic approach for biomedical entity linking. The goal is to reproduce the pipeline and then systematically test improvements.

## Team
- Moritz, Mohamed, Snehpreet (NLP Lab, 9CP, University of Bonn)
- Supervisor: Frederik

## Pipeline Architecture (4 Phases)
The BioLinkerAI pipeline we are reproducing:

1. **Linguistic Rules (Entity Extraction)**: Rule-based system identifies surface forms/mentions in text using linguistic rules (stopwords, verbs, compound words). Based on [6,15] in the paper.
2. **Candidate Generation**: Search Background Knowledge (UMLS/MeSH) via Elasticsearch (string similarity), enrich with aliases (sameAs links), rank with BM25.
3. **Domain-Specific Rules (Re-Ranking)**: Hard-coded rules re-rank candidates (confidence score from multiple KGs, semantic type filtering, definition-based prioritization).
4. **LLM Disambiguation**: Ranked candidates + context go to an LLM (open-source: Qwen3-8B/30B) which selects the best match.

## Datasets
- **BioRED** (primary): MeSH identifiers, entities pre-annotated — in `Data/BioRED/`
- **BC5CDR**: 1500 PubMed abstracts, MeSH vocabulary — in `Data/CDR_Data/`
- **MedMentions**: 4392 abstracts, UMLS — in `Data/MedMention/`

All datasets parsed via `pubtator_parser.py` into 3 DataFrames: metadata, annotations, relations.

## BioLinkerAI Reference Numbers (Table 3 in paper)
- MedMentions: 81.3 overall (GPT-4), 79.4 (Llama2-70B)
- BC5CDR: 93.3 overall (GPT-4), 93.1 (Llama2-70B)
- Best prior baseline: BioEFG at 65.4 (unseen, BC5CDR)

## Branch Strategy
- `main` — stable, working code
- `baseline/...` — implementing the BioLinkerAI reproduction
  - `baseline/linguistic-rules` — Phase 1
  - `baseline/candidate-generation` — Phase 2
  - `baseline/domain-rules-ranking` — Phase 3
  - `baseline/llm-disambiguation` — Phase 4
  - `baseline/evaluation` — evaluation pipeline
- `improvement/...` — our own extensions (later)

## Models
- Open-source (primary): Qwen3-8B, Qwen3-30B (via LMStudio)
- Closed-source (comparison): Claude Opus 4.6, Gemini 3.1 Pro, GPT 5.5

## Key Decisions
- Evaluation on BioRED with gold entities (entities pre-annotated, we only do linking)
- Phase 1 (linguistic rules) built for completeness but not needed for BioRED evaluation
- Accuracy@1 as primary metric, also Accuracy@k and Macro-F1 by entity type

## Planned Improvements (after baseline)
- Embedding-based retrieval (SapBERT/BioLinkBERT + FAISS)
- Abbreviation expansion module
- Context-window enrichment / GRF
- Chain-of-thought prompting
- Multi-model ensemble / voting
- Confidence-based cascading (small model for easy cases)
- Taxonomy-graph consistency check
- LoRA fine-tuning
- Coreference resolution
- Learned ranking (XGBoost) instead of hand-coded rules

## Tech Stack
- Python, pandas, Elasticsearch
- LMStudio for local LLM inference
- Git/GitHub for version control
