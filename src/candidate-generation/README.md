# Phase 2: Candidate Generation

> **Input:** An entity mention (e.g., `"famotidine"`)  
> **Output:** Ranked list of candidate MeSH entities with scores  
> **Paper reference:** Section 3, "Candidates Generation", Fig. 2 (center)

## What does this do?

Takes a mention like `"famotidine"` and searches the MeSH knowledge base (~355K entities) to find the most likely matches. Returns a ranked list of candidates, e.g.:

```
1. [D015738] Famotidine      (score=100)  ← correct
2. [D000069604] Famciclovir  (score=72)
3. [D016593] Terfenadine     (score=65)
...
```

## How it works

```
Entity Mention (e.g., "famotidine")
   │
   ▼
┌──────────────────────────────────┐
│  Synonym Enrichment (optional)   │
│                                  │
│  Add extra synonyms from         │
│  Wikidata, DBpedia, and/or UMLS  │
│  so the index knows more names   │
│  per entity (e.g., trade names,  │
│  Wikipedia redirects).           │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  MeSHIndex.search()              │
│                                  │
│  Backend A: rapidfuzz            │
│    → fuzzy string matching       │
│  Backend B: Elasticsearch        │
│    → BM25 scoring + fuzzy        │
└──────────────┬───────────────────┘
               │
               ▼
┌──────────────────────────────────┐
│  Deduplicate + Rank              │
│                                  │
│  If multiple synonyms point to   │
│  the same entity, keep the best  │
│  score. Return top-k results.    │
└──────────────┬───────────────────┘
               │
               ▼
       List of CandidateEntity
       (mesh_id, label, synonyms,
        definition, score)
```

## Files

| File | What it does |
|---|---|
| `mesh_index.py` | `MeSHIndex` class — parses MeSH XML files and builds a search index. Supports two backends: `rapidfuzz` (in-memory fuzzy matching) and `elasticsearch` (BM25 scoring). Core method: `.search(query, top_k)`. |
| `candidate_retriever.py` | `CandidateRetriever` class — wraps the index for pipeline use. Also contains `evaluate_candidate_recall()` to measure Accuracy@k on gold data, with MeSH ID version mapping. |
| `wikidata_enricher.py` | `WikidataEnricher` class — queries the public Wikidata SPARQL API for extra synonyms/aliases linked to MeSH IDs. No account needed. Results are cached in `wikidata_cache.json`. |
| `dbpedia_enricher.py` | `DBpediaEnricher` class — queries the public DBpedia SPARQL endpoint for labels and Wikipedia redirect titles linked to MeSH IDs. Redirects are especially useful because they contain informal/alternative names (e.g., "Heart attack" → "Myocardial infarction"). Cached in `dbpedia_cache.json`. |
| `umls_enricher.py` | `UMLSEnricher` class — parses UMLS MRCONSO.RRF to find synonyms from other vocabularies (DrugBank, SNOMED, RxNorm, etc.) via shared CUI mappings. Results are cached in `umls_cache.json`. |
| `error_analysis.py` | Analyzes which mentions fail at top-k and categorizes them (abbreviations, short mentions, no string overlap, etc.). |

## Quick usage

```python
from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever

# Build index with all enrichment sources
index = MeSHIndex(backend="rapidfuzz")  # or "elasticsearch"
index.build_from_xml(
    "Data/MeSH/desc2026.xml",
    "Data/MeSH/supp2026.xml",
    enrich_wikidata=True,                       # adds Wikidata synonyms
    enrich_dbpedia=True,                        # adds DBpedia labels + Wikipedia redirects
    enrich_umls="Data/UMLS/MRCONSO.RRF",        # adds UMLS synonyms
)

# Search
retriever = CandidateRetriever(index, top_k=10)
candidates = retriever.retrieve("seizures")

for c in candidates:
    print(f"[{c.mesh_id}] {c.preferred_label} (score={c.score:.0f})")
```

## CLI usage

```bash
# Basic (rapidfuzz only)
python3 candidate_retriever.py

# With all enrichment sources
python3 candidate_retriever.py --backend rapidfuzz --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF

# With Elasticsearch backend
python3 candidate_retriever.py --backend elasticsearch --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF

# Error analysis (shows why mentions fail)
python3 error_analysis.py --backend rapidfuzz --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF
```

## Data dependencies

**MeSH XML files** (not in git, ~700MB total):
```bash
curl -o Data/MeSH/desc2026.xml https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml
curl -o Data/MeSH/supp2026.xml https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp2026.xml
```

**UMLS MRCONSO.RRF** (optional, requires free NLM account):
Register at https://uts.nlm.nih.gov/uts/, download the Metathesaurus, and place `MRCONSO.RRF` at `Data/UMLS/MRCONSO.RRF`.

**Wikidata / DBpedia**: No download needed — queried via public SPARQL APIs and cached locally.
