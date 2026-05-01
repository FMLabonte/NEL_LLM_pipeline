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
│  MeSHIndex.search()              │
│                                  │
│  Compares the mention against    │
│  all ~1.5M labels/synonyms in   │
│  the MeSH database using fuzzy   │
│  string matching (rapidfuzz)     │
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
| `mesh_index.py` | `MeSHIndex` class — parses MeSH XML files (desc2026.xml + supp2026.xml) and builds an in-memory search index. Core method: `.search(query, top_k)`. |
| `candidate_retriever.py` | `CandidateRetriever` class — wraps the index for pipeline use. Also contains `evaluate_candidate_recall()` to measure Accuracy@k on gold data. |

## Quick usage

```python
from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever

# Build index (once, takes ~10s)
index = MeSHIndex()
index.build_from_xml("Data/MeSH/desc2026.xml", "Data/MeSH/supp2026.xml")

# Search
retriever = CandidateRetriever(index, top_k=10)
candidates = retriever.retrieve("seizures")

for c in candidates:
    print(f"[{c.mesh_id}] {c.preferred_label} (score={c.score:.0f})")
```

## Data dependency

Requires MeSH XML files (not in git, ~700MB total):
```bash
curl -o Data/MeSH/desc2026.xml https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/desc2026.xml
curl -o Data/MeSH/supp2026.xml https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/supp2026.xml
```
