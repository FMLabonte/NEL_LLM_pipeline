# Phase 1: Linguistic Rules (Entity Extraction)

> **Input:** Raw biomedical text  
> **Output:** List of entity mentions with character offsets  
> **Paper reference:** Section 3, "A Rule-governed framework", Fig. 2 (left side)

## What does this do?

Takes a sentence like `"She is having a fever and her temperature is around 39"` and extracts the entity mentions: **"fever"** and **"temperature"**.

## How it works

```
Raw Text
   │
   ▼
┌──────────────────────────────┐
│  Tokenize + POS-tag (spaCy)  │
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Rule 1: Remove stopwords    │  "the", "a", "and" → gone
│  Rule 2: Remove verbs        │  "is", "having" → gone
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Rule 3: Merge adjacent      │  "body" + "temperature"
│          candidate tokens    │  → "body temperature"
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Rule 4: Merge across        │  "cancer" + "of the" + "lung"
│          prepositions        │  → "cancer of the lung"
└──────────────┬───────────────┘
               │
               ▼
       List of EntityMention
       (text, start, end)
```

## Files

| File | What it does |
|---|---|
| `rules.py` | The 4 rules as individual functions. Each one can be tested on its own. |
| `entity_extractor.py` | `LinguisticEntityExtractor` class — chains all rules together. Call `.extract(text)` to get mentions. |

## Quick usage

```python
from entity_extractor import LinguisticEntityExtractor

extractor = LinguisticEntityExtractor()
mentions = extractor.extract("cancer of the lung is a serious disease")

for m in mentions:
    print(m.text, m.start, m.end)
# → "cancer of the lung"  0  18
# → "disease"             32  39
```

## Note

For BioRED evaluation, entities are already annotated (gold entities), so this phase is skipped. It's needed when running the pipeline on raw, unannotated text.
