"""
BioLinkerAI Reproduction Pipeline
=================================
Main pipeline that chains all 4 phases:
  Phase 1: Linguistic Rules (Entity Extraction)
  Phase 2: Candidate Generation (Elasticsearch + BM25)
  Phase 3: Domain-Specific Rules (Re-Ranking)
  Phase 4: LLM Disambiguation

Each phase is implemented in its own module under src/.
This file orchestrates the full pipeline and provides
both a gold-entity mode (for BioRED evaluation) and
a raw-text mode (using Phase 1 for entity extraction).
"""


def run_pipeline(text: str, gold_entities: list | None = None):
    """
    Run the full entity linking pipeline.

    Parameters
    ----------
    text : str
        The input biomedical text (e.g., a PubMed abstract).
    gold_entities : list or None
        If provided, skip Phase 1 and use these pre-annotated entities.
        Each entry should have: text, start, end, entity_type.

    Returns
    -------
    list of dict
        Each dict contains: mention, predicted_mesh_id, confidence, candidates.
    """

    # --- Phase 1: Entity Extraction ---
    if gold_entities is not None:
        mentions = gold_entities
    else:
        # TODO: Import and use LinguisticEntityExtractor
        raise NotImplementedError("Phase 1 (linguistic rules) not yet integrated")

    # --- Phase 2: Candidate Generation ---
    # TODO: For each mention, search Background Knowledge (Elasticsearch)
    #       Enrich with aliases, rank with BM25

    # --- Phase 3: Domain-Specific Rules ---
    # TODO: Re-rank candidates using domain-specific rules (Rules 5-7)

    # --- Phase 4: LLM Disambiguation ---
    # TODO: Pass ranked candidates + context to LLM, get best match

    raise NotImplementedError("Pipeline phases 2-4 not yet implemented")
