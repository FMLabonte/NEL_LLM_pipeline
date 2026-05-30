"""
Hybrid Scorer: Combines String Similarity + Embedding Similarity
=================================================================
Re-scores candidates using a weighted combination of:
  - rapidfuzz string similarity (token_sort_ratio, 0-100)
  - SapBERT embedding cosine similarity (scaled to 0-100)

The intuition: rapidfuzz is great for exact/near-exact matches
("seizures" → "Seizures"), but fails on semantic matches
("heart attack" → "Myocardial Infarction"). Embedding similarity
captures semantic meaning but may rank irrelevant-but-similar
concepts too high. The combination gets the best of both.

Formula:
    hybrid_score = α × string_score + (1 - α) × embedding_score

Where α controls the balance:
  - α = 1.0: pure string matching (current baseline)
  - α = 0.0: pure embedding similarity
  - α = 0.7: default, favoring string match but with semantic boost

Usage:
    from hybrid_scorer import HybridScorer

    scorer = HybridScorer(embedding_retriever, alpha=0.7)
    reranked = scorer.rescore(mention, candidates)
"""


class HybridScorer:
    """
    Combines rapidfuzz string similarity and embedding similarity
    into a single hybrid score for candidate re-ranking.

    Parameters
    ----------
    embedding_retriever : EmbeddingRetriever
        The embedding retriever (must be initialized with model loaded).
    alpha : float
        Weight for string similarity score (0-1).
        alpha=0.7 means 70% string + 30% embedding.
    """

    def __init__(self, embedding_retriever, alpha: float = 0.7):
        self.embedding_retriever = embedding_retriever
        self.alpha = alpha

    def rescore(self, mention: str, candidates: list) -> list:
        """
        Re-score candidates using hybrid string + embedding similarity.

        Parameters
        ----------
        mention : str
            The entity mention text.
        candidates : list[CandidateEntity]
            Candidates with rapidfuzz scores.

        Returns
        -------
        list[CandidateEntity]
            Re-ranked candidates with updated hybrid scores.
        """
        if not candidates:
            return candidates

        alpha = self.alpha

        # Get embedding similarity for all candidates
        emb_scores = self.embedding_retriever.score_candidates(mention, candidates)

        # Compute hybrid scores
        for c in candidates:
            string_score = c.score  # rapidfuzz score (0-100)
            embedding_score = emb_scores.get(c.mesh_id, 0.0)  # embedding score (0-100)

            c.score = alpha * string_score + (1 - alpha) * embedding_score

        # Sort by hybrid score (descending)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def rescore_with_details(self, mention: str, candidates: list) -> list[dict]:
        """
        Like rescore(), but returns detailed scoring info.

        Returns list of dicts with: candidate, string_score,
        embedding_score, hybrid_score.
        """
        if not candidates:
            return []

        emb_scores = self.embedding_retriever.score_candidates(mention, candidates)

        results = []
        for c in candidates:
            string_score = c.score
            embedding_score = emb_scores.get(c.mesh_id, 0.0)
            hybrid_score = self.alpha * string_score + (1 - self.alpha) * embedding_score

            results.append({
                "candidate": c,
                "string_score": string_score,
                "embedding_score": embedding_score,
                "hybrid_score": hybrid_score,
            })

            c.score = hybrid_score

        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return results
