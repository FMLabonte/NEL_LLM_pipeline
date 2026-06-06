"""
Sentence-Context Scorer for Disease Re-Ranking
================================================
Re-ranks Disease candidates by comparing the full sentence context
(not just the mention) against candidate labels using SapBERT.

Why this helps: The standard hybrid scorer compares mention→candidate,
e.g., "scleroderma renal crisis" vs "Scleroderma, Systemic" and
"Kidney Diseases". Both get similar scores because the mention
partially matches both. But the SENTENCE "The patient developed
scleroderma renal crisis with acute kidney failure" contains
"kidney failure" which strongly signals toward kidney disease.

By encoding the full sentence and comparing it against candidate
labels, we capture contextual clues that pure mention matching misses.

This scorer is applied ONLY to Disease entities where the top candidates
are ambiguous (score gap below a threshold), because:
  1. Disease accuracy (74.4%) trails Chemical (85.0%) significantly
  2. Chemical mentions are usually unambiguous at the string level
  3. The sentence context is most useful for Disease disambiguation

Usage:
    from sentence_context_scorer import SentenceContextScorer

    scorer = SentenceContextScorer(embedding_retriever)
    reranked = scorer.rescore(
        mention="scleroderma renal crisis",
        candidates=candidates,
        sentence="The patient developed scleroderma renal crisis...",
        entity_type="Disease",
    )
"""

import numpy as np


class SentenceContextScorer:
    """
    Re-ranks Disease candidates using sentence-level embedding similarity.

    Uses SapBERT to encode the full sentence containing the mention,
    then computes cosine similarity against each candidate's preferred
    label. This captures contextual signals (e.g., anatomical terms,
    symptoms, co-occurring diseases) that pure mention matching misses.

    Parameters
    ----------
    embedding_retriever : EmbeddingRetriever
        The embedding retriever (must be initialized with model loaded).
        Used for its _encode_batch() method.
    weight : float
        Weight for the sentence-context score when blending with the
        existing score. Higher = more influence from context.
        Default: 5.0 (adds up to 5 points on the 0-100 scale).
    ambiguity_threshold : float
        Only apply sentence-context re-ranking when the score gap
        between top-1 and top-2 is below this threshold. If top-1
        is clearly ahead, don't risk flipping it.
        Default: 15.0 (apply when top-2 is within 15 points of top-1).
    disease_only : bool
        If True, only apply to Disease/DiseaseOrPhenotypicFeature entities.
        Default: True.
    """

    def __init__(
        self,
        embedding_retriever,
        weight: float = 5.0,
        ambiguity_threshold: float = 15.0,
        disease_only: bool = True,
    ):
        self.retriever = embedding_retriever
        self.weight = weight
        self.ambiguity_threshold = ambiguity_threshold
        self.disease_only = disease_only

    def rescore(
        self,
        mention: str,
        candidates: list,
        sentence: str,
        entity_type: str | None = None,
    ) -> list:
        """
        Re-rank candidates using sentence-level context similarity.

        Parameters
        ----------
        mention : str
            The entity mention text.
        candidates : list[CandidateEntity]
            Candidates already scored by prior phases.
        sentence : str
            The sentence containing the mention (1-3 sentences).
        entity_type : str or None
            The mention's entity type.

        Returns
        -------
        list[CandidateEntity]
            Re-ranked candidates with updated scores.
        """
        if not candidates or len(candidates) < 2:
            return candidates

        # Only apply to Disease entities (if disease_only is set)
        if self.disease_only and entity_type not in (
            "Disease", "DiseaseOrPhenotypicFeature"
        ):
            return candidates

        # Check ambiguity: only re-rank when top candidates are close
        top1_score = candidates[0].score
        top2_score = candidates[1].score
        score_gap = top1_score - top2_score

        if score_gap > self.ambiguity_threshold:
            return candidates  # top-1 is clearly ahead, don't touch

        # Encode the sentence context
        sentence_emb = self.retriever._encode_batch([sentence])  # (1, dim)

        # Encode candidate labels (preferred_label + matched synonym)
        texts_to_encode = []
        for c in candidates:
            # Use both the preferred label and the matched synonym
            # to give the model the best chance of matching
            label = c.preferred_label
            matched = getattr(c, "matched_synonym", "")
            if matched and matched.lower() != label.lower():
                label = f"{label} {matched}"
            texts_to_encode.append(label)

        label_embs = self.retriever.encode_texts(texts_to_encode)  # (n, dim)

        # Compute cosine similarity (vectors are L2-normalized)
        similarities = np.dot(label_embs, sentence_emb.T).flatten()

        # Scale to 0-100 range (cosine similarity is -1 to 1, but
        # typically 0.2-0.8 for biomedical text)
        sim_scores = (similarities + 1) * 50  # map [-1,1] → [0,100]

        # Apply as weighted boost to existing scores
        for i, c in enumerate(candidates):
            context_boost = sim_scores[i] * self.weight / 100.0
            c.score += context_boost

        # Re-sort by updated scores
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def rescore_with_details(
        self,
        mention: str,
        candidates: list,
        sentence: str,
        entity_type: str | None = None,
    ) -> list[dict]:
        """
        Like rescore(), but returns detailed scoring info for analysis.
        """
        if not candidates or len(candidates) < 2:
            return [{"candidate": c, "context_sim": 0.0, "boost": 0.0}
                    for c in candidates]

        if self.disease_only and entity_type not in (
            "Disease", "DiseaseOrPhenotypicFeature"
        ):
            return [{"candidate": c, "context_sim": 0.0, "boost": 0.0,
                      "skipped": "not_disease"} for c in candidates]

        top1_score = candidates[0].score
        top2_score = candidates[1].score
        score_gap = top1_score - top2_score

        if score_gap > self.ambiguity_threshold:
            return [{"candidate": c, "context_sim": 0.0, "boost": 0.0,
                      "skipped": "not_ambiguous"} for c in candidates]

        sentence_emb = self.retriever._encode_batch([sentence])
        texts_to_encode = []
        for c in candidates:
            label = c.preferred_label
            matched = getattr(c, "matched_synonym", "")
            if matched and matched.lower() != label.lower():
                label = f"{label} {matched}"
            texts_to_encode.append(label)

        label_embs = self.retriever.encode_texts(texts_to_encode)
        similarities = np.dot(label_embs, sentence_emb.T).flatten()
        sim_scores = (similarities + 1) * 50

        results = []
        for i, c in enumerate(candidates):
            context_boost = sim_scores[i] * self.weight / 100.0
            original_score = c.score
            c.score += context_boost
            results.append({
                "candidate": c,
                "original_score": original_score,
                "context_sim": float(similarities[i]),
                "context_score": float(sim_scores[i]),
                "boost": float(context_boost),
            })

        results.sort(key=lambda x: x["candidate"].score, reverse=True)
        candidates.sort(key=lambda c: c.score, reverse=True)
        return results


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("SentenceContextScorer requires an EmbeddingRetriever.")
    print("Run via evaluate_pipeline.py with --sentence-context flag.")
