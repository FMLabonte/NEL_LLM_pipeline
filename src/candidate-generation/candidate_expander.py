"""
Candidate Expander
===================
Post-retrieval candidate expansion strategies for Phase 2.
Adds candidates that pure fuzzy string matching misses.

Implements the following strategies (based on T.A. feedback):

  1. UMLS Bridge Expansion:
     If the mention exists in UMLS but not in MeSH, use MRREL relations
     to find related MeSH entities and add them as candidates.

  2. Multi-Word Decomposition:
     For multi-word mentions, search each significant word individually.
     If a MeSH entity appears in multiple sub-searches, it gets boosted.
     Helps with mentions like "scleroderma renal crisis" where individual
     words might find relevant candidates.

  3. Parent Category Injection:
     If multiple top candidates share a MeSH tree prefix, add the parent
     category as a candidate. Helps when the gold is a broader term
     than what fuzzy matching finds.

Usage:
    from candidate_expander import CandidateExpander
    from umls_relation_expander import UMLSRelationExpander

    bridge = UMLSRelationExpander(mrconso, mrrel)
    bridge.build_bridge()

    expander = CandidateExpander(index, retriever, umls_bridge=bridge)
    expanded = expander.expand(
        mention="scleroderma renal crisis",
        candidates=phase2_candidates,
        entity_type="Disease",
    )
"""

from collections import defaultdict

from mesh_index import MeSHIndex, CandidateEntity
from candidate_retriever import CandidateRetriever


# Stopwords to skip in multi-word decomposition
_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "to", "for", "is",
    "are", "was", "were", "be", "been", "has", "have", "had", "with",
    "by", "at", "from", "on", "as", "not", "but", "its", "it", "this",
    "that", "which", "may", "can", "also", "more", "than", "other",
    "such", "used", "using", "use", "one", "two", "these", "those",
    "type", "induced", "related", "associated", "due", "like",
    "acute", "chronic", "severe", "mild", "early", "late",
}


class CandidateExpander:
    """
    Expands the candidate list using multiple strategies.

    Parameters
    ----------
    index : MeSHIndex
        The MeSH search index.
    retriever : CandidateRetriever
        The candidate retriever for sub-searches.
    umls_bridge : UMLSRelationExpander or None
        UMLS relation bridge for cross-KB expansion.
    bridge_score : float
        Score assigned to UMLS bridge candidates (default: 60.0).
    subword_score_factor : float
        Score multiplier for sub-word search results (default: 0.6).
    parent_score : float
        Score assigned to injected parent candidates (default: 55.0).
    min_word_len : int
        Minimum word length for multi-word decomposition (default: 3).
    """

    def __init__(
        self,
        index: MeSHIndex,
        retriever: CandidateRetriever,
        umls_bridge=None,
        bridge_score: float = 60.0,
        subword_score_factor: float = 0.6,
        parent_score: float = 55.0,
        min_word_len: int = 3,
    ):
        self.index = index
        self.retriever = retriever
        self.umls_bridge = umls_bridge
        self.bridge_score = bridge_score
        self.subword_score_factor = subword_score_factor
        self.parent_score = parent_score
        self.min_word_len = min_word_len

    def expand(
        self,
        mention: str,
        candidates: list[CandidateEntity],
        entity_type: str | None = None,
        top_k: int = 10,
    ) -> list[CandidateEntity]:
        """
        Apply all expansion strategies and return an expanded candidate list.

        The original candidates keep their scores. New candidates from
        expansion strategies get lower scores so they don't override
        high-confidence fuzzy matches, but can fill in when fuzzy matching fails.

        Parameters
        ----------
        mention : str
            The entity mention text.
        candidates : list[CandidateEntity]
            Initial candidates from Phase 2 fuzzy search.
        entity_type : str or None
            Entity type for filtering (e.g., "Disease", "Chemical").
        top_k : int
            Max number of candidates to return.

        Returns
        -------
        list[CandidateEntity]
            Expanded candidate list, sorted by score (descending).
        """
        existing_ids = {c.mesh_id for c in candidates}
        new_candidates = []

        # ── Strategy 1: UMLS Bridge Expansion ──
        bridge_candidates = self._umls_bridge_expansion(mention, existing_ids)
        new_candidates.extend(bridge_candidates)
        existing_ids.update(c.mesh_id for c in bridge_candidates)

        # ── Strategy 2: Multi-Word Decomposition ──
        multiword_candidates = self._multiword_expansion(mention, existing_ids)
        new_candidates.extend(multiword_candidates)
        existing_ids.update(c.mesh_id for c in multiword_candidates)

        # ── Strategy 3: Parent Category Injection ──
        parent_candidates = self._parent_injection(candidates, existing_ids)
        new_candidates.extend(parent_candidates)

        # Merge: original candidates + new candidates, sort by score
        all_candidates = list(candidates) + new_candidates

        # Deduplicate (keep highest score for each mesh_id)
        seen = {}
        for c in all_candidates:
            if c.mesh_id not in seen or c.score > seen[c.mesh_id].score:
                seen[c.mesh_id] = c
        merged = sorted(seen.values(), key=lambda c: c.score, reverse=True)

        return merged[:top_k]

    # ── Strategy 1: UMLS Bridge ──────────────────────────────────────────

    def _umls_bridge_expansion(
        self,
        mention: str,
        existing_ids: set[str],
    ) -> list[CandidateEntity]:
        """
        Look up the mention in the UMLS bridge to find related MeSH entities.
        These are entities reachable via MRREL relations (parent, broader, etc.)
        from the mention's UMLS concept.
        """
        if self.umls_bridge is None:
            return []

        related_mesh_ids = self.umls_bridge.lookup(mention)
        if not related_mesh_ids:
            return []

        candidates = []
        for mesh_id in related_mesh_ids:
            if mesh_id in existing_ids:
                continue

            entity = self.index.entities.get(mesh_id)
            if entity is None:
                continue

            candidates.append(CandidateEntity(
                mesh_id=mesh_id,
                preferred_label=entity.preferred_label,
                synonyms=entity.synonyms,
                definition=entity.definition,
                tree_numbers=entity.tree_numbers,
                score=self.bridge_score,
                matched_synonym=f"[UMLS bridge: {mention}]",
            ))

        return candidates

    # ── Strategy 2: Multi-Word Decomposition ─────────────────────────────

    def _multiword_expansion(
        self,
        mention: str,
        existing_ids: set[str],
    ) -> list[CandidateEntity]:
        """
        For multi-word mentions, search each significant word individually.
        Candidates found via multiple sub-words get a score boost.
        """
        words = mention.lower().split()
        if len(words) < 2:
            return []

        # Filter to significant words
        significant_words = [
            w for w in words
            if len(w) >= self.min_word_len and w not in _STOPWORDS
        ]
        if not significant_words:
            return []

        # Search each word and track which MeSH IDs appear
        mesh_id_hits: dict[str, list[CandidateEntity]] = defaultdict(list)
        mesh_id_word_count: dict[str, int] = defaultdict(int)

        for word in significant_words:
            sub_candidates = self.retriever.retrieve(word, top_k=5)
            seen_for_word = set()
            for c in sub_candidates:
                if c.mesh_id not in existing_ids and c.mesh_id not in seen_for_word:
                    seen_for_word.add(c.mesh_id)
                    mesh_id_hits[c.mesh_id].append(c)
                    mesh_id_word_count[c.mesh_id] += 1

        # Build candidates: boost entities found by multiple words
        candidates = []
        for mesh_id, hit_list in mesh_id_hits.items():
            best_hit = max(hit_list, key=lambda c: c.score)
            word_count = mesh_id_word_count[mesh_id]

            # Score: base score * factor, boosted by word count
            score = best_hit.score * self.subword_score_factor
            if word_count >= 2:
                score *= (1.0 + 0.3 * (word_count - 1))  # +30% per additional word match

            candidates.append(CandidateEntity(
                mesh_id=mesh_id,
                preferred_label=best_hit.preferred_label,
                synonyms=best_hit.synonyms,
                definition=best_hit.definition,
                tree_numbers=best_hit.tree_numbers,
                score=score,
                matched_synonym=f"[sub-word: {word_count} words matched]",
            ))

        # Sort by score, return top results
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:5]

    # ── Strategy 3: Parent Category Injection ────────────────────────────

    def _parent_injection(
        self,
        candidates: list[CandidateEntity],
        existing_ids: set[str],
    ) -> list[CandidateEntity]:
        """
        If multiple top candidates share a MeSH tree prefix,
        inject the parent category as an additional candidate.

        Example: if top-10 has candidates in C12.777.419.570 and C12.777.419.340,
        their common parent C12.777.419 = "Kidney Diseases" gets added.
        """
        if not candidates:
            return []

        # Count how many top candidates share each tree prefix
        tree_prefix_count: dict[str, int] = defaultdict(int)

        for c in candidates[:10]:
            entity = self.index.entities.get(c.mesh_id)
            if entity is None:
                continue
            for tn in entity.tree_numbers:
                parts = tn.split(".")
                # Track each ancestor prefix (skip the full tree number itself)
                for depth in range(1, len(parts)):
                    prefix = ".".join(parts[:depth])
                    tree_prefix_count[prefix] += 1

        # Find prefixes shared by 2+ candidates (potential parent categories)
        new_candidates = []
        injected_ids = set()

        # Sort by specificity (longer prefix = more specific parent)
        sorted_prefixes = sorted(
            tree_prefix_count.items(),
            key=lambda x: (-len(x[0].split(".")), -x[1]),
        )

        for prefix, count in sorted_prefixes:
            if count < 2:
                continue

            # Find the entity with this tree number
            for mesh_id, entity in self.index.entities.items():
                if prefix in entity.tree_numbers:
                    if mesh_id in existing_ids or mesh_id in injected_ids:
                        break

                    # Only inject if reasonably specific (depth >= 2)
                    if len(prefix.split(".")) >= 2:
                        new_candidates.append(CandidateEntity(
                            mesh_id=mesh_id,
                            preferred_label=entity.preferred_label,
                            synonyms=entity.synonyms,
                            definition=entity.definition,
                            tree_numbers=entity.tree_numbers,
                            score=self.parent_score + count,  # higher count → higher score
                            matched_synonym=f"[parent of {count} candidates: {prefix}]",
                        ))
                        injected_ids.add(mesh_id)
                    break

            # Limit to 3 parent injections
            if len(new_candidates) >= 3:
                break

        return new_candidates


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))

    from mesh_index import MeSHIndex
    from candidate_retriever import CandidateRetriever

    print("Building MeSH index...")
    index = MeSHIndex(backend="rapidfuzz")
    index.build_from_xml(
        descriptor_path=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
        supplementary_path=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
    )
    retriever = CandidateRetriever(index, top_k=10)

    # Try without UMLS bridge first (multi-word + parent injection only)
    expander = CandidateExpander(index, retriever)

    test_cases = [
        ("scleroderma renal crisis", "D007674", "Disease"),
        ("myelosuppression", "D001855", "Disease"),
        ("psychotic symptoms", "D011605", "Disease"),
    ]

    for mention, gold_id, entity_type in test_cases:
        print(f'\n{"=" * 60}')
        print(f'Mention: "{mention}" (gold: {gold_id})')

        # Phase 2 candidates
        candidates = retriever.retrieve(mention, top_k=10)
        print(f"\nPhase 2 top-3:")
        for c in candidates[:3]:
            marker = " ◀ GOLD" if c.mesh_id == gold_id else ""
            print(f"  [{c.mesh_id}] {c.preferred_label} ({c.score:.1f}){marker}")

        # Expanded candidates
        expanded = expander.expand(mention, candidates, entity_type, top_k=15)
        print(f"\nExpanded top-5:")
        for c in expanded[:5]:
            marker = " ◀ GOLD" if c.mesh_id == gold_id else ""
            print(f"  [{c.mesh_id}] {c.preferred_label} ({c.score:.1f}, {c.matched_synonym}){marker}")

        gold_in_expanded = any(c.mesh_id == gold_id for c in expanded)
        print(f"\nGold in expanded: {'✓' if gold_in_expanded else '✗'}")
