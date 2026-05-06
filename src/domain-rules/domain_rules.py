"""
Domain-Specific Rules (Re-Ranking)
====================================
Phase 3 of the BioLinkerAI pipeline.

Applies domain-specific rules to re-rank candidate entities from Phase 2.
These rules use biomedical knowledge to boost or penalize candidates
beyond what string similarity alone can achieve.

Rules implemented (Rules 5-7 from BioLinkerAI, based on [6,15]):

  Rule 5 — Multi-KG Confidence:
    Candidates found in multiple knowledge graphs (MeSH, Wikidata, DBpedia, UMLS)
    receive a confidence boost. The intuition: if the same concept appears under
    the same name in multiple independent KBs, the match is more reliable.

  Rule 6 — Semantic Type Filtering:
    If the mention's entity type is known (e.g., "Disease" or "Chemical"),
    penalize candidates whose MeSH tree numbers indicate the wrong category.
    MeSH tree numbers encode a semantic hierarchy: C = Diseases, D = Chemicals, etc.

  Rule 7 — Definition-Based Re-Ranking:
    Boost candidates whose MeSH definition (scope note) contains keywords from
    the mention or the surrounding context. This helps when the mention text
    doesn't match the candidate name but does match its definition.

Usage:
    from domain_rules import DomainRuleReranker

    reranker = DomainRuleReranker(mesh_index)
    reranked = reranker.rerank(
        mention="scleroderma renal crisis",
        candidates=candidates,
        entity_type="Disease",
        context="The patient developed scleroderma renal crisis...",
    )

Paper reference: Section 3, "Domain-Specific Rules", Table 2 (ablation)
    Without domain rules: 87.3% → With: 93.3% on BC5CDR (+6.0%)
"""

import re
from dataclasses import dataclass

# ── MeSH Tree Number → Semantic Category Mapping ──────────────────────────
# MeSH descriptors have tree numbers like "C04.588.614" where the first
# letter encodes the top-level category. We map these to the entity types
# used in BC5CDR (Disease, Chemical) and BioRED.
#
# Full MeSH tree: https://meshb.nlm.nih.gov/treeView

TREE_TO_CATEGORY = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques",
    "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes",
    "H": "Disciplines and Occupations",
    "I": "Anthropology, Education, Sociology",
    "J": "Technology, Industry, and Agriculture",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}

# Map BC5CDR/BioRED entity types to compatible MeSH tree categories
ENTITY_TYPE_TO_TREES = {
    "Disease": {"C", "F03"},     # C = Diseases, F03 = Mental Disorders
    "Chemical": {"D"},            # D = Chemicals and Drugs
    # BioRED entity types
    "DiseaseOrPhenotypicFeature": {"C", "F03"},
    "ChemicalEntity": {"D"},
    "GeneOrGeneProduct": {"D12", "D08"},  # D12 = Amino Acids/Proteins, D08 = Enzymes
    "SequenceVariant": set(),     # no direct MeSH tree
    "OrganismTaxon": {"B"},      # B = Organisms
    "CellLine": set(),
}


def _get_candidate_categories(tree_numbers: list[str]) -> set[str]:
    """
    Extract top-level MeSH categories from tree numbers.

    Returns set of single letters (e.g., {"C", "D"}) and relevant
    sub-categories (e.g., {"F03"} for mental disorders).
    """
    categories = set()
    for tn in tree_numbers:
        if tn:
            categories.add(tn[0])           # top-level: "C", "D", etc.
            if len(tn) >= 3:
                categories.add(tn[:3])      # sub-level: "F03", "D12", etc.
    return categories


def _tokenize_simple(text: str) -> set[str]:
    """
    Simple tokenization for keyword overlap — lowercase, split on
    non-alphanumeric, filter short tokens.
    """
    tokens = re.findall(r'[a-z0-9]+', text.lower())
    # Remove very common words and short tokens
    stopwords = {
        "a", "an", "the", "and", "or", "of", "in", "to", "for", "is",
        "are", "was", "were", "be", "been", "has", "have", "had", "with",
        "by", "at", "from", "on", "as", "not", "but", "its", "it", "this",
        "that", "which", "may", "can", "also", "more", "than", "other",
        "such", "used", "using", "use", "one", "two", "these", "those",
    }
    return {t for t in tokens if len(t) > 2 and t not in stopwords}


# ── Main reranker class ───────────────────────────────────────────────────

class DomainRuleReranker:
    """
    Re-ranks Phase 2 candidates using domain-specific rules.

    Parameters
    ----------
    mesh_index : MeSHIndex
        The built MeSH index (used for entity lookups and synonym data).
    rule5_boost : float
        Score boost per additional KG source for Rule 5 (default: 5.0).
    rule6_penalty : float
        Score penalty for semantic type mismatch in Rule 6 (default: -30.0).
    rule7_boost : float
        Score boost for definition keyword overlap in Rule 7 (default: 10.0).
    wikidata_synonyms : dict or None
        MeSH ID → list of Wikidata synonyms (to check multi-KG presence).
    dbpedia_synonyms : dict or None
        MeSH ID → list of DBpedia synonyms.
    umls_synonyms : dict or None
        MeSH ID → list of UMLS synonyms.
    """

    def __init__(
        self,
        mesh_index=None,
        rule5_boost: float = 2.0,
        rule6_penalty: float = -30.0,
        rule7_boost: float = 3.0,
        top1_protection_threshold: float = 90.0,
        wikidata_synonyms: dict | None = None,
        dbpedia_synonyms: dict | None = None,
        umls_synonyms: dict | None = None,
    ):
        self.mesh_index = mesh_index
        self.rule5_boost = rule5_boost
        self.rule6_penalty = rule6_penalty
        self.rule7_boost = rule7_boost
        self.top1_protection_threshold = top1_protection_threshold
        self.wikidata_synonyms = wikidata_synonyms or {}
        self.dbpedia_synonyms = dbpedia_synonyms or {}
        self.umls_synonyms = umls_synonyms or {}

    def rerank(
        self,
        mention: str,
        candidates: list,
        entity_type: str | None = None,
        context: str = "",
    ) -> list:
        """
        Apply all domain rules and return re-ranked candidates.

        Parameters
        ----------
        mention : str
            The entity mention text.
        candidates : list[CandidateEntity]
            Ranked candidates from Phase 2.
        entity_type : str or None
            The mention's entity type (e.g., "Disease", "Chemical").
            If None, Rule 6 (semantic type filtering) is skipped.
        context : str
            Surrounding text (sentence or abstract) for Rule 7.

        Returns
        -------
        list[CandidateEntity]
            Re-ranked candidates (best first). The score field is updated
            to reflect the combined Phase 2 + Phase 3 score.
        """
        if not candidates:
            return candidates

        # Top-1 protection: if Phase 2 top-1 has a very high score
        # (near-exact match), don't let rules flip it — the string
        # match is already highly confident.
        top1_score = candidates[0].score
        protect_top1 = top1_score >= self.top1_protection_threshold

        scored = []
        for i, c in enumerate(candidates):
            adjustments = {}

            # Rule 5: Multi-KG confidence boost
            # Only apply as a relative signal: boost difference between
            # candidates, not absolute (to avoid uniform shift)
            r5 = self._rule5_multi_kg_confidence(c.mesh_id)
            adjustments["rule5"] = r5

            # Rule 6: Semantic type filtering
            r6 = self._rule6_semantic_type(c, entity_type)
            adjustments["rule6"] = r6

            # Rule 7: Definition-based re-ranking
            r7 = self._rule7_definition_overlap(c, mention, context)
            adjustments["rule7"] = r7

            # Combined score
            total_adjustment = sum(adjustments.values())

            # Top-1 protection: if protecting, give top-1 candidate
            # a bonus equal to the max possible rule adjustment,
            # so rules can only flip top-1 if they strongly disagree
            if protect_top1 and i == 0:
                total_adjustment += 15.0  # protection bonus

            new_score = c.score + total_adjustment

            scored.append((c, new_score, adjustments))

        # Sort by new score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Update candidate scores and return
        reranked = []
        for c, new_score, _ in scored:
            # Create a copy-like approach: we modify score in place
            # (CandidateEntity is a dataclass, so this is fine)
            c.score = new_score
            reranked.append(c)

        return reranked

    def rerank_with_details(
        self,
        mention: str,
        candidates: list,
        entity_type: str | None = None,
        context: str = "",
    ) -> list[dict]:
        """
        Like rerank(), but returns detailed scoring info for analysis.

        Returns
        -------
        list[dict]
            Each dict has: candidate, original_score, new_score,
            rule5, rule6, rule7 adjustments.
        """
        if not candidates:
            return []

        top1_score = candidates[0].score
        protect_top1 = top1_score >= self.top1_protection_threshold

        results = []
        for i, c in enumerate(candidates):
            original_score = c.score

            r5 = self._rule5_multi_kg_confidence(c.mesh_id)
            r6 = self._rule6_semantic_type(c, entity_type)
            r7 = self._rule7_definition_overlap(c, mention, context)

            adjustment = r5 + r6 + r7
            if protect_top1 and i == 0:
                adjustment += 15.0

            new_score = original_score + adjustment

            results.append({
                "candidate": c,
                "original_score": original_score,
                "new_score": new_score,
                "rule5_boost": r5,
                "rule6_penalty": r6,
                "rule7_boost": r7,
            })

        results.sort(key=lambda x: x["new_score"], reverse=True)
        return results

    # ── Rule 5: Multi-KG Confidence ────────────────────────────────────

    def _rule5_multi_kg_confidence(self, mesh_id: str) -> float:
        """
        Rule 5: Boost candidates found in multiple knowledge graphs.

        A candidate that appears in MeSH + Wikidata + UMLS + DBpedia is
        more likely correct than one found only in MeSH. Each additional
        KG source adds a confidence boost.

        Returns
        -------
        float
            Score adjustment (>= 0).
        """
        kg_count = 1  # MeSH itself counts as 1

        if mesh_id in self.wikidata_synonyms:
            kg_count += 1
        if mesh_id in self.dbpedia_synonyms:
            kg_count += 1
        if mesh_id in self.umls_synonyms:
            kg_count += 1

        # Boost for each additional KG beyond MeSH
        extra_kgs = kg_count - 1
        return extra_kgs * self.rule5_boost

    # ── Rule 6: Semantic Type Filtering ────────────────────────────────

    def _rule6_semantic_type(self, candidate, entity_type: str | None) -> float:
        """
        Rule 6: Penalize candidates with wrong semantic type.

        If the mention is typed (e.g., "Disease"), candidates whose MeSH
        tree numbers indicate a different category (e.g., "Chemical") are
        penalized. This prevents linking diseases to chemicals and vice versa.

        Returns
        -------
        float
            Score adjustment (<= 0 for mismatch, 0 for match or unknown).
        """
        if entity_type is None:
            return 0.0

        expected_trees = ENTITY_TYPE_TO_TREES.get(entity_type, set())
        if not expected_trees:
            return 0.0  # unknown entity type, skip

        # Get candidate's MeSH tree categories
        tree_numbers = getattr(candidate, "tree_numbers", [])
        if not tree_numbers:
            # Supplementary concepts don't have tree numbers — no penalty
            return 0.0

        candidate_cats = _get_candidate_categories(tree_numbers)

        # Check if any of the candidate's categories match the expected type
        if candidate_cats & expected_trees:
            return 0.0  # match — no penalty
        else:
            return self.rule6_penalty  # mismatch — penalize

    # ── Rule 7: Definition-Based Re-Ranking ────────────────────────────

    def _rule7_definition_overlap(
        self,
        candidate,
        mention: str,
        context: str,
    ) -> float:
        """
        Rule 7: Boost candidates whose definition overlaps with mention/context.

        If the candidate's MeSH definition (scope note) shares keywords
        with the mention text or surrounding context, it's more likely to
        be the correct match. This helps with cases where the mention text
        doesn't match the candidate name but is semantically related.

        Example: "scleroderma renal crisis" → definition of "Kidney Diseases"
        might contain "renal" or "kidney".

        Returns
        -------
        float
            Score adjustment (>= 0).
        """
        definition = getattr(candidate, "definition", "")
        if not definition:
            return 0.0

        # Tokenize mention, context, and definition
        mention_tokens = _tokenize_simple(mention)
        context_tokens = _tokenize_simple(context) if context else set()
        definition_tokens = _tokenize_simple(definition)

        if not definition_tokens:
            return 0.0

        # Check overlap between mention → definition
        mention_overlap = mention_tokens & definition_tokens
        # Check overlap between context → definition (weaker signal)
        context_overlap = context_tokens & definition_tokens

        # Score based on overlap (capped to prevent runaway boosts)
        # Mention overlap is stronger signal than context overlap
        boost = 0.0
        if mention_overlap:
            boost += min(len(mention_overlap), 2) * self.rule7_boost
        if context_overlap:
            boost += min(len(context_overlap), 2) * (self.rule7_boost * 0.3)

        # Hard cap: prevent broad definitions from dominating
        boost = min(boost, self.rule7_boost * 3)

        return boost


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataclasses import dataclass as dc, field as f

    @dc
    class FakeCandidate:
        mesh_id: str
        preferred_label: str
        synonyms: list = f(default_factory=list)
        definition: str = ""
        tree_numbers: list = f(default_factory=list)
        score: float = 0.0
        matched_synonym: str = ""

    # Simulate Phase 2 candidates for "scleroderma renal crisis" (gold: D007674 = Kidney Diseases)
    candidates = [
        FakeCandidate(
            "C000702033", "Scleroderma citrinum", [], "", ["B01.300.381"],
            score=79.1, matched_synonym="scleroderma citrinum",
        ),
        FakeCandidate(
            "D012595", "Scleroderma, Systemic", [],
            "A chronic multi-system disorder of CONNECTIVE TISSUE.",
            ["C17.300.799", "C17.800.784"],
            score=75.0, matched_synonym="scleroderma",
        ),
        FakeCandidate(
            "D007674", "Kidney Diseases", [],
            "Pathological processes of the KIDNEY or its component tissues. Includes renal insufficiency and nephropathy.",
            ["C12.777.419", "C13.351.968.419"],
            score=50.0, matched_synonym="kidney diseases",
        ),
    ]

    # Simulate enrichment data
    wikidata = {"D007674": ["Kidney disease", "Nephropathy"]}
    umls = {"D007674": ["Renal disease", "Kidney disorder"], "D012595": ["Scleroderma"]}
    dbpedia = {"D007674": ["Kidney disease"]}

    reranker = DomainRuleReranker(
        wikidata_synonyms=wikidata,
        dbpedia_synonyms=dbpedia,
        umls_synonyms=umls,
    )

    print("Before re-ranking:")
    for c in candidates:
        print(f"  [{c.mesh_id}] {c.preferred_label} (score={c.score:.1f})")

    details = reranker.rerank_with_details(
        mention="scleroderma renal crisis",
        candidates=candidates,
        entity_type="Disease",
        context="The patient developed scleroderma renal crisis with acute kidney failure.",
    )

    print("\nAfter re-ranking (with details):")
    for d in details:
        c = d["candidate"]
        print(f"  [{c.mesh_id}] {c.preferred_label}")
        print(f"    Original: {d['original_score']:.1f} → New: {d['new_score']:.1f}")
        print(f"    Rule 5 (multi-KG): {d['rule5_boost']:+.1f}")
        print(f"    Rule 6 (sem type): {d['rule6_penalty']:+.1f}")
        print(f"    Rule 7 (definition): {d['rule7_boost']:+.1f}")
