"""
MeSH Search Index
==================
Builds and manages an in-memory search index over MeSH entities.

Parses the MeSH descriptor XML (desc2026.xml) and supplementary
concepts XML (supp2026.xml) into a searchable index. For each entity,
stores: ID, preferred label, synonyms, definition, and tree numbers.

Search is done via fuzzy string matching (rapidfuzz) against all
labels and synonyms. This can later be swapped out for Elasticsearch
for better performance on large-scale data.

Usage:
    index = MeSHIndex()
    index.build_from_xml("Data/MeSH/desc2026.xml")
    results = index.search("famotidine", top_k=10)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    # Fallback to difflib if rapidfuzz is not installed
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False
    print("Warning: rapidfuzz not installed, using difflib (slower). "
          "Install with: pip install rapidfuzz")


# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class MeSHEntity:
    """
    A single entity from the MeSH knowledge base.

    Attributes
    ----------
    mesh_id : str
        Unique identifier (e.g., "D015738" for descriptors, "C467567" for supplements).
    preferred_label : str
        The official/preferred name (e.g., "Famotidine").
    synonyms : list[str]
        Alternative names including the preferred label
        (e.g., ["Famotidine", "Pepcid", "MK-208"]).
    definition : str
        Scope note / definition from MeSH.
    tree_numbers : list[str]
        Hierarchical classification codes (e.g., ["D03.438.221.173"]).
    """
    mesh_id: str
    preferred_label: str
    synonyms: list[str] = field(default_factory=list)
    definition: str = ""
    tree_numbers: list[str] = field(default_factory=list)


@dataclass
class CandidateEntity:
    """
    A candidate entity returned by the search index,
    including a relevance score.
    """
    mesh_id: str
    preferred_label: str
    synonyms: list[str]
    definition: str
    tree_numbers: list[str]
    score: float              # similarity score (0-100)
    matched_synonym: str      # which label/synonym was matched


# ── Main index class ───────────────────────────────────────────────────────

class MeSHIndex:
    """
    In-memory search index over MeSH entities.

    Builds a lookup from all labels/synonyms to their MeSH entities,
    then uses fuzzy string matching to find candidates for a given mention.

    Parameters
    ----------
    case_sensitive : bool
        Whether search should be case-sensitive. Default False.
    """

    def __init__(self, case_sensitive: bool = False):
        self.case_sensitive = case_sensitive

        # mesh_id → MeSHEntity
        self.entities: dict[str, MeSHEntity] = {}

        # Flat list of (normalized_label, mesh_id) for search
        # Each synonym gets its own entry pointing to the entity
        self._label_index: list[tuple[str, str]] = []

    @property
    def size(self) -> int:
        """Number of entities in the index."""
        return len(self.entities)

    @property
    def label_count(self) -> int:
        """Total number of searchable labels (including synonyms)."""
        return len(self._label_index)

    # ── Building the index ─────────────────────────────────────────────

    def build_from_xml(
        self,
        descriptor_path: str | None = None,
        supplementary_path: str | None = None,
    ):
        """
        Parse MeSH XML file(s) and build the search index.

        Parameters
        ----------
        descriptor_path : str or None
            Path to desc2026.xml (main MeSH descriptors).
        supplementary_path : str or None
            Path to supp2026.xml (supplementary concepts, mostly chemicals).
        """
        if descriptor_path:
            self._parse_descriptors(descriptor_path)
        if supplementary_path:
            self._parse_supplementary(supplementary_path)

        # Build the flat label index for fast searching
        self._build_label_index()

        print(f"MeSH index built: {self.size} entities, {self.label_count} searchable labels")

    def _parse_descriptors(self, path: str):
        """Parse MeSH descriptor XML (desc2026.xml)."""
        print(f"Parsing descriptors from {path}...")
        count = 0

        for event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag != "DescriptorRecord":
                continue

            mesh_id = elem.find("DescriptorUI").text
            preferred = elem.find("DescriptorName/String").text

            # Collect all synonyms (entry terms) from all concepts
            synonyms = list(set(
                term.text for term in elem.findall(".//Term/String")
                if term.text
            ))

            # Scope note (definition)
            scope_elem = elem.find(".//ScopeNote")
            definition = scope_elem.text.strip() if scope_elem is not None and scope_elem.text else ""

            # Tree numbers (semantic hierarchy)
            tree_numbers = [t.text for t in elem.findall(".//TreeNumber") if t.text]

            self.entities[mesh_id] = MeSHEntity(
                mesh_id=mesh_id,
                preferred_label=preferred,
                synonyms=synonyms,
                definition=definition,
                tree_numbers=tree_numbers,
            )

            # Free memory as we go (important for large XML files)
            elem.clear()

            # Progress indicator
            count += 1
            if count % 5000 == 0:
                print(f"  ... {count} descriptors parsed", flush=True)

        print(f"  Done: {count} descriptors loaded.")

    def _parse_supplementary(self, path: str):
        """Parse MeSH supplementary concepts XML (supp2026.xml)."""
        print(f"Parsing supplementary concepts from {path}...")
        count = 0

        for event, elem in ET.iterparse(path, events=("end",)):
            if elem.tag != "SupplementalRecord":
                continue

            mesh_id = elem.find("SupplementalRecordUI").text
            preferred = elem.find("SupplementalRecordName/String").text

            synonyms = list(set(
                term.text for term in elem.findall(".//Term/String")
                if term.text
            ))

            scope_elem = elem.find(".//ScopeNote")
            definition = scope_elem.text.strip() if scope_elem is not None and scope_elem.text else ""

            self.entities[mesh_id] = MeSHEntity(
                mesh_id=mesh_id,
                preferred_label=preferred,
                synonyms=synonyms,
                definition=definition,
                tree_numbers=[],  # supplementary concepts don't have tree numbers
            )

            elem.clear()

            # Progress indicator (supplementary has ~324K records, so update more often)
            count += 1
            if count % 25000 == 0:
                print(f"  ... {count} supplementary concepts parsed", flush=True)

        print(f"  Done: {count} supplementary concepts loaded.")

    def _build_label_index(self):
        """
        Build the flat label index for search.
        Maps every label/synonym to its mesh_id for fast lookup.
        """
        self._label_index = []
        for mesh_id, entity in self.entities.items():
            for synonym in entity.synonyms:
                normalized = synonym if self.case_sensitive else synonym.lower()
                self._label_index.append((normalized, mesh_id))

    # ── Searching ──────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[CandidateEntity]:
        """
        Search for candidate entities matching a mention.

        Uses fuzzy string matching against all labels/synonyms in the index.
        Exact matches get the highest score (100).

        Parameters
        ----------
        query : str
            The entity mention to search for (e.g., "famotidine").
        top_k : int
            Number of top candidates to return.

        Returns
        -------
        list[CandidateEntity]
            Top-k candidates sorted by score (descending).
        """
        if not self._label_index:
            raise RuntimeError("Index is empty. Call build_from_xml() first.")

        normalized_query = query if self.case_sensitive else query.lower()

        if HAS_RAPIDFUZZ:
            return self._search_rapidfuzz(normalized_query, top_k)
        else:
            return self._search_difflib(normalized_query, top_k)

    def _search_rapidfuzz(self, query: str, top_k: int) -> list[CandidateEntity]:
        """Search using rapidfuzz (fast fuzzy matching)."""
        # Extract all labels for rapidfuzz process
        labels = [label for label, _ in self._label_index]

        # Find best matches using token_sort_ratio which handles
        # word order differences (e.g., "lung cancer" vs "cancer, lung")
        results = process.extract(
            query,
            labels,
            scorer=fuzz.token_sort_ratio,
            limit=top_k * 3,  # get more than needed, then deduplicate by mesh_id
        )

        # Deduplicate: keep the best score per mesh_id
        best_per_entity: dict[str, tuple[float, str]] = {}
        for matched_label, score, idx in results:
            mesh_id = self._label_index[idx][1]
            if mesh_id not in best_per_entity or score > best_per_entity[mesh_id][0]:
                best_per_entity[mesh_id] = (score, matched_label)

        # Build CandidateEntity objects, sorted by score
        candidates = []
        for mesh_id, (score, matched_label) in sorted(
            best_per_entity.items(), key=lambda x: x[1][0], reverse=True
        )[:top_k]:
            entity = self.entities[mesh_id]
            candidates.append(CandidateEntity(
                mesh_id=mesh_id,
                preferred_label=entity.preferred_label,
                synonyms=entity.synonyms,
                definition=entity.definition,
                tree_numbers=entity.tree_numbers,
                score=score,
                matched_synonym=matched_label,
            ))

        return candidates

    def _search_difflib(self, query: str, top_k: int) -> list[CandidateEntity]:
        """Fallback search using difflib (slower but no dependencies)."""
        scores: dict[str, tuple[float, str]] = {}

        for label, mesh_id in self._label_index:
            ratio = SequenceMatcher(None, query, label).ratio() * 100

            if mesh_id not in scores or ratio > scores[mesh_id][0]:
                scores[mesh_id] = (ratio, label)

        # Sort by score and take top-k
        top = sorted(scores.items(), key=lambda x: x[1][0], reverse=True)[:top_k]

        candidates = []
        for mesh_id, (score, matched_label) in top:
            entity = self.entities[mesh_id]
            candidates.append(CandidateEntity(
                mesh_id=mesh_id,
                preferred_label=entity.preferred_label,
                synonyms=entity.synonyms,
                definition=entity.definition,
                tree_numbers=entity.tree_numbers,
                score=score,
                matched_synonym=matched_label,
            ))

        return candidates

    def lookup(self, mesh_id: str) -> MeSHEntity | None:
        """
        Direct lookup of an entity by its MeSH ID.

        Parameters
        ----------
        mesh_id : str
            The MeSH identifier (e.g., "D015738").

        Returns
        -------
        MeSHEntity or None
        """
        return self.entities.get(mesh_id)


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    index = MeSHIndex()

    # Build from MeSH XML — adjust paths as needed
    t0 = time.time()
    index.build_from_xml(
        descriptor_path="../../Data/MeSH/desc2026.xml",
        supplementary_path="../../Data/MeSH/supp2026.xml",
    )
    print(f"Index built in {time.time() - t0:.1f}s\n")

    # Test searches
    test_queries = ["famotidine", "delirium", "seizures", "IDM", "doxorubicin"]

    for query in test_queries:
        print(f'Search: "{query}"')
        t0 = time.time()
        results = index.search(query, top_k=5)
        elapsed = time.time() - t0

        for i, c in enumerate(results):
            print(f"  {i+1}. [{c.mesh_id}] {c.preferred_label} "
                  f"(score={c.score:.1f}, matched='{c.matched_synonym}')")
        print(f"  ({elapsed*1000:.0f}ms)\n")
