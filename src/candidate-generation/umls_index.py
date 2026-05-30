"""
UMLS Search Index
==================
Builds and manages a search index over UMLS concepts from MRCONSO.RRF.

Unlike MeSHIndex (which only covers ~350k MeSH entries), the UMLS index
aggregates synonyms from multiple vocabularies (SNOMED-CT, NCI, CHV, etc.)
and can contain millions of entries. This provides:
  - Much broader synonym coverage per concept
  - Direct CUI-based matching (no CUI→MeSH mapping needed for MedMentions)
  - For MeSH-based datasets: CUI results are mapped back to MeSH IDs

Search uses a two-stage approach for speed:
  1. Token-based pre-filtering (inverted index) to find candidate labels
  2. RapidFuzz scoring on the filtered candidates only

Usage:
    index = UMLSIndex()
    index.build_from_mrconso("Data/UMLS/MRCONSO.RRF")
    results = index.search("myocardial infarction", top_k=10)

    # With vocabulary filter:
    index = UMLSIndex(vocabularies=["MSH", "SNOMEDCT_US", "NCI"])
    index.build_from_mrconso("Data/UMLS/MRCONSO.RRF")
"""

import os
import re
import json
import time
import pickle
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

# Import CandidateEntity from mesh_index so both indexes return the same type
from mesh_index import CandidateEntity


# ── Default vocabulary set ─────────────────────────────────────────────────
# These are the most useful biomedical vocabularies for entity linking.
# MSH = MeSH, SNOMEDCT_US = SNOMED-CT, NCI = NCI Thesaurus,
# CHV = Consumer Health Vocabulary, MTH = UMLS Metathesaurus names,
# OMIM = Online Mendelian Inheritance in Man, HPO = Human Phenotype Ontology,
# RXNORM = RxNorm (drugs)
DEFAULT_VOCABULARIES = [
    "MSH",           # MeSH — our primary ontology
    "SNOMEDCT_US",   # SNOMED-CT — very comprehensive clinical terms
    "NCI",           # NCI Thesaurus — cancer/disease terms
    "CHV",           # Consumer Health Vocabulary — lay synonyms
    "MTH",           # UMLS Metathesaurus preferred names
    "OMIM",          # Genetic disorders
    "HPO",           # Human Phenotype Ontology
    "RXNORM",        # Drug names
]


@dataclass
class UMLSEntity:
    """
    A single UMLS concept.

    Attributes
    ----------
    cui : str
        Concept Unique Identifier (e.g., "C0027051").
    preferred_label : str
        The preferred name (from MTH or first encountered).
    synonyms : list[str]
        All unique names/synonyms across included vocabularies.
    semantic_types : list[str]
        Semantic type abbreviations (if available from MRSTY).
    source_vocabularies : list[str]
        Which vocabularies contributed synonyms (e.g., ["MSH", "SNOMEDCT_US"]).
    mesh_ids : list[str]
        MeSH descriptor IDs mapped to this CUI (from SAB=MSH rows).
    """
    cui: str
    preferred_label: str
    synonyms: list[str] = field(default_factory=list)
    semantic_types: list[str] = field(default_factory=list)
    source_vocabularies: list[str] = field(default_factory=list)
    mesh_ids: list[str] = field(default_factory=list)


class UMLSIndex:
    """
    Search index over UMLS concepts from MRCONSO.RRF.

    Uses a two-stage search approach:
      1. Inverted token index for fast pre-filtering
      2. RapidFuzz scoring on filtered candidates

    Parameters
    ----------
    vocabularies : list[str] | None
        Which UMLS source vocabularies to include. None = DEFAULT_VOCABULARIES.
        Use ["ALL"] to include all English entries (very large, ~10M rows).
    cache_dir : str | None
        Directory for caching the built index. Defaults to MRCONSO parent dir.
    min_token_length : int
        Minimum token length for the inverted index (default 2).
    """

    def __init__(
        self,
        vocabularies: list[str] | None = None,
        cache_dir: str | None = None,
        min_token_length: int = 2,
    ):
        if not HAS_RAPIDFUZZ:
            print("Warning: rapidfuzz not installed. UMLSIndex.search() will not work. "
                  "Install with: pip install rapidfuzz")

        self.vocabularies = vocabularies or DEFAULT_VOCABULARIES
        self.cache_dir = cache_dir
        self.min_token_length = min_token_length
        self.return_cui = False  # Set to True for CUI-level evaluation (MedMentions)

        # CUI → UMLSEntity
        self.entities: dict[str, UMLSEntity] = {}

        # Flat list of (normalized_label, cui) for search
        self._label_index: list[tuple[str, str]] = []

        # Inverted token index: token → set of indices into _label_index
        self._token_index: dict[str, set[int]] = defaultdict(set)

        # CUI → set of MeSH IDs (for mapping back to MeSH)
        self._cui_to_mesh: dict[str, set[str]] = defaultdict(set)

        # MeSH ID → CUI (reverse mapping)
        self._mesh_to_cui: dict[str, str] = {}

    def build_from_mrconso(self, mrconso_path: str):
        """
        Build the UMLS index from MRCONSO.RRF.

        Tries to load from cache first. If cache miss, parses the full file.
        """
        mrconso_path = str(Path(mrconso_path).resolve())
        if not Path(mrconso_path).exists():
            raise FileNotFoundError(f"MRCONSO.RRF not found: {mrconso_path}")

        # Determine cache path
        if self.cache_dir:
            cache_base = Path(self.cache_dir)
        else:
            cache_base = Path(mrconso_path).parent

        # Cache key includes vocabulary selection
        vocab_key = "_".join(sorted(self.vocabularies))
        cache_hash = hashlib.md5(vocab_key.encode()).hexdigest()[:8]
        cache_path = cache_base / f"umls_index_cache_{cache_hash}.pkl"

        # Try loading from cache
        if cache_path.exists():
            cache_mtime = cache_path.stat().st_mtime
            source_mtime = Path(mrconso_path).stat().st_mtime
            if cache_mtime > source_mtime:
                print(f"  Loading UMLS index from cache: {cache_path.name}")
                try:
                    self._load_cache(cache_path)
                    print(f"  Loaded {len(self.entities):,} concepts, "
                          f"{len(self._label_index):,} labels")
                    return
                except Exception as e:
                    print(f"  Cache load failed ({e}), rebuilding...")

        # Build from MRCONSO.RRF
        print(f"  Building UMLS index from MRCONSO.RRF...")
        if "ALL" in self.vocabularies:
            print(f"  Including ALL English vocabularies")
        else:
            print(f"  Vocabularies: {', '.join(self.vocabularies)}")

        t0 = time.time()
        self._parse_mrconso(mrconso_path)
        t1 = time.time()

        print(f"  Parsed {len(self.entities):,} concepts, "
              f"{len(self._label_index):,} labels in {t1-t0:.1f}s")

        # Build inverted token index
        print(f"  Building token index...")
        self._build_token_index()
        t2 = time.time()
        print(f"  Token index built in {t2-t1:.1f}s "
              f"({len(self._token_index):,} unique tokens)")

        # Save cache
        print(f"  Saving cache to {cache_path.name}...")
        self._save_cache(cache_path)
        t3 = time.time()
        print(f"  Cache saved in {t3-t2:.1f}s")

    def _parse_mrconso(self, mrconso_path: str):
        """Parse MRCONSO.RRF and build entity + label structures."""
        include_all = "ALL" in self.vocabularies
        vocab_set = set(self.vocabularies)

        # Temporary structures
        cui_labels: dict[str, dict] = {}  # cui → {"preferred": str, "synonyms": set, "vocabs": set, "mesh_ids": set}

        with open(mrconso_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line_num % 2_000_000 == 0:
                    print(f"    ... processed {line_num:,} rows")

                parts = line.strip().split("|")
                if len(parts) < 15:
                    continue

                cui = parts[0]    # CUI
                lang = parts[1]   # LAT (language)
                ts = parts[2]     # TS (term status: P=preferred, S=synonym)
                sab = parts[11]   # SAB (source vocabulary)
                tty = parts[12]   # TTY (term type)
                label = parts[14] # STR (string)
                sdui = parts[10]  # SDUI (source descriptor UI — MeSH ID for SAB=MSH)

                # Filter: English only
                if lang != "ENG":
                    continue

                # Filter by vocabulary (unless ALL)
                if not include_all and sab not in vocab_set:
                    continue

                # Skip empty labels
                if not label or not label.strip():
                    continue

                # Initialize CUI entry
                if cui not in cui_labels:
                    cui_labels[cui] = {
                        "preferred": None,
                        "synonyms": set(),
                        "vocabs": set(),
                        "mesh_ids": set(),
                    }

                entry = cui_labels[cui]
                entry["synonyms"].add(label)
                entry["vocabs"].add(sab)

                # Track preferred name (TS=P or TTY=MH from MeSH)
                if entry["preferred"] is None:
                    if ts == "P" or tty in ("MH", "NM", "PT", "PN"):
                        entry["preferred"] = label

                # Track MeSH ID mapping
                if sab == "MSH" and sdui and sdui.strip():
                    entry["mesh_ids"].add(sdui.strip())

        # Build entities and label index
        label_set = set()  # track (normalized_label, cui) to deduplicate
        for cui, data in cui_labels.items():
            preferred = data["preferred"] or next(iter(data["synonyms"]))
            synonyms = sorted(data["synonyms"])
            mesh_ids = sorted(data["mesh_ids"])
            vocabs = sorted(data["vocabs"])

            entity = UMLSEntity(
                cui=cui,
                preferred_label=preferred,
                synonyms=synonyms,
                source_vocabularies=vocabs,
                mesh_ids=mesh_ids,
            )
            self.entities[cui] = entity

            # CUI↔MeSH mappings
            for mid in mesh_ids:
                self._cui_to_mesh[cui].add(mid)
                self._mesh_to_cui[mid] = cui

            # Build label index (deduplicated by normalized form)
            for label in synonyms:
                norm = label.lower().strip()
                key = (norm, cui)
                if key not in label_set:
                    label_set.add(key)
                    self._label_index.append((norm, cui))

    def _build_token_index(self):
        """Build inverted token index for fast pre-filtering."""
        self._token_index = defaultdict(set)
        tokenize = self._tokenize

        for idx, (label, _) in enumerate(self._label_index):
            for token in tokenize(label):
                self._token_index[token].add(idx)

    def _tokenize(self, text: str) -> set[str]:
        """Tokenize a string into lowercase alphanumeric tokens."""
        tokens = re.findall(r'[a-z0-9]+', text.lower())
        return {t for t in tokens if len(t) >= self.min_token_length}

    def _save_cache(self, cache_path: Path):
        """Save the built index to a pickle file."""
        data = {
            "entities": self.entities,
            "label_index": self._label_index,
            "token_index": dict(self._token_index),  # convert defaultdict
            "cui_to_mesh": dict(self._cui_to_mesh),
            "mesh_to_cui": self._mesh_to_cui,
        }
        with open(cache_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    def _load_cache(self, cache_path: Path):
        """Load index from pickle cache."""
        with open(cache_path, "rb") as f:
            data = pickle.load(f)

        self.entities = data["entities"]
        self._label_index = data["label_index"]
        self._token_index = defaultdict(set, data["token_index"])
        self._cui_to_mesh = defaultdict(set, data["cui_to_mesh"])
        self._mesh_to_cui = data["mesh_to_cui"]

    # ── Search ──────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10, return_cui: bool = False) -> list[CandidateEntity]:
        """
        Search for UMLS concepts matching the query.

        Uses two-stage retrieval:
          1. Token overlap to find candidate labels (fast)
          2. RapidFuzz scoring on candidates (precise)

        Parameters
        ----------
        return_cui : bool
            If True, always use CUI as the mesh_id field (for MedMentions evaluation).
            If False (default), use MeSH ID when available, CUI otherwise.
        """
        if not self._label_index:
            raise RuntimeError("Index is empty. Call build_from_mrconso() first.")

        # Stage 1: Token-based pre-filtering
        query_tokens = self._tokenize(query.lower())
        if not query_tokens:
            return []

        # Find label indices that share at least one token with the query
        candidate_indices = set()
        for token in query_tokens:
            if token in self._token_index:
                candidate_indices.update(self._token_index[token])

        if not candidate_indices:
            return []

        # Stage 2: Score with rapidfuzz
        candidate_labels = [(self._label_index[i][0], i) for i in candidate_indices]

        # Score each candidate
        scored: list[tuple[float, int, str]] = []
        query_lower = query.lower().strip()
        for label, idx in candidate_labels:
            score = fuzz.token_sort_ratio(query_lower, label)
            if score >= 40:  # minimum threshold
                scored.append((score, idx, label))

        # Sort by score descending
        scored.sort(key=lambda x: x[0], reverse=True)

        # Deduplicate by CUI: keep best score per CUI
        best_per_cui: dict[str, tuple[float, str]] = {}
        for score, idx, matched_label in scored:
            cui = self._label_index[idx][1]
            if cui not in best_per_cui or score > best_per_cui[cui][0]:
                best_per_cui[cui] = (score, matched_label)

        # Build CandidateEntity objects
        candidates = []
        for cui, (score, matched_label) in sorted(
            best_per_cui.items(), key=lambda x: x[1][0], reverse=True
        )[:top_k]:
            entity = self.entities[cui]

            # Use MeSH ID if available (unless return_cui mode), otherwise CUI
            use_cui = return_cui or self.return_cui
            if use_cui:
                entity_id = cui
            else:
                entity_id = entity.mesh_ids[0] if entity.mesh_ids else cui

            candidates.append(CandidateEntity(
                mesh_id=entity_id,  # MeSH ID or CUI
                preferred_label=entity.preferred_label,
                synonyms=entity.synonyms[:20],  # limit for memory
                definition="",  # MRCONSO doesn't have definitions (MRDEF does)
                tree_numbers=[],  # not available in UMLS directly
                score=score,
                matched_synonym=matched_label,
            ))

        return candidates

    def search_cui(self, query: str, top_k: int = 10) -> list[tuple[str, str, float]]:
        """
        Search and return raw CUI results.
        Returns list of (cui, matched_label, score).
        """
        if not self._label_index:
            raise RuntimeError("Index is empty. Call build_from_mrconso() first.")

        query_tokens = self._tokenize(query.lower())
        if not query_tokens:
            return []

        candidate_indices = set()
        for token in query_tokens:
            if token in self._token_index:
                candidate_indices.update(self._token_index[token])

        if not candidate_indices:
            return []

        query_lower = query.lower().strip()
        best_per_cui: dict[str, tuple[float, str]] = {}

        for idx in candidate_indices:
            label, cui = self._label_index[idx]
            score = fuzz.token_sort_ratio(query_lower, label)
            if score >= 40:
                if cui not in best_per_cui or score > best_per_cui[cui][0]:
                    best_per_cui[cui] = (score, label)

        results = sorted(best_per_cui.items(), key=lambda x: x[1][0], reverse=True)[:top_k]
        return [(cui, label, score) for cui, (score, label) in results]

    # ── Mapping utilities ───────────────────────────────────────────────────

    def cui_to_mesh(self, cui: str) -> set[str]:
        """Get MeSH IDs for a given CUI. Returns empty set if no mapping."""
        return self._cui_to_mesh.get(cui, set())

    def mesh_to_cui(self, mesh_id: str) -> str | None:
        """Get CUI for a given MeSH ID. Returns None if not found."""
        return self._mesh_to_cui.get(mesh_id)

    def lookup(self, cui: str) -> UMLSEntity | None:
        """Look up a concept by CUI."""
        return self.entities.get(cui)

    @property
    def size(self) -> int:
        """Number of concepts in the index (compatible with MeSHIndex.size)."""
        return len(self.entities)

    def __len__(self) -> int:
        return len(self.entities)

    def __contains__(self, cui: str) -> bool:
        return cui in self.entities

    # ── Statistics ──────────────────────────────────────────────────────────

    def print_stats(self):
        """Print index statistics."""
        total_labels = len(self._label_index)
        total_cuis = len(self.entities)
        cuis_with_mesh = sum(1 for e in self.entities.values() if e.mesh_ids)
        avg_synonyms = sum(len(e.synonyms) for e in self.entities.values()) / max(total_cuis, 1)

        print(f"\nUMLS Index Statistics:")
        print(f"  Total concepts (CUIs):   {total_cuis:,}")
        print(f"  Total labels:            {total_labels:,}")
        print(f"  CUIs with MeSH mapping:  {cuis_with_mesh:,} "
              f"({100*cuis_with_mesh/max(total_cuis,1):.1f}%)")
        print(f"  Avg synonyms per CUI:    {avg_synonyms:.1f}")
        print(f"  Token index entries:     {len(self._token_index):,}")

        # Vocabulary breakdown
        vocab_counts: dict[str, int] = defaultdict(int)
        for e in self.entities.values():
            for v in e.source_vocabularies:
                vocab_counts[v] += 1
        print(f"  Vocabulary coverage:")
        for v, c in sorted(vocab_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {v:20s} {c:>8,} concepts")
