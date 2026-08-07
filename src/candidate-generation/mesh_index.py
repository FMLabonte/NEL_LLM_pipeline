"""
MeSH Search Index
==================
Builds and manages a search index over MeSH entities.

Supports two search backends:
  - "rapidfuzz" (default): In-memory fuzzy string matching. No setup needed.
  - "elasticsearch": BM25-based search via a local Elasticsearch instance (e.g., via Docker).

Parses the MeSH descriptor XML (desc2026.xml) and supplementary
concepts XML (supp2026.xml) into a searchable index. For each entity,
stores: ID, preferred label, synonyms, definition, and tree numbers.

Usage:
    # With rapidfuzz (default, no setup needed):
    index = MeSHIndex(backend="rapidfuzz")
    index.build_from_xml("Data/MeSH/desc2026.xml")
    results = index.search("famotidine", top_k=10)

    # With Elasticsearch (needs ES running on localhost:9200):
    index = MeSHIndex(backend="elasticsearch")
    index.build_from_xml("Data/MeSH/desc2026.xml")
    results = index.search("famotidine", top_k=10)
"""

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

# ── Optional dependency imports ────────────────────────────────────────────

try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    from difflib import SequenceMatcher
    HAS_RAPIDFUZZ = False

try:
    from elasticsearch import Elasticsearch, helpers
    HAS_ELASTICSEARCH = True
except ImportError:
    HAS_ELASTICSEARCH = False


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
    score: float              # similarity score (0-100 for rapidfuzz, BM25 score for ES)
    matched_synonym: str      # which label/synonym was matched


# ── Main index class ───────────────────────────────────────────────────────

class MeSHIndex:
    """
    Search index over MeSH entities.

    Supports two backends:
      - "rapidfuzz": In-memory fuzzy matching (token_sort_ratio). Default.
      - "elasticsearch": BM25 search via local Elasticsearch instance.

    Parameters
    ----------
    backend : str
        Which search backend to use: "rapidfuzz" or "elasticsearch".
    case_sensitive : bool
        Whether search should be case-sensitive (rapidfuzz only). Default False.
    es_url : str
        Elasticsearch URL. Default "http://localhost:9200".
    es_index_name : str
        Name of the Elasticsearch index. Default "mesh_entities".
    """

    def __init__(
        self,
        backend: str = "rapidfuzz",
        case_sensitive: bool = False,
        es_url: str = "http://localhost:9200",
        es_index_name: str = "mesh_entities",
    ):
        # Validate backend choice
        if backend not in ("rapidfuzz", "elasticsearch"):
            raise ValueError(f"Unknown backend '{backend}'. Use 'rapidfuzz' or 'elasticsearch'.")

        if backend == "elasticsearch" and not HAS_ELASTICSEARCH:
            raise ImportError(
                "elasticsearch package not installed. "
                "Install with: pip install elasticsearch"
            )

        if backend == "rapidfuzz" and not HAS_RAPIDFUZZ:
            print("Warning: rapidfuzz not installed, using difflib fallback (slower). "
                  "Install with: pip install rapidfuzz")

        self.backend = backend
        self.case_sensitive = case_sensitive

        # mesh_id -> MeSHEntity (used by both backends for lookup)
        self.entities: dict[str, MeSHEntity] = {}

        # ── rapidfuzz-specific ──
        # Flat list of (normalized_label, mesh_id) for in-memory search
        self._label_index: list[tuple[str, str]] = []

        # ── elasticsearch-specific ──
        self._es_url = es_url
        self._es_index_name = es_index_name
        self._es_client: Elasticsearch | None = None

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
        enrich_wikidata: bool = False,
        enrich_dbpedia: bool = False,
        enrich_umls: str | None = None,
        enrich_mondo: bool = False,
        enrich_mrdef: str | None = None,
    ):
        """
        Parse MeSH XML file(s) and build the search index.

        Parameters
        ----------
        descriptor_path : str or None
            Path to desc2026.xml (main MeSH descriptors).
        supplementary_path : str or None
            Path to supp2026.xml (supplementary concepts, mostly chemicals).
        enrich_wikidata : bool
            If True, fetch additional synonyms from Wikidata before building
            the search index. Cached after first run (~1-2 min).
        enrich_dbpedia : bool
            If True, fetch additional labels and Wikipedia redirects from
            DBpedia. Cached after first run (~1-2 min).
        enrich_umls : str or None
            Path to MRCONSO.RRF file. If provided, enriches entities with
            synonyms from all UMLS vocabularies. Cached after first run.
        enrich_mondo : bool
            If True, fetch additional disease synonyms from the MONDO
            ontology. Particularly useful for improving Disease linking.
            Cached after first download (~90MB).
        enrich_mrdef : str or None
            Path to MRDEF.RRF file. If provided, enriches entities with
            definitions from UMLS (NCI, MSH, CSP, etc.). Especially useful
            for supplementary concepts which have no MeSH scope note.
            Requires cui_to_mesh_cache.json in the same directory.
        """
        if descriptor_path:
            self._parse_descriptors(descriptor_path)
        if supplementary_path:
            self._parse_supplementary(supplementary_path)

        # Optionally enrich with additional synonym sources
        if enrich_umls:
            self._enrich_from_umls(enrich_umls)
        if enrich_wikidata:
            self._enrich_from_wikidata()
        if enrich_dbpedia:
            self._enrich_from_dbpedia()
        if enrich_mondo:
            self._enrich_from_mondo()

        # Enrich definitions from MRDEF (after synonyms, before index build)
        if enrich_mrdef:
            self._enrich_definitions_from_mrdef(enrich_mrdef)

        # Build the appropriate search index based on backend
        if self.backend == "elasticsearch":
            self._build_es_index()
        else:
            self._build_label_index()

        print(f"MeSH index ready: {self.size} entities ({self.backend})")

    def _enrich_from_wikidata(self):
        """
        Add extra synonyms from Wikidata to existing MeSH entities.
        Only adds synonyms for entities already in our index.
        """
        from wikidata_enricher import WikidataEnricher

        enricher = WikidataEnricher()
        wikidata_synonyms = enricher.fetch_mesh_synonyms()

        added_count = 0
        matched_entities = 0

        for mesh_id, extra_syns in wikidata_synonyms.items():
            if mesh_id in self.entities:
                entity = self.entities[mesh_id]
                # Only add synonyms we don't already have (case-insensitive check)
                existing_lower = {s.lower() for s in entity.synonyms}
                new_syns = [s for s in extra_syns if s.lower() not in existing_lower]
                if new_syns:
                    entity.synonyms.extend(new_syns)
                    added_count += len(new_syns)
                    matched_entities += 1

        print(f"  + Wikidata: {added_count} synonyms ({matched_entities} entities)")

    def _enrich_from_dbpedia(self):
        """
        Add extra synonyms from DBpedia to existing MeSH entities.
        Includes Wikipedia redirect titles (informal/alternative names).
        """
        from dbpedia_enricher import DBpediaEnricher

        enricher = DBpediaEnricher()
        dbpedia_synonyms = enricher.fetch_mesh_synonyms()

        added_count = 0
        matched_entities = 0

        for mesh_id, extra_syns in dbpedia_synonyms.items():
            if mesh_id in self.entities:
                entity = self.entities[mesh_id]
                existing_lower = {s.lower() for s in entity.synonyms}
                new_syns = [s for s in extra_syns if s.lower() not in existing_lower]
                if new_syns:
                    entity.synonyms.extend(new_syns)
                    added_count += len(new_syns)
                    matched_entities += 1

        print(f"  + DBpedia: {added_count} synonyms ({matched_entities} entities)")

    def _enrich_from_umls(self, mrconso_path: str):
        """
        Add extra synonyms from UMLS to existing MeSH entities.
        Uses CUI mappings to find synonyms from DrugBank, SNOMED, RxNorm, etc.
        """
        from umls_enricher import UMLSEnricher

        enricher = UMLSEnricher(mrconso_path)
        umls_synonyms = enricher.get_mesh_synonyms()

        added_count = 0
        matched_entities = 0

        for mesh_id, extra_syns in umls_synonyms.items():
            if mesh_id in self.entities:
                entity = self.entities[mesh_id]
                # Only add synonyms we don't already have (case-insensitive check)
                existing_lower = {s.lower() for s in entity.synonyms}
                new_syns = [s for s in extra_syns if s.lower() not in existing_lower]
                if new_syns:
                    entity.synonyms.extend(new_syns)
                    added_count += len(new_syns)
                    matched_entities += 1

        print(f"  + UMLS: {added_count} synonyms ({matched_entities} entities)")

    def _enrich_from_mondo(self):
        """
        Add extra disease synonyms from the MONDO ontology to existing MeSH entities.
        MONDO unifies OMIM, Orphanet, Disease Ontology, and others — providing
        extensive synonym coverage especially for diseases.
        """
        from mondo_enricher import MONDOEnricher

        enricher = MONDOEnricher()
        mondo_synonyms = enricher.fetch_mesh_synonyms()

        added_count = 0
        matched_entities = 0

        for mesh_id, extra_syns in mondo_synonyms.items():
            if mesh_id in self.entities:
                entity = self.entities[mesh_id]
                existing_lower = {s.lower() for s in entity.synonyms}
                new_syns = [s for s in extra_syns if s.lower() not in existing_lower]
                if new_syns:
                    entity.synonyms.extend(new_syns)
                    added_count += len(new_syns)
                    matched_entities += 1

        print(f"  + MONDO: {added_count} synonyms ({matched_entities} entities)")

    def _enrich_definitions_from_mrdef(self, mrdef_path: str):
        """
        Add definitions from UMLS MRDEF.RRF to entities that lack one.

        MRDEF contains definitions from 16 sources (NCI, MSH, CSP, GO, HPO, etc.).
        This is especially valuable for supplementary concepts (C-prefix MeSH IDs)
        which have NO scope note in the MeSH XML — 324k entities with zero definition.

        For entities that already have a definition, MRDEF is skipped (MeSH scope
        notes are the primary source). For entities without a definition, the best
        available MRDEF definition is used, with source priority:
        MSH > NCI > CSP > PDQ > MEDLINEPLUS > others.

        Parameters
        ----------
        mrdef_path : str
            Path to MRDEF.RRF file (e.g., "Data/UMLS/MRDEF.RRF").
        """
        import json
        from pathlib import Path

        mrdef_dir = Path(mrdef_path).parent
        cui_cache_path = mrdef_dir / "cui_to_mesh_cache.json"

        if not cui_cache_path.exists():
            print(f"  ! MRDEF skipped: {cui_cache_path} not found")
            return

        # Load CUI → MeSH mapping
        with open(cui_cache_path) as f:
            cui_to_mesh_raw = json.load(f)

        # Build MeSH → CUI(s) reverse mapping
        mesh_to_cuis: dict[str, list[str]] = {}
        for cui, mesh_val in cui_to_mesh_raw.items():
            mesh_ids = mesh_val if isinstance(mesh_val, list) else [mesh_val]
            for mid in mesh_ids:
                if mid not in mesh_to_cuis:
                    mesh_to_cuis[mid] = []
                mesh_to_cuis[mid].append(cui)

        # Load MRDEF definitions indexed by CUI
        cui_defs: dict[str, dict[str, str]] = {}
        with open(mrdef_path, "r") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) >= 6:
                    cui, source, defn = parts[0], parts[4], parts[5]
                    if cui not in cui_defs:
                        cui_defs[cui] = {}
                    # Keep the longer definition per source
                    if source not in cui_defs[cui] or len(defn) > len(cui_defs[cui][source]):
                        cui_defs[cui][source] = defn

        # Source priority: prefer biomedical sources with clear definitions
        source_priority = ["MSH", "NCI", "CSP", "PDQ", "MEDLINEPLUS", "CHV",
                           "HPO", "GO", "SPN", "FMA", "AIR", "MCM"]

        enriched_count = 0
        supplemented_count = 0

        for mesh_id, entity in self.entities.items():
            # Only enrich entities that have NO definition
            if entity.definition:
                continue

            # Find CUIs for this MeSH entity
            cuis = mesh_to_cuis.get(mesh_id, [])
            if not cuis:
                continue

            # Find the best definition across all CUIs and sources
            best_def = ""
            best_source = ""
            best_priority = len(source_priority) + 1

            for cui in cuis:
                defs = cui_defs.get(cui, {})
                for source, defn in defs.items():
                    try:
                        priority = source_priority.index(source)
                    except ValueError:
                        priority = len(source_priority)  # unknown source = lowest

                    # Prefer higher priority source, or longer definition at same priority
                    if priority < best_priority or (priority == best_priority and len(defn) > len(best_def)):
                        best_def = defn
                        best_source = source
                        best_priority = priority

            if best_def:
                entity.definition = best_def
                enriched_count += 1

        print(f"  + MRDEF definitions: {enriched_count} entities enriched "
              f"(from {len(cui_defs)} CUIs)")

    def _parse_descriptors(self, path: str):
        """Parse MeSH descriptor XML (desc2026.xml)."""
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

            count += 1

        print(f"  Descriptors: {count}", end="", flush=True)

    def _parse_supplementary(self, path: str):
        """Parse MeSH supplementary concepts XML (supp2026.xml)."""
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

            count += 1

        print(f" + Supplementary: {count}")

    # ── rapidfuzz index ────────────────────────────────────────────────

    def _build_label_index(self):
        """
        Build the flat label index for rapidfuzz search.
        Maps every label/synonym to its mesh_id for fast lookup.
        """
        self._label_index = []
        for mesh_id, entity in self.entities.items():
            for synonym in entity.synonyms:
                normalized = synonym if self.case_sensitive else synonym.lower()
                self._label_index.append((normalized, mesh_id))

        print(f" | {len(self._label_index)} labels indexed")

    # ── Elasticsearch index ────────────────────────────────────────────

    def _build_es_index(self):
        """
        Index all MeSH entities into Elasticsearch.
        Creates one document per entity with all synonyms as a searchable field.
        Elasticsearch uses BM25 scoring by default.
        """
        # ES 8.x Python client: pass URL as string, not list.
        # Also need to explicitly set _meta header to avoid warnings.
        # Connect to Elasticsearch
        self._es_client = Elasticsearch(self._es_url)

        # Check connection — catch the actual error for debugging
        try:
            info = self._es_client.info()
            pass  # connected OK
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to Elasticsearch at {self._es_url}: {e}\n"
                "Make sure it's running (e.g., docker start elasticsearch)."
            )

        # Delete old index if it exists and recreate
        if self._es_client.indices.exists(index=self._es_index_name):
            self._es_client.indices.delete(index=self._es_index_name)

        # Create index with custom mapping optimized for entity search
        # We use a "text" field for BM25 search over all labels/synonyms
        mapping = {
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                # Custom analyzer: lowercase + standard tokenizer
                "analysis": {
                    "analyzer": {
                        "entity_analyzer": {
                            "type": "custom",
                            "tokenizer": "standard",
                            "filter": ["lowercase"],
                        }
                    }
                }
            },
            "mappings": {
                "properties": {
                    "mesh_id": {"type": "keyword"},
                    "preferred_label": {"type": "text", "analyzer": "entity_analyzer"},
                    # All synonyms joined into one text field for BM25 search
                    "all_labels": {"type": "text", "analyzer": "entity_analyzer"},
                    # Also store each synonym as a keyword for exact matching
                    "synonyms_exact": {"type": "keyword", "normalizer": "lowercase"},
                    "definition": {"type": "text", "analyzer": "entity_analyzer"},
                }
            }
        }

        # Add lowercase normalizer for keyword fields
        mapping["settings"]["analysis"]["normalizer"] = {
            "lowercase": {
                "type": "custom",
                "filter": ["lowercase"],
            }
        }

        self._es_client.indices.create(index=self._es_index_name, body=mapping)

        # Bulk-index all entities
        def _generate_actions():
            for mesh_id, entity in self.entities.items():
                yield {
                    "_index": self._es_index_name,
                    "_id": mesh_id,
                    "_source": {
                        "mesh_id": mesh_id,
                        "preferred_label": entity.preferred_label,
                        # Join all synonyms into a single text for BM25
                        "all_labels": " | ".join(entity.synonyms),
                        # Also index each synonym individually for exact match boosting
                        "synonyms_exact": [s.lower() for s in entity.synonyms],
                        "definition": entity.definition,
                    }
                }

        success, errors = helpers.bulk(
            self._es_client,
            _generate_actions(),
            chunk_size=5000,
            raise_on_error=False,
        )
        err_count = len(errors) if isinstance(errors, list) else errors
        if err_count:
            print(f"  ES index: {success} indexed, {err_count} errors")
        else:
            print(f"  ES index: {success} documents")

        # Refresh so documents are immediately searchable
        self._es_client.indices.refresh(index=self._es_index_name)

    # ── Searching ──────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 10) -> list[CandidateEntity]:
        """
        Search for candidate entities matching a mention.

        Uses the configured backend:
          - rapidfuzz: fuzzy string matching (token_sort_ratio)
          - elasticsearch: BM25 with exact-match boosting

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
        if self.backend == "elasticsearch":
            return self._search_elasticsearch(query, top_k)
        elif HAS_RAPIDFUZZ:
            normalized = query if self.case_sensitive else query.lower()
            return self._search_rapidfuzz(normalized, top_k)
        else:
            normalized = query if self.case_sensitive else query.lower()
            return self._search_difflib(normalized, top_k)

    def _search_elasticsearch(self, query: str, top_k: int) -> list[CandidateEntity]:
        """
        Search using Elasticsearch BM25 scoring.

        Uses a combination of:
          1. Exact match on synonyms (highest boost) — catches perfect matches
          2. BM25 on all_labels field — good for multi-word queries
          3. BM25 on preferred_label — slight boost for preferred name matches
        """
        if self._es_client is None:
            raise RuntimeError("Elasticsearch not initialized. Call build_from_xml() first.")

        # Multi-field query with boosting:
        # - Exact synonym match gets highest score (catches "famotidine" -> "Famotidine")
        # - BM25 on all_labels catches partial/fuzzy matches
        # - BM25 on preferred_label gives slight preference to preferred names
        es_query = {
            "bool": {
                "should": [
                    # Exact match on any synonym (strongest signal)
                    {
                        "term": {
                            "synonyms_exact": {
                                "value": query.lower(),
                                "boost": 10.0,
                            }
                        }
                    },
                    # BM25 on all labels/synonyms
                    {
                        "match": {
                            "all_labels": {
                                "query": query,
                                "boost": 2.0,
                                "fuzziness": "AUTO",  # allows 1-2 char edits (typos, plurals)
                            }
                        }
                    },
                    # BM25 on preferred label
                    {
                        "match": {
                            "preferred_label": {
                                "query": query,
                                "boost": 1.5,
                            }
                        }
                    },
                ],
                "minimum_should_match": 1,
            }
        }

        response = self._es_client.search(
            index=self._es_index_name,
            query=es_query,
            size=top_k,
        )

        # Convert ES hits to CandidateEntity objects
        candidates = []
        for hit in response["hits"]["hits"]:
            mesh_id = hit["_source"]["mesh_id"]
            entity = self.entities[mesh_id]
            score = hit["_score"]

            # Find which synonym was the best match
            # (ES doesn't tell us directly, so we pick the closest one)
            matched = entity.preferred_label
            query_lower = query.lower()
            for syn in entity.synonyms:
                if syn.lower() == query_lower:
                    matched = syn
                    break

            candidates.append(CandidateEntity(
                mesh_id=mesh_id,
                preferred_label=entity.preferred_label,
                synonyms=entity.synonyms,
                definition=entity.definition,
                tree_numbers=entity.tree_numbers,
                score=score,
                matched_synonym=matched,
            ))

        return candidates

    def _search_rapidfuzz(self, query: str, top_k: int) -> list[CandidateEntity]:
        """Search using rapidfuzz (fast fuzzy matching)."""
        if not self._label_index:
            raise RuntimeError("Index is empty. Call build_from_xml() first.")

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
        if not self._label_index:
            raise RuntimeError("Index is empty. Call build_from_xml() first.")

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
    import argparse
    import time

    parser = argparse.ArgumentParser(description="MeSH Index Demo")
    parser.add_argument(
        "--backend", choices=["rapidfuzz", "elasticsearch"], default="rapidfuzz",
        help="Search backend to use (default: rapidfuzz)",
    )
    parser.add_argument(
        "--es-url", default="http://localhost:9200",
        help="Elasticsearch URL (default: http://localhost:9200)",
    )
    parser.add_argument(
        "--wikidata", action="store_true",
        help="Enrich MeSH entities with Wikidata synonyms",
    )
    parser.add_argument(
        "--dbpedia", action="store_true",
        help="Enrich MeSH entities with DBpedia labels and Wikipedia redirects",
    )
    parser.add_argument(
        "--umls", type=str, default=None,
        help="Path to MRCONSO.RRF for UMLS synonym enrichment",
    )
    parser.add_argument(
        "--mondo", action="store_true",
        help="Enrich MeSH entities with MONDO disease ontology synonyms",
    )
    args = parser.parse_args()

    print(f"Using backend: {args.backend}")
    index = MeSHIndex(backend=args.backend, es_url=args.es_url)

    # Build from MeSH XML — adjust paths as needed
    t0 = time.time()
    index.build_from_xml(
        descriptor_path="../../Data/MeSH/desc2026.xml",
        supplementary_path="../../Data/MeSH/supp2026.xml",
        enrich_wikidata=args.wikidata,
        enrich_dbpedia=args.dbpedia,
        enrich_umls=args.umls,
        enrich_mondo=args.mondo,
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
