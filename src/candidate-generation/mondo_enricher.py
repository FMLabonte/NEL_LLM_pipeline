"""
MONDO Disease Ontology Enricher
================================
Fetches additional disease synonyms from the MONDO ontology for MeSH entities.

MONDO (Monarch Disease Ontology) unifies disease definitions from OMIM, Orphanet,
Disease Ontology, NCIT, and others into a single coherent ontology. It provides
extensive synonym coverage — especially for diseases — including:
  - Exact synonyms (clinically equivalent names)
  - Related synonyms (informal/colloquial names)
  - Narrow synonyms (more specific subtypes)
  - Cross-references to MeSH, OMIM, DOID, NCIT, ICD-10, etc.

This is particularly useful because our pipeline's Disease accuracy (74.1%)
significantly trails Chemical accuracy (83.3%). MONDO's disease-focused synonyms
can help close this gap by providing names that MeSH, UMLS, Wikidata, and
DBpedia may miss.

Example: MeSH "Diabetes Mellitus, Type 2" (D003924) might gain synonyms like
"type 2 diabetes", "T2DM", "adult-onset diabetes", "NIDDM",
"non-insulin-dependent diabetes" from MONDO.

No API key needed — downloads the public MONDO JSON-LD release from GitHub.

Usage:
    from mondo_enricher import MONDOEnricher

    enricher = MONDOEnricher()
    extra_synonyms = enricher.fetch_mesh_synonyms()
    # Returns: {"D003924": ["type 2 diabetes", "T2DM", "adult-onset diabetes", ...], ...}
"""

import json
import gzip
import urllib.request
from pathlib import Path


# MONDO releases are on GitHub — the JSON file contains the full ontology
# with all synonyms, cross-references, and definitions.
MONDO_JSON_URL = "https://github.com/monarch-initiative/mondo/releases/latest/download/mondo.json"

# Cache paths
DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "mondo_cache.json"
DEFAULT_RAW_PATH = Path(__file__).parent / "cache" / "mondo_raw.json"


class MONDOEnricher:
    """
    Fetches extra disease synonyms for MeSH entities from the MONDO ontology.

    Downloads the MONDO ontology (JSON format), extracts all disease classes
    that have MeSH cross-references, and collects their synonyms.

    Parameters
    ----------
    cache_path : str or Path or None
        Path to processed cache file (MeSH ID → synonyms mapping).
        Set to None to disable caching.
    raw_path : str or Path or None
        Path to store the downloaded MONDO JSON file.
        Avoids re-downloading the ~90MB file on subsequent runs.
    """

    def __init__(
        self,
        cache_path: str | Path | None = DEFAULT_CACHE_PATH,
        raw_path: str | Path | None = DEFAULT_RAW_PATH,
    ):
        self.cache_path = Path(cache_path) if cache_path else None
        self.raw_path = Path(raw_path) if raw_path else None

    def fetch_mesh_synonyms(self, force_refresh: bool = False) -> dict[str, list[str]]:
        """
        Get additional disease synonyms for MeSH entities from MONDO.

        Returns a dict mapping MeSH IDs to lists of extra synonyms.
        Uses a local cache to avoid re-processing the ontology every time.

        Parameters
        ----------
        force_refresh : bool
            If True, ignore cache and re-download/re-process MONDO.

        Returns
        -------
        dict[str, list[str]]
            Mapping from MeSH ID (e.g., "D003924") to list of extra synonyms.
        """
        # Try loading from processed cache first
        if not force_refresh and self.cache_path and self.cache_path.exists():
            print(f"Loading MONDO synonyms from cache: {self.cache_path}")
            return self._load_cache()

        # Download or load raw MONDO JSON
        mondo_data = self._get_mondo_json(force_refresh)

        # Extract MeSH → synonyms mapping
        print("Processing MONDO ontology for MeSH cross-references...")
        synonyms = self._extract_mesh_synonyms(mondo_data)

        total_syns = sum(len(s) for s in synonyms.values())
        print(f"  Total: {total_syns} synonyms for {len(synonyms)} MeSH entities from MONDO")

        # Save to cache
        if self.cache_path:
            self._save_cache(synonyms)
            print(f"  Cached to {self.cache_path}")

        return synonyms

    def _get_mondo_json(self, force_refresh: bool = False) -> dict:
        """
        Download or load the MONDO ontology JSON file.

        The file is ~90MB and contains the full ontology graph.
        Cached locally after first download.
        """
        # Try loading cached raw file
        if not force_refresh and self.raw_path and self.raw_path.exists():
            print(f"Loading MONDO ontology from local file: {self.raw_path}")
            with open(self.raw_path, "r", encoding="utf-8") as f:
                return json.load(f)

        # Download from GitHub
        print(f"Downloading MONDO ontology from GitHub...")
        print(f"  URL: {MONDO_JSON_URL}")
        print(f"  (This is ~90MB and may take a few minutes on first run)")

        req = urllib.request.Request(MONDO_JSON_URL, headers={
            "User-Agent": "NEL_LLM_pipeline/1.0 (University of Bonn NLP Lab; "
                          "biomedical entity linking research)",
            "Accept": "application/json",
        })

        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                data = json.loads(response.read().decode("utf-8"))
                print(f"  Downloaded successfully")

                # Save raw file for future use
                if self.raw_path:
                    self.raw_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(self.raw_path, "w", encoding="utf-8") as f:
                        json.dump(data, f)
                    print(f"  Saved raw MONDO file to {self.raw_path}")

                return data

        except Exception as e:
            print(f"  Warning: MONDO download failed: {e}")
            print("  Continuing without MONDO enrichment.")
            return {}

    def _extract_mesh_synonyms(self, mondo_data: dict) -> dict[str, list[str]]:
        """
        Extract MeSH ID → synonym mappings from the MONDO ontology graph.

        MONDO stores diseases as graph nodes with:
          - "lbl": preferred label
          - "meta.synonyms": list of synonym objects with "val" and "pred"
          - "meta.xrefs": cross-references including MeSH IDs (format "MESH:D003924")

        We find all nodes with a MeSH xref and collect their synonyms.
        """
        synonyms: dict[str, set[str]] = {}

        # MONDO JSON structure: {"graphs": [{"nodes": [...], ...}]}
        graphs = mondo_data.get("graphs", [])
        if not graphs:
            print("  Warning: No graphs found in MONDO JSON")
            return {}

        nodes = graphs[0].get("nodes", [])
        print(f"  Processing {len(nodes)} MONDO nodes...")

        mondo_classes = 0
        mesh_linked = 0

        for node in nodes:
            # Skip non-class nodes (properties, individuals, etc.)
            node_type = node.get("type", "")
            if node_type != "CLASS":
                continue

            mondo_classes += 1
            meta = node.get("meta", {})

            # Find MeSH cross-references
            mesh_ids = self._extract_mesh_xrefs(meta)
            if not mesh_ids:
                continue

            mesh_linked += 1

            # Collect all names for this concept
            names = set()

            # 1. Preferred label
            label = node.get("lbl", "")
            if label:
                names.add(label)

            # 2. Synonyms (exact, related, narrow, broad)
            for syn_obj in meta.get("synonyms", []):
                syn_val = syn_obj.get("val", "").strip()
                if syn_val:
                    names.add(syn_val)

            # 3. Definition text is NOT added as synonym (too long / noisy)

            # Map to all linked MeSH IDs
            for mesh_id in mesh_ids:
                if mesh_id not in synonyms:
                    synonyms[mesh_id] = set()
                synonyms[mesh_id].update(names)

        print(f"  MONDO classes: {mondo_classes}")
        print(f"  Classes with MeSH xref: {mesh_linked}")
        print(f"  Unique MeSH IDs found: {len(synonyms)}")

        # Convert sets to sorted lists
        return {mid: sorted(syns) for mid, syns in synonyms.items() if syns}

    def _extract_mesh_xrefs(self, meta: dict) -> list[str]:
        """
        Extract MeSH IDs from a node's cross-references.

        MONDO stores xrefs as: {"val": "MESH:D003924"} or
        in the xrefs list or in basicPropertyValues.

        Returns list of MeSH IDs (without the "MESH:" prefix).
        """
        mesh_ids = []

        # Check xrefs list
        for xref in meta.get("xrefs", []):
            val = xref.get("val", "")
            if val.startswith("MESH:"):
                mesh_id = val[5:]  # Remove "MESH:" prefix
                if mesh_id:
                    mesh_ids.append(mesh_id)

        # Also check definition xrefs (some MeSH refs are here)
        definition = meta.get("definition", {})
        for xref in definition.get("xrefs", []):
            if isinstance(xref, str) and xref.startswith("MESH:"):
                mesh_id = xref[5:]
                if mesh_id and mesh_id not in mesh_ids:
                    mesh_ids.append(mesh_id)

        return mesh_ids

    def _load_cache(self) -> dict[str, list[str]]:
        """Load cached synonyms from JSON file."""
        with open(self.cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} MeSH mappings from cache")
        return data

    def _save_cache(self, synonyms: dict[str, list[str]]):
        """Save synonyms to JSON cache file."""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(synonyms, f, ensure_ascii=False, indent=2)


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enricher = MONDOEnricher()
    synonyms = enricher.fetch_mesh_synonyms()

    print(f"\nTotal MeSH entities with MONDO synonyms: {len(synonyms)}")
    total_syns = sum(len(s) for s in synonyms.values())
    print(f"Total synonyms: {total_syns}")
    print(f"Avg synonyms per entity: {total_syns / len(synonyms):.1f}")

    # Show some disease examples
    examples = [
        ("D003920", "Diabetes Mellitus"),
        ("D003924", "Diabetes Mellitus, Type 2"),
        ("D006973", "Hypertension"),
        ("D009203", "Myocardial Infarction"),
        ("D001249", "Asthma"),
        ("D006937", "Hypercholesterolemia"),
        ("D000544", "Alzheimer Disease"),
    ]
    for mesh_id, name in examples:
        if mesh_id in synonyms:
            syns = synonyms[mesh_id]
            print(f"\n  {mesh_id} ({name}):")
            print(f"    {len(syns)} synonyms: {syns[:8]}")
            if len(syns) > 8:
                print(f"    ... and {len(syns) - 8} more")
        else:
            print(f"\n  {mesh_id} ({name}): (not in MONDO)")
