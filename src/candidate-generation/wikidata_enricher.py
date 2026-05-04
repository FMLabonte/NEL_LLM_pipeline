"""
Wikidata Synonym Enricher
==========================
Fetches additional labels and aliases from Wikidata for MeSH entities.

Wikidata contains many biomedical entities linked to MeSH IDs (property P486).
By querying these, we can add extra synonyms (e.g., common names, trade names)
that are not in the MeSH XML files. This improves candidate recall because
mentions in text often use informal or alternative names.

Example: MeSH "Famotidine" (D015738) might get additional aliases like
"Pepcid AC", "Fluxid", etc. from Wikidata.

No API key or registration needed — uses the public Wikidata SPARQL endpoint.

Usage:
    from wikidata_enricher import WikidataEnricher

    enricher = WikidataEnricher()
    extra_synonyms = enricher.fetch_mesh_synonyms()
    # Returns: {"D015738": ["Pepcid AC", "Fluxid", ...], ...}
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path


# Wikidata public SPARQL endpoint
WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"

# Cache file to avoid re-querying Wikidata every time
DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "wikidata_cache.json"


class WikidataEnricher:
    """
    Fetches extra synonyms for MeSH entities from Wikidata.

    Queries Wikidata for all entities that have a MeSH ID (property P486),
    and collects their English labels and aliases. These can then be added
    to the MeSH index as additional searchable synonyms.

    Parameters
    ----------
    cache_path : str or Path or None
        Path to cache file. If the cache exists, data is loaded from there
        instead of querying Wikidata. Set to None to disable caching.
    """

    def __init__(self, cache_path: str | Path | None = DEFAULT_CACHE_PATH):
        self.cache_path = Path(cache_path) if cache_path else None

    def fetch_mesh_synonyms(self, force_refresh: bool = False) -> dict[str, list[str]]:
        """
        Get additional synonyms for MeSH entities from Wikidata.

        Returns a dict mapping MeSH IDs to lists of extra synonyms.
        Uses a local cache to avoid hitting the SPARQL endpoint every time.

        Parameters
        ----------
        force_refresh : bool
            If True, ignore cache and re-query Wikidata.

        Returns
        -------
        dict[str, list[str]]
            Mapping from MeSH ID (e.g., "D015738") to list of extra synonyms.
        """
        # Try loading from cache first
        if not force_refresh and self.cache_path and self.cache_path.exists():
            print(f"Loading Wikidata synonyms from cache: {self.cache_path}")
            return self._load_cache()

        # Query Wikidata SPARQL endpoint
        print("Fetching synonyms from Wikidata SPARQL endpoint...")
        print("  (This may take 1-2 minutes on first run, then cached)")
        synonyms = self._query_wikidata()

        # Save to cache
        if self.cache_path:
            self._save_cache(synonyms)
            print(f"  Cached {len(synonyms)} MeSH mappings to {self.cache_path}")

        return synonyms

    def _query_wikidata(self) -> dict[str, list[str]]:
        """
        Query Wikidata SPARQL for all entities with MeSH IDs.

        Gets both the main English label and all English aliases (altLabel).
        Splits into multiple queries to avoid timeout on the public endpoint.
        """
        # SPARQL query: find all items with a MeSH ID, get their English
        # labels and aliases. We use SERVICE wikibase:label for labels
        # and skos:altLabel for aliases.
        query = """
        SELECT ?meshId ?label ?alias WHERE {
          ?item wdt:P486 ?meshId .
          ?item rdfs:label ?label .
          FILTER(LANG(?label) = "en")
          OPTIONAL {
            ?item skos:altLabel ?alias .
            FILTER(LANG(?alias) = "en")
          }
        }
        """

        results = self._execute_sparql(query)

        # Group by MeSH ID
        synonyms: dict[str, set[str]] = {}
        for row in results:
            mesh_id = row["meshId"]["value"]
            label = row["label"]["value"]

            if mesh_id not in synonyms:
                synonyms[mesh_id] = set()

            synonyms[mesh_id].add(label)

            # Alias is optional (from OPTIONAL clause)
            if "alias" in row:
                synonyms[mesh_id].add(row["alias"]["value"])

        # Convert sets to sorted lists
        return {mid: sorted(syns) for mid, syns in synonyms.items()}

    def _execute_sparql(self, query: str) -> list[dict]:
        """
        Execute a SPARQL query against the Wikidata endpoint.

        Uses urllib (no extra dependencies needed). Handles the JSON response
        and returns the list of result bindings.
        """
        # URL-encode the query
        params = urllib.parse.urlencode({
            "query": query,
            "format": "json",
        })

        url = f"{WIKIDATA_SPARQL_URL}?{params}"

        # Wikidata requires a User-Agent header
        req = urllib.request.Request(url, headers={
            "User-Agent": "NEL_LLM_pipeline/1.0 (University of Bonn NLP Lab; biomedical entity linking research)",
            "Accept": "application/json",
        })

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
                results = data["results"]["bindings"]
                print(f"  Received {len(results)} rows from Wikidata")
                return results
        except Exception as e:
            print(f"  Warning: Wikidata query failed: {e}")
            print("  Continuing without Wikidata enrichment.")
            return []

    def _load_cache(self) -> dict[str, list[str]]:
        """Load cached synonyms from JSON file."""
        with open(self.cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} MeSH mappings from cache")
        return data

    def _save_cache(self, synonyms: dict[str, list[str]]):
        """Save synonyms to JSON cache file."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(synonyms, f, ensure_ascii=False, indent=2)


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    enricher = WikidataEnricher()
    synonyms = enricher.fetch_mesh_synonyms()

    print(f"\nTotal MeSH entities with Wikidata synonyms: {len(synonyms)}")

    # Show some examples
    examples = ["D015738", "D004317", "D006973", "D003693", "D001241"]
    for mesh_id in examples:
        if mesh_id in synonyms:
            print(f"\n  {mesh_id}: {synonyms[mesh_id]}")
        else:
            print(f"\n  {mesh_id}: (not in Wikidata)")
