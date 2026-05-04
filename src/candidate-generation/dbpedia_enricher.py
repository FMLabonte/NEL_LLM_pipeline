"""
DBpedia Synonym Enricher
=========================
Fetches additional labels and redirects from DBpedia for MeSH entities.

DBpedia is a structured extraction of Wikipedia. It contains many biomedical
entities linked to MeSH IDs (via the dbo:meshId property). The key advantage
over Wikidata is that DBpedia also includes Wikipedia redirect titles — these
are often informal or alternative names used in practice.

Example: "Heart attack" is a Wikipedia redirect to "Myocardial infarction",
so DBpedia knows both names. This helps when a mention uses the colloquial
name instead of the official MeSH term.

No API key or registration needed — uses the public DBpedia SPARQL endpoint.

Usage:
    from dbpedia_enricher import DBpediaEnricher

    enricher = DBpediaEnricher()
    extra_synonyms = enricher.fetch_mesh_synonyms()
    # Returns: {"D009203": ["Heart attack", "Myocardial infarction", ...], ...}
"""

import json
import time
import urllib.request
import urllib.parse
from pathlib import Path


# DBpedia public SPARQL endpoint
DBPEDIA_SPARQL_URL = "https://dbpedia.org/sparql"

# Cache file to avoid re-querying DBpedia every time
DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "dbpedia_cache.json"


class DBpediaEnricher:
    """
    Fetches extra synonyms for MeSH entities from DBpedia.

    Queries DBpedia for all entities that have a MeSH ID, and collects
    their English labels and Wikipedia redirect names. These can then
    be added to the MeSH index as additional searchable synonyms.

    Parameters
    ----------
    cache_path : str or Path or None
        Path to cache file. If the cache exists, data is loaded from there
        instead of querying DBpedia. Set to None to disable caching.
    """

    def __init__(self, cache_path: str | Path | None = DEFAULT_CACHE_PATH):
        self.cache_path = Path(cache_path) if cache_path else None

    def fetch_mesh_synonyms(self, force_refresh: bool = False) -> dict[str, list[str]]:
        """
        Get additional synonyms for MeSH entities from DBpedia.

        Returns a dict mapping MeSH IDs to lists of extra synonyms.
        Uses a local cache to avoid hitting the SPARQL endpoint every time.

        Parameters
        ----------
        force_refresh : bool
            If True, ignore cache and re-query DBpedia.

        Returns
        -------
        dict[str, list[str]]
            Mapping from MeSH ID (e.g., "D009203") to list of extra synonyms.
        """
        # Try loading from cache first
        if not force_refresh and self.cache_path and self.cache_path.exists():
            print(f"Loading DBpedia synonyms from cache: {self.cache_path}")
            return self._load_cache()

        # Query DBpedia SPARQL endpoint
        print("Fetching synonyms from DBpedia SPARQL endpoint...")
        print("  (This may take 1-2 minutes on first run, then cached)")

        synonyms: dict[str, set[str]] = {}

        # Query 1: Get labels for entities with MeSH IDs
        labels = self._fetch_labels()
        for mesh_id, names in labels.items():
            if mesh_id not in synonyms:
                synonyms[mesh_id] = set()
            synonyms[mesh_id].update(names)

        # Query 2: Get Wikipedia redirect names (alternative titles)
        redirects = self._fetch_redirects()
        for mesh_id, names in redirects.items():
            if mesh_id not in synonyms:
                synonyms[mesh_id] = set()
            synonyms[mesh_id].update(names)

        # Convert to sorted lists
        result = {mid: sorted(syns) for mid, syns in synonyms.items() if syns}

        total_syns = sum(len(s) for s in result.values())
        print(f"  Total: {total_syns} synonyms for {len(result)} MeSH entities from DBpedia")

        # Save to cache
        if self.cache_path:
            self._save_cache(result)
            print(f"  Cached to {self.cache_path}")

        return result

    def _fetch_labels(self) -> dict[str, set[str]]:
        """
        Fetch English labels for all DBpedia entities with a MeSH ID.
        Uses dbo:meshId to find entities, rdfs:label for their names.
        """
        query = """
        SELECT ?meshId ?label WHERE {
          ?entity dbo:meshId ?meshId .
          ?entity rdfs:label ?label .
          FILTER(LANG(?label) = "en")
        }
        """

        results = self._execute_sparql(query)
        synonyms: dict[str, set[str]] = {}

        for row in results:
            mesh_id = row["meshId"]["value"].strip()
            label = row["label"]["value"].strip()
            if mesh_id and label:
                if mesh_id not in synonyms:
                    synonyms[mesh_id] = set()
                synonyms[mesh_id].add(label)

        print(f"  Labels: {sum(len(s) for s in synonyms.values())} from {len(synonyms)} entities")
        return synonyms

    def _fetch_redirects(self) -> dict[str, set[str]]:
        """
        Fetch Wikipedia redirect names for DBpedia entities with MeSH IDs.
        Redirects are alternative page titles that point to the same article,
        often informal names (e.g., "Heart attack" → "Myocardial infarction").
        """
        query = """
        SELECT ?meshId ?redirectLabel WHERE {
          ?entity dbo:meshId ?meshId .
          ?redirect dbo:wikiPageRedirects ?entity .
          ?redirect rdfs:label ?redirectLabel .
          FILTER(LANG(?redirectLabel) = "en")
        }
        """

        results = self._execute_sparql(query)
        synonyms: dict[str, set[str]] = {}

        for row in results:
            mesh_id = row["meshId"]["value"].strip()
            label = row["redirectLabel"]["value"].strip()
            if mesh_id and label:
                if mesh_id not in synonyms:
                    synonyms[mesh_id] = set()
                synonyms[mesh_id].add(label)

        print(f"  Redirects: {sum(len(s) for s in synonyms.values())} from {len(synonyms)} entities")
        return synonyms

    def _execute_sparql(self, query: str) -> list[dict]:
        """
        Execute a SPARQL query against the DBpedia endpoint.
        """
        params = urllib.parse.urlencode({
            "query": query,
            "format": "application/sparql-results+json",
        })

        url = f"{DBPEDIA_SPARQL_URL}?{params}"

        req = urllib.request.Request(url, headers={
            "User-Agent": "NEL_LLM_pipeline/1.0 (University of Bonn NLP Lab; biomedical entity linking research)",
            "Accept": "application/sparql-results+json",
        })

        try:
            with urllib.request.urlopen(req, timeout=180) as response:
                data = json.loads(response.read().decode("utf-8"))
                results = data["results"]["bindings"]
                print(f"  Received {len(results)} rows from DBpedia")
                return results
        except Exception as e:
            print(f"  Warning: DBpedia query failed: {e}")
            print("  Continuing without DBpedia enrichment.")
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
    enricher = DBpediaEnricher()
    synonyms = enricher.fetch_mesh_synonyms()

    print(f"\nTotal MeSH entities with DBpedia synonyms: {len(synonyms)}")

    # Show some examples
    examples = ["D015738", "D004317", "D006973", "D003693", "D009203"]
    for mesh_id in examples:
        if mesh_id in synonyms:
            print(f"\n  {mesh_id}: {synonyms[mesh_id][:10]}")
        else:
            print(f"\n  {mesh_id}: (not in DBpedia)")
