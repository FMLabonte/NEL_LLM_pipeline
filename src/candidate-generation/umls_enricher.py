"""
UMLS Synonym Enricher
======================
Extracts additional synonyms for MeSH entities from the UMLS Metathesaurus.

UMLS connects many biomedical vocabularies (MeSH, SNOMED, DrugBank, RxNorm, etc.)
via shared Concept Unique Identifiers (CUIs). By looking up MeSH entities in UMLS,
we can find all the names/synonyms that other vocabularies have for the same concept.

Example: MeSH "Famotidine" (D015738) has CUI C0015620 in UMLS.
Via that CUI, we find synonyms from DrugBank ("Famotidinum", "Famotidina"),
SNOMED ("Famotidine-containing product"), RxNorm ("famotidine"), etc.

Requires: MRCONSO.RRF from the UMLS Metathesaurus download.
          Place at Data/UMLS/MRCONSO.RRF

Usage:
    from umls_enricher import UMLSEnricher

    enricher = UMLSEnricher("Data/UMLS/MRCONSO.RRF")
    extra_synonyms = enricher.get_mesh_synonyms()
    # Returns: {"D015738": ["Famotidina", "Famotidinum", ...], ...}
"""

import json
from pathlib import Path


# Cache file to avoid reparsing the 17M line file every time
DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "umls_cache.json"


# ── MRCONSO.RRF column indices (pipe-delimited) ──────────────────────────
# Full spec: https://www.ncbi.nlm.nih.gov/books/NBK9685/table/ch03.T.concept_names_and_sources_file_mr/
COL_CUI  = 0    # Concept Unique Identifier (e.g., "C0015620")
COL_LAT  = 1    # Language (e.g., "ENG")
COL_SDUI = 10   # Source Descriptor UI (e.g., "D015738" for MeSH)
COL_SAB  = 11   # Source Abbreviation (e.g., "MSH", "SNOMEDCT_US", "DRUGBANK")
COL_STR  = 14   # String (the actual name/synonym)


class UMLSEnricher:
    """
    Extracts extra synonyms for MeSH entities from UMLS MRCONSO.RRF.

    The approach:
      1. Scan MRCONSO for rows where SAB="MSH" → maps MeSH ID → CUI
      2. Scan MRCONSO for all English rows with those CUIs → collects synonyms
      3. Returns new synonyms grouped by MeSH ID

    Parameters
    ----------
    mrconso_path : str or Path
        Path to MRCONSO.RRF file.
    cache_path : str or Path or None
        Path to cache file. Set to None to disable caching.
    """

    def __init__(
        self,
        mrconso_path: str | Path,
        cache_path: str | Path | None = DEFAULT_CACHE_PATH,
    ):
        self.mrconso_path = Path(mrconso_path)
        self.cache_path = Path(cache_path) if cache_path else None

        if not self.mrconso_path.exists():
            raise FileNotFoundError(
                f"MRCONSO.RRF not found at {self.mrconso_path}. "
                "Download from https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html"
            )

    def get_mesh_synonyms(self, force_refresh: bool = False) -> dict[str, list[str]]:
        """
        Get additional synonyms for MeSH entities from UMLS.

        Parameters
        ----------
        force_refresh : bool
            If True, ignore cache and re-parse MRCONSO.RRF.

        Returns
        -------
        dict[str, list[str]]
            Mapping from MeSH ID to list of extra synonyms from other vocabularies.
        """
        # Try cache first
        if not force_refresh and self.cache_path and self.cache_path.exists():
            print(f"Loading UMLS synonyms from cache: {self.cache_path}")
            return self._load_cache()

        # Parse MRCONSO.RRF (two passes for memory efficiency)
        synonyms = self._parse_mrconso()

        # Save to cache
        if self.cache_path:
            self._save_cache(synonyms)
            print(f"  Cached {len(synonyms)} MeSH mappings to {self.cache_path}")

        return synonyms

    def _parse_mrconso(self) -> dict[str, list[str]]:
        """
        Parse MRCONSO.RRF in two passes:
          Pass 1: Find all MeSH ID → CUI mappings (rows where SAB="MSH")
          Pass 2: For those CUIs, collect all English synonyms from ALL vocabularies
        """
        # ── Pass 1: MeSH ID → CUI mapping ──
        print("  Pass 1: Finding MeSH → CUI mappings...")
        mesh_to_cui: dict[str, str] = {}     # MeSH ID → CUI
        cui_to_mesh: dict[str, str] = {}     # CUI → MeSH ID (reverse lookup)
        line_count = 0

        with open(self.mrconso_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                if line_count % 2_000_000 == 0:
                    print(f"    ... {line_count // 1_000_000}M lines scanned", flush=True)

                parts = line.split("|")
                # Only look at MeSH source entries
                if parts[COL_SAB] == "MSH":
                    cui = parts[COL_CUI]
                    mesh_id = parts[COL_SDUI]
                    if mesh_id and mesh_id not in mesh_to_cui:
                        mesh_to_cui[mesh_id] = cui
                        cui_to_mesh[cui] = mesh_id

        print(f"    Found {len(mesh_to_cui)} MeSH → CUI mappings")

        # ── Pass 2: Collect English synonyms for those CUIs ──
        print("  Pass 2: Collecting English synonyms from all vocabularies...")
        synonyms: dict[str, set[str]] = {mid: set() for mid in mesh_to_cui}
        line_count = 0

        with open(self.mrconso_path, "r", encoding="utf-8") as f:
            for line in f:
                line_count += 1
                if line_count % 2_000_000 == 0:
                    print(f"    ... {line_count // 1_000_000}M lines scanned", flush=True)

                parts = line.split("|")

                # Only English entries
                if parts[COL_LAT] != "ENG":
                    continue

                cui = parts[COL_CUI]

                # Only CUIs we care about (mapped to a MeSH entity)
                if cui not in cui_to_mesh:
                    continue

                name = parts[COL_STR]
                mesh_id = cui_to_mesh[cui]

                if name:
                    synonyms[mesh_id].add(name)

        # Remove empty entries and convert to sorted lists
        result = {
            mid: sorted(syns)
            for mid, syns in synonyms.items()
            if syns
        }

        total_syns = sum(len(s) for s in result.values())
        print(f"    Collected {total_syns} synonyms for {len(result)} MeSH entities")

        return result

    def _load_cache(self) -> dict[str, list[str]]:
        """Load cached synonyms from JSON file."""
        with open(self.cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} MeSH mappings from cache")
        return data

    def _save_cache(self, synonyms: dict[str, list[str]]):
        """Save synonyms to JSON cache file."""
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(synonyms, f, ensure_ascii=False)


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    mrconso_path = PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF"

    enricher = UMLSEnricher(mrconso_path)
    synonyms = enricher.get_mesh_synonyms()

    print(f"\nTotal MeSH entities with UMLS synonyms: {len(synonyms)}")

    # Show some examples
    examples = ["D015738", "D004317", "D006973", "D003693", "D001241"]
    for mesh_id in examples:
        if mesh_id in synonyms:
            # Show first 10 synonyms
            syns = synonyms[mesh_id]
            print(f"\n  {mesh_id} ({len(syns)} synonyms): {syns[:10]}")
        else:
            print(f"\n  {mesh_id}: (not found in UMLS)")
