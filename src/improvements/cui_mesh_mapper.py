"""
CUI → MeSH ID Mapper
=====================
Maps UMLS Concept Unique Identifiers (CUIs) to MeSH Descriptor/Supplemental IDs
using MRCONSO.RRF from the UMLS Metathesaurus.

This is needed for evaluating on MedMentions, which uses UMLS CUIs as gold
standard IDs, while our pipeline returns MeSH IDs.

MRCONSO.RRF format (pipe-delimited):
    CUI|LAT|TS|LUI|STT|SUI|ISPREF|AUI|SAUI|SCUI|SDUI|SAB|TTY|CODE|STR|SRL|SUPPRESS|CVF

    - Column 0  (CUI):  Concept Unique Identifier (e.g., C0027051)
    - Column 11 (SAB):  Source abbreviation (e.g., MSH = MeSH)
    - Column 10 (SDUI): Source Descriptor UI (e.g., D009203 for MeSH)

We extract rows where SAB == "MSH" and build a CUI → set(MeSH IDs) mapping.
One CUI can map to multiple MeSH IDs (e.g., a CUI covering both a descriptor
and a supplementary concept).

Usage:
    from cui_mesh_mapper import CUIToMeSHMapper

    mapper = CUIToMeSHMapper("Data/UMLS/MRCONSO.RRF")
    mesh_ids = mapper.cui_to_mesh("C0027051")  # → {"D009203"}
    mesh_ids = mapper.cui_to_mesh("C9999999")  # → set()  (unmappable)
"""

import json
from pathlib import Path


class CUIToMeSHMapper:
    """
    Maps UMLS CUIs to MeSH IDs using MRCONSO.RRF.

    Builds the mapping on first use and caches it to disk as JSON
    for fast subsequent loads.
    """

    def __init__(self, mrconso_path: str, cache_dir: str = None):
        self.mrconso_path = Path(mrconso_path)
        self.cache_dir = Path(cache_dir) if cache_dir else self.mrconso_path.parent
        self.cache_path = self.cache_dir / "cui_to_mesh_cache.json"
        self._mapping: dict[str, list[str]] = {}
        self._loaded = False

    def _build_mapping(self) -> dict[str, list[str]]:
        """Parse MRCONSO.RRF and extract CUI → MeSH ID mappings."""
        print(f"  Building CUI→MeSH mapping from {self.mrconso_path}...")
        mapping: dict[str, set[str]] = {}

        with open(self.mrconso_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("|")
                if len(parts) < 12:
                    continue
                sab = parts[11]   # Source abbreviation
                if sab != "MSH":
                    continue
                cui = parts[0]    # CUI
                sdui = parts[10]  # MeSH descriptor/supplemental ID
                if cui and sdui:
                    if cui not in mapping:
                        mapping[cui] = set()
                    mapping[cui].add(sdui)

        # Convert sets to sorted lists for JSON serialization
        result = {cui: sorted(ids) for cui, ids in mapping.items()}
        print(f"  Built mapping: {len(result)} CUIs → MeSH IDs")
        return result

    def _save_cache(self, mapping: dict[str, list[str]]):
        """Save mapping to JSON cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "w") as f:
            json.dump(mapping, f)
        print(f"  Cached CUI→MeSH mapping to {self.cache_path}")

    def _load_cache(self) -> dict[str, list[str]] | None:
        """Load mapping from JSON cache if available."""
        if self.cache_path.exists():
            with open(self.cache_path, "r") as f:
                mapping = json.load(f)
            print(f"  Loaded CUI→MeSH cache: {len(mapping)} entries")
            return mapping
        return None

    def load(self):
        """Load or build the CUI→MeSH mapping."""
        if self._loaded:
            return

        # Try cache first
        cached = self._load_cache()
        if cached is not None:
            self._mapping = cached
            self._loaded = True
            return

        # Build from MRCONSO
        if not self.mrconso_path.exists():
            print(f"  WARNING: MRCONSO not found at {self.mrconso_path}")
            print(f"  CUI→MeSH mapping will be empty — MedMentions evaluation won't work")
            self._loaded = True
            return

        self._mapping = self._build_mapping()
        self._save_cache(self._mapping)
        self._loaded = True

    def cui_to_mesh(self, cui: str) -> set[str]:
        """
        Map a CUI to its MeSH ID(s).

        Returns a set of MeSH IDs, or empty set if not mappable.
        """
        if not self._loaded:
            self.load()
        mesh_ids = self._mapping.get(cui, [])
        return set(mesh_ids)

    def is_mappable(self, cui: str) -> bool:
        """Check if a CUI has a MeSH mapping."""
        if not self._loaded:
            self.load()
        return cui in self._mapping

    def __len__(self):
        if not self._loaded:
            self.load()
        return len(self._mapping)

    def __contains__(self, cui: str):
        if not self._loaded:
            self.load()
        return cui in self._mapping


# ── Quick test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    mrconso = sys.argv[1] if len(sys.argv) > 1 else "Data/UMLS/MRCONSO.RRF"
    mapper = CUIToMeSHMapper(mrconso)
    mapper.load()

    # Test some known mappings
    test_cuis = ["C0027051", "C0010674", "C0012345", "C0854135"]
    for cui in test_cuis:
        mesh_ids = mapper.cui_to_mesh(cui)
        print(f"  {cui} → {mesh_ids if mesh_ids else '(no MeSH mapping)'}")
