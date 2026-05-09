"""
UMLS Relation-Based Candidate Expansion
=========================================
Builds a "bridge" from non-MeSH UMLS concept names to related MeSH entities
using MRREL.RRF relations. This addresses the biggest failure category in
Phase 2: mentions that exist in UMLS but have no direct MeSH mapping, so
fuzzy string matching against MeSH labels can never find them.

Example:
    "myelosuppression" (CUI C0280962) has no MeSH ID,
    but MRREL says it ISA "Anemia" (C0002871 → MeSH D000740).
    So we add "myelosuppression" to our search index, pointing to D000740.

Build process (one-time, cached):
    1. MRCONSO: Build CUI → MeSH ID mapping (which CUIs have MeSH?)
    2. MRREL: Build CUI → related CUIs (parent, child, broader, narrower)
    3. MRCONSO: For CUIs WITHOUT MeSH, collect English names
       and bridge them to related MeSH IDs via relations (1-2 hops)

Usage:
    expander = UMLSRelationExpander(
        mrconso_path="Data/UMLS/MRCONSO.RRF",
        mrrel_path="Data/UMLS/MRREL.RRF",
    )
    bridge = expander.build_bridge()  # one-time, cached
    mesh_ids = expander.lookup("myelosuppression")
    # → [("D000740", 1), ("D009503", 1), ...]  (MeSH ID, hop distance)
"""

import json
import time
from pathlib import Path
from collections import defaultdict


class UMLSRelationExpander:
    """
    Builds and queries a bridge from non-MeSH UMLS names to related MeSH IDs.

    Parameters
    ----------
    mrconso_path : str
        Path to MRCONSO.RRF.
    mrrel_path : str
        Path to MRREL.RRF.
    cache_dir : str or Path
        Directory for cached bridge data.
    """

    def __init__(
        self,
        mrconso_path: str,
        mrrel_path: str,
        cache_dir: str | Path | None = None,
    ):
        self.mrconso_path = mrconso_path
        self.mrrel_path = mrrel_path

        if cache_dir is None:
            cache_dir = Path(__file__).parent / "cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

        self.bridge: dict[str, list[str]] = {}  # name (lower) → list of MeSH IDs
        self.cui_to_mesh: dict[str, set[str]] = {}  # CUI → set of MeSH IDs

    def build_bridge(self, force_rebuild: bool = False) -> dict[str, list[str]]:
        """
        Build or load the bridge cache.

        Returns
        -------
        dict[str, list[str]]
            Mapping from lowercase name → list of related MeSH IDs.
        """
        cache_path = self.cache_dir / "umls_bridge_cache.json"

        if cache_path.exists() and not force_rebuild:
            print(f"Loading UMLS bridge cache from {cache_path}...")
            with open(cache_path, "r") as f:
                self.bridge = json.load(f)
            print(f"  {len(self.bridge)} bridge entries loaded")
            return self.bridge

        print("Building UMLS relation bridge (this takes ~30-60s)...")
        t0 = time.time()

        # ── Step 1: CUI → MeSH IDs ──
        print("  Step 1: Building CUI → MeSH ID mapping from MRCONSO...")
        self.cui_to_mesh = self._build_cui_to_mesh()
        mesh_cuis = set(self.cui_to_mesh.keys())
        print(f"    {len(mesh_cuis)} CUIs have MeSH mappings")

        # ── Step 2: CUI → related CUIs from MRREL ──
        print("  Step 2: Building CUI relation graph from MRREL...")
        cui_parents, cui_children, cui_broader, cui_narrower = self._build_relations()
        print(f"    PAR/CHD: {sum(len(v) for v in cui_parents.values())} parent links")
        print(f"    RB/RN: {sum(len(v) for v in cui_broader.values())} broader links")

        # ── Step 3: Build bridge for non-MeSH CUIs ──
        print("  Step 3: Building name → MeSH bridge from MRCONSO...")
        self.bridge = self._build_name_bridge(
            mesh_cuis, cui_parents, cui_children, cui_broader, cui_narrower,
        )
        print(f"    {len(self.bridge)} bridge entries built")

        # ── Save cache ──
        with open(cache_path, "w") as f:
            json.dump(self.bridge, f)
        elapsed = time.time() - t0
        print(f"  Bridge built in {elapsed:.0f}s, saved to {cache_path}")

        return self.bridge

    def _build_cui_to_mesh(self) -> dict[str, set[str]]:
        """Scan MRCONSO to find which CUIs map to MeSH IDs."""
        cui_to_mesh: dict[str, set[str]] = defaultdict(set)

        with open(self.mrconso_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("|")
                cui = parts[0]
                source = parts[11]
                code = parts[10]  # SCUI or code — for MSH this is the MeSH ID

                if source.startswith("MSH") and code:
                    cui_to_mesh[cui].add(code)

        return dict(cui_to_mesh)

    def _build_relations(self):
        """
        Scan MRREL to build relation maps.
        Only keeps hierarchical relations (PAR/CHD, RB/RN) that are
        most useful for bridging to broader/narrower concepts.
        """
        cui_parents = defaultdict(set)    # CUI → parent CUIs (via PAR or inverse_isa)
        cui_children = defaultdict(set)   # CUI → child CUIs
        cui_broader = defaultdict(set)    # CUI → broader CUIs (via RB)
        cui_narrower = defaultdict(set)   # CUI → narrower CUIs (via RN)

        with open(self.mrrel_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("|")
                cui1 = parts[0]
                rel = parts[3]   # REL: PAR, CHD, RB, RN, RO, SY, etc.
                cui2 = parts[4]

                if rel == "PAR":
                    # CUI1 has parent CUI2
                    cui_parents[cui1].add(cui2)
                    cui_children[cui2].add(cui1)
                elif rel == "CHD":
                    # CUI1 has child CUI2
                    cui_children[cui1].add(cui2)
                    cui_parents[cui2].add(cui1)
                elif rel == "RB":
                    # CUI1 has broader concept CUI2
                    cui_broader[cui1].add(cui2)
                    cui_narrower[cui2].add(cui1)
                elif rel == "RN":
                    # CUI1 has narrower concept CUI2
                    cui_narrower[cui1].add(cui2)
                    cui_broader[cui2].add(cui1)

        return (
            dict(cui_parents),
            dict(cui_children),
            dict(cui_broader),
            dict(cui_narrower),
        )

    def _build_name_bridge(
        self,
        mesh_cuis: set[str],
        cui_parents: dict,
        cui_children: dict,
        cui_broader: dict,
        cui_narrower: dict,
    ) -> dict[str, list[str]]:
        """
        For CUIs without MeSH, find English names and bridge them
        to related MeSH IDs via 1-2 hops of hierarchical relations.
        """
        # First: for each CUI, compute which MeSH IDs it can reach
        # via relations (1 hop first, then 2 hops if needed)
        cui_to_bridge_mesh: dict[str, set[str]] = {}

        # Collect all CUIs that appear in any relation
        all_relation_cuis = set()
        for d in [cui_parents, cui_children, cui_broader, cui_narrower]:
            all_relation_cuis.update(d.keys())
            for vals in d.values():
                all_relation_cuis.update(vals)

        # Only process CUIs that DON'T have MeSH (those that do are already indexed)
        non_mesh_cuis = all_relation_cuis - mesh_cuis

        for cui in non_mesh_cuis:
            related_mesh = set()

            # 1-hop: check parents, broader, children, narrower
            # Prioritize parents/broader (going UP the hierarchy)
            for related_cui in cui_parents.get(cui, set()):
                if related_cui in self.cui_to_mesh:
                    related_mesh.update(self.cui_to_mesh[related_cui])
            for related_cui in cui_broader.get(cui, set()):
                if related_cui in self.cui_to_mesh:
                    related_mesh.update(self.cui_to_mesh[related_cui])
            # Also check children/narrower (going DOWN)
            for related_cui in cui_children.get(cui, set()):
                if related_cui in self.cui_to_mesh:
                    related_mesh.update(self.cui_to_mesh[related_cui])
            for related_cui in cui_narrower.get(cui, set()):
                if related_cui in self.cui_to_mesh:
                    related_mesh.update(self.cui_to_mesh[related_cui])

            # 2-hop: if no MeSH found at 1 hop, try going through parents' parents
            if not related_mesh:
                for hop1_cui in cui_parents.get(cui, set()):
                    for hop2_cui in cui_parents.get(hop1_cui, set()):
                        if hop2_cui in self.cui_to_mesh:
                            related_mesh.update(self.cui_to_mesh[hop2_cui])
                    for hop2_cui in cui_broader.get(hop1_cui, set()):
                        if hop2_cui in self.cui_to_mesh:
                            related_mesh.update(self.cui_to_mesh[hop2_cui])
                # Also try broader → broader
                for hop1_cui in cui_broader.get(cui, set()):
                    for hop2_cui in cui_broader.get(hop1_cui, set()):
                        if hop2_cui in self.cui_to_mesh:
                            related_mesh.update(self.cui_to_mesh[hop2_cui])
                    for hop2_cui in cui_parents.get(hop1_cui, set()):
                        if hop2_cui in self.cui_to_mesh:
                            related_mesh.update(self.cui_to_mesh[hop2_cui])

            if related_mesh:
                # Cap at 15 MeSH IDs per CUI to avoid bloat
                cui_to_bridge_mesh[cui] = set(list(related_mesh)[:15])

        print(f"    {len(cui_to_bridge_mesh)} non-MeSH CUIs bridged to MeSH")

        # Now scan MRCONSO again to collect English names for these CUIs
        bridge: dict[str, set[str]] = defaultdict(set)

        with open(self.mrconso_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("|")
                cui = parts[0]

                if cui not in cui_to_bridge_mesh:
                    continue

                lang = parts[1]
                if lang != "ENG":
                    continue

                name = parts[14].strip().lower()
                if len(name) < 3:
                    continue

                bridge[name].update(cui_to_bridge_mesh[cui])

        # Convert sets to lists for JSON serialization
        return {name: list(mesh_ids) for name, mesh_ids in bridge.items()}

    def lookup(self, mention: str) -> list[str]:
        """
        Look up a mention in the bridge cache.

        Parameters
        ----------
        mention : str
            The entity mention text.

        Returns
        -------
        list[str]
            Related MeSH IDs (empty if not found).
        """
        return self.bridge.get(mention.lower().strip(), [])

    def add_bridge_labels_to_index(self, mesh_index) -> int:
        """
        Add bridge labels to the MeSH search index.
        Each bridge name becomes a searchable label pointing to the related MeSH entity.

        Parameters
        ----------
        mesh_index : MeSHIndex
            The MeSH index to add labels to.

        Returns
        -------
        int
            Number of labels added.
        """
        added = 0
        for name, mesh_ids in self.bridge.items():
            for mesh_id in mesh_ids:
                if mesh_id in mesh_index.entities:
                    mesh_index._label_index.append((name, mesh_id))
                    added += 1
        print(f"  UMLS bridge: added {added} labels to search index")
        return added


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

    expander = UMLSRelationExpander(
        mrconso_path=str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF"),
        mrrel_path=str(PROJECT_ROOT / "Data" / "UMLS" / "MRREL.RRF"),
    )

    bridge = expander.build_bridge(force_rebuild=True)

    # Test our failure cases
    test_mentions = [
        "scleroderma renal crisis",
        "myelosuppression",
        "psychotic symptoms",
        "psychotic symptom",
        "bone marrow suppression",
        "clopidogrel",
    ]

    print("\n" + "=" * 60)
    print("Bridge lookup tests")
    print("=" * 60)
    for mention in test_mentions:
        mesh_ids = expander.lookup(mention)
        print(f'\n  "{mention}" → {len(mesh_ids)} MeSH IDs:')
        for mid in mesh_ids[:10]:
            print(f"    {mid}")
