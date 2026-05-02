"""
Candidate Retriever
====================
Phase 2 of the BioLinkerAI pipeline.

Given an entity mention (surface form), retrieves a ranked list of
candidate entities from the MeSH knowledge base. This implements the
candidate generation process described in Section 3 of the paper:
    1. Search the Background Knowledge via string similarity
    2. Enrich candidates with aliases
    3. Rank with BM25 / similarity scoring

Usage:
    from mesh_index import MeSHIndex
    from candidate_retriever import CandidateRetriever

    index = MeSHIndex()
    index.build_from_xml("Data/MeSH/desc2026.xml", "Data/MeSH/supp2026.xml")

    retriever = CandidateRetriever(index)
    candidates = retriever.retrieve("famotidine", top_k=10)
"""

import sys
from pathlib import Path

import pandas as pd

from mesh_index import MeSHIndex, CandidateEntity

# Add project root to path so we can import pubtator_parser
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from pubtator_parser import parse_pubtator


# ── Main retriever class ───────────────────────────────────────────────────

class CandidateRetriever:
    """
    Retrieves and ranks candidate entities for a given mention.

    Wraps the MeSHIndex search and adds the full retrieval pipeline
    as described in the BioLinkerAI paper (Phase 2).

    Parameters
    ----------
    index : MeSHIndex
        A built MeSH search index.
    top_k : int
        Default number of candidates to retrieve per mention.
    """

    def __init__(self, index: MeSHIndex, top_k: int = 10):
        self.index = index
        self.top_k = top_k

    def retrieve(self, mention: str, top_k: int | None = None) -> list[CandidateEntity]:
        """
        Retrieve top-k candidate entities for a mention.

        Parameters
        ----------
        mention : str
            The entity surface form (e.g., "famotidine", "seizures").
        top_k : int or None
            Number of candidates to return. Uses default if None.

        Returns
        -------
        list[CandidateEntity]
            Ranked list of candidates (best match first).
        """
        k = top_k or self.top_k
        candidates = self.index.search(mention, top_k=k)
        return candidates

    def retrieve_batch(
        self,
        mentions: list[str],
        top_k: int | None = None,
    ) -> dict[str, list[CandidateEntity]]:
        """
        Retrieve candidates for a batch of mentions.

        Parameters
        ----------
        mentions : list[str]
            List of entity surface forms.
        top_k : int or None
            Number of candidates per mention.

        Returns
        -------
        dict[str, list[CandidateEntity]]
            Mapping from mention text to its candidate list.
        """
        return {
            mention: self.retrieve(mention, top_k)
            for mention in mentions
        }


# ── Evaluation ─────────────────────────────────────────────────────────────

def _build_id_mapping(index: MeSHIndex, mrconso_path: str | None = None) -> dict[str, set[str]]:
    """
    Build a mapping from old/retired MeSH IDs to current MeSH IDs.

    Some entities in older datasets (like BC5CDR from 2016) use MeSH IDs
    that have since been retired or promoted. For example, Clopidogrel
    was C055162 (supplementary) but is now D000077144 (descriptor).

    Approach: Use UMLS CUI mappings if MRCONSO is available (most accurate).
    Fallback: Build a name-based mapping from the index itself.

    Parameters
    ----------
    index : MeSHIndex
        The built MeSH index (used for name-based fallback).
    mrconso_path : str or None
        Path to MRCONSO.RRF for CUI-based mapping.

    Returns
    -------
    dict[str, set[str]]
        Mapping from old MeSH ID → set of equivalent current MeSH IDs.
    """
    id_map: dict[str, set[str]] = {}

    if mrconso_path and Path(mrconso_path).exists():
        # ── CUI-based mapping (most accurate) ──
        # Step 1: Collect all CUI → MeSH ID associations
        print("  Building ID mapping from MRCONSO.RRF...")
        cui_to_mesh_ids: dict[str, set[str]] = {}

        with open(mrconso_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split("|")
                # Only MeSH source entries (SAB starts with "MSH")
                if parts[11].startswith("MSH"):
                    cui = parts[0]
                    mesh_id = parts[10]
                    if mesh_id:
                        if cui not in cui_to_mesh_ids:
                            cui_to_mesh_ids[cui] = set()
                        cui_to_mesh_ids[cui].add(mesh_id)

        # Step 2: For any CUI with multiple MeSH IDs, create cross-mappings
        for cui, mesh_ids in cui_to_mesh_ids.items():
            if len(mesh_ids) > 1:
                for mid in mesh_ids:
                    id_map[mid] = mesh_ids - {mid}

        print(f"  Found {len(id_map)} MeSH IDs with alternative mappings")

    # ── Name-based fallback: build label → mesh_id lookup from index ──
    # This catches cases where old C-IDs are completely gone from MRCONSO
    label_to_ids: dict[str, set[str]] = {}
    for mesh_id, entity in index.entities.items():
        label_lower = entity.preferred_label.lower()
        if label_lower not in label_to_ids:
            label_to_ids[label_lower] = set()
        label_to_ids[label_lower].add(mesh_id)

    # Store for use during evaluation
    id_map["__label_lookup__"] = None  # sentinel
    id_map["__label_to_ids__"] = label_to_ids  # type: ignore

    return id_map


def evaluate_candidate_recall(
    retriever: CandidateRetriever,
    annotations_df: pd.DataFrame,
    k_values: list[int] | None = None,
    mrconso_path: str | None = None,
) -> dict[str, float]:
    """
    Evaluate candidate generation quality on a gold-annotated dataset.

    For each unique (mention, mesh_id) pair, checks whether the correct
    MeSH ID appears in the top-k candidates. Reports Accuracy@k.

    Handles MeSH ID version mismatches: if a gold ID (e.g., C055162)
    has been retired/promoted to a new ID (e.g., D000077144), the new
    ID is also accepted as correct.

    Parameters
    ----------
    retriever : CandidateRetriever
        The retriever to evaluate.
    annotations_df : pd.DataFrame
        Gold annotations with columns: mention, mesh_id, entity_type.
        Loaded via pubtator_parser.parse_pubtator().
    k_values : list[int] or None
        Values of k to evaluate. Default: [1, 5, 10, 20].
    mrconso_path : str or None
        Path to MRCONSO.RRF for ID mapping. If None, uses name-based fallback.

    Returns
    -------
    dict[str, float]
        Results dict with keys like "accuracy@1", "accuracy@5", etc.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    max_k = max(k_values)

    # Build ID mapping for version mismatches
    id_map = _build_id_mapping(retriever.index, mrconso_path)
    label_to_ids = id_map.pop("__label_to_ids__", {})  # type: ignore
    id_map.pop("__label_lookup__", None)

    # Deduplicate: evaluate each unique (mention, mesh_id) pair once
    # Skip unlinkable entities (mesh_id == "-1")
    eval_pairs = (
        annotations_df[annotations_df["mesh_id"] != "-1"]
        [["mention", "mesh_id"]]
        .drop_duplicates()
    )

    n_pairs = len(eval_pairs)
    print(f"Evaluating {n_pairs} unique (mention, mesh_id) pairs...")

    # Track hits at each k
    hits = {k: 0 for k in k_values}
    total = 0
    id_mapping_saves = 0  # count how many we rescued via ID mapping

    for _, row in eval_pairs.iterrows():
        mention = row["mention"]
        gold_id = row["mesh_id"]

        # Handle composite gold IDs (e.g., "C467567|D003907")
        gold_ids = set(gold_id.split("|"))

        # Expand gold IDs with known equivalents (old → new mappings)
        expanded_gold_ids = set(gold_ids)
        for gid in gold_ids:
            # CUI-based mapping
            if gid in id_map:
                expanded_gold_ids.update(id_map[gid])
            # Name-based fallback: if gold ID not in index, look up by name
            if gid not in retriever.index.entities:
                gold_entity = retriever.index.lookup(gid)
                if gold_entity is None:
                    # Try finding by name from any candidate with matching label
                    # We check MRCONSO mapping first, then try label lookup
                    pass  # CUI mapping above handles most cases

        # Also check: for any gold ID not in index, find equivalent by label
        for gid in list(gold_ids):
            if gid not in retriever.index.entities:
                # This gold ID doesn't exist in our index — maybe it was promoted
                # Try to find it via label matching in candidates
                # (we'll check this against returned candidates below)
                pass

        # Retrieve candidates
        candidates = retriever.retrieve(mention, top_k=max_k)
        candidate_ids = [c.mesh_id for c in candidates]

        # For gold IDs not in index: accept any candidate whose preferred_label
        # matches a known label for this concept (name-based rescue)
        for gid in gold_ids:
            if gid not in retriever.index.entities:
                # Find what name this entity should have via UMLS or dataset context
                # Check if any returned candidate has the same name
                for c in candidates:
                    c_label = c.preferred_label.lower()
                    if c_label in label_to_ids:
                        # If there are entities with this label, check if any
                        # share a mapping with our gold ID
                        if gid in id_map and c.mesh_id in id_map[gid]:
                            expanded_gold_ids.add(c.mesh_id)

        # Check if gold ID (or equivalent) is in top-k for each k
        was_remapped = expanded_gold_ids != gold_ids
        for k in k_values:
            top_k_ids = set(candidate_ids[:k])
            if expanded_gold_ids & top_k_ids:
                hits[k] += 1
                if was_remapped and not (gold_ids & top_k_ids):
                    # This was rescued by ID mapping
                    if k == max_k:
                        id_mapping_saves += 1

        total += 1

        # Progress indicator
        if total % 100 == 0:
            print(f"  ... {total}/{n_pairs} mentions evaluated ({total*100//n_pairs}%)", flush=True)

    # Compute accuracy
    results = {}
    for k in k_values:
        acc = (hits[k] / total * 100) if total > 0 else 0.0
        results[f"accuracy@{k}"] = acc
        print(f"  Accuracy@{k}: {acc:.1f}% ({hits[k]}/{total})")

    if id_mapping_saves > 0:
        print(f"  ({id_mapping_saves} mentions rescued by ID remapping)")

    results["total_pairs"] = total
    results["id_mapping_saves"] = id_mapping_saves
    return results


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(description="Candidate Retriever Demo + Evaluation")
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
    args = parser.parse_args()

    # ── Step 1: Build the MeSH index ──
    print("=" * 60)
    print(f"Building MeSH index (backend={args.backend}, "
          f"wikidata={args.wikidata}, dbpedia={args.dbpedia}, "
          f"umls={'yes' if args.umls else 'no'})...")
    print("=" * 60)

    index = MeSHIndex(backend=args.backend, es_url=args.es_url)
    index.build_from_xml(
        descriptor_path=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
        supplementary_path=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
        enrich_wikidata=args.wikidata,
        enrich_dbpedia=args.dbpedia,
        enrich_umls=args.umls,
    )

    retriever = CandidateRetriever(index, top_k=10)

    # ── Step 2: Test with some mentions from BC5CDR ──
    print("\n" + "=" * 60)
    print("Sample retrievals from BC5CDR mentions")
    print("=" * 60)

    test_mentions = [
        ("Famotidine", "D015738"),
        ("delirium", "D003693"),
        ("seizures", "D012640"),
        ("doxorubicin", "D004317"),
        ("hypotension", "D007022"),
        ("IDM", "D007213"),         # abbreviation for indomethacin
    ]

    for mention, gold_id in test_mentions:
        candidates = retriever.retrieve(mention, top_k=5)
        gold_entity = index.lookup(gold_id)
        gold_label = gold_entity.preferred_label if gold_entity else "???"

        # Check if gold is in top-k
        candidate_ids = [c.mesh_id for c in candidates]
        found_at = None
        for i, cid in enumerate(candidate_ids):
            if cid == gold_id:
                found_at = i + 1
                break

        status = f"✓ found at rank {found_at}" if found_at else "✗ NOT in top-5"

        print(f'\n"{mention}" (gold: {gold_id} = {gold_label}) → {status}')
        for i, c in enumerate(candidates):
            marker = " ◀ GOLD" if c.mesh_id == gold_id else ""
            print(f"  {i+1}. [{c.mesh_id}] {c.preferred_label} "
                  f"(score={c.score:.1f}){marker}")

    # ── Step 3: Evaluate on BC5CDR test set ──
    print("\n" + "=" * 60)
    print("Evaluation on BC5CDR test set")
    print("=" * 60)

    meta, anns, rels = parse_pubtator(
        str(PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.PubTator.txt")
    )

    t0 = time.time()
    results = evaluate_candidate_recall(
        retriever, anns, k_values=[1, 5, 10, 20], mrconso_path=args.umls,
    )
    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed:.1f}s")
