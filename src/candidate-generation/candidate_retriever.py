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

def evaluate_candidate_recall(
    retriever: CandidateRetriever,
    annotations_df: pd.DataFrame,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Evaluate candidate generation quality on a gold-annotated dataset.

    For each unique (mention, mesh_id) pair, checks whether the correct
    MeSH ID appears in the top-k candidates. Reports Accuracy@k.

    This measures the upper bound on pipeline performance — if the correct
    candidate is not retrieved, no amount of re-ranking can fix it.

    Parameters
    ----------
    retriever : CandidateRetriever
        The retriever to evaluate.
    annotations_df : pd.DataFrame
        Gold annotations with columns: mention, mesh_id, entity_type.
        Loaded via pubtator_parser.parse_pubtator().
    k_values : list[int] or None
        Values of k to evaluate. Default: [1, 5, 10, 20].

    Returns
    -------
    dict[str, float]
        Results dict with keys like "accuracy@1", "accuracy@5", etc.
    """
    if k_values is None:
        k_values = [1, 5, 10, 20]

    max_k = max(k_values)

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

    for _, row in eval_pairs.iterrows():
        mention = row["mention"]
        gold_id = row["mesh_id"]

        # Handle composite gold IDs (e.g., "C467567|D003907")
        gold_ids = set(gold_id.split("|"))

        # Retrieve candidates
        candidates = retriever.retrieve(mention, top_k=max_k)
        candidate_ids = [c.mesh_id for c in candidates]

        # Check if gold ID is in top-k for each k
        for k in k_values:
            top_k_ids = set(candidate_ids[:k])
            if gold_ids & top_k_ids:  # any overlap = hit
                hits[k] += 1

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

    results["total_pairs"] = total
    return results


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    # ── Step 1: Build the MeSH index ──
    print("=" * 60)
    print("Building MeSH index...")
    print("=" * 60)

    index = MeSHIndex()
    index.build_from_xml(
        descriptor_path=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
        supplementary_path=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
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
    results = evaluate_candidate_recall(retriever, anns, k_values=[1, 5, 10, 20])
    elapsed = time.time() - t0
    print(f"\nEvaluation completed in {elapsed:.1f}s")
