"""
Phase 3 Evaluation: Domain-Specific Rules on BC5CDR
=====================================================
Evaluates how much the domain-specific re-ranking rules (Phase 3)
improve over raw Phase 2 candidate generation.

Runs Phase 2 → Phase 3 and compares Accuracy@1 before and after re-ranking.

Usage:
    python3 evaluate.py
    python3 evaluate.py --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF
"""

import sys
import time
import json
import argparse
from pathlib import Path

# Add project root and candidate-generation to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))

from pubtator_parser import parse_pubtator
from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever, _build_id_mapping
from domain_rules import DomainRuleReranker


def load_enrichment_caches(args) -> tuple[dict, dict, dict]:
    """Load cached enrichment data for Rule 5 (multi-KG confidence)."""
    cache_dir = PROJECT_ROOT / "src" / "candidate-generation" / "cache"

    wikidata = {}
    dbpedia = {}
    umls = {}

    if args.wikidata:
        wikidata_path = cache_dir / "wikidata_cache.json"
        if wikidata_path.exists():
            with open(wikidata_path, "r") as f:
                wikidata = json.load(f)
            print(f"  Loaded Wikidata cache: {len(wikidata)} entities")
        else:
            print(f"  Warning: Wikidata cache not found at {wikidata_path}")
            print(f"  Run Phase 2 with --wikidata first to generate it.")

    if args.dbpedia:
        dbpedia_path = cache_dir / "dbpedia_cache.json"
        if dbpedia_path.exists():
            with open(dbpedia_path, "r") as f:
                dbpedia = json.load(f)
            print(f"  Loaded DBpedia cache: {len(dbpedia)} entities")
        else:
            print(f"  Warning: DBpedia cache not found at {dbpedia_path}")

    if args.umls:
        umls_path = cache_dir / "umls_cache.json"
        if umls_path.exists():
            with open(umls_path, "r") as f:
                umls = json.load(f)
            print(f"  Loaded UMLS cache: {len(umls)} entities")
        else:
            print(f"  Warning: UMLS cache not found at {umls_path}")

    return wikidata, dbpedia, umls


def run_evaluation(args):
    """Run Phase 2 → Phase 3 evaluation."""

    # ── Step 1: Build MeSH index ──
    print("=" * 60)
    print(f"Phase 2: Building MeSH index (backend={args.backend})...")
    print("=" * 60)

    index = MeSHIndex(backend=args.backend, es_url=args.es_url)
    index.build_from_xml(
        descriptor_path=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
        supplementary_path=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
        enrich_wikidata=args.wikidata,
        enrich_dbpedia=args.dbpedia,
        enrich_umls=args.umls,
    )

    retriever = CandidateRetriever(index, top_k=args.top_k)

    # ── Step 2: Load enrichment caches for Rule 5 ──
    print("\nLoading enrichment caches for Rule 5 (multi-KG confidence)...")
    wikidata, dbpedia, umls = load_enrichment_caches(args)

    # ── Step 3: Create reranker ──
    reranker = DomainRuleReranker(
        mesh_index=index,
        rule5_boost=args.rule5_boost,
        rule6_penalty=args.rule6_penalty,
        rule7_boost=args.rule7_boost,
        wikidata_synonyms=wikidata,
        dbpedia_synonyms=dbpedia,
        umls_synonyms=umls,
    )

    # ── Step 4: Load BC5CDR test set ──
    print("\n" + "=" * 60)
    print("Loading BC5CDR test set...")
    print("=" * 60)

    meta, anns, rels = parse_pubtator(
        str(PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.PubTator.txt")
    )

    # Build context lookup
    context_lookup = {}
    for _, row in meta.iterrows():
        pmid = row["pmid"]
        context_lookup[pmid] = f"{row.get('title', '')} {row.get('abstract', '')}".strip()

    # Build ID mapping for evaluation
    id_map = _build_id_mapping(index, args.umls)
    label_to_ids = id_map.pop("__label_to_ids__", {})
    id_map.pop("__label_lookup__", None)

    # Deduplicate and filter
    eval_df = anns[anns["mesh_id"] != "-1"].drop_duplicates(subset=["mention", "mesh_id"])

    if args.limit:
        eval_df = eval_df.head(args.limit)

    n_pairs = len(eval_df)
    print(f"  {n_pairs} unique (mention, mesh_id) pairs to evaluate")

    # ── Step 5: Run evaluation ──
    print("\n" + "=" * 60)
    print("Running Phase 2 → Phase 3 evaluation...")
    print("=" * 60)

    total = 0
    p2_correct = 0
    p3_correct = 0
    p3_improved = 0
    p3_degraded = 0

    # Per-rule counters
    rule5_triggered = 0
    rule6_triggered = 0
    rule7_triggered = 0

    changes_log = []
    t0 = time.time()

    for _, row in eval_df.iterrows():
        mention = row["mention"]
        gold_id = row["mesh_id"]
        pmid = row["pmid"]
        entity_type = row.get("entity_type", None)

        # Expand gold IDs for version mismatches
        gold_ids = set(gold_id.split("|"))
        expanded_gold_ids = set(gold_ids)
        for gid in gold_ids:
            if gid in id_map:
                expanded_gold_ids.update(id_map[gid])

        # Phase 2: Retrieve candidates
        candidates = retriever.retrieve(mention, top_k=args.top_k)
        if not candidates:
            total += 1
            continue

        # Phase 2 top-1 check
        p2_top1 = candidates[0].mesh_id
        p2_hit = p2_top1 in expanded_gold_ids

        # Phase 3: Re-rank with domain rules
        context = context_lookup.get(pmid, "")
        details = reranker.rerank_with_details(
            mention=mention,
            candidates=candidates,
            entity_type=entity_type,
            context=context,
        )

        # Phase 3 top-1 check
        if details:
            p3_top1 = details[0]["candidate"].mesh_id
            p3_hit = p3_top1 in expanded_gold_ids

            # Track which rules fired
            for d in details:
                if d["rule5_boost"] > 0:
                    rule5_triggered += 1
                    break
            for d in details:
                if d["rule6_penalty"] < 0:
                    rule6_triggered += 1
                    break
            for d in details:
                if d["rule7_boost"] > 0:
                    rule7_triggered += 1
                    break
        else:
            p3_hit = False

        if p2_hit:
            p2_correct += 1
        if p3_hit:
            p3_correct += 1
        if p3_hit and not p2_hit:
            p3_improved += 1
        if p2_hit and not p3_hit:
            p3_degraded += 1

        # Log changes
        if p2_hit != p3_hit:
            changes_log.append({
                "mention": mention,
                "gold_id": gold_id,
                "entity_type": entity_type,
                "p2_top1": p2_top1,
                "p2_top1_label": candidates[0].preferred_label,
                "p3_top1": p3_top1 if details else "?",
                "p3_top1_label": details[0]["candidate"].preferred_label if details else "?",
                "p2_correct": p2_hit,
                "p3_correct": p3_hit,
                "rule5": details[0]["rule5_boost"] if details else 0,
                "rule6": details[0]["rule6_penalty"] if details else 0,
                "rule7": details[0]["rule7_boost"] if details else 0,
            })

        total += 1
        if total % 200 == 0:
            elapsed = time.time() - t0
            print(f"  ... {total}/{n_pairs} ({total*100//n_pairs}%) "
                  f"— P2: {p2_correct*100/total:.1f}% "
                  f"P3: {p3_correct*100/total:.1f}%",
                  flush=True)

    elapsed = time.time() - t0

    # ── Results ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    p2_acc = p2_correct / total * 100 if total > 0 else 0
    p3_acc = p3_correct / total * 100 if total > 0 else 0

    print(f"  Total mentions:          {total}")
    print(f"  Phase 2 Accuracy@1:      {p2_acc:.1f}% ({p2_correct}/{total})")
    print(f"  Phase 3 Accuracy@1:      {p3_acc:.1f}% ({p3_correct}/{total})")
    print(f"  Improvement (P3 vs P2):  {p3_acc - p2_acc:+.1f}%")
    print(f"")
    print(f"  P3 improved over P2:     {p3_improved} mentions")
    print(f"  P3 degraded vs P2:       {p3_degraded} mentions")
    print(f"  Net change:              {p3_improved - p3_degraded:+d} mentions")
    print(f"")
    print(f"  Rule 5 (multi-KG) triggered on:    {rule5_triggered} mentions")
    print(f"  Rule 6 (sem type) triggered on:    {rule6_triggered} mentions")
    print(f"  Rule 7 (definition) triggered on:  {rule7_triggered} mentions")
    print(f"")
    print(f"  Rule weights: R5={args.rule5_boost}, R6={args.rule6_penalty}, R7={args.rule7_boost}")
    print(f"  Time: {elapsed:.0f}s")

    # Show examples
    improvements = [r for r in changes_log if r["p3_correct"]]
    degradations = [r for r in changes_log if r["p2_correct"]]

    if improvements:
        print(f"\n── Examples: Phase 3 IMPROVED ({len(improvements)} total) ──")
        for r in improvements[:5]:
            print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
            print(f"    P2: [{r['p2_top1']}] {r['p2_top1_label']} (wrong)")
            print(f"    P3: [{r['p3_top1']}] {r['p3_top1_label']} (correct)")
            print(f"    Rules: R5={r['rule5']:+.1f} R6={r['rule6']:+.1f} R7={r['rule7']:+.1f}")

    if degradations:
        print(f"\n── Examples: Phase 3 DEGRADED ({len(degradations)} total) ──")
        for r in degradations[:5]:
            print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
            print(f"    P2: [{r['p2_top1']}] {r['p2_top1_label']} (correct)")
            print(f"    P3: [{r['p3_top1']}] {r['p3_top1_label']} (wrong)")
            print(f"    Rules: R5={r['rule5']:+.1f} R6={r['rule6']:+.1f} R7={r['rule7']:+.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 Evaluation: Domain-Specific Rules")

    # Phase 2 settings
    parser.add_argument("--backend", choices=["rapidfuzz", "elasticsearch"], default="rapidfuzz")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--top-k", type=int, default=10, help="Phase 2 candidates to retrieve")
    parser.add_argument("--wikidata", action="store_true", help="Use Wikidata enrichment")
    parser.add_argument("--dbpedia", action="store_true", help="Use DBpedia enrichment")
    parser.add_argument("--umls", type=str, default=None, help="Path to MRCONSO.RRF")

    # Rule weight tuning
    parser.add_argument("--rule5-boost", type=float, default=5.0, help="Multi-KG boost per source")
    parser.add_argument("--rule6-penalty", type=float, default=-30.0, help="Semantic type mismatch penalty")
    parser.add_argument("--rule7-boost", type=float, default=10.0, help="Definition overlap boost per keyword")

    # Evaluation settings
    parser.add_argument("--limit", type=int, default=None, help="Limit to N mentions")

    args = parser.parse_args()
    run_evaluation(args)
