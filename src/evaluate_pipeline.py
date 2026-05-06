"""
Pipeline Evaluation: Full BioLinkerAI on BC5CDR
================================================
End-to-end evaluation that runs Phase 2 → Phase 3 → Phase 4
on the BC5CDR test set with gold entities, showing the
contribution of each phase.

Reports:
  - Phase 2 only (candidate generation) Accuracy@1
  - Phase 2 + 3 (with domain rules) Accuracy@1
  - Phase 2 + 3 + 4 (with LLM) Accuracy@1
  - Per-phase improvement / degradation counts

Usage:
    # Phase 2 + 3 only (no LLM):
    python3 evaluate_pipeline.py --no-phase4

    # Full pipeline with LLM:
    python3 evaluate_pipeline.py --model qwen3-4b-2507

    # With all enrichment:
    python3 evaluate_pipeline.py --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF

    # Quick test:
    python3 evaluate_pipeline.py --limit 100 --no-phase4
"""

import sys
import time
import json
import argparse
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "domain-rules"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "llm-disambiguation"))

from pubtator_parser import parse_pubtator
from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever, _build_id_mapping
from domain_rules import DomainRuleReranker
from llm_disambiguator import LLMDisambiguator


def extract_sentence(text: str, mention: str, start: int = -1, window: int = 200) -> str:
    """Extract a sentence-level context window around the mention."""
    # Find mention position if start not given
    if start < 0:
        pos = text.lower().find(mention.lower())
        if pos < 0:
            return text[:window]
        start = pos

    # Window around mention
    win_start = max(0, start - window // 2)
    win_end = min(len(text), start + len(mention) + window // 2)
    snippet = text[win_start:win_end]

    # Trim to sentence boundaries
    first_period = snippet.find(". ")
    if first_period > 0 and first_period < len(snippet) // 3:
        snippet = snippet[first_period + 2:]
    last_period = snippet.rfind(". ")
    if last_period > len(snippet) * 2 // 3:
        snippet = snippet[:last_period + 1]

    return snippet.strip()


def run_evaluation(args):
    """Run full pipeline evaluation."""

    # ── Step 1: Build MeSH index ──
    print("=" * 60)
    print(f"Building MeSH index (backend={args.backend})...")
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

    # ── Step 2: Load enrichment caches for Phase 3 ──
    cache_dir = PROJECT_ROOT / "src" / "candidate-generation" / "cache"
    wikidata, dbpedia, umls_cache = {}, {}, {}

    for name, label in [("wikidata_cache.json", "Wikidata"),
                        ("dbpedia_cache.json", "DBpedia"),
                        ("umls_cache.json", "UMLS")]:
        path = cache_dir / name
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
            if "wikidata" in name:
                wikidata = data
            elif "dbpedia" in name:
                dbpedia = data
            else:
                umls_cache = data
            print(f"  Loaded {label} cache: {len(data)} entities")

    # ── Step 3: Create Phase 3 reranker ──
    reranker = DomainRuleReranker(
        mesh_index=index,
        rule5_boost=args.rule5_boost,
        rule6_penalty=args.rule6_penalty,
        rule7_boost=args.rule7_boost,
        wikidata_synonyms=wikidata,
        dbpedia_synonyms=dbpedia,
        umls_synonyms=umls_cache,
    )

    # ── Step 4: Create Phase 4 disambiguator (optional) ──
    disambiguator = None
    if not args.no_phase4:
        try:
            disambiguator = LLMDisambiguator(
                model=args.model,
                base_url=args.base_url,
                temperature=args.temperature,
            )
        except Exception as e:
            print(f"Warning: Could not connect to LLM: {e}")
            print("Phase 4 will be skipped.")

    # ── Step 5: Load BC5CDR test set ──
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
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        context_lookup[pmid] = {
            "title": title,
            "abstract": abstract,
            "full_text": f"{title} {abstract}".strip(),
        }

    # Build ID mapping
    id_map = _build_id_mapping(index, args.umls)
    label_to_ids = id_map.pop("__label_to_ids__", {})
    id_map.pop("__label_lookup__", None)

    # Deduplicate and filter
    eval_df = anns[anns["mesh_id"] != "-1"].drop_duplicates(subset=["mention", "mesh_id"])

    if args.limit:
        eval_df = eval_df.head(args.limit)

    n_pairs = len(eval_df)
    print(f"  {n_pairs} unique (mention, mesh_id) pairs to evaluate")

    # ── Step 6: Run evaluation ──
    print("\n" + "=" * 60)
    phases_label = "Phase 2 → 3"
    if disambiguator:
        phases_label += " → 4"
    print(f"Running {phases_label} evaluation...")
    print("=" * 60)

    total = 0
    p2_correct = 0
    p3_correct = 0
    p4_correct = 0

    p3_improved_over_p2 = 0
    p3_degraded_vs_p2 = 0
    p4_improved_over_p3 = 0
    p4_degraded_vs_p3 = 0

    llm_calls = 0
    llm_fallbacks = 0

    changes_log = []
    t0 = time.time()

    for _, row in eval_df.iterrows():
        mention = row["mention"]
        gold_id = row["mesh_id"]
        pmid = row["pmid"]
        entity_type = row.get("entity_type", None)

        # Expand gold IDs
        gold_ids = set(gold_id.split("|"))
        expanded_gold_ids = set(gold_ids)
        for gid in gold_ids:
            if gid in id_map:
                expanded_gold_ids.update(id_map[gid])

        # ── Phase 2: Retrieve candidates ──
        candidates = retriever.retrieve(mention, top_k=args.top_k)
        if not candidates:
            total += 1
            continue

        p2_top1 = candidates[0].mesh_id
        p2_hit = p2_top1 in expanded_gold_ids

        # ── Phase 3: Re-rank with domain rules ──
        # Make copies of scores so Phase 3 doesn't modify Phase 2's view
        p2_scores = {c.mesh_id: c.score for c in candidates}

        reranked = reranker.rerank(
            mention=mention,
            candidates=candidates,
            entity_type=entity_type,
            context=context_lookup.get(pmid, {}).get("full_text", ""),
        )

        p3_top1 = reranked[0].mesh_id
        p3_hit = p3_top1 in expanded_gold_ids

        # ── Phase 4: LLM disambiguation (optional) ──
        p4_top1 = p3_top1
        p4_hit = p3_hit
        confidence = "phase3"

        if disambiguator:
            ctx = context_lookup.get(pmid, {})
            sentence = extract_sentence(
                ctx.get("full_text", ""),
                mention,
            )
            llm_candidates = reranked[:args.llm_top_k]

            llm_result = disambiguator.disambiguate(
                mention=mention,
                candidates=llm_candidates,
                context=sentence,
                title=ctx.get("title", ""),
            )

            p4_top1 = llm_result.mesh_id
            p4_hit = p4_top1 in expanded_gold_ids
            confidence = llm_result.confidence
            llm_calls += 1
            if confidence == "fallback":
                llm_fallbacks += 1

        # ── Track results ──
        if p2_hit:
            p2_correct += 1
        if p3_hit:
            p3_correct += 1
        if p4_hit:
            p4_correct += 1

        if p3_hit and not p2_hit:
            p3_improved_over_p2 += 1
        if p2_hit and not p3_hit:
            p3_degraded_vs_p2 += 1
        if p4_hit and not p3_hit:
            p4_improved_over_p3 += 1
        if p3_hit and not p4_hit:
            p4_degraded_vs_p3 += 1

        # Log interesting changes
        if p2_hit != p3_hit or p3_hit != p4_hit:
            changes_log.append({
                "mention": mention,
                "gold_id": gold_id,
                "entity_type": entity_type,
                "p2_top1": p2_top1,
                "p2_label": candidates[0].preferred_label if candidates else "?",
                "p3_top1": p3_top1,
                "p3_label": reranked[0].preferred_label if reranked else "?",
                "p4_top1": p4_top1,
                "p2_correct": p2_hit,
                "p3_correct": p3_hit,
                "p4_correct": p4_hit,
                "confidence": confidence,
            })

        total += 1
        if total % 200 == 0:
            elapsed = time.time() - t0
            line = (f"  ... {total}/{n_pairs} ({total*100//n_pairs}%) "
                    f"— P2: {p2_correct*100/total:.1f}% "
                    f"P3: {p3_correct*100/total:.1f}%")
            if disambiguator:
                line += f" P4: {p4_correct*100/total:.1f}%"
            print(line, flush=True)

    elapsed = time.time() - t0

    # ── Results ──
    print("\n" + "=" * 60)
    print("RESULTS — Full Pipeline Evaluation")
    print("=" * 60)

    p2_acc = p2_correct / total * 100 if total > 0 else 0
    p3_acc = p3_correct / total * 100 if total > 0 else 0
    p4_acc = p4_correct / total * 100 if total > 0 else 0

    print(f"  Total mentions:              {total}")
    print(f"")
    print(f"  Phase 2 Accuracy@1:          {p2_acc:.1f}% ({p2_correct}/{total})")
    print(f"  Phase 2+3 Accuracy@1:        {p3_acc:.1f}% ({p3_correct}/{total})")
    print(f"    P3 vs P2:                  {p3_acc - p2_acc:+.1f}% "
          f"(+{p3_improved_over_p2} / -{p3_degraded_vs_p2})")

    if disambiguator:
        print(f"  Phase 2+3+4 Accuracy@1:      {p4_acc:.1f}% ({p4_correct}/{total})")
        print(f"    P4 vs P3:                  {p4_acc - p3_acc:+.1f}% "
              f"(+{p4_improved_over_p3} / -{p4_degraded_vs_p3})")
        print(f"    P4 vs P2:                  {p4_acc - p2_acc:+.1f}%")
        print(f"")
        print(f"  LLM calls:                   {llm_calls}")
        print(f"  LLM fallbacks (parse fail):  {llm_fallbacks}")

    print(f"")
    print(f"  Rule weights: R5={args.rule5_boost}, R6={args.rule6_penalty}, R7={args.rule7_boost}")
    print(f"  Time: {elapsed:.0f}s")

    # ── Show example changes ──
    if changes_log:
        improvements_p3 = [r for r in changes_log if r["p3_correct"] and not r["p2_correct"]]
        degradations_p3 = [r for r in changes_log if r["p2_correct"] and not r["p3_correct"]]

        if improvements_p3:
            print(f"\n── Phase 3 IMPROVED over Phase 2 ({len(improvements_p3)} total) ──")
            for r in improvements_p3[:3]:
                print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                print(f"    P2: {r['p2_top1']} (wrong) → P3: {r['p3_top1']} (correct)")

        if degradations_p3:
            print(f"\n── Phase 3 DEGRADED vs Phase 2 ({len(degradations_p3)} total) ──")
            for r in degradations_p3[:3]:
                print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                print(f"    P2: {r['p2_top1']} (correct) → P3: {r['p3_top1']} (wrong)")

        if disambiguator:
            improvements_p4 = [r for r in changes_log if r["p4_correct"] and not r["p3_correct"]]
            degradations_p4 = [r for r in changes_log if r["p3_correct"] and not r["p4_correct"]]

            if improvements_p4:
                print(f"\n── Phase 4 IMPROVED over Phase 3 ({len(improvements_p4)} total) ──")
                for r in improvements_p4[:3]:
                    print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                    print(f"    P3: {r['p3_top1']} (wrong) → P4: {r['p4_top1']} (correct)")

            if degradations_p4:
                print(f"\n── Phase 4 DEGRADED vs Phase 3 ({len(degradations_p4)} total) ──")
                for r in degradations_p4[:3]:
                    print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                    print(f"    P3: {r['p3_top1']} (correct) → P4: {r['p4_top1']} (wrong)")

    # ── Save detailed log ──
    log_path = PROJECT_ROOT / "src" / "pipeline_evaluation_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "total": total,
            "phase2_accuracy": p2_acc,
            "phase3_accuracy": p3_acc,
            "phase4_accuracy": p4_acc if disambiguator else None,
            "changes": changes_log,
        }, f, indent=2)
    print(f"\nDetailed log saved to {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full Pipeline Evaluation on BC5CDR")

    # Phase 2 settings
    parser.add_argument("--backend", choices=["rapidfuzz", "elasticsearch"], default="rapidfuzz")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--top-k", type=int, default=10, help="Phase 2 candidates")
    parser.add_argument("--wikidata", action="store_true", help="Wikidata enrichment")
    parser.add_argument("--dbpedia", action="store_true", help="DBpedia enrichment")
    parser.add_argument("--umls", type=str, default=None, help="Path to MRCONSO.RRF")

    # Phase 3 settings
    parser.add_argument("--rule5-boost", type=float, default=2.0)
    parser.add_argument("--rule6-penalty", type=float, default=-30.0)
    parser.add_argument("--rule7-boost", type=float, default=3.0)

    # Phase 4 settings
    parser.add_argument("--no-phase4", action="store_true", help="Skip Phase 4 (LLM)")
    parser.add_argument("--model", default="qwen3-4b-2507", help="LLM model name")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LLM API URL")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--llm-top-k", type=int, default=5, help="Candidates to pass to LLM")

    # Evaluation settings
    parser.add_argument("--limit", type=int, default=None, help="Limit to N mentions")

    args = parser.parse_args()
    run_evaluation(args)
