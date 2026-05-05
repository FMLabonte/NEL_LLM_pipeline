"""
Phase 4 Evaluation: LLM Disambiguation on BC5CDR
==================================================
End-to-end evaluation that:
  1. Loads the BC5CDR test set
  2. Builds the MeSH index (Phase 2)
  3. For each mention: retrieves candidates → asks the LLM to pick the best one
  4. Computes Accuracy@1 (does the LLM pick the correct MeSH entity?)

Handles MeSH ID version mismatches via UMLS CUI mapping (same as Phase 2 eval).

Usage:
    # Start LMStudio server first, then:
    python3 evaluate.py --model qwen3-4b-2507

    # With all Phase 2 enrichment:
    python3 evaluate.py --model qwen3-4b-2507 --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF

    # Limit to N mentions (for quick testing):
    python3 evaluate.py --model qwen3-4b-2507 --limit 50
"""

import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime

# Add project root and candidate-generation to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))

from pubtator_parser import parse_pubtator
from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever, _build_id_mapping
from llm_disambiguator import LLMDisambiguator


def build_context_lookup(metadata_df, annotations_df):
    """
    Build a lookup from (pmid, mention, start, end) → (title, abstract, sentence).

    Returns a dict mapping pmid → {title, abstract} for context retrieval.
    """
    context = {}
    for _, row in metadata_df.iterrows():
        pmid = row["pmid"]
        title = row.get("title", "")
        abstract = row.get("abstract", "")
        context[pmid] = {
            "title": title,
            "abstract": abstract,
            "full_text": f"{title} {abstract}".strip(),
        }
    return context


def extract_sentence(text: str, mention: str, start: int, end: int, window: int = 200) -> str:
    """
    Extract a sentence-like window around the mention in the text.

    Takes `window` characters before and after the mention, then trims to
    sentence boundaries (periods) if possible.
    """
    # Get window around mention
    win_start = max(0, start - window)
    win_end = min(len(text), end + window)
    excerpt = text[win_start:win_end]

    # Try to trim to sentence boundaries
    # Find last period before mention
    mention_offset = start - win_start
    prefix = excerpt[:mention_offset]
    last_period = prefix.rfind(". ")
    if last_period != -1:
        excerpt = excerpt[last_period + 2:]

    # Find first period after mention
    suffix_start = mention_offset + (end - start)
    suffix = excerpt[suffix_start:]
    first_period = suffix.find(". ")
    if first_period != -1:
        excerpt = excerpt[:suffix_start + first_period + 1]

    return excerpt.strip()


def run_evaluation(args):
    """Run the full Phase 2 + Phase 4 evaluation."""

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

    # ── Step 2: Load BC5CDR test set ──
    print("\n" + "=" * 60)
    print("Loading BC5CDR test set...")
    print("=" * 60)

    meta, anns, rels = parse_pubtator(
        str(PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.PubTator.txt")
    )

    context_lookup = build_context_lookup(meta, anns)

    # Deduplicate and filter
    eval_pairs = (
        anns[anns["mesh_id"] != "-1"]
        .drop_duplicates(subset=["mention", "mesh_id"])
    )

    if args.limit:
        eval_pairs = eval_pairs.head(args.limit)
        print(f"  Limited to {args.limit} mentions for testing")

    n_pairs = len(eval_pairs)
    print(f"  {n_pairs} unique (mention, mesh_id) pairs to evaluate")

    # ── Step 3: Build ID mapping for evaluation ──
    id_map = _build_id_mapping(index, args.umls)
    label_to_ids = id_map.pop("__label_to_ids__", {})
    id_map.pop("__label_lookup__", None)

    # ── Step 4: Connect to LLM ──
    print("\n" + "=" * 60)
    print(f"Phase 4: Connecting to LLM ({args.model})...")
    print("=" * 60)

    disambiguator = LLMDisambiguator(
        model=args.model,
        base_url=args.base_url,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )

    # ── Step 5: Run evaluation ──
    print("\n" + "=" * 60)
    print("Running Phase 2 → Phase 4 evaluation...")
    print("=" * 60)

    # Counters
    total = 0
    phase2_correct = 0     # Phase 2 top-1 is correct
    phase4_correct = 0     # Phase 4 LLM choice is correct
    llm_choices = 0        # LLM made a valid choice (vs fallback)
    fallback_count = 0     # LLM parse failed, fell back to top-1
    phase4_improved = 0    # Phase 4 correct but Phase 2 wrong
    phase4_degraded = 0    # Phase 2 correct but Phase 4 wrong

    results_log = []
    t0 = time.time()

    for _, row in eval_pairs.iterrows():
        mention = row["mention"]
        gold_id = row["mesh_id"]
        pmid = row["pmid"]
        start = int(row["start"]) if "start" in row and not (hasattr(row["start"], '__class__') and row["start"].__class__.__name__ == 'NAType') else 0
        end = int(row["end"]) if "end" in row and not (hasattr(row["end"], '__class__') and row["end"].__class__.__name__ == 'NAType') else 0

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

        # Check Phase 2 top-1
        p2_top1_id = candidates[0].mesh_id
        p2_correct = p2_top1_id in expanded_gold_ids

        # Get context
        ctx = context_lookup.get(pmid, {})
        full_text = ctx.get("full_text", "")
        title = ctx.get("title", "")

        # Extract sentence context around the mention
        sentence = extract_sentence(full_text, mention, start, end)
        if not sentence:
            sentence = full_text[:500]  # fallback to first 500 chars

        # Phase 4: LLM disambiguation
        result = disambiguator.disambiguate(
            mention=mention,
            candidates=candidates[:args.llm_top_k],  # limit candidates for LLM
            context=sentence,
            title=title,
        )

        # Check Phase 4 result
        p4_id = result.mesh_id
        p4_correct = p4_id in expanded_gold_ids

        # Also check if the LLM picked an entity whose other MeSH IDs match
        if not p4_correct:
            for gid in gold_ids:
                if gid not in index.entities:
                    for c in candidates[:args.llm_top_k]:
                        if c.mesh_id == p4_id:
                            c_label = c.preferred_label.lower()
                            if c_label in label_to_ids and gid in id_map and c.mesh_id in id_map[gid]:
                                p4_correct = True

        # Update counters
        if p2_correct:
            phase2_correct += 1
        if p4_correct:
            phase4_correct += 1
        if result.confidence == "llm":
            llm_choices += 1
        else:
            fallback_count += 1
        if p4_correct and not p2_correct:
            phase4_improved += 1
        if p2_correct and not p4_correct:
            phase4_degraded += 1

        total += 1

        # Log interesting cases
        if p2_correct != p4_correct:
            results_log.append({
                "mention": mention,
                "gold_id": gold_id,
                "p2_top1": p2_top1_id,
                "p2_top1_label": candidates[0].preferred_label,
                "p4_choice": p4_id,
                "p4_label": result.preferred_label,
                "p4_rank": result.chosen_rank,
                "p2_correct": p2_correct,
                "p4_correct": p4_correct,
                "raw_response": result.raw_response,
            })

        # Progress
        if total % 50 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            eta = (n_pairs - total) / rate if rate > 0 else 0
            print(f"  ... {total}/{n_pairs} ({total*100//n_pairs}%) "
                  f"— P2: {phase2_correct*100/total:.1f}% "
                  f"P4: {phase4_correct*100/total:.1f}% "
                  f"— {rate:.1f} mentions/s, ETA: {eta/60:.0f}min",
                  flush=True)

    elapsed = time.time() - t0

    # ── Results ──
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    p2_acc = phase2_correct / total * 100 if total > 0 else 0
    p4_acc = phase4_correct / total * 100 if total > 0 else 0

    print(f"  Total mentions:          {total}")
    print(f"  Phase 2 Accuracy@1:      {p2_acc:.1f}% ({phase2_correct}/{total})")
    print(f"  Phase 4 Accuracy@1:      {p4_acc:.1f}% ({phase4_correct}/{total})")
    print(f"  Improvement (P4 vs P2):  {p4_acc - p2_acc:+.1f}%")
    print(f"")
    print(f"  LLM valid choices:       {llm_choices} ({llm_choices*100/total:.1f}%)")
    print(f"  Fallback to top-1:       {fallback_count} ({fallback_count*100/total:.1f}%)")
    print(f"  P4 improved over P2:     {phase4_improved} mentions")
    print(f"  P4 degraded vs P2:       {phase4_degraded} mentions")
    print(f"  Net change:              {phase4_improved - phase4_degraded:+d} mentions")
    print(f"")
    print(f"  Time: {elapsed:.0f}s ({total/elapsed:.1f} mentions/s)")
    print(f"  Model: {args.model}")

    # Save detailed results
    if results_log:
        log_path = Path(__file__).parent / "evaluation_log.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "model": args.model,
                "backend": args.backend,
                "top_k": args.top_k,
                "llm_top_k": args.llm_top_k,
                "total": total,
                "phase2_accuracy": p2_acc,
                "phase4_accuracy": p4_acc,
                "improvements": phase4_improved,
                "degradations": phase4_degraded,
                "changes": results_log,
            }, f, ensure_ascii=False, indent=2)
        print(f"\n  Detailed change log saved to: {log_path}")

    # Show some examples of improvements and degradations
    improvements = [r for r in results_log if r["p4_correct"] and not r["p2_correct"]]
    degradations = [r for r in results_log if r["p2_correct"] and not r["p4_correct"]]

    if improvements:
        print(f"\n── Examples: Phase 4 IMPROVED ({len(improvements)} total) ──")
        for r in improvements[:5]:
            print(f'  "{r["mention"]}" gold={r["gold_id"]}')
            print(f"    P2 top-1: [{r['p2_top1']}] {r['p2_top1_label']} (wrong)")
            print(f"    P4 chose: [{r['p4_choice']}] {r['p4_label']} (correct, rank {r['p4_rank']})")

    if degradations:
        print(f"\n── Examples: Phase 4 DEGRADED ({len(degradations)} total) ──")
        for r in degradations[:5]:
            print(f'  "{r["mention"]}" gold={r["gold_id"]}')
            print(f"    P2 top-1: [{r['p2_top1']}] {r['p2_top1_label']} (correct)")
            print(f"    P4 chose: [{r['p4_choice']}] {r['p4_label']} (wrong, rank {r['p4_rank']})")
            print(f"    LLM raw: '{r['raw_response']}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4 Evaluation: LLM Disambiguation")

    # LLM settings
    parser.add_argument("--model", default="qwen3-4b-2507", help="Model name in LMStudio")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LLM API base URL")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max response tokens")

    # Phase 2 settings
    parser.add_argument("--backend", choices=["rapidfuzz", "elasticsearch"], default="rapidfuzz")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--top-k", type=int, default=10, help="Phase 2: candidates to retrieve")
    parser.add_argument("--llm-top-k", type=int, default=5, help="Phase 4: candidates to show LLM")
    parser.add_argument("--wikidata", action="store_true", help="Enrich with Wikidata")
    parser.add_argument("--dbpedia", action="store_true", help="Enrich with DBpedia")
    parser.add_argument("--umls", type=str, default=None, help="Path to MRCONSO.RRF")

    # Evaluation settings
    parser.add_argument("--limit", type=int, default=None, help="Limit to N mentions (for testing)")

    args = parser.parse_args()
    run_evaluation(args)
