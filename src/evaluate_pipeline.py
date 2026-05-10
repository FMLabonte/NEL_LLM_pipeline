"""
Pipeline Evaluation: Full BioLinkerAI on BC5CDR
================================================
End-to-end evaluation that runs Phase 2 → (2b) → Phase 3 → Phase 4
on the BC5CDR test set with gold entities, showing the
contribution of each phase.

Reports:
  - Phase 2 only (candidate generation) Accuracy@1
  - Phase 2 + 2b (with candidate expansion) Accuracy@1
  - Phase 2 + 2b + 3 (with domain rules) Accuracy@1
  - Phase 2 + 2b + 3 + 4 (with LLM) Accuracy@1
  - Per-phase improvement / degradation counts

Usage:
    # Phase 2 + 3 only (no LLM, no expansion):
    python3 evaluate_pipeline.py --no-phase4 --no-expansion

    # Phase 2 + 2b + 3 (with expansion, no LLM):
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
sys.path.insert(0, str(PROJECT_ROOT / "src" / "improvements"))

from pubtator_parser import parse_pubtator
from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever, _build_id_mapping
from candidate_expander import CandidateExpander
from umls_relation_expander import UMLSRelationExpander
from domain_rules import DomainRuleReranker
from llm_disambiguator import LLMDisambiguator
from abbreviation_expander import AbbreviationExpander

try:
    from embedding_retriever import EmbeddingRetriever
    from hybrid_scorer import HybridScorer
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False


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

    # ── Step 1b: Build Candidate Expander (optional) ──
    expander = None
    if not args.no_expansion:
        umls_bridge = None
        mrconso = args.umls or str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF")
        mrrel = args.mrrel or str(PROJECT_ROOT / "Data" / "UMLS" / "MRREL.RRF")
        if Path(mrrel).exists() and Path(mrconso).exists():
            umls_bridge = UMLSRelationExpander(mrconso, mrrel)
            umls_bridge.build_bridge()
            print(f"  UMLS bridge loaded: {len(umls_bridge.bridge)} entries")
        else:
            print("  UMLS bridge: MRREL/MRCONSO not found, skipping UMLS bridge expansion")

        expander = CandidateExpander(
            index, retriever, umls_bridge=umls_bridge,
        )
        print("  Candidate expander initialized (UMLS bridge + multi-word + parent injection)")
    else:
        print("  Candidate expansion: DISABLED")

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

    # ── Step 4b: Abbreviation Expander (optional) ──
    abbrev_expander = None
    if not args.no_abbreviation_expansion:
        abbrev_expander = AbbreviationExpander()
        print("  Abbreviation expander: ENABLED")
    else:
        print("  Abbreviation expander: DISABLED")

    # ── Step 4c: Embedding Retriever (optional) ──
    emb_retriever = None
    if args.embedding and HAS_EMBEDDING:
        cache_dir = str(PROJECT_ROOT / "src" / "improvements" / "cache" / "faiss")
        emb_retriever = EmbeddingRetriever(
            mesh_index=index,
            model_name=args.embedding_model,
            batch_size=args.embedding_batch_size,
        )
        emb_retriever.build_or_load(cache_dir)
        print(f"  Embedding retriever: ENABLED ({args.embedding_model})")
        print(f"  Hybrid scoring: alpha={args.hybrid_alpha} "
              f"({args.hybrid_alpha*100:.0f}% string + {(1-args.hybrid_alpha)*100:.0f}% embedding)")
    elif args.embedding and not HAS_EMBEDDING:
        print("  Embedding retriever: DISABLED (torch/transformers/faiss not installed)")
        print("    Install with: pip install torch transformers faiss-cpu")
    else:
        print("  Embedding retriever: DISABLED")

    # ── Step 4d: Hybrid Scorer (created alongside embedding retriever) ──
    hybrid_scorer = None
    if emb_retriever is not None:
        hybrid_scorer = HybridScorer(emb_retriever, alpha=args.hybrid_alpha)

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
    phases_label = ""
    if abbrev_expander:
        phases_label += "Abbrev → "
    phases_label += "Phase 2"
    if expander:
        phases_label += " → 2b"
    phases_label += " → 3"
    if disambiguator:
        phases_label += " → 4"
    print(f"Running {phases_label} evaluation...")
    if disambiguator:
        print(f"  LLM: {disambiguator.model}, top_k={args.llm_top_k}")
    print("=" * 60)

    total = 0
    p2_correct = 0
    p2b_correct = 0  # Phase 2 + expansion
    p3_correct = 0
    p4_correct = 0

    # Accuracy@k tracking
    K_VALUES = [1, 5, 10, 20]
    p2_at_k = {k: 0 for k in K_VALUES}   # Phase 2 Accuracy@k
    p2b_at_k = {k: 0 for k in K_VALUES}  # Phase 2+expansion Accuracy@k
    p3_at_k = {k: 0 for k in K_VALUES}   # Phase 2+expansion+Phase 3 Accuracy@k

    p2b_improved_over_p2 = 0
    p2b_degraded_vs_p2 = 0
    p3_improved_over_p2b = 0
    p3_degraded_vs_p2b = 0
    p4_improved_over_p3 = 0
    p4_degraded_vs_p3 = 0

    # Expansion diagnostics
    expansion_added_candidates = 0  # total new candidates added by expansion
    expansion_added_gold = 0        # how often expansion brought the gold into the list
    expansion_mentions_affected = 0 # how many mentions got new candidates

    # Abbreviation expansion diagnostics
    abbrev_expanded_count = 0     # how many mentions were expanded
    abbrev_improved_count = 0     # how many went from wrong to correct thanks to expansion

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

        # ── Pre-Phase 2: Abbreviation Expansion ──
        abbreviation_expanded = None
        if abbrev_expander is not None:
            ctx = context_lookup.get(pmid, {})
            full_text = ctx.get("full_text", "")
            doc_title = ctx.get("title", "")
            expanded = abbrev_expander.expand_mention(
                mention, context=full_text, title=doc_title,
            )
            if expanded:
                abbreviation_expanded = expanded
                abbrev_expanded_count += 1

        # ── Phase 2: Retrieve candidates ──
        candidates = retriever.retrieve(mention, top_k=args.top_k)

        # If abbreviation was expanded, also retrieve for expanded form and merge
        p2_original_top1 = candidates[0].mesh_id if candidates else "NONE"
        if abbreviation_expanded:
            expanded_candidates = retriever.retrieve(
                abbreviation_expanded, top_k=args.top_k,
            )
            if expanded_candidates:
                # Merge both sets, keeping the highest score per mesh_id
                by_id = {}
                for c in candidates:
                    by_id[c.mesh_id] = c
                for ec in expanded_candidates:
                    if ec.mesh_id not in by_id or ec.score > by_id[ec.mesh_id].score:
                        by_id[ec.mesh_id] = ec
                candidates = sorted(by_id.values(), key=lambda c: c.score, reverse=True)
                # Track if expansion changed top-1
                if candidates[0].mesh_id in expanded_gold_ids and p2_original_top1 not in expanded_gold_ids:
                    abbrev_improved_count += 1

        # ── Embedding retrieval (hybrid merge + re-scoring) ──
        if emb_retriever is not None:
            emb_candidates = emb_retriever.retrieve(mention, top_k=args.top_k)
            if emb_candidates:
                # Merge: add embedding candidates not already in the list
                existing_ids = {c.mesh_id for c in candidates}
                for ec in emb_candidates:
                    if ec.mesh_id not in existing_ids:
                        candidates.append(ec)
                        existing_ids.add(ec.mesh_id)

            # Hybrid re-scoring: combine string + embedding scores
            if hybrid_scorer is not None:
                candidates = hybrid_scorer.rescore(mention, candidates)

        if not candidates:
            total += 1
            continue

        p2_top1 = candidates[0].mesh_id
        p2_hit = p2_top1 in expanded_gold_ids

        # Accuracy@k for Phase 2
        p2_ids = [c.mesh_id for c in candidates]
        for k in K_VALUES:
            if any(mid in expanded_gold_ids for mid in p2_ids[:k]):
                p2_at_k[k] += 1

        # ── Phase 2b: Candidate Expansion ──
        p2_candidate_ids = {c.mesh_id for c in candidates}
        if expander is not None:
            # expansion_top_k must be at least as large as the input list
            # so we never shrink the candidate list, only grow it
            exp_top_k = max(args.expansion_top_k, len(candidates) + 10)
            candidates = expander.expand(
                mention=mention,
                candidates=candidates,
                entity_type=entity_type,
                top_k=exp_top_k,
            )
            # Diagnostics: how many new candidates were added?
            new_ids = {c.mesh_id for c in candidates} - p2_candidate_ids
            if new_ids:
                expansion_mentions_affected += 1
                expansion_added_candidates += len(new_ids)
                # Did expansion bring the gold into the list?
                if new_ids & expanded_gold_ids:
                    expansion_added_gold += 1

        p2b_top1 = candidates[0].mesh_id
        p2b_hit = p2b_top1 in expanded_gold_ids

        # Accuracy@k for Phase 2b
        p2b_ids = [c.mesh_id for c in candidates]
        for k in K_VALUES:
            if any(mid in expanded_gold_ids for mid in p2b_ids[:k]):
                p2b_at_k[k] += 1

        # ── Phase 3: Re-rank with domain rules ──
        reranked = reranker.rerank(
            mention=mention,
            candidates=candidates,
            entity_type=entity_type,
            context=context_lookup.get(pmid, {}).get("full_text", ""),
        )

        p3_top1 = reranked[0].mesh_id
        p3_hit = p3_top1 in expanded_gold_ids

        # Accuracy@k for Phase 3
        p3_ids = [c.mesh_id for c in reranked]
        for k in K_VALUES:
            if any(mid in expanded_gold_ids for mid in p3_ids[:k]):
                p3_at_k[k] += 1

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
        if p2b_hit:
            p2b_correct += 1
        if p3_hit:
            p3_correct += 1
        if p4_hit:
            p4_correct += 1

        if p2b_hit and not p2_hit:
            p2b_improved_over_p2 += 1
        if p2_hit and not p2b_hit:
            p2b_degraded_vs_p2 += 1
        if p3_hit and not p2b_hit:
            p3_improved_over_p2b += 1
        if p2b_hit and not p3_hit:
            p3_degraded_vs_p2b += 1
        if p4_hit and not p3_hit:
            p4_improved_over_p3 += 1
        if p3_hit and not p4_hit:
            p4_degraded_vs_p3 += 1

        # Log interesting changes
        if p2_hit != p2b_hit or p2b_hit != p3_hit or p3_hit != p4_hit:
            changes_log.append({
                "mention": mention,
                "gold_id": gold_id,
                "entity_type": entity_type,
                "p2_top1": p2_top1,
                "p2b_top1": p2b_top1,
                "p3_top1": p3_top1,
                "p3_label": reranked[0].preferred_label if reranked else "?",
                "p4_top1": p4_top1,
                "p2_correct": p2_hit,
                "p2b_correct": p2b_hit,
                "p3_correct": p3_hit,
                "p4_correct": p4_hit,
                "confidence": confidence,
            })

        total += 1
        # Print progress: every 50 with LLM, every 200 without
        progress_interval = 50 if disambiguator else 200
        if total % progress_interval == 0:
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            eta = (n_pairs - total) / rate if rate > 0 else 0
            line = (f"  ... {total}/{n_pairs} ({total*100//n_pairs}%) "
                    f"— P2: {p2_correct*100/total:.1f}%")
            if expander is not None:
                line += f" P2b: {p2b_correct*100/total:.1f}%"
            line += f" P3: {p3_correct*100/total:.1f}%"
            if disambiguator:
                line += f" P4: {p4_correct*100/total:.1f}%"
            line += f" [{elapsed:.0f}s, ETA {eta:.0f}s]"
            print(line, flush=True)

    elapsed = time.time() - t0

    # ── Results ──
    print("\n" + "=" * 60)
    print("RESULTS — Full Pipeline Evaluation")
    print("=" * 60)

    p2_acc = p2_correct / total * 100 if total > 0 else 0
    p2b_acc = p2b_correct / total * 100 if total > 0 else 0
    p3_acc = p3_correct / total * 100 if total > 0 else 0
    p4_acc = p4_correct / total * 100 if total > 0 else 0

    print(f"  Total mentions:              {total}")
    print(f"")

    # ── Accuracy@1 per phase ──
    print(f"  Phase 2 Accuracy@1:          {p2_acc:.1f}% ({p2_correct}/{total})")

    if expander is not None:
        print(f"  Phase 2+2b Accuracy@1:       {p2b_acc:.1f}% ({p2b_correct}/{total})")
        print(f"    P2b vs P2:                 {p2b_acc - p2_acc:+.1f}% "
              f"(+{p2b_improved_over_p2} / -{p2b_degraded_vs_p2})")

    print(f"  Phase 2(+2b)+3 Accuracy@1:   {p3_acc:.1f}% ({p3_correct}/{total})")
    print(f"    P3 vs P2b:                 {p3_acc - p2b_acc:+.1f}% "
          f"(+{p3_improved_over_p2b} / -{p3_degraded_vs_p2b})")

    if disambiguator:
        print(f"  Phase 2+2b+3+4 Accuracy@1:   {p4_acc:.1f}% ({p4_correct}/{total})")
        print(f"    P4 vs P3:                  {p4_acc - p3_acc:+.1f}% "
              f"(+{p4_improved_over_p3} / -{p4_degraded_vs_p3})")
        print(f"    P4 vs P2:                  {p4_acc - p2_acc:+.1f}%")
        print(f"")
        print(f"  LLM calls:                   {llm_calls}")
        print(f"  LLM fallbacks (parse fail):  {llm_fallbacks}")

    # ── Accuracy@k table ──
    print(f"\n{'─' * 60}")
    print(f"  Accuracy@k:")
    print(f"  {'k':>3}  {'Phase 2':>12}  {'Phase 2+2b':>12}  {'Phase 2b+3':>12}")
    for k in K_VALUES:
        p2_k = p2_at_k[k] / total * 100 if total > 0 else 0
        p2b_k = p2b_at_k[k] / total * 100 if total > 0 else 0
        p3_k = p3_at_k[k] / total * 100 if total > 0 else 0
        diff_str = f"(+{p2b_k - p2_k:.1f})" if p2b_k > p2_k else ""
        print(f"  {k:>3}  {p2_k:>11.1f}%  {p2b_k:>11.1f}%  {p3_k:>11.1f}%  {diff_str}")

    # ── Expansion diagnostics ──
    if expander is not None:
        print(f"\n{'─' * 60}")
        print(f"  Expansion diagnostics:")
        print(f"    Mentions with new candidates:  {expansion_mentions_affected}/{total} "
              f"({expansion_mentions_affected*100/total:.1f}%)")
        print(f"    Total new candidates added:    {expansion_added_candidates}")
        if expansion_mentions_affected > 0:
            print(f"    Avg new candidates per mention:{expansion_added_candidates/expansion_mentions_affected:.1f}")
        print(f"    Gold brought into list:        {expansion_added_gold} "
              f"({expansion_added_gold*100/total:.2f}%)")

    # ── Abbreviation expansion diagnostics ──
    if abbrev_expander is not None:
        print(f"\n{'─' * 60}")
        print(f"  Abbreviation expansion diagnostics:")
        print(f"    Mentions expanded:             {abbrev_expanded_count}/{total} "
              f"({abbrev_expanded_count*100/total:.1f}%)")
        print(f"    Directly improved top-1:       {abbrev_improved_count}")

    print(f"\n{'─' * 60}")
    expansion_str = "ON" if expander else "OFF"
    print(f"  Candidate expansion:         {expansion_str}")
    if expander:
        print(f"  Expansion top_k:             {args.expansion_top_k}")
    abbrev_str = "ON" if abbrev_expander else "OFF"
    print(f"  Abbreviation expansion:      {abbrev_str}")
    emb_str = f"ON ({args.embedding_model})" if emb_retriever else "OFF"
    print(f"  Embedding retrieval:         {emb_str}")
    print(f"  Rule weights: R5={args.rule5_boost}, R6={args.rule6_penalty}, R7={args.rule7_boost}")
    print(f"  Time: {elapsed:.0f}s")

    # ── Show example changes ──
    if changes_log:
        # Phase 2b vs Phase 2
        if expander is not None:
            improvements_p2b = [r for r in changes_log if r["p2b_correct"] and not r["p2_correct"]]
            degradations_p2b = [r for r in changes_log if r["p2_correct"] and not r["p2b_correct"]]

            if improvements_p2b:
                print(f"\n── Expansion IMPROVED over Phase 2 ({len(improvements_p2b)} total) ──")
                for r in improvements_p2b[:5]:
                    print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                    print(f"    P2: {r['p2_top1']} → P2b: {r['p2b_top1']} (correct)")

            if degradations_p2b:
                print(f"\n── Expansion DEGRADED vs Phase 2 ({len(degradations_p2b)} total) ──")
                for r in degradations_p2b[:5]:
                    print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                    print(f"    P2: {r['p2_top1']} (correct) → P2b: {r['p2b_top1']} (wrong)")

        # Phase 3 vs Phase 2b
        improvements_p3 = [r for r in changes_log if r["p3_correct"] and not r["p2b_correct"]]
        degradations_p3 = [r for r in changes_log if r["p2b_correct"] and not r["p3_correct"]]

        if improvements_p3:
            print(f"\n── Phase 3 IMPROVED over Phase 2b ({len(improvements_p3)} total) ──")
            for r in improvements_p3[:3]:
                print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                print(f"    P2b: {r['p2b_top1']} (wrong) → P3: {r['p3_top1']} (correct)")

        if degradations_p3:
            print(f"\n── Phase 3 DEGRADED vs Phase 2b ({len(degradations_p3)} total) ──")
            for r in degradations_p3[:3]:
                print(f'  "{r["mention"]}" [{r["entity_type"]}] gold={r["gold_id"]}')
                print(f"    P2b: {r['p2b_top1']} (correct) → P3: {r['p3_top1']} (wrong)")

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
            "phase2b_accuracy": p2b_acc if expander else None,
            "phase3_accuracy": p3_acc,
            "phase4_accuracy": p4_acc if disambiguator else None,
            "accuracy_at_k": {
                f"phase2@{k}": p2_at_k[k] / total * 100 if total else 0 for k in K_VALUES
            } | {
                f"phase2b@{k}": p2b_at_k[k] / total * 100 if total else 0 for k in K_VALUES
            } | {
                f"phase3@{k}": p3_at_k[k] / total * 100 if total else 0 for k in K_VALUES
            },
            "expansion_enabled": expander is not None,
            "expansion_diagnostics": {
                "mentions_affected": expansion_mentions_affected,
                "candidates_added": expansion_added_candidates,
                "gold_brought_in": expansion_added_gold,
            } if expander else None,
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

    # Phase 2b: Expansion settings
    parser.add_argument("--no-expansion", action="store_true", help="Skip candidate expansion (UMLS bridge, multi-word, parent)")
    parser.add_argument("--mrrel", type=str, default=None, help="Path to MRREL.RRF for UMLS bridge expansion")
    parser.add_argument("--expansion-top-k", type=int, default=30, help="Max candidates after expansion (default: 30)")

    # Abbreviation expansion
    parser.add_argument("--no-abbreviation-expansion", action="store_true", help="Skip abbreviation expansion")

    # Embedding retrieval
    parser.add_argument("--embedding", action="store_true", help="Enable embedding-based retrieval (SapBERT + FAISS)")
    parser.add_argument("--embedding-model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                        help="HuggingFace model for embedding retrieval")
    parser.add_argument("--embedding-batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--hybrid-alpha", type=float, default=0.7,
                        help="Hybrid scoring weight: 0=pure embedding, 1=pure string (default: 0.7)")

    # Phase 3 settings
    parser.add_argument("--rule5-boost", type=float, default=2.0)
    parser.add_argument("--rule6-penalty", type=float, default=-30.0)
    parser.add_argument("--rule7-boost", type=float, default=3.0)

    # Phase 4 settings
    parser.add_argument("--no-phase4", action="store_true", help="Skip Phase 4 (LLM)")
    parser.add_argument("--model", default="qwen3.5-9b", help="LLM model name")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LLM API URL")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--llm-top-k", type=int, default=10, help="Candidates to pass to LLM")

    # Evaluation settings
    parser.add_argument("--limit", type=int, default=None, help="Limit to N mentions")

    args = parser.parse_args()
    run_evaluation(args)
