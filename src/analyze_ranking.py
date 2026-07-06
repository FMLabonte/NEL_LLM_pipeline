"""
Ranking Distribution Analysis
===============================
Analyzes WHERE the gold candidate sits in the ranking at each phase.

Key questions:
  1. Gold rank distribution: At Phase 2/2b/3, what rank is the gold candidate?
  2. Phase 3 movement: Does Phase 3 push gold UP or DOWN?
  3. Score gap analysis: How far is gold from top-1 when it's not #1?
  4. Entity type breakdown: Which types have worst ranking?
  5. "Fixable" analysis: How many could an LLM theoretically fix?

Usage:
    python3 src/analyze_ranking.py --dataset medmentions --limit 1000
    python3 src/analyze_ranking.py --dataset bc5cdr
"""

import sys
import time
import json
import argparse
from pathlib import Path
from collections import defaultdict

# ── Path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "domain-rules"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "improvements"))

from pubtator_parser import parse_pubtator
from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever, _build_id_mapping
from cui_mesh_mapper import CUIToMeSHMapper
from candidate_expander import CandidateExpander
from umls_relation_expander import UMLSRelationExpander
from domain_rules import DomainRuleReranker
from abbreviation_expander import AbbreviationExpander
from string_normalizer import generate_variants

try:
    from embedding_retriever import EmbeddingRetriever
    from hybrid_scorer import HybridScorer
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False

try:
    from llm_abbreviation_expander import LLMAbbreviationExpander
    HAS_LLM_ABBREV = True
except ImportError:
    HAS_LLM_ABBREV = False

try:
    from document_topic_scorer import DocumentTopicScorer
    HAS_TOPIC = True
except ImportError:
    HAS_TOPIC = False


def find_gold_rank(candidates, expanded_gold_ids):
    """Find the rank (1-based) of the gold candidate. Returns None if not found."""
    for i, c in enumerate(candidates, 1):
        if c.mesh_id in expanded_gold_ids:
            return i
    return None


def run_analysis(args):
    """Run ranking distribution analysis."""

    # ── Build index (same setup as evaluate_pipeline.py) ──
    print(f"Building MeSH index ({args.backend})...")
    index = MeSHIndex(backend=args.backend)
    index.build_from_xml(
        descriptor_path=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
        supplementary_path=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
        enrich_wikidata=False,
        enrich_dbpedia=False,
        enrich_umls=args.umls,
    )

    retriever = CandidateRetriever(index, top_k=args.top_k)

    # ── Candidate Expander ──
    expander = None
    if not args.no_expansion:
        umls_bridge = None
        mrconso = args.umls or str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF")
        mrrel = args.mrrel or str(PROJECT_ROOT / "Data" / "UMLS" / "MRREL.RRF")
        if Path(mrrel).exists() and Path(mrconso).exists():
            umls_bridge = UMLSRelationExpander(mrconso, mrrel)
            umls_bridge.build_bridge()
        expander = CandidateExpander(index, retriever, umls_bridge=umls_bridge)

    # ── Enrichment caches ──
    cache_dir = PROJECT_ROOT / "src" / "candidate-generation" / "cache"
    wikidata, dbpedia, umls_cache = {}, {}, {}
    for name in ["wikidata_cache.json", "dbpedia_cache.json", "umls_cache.json"]:
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

    # ── Phase 3 Reranker ──
    reranker = DomainRuleReranker(
        mesh_index=index,
        wikidata_synonyms=wikidata,
        dbpedia_synonyms=dbpedia,
        umls_synonyms=umls_cache,
    )

    # ── Abbreviation Expander ──
    abbrev_expander = AbbreviationExpander()

    # ── LLM Abbreviation Expander ──
    llm_abbrev_expander = None
    if args.llm_abbreviation and HAS_LLM_ABBREV:
        try:
            llm_abbrev_expander = LLMAbbreviationExpander(
                model=args.llm_abbreviation_model or "qwen3.5-9b",
                base_url=args.local_base_url,
                temperature=0.3,
            )
        except Exception:
            pass

    # ── Embedding Retriever ──
    emb_retriever = None
    hybrid_scorer = None
    if args.embedding and HAS_EMBEDDING:
        cache_dir_emb = str(PROJECT_ROOT / "src" / "improvements" / "cache" / "faiss")
        emb_retriever = EmbeddingRetriever(
            mesh_index=index,
            model_name="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        )
        emb_retriever.build_or_load(cache_dir_emb)
        hybrid_scorer = HybridScorer(emb_retriever, alpha=0.7)

    # ── Topic Scorer ──
    topic_scorer = None
    if HAS_TOPIC and not args.no_topic_scoring:
        topic_scorer = DocumentTopicScorer()

    # ── Load dataset ──
    dataset_name = args.dataset.upper()
    print(f"Loading {dataset_name}...")

    if args.dataset == "bc5cdr":
        data_path = str(PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.PubTator.txt")
    elif args.dataset == "medmentions":
        data_path = str(PROJECT_ROOT / "Data" / "MedMention" / "MedMentions_st21pv_pubtator.txt")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    meta, anns, rels = parse_pubtator(data_path)

    # Build context lookup
    context_lookup = {}
    for _, row in meta.iterrows():
        pmid = row["pmid"]
        context_lookup[pmid] = {
            "title": row.get("title", ""),
            "abstract": row.get("abstract", ""),
            "full_text": f"{row.get('title', '')} {row.get('abstract', '')}".strip(),
        }

    # Build ID mapping
    id_map = _build_id_mapping(index, args.umls)
    label_to_ids = id_map.pop("__label_to_ids__", {})
    id_map.pop("__label_lookup__", None)

    # Prepare eval dataframe
    eval_df = anns.copy()

    if args.dataset == "medmentions":
        eval_df["original_cui"] = eval_df["mesh_id"]
        eval_df["cui"] = eval_df["mesh_id"].str.replace("UMLS:", "", regex=False)
        mrconso_path = args.umls or str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF")
        cui_mapper = CUIToMeSHMapper(mrconso_path)
        cui_mapper.load()
        eval_df["has_mesh"] = eval_df["cui"].apply(lambda c: cui_mapper.is_mappable(c))
        eval_df = eval_df[eval_df["has_mesh"]].copy()

        def cui_to_mesh_str(cui):
            mesh_ids = cui_mapper.cui_to_mesh(cui)
            return "|".join(sorted(mesh_ids)) if mesh_ids else cui
        eval_df["mesh_id"] = eval_df["cui"].apply(cui_to_mesh_str)
        eval_df = eval_df.drop(columns=["has_mesh"])

    eval_df = eval_df[eval_df["mesh_id"] != "-1"].drop_duplicates(subset=["mention", "mesh_id"])
    if args.limit:
        eval_df = eval_df.head(args.limit)

    n_pairs = len(eval_df)
    print(f"  Analyzing {n_pairs} mention-entity pairs")

    # ── Tracking structures ──
    # Gold rank at each phase (None = not in list)
    p2_gold_ranks = []    # Phase 2
    p2b_gold_ranks = []   # Phase 2+expansion
    p3_gold_ranks = []    # Phase 3

    # Phase 3 movement tracking
    p3_movements = []     # (before_rank, after_rank, mention, entity_type)

    # Score gap: top-1 score minus gold score (when gold != top-1)
    p3_score_gaps = []    # (gap, gold_rank, mention, entity_type)

    # Per entity type
    type_ranks_p2 = defaultdict(list)
    type_ranks_p3 = defaultdict(list)

    # Detailed cases for inspection
    p3_degraded_cases = []  # Phase 3 pushed gold DOWN
    p3_improved_cases = []  # Phase 3 pushed gold UP
    not_in_list = []        # Gold not in candidate list at all

    t0 = time.time()

    for idx, (_, row) in enumerate(eval_df.iterrows()):
        mention = row["mention"]
        gold_id = row["mesh_id"]
        pmid = row["pmid"]
        entity_type = row.get("entity_type", None)

        gold_ids = set(gold_id.split("|"))
        expanded_gold_ids = set(gold_ids)
        for gid in gold_ids:
            if gid in id_map:
                expanded_gold_ids.update(id_map[gid])

        # ── Abbreviation Expansion ──
        abbreviation_expanded = None
        ctx = context_lookup.get(pmid, {})
        full_text = ctx.get("full_text", "")

        expanded = abbrev_expander.expand_mention(
            mention, context=full_text, title=ctx.get("title", ""),
        )
        if expanded:
            abbreviation_expanded = expanded

        if abbreviation_expanded is None and llm_abbrev_expander is not None:
            if abbrev_expander._is_likely_abbreviation(mention):
                llm_expanded = llm_abbrev_expander.expand(
                    mention, context=full_text, title=ctx.get("title", ""),
                )
                if llm_expanded:
                    abbreviation_expanded = llm_expanded

        # ── Phase 2 ──
        candidates = retriever.retrieve(mention, top_k=args.top_k)
        variants = generate_variants(mention)
        existing_ids = {c.mesh_id for c in candidates}
        for variant in variants[1:]:
            var_candidates = retriever.retrieve(variant, top_k=args.top_k)
            for vc in var_candidates:
                if vc.mesh_id not in existing_ids:
                    candidates.append(vc)
                    existing_ids.add(vc.mesh_id)

        if abbreviation_expanded:
            expanded_candidates = retriever.retrieve(abbreviation_expanded, top_k=args.top_k)
            if expanded_candidates:
                by_id = {}
                for c in candidates:
                    by_id[c.mesh_id] = c
                for ec in expanded_candidates:
                    if ec.mesh_id not in by_id or ec.score > by_id[ec.mesh_id].score:
                        by_id[ec.mesh_id] = ec
                candidates = sorted(by_id.values(), key=lambda c: c.score, reverse=True)

        # Embedding retrieval
        if emb_retriever is not None:
            emb_candidates = emb_retriever.retrieve(mention, top_k=args.top_k)
            if emb_candidates:
                existing_ids = {c.mesh_id for c in candidates}
                for ec in emb_candidates:
                    if ec.mesh_id not in existing_ids:
                        candidates.append(ec)
                        existing_ids.add(ec.mesh_id)
            if hybrid_scorer is not None:
                candidates = hybrid_scorer.rescore(mention, candidates)

        if not candidates:
            p2_gold_ranks.append(None)
            p2b_gold_ranks.append(None)
            p3_gold_ranks.append(None)
            not_in_list.append({"mention": mention, "gold_id": gold_id, "entity_type": entity_type})
            continue

        # Phase 2 gold rank
        p2_rank = find_gold_rank(candidates, expanded_gold_ids)
        p2_gold_ranks.append(p2_rank)
        et_key = entity_type or "Unknown"
        type_ranks_p2[et_key].append(p2_rank)

        # ── Phase 2b: Expansion ──
        if expander is not None:
            exp_top_k = max(args.expansion_top_k, len(candidates) + 10)
            candidates = expander.expand(
                mention=mention,
                candidates=candidates,
                entity_type=entity_type,
                top_k=exp_top_k,
            )

        p2b_rank = find_gold_rank(candidates, expanded_gold_ids)
        p2b_gold_ranks.append(p2b_rank)

        # ── Phase 3: Rerank ──
        doc_text = context_lookup.get(pmid, {}).get("full_text", "")
        reranked = reranker.rerank(
            mention=mention,
            candidates=candidates,
            entity_type=entity_type,
            context=doc_text,
        )

        if topic_scorer is not None:
            topic = topic_scorer.detect_topic(doc_text)
            reranked = topic_scorer.rescore(reranked, topic)

        p3_rank = find_gold_rank(reranked, expanded_gold_ids)
        p3_gold_ranks.append(p3_rank)
        type_ranks_p3[et_key].append(p3_rank)

        # Track Phase 3 movement
        if p2b_rank is not None and p3_rank is not None:
            movement = p2b_rank - p3_rank  # positive = moved UP (good)
            p3_movements.append((p2b_rank, p3_rank, mention, et_key, gold_id))

            if p3_rank > p2b_rank:  # pushed DOWN
                p3_degraded_cases.append({
                    "mention": mention,
                    "entity_type": et_key,
                    "gold_id": gold_id,
                    "before_rank": p2b_rank,
                    "after_rank": p3_rank,
                    "top1_label": reranked[0].preferred_label,
                    "top1_id": reranked[0].mesh_id,
                })
            elif p3_rank < p2b_rank:  # pushed UP
                p3_improved_cases.append({
                    "mention": mention,
                    "entity_type": et_key,
                    "gold_id": gold_id,
                    "before_rank": p2b_rank,
                    "after_rank": p3_rank,
                })

        if p3_rank is not None and p3_rank > 1:
            gap = reranked[0].score - reranked[p3_rank - 1].score
            p3_score_gaps.append((gap, p3_rank, mention, et_key))

        if p2b_rank is None and p3_rank is None:
            not_in_list.append({
                "mention": mention,
                "gold_id": gold_id,
                "entity_type": et_key,
                "n_candidates": len(candidates),
            })

        # Progress
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - t0
            print(f"  ... {idx+1}/{n_pairs} ({(idx+1)*100//n_pairs}%) [{elapsed:.0f}s]", flush=True)

    elapsed = time.time() - t0

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("=" * 70)
    print(f"RANKING DISTRIBUTION ANALYSIS — {dataset_name}")
    print("=" * 70)

    # ── 1. Gold Rank Distribution ──
    print()
    print("1. GOLD CANDIDATE RANK DISTRIBUTION")
    print("-" * 70)

    for phase_name, ranks in [("Phase 2", p2_gold_ranks),
                               ("Phase 2b", p2b_gold_ranks),
                               ("Phase 3", p3_gold_ranks)]:
        total = len(ranks)
        not_found = sum(1 for r in ranks if r is None)
        found = [r for r in ranks if r is not None]

        print(f"\n  {phase_name} (n={total}):")
        print(f"    Not in list:  {not_found} ({not_found*100/total:.1f}%)")

        # Rank histogram
        rank_counts = defaultdict(int)
        for r in found:
            if r <= 10:
                rank_counts[r] += 1
            elif r <= 20:
                rank_counts["11-20"] += 1
            elif r <= 30:
                rank_counts["21-30"] += 1
            else:
                rank_counts["31+"] += 1

        print(f"    Rank distribution:")
        cumulative = 0
        for rank in range(1, 11):
            count = rank_counts.get(rank, 0)
            cumulative += count
            bar = "#" * (count * 40 // total)
            pct = count * 100 / total
            cum_pct = cumulative * 100 / total
            print(f"      Rank {rank:>2}: {count:>5} ({pct:>5.1f}%) cum={cum_pct:>5.1f}%  {bar}")

        for bucket in ["11-20", "21-30", "31+"]:
            count = rank_counts.get(bucket, 0)
            cumulative += count
            pct = count * 100 / total
            cum_pct = cumulative * 100 / total
            print(f"      {bucket:>6}: {count:>5} ({pct:>5.1f}%) cum={cum_pct:>5.1f}%")

    # ── 2. Phase 3 Movement ──
    print()
    print("2. PHASE 3 RANKING MOVEMENT (Phase 2b → Phase 3)")
    print("-" * 70)

    moved_up = [(b, a) for b, a, _, _, _ in p3_movements if a < b]
    moved_down = [(b, a) for b, a, _, _, _ in p3_movements if a > b]
    stayed = [(b, a) for b, a, _, _, _ in p3_movements if a == b]

    print(f"  Total with gold in list: {len(p3_movements)}")
    print(f"  Moved UP (good):    {len(moved_up):>5} ({len(moved_up)*100/len(p3_movements):.1f}%)")
    print(f"  Stayed same:        {len(stayed):>5} ({len(stayed)*100/len(p3_movements):.1f}%)")
    print(f"  Moved DOWN (bad):   {len(moved_down):>5} ({len(moved_down)*100/len(p3_movements):.1f}%)")

    if moved_up:
        avg_up = sum(b - a for b, a in moved_up) / len(moved_up)
        print(f"  Avg UP movement:    {avg_up:.1f} ranks")
    if moved_down:
        avg_down = sum(a - b for b, a in moved_down) / len(moved_down)
        print(f"  Avg DOWN movement:  {avg_down:.1f} ranks")

    # Movement from rank 1
    was_rank1 = [(b, a, m, et, g) for b, a, m, et, g in p3_movements if b == 1]
    lost_rank1 = [(b, a, m, et, g) for b, a, m, et, g in was_rank1 if a > 1]
    print(f"\n  Gold was Rank 1 at Phase 2b: {len(was_rank1)}")
    print(f"  Phase 3 pushed it AWAY from #1: {len(lost_rank1)} ({len(lost_rank1)*100/max(len(was_rank1),1):.1f}%)")

    if lost_rank1:
        print(f"  Examples of Phase 3 pushing gold from #1:")
        for b, a, mention, et, gold in lost_rank1[:10]:
            print(f"    \"{mention}\" [{et}] gold={gold}: rank 1 → {a}")

    # Movement TO rank 1
    gained_rank1 = [(b, a, m, et, g) for b, a, m, et, g in p3_movements if b > 1 and a == 1]
    print(f"\n  Phase 3 promoted gold TO #1: {len(gained_rank1)}")
    if gained_rank1:
        print(f"  Examples:")
        for b, a, mention, et, gold in gained_rank1[:10]:
            print(f"    \"{mention}\" [{et}] gold={gold}: rank {b} → 1")

    # ── 3. Score Gap Analysis ──
    print()
    print("3. SCORE GAP: Top-1 vs Gold (when gold is NOT #1, Phase 3)")
    print("-" * 70)

    if p3_score_gaps:
        gaps_by_rank = defaultdict(list)
        for gap, rank, mention, et in p3_score_gaps:
            gaps_by_rank[rank].append(gap)

        print(f"  {'Gold Rank':>10}  {'Count':>6}  {'Avg Gap':>8}  {'Median':>8}  {'Min':>8}  {'Max':>8}")
        for rank in sorted(gaps_by_rank.keys()):
            if rank > 10:
                break
            gaps = sorted(gaps_by_rank[rank])
            avg = sum(gaps) / len(gaps)
            median = gaps[len(gaps) // 2]
            print(f"  {rank:>10}  {len(gaps):>6}  {avg:>8.1f}  {median:>8.1f}  {gaps[0]:>8.1f}  {gaps[-1]:>8.1f}")

        # How many have small gaps (LLM could reasonably flip)
        small_gap = sum(1 for g, r, _, _ in p3_score_gaps if g < 10)
        medium_gap = sum(1 for g, r, _, _ in p3_score_gaps if 10 <= g < 30)
        large_gap = sum(1 for g, r, _, _ in p3_score_gaps if g >= 30)
        print(f"\n  Gap categories (gold NOT #1):")
        print(f"    Small gap (<10):   {small_gap:>5} — LLM should be able to flip these")
        print(f"    Medium gap (10-30):{medium_gap:>5} — harder for LLM")
        print(f"    Large gap (≥30):   {large_gap:>5} — unlikely LLM can fix")

    # ── 4. Entity Type Breakdown ──
    print()
    print("4. RANKING BY ENTITY TYPE (Phase 3)")
    print("-" * 70)

    print(f"  {'Type':>8}  {'Total':>6}  {'Rank 1':>7}  {'Rank 2-5':>9}  {'Rank 6-10':>9}  {'Not in':>7}  {'@1%':>6}  {'@5%':>6}  {'@10%':>6}")
    for et in sorted(type_ranks_p3.keys(), key=lambda x: len(type_ranks_p3[x]), reverse=True):
        ranks = type_ranks_p3[et]
        total = len(ranks)
        rank1 = sum(1 for r in ranks if r == 1)
        rank2_5 = sum(1 for r in ranks if r is not None and 2 <= r <= 5)
        rank6_10 = sum(1 for r in ranks if r is not None and 6 <= r <= 10)
        not_found = sum(1 for r in ranks if r is None)
        at1 = rank1 * 100 / total
        at5 = (rank1 + rank2_5) * 100 / total
        at10 = (rank1 + rank2_5 + rank6_10) * 100 / total
        print(f"  {et:>8}  {total:>6}  {rank1:>7}  {rank2_5:>9}  {rank6_10:>9}  {not_found:>7}  {at1:>5.1f}%  {at5:>5.1f}%  {at10:>5.1f}%")

    # ── 5. "Fixable" Analysis ──
    print()
    print("5. LLM OPPORTUNITY ANALYSIS (Phase 3 → what could Phase 4 fix?)")
    print("-" * 70)

    p3_wrong = [r for r in p3_gold_ranks if r is not None and r > 1]
    p3_right = [r for r in p3_gold_ranks if r == 1]
    p3_missing = [r for r in p3_gold_ranks if r is None]

    print(f"  Phase 3 correct (rank 1):       {len(p3_right):>5} ({len(p3_right)*100/n_pairs:.1f}%)")
    print(f"  Phase 3 wrong but gold in list:  {len(p3_wrong):>5} ({len(p3_wrong)*100/n_pairs:.1f}%)")
    print(f"  Gold not in candidate list:      {len(p3_missing):>5} ({len(p3_missing)*100/n_pairs:.1f}%)")
    print()

    # Of the wrong ones, where is gold?
    fixable_top5 = sum(1 for r in p3_wrong if r <= 5)
    fixable_top10 = sum(1 for r in p3_wrong if r <= 10)
    fixable_top20 = sum(1 for r in p3_wrong if r <= 20)

    print(f"  Of the {len(p3_wrong)} wrong cases, gold is at:")
    print(f"    Rank 2-5:   {fixable_top5:>5} — easy for LLM to fix")
    print(f"    Rank 6-10:  {fixable_top10 - fixable_top5:>5} — possible for LLM")
    print(f"    Rank 11-20: {fixable_top20 - fixable_top10:>5} — hard (not in LLM's top-10)")
    print(f"    Rank 21+:   {len(p3_wrong) - fixable_top20:>5} — not fixable by LLM")
    print()

    theoretical_max = len(p3_right) + fixable_top10
    theoretical_pct = theoretical_max * 100 / n_pairs
    print(f"  THEORETICAL MAXIMUM with perfect LLM (top-10):")
    print(f"    {len(p3_right)} (already correct) + {fixable_top10} (fixable) = {theoretical_max}")
    print(f"    = {theoretical_pct:.1f}% accuracy")
    print(f"    Current best (v6): 74.5%")
    print(f"    BioLinkerAI:       81.3%")
    print(f"    Headroom:          {theoretical_pct - 74.5:.1f}%")

    # ── 6. Phase 3 Degradation Details ──
    print()
    print("6. PHASE 3 WORST DEGRADATIONS (pushed gold furthest from #1)")
    print("-" * 70)

    worst = sorted(p3_degraded_cases, key=lambda x: x["after_rank"] - x["before_rank"], reverse=True)
    for case in worst[:15]:
        delta = case["after_rank"] - case["before_rank"]
        print(f"  \"{case['mention']}\" [{case['entity_type']}]")
        print(f"    gold={case['gold_id']}: rank {case['before_rank']} → {case['after_rank']} (Δ+{delta})")
        print(f"    Phase 3 top-1: {case['top1_label']} [{case['top1_id']}]")

    # ── 7. Not in list ──
    print()
    print(f"7. GOLD NOT IN CANDIDATE LIST: {len(not_in_list)} mentions")
    print("-" * 70)

    if not_in_list:
        type_missing = defaultdict(int)
        for case in not_in_list:
            type_missing[case.get("entity_type", "Unknown")] += 1
        print(f"  By entity type:")
        for et in sorted(type_missing.keys(), key=lambda x: type_missing[x], reverse=True):
            print(f"    {et:>8}: {type_missing[et]}")
        print(f"\n  Examples:")
        for case in not_in_list[:10]:
            print(f"    \"{case['mention']}\" [{case.get('entity_type', '?')}] gold={case['gold_id']}")

    # ── 8. JSON Export (optional) ──
    if args.save_json:
        export = {
            "dataset": dataset_name,
            "n_pairs": n_pairs,
            "p2_gold_ranks": p2_gold_ranks,
            "p2b_gold_ranks": p2b_gold_ranks,
            "p3_gold_ranks": p3_gold_ranks,
            "p3_movements": [
                {"before": b, "after": a, "mention": m, "type": et, "gold": g}
                for b, a, m, et, g in p3_movements
            ],
            "p3_degraded_cases": p3_degraded_cases,
            "p3_improved_cases": p3_improved_cases,
            "not_in_list": not_in_list,
            "type_ranks_p3": {k: v for k, v in type_ranks_p3.items()},
        }
        with open(args.save_json, "w") as f:
            json.dump(export, f, indent=2)
        print(f"\nRaw data saved to {args.save_json}")

    print(f"\nAnalysis completed in {elapsed:.0f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ranking Distribution Analysis")
    parser.add_argument("--dataset", choices=["bc5cdr", "medmentions"], default="medmentions")
    parser.add_argument("--backend", default="rapidfuzz")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--umls", type=str, default=None)
    parser.add_argument("--mrrel", type=str, default=None)
    parser.add_argument("--no-expansion", action="store_true")
    parser.add_argument("--no-topic-scoring", action="store_true")
    parser.add_argument("--embedding", action="store_true")
    parser.add_argument("--llm-abbreviation", action="store_true")
    parser.add_argument("--llm-abbreviation-model", type=str, default=None)
    parser.add_argument("--local-base-url", default="http://localhost:1234/v1")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--expansion-top-k", type=int, default=30)
    parser.add_argument("--save-json", type=str, default=None,
                        help="Save raw ranking data to JSON for visualization")
    args = parser.parse_args()
    run_analysis(args)
