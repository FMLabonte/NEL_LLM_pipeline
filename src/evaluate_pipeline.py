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

try:
    from umls_index import UMLSIndex
    HAS_UMLS_INDEX = True
except ImportError:
    HAS_UMLS_INDEX = False
from cui_mesh_mapper import CUIToMeSHMapper
from candidate_expander import CandidateExpander
from umls_relation_expander import UMLSRelationExpander
from domain_rules import DomainRuleReranker
from llm_disambiguator import LLMDisambiguator
from abbreviation_expander import AbbreviationExpander
from string_normalizer import generate_variants
from document_topic_scorer import DocumentTopicScorer

try:
    from embedding_retriever import EmbeddingRetriever, MultiEmbeddingRetriever
    from hybrid_scorer import HybridScorer
    HAS_EMBEDDING = True
except ImportError:
    HAS_EMBEDDING = False

try:
    from llm_abbreviation_expander import LLMAbbreviationExpander
    HAS_LLM_ABBREV = True
except ImportError:
    HAS_LLM_ABBREV = False


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

    # ── Step 1: Build search index ──
    umls_idx = None  # Reference to UMLSIndex if used (for CUI mapping later)

    if args.umls_index:
        # UMLS index mode: search against full UMLS (broader synonym coverage)
        if not HAS_UMLS_INDEX:
            raise ImportError("UMLSIndex not available. Check umls_index.py in candidate-generation/")
        print("=" * 60)
        print(f"Building UMLS index...")
        print("=" * 60)

        mrconso = args.umls or str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF")
        vocabs = args.umls_vocabs.split(",") if args.umls_vocabs else None
        umls_idx = UMLSIndex(vocabularies=vocabs)
        umls_idx.build_from_mrconso(mrconso)
        umls_idx.print_stats()

        # UMLSIndex has a compatible .search() method returning CandidateEntity
        # We wrap it in a simple retriever-like object
        index = umls_idx  # UMLSIndex has .search(query, top_k)
        retriever = CandidateRetriever(index, top_k=args.top_k)

        # Also build a lightweight MeSH index for Phase 3 domain rules (needs tree numbers)
        print("\n  Building MeSH index for domain rules (Phase 3)...")
        mesh_index_for_rules = MeSHIndex(backend=args.backend)
        mesh_index_for_rules.build_from_xml(
            descriptor_path=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
            supplementary_path=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
            enrich_wikidata=False,
            enrich_dbpedia=False,
            enrich_umls=None,
        )
    else:
        # Standard MeSH index mode
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
            enrich_mondo=args.mondo,
        )
        mesh_index_for_rules = index  # same object

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
            mesh_index_for_rules, retriever, umls_bridge=umls_bridge,
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
        mesh_index=mesh_index_for_rules,
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

    # ── Step 4b2: LLM Abbreviation Expander (optional fallback) ──
    llm_abbrev_expander = None
    if args.llm_abbreviation and HAS_LLM_ABBREV:
        try:
            llm_abbrev_expander = LLMAbbreviationExpander(
                model=args.llm_abbreviation_model or args.model,
                base_url=args.base_url,
                temperature=0.3,
                debug=args.llm_abbreviation_debug,
            )
            print("  LLM abbreviation expander: ENABLED (fallback after rule-based)")
        except Exception as e:
            print(f"  LLM abbreviation expander: FAILED ({e})")
    elif args.llm_abbreviation and not HAS_LLM_ABBREV:
        print("  LLM abbreviation expander: DISABLED (openai package not installed)")
    else:
        print("  LLM abbreviation expander: DISABLED")

    # ── Step 4c: Embedding Retriever (optional) ──
    emb_retriever = None
    if args.embedding and HAS_EMBEDDING:
        cache_dir = str(PROJECT_ROOT / "src" / "improvements" / "cache" / "faiss")
        # Embedding retriever always uses MeSH index (for FAISS label encoding)
        emb_index = mesh_index_for_rules
        if args.embedding_multi:
            # Multi-model: SapBERT + BioLinkBERT
            models = [
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                "michiyasunaga/BioLinkBERT-base",
            ]
            emb_retriever = MultiEmbeddingRetriever(
                mesh_index=emb_index,
                model_names=models,
                batch_size=args.embedding_batch_size,
            )
            emb_retriever.build_or_load(cache_dir)
            print(f"  Embedding retriever: MULTI-MODEL ({', '.join(m.split('/')[-1] for m in models)})")
        else:
            # Single model
            emb_retriever = EmbeddingRetriever(
                mesh_index=emb_index,
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

    # ── Step 4e: Document Topic Scorer (optional) ──
    topic_scorer = None
    if not args.no_topic_scoring:
        topic_scorer = DocumentTopicScorer(
            boost=args.topic_boost,
            penalty=args.topic_penalty,
            min_signal=args.topic_min_signal,
        )
        print(f"  Document topic scoring: ENABLED (boost={args.topic_boost}, penalty={args.topic_penalty})")
    else:
        print("  Document topic scoring: DISABLED")

    # ── Step 5: Load dataset ──
    print("\n" + "=" * 60)
    dataset_name = args.dataset.upper()
    print(f"Loading {dataset_name} test set...")
    print("=" * 60)

    if args.dataset == "bc5cdr":
        data_path = str(PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516" / "CDR_TestSet.PubTator.txt")
    elif args.dataset == "biored":
        data_path = str(PROJECT_ROOT / "Data" / "BioRED" / "Test.PubTator")
    elif args.dataset == "medmentions":
        data_path = str(PROJECT_ROOT / "Data" / "MedMention" / "MedMentions_st21pv_pubtator.txt")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    meta, anns, rels = parse_pubtator(data_path)

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

    # Build ID mapping (uses MeSH index for version remapping)
    id_map = _build_id_mapping(mesh_index_for_rules, args.umls)
    label_to_ids = id_map.pop("__label_to_ids__", {})
    id_map.pop("__label_lookup__", None)

    # Filter to MeSH-linkable entities only
    # BioRED: only DiseaseOrPhenotypicFeature + ChemicalEntity have MeSH IDs
    # MedMentions: uses UMLS IDs (not MeSH) — filter to those mappable to MeSH
    eval_df = anns.copy()

    if args.dataset == "biored":
        # Only keep entities with MeSH-like IDs (start with D or C followed by digits)
        mesh_mask = eval_df["mesh_id"].str.match(r'^[DC]\d+', na=False)
        skipped = len(eval_df) - mesh_mask.sum()
        eval_df = eval_df[mesh_mask]
        print(f"  Filtered to MeSH-linkable entities: {len(eval_df)} (skipped {skipped} non-MeSH)")
        # Show entity type breakdown
        print(f"  Entity types: {dict(eval_df['entity_type'].value_counts())}")

    elif args.dataset == "medmentions":
        # MedMentions uses UMLS CUI format "UMLS:C0010674"
        eval_df["original_cui"] = eval_df["mesh_id"]
        eval_df["cui"] = eval_df["mesh_id"].str.replace("UMLS:", "", regex=False)

        if args.umls_index and umls_idx is not None:
            # UMLS index mode: search returns CUIs directly
            umls_idx.return_cui = True  # ensure search returns CUIs, not MeSH IDs
            eval_df["mesh_id"] = eval_df["cui"]  # use raw CUI as gold ID

            if args.fair_comparison:
                # Fair comparison mode: only keep CUIs that have a MeSH mapping
                # (same subset as MeSH index path, for apples-to-apples comparison)
                mrconso_path = args.umls or str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF")
                cui_mapper = CUIToMeSHMapper(mrconso_path)
                cui_mapper.load()
                eval_df["has_mesh"] = eval_df["cui"].apply(lambda c: cui_mapper.is_mappable(c))
                skipped = (~eval_df["has_mesh"]).sum()
                eval_df = eval_df[eval_df["has_mesh"]].copy()
                eval_df = eval_df.drop(columns=["has_mesh"])
                print(f"  UMLS index (FAIR COMPARISON): {len(eval_df)} annotations with MeSH mapping "
                      f"(skipped {skipped} without MeSH)")
                print(f"  Unique CUIs: {eval_df['cui'].nunique()}")
            else:
                print(f"  UMLS index mode: evaluating ALL {len(eval_df)} annotations (no CUI→MeSH filtering)")
                print(f"  Unique CUIs: {eval_df['cui'].nunique()}")
        else:
            # MeSH index mode: need CUI→MeSH mapping (only 63% of annotations)
            mrconso_path = args.umls or str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF")
            cui_mapper = CUIToMeSHMapper(mrconso_path)
            cui_mapper.load()

            # Filter to CUIs that have a MeSH mapping
            eval_df["has_mesh"] = eval_df["cui"].apply(lambda c: cui_mapper.is_mappable(c))
            skipped = (~eval_df["has_mesh"]).sum()
            eval_df = eval_df[eval_df["has_mesh"]].copy()

            # Replace CUI with MeSH ID(s) — take first MeSH ID as primary gold
            # (store all as pipe-separated for multi-ID matching)
            def cui_to_mesh_str(cui):
                mesh_ids = cui_mapper.cui_to_mesh(cui)
                return "|".join(sorted(mesh_ids)) if mesh_ids else cui
            eval_df["mesh_id"] = eval_df["cui"].apply(cui_to_mesh_str)

            eval_df = eval_df.drop(columns=["has_mesh"])
            print(f"  CUI→MeSH mapping: {len(eval_df)} annotations mappable (skipped {skipped} without MeSH)")
            print(f"  Unique CUIs mapped: {eval_df['cui'].nunique()}")

    # Remove entries with no valid ID and deduplicate
    eval_df = eval_df[eval_df["mesh_id"] != "-1"].drop_duplicates(subset=["mention", "mesh_id"])

    if args.limit:
        eval_df = eval_df.head(args.limit)

    n_pairs = len(eval_df)
    print(f"  {n_pairs} unique (mention, mesh_id) pairs to evaluate")

    # ── Step 6: Run evaluation ──
    print("\n" + "=" * 60)
    phases_label = ""
    if abbrev_expander:
        if llm_abbrev_expander:
            phases_label += "Abbrev(+LLM) → "
        else:
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
    K_VALUES = [1, 5, 10, 20, 30]
    p2_at_k = {k: 0 for k in K_VALUES}   # Phase 2 Accuracy@k
    p2b_at_k = {k: 0 for k in K_VALUES}  # Phase 2+expansion Accuracy@k
    p3_at_k = {k: 0 for k in K_VALUES}   # Phase 2+expansion+Phase 3 Accuracy@k

    p2b_improved_over_p2 = 0
    p2b_degraded_vs_p2 = 0
    p3_improved_over_p2b = 0
    p3_degraded_vs_p2b = 0
    p4_improved_over_p3 = 0
    p4_degraded_vs_p3 = 0

    # Per-entity-type tracking
    from collections import defaultdict
    type_total = defaultdict(int)
    type_p3_correct = defaultdict(int)

    # Expansion diagnostics
    expansion_added_candidates = 0  # total new candidates added by expansion
    expansion_added_gold = 0        # how often expansion brought the gold into the list
    expansion_mentions_affected = 0 # how many mentions got new candidates

    # Abbreviation expansion diagnostics
    abbrev_expanded_count = 0     # how many mentions were expanded (rule-based)
    abbrev_improved_count = 0     # how many went from wrong to correct thanks to expansion

    # LLM abbreviation expansion diagnostics
    llm_abbrev_expanded_count = 0   # how many mentions were expanded by LLM
    llm_abbrev_improved_count = 0   # how many went from wrong to correct thanks to LLM expansion

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
        abbreviation_source = None  # "rule" or "llm"
        if abbrev_expander is not None:
            ctx = context_lookup.get(pmid, {})
            full_text = ctx.get("full_text", "")
            doc_title = ctx.get("title", "")
            expanded = abbrev_expander.expand_mention(
                mention, context=full_text, title=doc_title,
            )
            if expanded:
                abbreviation_expanded = expanded
                abbreviation_source = "rule"
                abbrev_expanded_count += 1

        # LLM abbreviation expansion fallback: if rule-based didn't find
        # anything and the mention looks like an abbreviation, ask the LLM
        if abbreviation_expanded is None and llm_abbrev_expander is not None:
            # Only try if it looks like an abbreviation (short, uppercase-heavy)
            # Use the rule-based expander's check if available, otherwise
            # use a simple heuristic: short text with >50% uppercase
            is_abbrev = False
            if abbrev_expander is not None:
                is_abbrev = abbrev_expander._is_likely_abbreviation(mention)
            else:
                # Standalone check: 2-10 chars, >50% uppercase
                m = mention.strip()
                if 2 <= len(m) <= 10:
                    is_abbrev = sum(1 for c in m if c.isupper()) / len(m) >= 0.5

            if is_abbrev:
                ctx = context_lookup.get(pmid, {})
                llm_expanded = llm_abbrev_expander.expand(
                    mention,
                    context=ctx.get("full_text", ""),
                    title=ctx.get("title", ""),
                )
                if llm_expanded:
                    abbreviation_expanded = llm_expanded
                    abbreviation_source = "llm"
                    llm_abbrev_expanded_count += 1

        # ── Phase 2: Retrieve candidates ──
        # Search original mention + normalized variants
        candidates = retriever.retrieve(mention, top_k=args.top_k)
        if not args.no_string_normalization:
            variants = generate_variants(mention)
            existing_ids = {c.mesh_id for c in candidates}
            for variant in variants[1:]:  # skip first (= original)
                var_candidates = retriever.retrieve(variant, top_k=args.top_k)
                for vc in var_candidates:
                    if vc.mesh_id not in existing_ids:
                        candidates.append(vc)
                        existing_ids.add(vc.mesh_id)

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
                    if abbreviation_source == "llm":
                        llm_abbrev_improved_count += 1
                    else:
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
        doc_text = context_lookup.get(pmid, {}).get("full_text", "")
        reranked = reranker.rerank(
            mention=mention,
            candidates=candidates,
            entity_type=entity_type,
            context=doc_text,
        )

        # ── Document topic consistency (optional, after Phase 3) ──
        if topic_scorer is not None:
            topic = topic_scorer.detect_topic(doc_text)
            reranked = topic_scorer.rescore(reranked, topic)

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

        # Per-entity-type tracking
        et_key = entity_type or "Unknown"
        type_total[et_key] += 1
        if p3_hit:
            type_p3_correct[et_key] += 1

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
    print(f"RESULTS — {dataset_name} Pipeline Evaluation")
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
        print(f"    Mentions expanded (rule-based): {abbrev_expanded_count}/{total} "
              f"({abbrev_expanded_count*100/total:.1f}%)")
        print(f"    Directly improved top-1:        {abbrev_improved_count}")
        if llm_abbrev_expander is not None:
            print(f"    Mentions expanded (LLM):        {llm_abbrev_expanded_count}/{total} "
                  f"({llm_abbrev_expanded_count*100/total:.1f}%)")
            print(f"    LLM directly improved top-1:    {llm_abbrev_improved_count}")
            llm_abbrev_expander.print_stats()

    # ── Per-entity-type accuracy ──
    if len(type_total) > 1:
        print(f"\n{'─' * 60}")
        print(f"  Accuracy by entity type (Phase 3 @1):")
        for et in sorted(type_total.keys(), key=lambda x: type_total[x], reverse=True):
            t = type_total[et]
            c = type_p3_correct[et]
            pct = c / t * 100 if t > 0 else 0
            print(f"    {et:35s}  {pct:5.1f}%  ({c}/{t})")

    print(f"\n{'─' * 60}")
    print(f"  Dataset:                     {dataset_name}")
    expansion_str = "ON" if expander else "OFF"
    print(f"  Candidate expansion:         {expansion_str}")
    if expander:
        print(f"  Expansion top_k:             {args.expansion_top_k}")
    abbrev_str = "ON" if abbrev_expander else "OFF"
    if llm_abbrev_expander is not None:
        abbrev_str += " + LLM fallback"
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
    parser = argparse.ArgumentParser(description="Full Pipeline Evaluation")

    # Dataset selection
    parser.add_argument("--dataset", choices=["bc5cdr", "biored", "medmentions"],
                        default="bc5cdr", help="Dataset to evaluate on (default: bc5cdr)")

    # Phase 2 settings
    parser.add_argument("--backend", choices=["rapidfuzz", "elasticsearch"], default="rapidfuzz")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--top-k", type=int, default=10, help="Phase 2 candidates")
    parser.add_argument("--wikidata", action="store_true", help="Wikidata enrichment")
    parser.add_argument("--dbpedia", action="store_true", help="DBpedia enrichment")
    parser.add_argument("--mondo", action="store_true", help="MONDO disease ontology enrichment (improves Disease linking)")
    parser.add_argument("--umls", type=str, default=None, help="Path to MRCONSO.RRF")

    # UMLS index (alternative to MeSH index)
    parser.add_argument("--umls-index", action="store_true",
                        help="Use UMLS index instead of MeSH index (broader synonym coverage, "
                             "direct CUI matching for MedMentions)")
    parser.add_argument("--umls-vocabs", type=str, default=None,
                        help="Comma-separated UMLS vocabularies to include "
                             "(default: MSH,SNOMEDCT_US,NCI,CHV,MTH,OMIM,HPO,RXNORM). Use 'ALL' for everything.")
    parser.add_argument("--fair-comparison", action="store_true",
                        help="When using UMLS index on MedMentions: only evaluate CUIs that "
                             "have a MeSH mapping (same subset as MeSH index). Enables fair comparison.")

    # Phase 2b: Expansion settings
    parser.add_argument("--no-expansion", action="store_true", help="Skip candidate expansion (UMLS bridge, multi-word, parent)")
    parser.add_argument("--mrrel", type=str, default=None, help="Path to MRREL.RRF for UMLS bridge expansion")
    parser.add_argument("--expansion-top-k", type=int, default=30, help="Max candidates after expansion (default: 30)")

    # Improvements
    parser.add_argument("--no-abbreviation-expansion", action="store_true", help="Skip abbreviation expansion")
    parser.add_argument("--llm-abbreviation", action="store_true",
                        help="Enable LLM-based abbreviation expansion as fallback "
                             "when rule-based expansion fails")
    parser.add_argument("--llm-abbreviation-model", type=str, default=None,
                        help="Model for LLM abbreviation expansion (default: same as --model)")
    parser.add_argument("--llm-abbreviation-debug", action="store_true",
                        help="Show first 5 raw LLM responses for abbreviation expansion debugging")
    parser.add_argument("--no-string-normalization", action="store_true", help="Skip string normalization variants")
    parser.add_argument("--no-topic-scoring", action="store_true", help="Skip document topic consistency scoring")
    parser.add_argument("--topic-boost", type=float, default=3.0, help="Topic match boost (default: 3.0)")
    parser.add_argument("--topic-penalty", type=float, default=-3.0, help="Topic conflict penalty (default: -3.0)")
    parser.add_argument("--topic-min-signal", type=int, default=4, help="Min keyword weight to activate topic scoring (default: 4)")

    # Embedding retrieval
    parser.add_argument("--embedding", action="store_true", help="Enable embedding-based retrieval (SapBERT + FAISS)")
    parser.add_argument("--embedding-model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                        help="HuggingFace model for embedding retrieval")
    parser.add_argument("--embedding-batch-size", type=int, default=256, help="Batch size for encoding")
    parser.add_argument("--embedding-multi", action="store_true",
                        help="Use both SapBERT + BioLinkBERT (multi-model ensemble)")
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
