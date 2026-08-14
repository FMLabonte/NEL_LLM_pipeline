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
import re
import time
import json
import difflib
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
from prompts import list_prompts
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

try:
    from sentence_context_scorer import SentenceContextScorer
    HAS_SENTENCE_CONTEXT = True
except ImportError:
    HAS_SENTENCE_CONTEXT = False

try:
    from mini_disease_disambiguator import MiniDiseaseDisambiguator
    HAS_MINI_DISAMBIG = True
except ImportError:
    HAS_MINI_DISAMBIG = False

try:
    from learned_ranker import LearnedRanker, extract_features_for_mention
    HAS_LEARNED_RANKER = True
except Exception:
    HAS_LEARNED_RANKER = False


def locate_mention(text: str, mention: str, start: int = -1) -> int:
    """
    Character offset of `mention` in `text`, or -1.

    Prefers the gold annotation offset, but only after verifying it actually
    points at the mention (PubTator offsets are computed over title+abstract,
    which must line up with how we join them). Falls back to a word-boundary
    search, then to a plain substring search.

    The plain search alone is what the pipeline used to do, and it silently
    matches inside longer words — "dex" in "dexamethasone", "AL" in "renal",
    "Cr" in "increased". That mislocates ~3.4% of BC5CDR mentions, nearly all
    of them abbreviations, i.e. the ones that most need correct context.
    """
    if not mention:
        return -1
    if (
        start is not None and start >= 0
        and start + len(mention) <= len(text)
        and text[start:start + len(mention)].lower() == mention.lower()
    ):
        return start
    m = re.search(
        r'(?<![A-Za-z0-9])' + re.escape(mention.strip()) + r'(?![A-Za-z0-9])',
        text, re.IGNORECASE,
    )
    if m:
        return m.start()
    return text.lower().find(mention.lower())


def extract_sentence(text: str, mention: str, start: int = -1, window: int = 200) -> str:
    """Extract a sentence-level context window around the mention."""
    start = locate_mention(text, mention, start)
    if start < 0:
        return text[:window]

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
        print("Building UMLS index...")

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
        print("  + MeSH index for Phase 3 rules...")
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
        print(f"Building MeSH index ({args.backend})...")

        index = MeSHIndex(backend=args.backend, es_url=args.es_url)
        index.build_from_xml(
            descriptor_path=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
            supplementary_path=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
            enrich_wikidata=args.wikidata,
            enrich_dbpedia=args.dbpedia,
            enrich_umls=args.umls,
            enrich_mondo=args.mondo,
            enrich_mrdef=args.mrdef,
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
            pass  # UMLS bridge loaded
        else:
            pass  # UMLS bridge files not found

        expander = CandidateExpander(
            mesh_index_for_rules, retriever, umls_bridge=umls_bridge,
        )
    else:
        pass  # no expansion

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
            pass  # cache loaded silently

    # ── Step 3: Create Phase 3 reranker ──
    reranker = DomainRuleReranker(
        mesh_index=mesh_index_for_rules,
        rule5_boost=args.rule5_boost,
        rule6_penalty=args.rule6_penalty,
        rule7_boost=args.rule7_boost,
        rule8_specificity_boost=args.rule8_specificity_boost,
        top1_protection_threshold=args.top1_protection_threshold,
        wikidata_synonyms=wikidata,
        dbpedia_synonyms=dbpedia,
        umls_synonyms=umls_cache,
    )

    # ── Step 4: Create Phase 4 disambiguator (optional) ──
    # When dumping fine-tuning data we never call the LLM — we only need the
    # candidate lists — so skip the disambiguator (and its endpoint entirely).
    disambiguator = None
    if not args.no_phase4 and not args.dump_finetune:
        try:
            disambiguator = LLMDisambiguator(
                model=args.model,
                base_url=args.base_url,
                api_key=args.api_key,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                no_think=args.phase4_no_think,
                prompt_version=args.prompt_version,
                shuffle_candidates=args.shuffle_candidates,
                shuffle_seed=args.shuffle_seed if args.shuffle_candidates else None,
                max_definition_len=args.max_definition_len,
                structured_output=args.structured_output,
            )
        except Exception as e:
            print(f"Warning: Could not connect to LLM: {e}")
            print("Phase 4 will be skipped.")

    # ── Step 4b: Abbreviation Expander (optional) ──
    abbrev_expander = None
    if not args.no_abbreviation_expansion:
        abbrev_expander = AbbreviationExpander()

    # ── Step 4b2: LLM Abbreviation Expander (optional fallback) ──
    llm_abbrev_expander = None
    if args.llm_abbreviation and HAS_LLM_ABBREV:
        try:
            llm_abbrev_expander = LLMAbbreviationExpander(
                model=args.llm_abbreviation_model or args.model,
                base_url=args.local_base_url,
                temperature=0.3,
                debug=args.llm_abbreviation_debug,
            )
        except Exception as e:
            print(f"  Warning: LLM abbreviation expander failed: {e}")

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
        else:
            # Single model
            emb_retriever = EmbeddingRetriever(
                mesh_index=emb_index,
                model_name=args.embedding_model,
                batch_size=args.embedding_batch_size,
            )
            emb_retriever.build_or_load(cache_dir)
    elif args.embedding and not HAS_EMBEDDING:
        print("  Warning: embedding retriever unavailable (pip install torch transformers faiss-cpu)")

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

    # ── Step 4f: Sentence-Context Scorer (optional, Disease-focused) ──
    sentence_scorer = None
    if args.sentence_context and emb_retriever is not None and HAS_SENTENCE_CONTEXT:
        sentence_scorer = SentenceContextScorer(
            embedding_retriever=emb_retriever,
            weight=args.sentence_context_weight,
            ambiguity_threshold=args.sentence_context_threshold,
            disease_only=True,
        )
    elif args.sentence_context and emb_retriever is None:
        print("  Warning: sentence-context scoring requires --embedding")

    # ── Step 4g: Mini Disease Disambiguator (optional, Disease-focused) ──
    mini_disambig = None
    if args.mini_disambig and HAS_MINI_DISAMBIG:
        try:
            mini_disambig = MiniDiseaseDisambiguator(
                model=args.mini_disambig_model or args.model,
                base_url=args.local_base_url,
                ambiguity_threshold=args.mini_disambig_threshold,
                max_candidates=args.mini_disambig_max_candidates,
                debug=args.mini_disambig_debug,
            )
        except Exception as e:
            print(f"  Warning: Mini Disease Disambiguator failed: {e}")

    # ── Step 4h: Learned Ranker (optional, XGBoost) ──
    learned_ranker = None
    if args.learned_ranker and HAS_LEARNED_RANKER:
        try:
            learned_ranker = LearnedRanker(model_path=args.learned_ranker)
        except Exception as e:
            print(f"  Warning: Learned ranker failed: {e}")
    elif args.train_ranker and HAS_LEARNED_RANKER:
        pass  # will train after evaluation
    elif (args.learned_ranker or args.train_ranker) and not HAS_LEARNED_RANKER:
        print("  Warning: xgboost not installed for learned ranker")

    # ── Feature summary (compact) ──
    features_on = []
    if expander:
        features_on.append("expansion")
    if abbrev_expander:
        features_on.append("abbrev")
    if llm_abbrev_expander:
        features_on.append("llm-abbrev")
    if emb_retriever:
        model_short = args.embedding_model.split("/")[-1] if not args.embedding_multi else "multi"
        features_on.append(f"embedding({model_short})")
    if hybrid_scorer:
        features_on.append(f"hybrid(α={args.hybrid_alpha})")
    if topic_scorer:
        features_on.append("topic-scoring")
    if sentence_scorer:
        features_on.append("sentence-context")
    if mini_disambig:
        features_on.append("mini-disambig")
    if learned_ranker:
        features_on.append("learned-ranker")
    if args.train_ranker:
        features_on.append("ranker-training")
    print(f"  Features: {', '.join(features_on) if features_on else 'none'}")

    # ── Step 5: Load dataset ──
    dataset_name = args.dataset.upper()
    split_label = args.split.upper() if hasattr(args, 'split') else "TEST"
    print(f"Loading {dataset_name} {split_label}...")

    if args.dataset == "bc5cdr":
        split_map = {"train": "TrainingSet", "dev": "DevelopmentSet", "test": "TestSet"}
        split_name = split_map.get(args.split, "TestSet")
        data_path = str(PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516" / f"CDR_{split_name}.PubTator.txt")
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
        print(f"  Filtered to MeSH-linkable: {len(eval_df)} (skipped {skipped})")

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
                print(f"  UMLS fair-comparison: {len(eval_df)} annotations (skipped {skipped})")
            else:
                pass  # evaluating all annotations
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
            print(f"  CUI→MeSH: {len(eval_df)} mappable (skipped {skipped})")

    # Remove entries with no valid ID
    eval_df = eval_df[eval_df["mesh_id"] != "-1"]
    n_all_mentions = len(eval_df)

    # Deduplicate unless asked not to.
    #
    # This single line decides what our headline number means, and it is NOT
    # what BioLinkerAI reports. They evaluate "over the total number of input
    # mentions" (~9.7k on BC5CDR test); deduplicating leaves 2625, i.e. 27%.
    # BC5CDR is heavily repetitive — "seizures" occurs 99x, "cocaine" 92x —
    # and those repeated mentions are mostly easy exact matches. Deduplicating
    # therefore evaluates almost entirely on the hard long tail (47% of unique
    # pairs occur exactly once) and is NOT comparable to the published numbers.
    #
    # Report both. --keep-duplicates gives the comparable figure.
    if args.keep_duplicates:
        print(f"  Keeping duplicate mentions "
              f"(comparable to BioLinkerAI's denominator)")
    else:
        eval_df = eval_df.drop_duplicates(subset=["mention", "mesh_id"])
        pct = len(eval_df) / n_all_mentions * 100 if n_all_mentions else 0
        print(f"  Deduplicated to unique (mention, gold) pairs: "
              f"{len(eval_df)}/{n_all_mentions} ({pct:.1f}% of all mentions)")
        print(f"    NOTE: BioLinkerAI reports over ALL mentions. "
              f"Use --keep-duplicates for a comparable number.")

    n_unique_pairs = eval_df.drop_duplicates(subset=["mention", "mesh_id"]).shape[0]

    if args.limit:
        eval_df = eval_df.head(args.limit)

    n_pairs = len(eval_df)
    print(f"  Evaluating {n_pairs} mention-entity pairs"
          + (f" (limited to {args.limit})" if args.limit else ""))

    # ── Step 5b: Retrieval-based few-shot examples (optional) ──
    #
    # Replaces the three hard-coded examples with the k most similar annotated
    # mentions from the TRAINING split. The point is not in-context learning in
    # general — it is that our Phase 4 errors are annotation-convention errors
    # (BC5CDR gold is usually the broader MeSH descriptor), and retrieved
    # examples carry that convention in a way prompt wording does not.
    fewshot_retriever = None
    if args.retrieval_fewshot:
        if disambiguator is None:
            print("  --retrieval-fewshot ignored (Phase 4 is disabled)")
        else:
            try:
                from fewshot_retriever import FewShotRetriever
                label_lookup = {}
                if mesh_index_for_rules is not None:
                    try:
                        label_lookup = {
                            mid: e.preferred_label
                            for mid, e in mesh_index_for_rules.entities.items()
                        }
                    except AttributeError:
                        pass
                fewshot_retriever = FewShotRetriever(
                    model_name=args.embedding_model,
                    context_words=args.fewshot_context_words,
                )
                fewshot_retriever.build_or_load(
                    dataset=args.dataset,
                    cache_dir=str(PROJECT_ROOT / "src" / "improvements" / "cache"),
                    label_lookup=label_lookup,
                )
            except Exception as e:
                print(f"  Few-shot retrieval unavailable ({e}) — "
                      f"falling back to hard-coded examples")
                fewshot_retriever = None

    # ── Step 6: Run evaluation ──
    print("=" * 60)
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
    print(f"Running {phases_label}...")
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
    llm_skipped = 0  # confidence-based cascading: skipped because Phase 3 was confident

    # Learned ranker training data collection
    ranker_train_features = []  # list of list[dict] per mention
    ranker_train_labels = []    # list of list[int] per mention

    # ── Fine-tuning data dump (optional) ──
    # Reuses the exact same candidate generation + prompt builder as inference,
    # so train and test see the same input distribution. Writes chat-format
    # SFT JSONL. TRAIN SPLIT ONLY — refuses to run on test to prevent leakage.
    dump_fh = None
    dump_config = None
    dump_counts = defaultdict(int)   # per (mention, gold) pair, for capping
    dump_written = 0
    dump_skipped_nogold = 0
    if args.dump_finetune:
        if getattr(args, "split", "test") == "test":
            raise SystemExit(
                "Refusing to build fine-tuning data from the TEST split. "
                "Pass --split train (never train on what you evaluate on)."
            )
        import dataclasses as _dc
        from prompts import get_prompt_config, build_user_prompt as _build_up
        # Strip few-shot examples from the training prompt. Fine-tuning should
        # teach the convention in the weights, so the target inference prompt is
        # the bare one (no hardcoded or retrieved examples). Train and test must
        # use the same prompt — run the fine-tuned model WITHOUT --retrieval-fewshot.
        dump_config = _dc.replace(
            get_prompt_config(args.prompt_version), include_few_shot=False,
        )
        dump_path = Path(args.dump_finetune)
        dump_path.parent.mkdir(parents=True, exist_ok=True)
        dump_fh = open(dump_path, "w")
        print(f"  Fine-tuning dump -> {dump_path} "
              f"(split={args.split}, target={args.finetune_target}, "
              f"max {args.finetune_max_per_pair}/pair)")

    # Failure-stage decomposition (see the eval loop for the definitions)
    cg_failures = 0     # gold not in the candidate list the LLM sees
    ned_failures = 0    # gold in the list, but top-1 is wrong
    p4_ned_failures = 0  # same, measured after the LLM had its say
    n_gold_in_list = 0  # gold present in the top-k list the LLM sees
                        # -> denominator for disambiguation accuracy

    changes_log = []
    debug_llm_failures = []  # detailed logs for --debug-llm

    # Per-mention prediction dump (for paired significance testing across runs)
    pred_fh = None
    if args.dump_predictions:
        pp = Path(args.dump_predictions)
        pp.parent.mkdir(parents=True, exist_ok=True)
        pred_fh = open(pp, "w")

    t0 = time.time()

    for _, row in eval_df.iterrows():
        mention = row["mention"]
        gold_id = row["mesh_id"]
        pmid = row["pmid"]
        entity_type = row.get("entity_type", None)

        # Gold character offset — anchors every context window we build below.
        try:
            mention_start = int(row["start"])
        except (KeyError, TypeError, ValueError):
            mention_start = -1

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

        # >>> SANITY CHECK — mention header (uncomment to enable) <<<
        # if args.limit and args.limit <= 10:
        #     print(f"\n{'#'*70}")
        #     print(f"  MENTION {total+1}/{n_pairs}: \"{mention}\"")
        #     print(f"  Gold: {gold_id} | Type: {entity_type} | PMID: {pmid}")
        #     print(f"{'#'*70}")
        # >>> END SANITY CHECK <<<

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

        # >>> SANITY CHECK — Phase 2 candidates before re-ranking (uncomment to enable) <<<
        # if args.limit and args.limit <= 10:
        #     gold_in_list = any(c.mesh_id in expanded_gold_ids for c in candidates[:20])
        #     gold_rank = next(
        #         (i+1 for i, c in enumerate(candidates[:20])
        #          if c.mesh_id in expanded_gold_ids), None
        #     )
        #     print(f"\n  PHASE 2 CANDIDATES (top 10 of {len(candidates)}):"
        #           f"  [Gold in top-20: {'YES @'+str(gold_rank) if gold_in_list else 'NO'}]")
        #     for i, c in enumerate(candidates[:10], 1):
        #         marker = " ★GOLD" if c.mesh_id in expanded_gold_ids else ""
        #         print(f"    {i:>2}. {c.mesh_id:<12} {(c.preferred_label or '')[:40]:<40}"
        #               f" score={c.score:.1f}{marker}")
        # >>> END SANITY CHECK <<<

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

        # ── Sentence-context scoring (optional, Disease-focused) ──
        if sentence_scorer is not None:
            sentence = extract_sentence(doc_text, mention, start=mention_start)
            reranked = sentence_scorer.rescore(
                mention=mention,
                candidates=reranked,
                sentence=sentence,
                entity_type=entity_type,
            )

        # ── Mini Disease Disambiguator (optional, Disease-focused) ──
        if mini_disambig is not None:
            sentence = extract_sentence(doc_text, mention, start=mention_start)
            reranked = mini_disambig.disambiguate_if_ambiguous(
                mention=mention,
                candidates=reranked,
                sentence=sentence,
                entity_type=entity_type,
            )

        # ── Learned Ranker: re-rank with XGBoost (optional) ──
        if learned_ranker is not None:
            reranked = learned_ranker.rerank(
                mention=mention,
                candidates=reranked,
                entity_type=entity_type,
                context=doc_text,
            )

        # ── Learned Ranker: collect training data (optional) ──
        if args.train_ranker and HAS_LEARNED_RANKER:
            feats = extract_features_for_mention(
                mention, reranked, entity_type, doc_text,
            )
            labels = [
                1 if c.mesh_id in expanded_gold_ids else 0
                for c in reranked
            ]
            ranker_train_features.append(feats)
            ranker_train_labels.append(labels)

        p3_top1 = reranked[0].mesh_id
        p3_hit = p3_top1 in expanded_gold_ids

        # Phase 3 confidence = score gap between top-1 and top-2. Drives the
        # cascading decision AND is logged for every changed case so we can
        # read the optimal --phase4-threshold straight off the data: pick a
        # value above the gap of most improvements but below that of most
        # degradations.
        p3_score_gap = (
            reranked[0].score - reranked[1].score if len(reranked) >= 2 else 999.0
        )

        # Accuracy@k for Phase 3
        p3_ids = [c.mesh_id for c in reranked]
        for k in K_VALUES:
            if any(mid in expanded_gold_ids for mid in p3_ids[:k]):
                p3_at_k[k] += 1

        # ── Failure-stage decomposition (Ye & Mitchell, ACL 2025, Table 4) ──
        # Splits every error into "the gold was never retrieved" (candidate
        # generation) vs "the gold was right there and we picked wrong"
        # (disambiguation). Only the second kind is winnable by a better LLM,
        # so this ratio tells us where the remaining headroom actually is.
        gold_in_llm_list = any(
            mid in expanded_gold_ids for mid in p3_ids[:args.llm_top_k]
        )
        if gold_in_llm_list:
            n_gold_in_list += 1
        if not p3_hit:
            if gold_in_llm_list:
                ned_failures += 1
            else:
                cg_failures += 1

        # ── Fine-tuning example dump ──
        # One SFT example per mention whose gold is in the top-k candidate list
        # (can't teach "pick the gold" if the gold isn't among the choices).
        # Prompt is built with the SAME builder as inference, so the fine-tuned
        # model sees the identical input format. No few-shots on purpose: the
        # model should learn the annotation convention in its weights, which is
        # the whole point of fine-tuning (and lets you drop few-shots at test).
        if dump_fh is not None:
            llm_cands = reranked[:args.llm_top_k]
            gold_cand = next(
                (c for c in llm_cands if c.mesh_id in expanded_gold_ids), None
            )
            if gold_cand is None:
                dump_skipped_nogold += 1
            else:
                pair_key = (mention.lower(), gold_cand.mesh_id)
                if dump_counts[pair_key] < args.finetune_max_per_pair:
                    dump_counts[pair_key] += 1
                    ctx = context_lookup.get(pmid, {})
                    full_text = ctx.get("full_text", "")
                    if dump_config.context_words > 0 or dump_config.use_full_context:
                        d_ctx, d_off = full_text, mention_start
                    else:
                        d_ctx = extract_sentence(full_text, mention, start=mention_start)
                        d_off = -1
                    user_prompt = _build_up(
                        dump_config, mention, llm_cands, d_ctx,
                        ctx.get("title", ""), mention_start=d_off,
                    )
                    if args.finetune_target == "id":
                        target = json.dumps({"mesh_id": gold_cand.mesh_id})
                    else:
                        target = json.dumps({
                            "entity_name": gold_cand.preferred_label,
                            "mesh_id": gold_cand.mesh_id,
                        })
                    dump_fh.write(json.dumps({
                        "messages": [
                            {"role": "system", "content": dump_config.system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": target},
                        ],
                        "meta": {
                            "mention": mention,
                            "gold_id": gold_cand.mesh_id,
                            "pmid": pmid,
                            "entity_type": entity_type,
                        },
                    }, ensure_ascii=False) + "\n")
                    dump_written += 1

        # ── Phase 4: LLM disambiguation (optional, with cascading) ──
        p4_top1 = p3_top1
        p4_hit = p3_hit
        confidence = "phase3"

        if disambiguator:
            # Confidence-based cascading: if Phase 3 is confident (large score
            # gap between top-1 and top-2), skip the LLM call and keep Phase 3.
            score_gap = p3_score_gap
            threshold = args.phase4_threshold
            call_llm = (threshold <= 0.0) or (score_gap < threshold)

            if call_llm:
                ctx = context_lookup.get(pmid, {})
                full_text = ctx.get("full_text", "")
                llm_candidates = reranked[:args.llm_top_k]

                # Choose what "context" means for this prompt version.
                # Word-window and full-abstract prompts get the whole document
                # plus the gold offset, so the window is anchored exactly.
                # The legacy sentence path gets a pre-cut snippet, whose own
                # coordinates no longer match the document offset.
                pconf = disambiguator.prompt_config
                if (pconf.context_words > 0 or pconf.context_sentences > 0
                        or pconf.use_full_context):
                    # Bounded/full window: hand over the whole document + gold
                    # offset so the window is anchored exactly by the builder.
                    llm_context = full_text
                    llm_offset = mention_start
                else:
                    llm_context = extract_sentence(
                        full_text, mention, start=mention_start,
                    )
                    llm_offset = -1

                # Retrieved few-shot examples (opt-in via --retrieval-fewshot)
                shots = None
                if fewshot_retriever is not None:
                    shots = fewshot_retriever.retrieve(
                        mention=mention,
                        context=extract_sentence(
                            full_text, mention, start=mention_start,
                        ),
                        k=args.fewshot_k,
                        exclude_pmid=pmid,
                        exclude_exact_mention=args.fewshot_exclude_exact,
                    )

                llm_result = disambiguator.disambiguate(
                    mention=mention,
                    candidates=llm_candidates,
                    context=llm_context,
                    title=ctx.get("title", ""),
                    mention_start=llm_offset,
                    dynamic_examples=shots,
                )

                p4_top1 = llm_result.mesh_id
                p4_hit = p4_top1 in expanded_gold_ids
                confidence = llm_result.confidence
                llm_calls += 1
                if confidence == "fallback":
                    llm_fallbacks += 1

                # ── Debug LLM: log failures where gold was in candidate list ──
                if args.debug_llm and not p4_hit:
                    gold_in_list = any(c.mesh_id in expanded_gold_ids for c in llm_candidates)
                    if gold_in_list:
                        gold_rank = next(
                            (i + 1 for i, c in enumerate(llm_candidates)
                             if c.mesh_id in expanded_gold_ids), -1
                        )
                        gold_cand = next(
                            (c for c in llm_candidates if c.mesh_id in expanded_gold_ids), None
                        )
                        debug_llm_failures.append({
                            "mention": mention,
                            "entity_type": entity_type,
                            "gold_id": gold_id,
                            "gold_rank": gold_rank,
                            "gold_label": gold_cand.preferred_label if gold_cand else "?",
                            "gold_score": gold_cand.score if gold_cand else 0,
                            "gold_definition": (gold_cand.definition or "")[:200] if gold_cand else "",
                            "llm_chose_id": p4_top1,
                            "llm_chose_label": llm_result.preferred_label,
                            "llm_chose_rank": llm_result.chosen_rank,
                            "llm_response": llm_result.raw_response,
                            "top1_id": llm_candidates[0].mesh_id,
                            "top1_label": llm_candidates[0].preferred_label,
                            "top1_score": llm_candidates[0].score,
                            "n_candidates": len(llm_candidates),
                            "pmid": pmid,
                        })
            else:
                # Phase 3 was confident — skip LLM, keep Phase 3 result
                llm_skipped += 1

        # How many errors remain winnable after the LLM had its chance?
        if disambiguator and not p4_hit and gold_in_llm_list:
            p4_ned_failures += 1

        # >>> SANITY CHECK — final verdict (uncomment to enable) <<<
        # if args.limit and args.limit <= 10:
        #     print(f"\n  {'─'*66}")
        #     print(f"  VERDICT: \"{mention}\"")
        #     print(f"    Gold:     {gold_id}")
        #     print(f"    Phase 2:  {p2_top1} {'✓' if p2_hit else '✗'}")
        #     print(f"    Phase 2b: {p2b_top1} {'✓' if p2b_hit else '✗'}")
        #     print(f"    Phase 3:  {p3_top1} {'✓' if p3_hit else '✗'}")
        #     print(f"    Phase 4:  {p4_top1} {'✓' if p4_hit else '✗'}"
        #           f" (confidence: {confidence})")
        #     print(f"  {'#'*70}\n")
        # >>> END SANITY CHECK <<<

        # Per-mention prediction record (McNemar + decomposition analysis).
        # The extra fields let us stratify the LLM's contribution by retriever
        # confidence, mention difficulty, and gold rank (see analyze_decomposition.py).
        if pred_fh is not None:
            p3_label = reranked[0].preferred_label if reranked else ""
            surf = (difflib.SequenceMatcher(
                None, mention.lower(), p3_label.lower()).ratio() * 100
            ) if p3_label else 0.0
            gold_rank_full = next(
                (i + 1 for i, c in enumerate(reranked)
                 if c.mesh_id in expanded_gold_ids), -1
            )
            pred_fh.write(json.dumps({
                "pmid": pmid,
                "mention": mention,
                "gold_id": gold_id,
                "entity_type": entity_type,
                "p3_correct": bool(p3_hit),
                "p4_correct": bool(p4_hit),
                # decomposition fields
                "p3_top1": p3_top1,
                "p4_top1": p4_top1,
                "llm_changed": bool(p4_top1 != p3_top1),
                "p3_score_gap": round(p3_score_gap, 2),   # retriever confidence
                "p3_top1_score": round(reranked[0].score, 2) if reranked else 0.0,
                "gold_in_list": bool(gold_in_llm_list),
                "gold_rank": gold_rank_full,              # -1 if not retrieved
                "surface_sim": round(surf, 1),            # mention vs top-1 label
            }, ensure_ascii=False) + "\n")

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
                "p3_score_gap": round(p3_score_gap, 2),
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
                if llm_skipped > 0:
                    line += f" (LLM: {llm_calls}, skip: {llm_skipped})"
            line += f" [{elapsed:.0f}s, ETA {eta:.0f}s]"
            print(line, flush=True)

    elapsed = time.time() - t0

    if pred_fh is not None:
        pred_fh.close()
        print(f"\n  Per-mention predictions -> {args.dump_predictions}")

    # ── Fine-tuning dump: finalize ──
    if dump_fh is not None:
        dump_fh.close()
        print("\n" + "=" * 60)
        print(f"FINE-TUNING DATA written to {args.dump_finetune}")
        print("=" * 60)
        print(f"  Examples written:            {dump_written}")
        print(f"  Unique (mention, gold) pairs:{len(dump_counts)}")
        print(f"  Skipped (gold not in top-{args.llm_top_k}): {dump_skipped_nogold}")
        print(f"  Cap per pair:                {args.finetune_max_per_pair}")
        print(f"  Format: chat SFT JSONL (system/user/assistant), "
              f"target={args.finetune_target}")
        print(f"  Time: {elapsed:.0f}s")
        return

    # ── Results ──
    print("\n" + "=" * 60)
    print(f"RESULTS — {dataset_name} Pipeline Evaluation")
    print("=" * 60)

    p2_acc = p2_correct / total * 100 if total > 0 else 0
    p2b_acc = p2b_correct / total * 100 if total > 0 else 0
    p3_acc = p3_correct / total * 100 if total > 0 else 0
    p4_acc = p4_correct / total * 100 if total > 0 else 0

    # Disambiguation accuracy = of the cases where gold IS in the top-k list,
    # how often is it picked? This isolates the *picker* (rules / LLM) from
    # candidate generation, so it can be compared across models without being
    # dragged down by retrieval misses.
    disambig_p3 = p3_correct / n_gold_in_list * 100 if n_gold_in_list else 0
    disambig_p4 = p4_correct / n_gold_in_list * 100 if n_gold_in_list else 0

    print(f"  Total mentions:              {total}")
    print(f"")

    # ══ HEADLINE: final end-to-end accuracy ══════════════════════════════
    # This is the one number comparable to BioLinkerAI and the baselines:
    # top-1 correct over all mentions, LLM included (if Phase 4 is on).
    if disambiguator:
        print(f"  ►► FINAL Accuracy@1 (with LLM):  {p4_acc:.1f}% "
              f"({p4_correct}/{total})")
    else:
        print(f"  ►► FINAL Accuracy@1 (no LLM, Phase 3):  {p3_acc:.1f}% "
              f"({p3_correct}/{total})")
    print(f"")

    # ── Per-phase Accuracy@1 (how each stage moves the top-1) ──
    print(f"  Accuracy@1 by pipeline stage (top-1 correct / all mentions):")
    print(f"    Phase 2  (retrieval):        {p2_acc:.1f}% ({p2_correct}/{total})")
    if expander is not None:
        print(f"    Phase 2b (expansion):        {p2b_acc:.1f}% "
              f"({p2b_correct}/{total})  {p2b_acc - p2_acc:+.1f}")
    print(f"    Phase 3  (domain rules):     {p3_acc:.1f}% ({p3_correct}/{total})"
          f"  {p3_acc - p2b_acc:+.1f}  (+{p3_improved_over_p2b}/-{p3_degraded_vs_p2b})")
    if disambiguator:
        print(f"    Phase 4  (LLM):              {p4_acc:.1f}% ({p4_correct}/{total})"
              f"  {p4_acc - p3_acc:+.1f}  (+{p4_improved_over_p3}/-{p4_degraded_vs_p3})")

    # ── Disambiguation accuracy (picker skill, gold-retrieved cases only) ──
    if n_gold_in_list > 0:
        print(f"")
        print(f"  Disambiguation Accuracy (of {n_gold_in_list} cases with gold in "
              f"top-{args.llm_top_k} list — isolates the picker):")
        print(f"    Phase 3 rules pick gold:     {disambig_p3:.1f}%")
        if disambiguator:
            print(f"    Phase 4 LLM   pick gold:     {disambig_p4:.1f}%  "
                  f"{disambig_p4 - disambig_p3:+.1f}")

    if disambiguator:
        print(f"")
        print(f"  LLM calls:                   {llm_calls}")
        print(f"  LLM skipped (confident):     {llm_skipped}")
        print(f"  LLM fallbacks (parse fail):  {llm_fallbacks}")
        if llm_skipped > 0:
            print(f"  Cascading threshold:         {args.phase4_threshold:.1f} "
                  f"(called {llm_calls}/{llm_calls + llm_skipped} = "
                  f"{llm_calls/(llm_calls + llm_skipped)*100:.0f}%)")
        # Token usage & cost summary
        print(disambiguator.get_usage_summary())

    # ── Failure-stage decomposition ──
    # Where do our errors come from? Only NED failures are winnable by a
    # better prompt or a better model; CG failures need better retrieval.
    total_failures = cg_failures + ned_failures
    if total_failures > 0:
        print(f"\n{'─' * 60}")
        print(f"  Failure stage (Phase 3, gold vs top-{args.llm_top_k} list):")
        print(f"    Candidate generation (gold not retrieved): "
              f"{cg_failures:5d}  ({cg_failures/total_failures*100:.1f}% of errors)")
        print(f"    Disambiguation (gold present, picked wrong): "
              f"{ned_failures:5d}  ({ned_failures/total_failures*100:.1f}% of errors)")
        print(f"    -> Max reachable by a perfect disambiguator: "
              f"{(p3_correct + ned_failures)/total*100:.1f}%")
        if disambiguator:
            recovered = ned_failures - p4_ned_failures
            share = recovered / ned_failures * 100 if ned_failures else 0
            print(f"    LLM recovered {recovered} of {ned_failures} "
                  f"winnable cases ({share:.1f}%)")

    # ── Candidate Recall@k (retrieval ceiling — NO LLM) ──
    # This measures ONLY candidate generation: how often the gold appears in
    # the top-k candidates. It is the ceiling for any disambiguator and is a
    # different axis from the Accuracy@1 above — do not read @5/@10 here as if
    # the LLM produced them. The LLM only ever outputs a single choice (@1);
    # a genuine LLM-Recall@k would need a ranking-output prompt (not built).
    print(f"\n{'─' * 60}")
    print(f"  Candidate Recall@k  (gold present in top-k candidates — NO LLM):")
    print(f"  {'k':>3}  {'Phase 2':>12}  {'Phase 2+2b':>12}  {'Phase 2b+3':>12}")
    for k in K_VALUES:
        p2_k = p2_at_k[k] / total * 100 if total > 0 else 0
        p2b_k = p2b_at_k[k] / total * 100 if total > 0 else 0
        p3_k = p3_at_k[k] / total * 100 if total > 0 else 0
        diff_str = f"(+{p2b_k - p2_k:.1f})" if p2b_k > p2_k else ""
        print(f"  {k:>3}  {p2_k:>11.1f}%  {p2b_k:>11.1f}%  {p3_k:>11.1f}%  {diff_str}")
    print(f"  (Recall@1 == Phase-3 Accuracy@1; Recall@k for k>1 is the ceiling, "
          f"not an LLM result.)")

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

    # ── Mini Disease Disambiguator diagnostics ──
    if mini_disambig is not None:
        print(f"\n{'─' * 60}")
        mini_disambig.print_stats()

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
    def_len_str = "unlimited" if args.max_definition_len == 0 else str(args.max_definition_len)
    print(f"  Max definition length:       {def_len_str}")
    mrdef_str = "ON" if args.mrdef else "OFF"
    print(f"  MRDEF definitions:           {mrdef_str}")
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

    # ── Debug LLM failures ──
    if args.debug_llm and debug_llm_failures:
        print(f"\n{'═' * 60}")
        print(f"DEBUG LLM FAILURES — Gold in list but LLM chose wrong ({len(debug_llm_failures)} cases)")
        print(f"{'═' * 60}")

        # Summary statistics
        gold_ranks = [f["gold_rank"] for f in debug_llm_failures]
        from collections import Counter
        rank_dist = Counter(gold_ranks)
        print(f"\n  Gold rank distribution in failed cases:")
        for rank in sorted(rank_dist.keys()):
            bar = "█" * rank_dist[rank]
            print(f"    Rank {rank:2d}: {rank_dist[rank]:3d} {bar}")

        llm_chose_top1 = sum(1 for f in debug_llm_failures if f["llm_chose_rank"] == 1)
        print(f"\n  LLM stuck with Phase 3 top-1: {llm_chose_top1}/{len(debug_llm_failures)} "
              f"({llm_chose_top1*100/len(debug_llm_failures):.0f}%)")

        type_dist = Counter(f["entity_type"] for f in debug_llm_failures)
        print(f"\n  Failures by entity type:")
        for et, cnt in type_dist.most_common(10):
            print(f"    {et}: {cnt}")

        # Show first N detailed failures
        n_show = min(20, len(debug_llm_failures))
        print(f"\n── First {n_show} detailed failures ──")
        for i, f in enumerate(debug_llm_failures[:n_show]):
            print(f'\n  [{i+1}] "{f["mention"]}" [{f["entity_type"]}] (PMID: {f["pmid"]})')
            print(f"      Gold: {f['gold_label']} [{f['gold_id']}] — rank {f['gold_rank']}, "
                  f"score {f['gold_score']:.1f}")
            if f["gold_definition"]:
                print(f"      Gold def: {f['gold_definition']}")
            print(f"      LLM chose: {f['llm_chose_label']} [{f['llm_chose_id']}] — "
                  f"rank {f['llm_chose_rank']}")
            print(f"      Phase 3 top-1: {f['top1_label']} [{f['top1_id']}] — "
                  f"score {f['top1_score']:.1f}")
            print(f"      LLM response: {f['llm_response'][:300]}")

        # Save full debug log to JSON
        debug_path = PROJECT_ROOT / "results" / "debug_llm_failures.json"
        debug_path.parent.mkdir(exist_ok=True)
        with open(debug_path, "w") as f:
            json.dump(debug_llm_failures, f, indent=2, ensure_ascii=False)
        print(f"\n  Full debug log saved to {debug_path} ({len(debug_llm_failures)} entries)")

    # ── Train learned ranker if requested ──
    if args.train_ranker and HAS_LEARNED_RANKER and ranker_train_features:
        print(f"\n{'═' * 60}")
        print("TRAINING LEARNED RANKER (XGBoost)")
        print(f"{'═' * 60}")

        ranker = LearnedRanker()
        ranker.train(
            all_features=ranker_train_features,
            all_labels=ranker_train_labels,
            n_estimators=args.ranker_n_estimators,
            max_depth=args.ranker_max_depth,
            learning_rate=args.ranker_lr,
        )

        model_path = str(
            PROJECT_ROOT / "src" / "improvements" / "cache" / "ranker_model.pkl"
        )
        ranker.save(model_path)
        print(f"\n  To use this model for evaluation, run:")
        print(f"    python3 src/evaluate_pipeline.py --learned-ranker {model_path} ...")

    # ── Save detailed log ──
    log_path = PROJECT_ROOT / "src" / "pipeline_evaluation_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "total": total,
            # What the accuracy below is a percentage OF. Without this the
            # numbers cannot be compared across runs or against the paper.
            "evaluation_setup": {
                "dataset": dataset_name,
                "split": getattr(args, "split", "test"),
                "deduplicated": not args.keep_duplicates,
                "n_all_mentions": n_all_mentions,
                "n_unique_pairs": n_unique_pairs,
                "limit": args.limit,
                "comparable_to_published": bool(args.keep_duplicates and not args.limit),
            },
            "phase4_setup": {
                "model": args.model,
                "prompt_version": args.prompt_version,
                "llm_top_k": args.llm_top_k,
                "temperature": args.temperature,
                "context_words": args.context_words,
                "retrieval_fewshot": args.retrieval_fewshot,
                "fewshot_k": args.fewshot_k if args.retrieval_fewshot else None,
                "structured_output": args.structured_output,
            } if disambiguator else None,
            # Picker skill, isolated from candidate generation: correct / (gold
            # was in the top-k list). Compare these across models/prompts.
            "disambiguation_accuracy": {
                "gold_in_list_count": n_gold_in_list,
                "phase3_rules": (
                    p3_correct / n_gold_in_list * 100 if n_gold_in_list else 0
                ),
                "phase4_llm": (
                    p4_correct / n_gold_in_list * 100 if n_gold_in_list else 0
                ) if disambiguator else None,
            },
            # Retrieval ceiling (no LLM): gold present in top-k candidates.
            "candidate_recall_at_k": {
                f"phase2b+3@{k}": (p3_at_k[k] / total * 100 if total else 0)
                for k in K_VALUES
            },
            # Parse failures fall back to the Phase 3 top-1, which makes
            # Phase 4 look like it did nothing. Track it explicitly.
            "llm_diagnostics": {
                "calls": llm_calls,
                "skipped_confident": llm_skipped,
                "parse_fallbacks": llm_fallbacks,
                "parse_fallback_rate": (
                    llm_fallbacks / llm_calls * 100 if llm_calls else 0
                ),
            } if disambiguator else None,
            "failure_stage": {
                "candidate_generation": cg_failures,
                "disambiguation": ned_failures,
                "disambiguation_after_llm": p4_ned_failures if disambiguator else None,
                "max_reachable_accuracy": (
                    (p3_correct + ned_failures) / total * 100 if total else 0
                ),
            },
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
    parser.add_argument("--split", choices=["train", "dev", "test"],
                        default="test", help="Data split to use (default: test)")

    # Phase 2 settings
    parser.add_argument("--backend", choices=["rapidfuzz", "elasticsearch"], default="rapidfuzz")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--top-k", type=int, default=10, help="Phase 2 candidates")
    parser.add_argument("--wikidata", action="store_true", help="Wikidata enrichment")
    parser.add_argument("--dbpedia", action="store_true", help="DBpedia enrichment")
    parser.add_argument("--mondo", action="store_true", help="MONDO disease ontology enrichment (improves Disease linking)")
    parser.add_argument("--umls", type=str, default=None, help="Path to MRCONSO.RRF")
    parser.add_argument("--mrdef", type=str, default=None,
                        help="Path to MRDEF.RRF for definition enrichment. Adds definitions "
                             "from UMLS (NCI, MSH, CSP, etc.) to entities that lack one — "
                             "especially supplementary concepts (324k have no MeSH scope note).")

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
    parser.add_argument("--sentence-context", action="store_true",
                        help="Enable sentence-context re-ranking for Disease entities "
                             "(requires --embedding)")
    parser.add_argument("--sentence-context-weight", type=float, default=5.0,
                        help="Weight for sentence-context similarity boost (default: 5.0)")
    parser.add_argument("--sentence-context-threshold", type=float, default=15.0,
                        help="Only apply sentence-context when top-1 vs top-2 gap < threshold (default: 15.0)")
    parser.add_argument("--mini-disambig", action="store_true",
                        help="Enable Mini Disease Disambiguator (lightweight LLM for "
                             "ambiguous Disease mentions)")
    parser.add_argument("--mini-disambig-model", type=str, default=None,
                        help="Model for Mini Disease Disambiguator (default: same as --model)")
    parser.add_argument("--mini-disambig-threshold", type=float, default=10.0,
                        help="Score gap threshold to trigger LLM disambiguation (default: 10.0)")
    parser.add_argument("--mini-disambig-max-candidates", type=int, default=3,
                        help="Max candidates to show the LLM (default: 3)")
    parser.add_argument("--mini-disambig-debug", action="store_true",
                        help="Show debug info for Mini Disease Disambiguator")
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
    parser.add_argument("--rule7-boost", type=float, default=1.5)
    parser.add_argument("--rule8-specificity-boost", type=float, default=2.0)
    parser.add_argument("--top1-protection-threshold", type=float, default=75.0)

    # Phase 4 settings
    parser.add_argument("--no-phase4", action="store_true", help="Skip Phase 4 (LLM)")
    parser.add_argument("--model", default="qwen3.5-9b", help="LLM model name")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="LLM API URL")
    parser.add_argument("--api-key", default="lm-studio",
                        help="API key for LLM endpoint (default: lm-studio for local)")
    parser.add_argument("--local-base-url", default="http://localhost:1234/v1",
                        help="Local LMStudio URL for abbreviation expander / mini disambiguator "
                             "(default: http://localhost:1234/v1)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--llm-top-k", type=int, default=15, help="Candidates to pass to LLM (default: 15)")
    parser.add_argument("--phase4-threshold", type=float, default=0.0,
                        help="Confidence-based cascading: only call LLM when score gap between "
                             "top-1 and top-2 is below this threshold. 0 = call LLM for all "
                             "(default: 0). Typical values: 10-20.")
    parser.add_argument("--max-tokens", type=int, default=8192,
                        help="Max tokens for LLM response (default: 8192). "
                             "Higher values allow more reasoning. Typical range: 4096-16384.")
    parser.add_argument("--phase4-no-think", action="store_true",
                        help="Disable thinking mode for Phase 4 LLM (/no_think). "
                             "Much faster (~5s vs ~50s) but potentially lower quality.")
    available_prompts = list_prompts()
    prompt_help = "Phase 4 prompt version. Available: " + ", ".join(
        f"{k} ({v})" for k, v in available_prompts.items()
    )
    parser.add_argument("--prompt-version", choices=list(available_prompts.keys()), default="v4",
                        help=prompt_help)
    parser.add_argument("--debug-llm", action="store_true",
                        help="Log detailed LLM failure analysis: for every case where "
                             "gold was in candidate list but LLM picked wrong, show "
                             "the gold rank, LLM choice, and LLM response. "
                             "Saved to results/debug_llm_failures.json")
    parser.add_argument("--max-definition-len", type=int, default=1000,
                        help="Max characters for candidate definitions in LLM prompt "
                             "(default: 1000). Set to 0 for unlimited. "
                             "Old default was 150, which truncated 46%% of definitions.")
    parser.add_argument("--shuffle-candidates", action="store_true",
                        help="Shuffle candidate order before passing to LLM "
                             "(tests position bias)")
    parser.add_argument("--shuffle-seed", type=int, default=42,
                        help="Random seed for candidate shuffling (default: 42)")
    parser.add_argument("--structured-output", action="store_true",
                        help="Constrain the LLM output to the candidate IDs of "
                             "each request via a per-request JSON-schema enum "
                             "(structured outputs). Eliminates hallucinated IDs "
                             "and parse fallbacks; replaces the text parser. "
                             "Auto-disables if the backend rejects it.")
    parser.add_argument("--context-words", type=int, default=None,
                        help="Size of the mention context window in words, split "
                             "evenly left/right. Overrides the prompt version's "
                             "own setting. 64 was measured as optimal by Ye & "
                             "Mitchell (ACL 2025). 0 = legacy sentence window.")
    parser.add_argument("--context-sentences", type=int, default=None,
                        help="Sentence-based context window: N sentences on each "
                             "side of the mention's sentence (N=1 => 3 sentences "
                             "total). Overrides --context-words. Use for the "
                             "sentence-vs-word ablation.")
    parser.add_argument("--dump-predictions", type=str, default=None, metavar="PATH",
                        help="Write per-mention correctness (p3/p4) to JSONL for "
                             "paired significance testing (see mcnemar_compare.py).")

    # Retrieval-based few-shot examples
    parser.add_argument("--retrieval-fewshot", action="store_true",
                        help="Replace the hard-coded few-shot examples with the k "
                             "most similar annotated mentions from the TRAINING "
                             "split (SapBERT + FAISS). Teaches the dataset's "
                             "annotation convention, which is what most Phase 4 "
                             "errors are actually about.")
    parser.add_argument("--fewshot-k", type=int, default=5,
                        help="Number of retrieved examples in the prompt (default: 5)")
    parser.add_argument("--fewshot-context-words", type=int, default=32,
                        help="Context words per retrieved example (default: 32)")
    parser.add_argument("--fewshot-exclude-exact", action="store_true",
                        help="Drop retrieved examples whose mention string equals "
                             "the query. Train/test documents are disjoint, so "
                             "these are legitimate supervision and usually the most "
                             "useful examples — use this only for a leakage-free "
                             "ablation.")

    # Learned Ranker (XGBoost)
    parser.add_argument("--train-ranker", action="store_true",
                        help="Train XGBoost ranker on current data split and save model")
    parser.add_argument("--learned-ranker", type=str, default=None,
                        help="Path to trained XGBoost ranker model (re-ranks after Phase 3)")
    parser.add_argument("--ranker-n-estimators", type=int, default=300,
                        help="XGBoost n_estimators (default: 300)")
    parser.add_argument("--ranker-max-depth", type=int, default=6,
                        help="XGBoost max_depth (default: 6)")
    parser.add_argument("--ranker-lr", type=float, default=0.1,
                        help="XGBoost learning rate (default: 0.1)")

    # Fine-tuning data export
    parser.add_argument("--dump-finetune", type=str, default=None, metavar="PATH",
                        help="Instead of evaluating, write chat-format SFT JSONL "
                             "for fine-tuning the Phase-4 model. Reuses the exact "
                             "inference prompt + candidate generation. Requires "
                             "--split train (refuses test). No LLM endpoint needed.")
    parser.add_argument("--finetune-max-per-pair", type=int, default=5,
                        help="Cap examples per (mention, gold) pair so frequent "
                             "mentions (seizures x99) don't dominate (default: 5).")
    parser.add_argument("--finetune-target", choices=["full", "id"], default="full",
                        help="Target output format: 'full' = {entity_name, mesh_id} "
                             "(matches v8 prompt), 'id' = {mesh_id} only "
                             "(matches --structured-output). Default: full.")

    # Evaluation settings
    parser.add_argument("--limit", type=int, default=None, help="Limit to N mentions")
    parser.add_argument("--keep-duplicates", action="store_true",
                        help="Evaluate on ALL mentions instead of unique "
                             "(mention, gold) pairs. This is what BioLinkerAI and "
                             "the other published baselines report — accuracy over "
                             "the total number of input mentions. Deduplicating "
                             "drops BC5CDR test from ~9.7k to 2625 mentions (27%%) "
                             "and removes mostly easy, frequently repeated ones, "
                             "so the default number is NOT comparable to the paper.")

    args = parser.parse_args()

    # Prompt-level overrides: mention context window.
    # --context-sentences wins over --context-words (mutually exclusive modes).
    if args.context_words is not None or args.context_sentences is not None:
        from prompts import PROMPT_REGISTRY
        cfg = PROMPT_REGISTRY[args.prompt_version]
        if args.context_sentences is not None:
            cfg.context_sentences = args.context_sentences
            cfg.context_words = 0
        elif args.context_words is not None:
            cfg.context_words = args.context_words
            cfg.context_sentences = 0

    run_evaluation(args)
