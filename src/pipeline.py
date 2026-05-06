"""
BioLinkerAI Reproduction Pipeline
=================================
Main pipeline that chains all 4 phases:
  Phase 1: Linguistic Rules (Entity Extraction)
  Phase 2: Candidate Generation (MeSH index + fuzzy matching)
  Phase 3: Domain-Specific Rules (Re-Ranking)
  Phase 4: LLM Disambiguation

Each phase is implemented in its own module under src/.
This file orchestrates the full pipeline and provides
both a gold-entity mode (for evaluation) and a raw-text
mode (using Phase 1 for entity extraction).

Usage:
    from pipeline import BioLinkerPipeline

    pipeline = BioLinkerPipeline(mesh_xml="Data/MeSH/desc2026.xml")
    results = pipeline.link_text(
        "Famotidine-induced seizures were observed in the patient.",
    )
    # Or with gold entities:
    results = pipeline.link_entities(
        gold_entities=[{"text": "seizures", "entity_type": "Disease"}],
        context="Famotidine-induced seizures were observed in the patient.",
        title="Adverse effects of famotidine",
    )

Paper reference: Section 3, full pipeline overview
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field

# ── Path setup ────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "domain-rules"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "llm-disambiguation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "linguistic-rules"))

from mesh_index import MeSHIndex
from candidate_retriever import CandidateRetriever
from domain_rules import DomainRuleReranker
from llm_disambiguator import LLMDisambiguator


# ── Result data class ─────────────────────────────────────────────────────

@dataclass
class LinkingResult:
    """
    Result of entity linking for a single mention.

    Attributes
    ----------
    mention : str
        The original mention text.
    entity_type : str or None
        Entity type (e.g., "Disease", "Chemical") if known.
    mesh_id : str
        The predicted MeSH ID.
    preferred_label : str
        The preferred label of the predicted entity.
    phase2_top1 : str
        MeSH ID of the Phase 2 top-1 candidate (before re-ranking).
    phase3_top1 : str
        MeSH ID of the Phase 3 top-1 candidate (after domain rules).
    phase4_choice : str
        MeSH ID chosen by the LLM (Phase 4), or same as phase3_top1 if Phase 4 is off.
    confidence : str
        "llm" if the LLM made the final choice, "phase3" if Phase 4 was skipped,
        "fallback" if LLM parsing failed.
    candidates : list
        The candidate list after Phase 3 re-ranking.
    """
    mention: str
    entity_type: str | None
    mesh_id: str
    preferred_label: str
    phase2_top1: str
    phase3_top1: str
    phase4_choice: str
    confidence: str
    candidates: list = field(default_factory=list, repr=False)


# ── Main pipeline class ──────────────────────────────────────────────────

class BioLinkerPipeline:
    """
    Full BioLinkerAI pipeline: Phase 1 → Phase 2 → Phase 3 → Phase 4.

    Parameters
    ----------
    mesh_xml : str
        Path to MeSH descriptor XML (desc2026.xml).
    supp_xml : str or None
        Path to MeSH supplementary XML (supp2026.xml).
    backend : str
        Search backend for Phase 2 ("rapidfuzz" or "elasticsearch").
    es_url : str
        Elasticsearch URL (only used if backend="elasticsearch").
    top_k : int
        Number of candidates to retrieve in Phase 2.
    enrich_wikidata : bool
        Whether to enrich with Wikidata synonyms.
    enrich_dbpedia : bool
        Whether to enrich with DBpedia synonyms.
    enrich_umls : str or None
        Path to MRCONSO.RRF for UMLS enrichment (None = skip).
    use_phase3 : bool
        Whether to apply Phase 3 domain rules (default: True).
    use_phase4 : bool
        Whether to apply Phase 4 LLM disambiguation (default: True).
    llm_model : str
        LLM model name for Phase 4.
    llm_base_url : str
        LLM API base URL for Phase 4.
    llm_top_k : int
        Number of candidates to pass to the LLM.
    rule5_boost : float
        Phase 3 Rule 5 weight.
    rule6_penalty : float
        Phase 3 Rule 6 weight.
    rule7_boost : float
        Phase 3 Rule 7 weight.
    """

    def __init__(
        self,
        mesh_xml: str | None = None,
        supp_xml: str | None = None,
        backend: str = "rapidfuzz",
        es_url: str = "http://localhost:9200",
        top_k: int = 10,
        enrich_wikidata: bool = False,
        enrich_dbpedia: bool = False,
        enrich_umls: str | None = None,
        use_phase3: bool = True,
        use_phase4: bool = True,
        llm_model: str = "qwen3-4b-2507",
        llm_base_url: str = "http://localhost:1234/v1",
        llm_top_k: int = 5,
        rule5_boost: float = 2.0,
        rule6_penalty: float = -30.0,
        rule7_boost: float = 3.0,
    ):
        self.top_k = top_k
        self.llm_top_k = llm_top_k
        self.use_phase3 = use_phase3
        self.use_phase4 = use_phase4

        # ── Phase 1: Linguistic Entity Extractor (lazy init) ──
        self._extractor = None

        # ── Phase 2: MeSH Index + Candidate Retriever ──
        self.index = MeSHIndex(backend=backend, es_url=es_url)
        if mesh_xml:
            self.index.build_from_xml(
                descriptor_path=mesh_xml,
                supplementary_path=supp_xml or str(
                    Path(mesh_xml).parent / "supp2026.xml"
                ),
                enrich_wikidata=enrich_wikidata,
                enrich_dbpedia=enrich_dbpedia,
                enrich_umls=enrich_umls,
            )
        self.retriever = CandidateRetriever(self.index, top_k=top_k)

        # ── Phase 3: Domain Rule Reranker ──
        wikidata, dbpedia, umls = self._load_enrichment_caches()
        self.reranker = DomainRuleReranker(
            mesh_index=self.index,
            rule5_boost=rule5_boost,
            rule6_penalty=rule6_penalty,
            rule7_boost=rule7_boost,
            wikidata_synonyms=wikidata,
            dbpedia_synonyms=dbpedia,
            umls_synonyms=umls,
        )

        # ── Phase 4: LLM Disambiguator ──
        self.disambiguator = None
        if use_phase4:
            try:
                self.disambiguator = LLMDisambiguator(
                    model=llm_model,
                    base_url=llm_base_url,
                )
            except Exception as e:
                print(f"Warning: Could not initialize LLM disambiguator: {e}")
                print("Phase 4 will be skipped (falling back to Phase 3 output).")
                self.use_phase4 = False

    def _load_enrichment_caches(self) -> tuple[dict, dict, dict]:
        """Load cached enrichment data for Phase 3 Rule 5."""
        cache_dir = PROJECT_ROOT / "src" / "candidate-generation" / "cache"
        wikidata, dbpedia, umls = {}, {}, {}

        for name, var in [("wikidata_cache.json", "wikidata"),
                          ("dbpedia_cache.json", "dbpedia"),
                          ("umls_cache.json", "umls")]:
            path = cache_dir / name
            if path.exists():
                with open(path, "r") as f:
                    data = json.load(f)
                if var == "wikidata":
                    wikidata = data
                elif var == "dbpedia":
                    dbpedia = data
                else:
                    umls = data

        return wikidata, dbpedia, umls

    @property
    def extractor(self):
        """Lazy-load Phase 1 extractor (requires spaCy model)."""
        if self._extractor is None:
            from entity_extractor import LinguisticEntityExtractor
            self._extractor = LinguisticEntityExtractor()
        return self._extractor

    # ── Phase 1: Entity Extraction (raw text mode) ────────────────────────

    def extract_entities(self, text: str) -> list[dict]:
        """
        Phase 1: Extract entity mentions from raw text using linguistic rules.

        Parameters
        ----------
        text : str
            Raw biomedical text.

        Returns
        -------
        list[dict]
            Each dict has: text, start, end, tokens.
        """
        mentions = self.extractor.extract(text)
        return [
            {
                "text": m.text,
                "start": m.start,
                "end": m.end,
                "tokens": m.tokens,
                "entity_type": None,  # Phase 1 does not assign entity types
            }
            for m in mentions
        ]

    # ── Full pipeline: raw text ──────────────────────────────────────────

    def link_text(
        self,
        text: str,
        title: str = "",
    ) -> list[LinkingResult]:
        """
        Run the full pipeline on raw text (Phase 1 → 2 → 3 → 4).

        Parameters
        ----------
        text : str
            Raw biomedical text (e.g., a PubMed abstract).
        title : str
            Optional paper title (used as context for Phase 4).

        Returns
        -------
        list[LinkingResult]
            One result per extracted entity mention.
        """
        # Phase 1: Extract entities
        mentions = self.extract_entities(text)

        # Link each extracted mention
        return self.link_entities(
            gold_entities=mentions,
            context=text,
            title=title,
        )

    # ── Full pipeline: gold entities ─────────────────────────────────────

    def link_entities(
        self,
        gold_entities: list[dict],
        context: str = "",
        title: str = "",
    ) -> list[LinkingResult]:
        """
        Run Phase 2 → 3 → 4 on pre-annotated entities (skip Phase 1).

        Parameters
        ----------
        gold_entities : list[dict]
            Each dict has: text (mention surface form), entity_type (optional).
        context : str
            Document text (abstract) for context-based rules and LLM prompting.
        title : str
            Paper title for Phase 4 LLM prompting.

        Returns
        -------
        list[LinkingResult]
            One result per entity mention.
        """
        results = []

        for entity in gold_entities:
            mention = entity["text"]
            entity_type = entity.get("entity_type", None)

            result = self.link_single(
                mention=mention,
                entity_type=entity_type,
                context=context,
                title=title,
            )
            results.append(result)

        return results

    def link_single(
        self,
        mention: str,
        entity_type: str | None = None,
        context: str = "",
        title: str = "",
    ) -> LinkingResult:
        """
        Link a single mention through Phase 2 → 3 → 4.

        Parameters
        ----------
        mention : str
            The entity surface form.
        entity_type : str or None
            Entity type for Phase 3 semantic filtering.
        context : str
            Document text for context-based rules.
        title : str
            Paper title for LLM prompting.

        Returns
        -------
        LinkingResult
        """
        # ── Phase 2: Candidate Generation ──
        candidates = self.retriever.retrieve(mention, top_k=self.top_k)

        if not candidates:
            return LinkingResult(
                mention=mention,
                entity_type=entity_type,
                mesh_id="NONE",
                preferred_label="",
                phase2_top1="NONE",
                phase3_top1="NONE",
                phase4_choice="NONE",
                confidence="no_candidates",
            )

        phase2_top1 = candidates[0].mesh_id

        # ── Phase 3: Domain-Specific Rules (Re-Ranking) ──
        if self.use_phase3:
            candidates = self.reranker.rerank(
                mention=mention,
                candidates=candidates,
                entity_type=entity_type,
                context=context,
            )

        phase3_top1 = candidates[0].mesh_id

        # ── Phase 4: LLM Disambiguation ──
        if self.use_phase4 and self.disambiguator is not None:
            llm_candidates = candidates[:self.llm_top_k]
            llm_result = self.disambiguator.disambiguate(
                mention=mention,
                candidates=llm_candidates,
                context=context,
                title=title,
            )
            final_mesh_id = llm_result.mesh_id
            final_label = llm_result.preferred_label
            phase4_choice = llm_result.mesh_id
            confidence = llm_result.confidence
        else:
            final_mesh_id = phase3_top1
            final_label = candidates[0].preferred_label
            phase4_choice = phase3_top1
            confidence = "phase3"

        return LinkingResult(
            mention=mention,
            entity_type=entity_type,
            mesh_id=final_mesh_id,
            preferred_label=final_label,
            phase2_top1=phase2_top1,
            phase3_top1=phase3_top1,
            phase4_choice=phase4_choice,
            confidence=confidence,
            candidates=candidates,
        )


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("BioLinkerAI Pipeline — Quick Demo")
    print("=" * 60)

    pipeline = BioLinkerPipeline(
        mesh_xml=str(PROJECT_ROOT / "Data" / "MeSH" / "desc2026.xml"),
        supp_xml=str(PROJECT_ROOT / "Data" / "MeSH" / "supp2026.xml"),
        use_phase4=False,  # demo without LLM
    )

    # Gold entity mode
    print("\n── Gold Entity Mode (Phase 2 → 3) ──")
    results = pipeline.link_entities(
        gold_entities=[
            {"text": "seizures", "entity_type": "Disease"},
            {"text": "famotidine", "entity_type": "Chemical"},
        ],
        context="Famotidine-induced seizures were observed in the patient.",
        title="Adverse effects of famotidine",
    )

    for r in results:
        print(f'  "{r.mention}" → [{r.mesh_id}] {r.preferred_label}')
        print(f"    Phase 2: {r.phase2_top1}, Phase 3: {r.phase3_top1}")
        print(f"    Confidence: {r.confidence}")
