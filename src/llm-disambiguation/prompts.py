"""
Prompt Registry for LLM Disambiguation
========================================
Manages different prompt versions for Phase 4 (LLM entity linking).

Each version defines:
  - System prompt (role instruction)
  - How to build the user prompt (context, candidates, examples)
  - How to parse the LLM response (number vs JSON)
  - Settings (full context, few-shot, categories, etc.)

Add a new prompt version by:
  1. Create a PromptConfig with your settings
  2. Add it to PROMPT_REGISTRY
  3. Done — the disambiguator picks it up via --prompt-version

Usage:
    from prompts import get_prompt_config, build_user_prompt, parse_response

    config = get_prompt_config("v3")
    prompt = build_user_prompt(config, mention, candidates, context, title)
    chosen_idx = parse_response(config, llm_output, candidates)
"""

import json
import re
from dataclasses import dataclass


# ── Optional: Pydantic for structured response validation ────────────────

try:
    from pydantic import BaseModel
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

if HAS_PYDANTIC:
    class EntityLinkResponse(BaseModel):
        """Structured LLM response for entity linking."""
        entity_name: str
        mesh_id: str
else:
    EntityLinkResponse = None


# ── Prompt configuration ─────────────────────────────────────────────────

@dataclass
class PromptConfig:
    """
    Self-contained configuration for a prompt version.

    Controls every aspect of prompt construction and response parsing.
    """
    name: str
    description: str
    system_prompt: str
    response_format: str        # "number" or "json_entity"
    use_full_context: bool      # True = full abstract, False = sentence snippet
    include_few_shot: bool
    include_categories: bool
    include_match_score: bool
    allow_none: bool            # True = LLM can answer "NONE"
    ask_reasoning: bool         # True = ask for CoT reasoning before answer
    max_definition_len: int = 1000
    max_synonyms: int = 3
    include_semantic_type: bool = False  # show MeSH tree-derived semantic type per candidate
    context_words: int = 0      # >0 = word-based mention window of N words
                                # (Ye & Mitchell, ACL 2025, found 64 optimal);
                                # 0 = legacy sentence-based window
    context_sentences: int = 0  # >0 = sentence-based window: N sentences on each
                                # side of the mention's sentence. Takes priority
                                # over context_words. 0 = off.


# ── Prompt versions ──────────────────────────────────────────────────────

PROMPT_REGISTRY: dict[str, PromptConfig] = {

    # ── V1: Original prompt (used in early BC5CDR runs) ──────────────────
    "v1": PromptConfig(
        name="v1",
        description="Original: reasoning + number, NONE option, sentence context",
        system_prompt="""\
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH (Medical Subject Headings) identifier.

You will be given:
1. A biomedical text (title and abstract of a paper)
2. A highlighted mention (the entity to link) inside the sentence, including 2 sentences before and after that mention
3. A numbered list of candidate MeSH entities, which was already ranked using domain specific rules

Your job: Select the candidate that best matches the mention IN CONTEXT. Consider:
- The meaning of the mention in its specific context
- Whether the candidate's definition fits the usage
- Synonyms and alternative names

Think step by step: First, identify what the mention refers to in this context. Then, compare it against the candidates and pick the best match.

Respond with your reasoning in 1-2 sentences, then on a new line write ONLY the number of your chosen candidate (e.g., "1" or "3"). If none of the candidates match, write "NONE".""",
        response_format="number",
        use_full_context=False,
        include_few_shot=False,
        include_categories=False,
        include_match_score=True,
        allow_none=True,
        ask_reasoning=True,
    ),

    # ── V2: Concise prompt (current default) ─────────────────────────────
    "v2": PromptConfig(
        name="v2",
        description="Concise: number only, few-shot examples, categories, sentence context",
        system_prompt="""\
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH/UMLS identifier.

You will receive a mention highlighted in context and a ranked list of candidate entities. Carefully consider ALL candidates before choosing.

Focus on:
- The specific meaning of the mention in its sentence context (e.g., "cold" as temperature vs. common cold)
- Whether the candidate's definition and semantic category match the usage
- Synonyms and alternative names that link the mention to a candidate
- For abbreviations: which candidate's full name matches the acronym

Respond with ONLY the candidate number (e.g., "1" or "3"). No explanation needed.""",
        response_format="number",
        use_full_context=False,
        include_few_shot=True,
        include_categories=True,
        include_match_score=True,
        allow_none=False,
        ask_reasoning=False,
    ),

    # ── V3: Structured JSON output with entity names (T.A. suggestion) ───
    "v3": PromptConfig(
        name="v3",
        description="JSON output with entity name, full abstract context, few-shot, no scores",
        system_prompt="""\
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH/UMLS concept.

You will receive:
- The full abstract of a biomedical paper
- A highlighted mention with surrounding sentences
- A list of candidate entities with definitions and synonyms

Select the candidate whose meaning best matches how the mention is used in context. Consider the semantic category, definition, synonyms, and abbreviation expansions.

Respond in JSON format exactly like this:
{"entity_name": "Cystic Fibrosis", "mesh_id": "D003550"}

Rules:
- entity_name must exactly match one of the candidate names listed
- mesh_id must exactly match that candidate's identifier
- Output ONLY the JSON object, nothing else""",
        response_format="json_entity",
        use_full_context=True,
        include_few_shot=True,
        include_categories=True,
        include_match_score=False,
        allow_none=False,
        ask_reasoning=False,
    ),

    # ── V5: Enhanced v1 — reasoning + specificity fix + few-shot ────────
    # Based on v1 (our best MedMentions result: +6.8%), enhanced with:
    #   - Specificity line to prevent parent→child flip degradations
    #   - Few-shot examples from v2
    #   - Categories from v2
    #   - Explicit trust-the-ranking signal
    "v5": PromptConfig(
        name="v5",
        description="Enhanced v1: reasoning + specificity fix + few-shot + categories",
        system_prompt="""\
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH (Medical Subject Headings) identifier.

You will be given:
1. A biomedical text (title and abstract of a paper)
2. A highlighted mention (the entity to link) inside the sentence, including 2 sentences before and after that mention
3. A numbered list of candidate MeSH entities, pre-ranked by a domain-specific system. Higher match scores indicate stronger matches.

Your job: Select the candidate that best matches the mention IN CONTEXT. Consider:
- The meaning of the mention in its specific context
- Whether the candidate's definition fits the usage
- Synonyms and alternative names
- The semantic category of the candidate

IMPORTANT — Specificity rule: Always pick the concept at the SAME level of specificity as the mention. If the mention says "anxiety", pick the general concept (Anxiety), NOT a more specific subtype (Anxiety Disorders, Social Anxiety). If the mention says "breast cancer", pick the specific concept, not just "Neoplasms". Only choose a more specific or more general concept if the context explicitly supports it.

IMPORTANT — Trust the ranking: The candidates are pre-ranked by domain rules. The top-ranked candidate (highest match score) is correct in ~80% of cases. Only override it if you have strong contextual evidence that a lower-ranked candidate is a better match.

Think step by step: First, identify what the mention refers to in this context. Then, compare it against the candidates and pick the best match.

Respond with your reasoning in 1-2 sentences, then on a new line write ONLY the number of your chosen candidate (e.g., "1" or "3"). If none of the candidates match, write "NONE".""",
        response_format="number",
        use_full_context=False,
        include_few_shot=True,
        include_categories=True,
        include_match_score=True,
        allow_none=True,
        ask_reasoning=True,
    ),

    # ── V6: v5 reasoning + JSON output (0 parse fails) ─────────────────
    # Combines v5's strengths (reasoning, specificity, trust-ranking)
    # with JSON output format (0 parse fails vs 4-7% with number format).
    # Key insight: reasoning boosts accuracy, but number parsing loses it.
    "v6": PromptConfig(
        name="v6",
        description="v5 reasoning + specificity + JSON output (best of v5 format + v4 parsing)",
        system_prompt="""\
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH (Medical Subject Headings) identifier.

You will be given:
1. A biomedical text (title and abstract of a paper)
2. A highlighted mention (the entity to link) inside the sentence, including 2 sentences before and after that mention
3. A list of candidate MeSH entities, pre-ranked by a domain-specific system. Higher match scores indicate stronger matches.

Your job: Select the candidate that best matches the mention IN CONTEXT. Consider:
- The meaning of the mention in its specific context
- Whether the candidate's definition fits the usage
- Synonyms and alternative names
- The semantic category of the candidate

IMPORTANT — Specificity rule: Always pick the concept at the SAME level of specificity as the mention. If the mention says "anxiety", pick the general concept (Anxiety), NOT a more specific subtype (Anxiety Disorders, Social Anxiety). If the mention says "breast cancer", pick the specific concept, not just "Neoplasms". Only choose a more specific or more general concept if the context explicitly supports it.

IMPORTANT — Trust the ranking: The candidates are pre-ranked by domain rules. The top-ranked candidate (highest match score) is correct in ~80% of cases. Only override it if you have strong contextual evidence that a lower-ranked candidate is a better match.

Think step by step: First, identify what the mention refers to in this context. Then, compare it against the candidates and pick the best match.

Respond with 1-2 sentences of reasoning, then on a NEW LINE output ONLY a JSON object:
{"entity_name": "<candidate name>", "mesh_id": "<candidate ID>"}

Example response:
The mention "CF" in the context of lung disease and CFTR mutations refers to cystic fibrosis, not other CF abbreviations.
{"entity_name": "Cystic Fibrosis", "mesh_id": "D003550"}

Rules:
- entity_name must EXACTLY match one of the candidate names listed
- mesh_id must EXACTLY match that candidate's identifier
- Always end with the JSON object on its own line""",
        response_format="json_entity",
        use_full_context=False,
        include_few_shot=True,
        include_categories=True,
        include_match_score=True,
        allow_none=False,
        ask_reasoning=True,
    ),

    # ── V7: BioLinkerAI replication — simple prompt, semantic types ──────
    # Replicates BioLinkerAI's prompt style:
    #   - Short system prompt, no reasoning
    #   - Candidates listed with SemanticType (from MeSH tree numbers)
    #   - No synonyms, no match scores
    #   - Shorter definitions (200 chars)
    #   - Full abstract context
    #   - JSON output for reliable parsing
    "v7": PromptConfig(
        name="v7",
        description="BioLinkerAI replication: simple prompt, semantic types, full context, JSON output",
        system_prompt="""\
You are a biomedical entity linking expert. Given an entity mention in a biomedical text and a list of candidate concepts, select the best matching candidate.

Consider the description and semantic type of the candidates with respect to the context of the input text to make the decision.

Respond with ONLY a JSON object: {"entity_name": "<name>", "mesh_id": "<id>"}""",
        response_format="json_entity",
        use_full_context=True,
        include_few_shot=False,
        include_categories=False,
        include_match_score=False,
        allow_none=False,
        ask_reasoning=False,
        max_definition_len=200,
        max_synonyms=0,
        include_semantic_type=True,
    ),

    # ── V8: literature-informed baseline ─────────────────────────────────
    # Built from the findings in Ye & Mitchell (ACL 2025) plus our own error
    # analysis. Differences to v6:
    #   - no forced chain-of-thought: the simplest prompt beat both CoT and a
    #     reasoning model across all five datasets in their study, and our own
    #     degradations are over-specification, which reasoning makes worse
    #   - 64-word window around the mention (their measured optimum)
    #   - keeps match scores (Phase 3 ranking is a strong prior)
    #   - designed to be run with --retrieval-fewshot: the retrieved training
    #     examples carry the dataset's annotation convention
    "v8": PromptConfig(
        name="v8",
        description="Literature-informed: no CoT, 64-word window, scores, retrieval few-shots",
        system_prompt="""\
You are a biomedical entity linking expert. You are given a mention highlighted in its context and a list of candidate concepts, pre-ranked by a domain-specific system.

Select the candidate that best matches the mention in that context.

Two rules that matter more than they look:

1. MATCH THE LEVEL OF SPECIFICITY. Pick the concept at the same granularity as the mention. "anxiety" is Anxiety, not Anxiety Disorders. "kidney injury" is Kidney Diseases, not Acute Kidney Injury. "encephalopathy" is Brain Diseases, not a named specific encephalopathy. Only go more specific when the text explicitly names the specific condition.

2. TRUST THE RANKING. The top-ranked candidate is correct in roughly 80% of cases. Override it only when the context gives you clear evidence for a different candidate — not merely a plausible one.

Respond with ONLY a JSON object, no reasoning, no other text:
{"entity_name": "<candidate name>", "mesh_id": "<candidate ID>"}

entity_name and mesh_id must be copied exactly from one of the listed candidates.""",
        response_format="json_entity",
        use_full_context=False,
        include_few_shot=True,
        include_categories=True,
        include_match_score=True,
        allow_none=False,
        ask_reasoning=False,
        context_words=64,
    ),

    # ── V6-full: v6 with full abstract context ──────────────────────────
    # Same as v6 but uses the full abstract instead of 200-char window.
    # Tests whether more context helps the LLM.
    "v6-full": PromptConfig(
        name="v6-full",
        description="v6 with full abstract context (tests context window effect)",
        system_prompt="""\
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH (Medical Subject Headings) identifier.

You will be given:
1. A biomedical text (title and full abstract of a paper)
2. A highlighted mention (the entity to link) with surrounding context
3. A list of candidate MeSH entities, pre-ranked by a domain-specific system. Higher match scores indicate stronger matches.

Your job: Select the candidate that best matches the mention IN CONTEXT. Consider:
- The meaning of the mention in its specific context
- Whether the candidate's definition fits the usage
- Synonyms and alternative names
- The semantic category of the candidate

IMPORTANT — Specificity rule: Always pick the concept at the SAME level of specificity as the mention. If the mention says "anxiety", pick the general concept (Anxiety), NOT a more specific subtype (Anxiety Disorders, Social Anxiety). If the mention says "breast cancer", pick the specific concept, not just "Neoplasms". Only choose a more specific or more general concept if the context explicitly supports it.

IMPORTANT — Trust the ranking: The candidates are pre-ranked by domain rules. The top-ranked candidate (highest match score) is correct in ~80% of cases. Only override it if you have strong contextual evidence that a lower-ranked candidate is a better match.

Think step by step: First, identify what the mention refers to in this context. Then, compare it against the candidates and pick the best match.

Respond with 1-2 sentences of reasoning, then on a NEW LINE output ONLY a JSON object:
{"entity_name": "<candidate name>", "mesh_id": "<candidate ID>"}

Example response:
The mention "CF" in the context of lung disease and CFTR mutations refers to cystic fibrosis, not other CF abbreviations.
{"entity_name": "Cystic Fibrosis", "mesh_id": "D003550"}

Rules:
- entity_name must EXACTLY match one of the candidate names listed
- mesh_id must EXACTLY match that candidate's identifier
- Always end with the JSON object on its own line""",
        response_format="json_entity",
        use_full_context=True,
        include_few_shot=True,
        include_categories=True,
        include_match_score=True,
        allow_none=False,
        ask_reasoning=True,
    ),

    # ── V4: Best-of-v2+v3 — JSON output, scores, sentence context ──────
    "v4": PromptConfig(
        name="v4",
        description="JSON output + match scores + sentence context (best of v2 & v3)",
        system_prompt="""\
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH/UMLS identifier.

You will receive a mention highlighted in context and a ranked list of candidate entities. The candidates are pre-ranked by a domain-specific system — the match scores reflect this ranking. Carefully consider ALL candidates, but note that higher-scored candidates are more likely correct.

Focus on:
- The specific meaning of the mention in its sentence context (e.g., "cold" as temperature vs. common cold)
- Whether the candidate's definition and semantic category match the usage
- Synonyms and alternative names that link the mention to a candidate
- For abbreviations: which candidate's full name matches the acronym

Respond in JSON format exactly like this:
{"entity_name": "Cystic Fibrosis", "mesh_id": "D003550"}

Rules:
- entity_name must exactly match one of the candidate names listed
- mesh_id must exactly match that candidate's identifier
- Output ONLY the JSON object, nothing else""",
        response_format="json_entity",
        use_full_context=False,
        include_few_shot=True,
        include_categories=True,
        include_match_score=True,
        allow_none=False,
        ask_reasoning=False,
    ),
}


# ── Shared constants ─────────────────────────────────────────────────────

_TREE_CATS = {
    "A": "Anatomy", "B": "Organisms", "C": "Diseases",
    "D": "Chemicals and Drugs", "E": "Techniques", "F": "Psychology",
    "G": "Phenomena", "N": "Health Care",
}

_FEW_SHOT_EXAMPLES = [
    ('"hypertension" in "patients with hypertension and diabetes"',
     "Hypertensive disease [D006973]", "not Ocular Hypertension"),
    ('"AD" in "a mouse model of AD"',
     "Alzheimer Disease [D000544]", "not Autonomic Dysreflexia"),
    ('"lithium" in "lithium treatment for bipolar disorder"',
     "Lithium [D008094] (the element/drug)", "not Lithium Compounds"),
]


# ── Public API ───────────────────────────────────────────────────────────

def list_prompts() -> dict[str, str]:
    """Return {name: description} for all registered prompt versions."""
    return {k: v.description for k, v in PROMPT_REGISTRY.items()}


def get_prompt_config(version: str) -> PromptConfig:
    """Get a prompt configuration by version name."""
    if version not in PROMPT_REGISTRY:
        available = ", ".join(PROMPT_REGISTRY.keys())
        raise ValueError(
            f"Unknown prompt version '{version}'. Available: {available}"
        )
    return PROMPT_REGISTRY[version]


# ── User prompt builder ──────────────────────────────────────────────────

def build_user_prompt(
    config: PromptConfig,
    mention: str,
    candidates: list,
    context: str,
    title: str = "",
    mention_start: int = -1,
    dynamic_examples: list | None = None,
) -> str:
    """
    Build the user prompt for the LLM based on the prompt config.

    Parameters
    ----------
    config : PromptConfig
        Which prompt version to use.
    mention : str
        The entity mention to disambiguate.
    candidates : list[CandidateEntity]
        Ranked candidates (already optionally shuffled by the caller).
    context : str
        Document text — full abstract if config.use_full_context,
        or a sentence snippet otherwise.
    title : str
        The paper title.
    mention_start : int
        Character offset of the mention within `context`. Pass the gold
        annotation offset when available; -1 triggers a word-boundary search.
    dynamic_examples : list[dict] | None
        Retrieved few-shot examples, each {mention, context, label, id}.
        When given, these replace the hard-coded examples.
    """
    parts = []

    # ── Context section ──
    # With a word-based window we deliberately do NOT dump the surrounding
    # text as well: the window IS the context, and repeating the abstract
    # around it reintroduces exactly the noise the window is meant to remove.
    parts.append("## Biomedical Text")
    if title:
        parts.append(f"**Title:** {title}")

    bounded = config.context_words > 0 or config.context_sentences > 0
    if not bounded:
        label = "Abstract" if config.use_full_context else "Text"
        parts.append(f"**{label}:** {context}")
    parts.append("")

    # ── Mention with context window ──
    # Priority: sentence window > word window > legacy 2-sentence window.
    if config.context_sentences > 0:
        mention_window = _extract_mention_window(
            context, mention, n_sentences=config.context_sentences,
            start=mention_start,
        )
    elif config.context_words > 0:
        mention_window = _extract_word_window(
            context, mention, n_words=config.context_words, start=mention_start,
        )
    else:
        mention_window = _extract_mention_window(context, mention, n_sentences=2)
    parts.append(f'## Mention to Link: "{mention}"')
    parts.append(f"**Context window:** {mention_window}")
    parts.append("")

    # ── Candidate list ──
    parts.append("## Candidates")
    for i, c in enumerate(candidates, 1):
        # Number-based vs name-based listing
        if config.response_format == "json_entity":
            line = f"- **{c.preferred_label}** [{c.mesh_id}]"
        else:
            line = f"{i}. **{c.preferred_label}** [{c.mesh_id}]"
        parts.append(line)

        # Category from MeSH tree numbers (standard format)
        if config.include_categories and not config.include_semantic_type:
            tree_numbers = getattr(c, "tree_numbers", [])
            if tree_numbers:
                cats = sorted({
                    _TREE_CATS.get(tn[0], tn[0])
                    for tn in tree_numbers if tn
                })
                parts.append(f"   Category: {', '.join(cats)}")

        # Semantic type (BioLinkerAI-style, derived from MeSH tree numbers)
        if config.include_semantic_type:
            tree_numbers = getattr(c, "tree_numbers", [])
            if tree_numbers:
                cats = sorted({
                    _TREE_CATS.get(tn[0], "")
                    for tn in tree_numbers if tn
                })
                sem_type = ", ".join(c_name for c_name in cats if c_name)
                if sem_type:
                    parts.append(f"   Semantic type: {sem_type}")

        # Definition (optionally truncated; 0 = unlimited)
        if c.definition:
            defn = c.definition
            if config.max_definition_len > 0 and len(defn) > config.max_definition_len:
                defn = defn[:config.max_definition_len] + "..."
            parts.append(f"   Definition: {defn}")

        # Synonyms (skipped when max_synonyms=0)
        if config.max_synonyms > 0:
            other_syns = [
                s for s in c.synonyms
                if s.lower() != c.preferred_label.lower()
            ]
            if other_syns:
                shown = other_syns[:config.max_synonyms]
                parts.append(f"   Synonyms: {', '.join(shown)}")

        # Match score (optional — hidden in v3 to avoid position bias)
        if config.include_match_score:
            parts.append(f"   Match score: {c.score:.1f}")

        parts.append("")

    # ── Final instruction ──
    if config.response_format == "json_entity":
        parts.append("Which candidate best matches the mention in context?")
        parts.append(
            'Respond with ONLY a JSON object: '
            '{"entity_name": "<name>", "mesh_id": "<id>"}'
        )
    else:
        parts.append(
            "Which candidate best matches the mention in the given context? "
            "Reply with ONLY the number."
        )

    # ── Few-shot examples ──
    # Retrieved examples take priority over the hard-coded ones: they come
    # from the training split and carry the dataset's ANNOTATION CONVENTION,
    # which is what most disambiguation errors are actually about (BC5CDR
    # gold is usually the broader MeSH descriptor, e.g. "encephalopathy" →
    # Brain Diseases, not a semantically tighter subconcept).
    if dynamic_examples:
        parts.append("")
        parts.append(
            "Examples from annotated data — note how specific the chosen "
            "concept is relative to the mention, and follow the same convention:"
        )
        for ex in dynamic_examples:
            target = (
                f'{ex["label"]} [{ex["id"]}]' if ex.get("label") else f'[{ex["id"]}]'
            )
            parts.append(
                f'- "{ex["mention"]}" in "{ex["context"]}" → {target}'
            )
    elif config.include_few_shot:
        parts.append("")
        parts.append("Examples of correct linking:")
        for mention_ex, correct, wrong in _FEW_SHOT_EXAMPLES:
            parts.append(f"- {mention_ex} → {correct}, {wrong}")

    return "\n".join(parts)


# ── Response parsing ─────────────────────────────────────────────────────

def _strip_thinking_tags(response: str) -> str:
    """Strip <think>...</think> blocks from Qwen3.5 model output."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def parse_response(
    config: PromptConfig,
    response: str,
    candidates: list,
) -> int | None:
    """
    Parse the LLM response and return the 1-based candidate index.

    Returns None if parsing fails (caller should fall back to top-1).
    """
    text = _strip_thinking_tags(response).strip()

    if config.response_format == "json_entity":
        return _parse_json_entity(text, candidates)
    else:
        return _parse_number(text, len(candidates), config.allow_none)


# Matches a JSON object, tolerating one level of nesting.
_JSON_OBJ_RE = re.compile(r'\{(?:[^{}]|\{[^{}]*\})*\}')

# Placeholder values from the prompt template — if a model echoes the format
# spec before answering, we must not mistake the template for the answer.
_PLACEHOLDER_RE = re.compile(r'^\s*<.*>\s*$')


def _normalize_id(raw: str) -> str:
    """Normalize a MeSH/UMLS id: strip vocabulary prefixes, upper-case."""
    s = str(raw).strip().upper()
    for prefix in ("MESH:", "MSH:", "UMLS:", "MESHID:", "ID:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    return s


def _normalize_label(raw: str) -> str:
    """Normalize an entity label for comparison: lowercase, collapse spaces."""
    return re.sub(r'\s+', ' ', str(raw).strip().lower())


def _parse_json_entity(text: str, candidates: list) -> int | None:
    """
    Parse a JSON response like {"entity_name": "...", "mesh_id": "..."}.

    Scans the response back-to-front: prompt versions that ask for reasoning
    first (v6, v6-full) put the real answer LAST, and models frequently echo
    the format template earlier in the response. Taking the first match would
    return the template and silently fall back to the Phase 3 top-1.
    """
    entity_name = mesh_id = ""

    for raw in reversed(_JSON_OBJ_RE.findall(text)):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        # Validate with Pydantic if available
        if HAS_PYDANTIC and EntityLinkResponse is not None:
            try:
                parsed = EntityLinkResponse(**data)
                cand_name, cand_id = parsed.entity_name, parsed.mesh_id
            except Exception:
                cand_name = data.get("entity_name", "")
                cand_id = data.get("mesh_id", "")
        else:
            cand_name = data.get("entity_name", "")
            cand_id = data.get("mesh_id", "")

        cand_name, cand_id = str(cand_name or ""), str(cand_id or "")

        # Skip the format template: {"entity_name": "<name>", "mesh_id": "<id>"}
        if _PLACEHOLDER_RE.match(cand_name) or _PLACEHOLDER_RE.match(cand_id):
            continue
        if not cand_name and not cand_id:
            continue

        entity_name, mesh_id = cand_name, cand_id
        break

    if not entity_name and not mesh_id:
        return None

    # 1) Exact id match
    for i, c in enumerate(candidates, 1):
        if c.mesh_id == mesh_id:
            return i

    # 2) Normalized id match ("MESH:D003866", "d003866", ...)
    norm_id = _normalize_id(mesh_id)
    if norm_id:
        for i, c in enumerate(candidates, 1):
            if _normalize_id(c.mesh_id) == norm_id:
                return i

    # 3) Exact label match (case/whitespace-insensitive)
    norm_name = _normalize_label(entity_name)
    if norm_name:
        for i, c in enumerate(candidates, 1):
            if _normalize_label(c.preferred_label) == norm_name:
                return i

        # 4) Label match ignoring punctuation
        stripped = re.sub(r'[^a-z0-9 ]', '', norm_name)
        for i, c in enumerate(candidates, 1):
            if re.sub(r'[^a-z0-9 ]', '', _normalize_label(c.preferred_label)) == stripped:
                return i

        # 5) Substring fallback — pick the CLOSEST match, not the first one.
        #    Taking the first match systematically favours whichever candidate
        #    happens to rank higher, e.g. answering "Depressive Disorder" would
        #    select "Depressive Disorder, Major" if that sits above it.
        best_i, best_delta = None, None
        for i, c in enumerate(candidates, 1):
            label = _normalize_label(c.preferred_label)
            if norm_name in label or label in norm_name:
                delta = abs(len(label) - len(norm_name))
                if best_delta is None or delta < best_delta:
                    best_i, best_delta = i, delta
        if best_i is not None:
            return best_i

    return None


def _parse_number(
    text: str, num_candidates: int, allow_none: bool = False,
) -> int | None:
    """Parse a number-based response (e.g., "1", "3", or after CoT reasoning)."""
    if allow_none and text.upper() == "NONE":
        return None

    # Try last line first (CoT reasoning appears before the answer)
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        if last_line.isdigit():
            num = int(last_line)
            if 1 <= num <= num_candidates:
                return num
        match = re.search(r'\b(\d+)\b', last_line)
        if match:
            num = int(match.group(1))
            if 1 <= num <= num_candidates:
                return num

    # Fallback: plain number
    if text.isdigit():
        num = int(text)
        if 1 <= num <= num_candidates:
            return num

    # Fallback: last number in full response
    matches = re.findall(r'\b(\d+)\b', text)
    if matches:
        num = int(matches[-1])
        if 1 <= num <= num_candidates:
            return num

    return None


# ── Helpers ──────────────────────────────────────────────────────────────

def find_mention(text: str, mention: str, from_pos: int = 0) -> int:
    """
    Find `mention` in `text` on a word boundary. Returns -1 if not found.

    Plain str.find() matches substrings inside longer words — "Cr" inside
    "increased", "dex" inside "dexamethasone", "AL" inside "renal". Those are
    overwhelmingly abbreviations, i.e. precisely the mentions whose linking
    depends on getting the right context.

    Boundaries are alphanumeric-only so that mentions containing punctuation
    or digits (e.g. "5-FU", "TNF-alpha", "vitamin D3") still match.
    """
    if not mention:
        return -1
    pattern = (
        r'(?<![A-Za-z0-9])' + re.escape(mention.strip()) + r'(?![A-Za-z0-9])'
    )
    m = re.search(pattern, text[from_pos:], re.IGNORECASE)
    return from_pos + m.start() if m else -1


def _extract_word_window(
    context: str, mention: str, n_words: int = 64, start: int = -1,
) -> str:
    """
    Take `n_words` around the mention, split evenly left and right.

    This is the window Ye & Mitchell (ACL 2025) measured as optimal for LLM
    disambiguation — tighter than a full abstract (which adds noise) and
    wider than a single clause (which loses the disambiguating signal).

    `start` should be the gold annotation offset when available; otherwise
    the mention is located on a word boundary.
    """
    if start is None or start < 0 or not _offset_matches(context, mention, start):
        start = find_mention(context, mention)
    if start < 0:
        start = context.lower().find(mention.lower())
    if start < 0:
        return " ".join(context.split()[:n_words])

    half = max(1, n_words // 2)
    left_words = context[:start].split()[-half:]
    right_words = context[start + len(mention):].split()[:half]
    surface = context[start:start + len(mention)]

    return " ".join(left_words + [f"**{surface}**"] + right_words)


def _offset_matches(text: str, mention: str, start: int) -> bool:
    """True if `start` really points at `mention` inside `text`."""
    if start < 0 or start + len(mention) > len(text):
        return False
    return text[start:start + len(mention)].lower() == mention.lower()


def _extract_mention_window(
    context: str, mention: str, n_sentences: int = 2, start: int = -1,
) -> str:
    """
    Extract the sentence containing the mention plus n sentences before and
    after (so n_sentences=2 => up to 5 sentences). The mention is highlighted
    with **markers**. `start` is the gold character offset — when given, it
    pins the correct sentence even for repeated mentions or abbreviations.
    """
    context = context.strip()
    sentences = re.split(r'(?<=[.!?])\s+', context)
    if not sentences:
        return context

    mention_idx = None

    # Best: map the gold offset to the sentence that contains it.
    if (start is not None and 0 <= start < len(context)
            and context[start:start + len(mention)].lower() == mention.lower()):
        pos = 0
        for i, sent in enumerate(sentences):
            j = context.find(sent, pos)
            if j < 0:
                j = pos
            if j <= start < j + len(sent):
                mention_idx = i
                break
            pos = j + len(sent)

    # Otherwise locate on a WORD BOUNDARY. A plain substring search matches
    # "dex" inside "dexamethasone" or "AL" inside "renal", which puts the
    # window around a position where the mention does not actually occur.
    if mention_idx is None:
        for i, sent in enumerate(sentences):
            if find_mention(sent, mention) >= 0:
                mention_idx = i
                break

    if mention_idx is None:  # fall back to a loose search
        for i, sent in enumerate(sentences):
            if mention.lower() in sent.lower():
                mention_idx = i
                break

    if mention_idx is None:
        return context.replace(mention, f"**{mention}**", 1)

    win_start = max(0, mention_idx - n_sentences)
    win_end = min(len(sentences), mention_idx + n_sentences + 1)
    win_sents = list(sentences[win_start:win_end])

    # Highlight the mention inside ITS sentence — not the first occurrence in
    # the whole window, which for a repeated mention marks the wrong instance.
    local = mention_idx - win_start
    sent = win_sents[local]
    idx = find_mention(sent, mention)
    if idx < 0:
        idx = sent.lower().find(mention.lower())
    if idx >= 0:
        win_sents[local] = (
            sent[:idx] + f"**{sent[idx:idx + len(mention)]}**"
            + sent[idx + len(mention):]
        )
    return " ".join(win_sents)
