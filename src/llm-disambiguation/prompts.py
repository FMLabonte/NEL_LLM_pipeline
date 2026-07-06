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
    max_definition_len: int = 150
    max_synonyms: int = 3


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
    """
    parts = []

    # ── Context section ──
    parts.append("## Biomedical Text")
    if title:
        parts.append(f"**Title:** {title}")

    label = "Abstract" if config.use_full_context else "Text"
    parts.append(f"**{label}:** {context}")
    parts.append("")

    # ── Mention with context window (2 sentences before/after) ──
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

        # Category from MeSH tree numbers
        if config.include_categories:
            tree_numbers = getattr(c, "tree_numbers", [])
            if tree_numbers:
                cats = sorted({
                    _TREE_CATS.get(tn[0], tn[0])
                    for tn in tree_numbers if tn
                })
                parts.append(f"   Category: {', '.join(cats)}")

        # Definition (truncated)
        if c.definition:
            defn = c.definition
            if len(defn) > config.max_definition_len:
                defn = defn[:config.max_definition_len] + "..."
            parts.append(f"   Definition: {defn}")

        # Synonyms
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
    if config.include_few_shot:
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


def _parse_json_entity(text: str, candidates: list) -> int | None:
    """Parse a JSON response like {"entity_name": "...", "mesh_id": "..."}."""
    # Extract JSON object (handles markdown code blocks, extra text)
    json_match = re.search(r'\{[^}]+\}', text)
    if not json_match:
        return None

    try:
        data = json.loads(json_match.group())
    except json.JSONDecodeError:
        return None

    # Validate with Pydantic if available
    if HAS_PYDANTIC and EntityLinkResponse is not None:
        try:
            parsed = EntityLinkResponse(**data)
            entity_name = parsed.entity_name
            mesh_id = parsed.mesh_id
        except Exception:
            entity_name = data.get("entity_name", "")
            mesh_id = data.get("mesh_id", "")
    else:
        entity_name = str(data.get("entity_name", ""))
        mesh_id = str(data.get("mesh_id", ""))

    if not entity_name and not mesh_id:
        return None

    # 1) Match by mesh_id (exact)
    for i, c in enumerate(candidates, 1):
        if c.mesh_id == mesh_id:
            return i

    # 2) Match by entity_name (exact, case-insensitive)
    name_lower = entity_name.lower().strip()
    for i, c in enumerate(candidates, 1):
        if c.preferred_label.lower().strip() == name_lower:
            return i

    # 3) Fuzzy: substring match on entity_name
    for i, c in enumerate(candidates, 1):
        label_lower = c.preferred_label.lower()
        if name_lower in label_lower or label_lower in name_lower:
            return i

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

def _extract_mention_window(
    context: str, mention: str, n_sentences: int = 2,
) -> str:
    """
    Extract the sentence containing the mention plus n sentences
    before and after. The mention is highlighted with **markers**.
    """
    sentences = re.split(r'(?<=[.!?])\s+', context.strip())
    if not sentences:
        return context

    mention_lower = mention.lower()
    mention_idx = None
    for i, sent in enumerate(sentences):
        if mention_lower in sent.lower():
            mention_idx = i
            break

    if mention_idx is None:
        return context.replace(mention, f"**{mention}**", 1)

    start = max(0, mention_idx - n_sentences)
    end = min(len(sentences), mention_idx + n_sentences + 1)
    window = " ".join(sentences[start:end])

    idx = window.lower().find(mention_lower)
    if idx >= 0:
        original = window[idx:idx + len(mention)]
        window = window[:idx] + f"**{original}**" + window[idx + len(mention):]

    return window
