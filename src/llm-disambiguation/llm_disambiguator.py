"""
LLM Disambiguator
==================
Given a list of candidate entities and the document context, a LLM tries to select the best matching candidate.

The LLM receives:
  - The paper context (title and abstract)
  - The mention (surface form highlighted in its sentence. Additionally, 2 sentences before and after that)
  - A numbered list of candidate entities with labels, definitions, synonyms
  - An instruction to pick the best match

Connects to LMStudio (or any OpenAI-compatible API) via the openai package.

Usage:
    from llm_disambiguator import LLMDisambiguator

    disambiguator = LLMDisambiguator(
        model="qwen3-4b-2507",
        base_url="http://localhost:1234/v1")

    result = disambiguator.disambiguate(
        mention="...",
        candidates=[...],
        context="...",
        title="...")

    # result.mesh_id -> "..."
"""

import re
import time
import sys
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class DisambiguationResult:
    """
    Result of LLM disambiguation for a single mention.
    """
    mention: str # original mention text
    mesh_id: str # chosen by LLM
    preferred_label: str
    chosen_rank: int # -1 if LLM chose NONE
    confidence: str  # "llm" or "fallback"
    raw_response: str # for debugging purposes


# ── Prompt templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """
You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH (Medical Subject Headings) identifier.

You will be given:
1. A biomedical text (title and abstract of a paper)
2. A highlighted mention (the entity to link) inside the sentence, including 2 sentences before and after that mention
3. A numbered list of candidate MeSH entities, which was already ranked using domain specific rules

Your job: Select the candidate that best matches the mention IN CONTEXT. Consider:
- The meaning of the mention in its specific context
- Whether the candidate's definition fits the usage
- Synonyms and alternative names

IMPORTANT: Respond with ONLY the number of your chosen candidate (e.g., "1" or "3"). Nothing else. If none of the candidates match, respond with "NONE".
"""

def _build_user_prompt(
    mention: str,
    candidates: list,
    context: str,
    title: str = "",
    max_definition_len: int = 150,
    max_synonyms: int = 3,
) -> str:

    # Build context section
    parts = []
    parts.append("## Biomedical Text")
    if title:
        parts.append(f"**Title:** {title}")
    parts.append(f"**Text:** {context}")
    parts.append("")

    # Highlight the mention
    parts.append(f'## Mention to Link: "{mention}"')
    parts.append("")

    # Todo: Append also the sentence and the 2 before and after that one

    # Build candidate list
    parts.append("## Candidates")
    for i, c in enumerate(candidates, 1):
        # Label
        line = f"{i}. **{c.preferred_label}** [{c.mesh_id}]" # Todo: Check if this is enough. Can we add more?
        parts.append(line)

        # Definition (truncated)
        if c.definition:
            defn = c.definition
            if len(defn) > max_definition_len:
                defn = defn[:max_definition_len] + "..."
            parts.append(f"   Definition: {defn}")

        # Synonyms (top few, excluding the preferred label)
        other_syns = [s for s in c.synonyms if s.lower() != c.preferred_label.lower()]
        if other_syns:
            shown = other_syns[:max_synonyms]
            parts.append(f"   Synonyms: {', '.join(shown)}")

        # Score from domain specific rules
        parts.append(f"   Match score: {c.score:.1f}")
        parts.append("")

    parts.append("Which candidate best matches the mention in the given context? Reply with ONLY the number.")

    return "\n".join(parts)


def _parse_llm_response(response: str, num_candidates: int) -> int | None:
    text = response.strip()

    # Check for case NONE
    if text.upper() == "NONE":
        return None

    # Try to extract a number, handle cases like "1", "1.", "Candidate 1", "The best match is 1", etc.
    # First try: just a plain number
    if text.isdigit():
        num = int(text)
        if 1 <= num <= num_candidates:
            return num

    # Second try: find the first number in the response
    match = re.search(r'\b(\d+)\b', text)
    if match:
        num = int(match.group(1))
        if 1 <= num <= num_candidates:
            return num

    return None


# ── Main disambiguator class ──────────────────────────────────────────────

class LLMDisambiguator:
    """
    Uses an LLM to disambiguate between candidate entities.

    Connects to an OpenAI-compatible API (e.g., LMStudio).
    """

    def __init__(
        self,
        model: str = "qwen3-4b-2507",
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.0,
        max_tokens: int = 32,
        timeout: float = 30.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LMStudio doesn't need a real key
            timeout=timeout,
        )

        # Verify connection
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"Connected to LLM API at {base_url}")
            print(f"  Available models: {model_ids}")
            if model not in model_ids:
                print(f"  Warning: model '{model}' not in list. " # Todo: Fix, because its qwen/qwen3-4b-2507 in lm studio
                      f"LMStudio might use a different ID.")
        except Exception as e:
            print(f"Warning: Could not connect to LLM API at {base_url}: {e}")
            print("Make sure LMStudio is running with the server enabled.")

    def disambiguate(
        self,
        mention: str,
        candidates: list,
        context: str,
        title: str = "",
    ) -> DisambiguationResult:
        """
        This is the interface for the pipeline.
        """
        if not candidates:
            return DisambiguationResult(
                mention=mention,
                mesh_id="NONE",
                preferred_label="",
                chosen_rank=-1,
                confidence="fallback",
                raw_response="",
            )

        user_prompt = _build_user_prompt(mention, candidates, context, title)

        # Call LLM
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = response.choices[0].message.content.strip()
        except Exception as e:
            # API error — fall back to top-1 candidate
            print(f"  LLM API error for '{mention}': {e}")
            return DisambiguationResult(
                mention=mention,
                mesh_id=candidates[0].mesh_id,
                preferred_label=candidates[0].preferred_label,
                chosen_rank=1,
                confidence="fallback",
                raw_response=f"ERROR: {e}",
            )

        # Parse response
        chosen_idx = _parse_llm_response(raw, len(candidates))

        if chosen_idx is not None:
            chosen = candidates[chosen_idx - 1]
            return DisambiguationResult(
                mention=mention,
                mesh_id=chosen.mesh_id,
                preferred_label=chosen.preferred_label,
                chosen_rank=chosen_idx,
                confidence="llm",
                raw_response=raw,
            )
        else:
            # Parse failed or NONE — fall back to top-1
            return DisambiguationResult(
                mention=mention,
                mesh_id=candidates[0].mesh_id,
                preferred_label=candidates[0].preferred_label,
                chosen_rank=1,
                confidence="fallback",
                raw_response=raw,
            )

    def disambiguate_batch(
        self,
        items: list[dict],
        top_k: int = 5,
    ) -> list[DisambiguationResult]:
        """
       Uses the disambiguate() method for a batch of mentions.
        """
        results = []
        total = len(items)

        for i, item in enumerate(items):
            # Limit candidates to top_k for the LLM
            candidates = item["candidates"][:top_k]

            result = self.disambiguate(
                mention=item["mention"],
                candidates=candidates,
                context=item["context"],
                title=item.get("title", ""),
            )
            results.append(result)

            if (i + 1) % 50 == 0:
                print(f"  ... {i+1}/{total} mentions disambiguated", flush=True)

        return results