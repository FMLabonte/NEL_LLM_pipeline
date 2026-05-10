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

Think step by step: First, identify what the mention refers to in this context. Then, compare it against the candidates and pick the best match.

Respond with your reasoning in 1-2 sentences, then on a new line write ONLY the number of your chosen candidate (e.g., "1" or "3"). If none of the candidates match, write "NONE".
"""

def _extract_mention_window(context: str, mention: str, n_sentences: int = 2) -> str:
    """
    Extract the sentence containing the mention plus n sentences before and after.
    The mention is highlighted with **markers** in the output.
    """
    import re as _re

    # Split context into sentences (handles ". ", "? ", "! " and end-of-string)
    sentences = _re.split(r'(?<=[.!?])\s+', context.strip())
    if not sentences:
        return context

    # Find which sentence contains the mention (case-insensitive)
    mention_lower = mention.lower()
    mention_idx = None
    for i, sent in enumerate(sentences):
        if mention_lower in sent.lower():
            mention_idx = i
            break

    if mention_idx is None:
        # Mention isn't found in sentences, return full context with highlight
        highlighted = context.replace(mention, f"**{mention}**", 1)
        return highlighted

    # Extract window: n sentences before + mention sentence + n sentences after
    start = max(0, mention_idx - n_sentences)
    end = min(len(sentences), mention_idx + n_sentences + 1)
    window = sentences[start:end]

    # Highlight the mention in the relevant sentence
    window_text = " ".join(window)
    # Case-preserving highlight
    idx = window_text.lower().find(mention_lower)
    if idx >= 0:
        original = window_text[idx:idx + len(mention)]
        window_text = window_text[:idx] + f"**{original}**" + window_text[idx + len(mention):]

    return window_text


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

    # Extract mention with surrounding sentences (2 before + 2 after)
    mention_window = _extract_mention_window(context, mention, n_sentences=2)
    parts.append(f'## Mention to Link: "{mention}"')
    parts.append(f"**Context window:** {mention_window}")
    parts.append("")

    # Build candidate list
    parts.append("## Candidates")
    for i, c in enumerate(candidates, 1):
        # Label
        line = f"{i}. **{c.preferred_label}** [{c.mesh_id}]"
        parts.append(line)

        # Semantic category from MeSH tree numbers (e.g., "Diseases", "Chemicals and Drugs")
        tree_numbers = getattr(c, "tree_numbers", [])
        if tree_numbers:
            _TREE_CATS = {"A": "Anatomy", "B": "Organisms", "C": "Diseases",
                          "D": "Chemicals and Drugs", "E": "Techniques", "F": "Psychology",
                          "G": "Phenomena", "N": "Health Care"}
            cats = sorted({_TREE_CATS.get(tn[0], tn[0]) for tn in tree_numbers if tn})
            parts.append(f"   Category: {', '.join(cats)}")

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


def _strip_thinking_tags(response: str) -> str:
    """
    Strip <think>...</think> blocks from model output.
    Qwen3.5 and similar models emit thinking tokens wrapped in these tags.
    """
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def _parse_llm_response(response: str, num_candidates: int) -> int | None:
    # Strip thinking tags (Qwen3.5 outputs <think>...</think> before answer)
    text = _strip_thinking_tags(response).strip()

    # Check for case NONE
    if text.upper() == "NONE":
        return None

    # Try to extract a number from the LAST line (CoT reasoning comes first)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if lines:
        last_line = lines[-1]
        # If last line is just a number
        if last_line.isdigit():
            num = int(last_line)
            if 1 <= num <= num_candidates:
                return num
        # If last line contains a number
        match = re.search(r'\b(\d+)\b', last_line)
        if match:
            num = int(match.group(1))
            if 1 <= num <= num_candidates:
                return num

    # Fallback: find any number in the full response
    # First try: just a plain number
    if text.isdigit():
        num = int(text)
        if 1 <= num <= num_candidates:
            return num

    # Second try: find the last number in the response (most likely the answer)
    matches = re.findall(r'\b(\d+)\b', text)
    if matches:
        num = int(matches[-1])
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
        model: str = "qwen3.5-9b",
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.6,
        max_tokens: int = 512,
        timeout: float = 60.0,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",  # LMStudio doesn't need a real key
            timeout=timeout,
        )

        # Verify connection and auto-detect model name
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"Connected to LLM API at {base_url}")
            print(f"  Available models: {model_ids}")
            if model not in model_ids:
                # LMStudio often uses "org/model" format (e.g., "qwen/qwen3-4b-2507")
                # Auto-detect: if exactly one model is loaded, use that
                if len(model_ids) == 1:
                    self.model = model_ids[0]
                    print(f"  Auto-detected model: '{self.model}'")
                else:
                    # Try partial match (e.g., "qwen3-4b" matches "qwen/qwen3-4b-2507")
                    matches = [m for m in model_ids if model in m or m in model]
                    if len(matches) == 1:
                        self.model = matches[0]
                        print(f"  Auto-detected model: '{self.model}'")
                    else:
                        print(f"  Warning: model '{model}' not in list. Using first available.")
                        self.model = model_ids[0] if model_ids else model
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