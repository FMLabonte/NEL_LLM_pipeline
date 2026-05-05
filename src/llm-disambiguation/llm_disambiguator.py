"""
LLM Disambiguator
==================
Phase 4 of the BioLinkerAI pipeline.

Given a list of candidate entities (from Phase 2) and the document context,
uses an LLM to select the best matching entity for a given mention.

The LLM receives:
  - The paper context (title + abstract)
  - The mention (surface form highlighted in its sentence)
  - A numbered list of candidate entities with labels, definitions, synonyms
  - An instruction to pick the best match

Connects to LMStudio (or any OpenAI-compatible API) via the openai package.

Usage:
    from llm_disambiguator import LLMDisambiguator

    disambiguator = LLMDisambiguator(
        model="qwen3-4b-2507",
        base_url="http://localhost:1234/v1",
    )

    result = disambiguator.disambiguate(
        mention="seizures",
        candidates=candidates,       # list[CandidateEntity] from Phase 2
        context="Famotidine-induced seizures were observed...",
        title="Adverse effects of famotidine",
    )
    # result.mesh_id -> "D012640"
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

    Attributes
    ----------
    mention : str
        The original mention text.
    mesh_id : str
        The MeSH ID chosen by the LLM.
    preferred_label : str
        The preferred label of the chosen entity.
    chosen_rank : int
        The rank (1-based) of the chosen candidate in the original list.
        -1 if the LLM chose NONE or parsing failed.
    confidence : str
        "llm" if the LLM made a valid choice, "fallback" if we fell back
        to the top-1 candidate due to parse errors.
    raw_response : str
        The raw LLM response text (for debugging).
    """
    mention: str
    mesh_id: str
    preferred_label: str
    chosen_rank: int
    confidence: str  # "llm" or "fallback"
    raw_response: str


# ── Prompt templates ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a biomedical entity linking expert. Your task is to link entity mentions in biomedical texts to the correct MeSH (Medical Subject Headings) identifier.

You will be given:
1. A biomedical text (title and abstract of a paper)
2. A highlighted mention (the entity to link)
3. A numbered list of candidate MeSH entities

Your job: Select the candidate that best matches the mention IN CONTEXT. Consider:
- The meaning of the mention in its specific context
- Whether the candidate's definition fits the usage
- Synonyms and alternative names

IMPORTANT: Respond with ONLY the number of your chosen candidate (e.g., "1" or "3"). Nothing else. If none of the candidates match, respond with "NONE"."""


def _build_user_prompt(
    mention: str,
    candidates: list,
    context: str,
    title: str = "",
    max_definition_len: int = 150,
    max_synonyms: int = 3,
) -> str:
    """
    Build the user prompt for the LLM.

    Parameters
    ----------
    mention : str
        The entity mention to disambiguate.
    candidates : list[CandidateEntity]
        Ranked candidates from Phase 2.
    context : str
        The document text (abstract or sentence) containing the mention.
    title : str
        The paper title.
    max_definition_len : int
        Max characters for definition (truncated with "...").
    max_synonyms : int
        Max number of synonyms to show per candidate.
    """
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

    # Build candidate list
    parts.append("## Candidates")
    for i, c in enumerate(candidates, 1):
        # Label
        line = f"{i}. **{c.preferred_label}** [{c.mesh_id}]"
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

        # Score from Phase 2
        parts.append(f"   Match score: {c.score:.1f}")
        parts.append("")

    parts.append("Which candidate best matches the mention in the given context? Reply with ONLY the number.")

    return "\n".join(parts)


def _parse_llm_response(response: str, num_candidates: int) -> int | None:
    """
    Parse the LLM response to extract the chosen candidate number.

    Returns
    -------
    int or None
        The 1-based candidate index, or None if parsing failed / NONE chosen.
    """
    text = response.strip()

    # Check for NONE
    if text.upper() == "NONE":
        return None

    # Try to extract a number — handle cases like "1", "1.", "Candidate 1",
    # "The best match is 1", etc.
    # First try: just a plain number
    if text.isdigit():
        num = int(text)
        if 1 <= num <= num_candidates:
            return num

    # Second try: find first number in the response
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

    Parameters
    ----------
    model : str
        Model name as shown in LMStudio (e.g., "qwen3-4b-2507").
    base_url : str
        API base URL (e.g., "http://localhost:1234/v1").
    temperature : float
        Sampling temperature. 0 for deterministic output.
    max_tokens : int
        Max tokens in LLM response (should be small — we only need a number).
    timeout : float
        Request timeout in seconds.
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
                print(f"  Warning: model '{model}' not in list. "
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
        Disambiguate a mention using the LLM.

        Parameters
        ----------
        mention : str
            The entity mention text.
        candidates : list[CandidateEntity]
            Ranked candidates from Phase 2.
        context : str
            The document text containing the mention.
        title : str
            The paper title.

        Returns
        -------
        DisambiguationResult
            The LLM's choice with metadata.
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

        # Build prompt
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
        Disambiguate a batch of mentions.

        Parameters
        ----------
        items : list[dict]
            Each dict has keys: mention, candidates, context, title.
        top_k : int
            Only pass top_k candidates to the LLM (to keep prompts short).

        Returns
        -------
        list[DisambiguationResult]
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


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Disambiguator Demo")
    parser.add_argument("--model", default="qwen3-4b-2507", help="Model name in LMStudio")
    parser.add_argument("--base-url", default="http://localhost:1234/v1", help="API base URL")
    args = parser.parse_args()

    disambiguator = LLMDisambiguator(model=args.model, base_url=args.base_url)

    # Quick test with a fake candidate list
    from dataclasses import dataclass as dc, field as f

    @dc
    class FakeCandidate:
        mesh_id: str
        preferred_label: str
        synonyms: list = f(default_factory=list)
        definition: str = ""
        score: float = 0.0

    candidates = [
        FakeCandidate("D012640", "Seizures", ["Convulsions", "Epileptic seizure"], "A sudden onset of excessive activity in the brain.", 95.0),
        FakeCandidate("D004827", "Epilepsy", ["Seizure disorder"], "A brain disorder involving repeated seizures.", 80.0),
        FakeCandidate("D013575", "Syncope", ["Fainting"], "Transient loss of consciousness.", 60.0),
    ]

    result = disambiguator.disambiguate(
        mention="seizures",
        candidates=candidates,
        context="The patient experienced recurrent seizures after administration of the drug.",
        title="Adverse neurological effects of famotidine",
    )

    print(f"\nResult: [{result.mesh_id}] {result.preferred_label}")
    print(f"  Chosen rank: {result.chosen_rank}")
    print(f"  Confidence: {result.confidence}")
    print(f"  Raw response: '{result.raw_response}'")
