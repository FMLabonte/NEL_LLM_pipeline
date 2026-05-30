"""
LLM-Based Abbreviation Expander
=================================
Uses an LLM to expand biomedical abbreviations that the rule-based
AbbreviationExpander cannot resolve.

The rule-based expander handles two cases well:
  1. Parenthetical definitions: "acute myocardial infarction (AMI)"
  2. Known dictionary entries: "COPD" → chronic obstructive pulmonary disease

But it fails when:
  - The abbreviation is not defined in the abstract text
  - The abbreviation is not in the hardcoded dictionary
  - The abbreviation is context-dependent (e.g., "MS" could be
    "multiple sclerosis" or "mass spectrometry")

This module fills that gap by asking an LLM (via OpenAI-compatible API,
e.g., LMStudio with Qwen3) to expand the abbreviation using:
  - The biomedical context (title + abstract)
  - The LLM's training knowledge of biomedical abbreviations

Integration:
  - Used as a fallback AFTER the rule-based expander
  - Activated with --llm-abbreviation flag in evaluate_pipeline.py
  - Uses the same API endpoint as Phase 4 (LLM disambiguation)

Usage:
    from llm_abbreviation_expander import LLMAbbreviationExpander

    expander = LLMAbbreviationExpander(
        model="qwen3-8b",
        base_url="http://localhost:1234/v1",
    )

    expansion = expander.expand(
        mention="IDM",
        context="The IDM showed signs of hypoglycemia at birth...",
        title="Neonatal outcomes in diabetic pregnancies",
    )
    # → "infant of diabetic mother"
"""

import re
import time
from dataclasses import dataclass, field

from openai import OpenAI


# ── Prompt template ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a biomedical terminology expert. Expand the given abbreviation.

Rules:
- Return ONLY the expanded form, nothing else
- No explanation, no punctuation, no quotes
- If unsure, respond: UNKNOWN

Examples:
- AMI → acute myocardial infarction
- COPD → chronic obstructive pulmonary disease
- 5-FU → fluorouracil
- HTN → hypertension
- CRC → colorectal cancer\
"""


def _build_expansion_prompt(
    mention: str,
    context: str,
    title: str = "",
) -> str:
    """
    Build the user prompt for abbreviation expansion.

    Includes /no_think to disable Qwen3's thinking mode — this is a
    simple lookup task that doesn't benefit from chain-of-thought.
    Without /no_think, Qwen3 spends all tokens on <think> tags and
    returns empty content.
    """
    parts = []

    # /no_think disables Qwen3 thinking mode for this simple task
    parts.append("/no_think")
    parts.append("")

    if title:
        parts.append(f"Title: {title}")

    # Trim context to a reasonable window around the mention
    if context:
        mention_lower = mention.lower()
        pos = context.lower().find(mention_lower)
        if pos >= 0:
            start = max(0, pos - 150)
            end = min(len(context), pos + len(mention) + 150)
            snippet = context[start:end].strip()
            if start > 0:
                snippet = "..." + snippet
            if end < len(context):
                snippet = snippet + "..."
            parts.append(f"Context: {snippet}")
        else:
            parts.append(f"Context: {context[:300]}...")

    parts.append(f"Abbreviation: {mention}")

    return "\n".join(parts)


def _strip_thinking_tags(response: str) -> str:
    """Strip <think>...</think> blocks from model output (Qwen3 style)."""
    return re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL).strip()


def _extract_answer_from_reasoning(reasoning: str) -> str | None:
    """
    Extract the actual expansion from Qwen3's reasoning_content.

    When thinking mode is active, the content field is empty and
    the full response (including the answer) is in reasoning_content.
    The structure is typically:

        Thinking Process:
        1. Analyze the Request: ...
        2. Identify the Abbreviation: ...
        3. Conclusion: <the actual answer>

    Or sometimes the answer appears after markers like:
        "→", "Answer:", "Expansion:", "stands for", "the expansion is"

    We extract the answer by scanning from the end for the actual term.
    """
    if not reasoning:
        return None

    text = reasoning.strip()

    # Strategy 1: Look for explicit answer markers
    answer_markers = [
        r'(?:answer|expansion|result|output|response)\s*[:=]\s*(.+)',
        r'(?:stands for|refers to|expands to|full form is)\s+(.+)',
        r'→\s*(.+)',
        r'\*\*(.+?)\*\*\s*$',  # last bold text
    ]
    for pattern in answer_markers:
        matches = re.findall(pattern, text, re.IGNORECASE | re.MULTILINE)
        if matches:
            candidate = matches[-1].strip().rstrip(".")
            candidate = candidate.strip('"').strip("'").strip("*").strip()
            if 3 <= len(candidate) <= 80 and candidate.lower() not in (
                "unknown", "none", "n/a", "thinking process"
            ):
                return candidate

    # Strategy 2: Take the last non-empty, non-structural line
    # (the answer is typically at the very end of the thinking)
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # Walk backwards from the end
    for line in reversed(lines):
        clean = line.strip().rstrip(".").strip()

        # Skip structural lines
        if clean.lower().startswith(("thinking process", "1.", "2.", "3.",
                                      "4.", "5.", "*", "-", "##",
                                      "analyze", "identify", "consider",
                                      "conclusion:", "therefore")):
            # But "conclusion:" might be followed by the answer on same line
            if ":" in clean:
                after_colon = clean.split(":", 1)[1].strip().rstrip(".")
                after_colon = after_colon.strip('"').strip("'").strip("*").strip()
                if 3 <= len(after_colon) <= 80 and after_colon.lower() not in (
                    "unknown", "none", "n/a"
                ):
                    return after_colon
            continue

        # Skip lines with bullet-point-like structure
        if re.match(r'^[\*\-•]\s', clean):
            continue

        # Skip lines that are clearly meta-text
        meta_words = {"task:", "input", "context", "abbreviation:",
                      "role:", "note:", "step"}
        if any(clean.lower().startswith(w) for w in meta_words):
            continue

        # Remove markdown formatting
        clean = clean.replace("**", "").strip()

        # This line might be the answer
        if 3 <= len(clean) <= 80:
            return clean

    return None


def _parse_expansion(raw_response: str) -> str | None:
    """
    Parse the LLM response to extract the expansion.

    Returns None if the LLM said UNKNOWN or the response is not usable.

    Handles various LLM output styles:
      - Clean single-line: "acute myocardial infarction"
      - With thinking tags: "<think>...</think>\nacute myocardial infarction"
      - With explanation: "AMI stands for acute myocardial infarction"
      - With prefix: "The expansion is: acute myocardial infarction"
      - Multi-line with reasoning: "...\nacute myocardial infarction"
      - Quoted: '"acute myocardial infarction"'
      - Bold/formatted: "**acute myocardial infarction**"
    """
    text = _strip_thinking_tags(raw_response).strip()

    if not text:
        return None

    # Check for UNKNOWN variants
    text_upper = text.upper().strip().rstrip(".")
    if text_upper in ("UNKNOWN", "NONE", "N/A", "NOT SURE", "I DON'T KNOW",
                       "I'M NOT SURE", "CANNOT DETERMINE", "NOT ENOUGH CONTEXT"):
        return None

    # Remove markdown bold markers
    text = text.replace("**", "").strip()

    # Remove quotes if the LLM wrapped the answer
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # If multi-line, try to extract just the expansion
    if "\n" in text:
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return None

        # Strategy 1: If any line is short and looks like just a term
        # (no verbs, no punctuation besides hyphens), prefer that
        best_line = None
        for line in lines:
            clean = line.strip().rstrip(".")
            # Skip lines that look like explanations
            if any(w in clean.lower() for w in ["stands for", "refers to",
                    "is an abbreviation", "the abbreviation", "in this context",
                    "which means", "this is", "it means"]):
                # Extract after "stands for" / "refers to" etc.
                for marker in ["stands for ", "refers to ", "which means ",
                               "is an abbreviation for ", "abbreviation for "]:
                    idx = clean.lower().find(marker)
                    if idx >= 0:
                        candidate = clean[idx + len(marker):].strip().rstrip(".")
                        candidate = candidate.strip('"').strip("'").strip()
                        if 3 <= len(candidate) <= 80:
                            return candidate
                continue
            # A clean term line: no sentence-like structure
            if len(clean) <= 80 and clean[0].isalpha():
                best_line = clean
                break

        if best_line:
            text = best_line
        else:
            # Fallback: take the last short line (often the answer)
            short_lines = [l.strip().rstrip(".") for l in lines
                          if 3 <= len(l.strip()) <= 80]
            if short_lines:
                text = short_lines[-1]
            else:
                return None

    # Remove trailing period
    if text.endswith("."):
        text = text[:-1].strip()

    # Try to extract from "stands for X" / "refers to X" patterns
    for marker in ["stands for ", "refers to ", "which means ",
                   "is an abbreviation for ", "abbreviation for ",
                   "expands to ", "full form is ", "full form: ",
                   "expansion: ", "expanded form: "]:
        idx = text.lower().find(marker)
        if idx >= 0:
            candidate = text[idx + len(marker):].strip().rstrip(".")
            candidate = candidate.strip('"').strip("'").strip()
            if 3 <= len(candidate) <= 80:
                text = candidate
                break

    # Remove leading labels like "Answer: " or "Expansion: "
    for prefix in ["answer:", "expansion:", "result:", "full form:",
                   "the full form is", "it stands for"]:
        if text.lower().startswith(prefix):
            text = text[len(prefix):].strip()

    # Remove quotes again (after extraction)
    if (text.startswith('"') and text.endswith('"')) or \
       (text.startswith("'") and text.endswith("'")):
        text = text[1:-1].strip()

    # Remove markdown bold again
    text = text.replace("**", "").strip()

    # Final sanity checks
    if len(text) < 3:
        return None
    if len(text) > 100:
        return None

    # Check it's not still an UNKNOWN-like answer
    if text.upper().rstrip(".") in ("UNKNOWN", "NONE", "N/A"):
        return None

    return text


# ── Main class ───────────────────────────────────────────────────────────

@dataclass
class ExpansionResult:
    """Result of an LLM abbreviation expansion attempt."""
    mention: str
    expansion: str | None  # None if not expanded
    source: str            # "llm", "cache", or "failed"
    raw_response: str = ""


class LLMAbbreviationExpander:
    """
    Expands biomedical abbreviations using an LLM.

    Uses the same OpenAI-compatible API as LLMDisambiguator (e.g.,
    LMStudio with a local Qwen3 model).

    Parameters
    ----------
    model : str
        Model name (as registered in LMStudio).
    base_url : str
        API endpoint URL.
    temperature : float
        Sampling temperature (lower = more deterministic). Default 0.3
        because we want consistent, factual expansions.
    max_tokens : int
        Max response length. Short because we only need the expansion.
    cache_expansions : bool
        If True, cache (mention → expansion) to avoid redundant LLM calls
        when the same abbreviation appears multiple times in a dataset.
    """

    def __init__(
        self,
        model: str = "qwen3-8b",
        base_url: str = "http://localhost:1234/v1",
        temperature: float = 0.3,
        max_tokens: int = 256,
        cache_expansions: bool = True,
        debug: bool = False,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache_expansions = cache_expansions
        self.debug = debug

        # Session cache: abbreviation → expansion (or None)
        self._cache: dict[str, str | None] = {}
        self._stats = {"calls": 0, "cache_hits": 0, "expanded": 0,
                        "failed": 0, "api_errors": 0, "parse_failures": 0}

        self.client = OpenAI(
            base_url=base_url,
            api_key="lm-studio",
            timeout=30.0,
        )

        # Verify connection and auto-detect model
        try:
            models = self.client.models.list()
            model_ids = [m.id for m in models.data]
            print(f"  LLM Abbreviation Expander connected to {base_url}")
            print(f"    Available models: {model_ids}")
            if model not in model_ids:
                if len(model_ids) == 1:
                    self.model = model_ids[0]
                    print(f"    Auto-detected model: '{self.model}'")
                else:
                    matches = [m for m in model_ids if model in m or m in model]
                    if len(matches) == 1:
                        self.model = matches[0]
                        print(f"    Auto-detected model: '{self.model}'")
                    else:
                        self.model = model_ids[0] if model_ids else model
                        print(f"    Warning: '{model}' not found, using '{self.model}'")
        except Exception as e:
            print(f"  Warning: LLM Abbreviation Expander could not connect: {e}")

    def expand(
        self,
        mention: str,
        context: str = "",
        title: str = "",
    ) -> str | None:
        """
        Expand an abbreviation using the LLM.

        Parameters
        ----------
        mention : str
            The abbreviation to expand (e.g., "AMI", "IDM").
        context : str
            The surrounding text (abstract).
        title : str
            The paper title.

        Returns
        -------
        str or None
            The expanded form, or None if expansion failed/unknown.
        """
        # Check cache first (by normalized mention)
        cache_key = mention.upper()
        if self.cache_expansions and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            return self._cache[cache_key]

        # Build prompt and call LLM
        user_prompt = _build_expansion_prompt(mention, context, title)

        try:
            self._stats["calls"] += 1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            msg = response.choices[0].message
            raw = (msg.content or "").strip()

            # Fallback: if content is empty, Qwen3 may have put the
            # answer in reasoning_content (thinking mode splits output)
            if not raw:
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    if self.debug:
                        print(f"    [LLM-ABBREV] '{mention}': content empty, "
                              f"extracting from reasoning_content ({len(reasoning)} chars)")
                    # Extract the actual answer from the thinking process
                    extracted = _extract_answer_from_reasoning(reasoning)
                    if extracted:
                        raw = extracted
                    else:
                        # Last resort: use the full reasoning and hope
                        # _parse_expansion can find something
                        raw = reasoning.strip()
        except Exception as e:
            self._stats["failed"] += 1
            self._stats["api_errors"] += 1
            if self.debug:
                print(f"    [LLM-ABBREV] API error for '{mention}': {e}")
            if self.cache_expansions:
                self._cache[cache_key] = None
            return None

        # Parse the response
        expansion = _parse_expansion(raw)

        if self.debug and self._stats["calls"] <= 10:
            raw_short = raw[:150].replace("\n", "\\n") if raw else "(empty)"
            print(f"    [LLM-ABBREV] '{mention}' → raw: '{raw_short}' → parsed: '{expansion}'")

        if expansion:
            # Basic validation: expansion should be longer than abbreviation
            if len(expansion) <= len(mention):
                if self.debug:
                    print(f"    [LLM-ABBREV] '{mention}' → rejected (expansion '{expansion}' too short)")
                expansion = None
            else:
                self._stats["expanded"] += 1

        if expansion is None:
            self._stats["failed"] += 1
            self._stats["parse_failures"] += 1

        # Cache the result
        if self.cache_expansions:
            self._cache[cache_key] = expansion

        return expansion

    def expand_with_details(
        self,
        mention: str,
        context: str = "",
        title: str = "",
    ) -> ExpansionResult:
        """Like expand(), but returns detailed result object."""
        cache_key = mention.upper()
        if self.cache_expansions and cache_key in self._cache:
            self._stats["cache_hits"] += 1
            cached = self._cache[cache_key]
            return ExpansionResult(
                mention=mention,
                expansion=cached,
                source="cache",
            )

        user_prompt = _build_expansion_prompt(mention, context, title)

        try:
            self._stats["calls"] += 1
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            msg = response.choices[0].message
            raw = (msg.content or "").strip()
            if not raw:
                reasoning = getattr(msg, "reasoning_content", None)
                if reasoning:
                    extracted = _extract_answer_from_reasoning(reasoning)
                    raw = extracted if extracted else reasoning.strip()
        except Exception as e:
            self._stats["failed"] += 1
            return ExpansionResult(
                mention=mention,
                expansion=None,
                source="failed",
                raw_response=f"ERROR: {e}",
            )

        expansion = _parse_expansion(raw)

        if expansion and len(expansion) <= len(mention):
            expansion = None

        if expansion:
            self._stats["expanded"] += 1
            source = "llm"
        else:
            self._stats["failed"] += 1
            source = "failed"

        if self.cache_expansions:
            self._cache[cache_key] = expansion

        return ExpansionResult(
            mention=mention,
            expansion=expansion,
            source=source,
            raw_response=raw,
        )

    def get_stats(self) -> dict:
        """Return expansion statistics."""
        return dict(self._stats)

    def print_stats(self):
        """Print expansion statistics."""
        s = self._stats
        print(f"  LLM Abbreviation Expander stats:")
        print(f"    LLM calls:      {s['calls']}")
        print(f"    Cache hits:     {s['cache_hits']}")
        print(f"    Expanded:       {s['expanded']}")
        print(f"    Failed:         {s['failed']}")
        print(f"      API errors:   {s['api_errors']}")
        print(f"      Parse fails:  {s['parse_failures']}")


# ── Quick demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("LLM Abbreviation Expander — Demo")
    print("=" * 60)

    expander = LLMAbbreviationExpander()

    test_cases = [
        {
            "mention": "IDM",
            "title": "Neonatal complications in diabetic pregnancies",
            "context": "The IDM showed signs of neonatal hypoglycemia at birth. "
                       "Maternal diabetes was poorly controlled during pregnancy.",
        },
        {
            "mention": "TdP",
            "title": "Drug-induced cardiac arrhythmias",
            "context": "The patient developed TdP after administration of the "
                       "antiarrhythmic drug. QT interval prolongation was observed.",
        },
        {
            "mention": "PD",
            "title": "Dopaminergic pathways in neurodegenerative disease",
            "context": "Patients with PD exhibited reduced dopamine transporter "
                       "binding in the striatum. Tremor was the presenting symptom.",
        },
        {
            "mention": "MS",
            "title": "Proteomic analysis of plasma biomarkers",
            "context": "Samples were analyzed by LC-MS to identify differentially "
                       "expressed proteins. MS data were processed using MaxQuant.",
        },
        {
            "mention": "CRC",
            "title": "Screening strategies for colorectal malignancies",
            "context": "Early detection of CRC through colonoscopy significantly "
                       "reduces mortality. The 5-year survival rate improves.",
        },
    ]

    for tc in test_cases:
        result = expander.expand_with_details(**tc)
        print(f'\n  "{tc["mention"]}" → {result.expansion or "(UNKNOWN)"}')
        print(f"    Source: {result.source}")
        if result.raw_response:
            # Show first 80 chars of raw response
            raw_short = result.raw_response[:80].replace("\n", " ")
            print(f"    Raw: {raw_short}...")

    print()
    expander.print_stats()
