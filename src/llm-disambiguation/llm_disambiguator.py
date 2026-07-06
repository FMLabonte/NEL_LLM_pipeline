"""
LLM Disambiguator
==================
Given a list of candidate entities and the document context, an LLM
selects the best matching candidate.

Prompt construction, response parsing, and version management are
handled by prompts.py — this module handles the API connection,
retry logic, token tracking, and pipeline interface.

Usage:
    from llm_disambiguator import LLMDisambiguator

    disambiguator = LLMDisambiguator(
        model="openai/gpt-4o-mini",
        base_url="http://131.220.150.238:8080",
        api_key="sk-...",
        prompt_version="v3",
    )

    result = disambiguator.disambiguate(
        mention="CF",
        candidates=[...],
        context="full abstract text...",
        title="Paper title",
    )
    # result.mesh_id -> "D003550"
"""

import time
import random
from dataclasses import dataclass

from openai import OpenAI

from prompts import (
    get_prompt_config,
    build_user_prompt,
    parse_response,
    list_prompts,
    PromptConfig,
)


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class DisambiguationResult:
    """Result of LLM disambiguation for a single mention."""
    mention: str
    mesh_id: str
    preferred_label: str
    chosen_rank: int        # 1-based index, -1 if fallback
    confidence: str         # "llm" or "fallback"
    raw_response: str


# ── Main disambiguator class ──────────────────────────────────────────────

class LLMDisambiguator:
    """
    Uses an LLM to disambiguate between candidate entities.

    Connects to an OpenAI-compatible API (LMStudio, LiteLLM proxy, etc.)
    and uses prompt configs from prompts.py.
    """

    def __init__(
        self,
        model: str = "qwen3.5-9b",
        base_url: str = "http://localhost:1234/v1",
        api_key: str = "lm-studio",
        temperature: float = 0.6,
        max_tokens: int = 8192,
        timeout: float = 300.0,
        no_think: bool = False,
        prompt_version: str = "v3",
        shuffle_candidates: bool = False,
        shuffle_seed: int | None = None,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens if not no_think else min(max_tokens, 512)
        self.no_think = no_think
        self.shuffle_candidates = shuffle_candidates

        # Load prompt config from registry
        self.prompt_config = get_prompt_config(prompt_version)

        # Set up shuffling RNG (deterministic if seed is given)
        self._rng = random.Random(shuffle_seed) if shuffle_candidates else None

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )

        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        # Pricing per 1M tokens (input, output)
        self._pricing = {
            "openai/gpt-4o-mini":              (0.15,  0.60),
            "openai/gpt-4.1-mini":             (0.40,  1.60),
            "openai/gpt-4.1-nano":             (0.10,  0.40),
            "openai/gpt-5-nano":               (0.20,  1.25),
            "openai/gpt-5-mini":               (0.75,  4.50),
            "openai/gpt-5.5":                  (5.00, 30.00),
            "mistral/mistral-small-latest":     (0.10,  0.30),
            "mistral/mistral-small":            (0.10,  0.30),
            "mistral/magistral-medium-latest":  (2.00,  5.00),
            "mistral/mistral-large-latest":     (2.00,  6.00),
        }

        # Verify connection and auto-detect model name
        try:
            quick_client = OpenAI(
                base_url=base_url, api_key=api_key, timeout=5.0,
            )
            models = quick_client.models.list()
            model_ids = [m.id for m in models.data]
            if model not in model_ids:
                if len(model_ids) == 1:
                    self.model = model_ids[0]
                else:
                    matches = [m for m in model_ids if model in m or m in model]
                    if len(matches) == 1:
                        self.model = matches[0]
        except Exception:
            pass

        # Log config (compact)
        think_label = "no_think" if self.no_think else "think"
        shuffle_label = (
            f", shuffle=seed{shuffle_seed}" if self.shuffle_candidates and shuffle_seed
            else ", shuffle=random" if self.shuffle_candidates
            else ""
        )
        print(f"  LLM: {self.model} | prompt={self.prompt_config.name} "
              f"({self.prompt_config.response_format}) | "
              f"{think_label}, max_tokens={self.max_tokens}{shuffle_label}")

    # ── Main disambiguation method ────────────────────────────────────────

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
            The entity mention text (e.g., "CF", "seizures").
        candidates : list[CandidateEntity]
            Ranked candidates from Phase 3. Order may be shuffled
            if shuffle_candidates is enabled.
        context : str
            Document text. Should be the full abstract if the prompt
            config uses full_context, otherwise a sentence snippet.
        title : str
            Paper title.
        """
        if not candidates:
            return DisambiguationResult(
                mention=mention, mesh_id="NONE", preferred_label="",
                chosen_rank=-1, confidence="fallback", raw_response="",
            )

        # Optionally shuffle candidate order (to test position bias)
        if self._rng is not None:
            candidates = list(candidates)  # copy
            self._rng.shuffle(candidates)

        # Build prompt
        user_prompt = build_user_prompt(
            self.prompt_config, mention, candidates, context, title,
        )

        # Append /no_think for Qwen3.5 models
        if self.no_think:
            user_prompt += "\n/no_think"

        messages = [
            {"role": "system", "content": self.prompt_config.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # Call LLM with retry on timeout
        raw = None
        max_retries = 2
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                raw = response.choices[0].message.content.strip()

                # Track token usage
                if hasattr(response, 'usage') and response.usage:
                    self.total_input_tokens += (
                        response.usage.prompt_tokens or 0
                    )
                    self.total_output_tokens += (
                        response.usage.completion_tokens or 0
                    )
                break

            except Exception as e:
                err_str = str(e).lower()
                is_timeout = "timeout" in err_str or "timed out" in err_str
                if is_timeout and attempt < max_retries:
                    print(f"  Timeout for '{mention}' "
                          f"(attempt {attempt}/{max_retries}), retrying...")
                    time.sleep(2)
                    continue

                label = "Timeout" if is_timeout else "API error"
                print(f"  LLM {label} for '{mention}': {e}")
                return DisambiguationResult(
                    mention=mention,
                    mesh_id=candidates[0].mesh_id,
                    preferred_label=candidates[0].preferred_label,
                    chosen_rank=1,
                    confidence="fallback",
                    raw_response=f"ERROR: {e}",
                )

        # Parse response using prompt-specific parser
        chosen_idx = parse_response(self.prompt_config, raw, candidates)

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
            # Parse failed — fall back to top-1 (from original ranking)
            return DisambiguationResult(
                mention=mention,
                mesh_id=candidates[0].mesh_id,
                preferred_label=candidates[0].preferred_label,
                chosen_rank=1,
                confidence="fallback",
                raw_response=raw,
            )

    # ── Token usage tracking ──────────────────────────────────────────────

    def get_usage_summary(self) -> str:
        """Return a formatted summary of token usage and estimated cost."""
        total = self.total_input_tokens + self.total_output_tokens
        lines = [
            f"  Token usage:     {self.total_input_tokens:,} input + "
            f"{self.total_output_tokens:,} output = {total:,} total",
        ]

        pricing = self._pricing.get(self.model)
        if pricing:
            input_price, output_price = pricing
            cost_in = self.total_input_tokens / 1_000_000 * input_price
            cost_out = self.total_output_tokens / 1_000_000 * output_price
            lines.append(
                f"  Estimated cost:  ${cost_in:.4f} (in) + "
                f"${cost_out:.4f} (out) = ${cost_in + cost_out:.4f}"
            )
        elif total > 0:
            lines.append(
                f"  Estimated cost:  unknown (no pricing for '{self.model}')"
            )

        return "\n".join(lines)

    # ── Batch disambiguation ──────────────────────────────────────────────

    def disambiguate_batch(
        self,
        items: list[dict],
        top_k: int = 5,
    ) -> list[DisambiguationResult]:
        """Disambiguate a batch of mentions."""
        results = []
        total = len(items)

        for i, item in enumerate(items):
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
