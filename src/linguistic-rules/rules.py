"""
Linguistic Rules for Entity Extraction
=======================================
Individual rule functions implementing the 4 linguistic rules
from BioLinkerAI (Section 3, derived from [6,15] — LNN-EL).

Rules:
    1. Stopwords are not entities or relations
    2. Verbs are not entities
    3. A single compound word comprises words without stopwords
    4. Entities with only stopwords between them are one entity

Additionally, based on Rule 7 from the domain-specific rules:
    - Entity labels may combine words and numbers (e.g., "Interleukin 1 Beta")
    - Numbers adjacent to entity tokens are included as candidates
"""


# ── Rule 1: Stopwords are not entities ──────────────────────────────────────

def is_stopword(token) -> bool:
    """
    Stopwords (e.g., "the", "is", "and", "of") cannot be standalone entities.
    However, they may appear *inside* multi-word entities (see Rule 4).

    Parameters
    ----------
    token : spacy.tokens.Token

    Returns
    -------
    bool
    """
    return token.is_stop


# ── Rule 2: Verbs are not entities ──────────────────────────────────────────

def is_verb(token) -> bool:
    """
    Verbs (e.g., "having", "reported", "affects") are never entities.
    We also exclude auxiliary verbs (e.g., "is", "was", "has").

    Parameters
    ----------
    token : spacy.tokens.Token

    Returns
    -------
    bool
    """
    return token.pos_ in ("VERB", "AUX")


# ── Candidate check (combines Rules 1 & 2) ─────────────────────────────────

def is_candidate(token) -> bool:
    """
    A token is a candidate if it passes both Rule 1 and Rule 2,
    and is not punctuation. Numbers ARE candidates (Rule 7:
    entity labels may combine words and numbers).

    Parameters
    ----------
    token : spacy.tokens.Token

    Returns
    -------
    bool
        True if the token could be part of an entity mention.
    """
    # Punctuation is never an entity
    if token.is_punct:
        return False

    # Rule 1: stopwords are not entities
    if is_stopword(token):
        return False

    # Rule 2: verbs are not entities
    if is_verb(token):
        return False

    return True


# ── Rule 3: Merge adjacent candidate tokens ─────────────────────────────────

def merge_adjacent_candidates(token_candidates: list[dict]) -> list[list[dict]]:
    """
    Rule 3: A single compound word comprises words without stopwords.

    Adjacent candidate tokens (no non-candidate tokens between them)
    are merged into a single entity span. This handles compound
    biomedical terms like "body temperature" or "Interleukin 1 Beta".

    Parameters
    ----------
    token_candidates : list of dict
        Each dict has keys: 'token' (spaCy Token), 'is_candidate' (bool).

    Returns
    -------
    list of list of dict
        Groups of token dicts, where each group is one entity span.
    """
    spans = []
    current_span = []

    for tc in token_candidates:
        if tc["is_candidate"]:
            # This token is a candidate — add to current span
            current_span.append(tc)
        else:
            # Non-candidate resets the current span
            if current_span:
                spans.append(current_span)
                current_span = []

    # For the last span
    if current_span:
        spans.append(current_span)

    return spans


# ── Rule 4: Merge spans separated only by stopwords ─────────────────────────

def merge_across_stopwords(
    spans: list[list[dict]],
    all_tokens: list[dict],
    max_gap: int = 3,
) -> list[list[dict]]:
    """
    Rule 4: Entities with only stopwords between them are one entity.

    If two entity spans are separated only by stopword tokens
    (and the gap is at most `max_gap` tokens), they are merged
    into a single entity span. This handles phrases like
    "cancer of the lung" where "of" and "the" are stopwords.

    The following token types block merging even if they are stopwords:
    - Conjunctions (e.g., "and", "or") — separate distinct concepts
    - Verbs / auxiliaries (e.g., "is", "was") — indicate a clause boundary
    This prevents merges like "temperature is around 39" or
    "fever and chills" while still allowing "cancer of the lung".

    Parameters
    ----------
    spans : list of list of dict
        Entity spans from Rule 3 (merge_adjacent_candidates).
    all_tokens : list of dict
        All tokens in the sentence with their candidate status.
    max_gap : int
        Maximum number of stopword tokens allowed between two spans
        for them to be merged. Default is 3 (handles "X of the Y").

    Returns
    -------
    list of list of dict
        Merged entity spans.
    """
    # Nothing can be merged
    if len(spans) <= 1:
        return spans

    merged = [spans[0]]

    for i in range(1, len(spans)):
        prev_span = merged[-1]
        curr_span = spans[i]

        # Find the token indices of the gap between spans
        prev_last_idx = prev_span[-1]["token"].i
        curr_first_idx = curr_span[0]["token"].i

        # Collect tokens in the gap
        gap_tokens = [
            tc for tc in all_tokens
            if prev_last_idx < tc["token"].i < curr_first_idx
        ]

        # Only merge across stopwords that are structural connectors
        # (prepositions like "of" and determiners like "the", "a").
        # This captures biomedical patterns like "cancer of the lung"
        # or "disease of the liver" while preventing overly broad merges
        # like "key role in inflammation" where "in" connects two
        # semantically distinct concepts.
        ALLOWED_GAP_POS = ("ADP", "DET")

        gap_allows_merge = (
                0 < len(gap_tokens) <= max_gap
                and all(
                tc["token"].is_stop and tc["token"].pos_ in ALLOWED_GAP_POS
                for tc in gap_tokens
            )
        )

        if gap_allows_merge:
            # Merge: extend previous span with gap tokens and current span
            merged[-1] = prev_span + gap_tokens + curr_span
        else:
            # Don't merge — keep as separate spans
            merged.append(curr_span)

    return merged
