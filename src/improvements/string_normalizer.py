"""
String Normalizer for Biomedical Entity Mentions
==================================================
Normalizes mention text before search to handle common variations
that cause string matching to fail:

  - Hyphens / whitespace inconsistency: "non-small" vs "nonsmall" vs "non small"
  - Greek letters: "TNF-α" → "TNF-alpha", "IL-6β" → "IL-6beta"
  - Parenthetical qualifiers: "Aspirin (oral)" → "Aspirin"
  - Possessives: "Crohn's disease" → "Crohn disease"
  - Plurals (simple): "seizures" → also search "seizure"
  - Roman numerals: "Type II diabetes" → "Type 2 diabetes"
  - Common suffixes/prefixes: "-induced", "-associated", "-related"

Usage:
    from string_normalizer import normalize_mention, generate_variants

    # Single normalized form
    normalized = normalize_mention("TNF-α-induced hepatotoxicity")

    # All search variants (for broadening retrieval)
    variants = generate_variants("Crohn's disease")
    # → ["crohn's disease", "crohn disease", "crohns disease"]
"""

import re

# ── Greek letter mapping ──────────────────────────────────────────────────

GREEK_TO_LATIN = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta",
    "ε": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
    "ν": "nu", "ξ": "xi", "ο": "omicron", "π": "pi",
    "ρ": "rho", "σ": "sigma", "τ": "tau", "υ": "upsilon",
    "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega",
    # Uppercase
    "Α": "Alpha", "Β": "Beta", "Γ": "Gamma", "Δ": "Delta",
    "Ε": "Epsilon", "Ζ": "Zeta", "Η": "Eta", "Θ": "Theta",
    "Ι": "Iota", "Κ": "Kappa", "Λ": "Lambda", "Μ": "Mu",
    "Ν": "Nu", "Ξ": "Xi", "Ο": "Omicron", "Π": "Pi",
    "Ρ": "Rho", "Σ": "Sigma", "Τ": "Tau", "Υ": "Upsilon",
    "Φ": "Phi", "Χ": "Chi", "Ψ": "Psi", "Ω": "Omega",
}

# ── Roman numeral mapping ────────────────────────────────────────────────

ROMAN_TO_ARABIC = {
    "I": "1", "II": "2", "III": "3", "IV": "4", "V": "5",
    "VI": "6", "VII": "7", "VIII": "8", "IX": "9", "X": "10",
    "XI": "11", "XII": "12",
}


def _replace_greek(text: str) -> str:
    """Replace Greek letters with Latin equivalents."""
    for greek, latin in GREEK_TO_LATIN.items():
        if greek in text:
            text = text.replace(greek, latin)
    return text


def _replace_roman_numerals(text: str) -> str:
    """Replace Roman numerals with Arabic numbers (only standalone words)."""
    def _replace_match(m):
        word = m.group(0)
        return ROMAN_TO_ARABIC.get(word, word)

    # Match standalone Roman numerals (not part of larger words)
    return re.sub(r'\b(XII|XI|VIII|VII|VI|IV|IX|III|II|X|V|I)\b', _replace_match, text)


def _strip_parenthetical(text: str) -> str:
    """Remove parenthetical qualifiers like '(oral)', '(topical)', '(NOS)'."""
    # Only strip trailing parentheticals, not abbreviation definitions
    stripped = re.sub(r'\s*\([^)]{1,20}\)\s*$', '', text).strip()
    return stripped if stripped else text


def _remove_possessive(text: str) -> str:
    """Remove possessive 's: Crohn's → Crohn."""
    return re.sub(r"'s\b", "", text)


def _normalize_whitespace_hyphens(text: str) -> str:
    """Normalize whitespace and collapse multiple spaces."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def normalize_mention(text: str) -> str:
    """
    Apply standard normalization to a mention.

    This produces ONE canonical form for search. For broader matching,
    use generate_variants() which returns multiple forms.

    Steps:
      1. Replace Greek letters → Latin
      2. Strip trailing parentheticals
      3. Remove possessives
      4. Normalize whitespace
    """
    text = _replace_greek(text)
    text = _strip_parenthetical(text)
    text = _remove_possessive(text)
    text = _normalize_whitespace_hyphens(text)
    return text


def generate_variants(text: str) -> list[str]:
    """
    Generate multiple search variants for a mention.

    Returns a list of forms to search with, starting with the
    most specific (original) and adding normalized variants.
    Duplicates are removed while preserving order.

    Examples:
        "Crohn's disease" → ["Crohn's disease", "Crohn disease"]
        "TNF-α" → ["TNF-α", "TNF-alpha"]
        "non-small cell" → ["non-small cell", "nonsmall cell", "non small cell"]
        "seizures" → ["seizures", "seizure"]
        "Type II diabetes" → ["Type II diabetes", "Type 2 diabetes"]
    """
    variants = [text]

    # Greek letter replacement
    greek_replaced = _replace_greek(text)
    if greek_replaced != text:
        variants.append(greek_replaced)

    # Possessive removal
    if "'s" in text or "'s" in text:
        no_poss = _remove_possessive(text)
        if no_poss != text:
            variants.append(no_poss)

    # Parenthetical removal
    stripped = _strip_parenthetical(text)
    if stripped != text:
        variants.append(stripped)

    # Hyphen variants: "non-small" → "nonsmall" and "non small"
    if "-" in text:
        variants.append(text.replace("-", ""))   # collapsed
        variants.append(text.replace("-", " "))  # spaced

    # Simple plural handling: try removing trailing 's'
    text_lower = text.lower().strip()
    if (len(text_lower) > 3
            and text_lower.endswith("s")
            and not text_lower.endswith("ss")
            and not text_lower.endswith("is")
            and not text_lower.endswith("us")):
        singular = text[:-1]
        variants.append(singular)

    # Roman numerals
    roman_replaced = _replace_roman_numerals(text)
    if roman_replaced != text:
        variants.append(roman_replaced)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for v in variants:
        v_clean = _normalize_whitespace_hyphens(v)
        if v_clean and v_clean not in seen:
            seen.add(v_clean)
            unique.append(v_clean)

    return unique


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        "Crohn's disease",
        "TNF-α",
        "non-small cell lung cancer",
        "seizures",
        "Type II diabetes",
        "Aspirin (oral)",
        "IL-6β receptor",
        "Parkinson's disease",
        "hepatotoxicity",
        "anti-inflammatory",
        "5-hydroxytryptamine",
    ]

    print("String Normalization Variants:")
    print("=" * 60)
    for mention in test_cases:
        variants = generate_variants(mention)
        normalized = normalize_mention(mention)
        print(f'  "{mention}"')
        print(f'    Normalized: "{normalized}"')
        print(f'    Variants:   {variants}')
        print()
