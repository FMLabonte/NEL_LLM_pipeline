"""
Abbreviation Expander
======================
Detects abbreviations in entity mentions and expands them using
context from the surrounding text (abstract/title).

This addresses ~20% of Phase 2 failures where the mention is an
abbreviation (e.g., "IDM", "PD", "AMI") that has no string overlap
with any MeSH label, making fuzzy matching fail.

Strategies (applied in order):

  1. Parenthetical Extraction (forward):
     "acute myocardial infarction (AMI)" → AMI = acute myocardial infarction

  2. Parenthetical Extraction (reverse):
     "(AMI) acute myocardial infarction" → AMI = acute myocardial infarction

  3. First-Letter Heuristic Validation:
     Checks if the abbreviation letters match the first letters of the
     expanded words, to filter false positives.

  4. Known Biomedical Abbreviations (fallback dictionary):
     Common abbreviations that are rarely defined inline in abstracts.

Usage:
    from abbreviation_expander import AbbreviationExpander

    expander = AbbreviationExpander()
    expansions = expander.extract_abbreviations(
        "Patients with acute myocardial infarction (AMI) were treated."
    )
    # → {"AMI": "acute myocardial infarction"}

    expanded = expander.expand_mention("AMI", context="... (AMI) ...")
    # → "acute myocardial infarction"
"""

import re
from dataclasses import dataclass


@dataclass
class AbbreviationMatch:
    """
    A detected abbreviation-expansion pair.

    Attributes
    ----------
    abbreviation : str
        The short form (e.g., "AMI").
    expansion : str
        The long form (e.g., "acute myocardial infarction").
    confidence : str
        How the match was found: "parenthetical", "reverse_parenthetical",
        "dictionary", or "heuristic".
    """
    abbreviation: str
    expansion: str
    confidence: str


# ── Known biomedical abbreviations (fallback) ────────────────────────────
# Common abbreviations that may not be defined in every abstract.
# Kept small and high-confidence to avoid false positives.

KNOWN_ABBREVIATIONS = {
    # Diseases
    "AMI": "acute myocardial infarction",
    "MI": "myocardial infarction",
    "CHF": "congestive heart failure",
    "COPD": "chronic obstructive pulmonary disease",
    "CKD": "chronic kidney disease",
    "AKI": "acute kidney injury",
    "DVT": "deep vein thrombosis",
    "PE": "pulmonary embolism",
    "SLE": "systemic lupus erythematosus",
    "RA": "rheumatoid arthritis",
    "MS": "multiple sclerosis",
    "PD": "Parkinson disease",
    "AD": "Alzheimer disease",
    "DM": "diabetes mellitus",
    "T2DM": "type 2 diabetes mellitus",
    "T1DM": "type 1 diabetes mellitus",
    "HTN": "hypertension",
    "AF": "atrial fibrillation",
    "HCC": "hepatocellular carcinoma",
    "CRC": "colorectal cancer",
    "NSCLC": "non-small cell lung cancer",
    "SCLC": "small cell lung cancer",
    "AML": "acute myeloid leukemia",
    "ALL": "acute lymphoblastic leukemia",
    "CML": "chronic myeloid leukemia",
    "NHL": "non-Hodgkin lymphoma",
    "IDM": "infant of diabetic mother",
    "ARDS": "acute respiratory distress syndrome",
    "TIA": "transient ischemic attack",
    "CVA": "cerebrovascular accident",
    "VTE": "venous thromboembolism",
    "ILD": "interstitial lung disease",
    "IPF": "idiopathic pulmonary fibrosis",
    "IBD": "inflammatory bowel disease",
    "IBS": "irritable bowel syndrome",
    "GERD": "gastroesophageal reflux disease",
    "UTI": "urinary tract infection",
    "PTSD": "post-traumatic stress disorder",
    "OCD": "obsessive-compulsive disorder",
    "BPD": "borderline personality disorder",
    "TBI": "traumatic brain injury",
    "NMS": "neuroleptic malignant syndrome",
    "SS": "serotonin syndrome",
    "TdP": "torsades de pointes",
    "HUS": "hemolytic uremic syndrome",
    "TTP": "thrombotic thrombocytopenic purpura",
    "DIC": "disseminated intravascular coagulation",
    "HELLP": "hemolysis elevated liver enzymes low platelet count",
    "PCOS": "polycystic ovary syndrome",
    "BPH": "benign prostatic hyperplasia",
    "RCC": "renal cell carcinoma",
    "ASD": "atrial septal defect",
    "VSD": "ventricular septal defect",
    "PDA": "patent ductus arteriosus",
    "DCM": "dilated cardiomyopathy",
    "HCM": "hypertrophic cardiomyopathy",
    # Chemicals / Drugs
    "5-FU": "fluorouracil",
    "MTX": "methotrexate",
    "CsA": "cyclosporine",
    "MMF": "mycophenolate mofetil",
    "NSAID": "nonsteroidal anti-inflammatory drug",
    "NSAIDs": "nonsteroidal anti-inflammatory drugs",
    "SSRI": "selective serotonin reuptake inhibitor",
    "SSRIs": "selective serotonin reuptake inhibitors",
    "ACEi": "angiotensin-converting enzyme inhibitor",
    "ARB": "angiotensin receptor blocker",
    "CCB": "calcium channel blocker",
    "PPI": "proton pump inhibitor",
    "TCA": "tricyclic antidepressant",
    "SNRI": "serotonin-norepinephrine reuptake inhibitor",
    "EPO": "erythropoietin",
    "G-CSF": "granulocyte colony-stimulating factor",
    "TNF": "tumor necrosis factor",
    "IL-2": "interleukin-2",
    "IFN": "interferon",
}


class AbbreviationExpander:
    """
    Extracts and expands abbreviations from biomedical text.

    Parameters
    ----------
    min_abbrev_len : int
        Minimum length for an abbreviation (default: 2).
    max_abbrev_len : int
        Maximum length for an abbreviation (default: 10).
    use_dictionary : bool
        Whether to use the built-in dictionary as fallback (default: True).
    """

    def __init__(
        self,
        min_abbrev_len: int = 2,
        max_abbrev_len: int = 10,
        use_dictionary: bool = True,
    ):
        self.min_abbrev_len = min_abbrev_len
        self.max_abbrev_len = max_abbrev_len
        self.use_dictionary = use_dictionary

        # Build case-insensitive dictionary lookup
        self._dict_lookup: dict[str, str] = {}
        if use_dictionary:
            for abbr, expansion in KNOWN_ABBREVIATIONS.items():
                self._dict_lookup[abbr.lower()] = expansion

    def extract_abbreviations(self, text: str) -> dict[str, AbbreviationMatch]:
        """
        Extract all abbreviation-expansion pairs from a text.

        Scans for parenthetical patterns in both directions:
          - "long form (SHORT)" → forward pattern
          - "(SHORT) long form" → reverse pattern

        Parameters
        ----------
        text : str
            The full text (title + abstract).

        Returns
        -------
        dict[str, AbbreviationMatch]
            Mapping from abbreviation (uppercase) → AbbreviationMatch.
        """
        abbreviations: dict[str, AbbreviationMatch] = {}

        # ── Strategy 1: Forward pattern — "long form (ABBREV)" ──
        # Match: word(s) followed by (ABBREVIATION)
        # The abbreviation is typically 2-10 chars, mostly uppercase
        forward_pattern = re.compile(
            r'((?:[A-Za-z][\w-]*[\s,]+){1,10})'  # 1-10 words before parenthesis
            r'\(([A-Za-z][A-Za-z0-9-]{0,9})\)'   # (ABBREV) in parentheses
        )

        for match in forward_pattern.finditer(text):
            long_form_raw = match.group(1).strip()
            short_form = match.group(2).strip()

            if not self._is_likely_abbreviation(short_form):
                continue

            # Trim the long form to only the words that match the abbreviation
            # e.g., "Patients with acute myocardial infarction" → trim to
            # "acute myocardial infarction" for AMI (3 letters → ~3 words)
            long_form = self._trim_long_form(short_form, long_form_raw)

            if self._validate_first_letters(short_form, long_form):
                abbreviations[short_form.upper()] = AbbreviationMatch(
                    abbreviation=short_form,
                    expansion=long_form,
                    confidence="parenthetical",
                )

        # ── Strategy 2: Reverse pattern — "(ABBREV) long form" ──
        reverse_pattern = re.compile(
            r'\(([A-Za-z][A-Za-z0-9-]{0,9})\)'  # (ABBREV)
            r'\s+((?:[A-Za-z][\w-]*(?:[\s,]+|$)){1,10})'  # 1-10 words after
        )

        for match in reverse_pattern.finditer(text):
            short_form = match.group(1).strip()
            long_form_raw = match.group(2).strip()

            if short_form.upper() in abbreviations:
                continue  # already found via forward pattern

            if not self._is_likely_abbreviation(short_form):
                continue

            # Trim to matching length
            long_form = self._trim_long_form(short_form, long_form_raw)

            if self._validate_first_letters(short_form, long_form):
                abbreviations[short_form.upper()] = AbbreviationMatch(
                    abbreviation=short_form,
                    expansion=long_form,
                    confidence="reverse_parenthetical",
                )

        return abbreviations

    def expand_mention(
        self,
        mention: str,
        context: str = "",
        title: str = "",
    ) -> str | None:
        """
        Try to expand a mention if it looks like an abbreviation.

        Checks:
          1. Is the mention short enough to be an abbreviation?
          2. Can we find the expansion in the context text?
          3. Is it in our dictionary?

        Parameters
        ----------
        mention : str
            The entity mention to potentially expand.
        context : str
            The abstract/full text where the mention appears.
        title : str
            The paper title (also searched for definitions).

        Returns
        -------
        str or None
            The expanded form if found, None otherwise.
        """
        if not self._is_likely_abbreviation(mention):
            return None

        # Search in context (title + abstract)
        full_text = f"{title} {context}".strip() if title else context

        if full_text:
            abbreviations = self.extract_abbreviations(full_text)

            # Check for exact match (case-insensitive)
            mention_upper = mention.upper()
            if mention_upper in abbreviations:
                return abbreviations[mention_upper].expansion

            # Check for match ignoring trailing 's' (plural)
            if mention_upper.endswith("S") and mention_upper[:-1] in abbreviations:
                return abbreviations[mention_upper[:-1]].expansion

        # Fallback: dictionary lookup
        if self.use_dictionary:
            dict_match = self._dict_lookup.get(mention.lower())
            if dict_match:
                return dict_match

        return None

    def expand_mention_all(
        self,
        mention: str,
        context: str = "",
        title: str = "",
    ) -> list[str]:
        """
        Return all possible expansions for a mention (for multi-search).

        Returns a list that always includes the original mention,
        plus any expansions found.

        Parameters
        ----------
        mention : str
            The entity mention.
        context : str
            Abstract text.
        title : str
            Paper title.

        Returns
        -------
        list[str]
            List of search terms: [original_mention, expansion1, ...].
        """
        terms = [mention]

        expansion = self.expand_mention(mention, context, title)
        if expansion and expansion.lower() != mention.lower():
            terms.append(expansion)

        return terms

    def _trim_long_form(self, abbreviation: str, long_form: str) -> str:
        """
        Trim the long form to only the words that plausibly match the abbreviation.

        The regex may capture too many preceding/following words.
        Tries both suffixes (for forward: "Patients with acute myocardial
        infarction" → "acute myocardial infarction") and prefixes (for reverse:
        "Parkinson disease was diagnosed in" → "Parkinson disease").
        """
        words = long_form.split()

        # Strip trailing 's' from abbreviation for matching (plural form)
        # e.g., "SSRIs" → use "SSRI" for matching but keep the expansion
        abbrev_clean = abbreviation
        if (len(abbreviation) > 2
                and abbreviation[-1].lower() == 's'
                and abbreviation[-2].isupper()):
            abbrev_clean = abbreviation[:-1]

        abbrev_letters = [c for c in abbrev_clean if c.isalpha()]
        n_letters = len(abbrev_letters)

        if len(words) <= n_letters + 1:
            return long_form  # already short enough

        best_form = long_form
        best_len = len(words)

        # Try suffixes (trim from the left): "X Y Z" → "Y Z" → "Z"
        for start in range(len(words)):
            candidate = " ".join(words[start:])
            if self._validate_first_letters(abbreviation, candidate):
                n_words = len(words) - start
                if n_words < best_len:
                    best_len = n_words
                    best_form = candidate

        # Try prefixes (trim from the right): "X Y Z" → "X Y" → "X"
        for end in range(len(words), 0, -1):
            candidate = " ".join(words[:end])
            if self._validate_first_letters(abbreviation, candidate):
                if end < best_len:
                    best_len = end
                    best_form = candidate

        return best_form

    def _is_likely_abbreviation(self, text: str) -> bool:
        """
        Check if a text looks like an abbreviation.

        Heuristics:
          - Length between min and max
          - High ratio of uppercase letters, OR
          - Very short (2-4 chars)
          - Not a common short word
        """
        text = text.strip()

        if len(text) < self.min_abbrev_len or len(text) > self.max_abbrev_len:
            return False

        # Filter out common short words that aren't abbreviations
        common_short = {
            "the", "and", "for", "are", "was", "not", "but", "had",
            "has", "its", "can", "may", "use", "all", "new", "two",
            "one", "low", "high", "old", "men", "age", "day", "rat",
            "mice", "cell", "drug", "dose", "risk", "case", "pain",
            "loss", "oral", "bone", "lung", "skin", "iron", "acid",
            "gene", "type", "mild", "lead", "gold", "zinc",
        }
        if text.lower() in common_short:
            return False

        # Short tokens (2-4 chars) with uppercase letters → likely abbreviation
        if len(text) <= 4:
            uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
            return uppercase_ratio >= 0.5

        # Longer tokens: must have high uppercase ratio or contain numbers
        uppercase_ratio = sum(1 for c in text if c.isupper()) / len(text)
        has_numbers = any(c.isdigit() for c in text)

        return uppercase_ratio >= 0.5 or has_numbers

    def _validate_first_letters(self, abbreviation: str, long_form: str) -> bool:
        """
        Validate that the abbreviation matches the first letters of the
        long form words.

        Uses a flexible matching approach:
          - Letters from abbreviation should match first letters of long form words
          - Allows skipping common short words (of, the, and, in, etc.)
          - Allows lowercase letters in abbreviation to match mid-word

        Parameters
        ----------
        abbreviation : str
            The short form (e.g., "AMI").
        long_form : str
            The candidate long form (e.g., "acute myocardial infarction").

        Returns
        -------
        bool
            True if the abbreviation plausibly matches the long form.
        """
        # Extract only alphabetic characters from the abbreviation
        abbrev_letters = [c.lower() for c in abbreviation if c.isalpha()]
        if not abbrev_letters:
            return False

        # Split long form into words, skip trivial ones
        skip_words = {"of", "the", "and", "in", "a", "an", "to", "or", "by", "with", "we", "studied"}
        words = [w for w in long_form.lower().split() if w not in skip_words]

        if not words:
            return False

        # Try to match abbreviation letters to word first-letters
        # Uses recursive backtracking to handle ambiguous cases like
        # NSCLC = "non-small cell lung cancer" where letters can come
        # from first letters of words OR subsequent chars within a word
        return self._match_letters(abbrev_letters, words, 0, 0)

    def _match_letters(
        self,
        abbrev_letters: list[str],
        words: list[str],
        a_idx: int,
        w_idx: int,
    ) -> bool:
        """Recursive matching of abbreviation letters to words."""
        # Base case: all abbreviation letters matched
        if a_idx >= len(abbrev_letters):
            return True

        # No more words to match against
        if w_idx >= len(words):
            # Allow if we matched at least 70% of letters
            return a_idx / len(abbrev_letters) >= 0.7

        word = words[w_idx]
        letter = abbrev_letters[a_idx]

        # Option 1: current word's first letter matches
        if word[0] == letter:
            # Try consuming just the first letter (move to next word)
            if self._match_letters(abbrev_letters, words, a_idx + 1, w_idx + 1):
                return True

            # Try consuming multiple letters from this word
            # (e.g., "small" → "s", "c" from "cell" next, but "sc" from "scleroderma")
            for char_idx in range(1, min(len(word), 4)):
                if (a_idx + char_idx < len(abbrev_letters)
                        and word[char_idx] == abbrev_letters[a_idx + char_idx]):
                    if self._match_letters(
                        abbrev_letters, words, a_idx + char_idx + 1, w_idx + 1
                    ):
                        return True
                else:
                    break

        # Option 2: skip this word (it might be a filler word we missed)
        if self._match_letters(abbrev_letters, words, a_idx, w_idx + 1):
            return True

        return False


# ── Quick test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    expander = AbbreviationExpander()

    # Test extraction
    test_texts = [
        "Patients with acute myocardial infarction (AMI) were treated with aspirin.",
        "The infant of a diabetic mother (IDM) showed hypoglycemia.",
        "We studied chronic obstructive pulmonary disease (COPD) in elderly patients.",
        "(PD) Parkinson disease was diagnosed in 30 patients.",
        "Non-small cell lung cancer (NSCLC) represents 85% of lung cancers.",
        "Treatment with selective serotonin reuptake inhibitors (SSRIs) was initiated.",
    ]

    print("=" * 60)
    print("Abbreviation Extraction Tests")
    print("=" * 60)

    for text in test_texts:
        abbrevs = expander.extract_abbreviations(text)
        print(f"\nText: {text[:70]}...")
        if abbrevs:
            for abbr, match in abbrevs.items():
                print(f"  {abbr} → {match.expansion} ({match.confidence})")
        else:
            print("  (no abbreviations found)")

    # Test expansion
    print("\n" + "=" * 60)
    print("Abbreviation Expansion Tests")
    print("=" * 60)

    test_cases = [
        ("AMI", "Patients with acute myocardial infarction (AMI) were treated."),
        ("IDM", "The infant of a diabetic mother (IDM) showed hypoglycemia."),
        ("PD", ""),  # no context → should fall back to dictionary
        ("COPD", ""),  # dictionary fallback
        ("seizures", ""),  # not an abbreviation
        ("HTN", "The patient had long-standing hypertension (HTN)."),
    ]

    for mention, context in test_cases:
        expansion = expander.expand_mention(mention, context)
        all_terms = expander.expand_mention_all(mention, context)
        status = f"→ {expansion}" if expansion else "→ (no expansion)"
        print(f'  "{mention}" {status}')
        print(f"    Search terms: {all_terms}")
