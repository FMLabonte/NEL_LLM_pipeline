"""
Linguistic Entity Extractor
============================
Phase 1 of the BioLinkerAI pipeline.

Extracts entity surface forms (mentions) from raw biomedical text
using the 4 linguistic rules described in the BioLinkerAI paper
(Section 3, "A Rule-governed framework").

Usage:
    from entity_extractor import LinguisticEntityExtractor

    extractor = LinguisticEntityExtractor()
    mentions = extractor.extract("She is having a fever and her temperature is around 39")
    # Returns: [EntityMention("fever", ...), EntityMention("temperature", ...)]
"""

import spacy
from dataclasses import dataclass, field

from rules import is_candidate, merge_adjacent_candidates, merge_across_stopwords


# ── Data class for extracted entity mentions ────────────────────────────────

@dataclass
class EntityMention:
    """
    Represents a single entity mention extracted from text.

    Attributes
    ----------
    text : str
        The surface form as it appears in the original text
        (e.g., "body temperature", "cancer of the lung").
    start : int
        Character offset of the first character in the original text.
    end : int
        Character offset one past the last character in the original text.
    tokens : list[str]
        The individual tokens that make up this mention.
    """
    text: str
    start: int
    end: int
    tokens: list[str] = field(default_factory=list)

    def __repr__(self):
        return f'EntityMention("{self.text}", start={self.start}, end={self.end})'


# ── Main extractor class ───────────────────────────────────────────────────

class LinguisticEntityExtractor:
    """
    Extracts entity mentions from raw biomedical text using the
    4 linguistic rules from BioLinkerAI.

    The extraction pipeline per sentence:
        1. Tokenize and POS-tag with spaCy
        2. Apply Rule 1 & 2: mark stopwords and verbs as non-candidates
        3. Apply Rule 3: merge adjacent candidate tokens into spans
        4. Apply Rule 4: merge spans separated only by stopwords
        5. Build EntityMention objects from the final spans

    Parameters
    ----------
    model_name : str
        spaCy model to use for tokenization and POS tagging.
        Default is "en_core_web_sm" (general English).
        For better biomedical tokenization, consider switching to
        "en_core_sci_sm" from scispaCy (pip install scispacy).
    custom_stopwords : set[str] or None
        Optional custom stopword list. If None, uses spaCy's defaults.
    max_stopword_gap : int
        Maximum number of stopword tokens between two entity spans
        for Rule 4 merging. Default is 3.
    """

    def __init__(
        self,
        model_name: str = "en_core_sci_sm", # en_core_web_sm as an alternative
        custom_stopwords: set[str] | None = None,
        max_stopword_gap: int = 3,
    ):
        self.nlp = spacy.load(model_name)
        self.max_stopword_gap = max_stopword_gap

        # Override spaCy's default stopwords if custom list is provided
        if custom_stopwords is not None:
            self.nlp.Defaults.stop_words = custom_stopwords

    def extract(self, text: str) -> list[EntityMention]:
        """
        Extract entity mentions from a full text (e.g., a PubMed abstract).

        Processes each sentence independently using the linguistic rules.

        Parameters
        ----------
        text : str
            Raw biomedical text.

        Returns
        -------
        list[EntityMention]
            Extracted entity mentions with text and character offsets.
        """
        doc = self.nlp(text)
        mentions = []

        for sent in doc.sents:
            sent_mentions = self._extract_from_sentence(sent)
            mentions.extend(sent_mentions)

        return mentions

    def _extract_from_sentence(self, sent) -> list[EntityMention]:
        """
        Applies Rules 1-4 in sequence to identify entity spans.

        Parameters
        ----------
        sent : spacy.tokens.Span
            A sentence span from a spaCy Doc.

        Returns
        -------
        list[EntityMention]
        """
        # ── Step 1: Classify each token as candidate or not ──
        # Rules 1 (stopwords) and 2 (verbs) are applied here via is_candidate().
        # Hyphens are treated as transparent connectors: if a hyphen sits
        # between two candidate tokens, all three are marked as candidates
        # so they merge into one entity (e.g., "3-methoxy-4-hydroxyphenethyleneglycol").
        token_infos = [
            {"token": token, "is_candidate": is_candidate(token)}
            for token in sent
        ]

        # Mark hyphens between two candidates as candidates themselves,
        # so Rule 3 merges them into a single span.
        for j in range(1, len(token_infos) - 1):
            tok = token_infos[j]["token"]
            if tok.text == "-" and token_infos[j - 1]["is_candidate"] and token_infos[j + 1]["is_candidate"]:
                token_infos[j]["is_candidate"] = True

        # ── Step 2: Rule 3 — merge adjacent candidate tokens ──
        # e.g., ["body", "temperature"] → one span ["body temperature"]
        spans = merge_adjacent_candidates(token_infos)

        # ── Step 3: Rule 4 — merge spans separated only by stopwords ──
        # e.g., ["cancer"] + ["lung"] with "of the" between → ["cancer of the lung"]
        spans = merge_across_stopwords(
            spans, token_infos, max_gap=self.max_stopword_gap
        )

        # ── Step 4: Filter out number-only spans ──
        # Standalone numbers (e.g., "39") are not entities on their own,
        # but numbers inside multi-token spans are fine (e.g., "Interleukin 1 Beta").
        spans = [
            span for span in spans
            if not all(info["token"].like_num or info["token"].text == "-" for info in span)
        ]

        # ── Step 5: Build EntityMention objects ──
        mentions = []
        for span in spans:
            tokens_in_span = [info["token"] for info in span]

            # Character offsets from the first and last token in the span
            start_char = tokens_in_span[0].idx
            end_char = tokens_in_span[-1].idx + len(tokens_in_span[-1])

            # Extract the original text (preserving whitespace/hyphens)
            mention_text = sent.doc.text[start_char:end_char]

            mentions.append(EntityMention(
                text=mention_text,
                start=start_char,
                end=end_char,
                tokens=[t.text for t in tokens_in_span],
            ))

        return mentions


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    extractor = LinguisticEntityExtractor()

    # Example from the paper (Fig. 2)
    test_text = "She is having a fever and her temperature is around 39"
    print(f"Input: \"{test_text}\"\n")

    mentions = extractor.extract(test_text)

    print(f"Extracted {len(mentions)} mention(s):")
    for m in mentions:
        print(f"  → \"{m.text}\" (chars {m.start}–{m.end}, tokens: {m.tokens})")

    # Additional test cases
    print("\n--- Additional test cases ---\n")

    test_cases = [
        "3-methoxy-4-hydroxyphenethyleneglycol is a metabolite",
        "cancer of the lung is a serious disease",
        "Interleukin 1 Beta plays a key role in inflammation",
        "Several studies reported somatic mutations of many genes",
    ]

    for text in test_cases:
        mentions = extractor.extract(text)
        print(f"Input: \"{text}\"")
        for m in mentions:
            print(f"  → \"{m.text}\" (tokens: {m.tokens})")
        print()
