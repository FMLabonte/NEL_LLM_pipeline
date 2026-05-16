"""
Document-Topic Consistency Scorer
==================================
Lightweight taxonomy consistency check: detects the dominant biomedical
topic(s) of a document from its abstract text, then boosts/penalizes
candidates whose MeSH tree numbers align (or conflict) with that topic.

Idea: If a document is about oncology (lots of cancer-related terms),
an ambiguous mention should prefer a Neoplasms (C04) candidate over
a Cardiovascular (C14) one.

This is a *lightweight* proxy for full document-level consistency
(which would require linking all mentions first, then cross-checking).
Instead, we extract topic signals directly from the text using keyword
matching against MeSH top-level categories.

Usage:
    from document_topic_scorer import DocumentTopicScorer

    scorer = DocumentTopicScorer(boost=3.0, penalty=-5.0)
    topic = scorer.detect_topic(abstract_text)
    candidates = scorer.rescore(candidates, topic)

Can be disabled with --no-topic-scoring flag.
"""

import re
from collections import Counter


# ── MeSH Tree top-level categories ──────────────────────────────────────
# https://www.nlm.nih.gov/mesh/intro_trees.html

TREE_CATEGORIES = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals and Drugs",
    "E": "Analytical, Diagnostic and Therapeutic Techniques, and Equipment",
    "F": "Psychiatry and Psychology",
    "G": "Phenomena and Processes",
    "H": "Disciplines and Occupations",
    "I": "Anthropology, Education, Sociology, and Social Phenomena",
    "J": "Technology, Industry, and Agriculture",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}

# ── Sub-category keywords ───────────────────────────────────────────────
# Maps keyword patterns → (tree_prefix, weight)
# More specific patterns get higher weight

TOPIC_KEYWORDS = {
    # ── Neoplasms (C04) ──
    "cancer": ("C04", 3),
    "tumor": ("C04", 3),
    "tumour": ("C04", 3),
    "carcinoma": ("C04", 3),
    "neoplasm": ("C04", 3),
    "oncol": ("C04", 2),
    "metastas": ("C04", 3),
    "malignant": ("C04", 2),
    "lymphoma": ("C04", 3),
    "leukemia": ("C04", 3),
    "leukaemia": ("C04", 3),
    "melanoma": ("C04", 3),
    "sarcoma": ("C04", 3),
    "glioma": ("C04", 3),
    "adenocarcinoma": ("C04", 3),

    # ── Cardiovascular (C14) ──
    "cardiovascular": ("C14", 3),
    "cardiac": ("C14", 2),
    "heart failure": ("C14", 3),
    "myocardial": ("C14", 3),
    "arrhythmia": ("C14", 3),
    "atherosclerosis": ("C14", 3),
    "hypertension": ("C14", 2),
    "stroke": ("C14", 2),
    "coronary": ("C14", 3),
    "aortic": ("C14", 2),

    # ── Nervous System (C10) ──
    "neurodegenerat": ("C10", 3),
    "alzheimer": ("C10", 3),
    "parkinson": ("C10", 3),
    "epilepsy": ("C10", 3),
    "seizure": ("C10", 2),
    "neuropath": ("C10", 2),
    "multiple sclerosis": ("C10", 3),
    "dementia": ("C10", 3),

    # ── Respiratory (C08) ──
    "pulmonary": ("C08", 2),
    "respiratory": ("C08", 2),
    "asthma": ("C08", 3),
    "copd": ("C08", 3),
    "pneumonia": ("C08", 3),
    "lung disease": ("C08", 3),

    # ── Digestive System (C06) ──
    "hepat": ("C06", 2),
    "liver": ("C06", 2),
    "gastric": ("C06", 2),
    "intestinal": ("C06", 2),
    "colitis": ("C06", 3),
    "cirrhosis": ("C06", 3),
    "pancreatitis": ("C06", 3),

    # ── Musculoskeletal (C05) ──
    "arthritis": ("C05", 3),
    "osteoporosis": ("C05", 3),
    "rheumat": ("C05", 2),

    # ── Endocrine (C19) ──
    "diabetes": ("C19", 3),
    "diabetic": ("C19", 3),
    "thyroid": ("C19", 2),
    "insulin": ("C19", 2),

    # ── Immune System (C20) ──
    "autoimmune": ("C20", 3),
    "immunodeficiency": ("C20", 3),
    "allerg": ("C20", 2),
    "lupus": ("C20", 3),

    # ── Infections (C01) ──
    "infection": ("C01", 2),
    "bacterial": ("C01", 2),
    "viral": ("C01", 2),
    "sepsis": ("C01", 3),
    "tuberculosis": ("C01", 3),
    "hiv": ("C01", 2),
    "hepatitis": ("C01", 3),
    "malaria": ("C01", 3),

    # ── Mental Disorders (F03) ──
    "depression": ("F03", 2),
    "anxiety": ("F03", 2),
    "schizophrenia": ("F03", 3),
    "bipolar": ("F03", 3),
    "psychiatric": ("F03", 2),

    # ── Kidney/Urological (C12/C13) ──
    "renal": ("C12", 2),
    "kidney": ("C12", 2),
    "nephro": ("C12", 2),
    "dialysis": ("C12", 2),

    # ── Chemicals/Drugs (D) — broad ──
    "inhibitor": ("D", 1),
    "receptor": ("D", 1),
    "enzyme": ("D", 1),
    "protein": ("D", 1),
    "antibod": ("D", 1),
    "cytokine": ("D", 1),
    "kinase": ("D", 1),
}


class DocumentTopicScorer:
    """
    Detects document topic from abstract text and re-scores candidates
    based on taxonomy consistency.

    Parameters
    ----------
    boost : float
        Score boost for candidates matching the document topic.
    penalty : float
        Score penalty for candidates conflicting with the document topic.
    min_signal : int
        Minimum total keyword weight to activate topic scoring.
        Below this threshold, the document topic is too uncertain.
    """

    def __init__(self, boost: float = 3.0, penalty: float = -3.0, min_signal: int = 4):
        self.boost = boost
        self.penalty = penalty
        self.min_signal = min_signal

    def detect_topic(self, text: str) -> dict[str, float]:
        """
        Detect dominant MeSH categories from document text.

        Returns dict of {tree_prefix: weight}, e.g. {"C04": 9, "D": 3}.
        Only returns categories with weight >= min_signal.
        """
        text_lower = text.lower()
        scores = Counter()

        for keyword, (tree_prefix, weight) in TOPIC_KEYWORDS.items():
            # Count occurrences (but cap at 3 to avoid one repeated word dominating)
            count = min(len(re.findall(re.escape(keyword), text_lower)), 3)
            if count > 0:
                scores[tree_prefix] += weight * count

        # Filter to categories with enough signal
        return {cat: w for cat, w in scores.items() if w >= self.min_signal}

    def rescore(self, candidates: list, topic: dict[str, float]) -> list:
        """
        Re-score candidates based on document topic consistency.

        Candidates whose tree numbers match the dominant topic get boosted.
        Candidates in clearly different disease categories get penalized
        (only if topic signal is strong and candidate is in a disease
        sub-category that conflicts).

        Parameters
        ----------
        candidates : list[CandidateEntity]
            Candidates to re-score (modified in place).
        topic : dict[str, float]
            Document topic weights from detect_topic().

        Returns
        -------
        list[CandidateEntity]
            Re-sorted candidates.
        """
        if not topic or not candidates:
            return candidates

        # Get the dominant sub-categories (e.g., "C04", "C14")
        # Only use specific sub-categories for penalty, not broad ones like "D"
        dominant_subcats = {cat for cat in topic if len(cat) > 1}
        # Top-level categories present (for boosting)
        dominant_top = {cat[0] for cat in topic}

        for c in candidates:
            tree_numbers = getattr(c, "tree_numbers", [])
            if not tree_numbers:
                continue

            # Extract this candidate's categories
            cand_subcats = set()
            cand_top = set()
            for tn in tree_numbers:
                if tn:
                    cand_top.add(tn[0])
                    if len(tn) >= 3:
                        cand_subcats.add(tn[:3])

            # Boost: candidate matches a dominant sub-category
            if cand_subcats & dominant_subcats:
                c.score += self.boost

            # Penalty: candidate is in a *different* disease sub-category
            # Only apply when:
            #   1. We have strong topic signal (specific sub-categories detected)
            #   2. The candidate is in a disease category (C)
            #   3. The candidate's sub-category doesn't match ANY dominant sub-category
            elif (dominant_subcats
                  and "C" in cand_top
                  and cand_subcats
                  and not (cand_subcats & dominant_subcats)
                  # Only penalize if we have a dominant disease sub-cat
                  and any(cat.startswith("C") for cat in dominant_subcats)):
                c.score += self.penalty

        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates

    def rescore_with_details(self, candidates: list, topic: dict[str, float]) -> list[dict]:
        """Like rescore(), but returns detailed info for debugging."""
        if not topic or not candidates:
            return [{"candidate": c, "adjustment": 0.0, "reason": "no_topic"} for c in candidates]

        dominant_subcats = {cat for cat in topic if len(cat) > 1}
        dominant_top = {cat[0] for cat in topic}
        has_disease_topic = any(cat.startswith("C") for cat in dominant_subcats)

        results = []
        for c in candidates:
            tree_numbers = getattr(c, "tree_numbers", [])
            adj = 0.0
            reason = "neutral"

            if tree_numbers:
                cand_subcats = set()
                cand_top = set()
                for tn in tree_numbers:
                    if tn:
                        cand_top.add(tn[0])
                        if len(tn) >= 3:
                            cand_subcats.add(tn[:3])

                if cand_subcats & dominant_subcats:
                    adj = self.boost
                    reason = f"topic_match:{cand_subcats & dominant_subcats}"
                elif (dominant_subcats and "C" in cand_top and cand_subcats
                      and not (cand_subcats & dominant_subcats) and has_disease_topic):
                    adj = self.penalty
                    reason = f"topic_conflict:{cand_subcats} vs {dominant_subcats}"

            c.score += adj
            results.append({"candidate": c, "adjustment": adj, "reason": reason})

        candidates.sort(key=lambda c: c.score, reverse=True)
        results.sort(key=lambda x: x["candidate"].score, reverse=True)
        return results


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    scorer = DocumentTopicScorer()

    test_abstracts = [
        "This study investigated the role of BRCA1 mutations in breast cancer. "
        "Patients with metastatic carcinoma showed poor prognosis. Tumor size "
        "was correlated with overall survival in this oncology cohort.",

        "We examined the effects of myocardial infarction on cardiac function. "
        "Heart failure patients showed reduced ejection fraction. Coronary artery "
        "disease was the primary cause of cardiovascular events.",

        "Alzheimer's disease is a neurodegenerative disorder characterized by "
        "cognitive decline. Patients with dementia showed amyloid plaques. "
        "Parkinson's disease was also observed as a comorbidity.",
    ]

    for i, abstract in enumerate(test_abstracts):
        topic = scorer.detect_topic(abstract)
        print(f"\nAbstract {i+1}:")
        print(f"  Topics: {topic}")
        dominant = max(topic, key=topic.get) if topic else "none"
        print(f"  Dominant: {dominant} ({TREE_CATEGORIES.get(dominant[0], dominant)})")
