"""
Learned Ranker for Entity Linking (XGBoost)
=============================================
Replaces or augments hand-tuned Phase 3 domain rules with a trained
XGBoost ranking model that learns optimal feature weights from data.

Features include string matching, embedding similarity, candidate
quality, MeSH taxonomy (tree) structure, and context overlap —
effectively combining domain rules AND taxonomy graph checking
into a single learned model.

Training workflow:
    # Step 1: Train on BC5CDR training set
    python3 src/evaluate_pipeline.py \\
        --split train --train-ranker \\
        --embedding --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF

    # Step 2: Evaluate on test set with learned ranker
    python3 src/evaluate_pipeline.py \\
        --learned-ranker src/improvements/cache/ranker_model.json \\
        --embedding --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF
"""

import json
import pickle
import numpy as np
from pathlib import Path

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

from sklearn.ensemble import GradientBoostingClassifier


# ── Feature definitions ─────────────────────────────────────────────────

FEATURE_NAMES = [
    # String / matching features
    "string_score",
    "mention_length",
    "mention_word_count",
    "label_length",
    "label_word_count",
    "exact_match",
    "partial_match",
    "word_overlap",
    "char_ratio",
    # Candidate quality
    "candidate_rank",
    "score_gap_to_top1",
    "score_ratio",
    "has_definition",
    "definition_length",
    "num_synonyms",
    # Taxonomy (MeSH tree) features
    "max_tree_depth",
    "min_tree_depth",
    "num_tree_numbers",
    "is_supplementary",
    "tree_cat_C",          # has Diseases tree
    "tree_cat_D",          # has Chemicals tree
    "tree_cat_match_entity",
    # Context features
    "mention_in_definition",
    "label_words_in_context",
    "definition_words_in_context",
]


# ── Feature extraction ───────────────────────────────────────────────────

def extract_candidate_features(
    mention: str,
    candidate,
    rank: int,
    top1_score: float,
    entity_type: str | None = None,
    context: str = "",
) -> dict:
    """
    Extract features for a single (mention, candidate) pair.

    These features capture signals from string matching, embedding
    similarity, candidate metadata, MeSH taxonomy structure, and
    context overlap — everything the hand-tuned domain rules use,
    plus additional signals.
    """
    mention_lower = mention.lower()
    label_lower = candidate.preferred_label.lower()
    mention_words = set(mention_lower.split())
    label_words = set(label_lower.split())

    # ── String / matching ──
    features = {
        "string_score": candidate.score,
        "mention_length": len(mention),
        "mention_word_count": len(mention_words),
        "label_length": len(candidate.preferred_label),
        "label_word_count": len(label_words),
        "exact_match": float(mention_lower == label_lower),
        "partial_match": float(
            mention_lower in label_lower or label_lower in mention_lower
        ),
        "word_overlap": (
            len(mention_words & label_words)
            / max(len(mention_words | label_words), 1)
        ),
        "char_ratio": len(mention) / max(len(candidate.preferred_label), 1),
    }

    # ── Candidate quality ──
    features["candidate_rank"] = rank
    features["score_gap_to_top1"] = top1_score - candidate.score
    features["score_ratio"] = candidate.score / max(top1_score, 0.01)
    features["has_definition"] = float(bool(candidate.definition))
    features["definition_length"] = (
        len(candidate.definition) if candidate.definition else 0
    )
    features["num_synonyms"] = (
        len(candidate.synonyms) if candidate.synonyms else 0
    )

    # ── Taxonomy (MeSH tree) features ──
    tree_numbers = getattr(candidate, "tree_numbers", []) or []
    if tree_numbers:
        depths = [len(tn.split(".")) for tn in tree_numbers]
        features["max_tree_depth"] = max(depths)
        features["min_tree_depth"] = min(depths)
        features["num_tree_numbers"] = len(tree_numbers)

        first_letters = {tn[0] for tn in tree_numbers if tn}
        features["tree_cat_C"] = float("C" in first_letters)
        features["tree_cat_D"] = float("D" in first_letters)

        # Does tree category match entity type?
        if entity_type in ("Disease", "DiseaseOrPhenotypicFeature"):
            features["tree_cat_match_entity"] = float("C" in first_letters)
        elif entity_type in ("Chemical", "ChemicalEntity"):
            features["tree_cat_match_entity"] = float("D" in first_letters)
        else:
            features["tree_cat_match_entity"] = 0.5
    else:
        features["max_tree_depth"] = 0
        features["min_tree_depth"] = 0
        features["num_tree_numbers"] = 0
        features["tree_cat_C"] = 0.0
        features["tree_cat_D"] = 0.0
        features["tree_cat_match_entity"] = 0.5

    features["is_supplementary"] = float(
        getattr(candidate, "is_supplementary", False)
    )

    # ── Context features ──
    definition = (candidate.definition or "").lower()
    features["mention_in_definition"] = (
        float(mention_lower in definition) if definition else 0.0
    )

    context_lower = context.lower() if context else ""
    if label_words and context_lower:
        features["label_words_in_context"] = (
            sum(1 for w in label_words if w in context_lower) / len(label_words)
        )
    else:
        features["label_words_in_context"] = 0.0

    # Definition words in context (captures domain rule 7 signal)
    if definition and context_lower:
        def_words = set(definition.split())
        # Filter to meaningful words (>3 chars)
        def_words = {w for w in def_words if len(w) > 3}
        if def_words:
            features["definition_words_in_context"] = (
                sum(1 for w in def_words if w in context_lower) / len(def_words)
            )
        else:
            features["definition_words_in_context"] = 0.0
    else:
        features["definition_words_in_context"] = 0.0

    return features


def extract_features_for_mention(
    mention: str,
    candidates: list,
    entity_type: str | None = None,
    context: str = "",
) -> list[dict]:
    """Extract features for all candidates of a single mention."""
    if not candidates:
        return []

    top1_score = candidates[0].score

    return [
        extract_candidate_features(
            mention=mention,
            candidate=c,
            rank=i,
            top1_score=top1_score,
            entity_type=entity_type,
            context=context,
        )
        for i, c in enumerate(candidates)
    ]


# ── Learned Ranker ───────────────────────────────────────────────────────

class LearnedRanker:
    """
    Learned ranker for candidate re-ranking.

    Uses sklearn GradientBoostingClassifier (pointwise) by default.
    Falls back gracefully — no native dependencies that segfault on macOS.

    The model predicts P(candidate is correct) for each candidate,
    then re-ranks by predicted probability.

    Parameters
    ----------
    model_path : str or None
        Path to a saved model (.pkl). If provided, loads immediately.
    """

    def __init__(self, model_path: str | None = None):
        self.model = None
        self.feature_names = FEATURE_NAMES
        self._backend = "sklearn"

        if model_path:
            if Path(model_path).exists():
                self.load(model_path)
            else:
                raise FileNotFoundError(
                    f"Ranker model not found: {model_path}"
                )

    def _features_to_matrix(self, feature_dicts: list[dict]) -> np.ndarray:
        """Convert list of feature dicts to numpy matrix."""
        return np.array(
            [
                [fd.get(name, 0.0) for name in self.feature_names]
                for fd in feature_dicts
            ],
            dtype=np.float32,
        )

    def train(
        self,
        all_features: list[list[dict]],
        all_labels: list[list[int]],
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        verbose: bool = True,
    ):
        """
        Train the ranker on collected features.

        Parameters
        ----------
        all_features : list of list of dict
            For each mention: list of feature dicts (one per candidate).
        all_labels : list of list of int
            For each mention: list of 0/1 labels (1 = gold candidate).
        """
        # Flatten into single arrays
        X_rows = []
        y_rows = []

        for mention_feats, mention_labels in zip(all_features, all_labels):
            if not mention_feats:
                continue
            X_rows.extend(mention_feats)
            y_rows.extend(mention_labels)

        X = self._features_to_matrix(X_rows)
        y = np.array(y_rows, dtype=np.int32)

        n_pos = int(y.sum())
        n_neg = len(y) - n_pos
        n_mentions = len(all_features)

        if verbose:
            print(f"\n  Training Learned Ranker (sklearn GradientBoosting):")
            print(f"    Mentions:          {n_mentions}")
            print(f"    Total candidates:  {len(y)}")
            print(f"    Positive labels:   {n_pos} ({n_pos/len(y)*100:.1f}%)")
            print(f"    Negative labels:   {n_neg}")
            print(f"    Params: n_estimators={n_estimators}, "
                  f"max_depth={max_depth}, lr={learning_rate}")

        # Use sample weights to handle class imbalance
        # (each mention has ~20 negatives and ~1 positive)
        weight_pos = n_neg / max(n_pos, 1)
        sample_weights = np.where(y == 1, weight_pos, 1.0)

        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=0.8,
            max_features=0.8,
            random_state=42,
            verbose=1 if verbose else 0,
        )

        self.model.fit(X, y, sample_weight=sample_weights)

        if verbose:
            # Show feature importance
            importance = self.model.feature_importances_
            sorted_idx = np.argsort(importance)[::-1]
            print(f"\n  Top features by importance:")
            for i in sorted_idx[:12]:
                print(
                    f"    {self.feature_names[i]:35s} "
                    f"{importance[i]:.4f}"
                )

            # Quick train-set accuracy check
            train_probs = self.model.predict_proba(X)[:, 1]
            # Check how often top-prob candidate is correct per mention
            idx = 0
            correct = 0
            total_m = 0
            for mention_feats, mention_labels in zip(
                all_features, all_labels
            ):
                if not mention_feats:
                    continue
                n = len(mention_feats)
                probs = train_probs[idx:idx + n]
                labs = y[idx:idx + n]
                best = np.argmax(probs)
                if labs[best] == 1:
                    correct += 1
                total_m += 1
                idx += n
            print(f"\n  Train-set re-ranking Accuracy@1: "
                  f"{correct/total_m*100:.1f}% ({correct}/{total_m})")

    def rerank(
        self,
        mention: str,
        candidates: list,
        entity_type: str | None = None,
        context: str = "",
    ) -> list:
        """
        Re-rank candidates using the trained model.

        Returns candidates sorted by predicted probability of being correct.
        """
        if self.model is None or not candidates:
            return candidates

        features = extract_features_for_mention(
            mention, candidates, entity_type, context,
        )
        X = self._features_to_matrix(features)
        # predict_proba returns [P(neg), P(pos)] — use P(pos)
        scores = self.model.predict_proba(X)[:, 1]

        # Sort candidates by predicted probability (higher = better)
        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        return [c for c, _ in ranked]

    def save(self, path: str):
        """Save trained model as pickle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {"model": self.model, "feature_names": self.feature_names},
                f,
            )
        print(f"  Saved ranker model to {path}")

    def load(self, path: str):
        """Load trained model from pickle."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.model = data["model"]
        self.feature_names = data.get("feature_names", FEATURE_NAMES)
        print(f"  Loaded ranker model from {path}")


# ── Quick test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("LearnedRanker ready (sklearn GradientBoosting).")
    print("Train via: evaluate_pipeline.py --split train --train-ranker")
    print("Use via:   evaluate_pipeline.py --learned-ranker PATH")
