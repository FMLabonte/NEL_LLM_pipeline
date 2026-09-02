"""
Standalone Okapi BM25 retriever over MeSH labels.

A *named* lexical retriever for the retriever-strength curve (Figure 1). Unlike the
default `rapidfuzz` backend (token-sort fuzzy matching) this is genuine Okapi BM25
(the `rank_bm25` implementation), so it can be cited as "BM25" without an
Elasticsearch daemon. Interface mirrors `EmbeddingRetriever`: `retrieve(mention,
top_k)` returns a list of `CandidateEntity` sorted by BM25 score.

Usage (inside evaluate_pipeline, via --bm25-only):
    r = BM25Retriever(mesh_index)
    r.build_or_load(cache_dir)
    cands = r.retrieve("heart attack", top_k=10)
"""
from __future__ import annotations

import pickle
import re
from pathlib import Path

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


class BM25Retriever:
    """Okapi BM25 over every MeSH label (preferred term + synonyms)."""

    def __init__(self, mesh_index, max_synonyms: int = 50):
        if not HAS_BM25:
            raise RuntimeError(
                "rank_bm25 not installed. Install with: pip install rank_bm25"
            )
        self.mesh_index = mesh_index
        self.max_synonyms = max_synonyms
        self.bm25 = None
        self._labels: list[str] = []       # surface label per BM25 doc
        self._mesh_ids: list[str] = []      # mesh_id per BM25 doc (parallel)

    # ── index construction ──
    def _collect_labels(self):
        labels, mesh_ids = [], []
        for mid, ent in self.mesh_index.entities.items():
            names = [ent.preferred_label] + list(ent.synonyms or [])[: self.max_synonyms]
            for name in names:
                if not name:
                    continue
                labels.append(name)
                mesh_ids.append(mid)
        return labels, mesh_ids

    def build_index(self):
        self._labels, self._mesh_ids = self._collect_labels()
        corpus = [_tokenize(l) for l in self._labels]
        print(f"  BM25: indexing {len(corpus):,} MeSH labels ...")
        self.bm25 = BM25Okapi(corpus)
        print("  BM25: index ready.")

    def build_or_load(self, cache_dir: str):
        cache = Path(cache_dir)
        cache.mkdir(parents=True, exist_ok=True)
        f = cache / "bm25_mesh.pkl"
        if f.exists():
            try:
                with open(f, "rb") as fh:
                    obj = pickle.load(fh)
                if obj.get("n_entities") == len(self.mesh_index.entities):
                    self.bm25 = obj["bm25"]
                    self._labels = obj["labels"]
                    self._mesh_ids = obj["mesh_ids"]
                    print(f"  BM25: loaded cached index ({len(self._labels):,} labels).")
                    return
                print("  BM25: cached index stale (entity count changed), rebuilding.")
            except Exception as e:
                print(f"  BM25: cache load failed ({e}), rebuilding.")
        self.build_index()
        with open(f, "wb") as fh:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "labels": self._labels,
                    "mesh_ids": self._mesh_ids,
                    "n_entities": len(self.mesh_index.entities),
                },
                fh,
            )

    # ── retrieval ──
    def retrieve(self, mention: str, top_k: int = 10) -> list:
        from mesh_index import CandidateEntity  # avoid circular import

        if self.bm25 is None:
            raise RuntimeError("BM25 index not built. Call build_or_load() first.")
        q = _tokenize(mention)
        if not q:
            return []
        scores = self.bm25.get_scores(q)  # np.ndarray over all labels

        # Deduplicate by mesh_id, keeping the highest-scoring label.
        best: dict[str, tuple[float, str]] = {}
        # Only look at the top (top_k * 40) label positions for speed.
        import numpy as np

        n_look = min(len(scores), max(top_k * 40, 200))
        top_idx = np.argpartition(scores, -n_look)[-n_look:]
        for idx in top_idx:
            s = float(scores[idx])
            if s <= 0:
                continue
            mid = self._mesh_ids[idx]
            if mid not in best or s > best[mid][0]:
                best[mid] = (s, self._labels[idx])

        cands = []
        for mid, (s, matched) in best.items():
            ent = self.mesh_index.entities.get(mid)
            if ent is None:
                continue
            cands.append(
                CandidateEntity(
                    mesh_id=mid,
                    preferred_label=ent.preferred_label,
                    synonyms=ent.synonyms,
                    definition=ent.definition,
                    tree_numbers=ent.tree_numbers,
                    score=s,
                    matched_synonym=matched,
                )
            )
        cands.sort(key=lambda c: c.score, reverse=True)
        return cands[:top_k]
