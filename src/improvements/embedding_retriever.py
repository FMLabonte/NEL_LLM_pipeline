"""
Embedding-Based Retriever (SapBERT + FAISS)
=============================================
Complementary retrieval method for Phase 2 that uses dense
embeddings instead of string matching.

SapBERT (Self-Alignment Pretraining for BERT) is pre-trained on
UMLS synonyms to place biomedical synonyms close together in
embedding space. This means "heart attack" and "Myocardial Infarction"
have high cosine similarity even though they share zero tokens.

Architecture:
  1. Build phase: Encode all MeSH labels (preferred + synonyms) with
     SapBERT → build a FAISS index over these embeddings.
  2. Query phase: Encode mention → find k-nearest neighbors in FAISS
     → return CandidateEntity objects with cosine similarity scores.

The FAISS index and metadata are cached to disk so the expensive
encoding step only runs once (~15-30 min for ~995K labels).

Usage:
    from embedding_retriever import EmbeddingRetriever

    retriever = EmbeddingRetriever(mesh_index)
    retriever.build_index()  # or retriever.load_index("cache/faiss/")
    candidates = retriever.retrieve("heart attack", top_k=10)

Paper reference: "Planned Improvements" in CLAUDE.md
"""

import os
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass

try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False


# ── Default model ────────────────────────────────────────────────────────

# SapBERT: trained on UMLS synonym pairs, best for biomedical entity linking
DEFAULT_MODEL = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext"

# Alternative: BioLinkBERT (general biomedical, not synonym-specialized)
# DEFAULT_MODEL = "michiyasunaga/BioLinkBERT-base"


class EmbeddingRetriever:
    """
    Dense embedding retriever using SapBERT + FAISS.

    Parameters
    ----------
    mesh_index : MeSHIndex
        The built MeSH index (used for entity data).
    model_name : str
        HuggingFace model name for encoding.
    device : str or None
        "cuda", "mps", or "cpu". Auto-detected if None.
    batch_size : int
        Batch size for encoding labels (adjust based on GPU memory).
    max_length : int
        Maximum token length for the model.
    """

    def __init__(
        self,
        mesh_index=None,
        model_name: str = DEFAULT_MODEL,
        device: str | None = None,
        batch_size: int = 256,
        max_length: int = 64,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install torch transformers"
            )
        if not HAS_FAISS:
            raise ImportError(
                "faiss is required. Install with: pip install faiss-cpu"
            )

        self.mesh_index = mesh_index
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length

        # Auto-detect device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Load model and tokenizer
        print(f"Loading embedding model: {model_name}")
        print(f"  Device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # FAISS index and metadata
        self.faiss_index = None
        self._index_labels: list[str] = []       # label at each FAISS position
        self._index_mesh_ids: list[str] = []      # mesh_id at each FAISS position
        self._embedding_dim: int = 0

    # ── Encoding ──────────────────────────────────────────────────────────

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        """
        Encode a batch of texts into normalized embeddings.

        Uses [CLS] token pooling (standard for SapBERT).
        Returns L2-normalized vectors for cosine similarity via inner product.
        """
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]

        # L2 normalize → inner product = cosine similarity
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    def encode_texts(self, texts: list[str], verbose: bool = False) -> np.ndarray:
        """
        Encode a list of texts into embeddings, processing in batches.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.
        verbose : bool
            If True, print progress (useful for large index builds).

        Returns
        -------
        np.ndarray
            Shape (len(texts), embedding_dim), L2-normalized.
        """
        all_embeddings = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            embs = self._encode_batch(batch)
            all_embeddings.append(embs)

            if verbose:
                batch_num = i // self.batch_size + 1
                if batch_num % 50 == 0 or batch_num == n_batches:
                    print(f"  Encoded {min(i + self.batch_size, len(texts))}/{len(texts)} labels "
                          f"({batch_num}/{n_batches} batches)", flush=True)

        return np.vstack(all_embeddings)

    # ── Building the FAISS index ──────────────────────────────────────────

    def build_index(self, max_synonyms_per_entity: int = 10):
        """
        Build a FAISS index from all MeSH entity labels.

        Encodes every preferred label and synonym, stores the mapping
        from FAISS position → (label, mesh_id).

        Parameters
        ----------
        max_synonyms_per_entity : int
            Max synonyms per entity to encode (limits index size).
        """
        if self.mesh_index is None:
            raise ValueError("No MeSH index provided. Pass mesh_index to constructor.")

        print(f"\nBuilding FAISS embedding index...")
        print(f"  Entities in MeSH index: {self.mesh_index.size}")

        # Collect all labels to encode
        labels = []
        mesh_ids = []

        for mesh_id, entity in self.mesh_index.entities.items():
            # Preferred label
            labels.append(entity.preferred_label)
            mesh_ids.append(mesh_id)

            # Synonyms (up to max, excluding preferred label)
            added = 0
            for syn in entity.synonyms:
                if syn.lower() != entity.preferred_label.lower():
                    labels.append(syn)
                    mesh_ids.append(mesh_id)
                    added += 1
                    if added >= max_synonyms_per_entity:
                        break

        print(f"  Total labels to encode: {len(labels)}")

        # Encode all labels
        print(f"  Encoding with {self.model_name}...")
        embeddings = self.encode_texts(labels, verbose=True)
        self._embedding_dim = embeddings.shape[1]

        # Build FAISS index (inner product = cosine similarity for normalized vectors)
        print(f"  Building FAISS index (dim={self._embedding_dim})...")
        self.faiss_index = faiss.IndexFlatIP(self._embedding_dim)
        self.faiss_index.add(embeddings.astype(np.float32))

        self._index_labels = labels
        self._index_mesh_ids = mesh_ids

        print(f"  FAISS index built: {self.faiss_index.ntotal} vectors")

    # ── Saving and loading ────────────────────────────────────────────────

    def save_index(self, cache_dir: str):
        """
        Save FAISS index and metadata to disk for fast reloading.
        """
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.faiss_index, str(cache_path / "faiss.index"))

        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self._embedding_dim,
            "num_vectors": len(self._index_labels),
            "labels": self._index_labels,
            "mesh_ids": self._index_mesh_ids,
        }
        with open(cache_path / "metadata.json", "w") as f:
            json.dump(metadata, f)

        print(f"  Index saved to {cache_dir}/")

    def load_index(self, cache_dir: str) -> bool:
        """
        Load a previously saved FAISS index from disk.

        Returns True if loaded successfully, False if cache not found.
        """
        cache_path = Path(cache_dir)
        index_file = cache_path / "faiss.index"
        meta_file = cache_path / "metadata.json"

        if not index_file.exists() or not meta_file.exists():
            return False

        print(f"Loading cached FAISS index from {cache_dir}/...")

        # Load metadata
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        # Verify model matches
        if metadata["model_name"] != self.model_name:
            print(f"  Warning: cached index was built with {metadata['model_name']}, "
                  f"but current model is {self.model_name}. Rebuilding.")
            return False

        self._embedding_dim = metadata["embedding_dim"]
        self._index_labels = metadata["labels"]
        self._index_mesh_ids = metadata["mesh_ids"]

        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_file))
        print(f"  Loaded: {self.faiss_index.ntotal} vectors (dim={self._embedding_dim})")

        return True

    def build_or_load(self, cache_dir: str, max_synonyms_per_entity: int = 10):
        """
        Try to load from cache, build if not available.
        """
        if not self.load_index(cache_dir):
            self.build_index(max_synonyms_per_entity=max_synonyms_per_entity)
            self.save_index(cache_dir)

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, mention: str, top_k: int = 10) -> list:
        """
        Retrieve top-k candidates for a mention using embedding similarity.

        Parameters
        ----------
        mention : str
            The entity mention to search for.
        top_k : int
            Number of candidates to return.

        Returns
        -------
        list[CandidateEntity]
            Candidates sorted by cosine similarity (highest first).
            Scores are scaled to 0-100 range for compatibility with rapidfuzz.
        """
        if self.faiss_index is None:
            raise RuntimeError("FAISS index not built. Call build_index() or load_index() first.")

        # Import CandidateEntity here to avoid circular imports
        from mesh_index import CandidateEntity

        # Encode the mention
        query_emb = self._encode_batch([mention])

        # Search FAISS
        scores, indices = self.faiss_index.search(query_emb.astype(np.float32), top_k * 3)
        # top_k * 3 because multiple labels may map to the same mesh_id

        # Deduplicate by mesh_id, keeping the highest score
        seen = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue  # FAISS returns -1 for unfilled slots
            mesh_id = self._index_mesh_ids[idx]
            matched_label = self._index_labels[idx]

            # Cosine similarity is in [-1, 1], scale to 0-100
            scaled_score = float(score) * 100.0

            if mesh_id not in seen or scaled_score > seen[mesh_id][0]:
                seen[mesh_id] = (scaled_score, matched_label)

        # Build CandidateEntity objects
        candidates = []
        for mesh_id, (score, matched_label) in seen.items():
            entity = self.mesh_index.entities.get(mesh_id)
            if entity is None:
                continue
            candidates.append(CandidateEntity(
                mesh_id=mesh_id,
                preferred_label=entity.preferred_label,
                synonyms=entity.synonyms,
                definition=entity.definition,
                tree_numbers=entity.tree_numbers,
                score=score,
                matched_synonym=matched_label,
            ))

        # Sort by score descending and return top_k
        candidates.sort(key=lambda c: c.score, reverse=True)
        return candidates[:top_k]

    def score_candidates(self, mention: str, candidates: list) -> dict[str, float]:
        """
        Compute embedding similarity between a mention and existing candidates.

        Encodes each candidate's preferred_label and returns cosine similarity.

        Parameters
        ----------
        mention : str
            The entity mention.
        candidates : list[CandidateEntity]
            Existing candidates to score.

        Returns
        -------
        dict[str, float]
            mesh_id → embedding similarity score (0-100 scale).
        """
        if not candidates:
            return {}

        # Encode the mention
        mention_emb = self._encode_batch([mention])  # (1, dim)

        # Encode preferred labels
        labels = [c.preferred_label for c in candidates]
        label_embs = self.encode_texts(labels)  # (n, dim)

        # Cosine similarity (vectors are already normalized)
        similarities = np.dot(label_embs, mention_emb.T).flatten()

        # Map to mesh_id, scaled to 0-100
        scores = {}
        for c, sim in zip(candidates, similarities):
            scores[c.mesh_id] = float(sim) * 100.0

        return scores

    def retrieve_with_details(self, mention: str, top_k: int = 10) -> dict:
        """
        Like retrieve(), but also returns the raw embedding similarity info.
        Useful for debugging and analysis.
        """
        candidates = self.retrieve(mention, top_k)
        query_emb = self._encode_batch([mention])

        return {
            "mention": mention,
            "candidates": candidates,
            "query_embedding_norm": float(np.linalg.norm(query_emb)),
        }


# ── Multi-Model Retriever ──────────────────────────────────────────────────

class MultiEmbeddingRetriever:
    """
    Combines multiple embedding retrievers (e.g., SapBERT + BioLinkBERT).

    Merges candidates from all models, keeping the highest score per mesh_id.
    For score_candidates(), averages the scores across models.

    Parameters
    ----------
    mesh_index : MeSHIndex
        The built MeSH index.
    model_names : list[str]
        List of HuggingFace model names.
    device : str or None
        Device for all models.
    batch_size : int
        Batch size for encoding.
    """

    def __init__(
        self,
        mesh_index=None,
        model_names: list[str] | None = None,
        device: str | None = None,
        batch_size: int = 256,
    ):
        if model_names is None:
            model_names = [
                "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                "michiyasunaga/BioLinkBERT-base",
            ]

        self.retrievers = []
        for name in model_names:
            r = EmbeddingRetriever(
                mesh_index=mesh_index,
                model_name=name,
                device=device,
                batch_size=batch_size,
            )
            self.retrievers.append(r)

    def build_or_load(self, base_cache_dir: str, max_synonyms_per_entity: int = 10):
        """Build or load FAISS index for each model."""
        for r in self.retrievers:
            # Each model gets its own cache subdirectory
            model_slug = r.model_name.replace("/", "_")
            cache_dir = f"{base_cache_dir}/{model_slug}"
            r.build_or_load(cache_dir, max_synonyms_per_entity)

    def retrieve(self, mention: str, top_k: int = 10) -> list:
        """
        Retrieve candidates from all models, merge by highest score.
        """
        from mesh_index import CandidateEntity

        best_by_id = {}  # mesh_id → CandidateEntity (with highest score)

        for r in self.retrievers:
            candidates = r.retrieve(mention, top_k=top_k)
            for c in candidates:
                if c.mesh_id not in best_by_id or c.score > best_by_id[c.mesh_id].score:
                    best_by_id[c.mesh_id] = c

        merged = sorted(best_by_id.values(), key=lambda c: c.score, reverse=True)
        return merged[:top_k]

    def score_candidates(self, mention: str, candidates: list) -> dict[str, float]:
        """
        Score candidates using all models, return average score per mesh_id.
        """
        all_scores = []
        for r in self.retrievers:
            scores = r.score_candidates(mention, candidates)
            all_scores.append(scores)

        # Average across models
        averaged = {}
        for c in candidates:
            model_scores = [s.get(c.mesh_id, 0.0) for s in all_scores]
            averaged[c.mesh_id] = sum(model_scores) / len(model_scores)

        return averaged


# ── Quick demo ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("Embedding Retriever — Quick Demo")
    print("=" * 60)
    print()

    # Test encoding only (without full MeSH index)
    retriever = EmbeddingRetriever(mesh_index=None)

    test_pairs = [
        ("heart attack", "Myocardial Infarction"),
        ("heart attack", "Cardiac Arrest"),
        ("lung cancer", "Pulmonary Neoplasms"),
        ("aspirin", "Acetylsalicylic Acid"),
        ("diabetes", "Diabetes Mellitus"),
        ("headache", "Cephalalgia"),
        ("high blood pressure", "Hypertension"),
        ("seizures", "Seizures"),
        # Negative pairs
        ("heart attack", "Hepatitis"),
        ("lung cancer", "Diabetes Mellitus"),
    ]

    print("Cosine similarity between biomedical term pairs:")
    print("-" * 60)

    for term1, term2 in test_pairs:
        emb1 = retriever._encode_batch([term1])
        emb2 = retriever._encode_batch([term2])
        similarity = float(np.dot(emb1[0], emb2[0]))
        print(f"  {term1:25s} ↔ {term2:25s}  sim={similarity:.3f}")
