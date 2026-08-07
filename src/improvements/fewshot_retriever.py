"""
Retrieval-based few-shot examples for Phase 4 (LLM disambiguation)
==================================================================

Instead of three hard-coded examples that have nothing to do with the mention
at hand, retrieve the most similar annotated mentions from the TRAINING split
and put those in the prompt.

Why this matters more than ordinary in-context learning:

Our Phase 4 degradations are not semantic errors, they are CONVENTION errors.
BC5CDR gold is annotated per MeSH indexing rules, which usually means the
broader descriptor:

    "encephalopathy"      → Brain Diseases       (not Kuru)
    "kidney injury"       → Kidney Diseases      (not Acute Kidney Injury)
    "depressive disorder" → Depressive Disorder  (not Major Depressive Disorder)

A model reasoning purely semantically over-specifies here every time, and no
amount of prompt wording reliably fixes that — which is exactly why our v1-v7
prompt sweep moved nothing. Retrieved examples carry the convention implicitly.

Method follows Ye & Mitchell, ACL 2025 (§2.2.3): embed mention + surrounding
context, index with FAISS, retrieve top-k at inference.

Usage
-----
    retriever = FewShotRetriever(model_name="cambridgeltl/SapBERT-...")
    retriever.build_or_load(dataset="bc5cdr", cache_dir="src/improvements/cache")
    examples = retriever.retrieve("CF", "a mouse model of CF lung disease", k=5)
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


# Where each dataset's training split lives, and how to read the gold id.
_TRAIN_SPLITS = {
    "bc5cdr": PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516"
              / "CDR_TrainingSet.PubTator.txt",
    "biored": PROJECT_ROOT / "Data" / "BioRED" / "Train.PubTator",
    "medmentions": PROJECT_ROOT / "Data" / "MedMention"
                   / "MedMentions_st21pv_pubtator.txt",
}


def extract_word_window(
    text: str, mention: str, n_words: int = 32, start: int = -1,
) -> str:
    """
    N words around the mention, split evenly left and right.

    Kept local rather than imported from src/llm-disambiguation/prompts.py:
    that directory has a hyphen in its name and is therefore not importable
    as a package without sys.path surgery.
    """
    if not mention:
        return " ".join(text.split()[:n_words])

    ok = (
        start is not None and start >= 0
        and start + len(mention) <= len(text)
        and text[start:start + len(mention)].lower() == mention.lower()
    )
    if not ok:
        m = re.search(
            r'(?<![A-Za-z0-9])' + re.escape(mention.strip()) + r'(?![A-Za-z0-9])',
            text, re.IGNORECASE,
        )
        start = m.start() if m else text.lower().find(mention.lower())
    if start < 0:
        return " ".join(text.split()[:n_words])

    half = max(1, n_words // 2)
    left = text[:start].split()[-half:]
    right = text[start + len(mention):].split()[:half]
    return " ".join(left + [text[start:start + len(mention)]] + right)


class FewShotRetriever:
    """Retrieves similar annotated mentions from a training split."""

    def __init__(
        self,
        model_name: str = "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        device: str | None = None,
        batch_size: int = 128,
        max_length: int = 64,
        context_words: int = 32,
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "FewShotRetriever needs transformers + torch. "
                "Install with: pip install torch transformers"
            )
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.context_words = context_words

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device

        print(f"  Loading few-shot encoder: {model_name} on {device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

        self.examples: list[dict] = []
        self._embeddings: np.ndarray | None = None
        self._faiss_index = None

    # ── Encoding ─────────────────────────────────────────────────────────

    def _encode(self, texts: list[str], verbose: bool = False) -> np.ndarray:
        """Encode texts with [CLS] pooling, L2-normalized (matches SapBERT)."""
        out = []
        n_batches = (len(texts) + self.batch_size - 1) // self.batch_size
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            inputs = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=self.max_length, return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                emb = self.model(**inputs).last_hidden_state[:, 0, :]
            emb = torch.nn.functional.normalize(emb, p=2, dim=1)
            out.append(emb.cpu().numpy())
            if verbose:
                b = i // self.batch_size + 1
                if b % 25 == 0 or b == n_batches:
                    print(f"    Encoded {min(i + self.batch_size, len(texts))}"
                          f"/{len(texts)} examples", flush=True)
        return np.vstack(out).astype("float32")

    @staticmethod
    def _query_text(mention: str, context: str) -> str:
        """The string that gets embedded — mention plus its local context."""
        ctx = " ".join((context or "").split())
        return f"{mention} : {ctx}" if ctx else mention

    # ── Building ─────────────────────────────────────────────────────────

    def build_from_dataset(
        self,
        dataset: str,
        label_lookup: dict | None = None,
        max_examples: int | None = 40000,
    ):
        """
        Read the training split and index every annotated mention.

        label_lookup : optional {id: preferred_label}. Without it the examples
        show the id only, which still teaches the convention but reads worse.
        """
        from pubtator_parser import parse_pubtator

        path = _TRAIN_SPLITS.get(dataset)
        if path is None or not Path(path).exists():
            raise FileNotFoundError(
                f"No training split for '{dataset}' at {path}. "
                f"Few-shot retrieval needs the train split."
            )

        print(f"  Reading training split: {Path(path).name}")
        meta, anns, _ = parse_pubtator(str(path))

        doc_text = {
            r["pmid"]: f"{r.get('title', '')} {r.get('abstract', '')}".strip()
            for _, r in meta.iterrows()
        }

        df = anns[anns["mesh_id"] != "-1"]
        # One example per (mention, gold) pair keeps the index diverse and small.
        df = df.drop_duplicates(subset=["mention", "mesh_id"])
        if max_examples and len(df) > max_examples:
            df = df.sample(n=max_examples, random_state=42)

        self.examples = []
        for _, r in df.iterrows():
            text = doc_text.get(r["pmid"], "")
            if not text:
                continue
            start = int(r["start"]) if "start" in r and r["start"] == r["start"] else -1
            window = extract_word_window(
                text, r["mention"], n_words=self.context_words, start=start,
            )
            gid = str(r["mesh_id"])
            self.examples.append({
                "mention": r["mention"],
                "context": window,
                "id": gid,
                "label": (label_lookup or {}).get(gid.split("|")[0], ""),
                "pmid": r["pmid"],
            })

        print(f"  Indexing {len(self.examples)} training examples...")
        texts = [self._query_text(e["mention"], e["context"]) for e in self.examples]
        self._embeddings = self._encode(texts, verbose=True)
        self._build_faiss()

    def _build_faiss(self):
        if HAS_FAISS and self._embeddings is not None:
            index = faiss.IndexFlatIP(self._embeddings.shape[1])
            index.add(self._embeddings)
            self._faiss_index = index

    # ── Caching ──────────────────────────────────────────────────────────

    def _cache_paths(self, cache_dir: str, dataset: str):
        safe = self.model_name.replace("/", "_")
        base = Path(cache_dir) / "fewshot" / f"{dataset}_{safe}"
        return base.with_suffix(".npy"), base.with_suffix(".json")

    def build_or_load(
        self,
        dataset: str,
        cache_dir: str,
        label_lookup: dict | None = None,
        max_examples: int | None = 40000,
    ):
        """Load a cached index if present, otherwise build and cache one."""
        emb_path, meta_path = self._cache_paths(cache_dir, dataset)

        if emb_path.exists() and meta_path.exists():
            try:
                self._embeddings = np.load(emb_path)
                with open(meta_path) as f:
                    self.examples = json.load(f)
                if len(self.examples) == len(self._embeddings):
                    print(f"  Loaded {len(self.examples)} cached few-shot examples")
                    self._build_faiss()
                    return
                print("  Few-shot cache inconsistent — rebuilding")
            except Exception as e:
                print(f"  Few-shot cache unreadable ({e}) — rebuilding")

        self.build_from_dataset(dataset, label_lookup, max_examples)

        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(emb_path, self._embeddings)
        with open(meta_path, "w") as f:
            json.dump(self.examples, f)
        print(f"  Cached few-shot index to {emb_path.parent}")

    # ── Retrieval ────────────────────────────────────────────────────────

    def retrieve(
        self,
        mention: str,
        context: str,
        k: int = 5,
        exclude_pmid: str | None = None,
        exclude_exact_mention: bool = False,
    ) -> list[dict]:
        """
        Return the k most similar training examples.

        exclude_pmid drops examples from the same document, which matters when
        one file holds several splits (MedMentions).

        exclude_exact_mention drops training examples whose mention string
        equals the query. Off by default: train and test documents are
        disjoint, so "seizures → D012640" seen in training is legitimate
        supervision and usually the single most useful example we can show.
        Ye & Mitchell (ACL 2025, A.2.2) measured this and kept such examples.
        Turn it on for a leakage-free ablation.
        """
        if not self.examples or self._embeddings is None:
            return []

        q = self._encode([self._query_text(mention, context)])

        # Fetch with a margin: the filters below can drop hits, and coming up
        # short silently weakens the prompt.
        n_fetch = min(len(self.examples), max(k * 3, k + 20))

        if self._faiss_index is not None:
            _, idx = self._faiss_index.search(q, n_fetch)
            order = idx[0]
        else:  # numpy fallback — vectors are normalized, so dot = cosine
            sims = self._embeddings @ q[0]
            order = np.argsort(-sims)[:n_fetch]

        out = []
        for i in order:
            if i < 0:
                continue
            ex = self.examples[int(i)]
            if exclude_pmid and ex.get("pmid") == exclude_pmid:
                continue
            if exclude_exact_mention and ex["mention"].lower() == mention.lower():
                continue
            out.append(ex)
            if len(out) >= k:
                break
        return out
