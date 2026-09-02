"""
Build the UMLS SapBERT/FAISS index ONCE (for MedMentions embedding retrieval).

Loads the cached UMLSIndex (483M pickle), encodes all ~3.65M UMLS labels with
SapBERT, and writes a ~11 GB FAISS index to
  src/improvements/cache/faiss_umls/<model_slug>/
so that later `evaluate_pipeline.py --umls-index --embedding` loads it in seconds.

Run this once on a GPU node (no vLLM needed). ~20-30 min.

Usage:
    python3 src/build_umls_faiss.py
    python3 src/build_umls_faiss.py --mrconso Data/UMLS/MRCONSO.RRF --batch-size 512
"""

import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src" / "candidate-generation"))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "improvements"))

from umls_index import UMLSIndex
from embedding_retriever import UMLSEmbeddingRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mrconso", default=str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF"),
                    help="Path to MRCONSO.RRF (only its existence/mtime is used; "
                         "the index loads from the cached pickle).")
    ap.add_argument("--model", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    print("=" * 60)
    print("Building UMLS SapBERT/FAISS index")
    print("=" * 60)

    # 1) Load the UMLS string index (from the 483M pickle cache)
    print("\n[1/2] Loading UMLS index (from cache)...")
    umls_idx = UMLSIndex()  # default 8 vocabs -> f553afc2 cache
    umls_idx.build_from_mrconso(args.mrconso)
    print(f"  {len(umls_idx.entities):,} concepts, {len(umls_idx._label_index):,} labels")

    # 2) Build + save the embedding index
    print("\n[2/2] Building embedding index...")
    cache_dir = str(PROJECT_ROOT / "src" / "improvements" / "cache" /
                    "faiss_umls" / args.model.replace("/", "_"))
    retriever = UMLSEmbeddingRetriever(
        umls_index=umls_idx,
        model_name=args.model,
        batch_size=args.batch_size,
    )
    retriever.build_or_load(cache_dir)
    print(f"\nDone. Index cached at:\n  {cache_dir}/")


if __name__ == "__main__":
    main()
