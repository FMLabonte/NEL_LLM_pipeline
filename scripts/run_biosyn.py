#!/usr/bin/env python3
"""
Faithful BioSyn batch runner  (Sung et al., ACL 2020; dmis-lab/BioSyn).

Runs the FULL BioSyn scorer (hybrid = sparse_weight * sparse + dense) with the
authors' pretrained checkpoints on OUR exact eval mentions, and writes a
candidates JSONL that our pipeline consumes via `evaluate_pipeline.py
--candidates-from`. This puts our identical LLM stage on BioSyn's candidates,
so BioSyn becomes a real point on the retriever-strength curve (Figure 1).

IMPORTANT — run this from inside the cloned BioSyn repo root (so `from
src.biosyn import ...` resolves). BC5CDR has separate disease/chemical models
and dictionaries; mentions are routed by entity_type exactly as BioSyn is
trained. We apply BioSyn's own TextPreprocess but NOT Ab3P / composite
resolution, so the mention set stays byte-identical to what every other
retriever in the curve sees (a fair LLM-gain, not BioSyn's headline number).

Example (BC5CDR):
    python run_biosyn.py \
        --mentions mentions_bc5cdr_test.jsonl \
        --disease-model  dmis-lab/biosyn-sapbert-bc5cdr-disease \
        --disease-dict   datasets/bc5cdr-disease/test_dictionary.txt \
        --chemical-model dmis-lab/biosyn-sapbert-bc5cdr-chemical \
        --chemical-dict  datasets/bc5cdr-chemical/test_dictionary.txt \
        --topk-dict 40 --topk-out 10 --use-cuda \
        --out preds_candidates_biosyn_bc5cdr.jsonl
"""
import argparse
import json
import re
import numpy as np

# ── Compatibility shims: run BioSyn's 2020 code on a MODERN stack (eval_env) ──
# Avoids rebuilding the old `tokenizers` pin from source under Python 3.12
# (which needs a Rust compiler). BioSyn's INFERENCE path only touches stable
# AutoModel/AutoTokenizer APIs; the sole break is `cached_download`, removed
# from recent huggingface_hub. We shim it (never called when models are local)
# and force torch.load(weights_only=False) for the trusted sparse_weight.pt.
import huggingface_hub as _hf
if not hasattr(_hf, "cached_download"):
    def _no_cached_download(*a, **k):
        raise RuntimeError(
            "cached_download unavailable in modern huggingface_hub; pass a LOCAL "
            "model dir (with sparse_encoder.pk + sparse_weight.pt) so no download "
            "is needed.")
    _hf.cached_download = _no_cached_download
if not hasattr(_hf, "hf_hub_url"):
    _hf.hf_hub_url = lambda *a, **k: None
import torch as _torch
_orig_torch_load = _torch.load
def _torch_load_full(*a, **k):
    k.setdefault("weights_only", False)  # sparse_weight.pt is a trusted nn.Parameter
    return _orig_torch_load(*a, **k)
_torch.load = _torch_load_full

from src.biosyn import DictionaryDataset, BioSyn, TextPreprocess

MESH_RE = re.compile(r"[DC]\d{6,7}")


def mesh_id_of(raw_id: str) -> str | None:
    """BioSyn ids look like 'D001260|208900' (MeSH|OMIM). Return the MeSH token."""
    m = MESH_RE.search(raw_id or "")
    return m.group(0) if m else None


def load_scorer(model_path, dict_path, use_cuda, max_length):
    biosyn = BioSyn(max_length=max_length, use_cuda=use_cuda)
    biosyn.load_model(model_name_or_path=model_path)
    dictionary = DictionaryDataset(dictionary_path=dict_path).data  # [:,0]=name [:,1]=cui
    names = dictionary[:, 0]
    # sklearn version mismatch: the shipped sparse_encoder.pk was pickled with
    # scikit-learn 0.24.2 and does NOT unpickle as *fitted* under modern sklearn
    # (NotFittedError). BioSyn's sparse encoder is just a char 1-2gram TF-IDF, so
    # we re-fit it on the dictionary. The learned DENSE encoder (the actual BioSyn
    # model) and the sparse_weight are untouched -> still faithful hybrid BioSyn.
    print(f"    (re)fitting sparse TF-IDF on {len(names):,} dictionary names ...")
    biosyn.init_sparse_encoder(names.tolist())
    print(f"    embedding dictionary ({len(names):,} names) ...")
    dict_sparse = biosyn.embed_sparse(names=names, show_progress=True)
    dict_dense = biosyn.embed_dense(names=names, show_progress=True)
    sparse_w = biosyn.get_sparse_weight().item()
    return {"biosyn": biosyn, "dict": dictionary,
            "sparse": dict_sparse, "dense": dict_dense, "sparse_w": sparse_w}


def predict(scorer, mention_text, topk_dict, topk_out):
    b = scorer["biosyn"]
    pp = TextPreprocess().run(mention_text)
    m_sparse = b.embed_sparse(names=[pp])
    m_dense = b.embed_dense(names=[pp])
    s = b.get_score_matrix(query_embeds=m_sparse, dict_embeds=scorer["sparse"])
    d = b.get_score_matrix(query_embeds=m_dense, dict_embeds=scorer["dense"])
    hybrid = scorer["sparse_w"] * s + d
    idxs = b.retrieve_candidate(score_matrix=hybrid, topk=topk_dict)[0]
    scores_row = hybrid[0]
    # Dedupe dictionary hits by MeSH id, keep the highest-scoring name.
    best = {}
    for i in idxs:
        name, raw_id = scorer["dict"][i][0], scorer["dict"][i][1]
        mid = mesh_id_of(raw_id)
        if mid is None:
            continue
        sc = float(scores_row[i])
        if mid not in best or sc > best[mid][0]:
            best[mid] = (sc, name)
    ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[:topk_out]
    return [{"cui": mid, "name": name, "score": sc} for mid, (sc, name) in ranked]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mentions", required=True, help="our dumped mentions JSONL")
    ap.add_argument("--disease-model", required=True)
    ap.add_argument("--disease-dict", required=True)
    ap.add_argument("--chemical-model", default=None,
                    help="omit for single-type datasets (e.g. NCBI-Disease)")
    ap.add_argument("--chemical-dict", default=None)
    ap.add_argument("--topk-dict", type=int, default=40,
                    help="dictionary hits to pull before de-duping by CUI")
    ap.add_argument("--topk-out", type=int, default=10,
                    help="unique-CUI candidates to keep per mention")
    ap.add_argument("--max-length", type=int, default=25)
    ap.add_argument("--use-cuda", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("Loading DISEASE scorer ...")
    disease = load_scorer(args.disease_model, args.disease_dict, args.use_cuda, args.max_length)
    chemical = None
    if args.chemical_model and args.chemical_dict:
        print("Loading CHEMICAL scorer ...")
        chemical = load_scorer(args.chemical_model, args.chemical_dict, args.use_cuda, args.max_length)

    mentions = [json.loads(l) for l in open(args.mentions) if l.strip()]
    print(f"Scoring {len(mentions):,} mentions ...")
    n_chem = n_dis = 0
    with open(args.out, "w") as out:
        for i, rec in enumerate(mentions):
            etype = (rec.get("entity_type") or "").lower()
            if chemical is not None and ("chem" in etype or "drug" in etype):
                scorer = chemical; n_chem += 1
            else:
                scorer = disease; n_dis += 1
            cands = predict(scorer, rec["mention"], args.topk_dict, args.topk_out)
            out.write(json.dumps({
                "pmid": str(rec["pmid"]),
                "start": rec.get("start", -1),
                "mention": rec["mention"],
                "candidates": cands,
            }) + "\n")
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{len(mentions)}")
    print(f"Done. disease={n_dis} chemical={n_chem}  ->  {args.out}")


if __name__ == "__main__":
    main()
