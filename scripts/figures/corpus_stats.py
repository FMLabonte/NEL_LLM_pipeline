"""Corpus statistics for the lab report (Data Set section).
Applies the same MeSH-linkable filters as src/evaluate_pipeline.py."""
import sys, json, re
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pubtator_parser import parse_pubtator

ROOT = Path(__file__).resolve().parents[2] / "Data"
CORPORA = {
    "BC5CDR":   ROOT / "CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt",
    "BioRED":   ROOT / "BioRED/Test.PubTator",
    "NCBI":     ROOT / "NCBI/NCBItestset_corpus.txt",
    "NLM-Chem": ROOT / "NLM-Chem/nlm_chem_test.pubtator",
    "MedMentions": ROOT / "MedMention/MedMentions_st21pv_pubtator.txt",
}

rows, typedist = [], {}
for name, path in CORPORA.items():
    meta, anns, _ = parse_pubtator(path)
    if name == "MedMentions":
        pm = {l.strip() for l in open(ROOT / "MedMention/corpus_pubtator_pmids_test.txt") if l.strip()}
        meta = meta[meta.pmid.astype(str).isin(pm)]; anns = anns[anns.pmid.astype(str).isin(pm)]
    n_all = len(anns)
    ids = anns["mesh_id"].astype(str)
    if name in ("NCBI", "NLM-Chem"):
        ids = ids.str.replace("MESH:", "", regex=False).str.replace("MeSH:", "", regex=False).str.replace("+", "|", regex=False)
        anns = anns.assign(mesh_id=ids)
    if name in ("BioRED", "NCBI", "NLM-Chem"):
        anns = anns[anns.mesh_id.astype(str).str.match(r"^[DC]\d+", na=False)]
    anns = anns[anns.mesh_id.astype(str) != "-1"]   # unlinked mentions, as in evaluate_pipeline.py
    text = (meta.title.fillna("") + " " + meta.abstract.fillna(""))
    ntok = text.str.split().str.len()
    mlen = anns.mention.astype(str).str.split().str.len()
    pairs = anns[["mention", "mesh_id"]].astype(str).apply(lambda r: (r.mention.lower(), r.mesh_id), axis=1)
    rows.append(dict(corpus=name, docs=meta.pmid.nunique(), tok_per_doc=round(ntok.mean()),
        mentions_all=n_all, mentions=len(anns), per_doc=round(len(anns)/meta.pmid.nunique(),1),
        concepts=anns.mesh_id.nunique(), uniq_pairs=pairs.nunique(),
        dup_share=round(100*(1-pairs.nunique()/len(anns)),1),
        mention_words=round(mlen.mean(),2), short_share=round(100*(anns.mention.astype(str).str.len()<=5).mean(),1)))
    typedist[name] = anns.entity_type.value_counts().to_dict()

df = pd.DataFrame(rows)
print(df.to_string(index=False))
print(json.dumps(typedist, indent=1))
FIGDIR = Path(__file__).resolve().parents[2] / "figures"; FIGDIR.mkdir(exist_ok=True)
df.to_csv(FIGDIR / "corpus_stats.csv", index=False)
json.dump(typedist, open(FIGDIR / "corpus_types.json", "w"), indent=1)
