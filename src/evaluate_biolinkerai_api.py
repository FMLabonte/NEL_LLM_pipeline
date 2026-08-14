"""
Evaluate the official BioLinkerAI API on OUR data with OUR metric.
================================================================

Gives a direct, same-denominator comparison against BioLinkerAI. Their public
API (https://labs.tib.eu/biolinkerai/api-use) is end-to-end: it extracts its
OWN entities from the text and links each to a UMLS CUI. Our benchmark is
gold-mention linking (entities pre-annotated). We bridge the two by:

  1. Sending each document (title + abstract) to /process-text once.
  2. Matching every GOLD mention to the API entity at the same offset
     (fallback: surface form), taking its linked CUI.
  3. Mapping CUI -> MeSH (MRCONSO) and scoring against the gold MeSH id,
     exactly as our own evaluation does.

Because the API also does recognition, we report two honest numbers:
  * End-to-end accuracy  = correct / all gold mentions
        (a gold mention their extractor missed counts as wrong)
  * Linking accuracy      = correct / gold mentions their extractor found
        (isolates their disambiguation quality — comparable to our Phase 4)
plus the extraction coverage (how many gold mentions they found at all).

Responses are cached per PMID, so re-runs never re-hit the API.

Usage:
    python3 src/evaluate_biolinkerai_api.py --dataset bc5cdr --split test \\
        --umls Data/UMLS/MRCONSO.RRF --keep-duplicates \\
        --dump-predictions preds_biolinkerai.jsonl
"""

import argparse
import json
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "improvements"))

from pubtator_parser import parse_pubtator
from cui_mesh_mapper import CUIToMeSHMapper

API_URL = "https://labs.tib.eu/biolinkerai/process-text"


# ── API client (stdlib only, cached) ───────────────────────────────────────

def call_api(text: str, k: int, url: str, timeout: float, max_retries: int):
    """POST one document to the BioLinkerAI API. Returns the parsed JSON."""
    payload = json.dumps({"input_text": text, "k": k}).encode("utf-8")
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            req = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"}, method="POST",
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(2 * attempt)
    raise RuntimeError(f"API failed after {max_retries} attempts: {last_err}")


def load_cache(path: Path) -> dict:
    cache = {}
    if path.exists():
        for line in open(path):
            try:
                r = json.loads(line)
                cache[str(r["pmid"])] = r["response"]
            except (json.JSONDecodeError, KeyError):
                continue
    return cache


# ── Gold <-> API matching ───────────────────────────────────────────────────

def api_entities(response: dict) -> list[dict]:
    """Flatten the API response to entity dicts with a CUI."""
    out = []
    for r in (response or {}).get("results", []):
        if r.get("category") != "entities":
            continue
        cand = r.get("best_candidate") or {}
        out.append({
            "surface_form": r.get("surface_form", ""),
            "start": r.get("start", -1),
            "end": r.get("end", -1),
            "cui": cand.get("id"),
        })
    return out


def match_gold(g_start: int, g_end: int, surface: str, ents: list[dict]):
    """Find the API entity for a gold mention: exact offset > overlap > surface."""
    for e in ents:                                    # exact offset
        if e["start"] == g_start and e["end"] == g_end:
            return e
    best, best_ov = None, 0                            # max overlap
    for e in ents:
        ov = min(g_end, e["end"]) - max(g_start, e["start"])
        if ov > best_ov:
            best, best_ov = e, ov
    if best is not None and best_ov > 0:
        return best
    sl = surface.lower()                               # surface-form fallback
    cands = [e for e in ents if e["surface_form"].lower() == sl]
    if cands:
        return min(cands, key=lambda e: abs(e["start"] - g_start))
    return None


def norm_mesh(x: str) -> str:
    s = str(x).strip().upper()
    for p in ("MESH:", "MSH:", "UMLS:"):
        if s.startswith(p):
            s = s[len(p):]
    return s


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["bc5cdr"], default="bc5cdr",
                    help="Only BC5CDR for now (MeSH gold; MedMentions needs the "
                         "split fix first).")
    ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
    ap.add_argument("--umls", default=str(PROJECT_ROOT / "Data" / "UMLS" / "MRCONSO.RRF"),
                    help="MRCONSO.RRF path (uses cached CUI->MeSH if present).")
    ap.add_argument("--k", type=int, default=50, help="API candidate pool (their default: 50).")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="Evaluate all mentions (comparable denominator), not "
                         "unique (mention, gold) pairs.")
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "results" / "biolinkerai_api_cache.jsonl"))
    ap.add_argument("--api-url", default=API_URL)
    ap.add_argument("--sleep", type=float, default=0.3, help="Pause between API calls (be gentle).")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--offline", action="store_true",
                    help="Use only cached responses; never call the API.")
    ap.add_argument("--dump-predictions", default=None,
                    help="Write per-mention correctness JSONL (for mcnemar_compare.py).")
    args = ap.parse_args()

    split_map = {"train": "TrainingSet", "dev": "DevelopmentSet", "test": "TestSet"}
    data_path = (PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516"
                 / f"CDR_{split_map[args.split]}.PubTator.txt")
    meta, anns, _ = parse_pubtator(str(data_path))

    doc_text = {r["pmid"]: f"{r.get('title','')} {r.get('abstract','')}".strip()
                for _, r in meta.iterrows()}

    df = anns[anns["mesh_id"] != "-1"].copy()
    df = df[df["mesh_id"].str.match(r"^[DC]\d+", na=False)]   # MeSH-linkable
    if not args.keep_duplicates:
        df = df.drop_duplicates(subset=["mention", "mesh_id"])
    if args.limit:
        df = df.head(args.limit)
    print(f"  Evaluating {len(df)} gold mentions "
          f"({'all' if args.keep_duplicates else 'deduplicated'})")

    mapper = CUIToMeSHMapper(args.umls)
    mapper.load()

    cache_path = Path(args.cache)
    cache = load_cache(cache_path)
    print(f"  Cache: {len(cache)} documents already fetched")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_fh = open(cache_path, "a")

    total = matched = correct = 0
    no_cui = unmappable = 0
    preds = []
    pmids = list(dict.fromkeys(df["pmid"].tolist()))
    ent_cache = {}   # pmid -> parsed entities

    for i, pmid in enumerate(pmids, 1):
        if pmid not in cache:
            if args.offline:
                continue
            text = doc_text.get(pmid, "")
            if not text:
                continue
            try:
                resp = call_api(text, args.k, args.api_url, args.timeout, args.max_retries)
            except RuntimeError as e:
                print(f"  [{pmid}] {e}")
                continue
            cache[pmid] = resp
            cache_fh.write(json.dumps({"pmid": pmid, "response": resp}) + "\n")
            cache_fh.flush()
            time.sleep(args.sleep)
        ent_cache[pmid] = api_entities(cache[pmid])
        if i % 50 == 0:
            print(f"  ... {i}/{len(pmids)} docs", flush=True)

    for _, row in df.iterrows():
        pmid = row["pmid"]
        if pmid not in ent_cache:
            continue
        total += 1
        gold_ids = {norm_mesh(g) for g in str(row["mesh_id"]).split("|")}
        e = match_gold(int(row["start"]), int(row["end"]), row["mention"], ent_cache[pmid])
        hit = False
        if e is not None:
            matched += 1
            cui = e["cui"]
            if not cui:
                no_cui += 1
            else:
                mesh = {norm_mesh(m) for m in mapper.cui_to_mesh(cui)}
                if not mesh:
                    unmappable += 1
                hit = bool(mesh & gold_ids)
        if hit:
            correct += 1
        preds.append({
            "pmid": pmid, "mention": row["mention"], "gold_id": row["mesh_id"],
            "entity_type": row.get("entity_type"),
            "p4_correct": hit, "p3_correct": hit,
        })

    cache_fh.close()

    print("\n" + "=" * 60)
    print("BioLinkerAI API — evaluation on our data")
    print("=" * 60)
    print(f"  Gold mentions:                 {total}")
    print(f"  Found by their extractor:      {matched} "
          f"({matched/total*100:.1f}% coverage)" if total else "")
    print(f"  ►► End-to-end Accuracy@1:      {correct/total*100:.1f}% "
          f"({correct}/{total})" if total else "")
    if matched:
        print(f"  Linking Accuracy (on found):   {correct/matched*100:.1f}% "
              f"({correct}/{matched})")
    print(f"    (of found: {no_cui} had no CUI, {unmappable} CUI not mappable to MeSH)")
    print(f"\n  Note: end-to-end includes their recognition; 'linking on found' is")
    print(f"  the fair comparison to our Phase 4. For the paper, apply the same")
    print(f"  MeSH id-remapping to both sides before the final number.")

    if args.dump_predictions:
        with open(args.dump_predictions, "w") as f:
            for p in preds:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print(f"\n  Per-mention predictions -> {args.dump_predictions}")


if __name__ == "__main__":
    main()
