"""
Diagnose the BioLinkerAI API — isolate WHY /process-text returns HTTP 500.

Runs a series of probes from harmless to realistic and prints, for each, the
HTTP status AND the response body (500 pages usually carry the real error /
stack trace, which urllib normally hides). Read top-to-bottom:

  * If probe 1 (their own documented example) already 500s
        -> their API is down / broken. Not our fault. Wait or email them.
  * If probe 1 works but a real BC5CDR abstract 500s
        -> our input triggers it (length? a character? the k value?).
          The later probes tell you which.

Run:  python3 src/test_biolinkerai_api.py
"""

import json
import sys
import urllib.request
import urllib.error
from pathlib import Path

URL = "https://labs.tib.eu/biolinkerai/process-text"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def probe(label, payload, url=URL, ua=None, timeout=60):
    """POST payload, print status + body (including error bodies)."""
    print("\n" + "=" * 70)
    print(f"PROBE: {label}")
    print(f"  payload keys: {list(payload)} | "
          f"input_text length: {len(payload.get('input_text',''))} chars | "
          f"k={payload.get('k', '(omitted)')}")
    headers = {"Content-Type": "application/json"}
    if ua:
        headers["User-Agent"] = ua
    data = json.dumps(payload).encode("utf-8")
    try:
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            print(f"  STATUS: {resp.status} OK")
            try:
                j = json.loads(body)
                n = len(j.get("results", []))
                print(f"  -> valid JSON, {n} results")
                if n:
                    e = j["results"][0]
                    print(f"     first: {e.get('surface_form')!r} -> "
                          f"{(e.get('best_candidate') or {}).get('id')}")
            except json.JSONDecodeError:
                print(f"  -> body (first 400 chars): {body[:400]}")
    except urllib.error.HTTPError as e:
        # THIS is the important part: the body of a 500 usually explains it.
        body = e.read().decode("utf-8", "replace") if e.fp else ""
        print(f"  STATUS: HTTP {e.code} {e.reason}")
        print(f"  ERROR BODY (first 800 chars):\n{body[:800]}")
    except Exception as e:
        print(f"  FAILED: {type(e).__name__}: {e}")


def main():
    # 1) Their own documented example — the ground truth for "is the API up?"
    probe("1. Documented example (Lepirudin/Garlic), k=50", {
        "input_text": "The serum concentration of Lepirudin can be decreased "
                      "when it is combined with Garlic",
        "k": 50,
    })

    # 2) Minimal input
    probe("2. Minimal text 'aspirin', k=50", {"input_text": "aspirin", "k": 50})

    # 3) Is the k field the problem?
    probe("3. Minimal text, k OMITTED", {"input_text": "aspirin"})
    probe("4. Minimal text, k=5", {"input_text": "aspirin", "k": 5})

    # 5) A real BC5CDR abstract (the kind that 500'd for us)
    try:
        sys.path.insert(0, str(PROJECT_ROOT))
        from pubtator_parser import parse_pubtator
        meta, _, _ = parse_pubtator(str(
            PROJECT_ROOT / "Data" / "CDR_Data" / "CDR.Corpus.v010516"
            / "CDR_TestSet.PubTator.txt"))
        row = meta.iloc[0]
        text = f"{row.get('title','')} {row.get('abstract','')}".strip()
        probe(f"5. Real BC5CDR abstract (pmid {row['pmid']}), k=50",
              {"input_text": text, "k": 50})
        probe(f"6. Same real abstract, k=5", {"input_text": text, "k": 5})
        # 7) Just the title (short real text)
        probe(f"7. Real title only, k=50",
              {"input_text": str(row.get("title", "")), "k": 50})
    except Exception as e:
        print(f"\n(could not load BC5CDR for probes 5-7: {e})")

    # 8) With a browser User-Agent (some servers reject urllib's default UA)
    probe("8. Documented example + browser User-Agent", {
        "input_text": "The serum concentration of Lepirudin can be decreased "
                      "when it is combined with Garlic",
        "k": 50,
    }, ua="Mozilla/5.0")

    print("\n" + "=" * 70)
    print("READING THE RESULTS:")
    print("  probe 1 fails too      -> their API is down/broken (server-side).")
    print("  probe 1 ok, 5/6 fail   -> long/real input crashes it; try 7 (title).")
    print("  only k=50 fails        -> lower k in the eval (--k 5).")
    print("  only UA version works  -> add a User-Agent header to the client.")


if __name__ == "__main__":
    main()
