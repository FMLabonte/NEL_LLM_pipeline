"""
BioC-XML  ->  PubTator  converter (for full-text corpora like NLM-Chem).

NLM-Chem ships as BioC XML full-text articles, not abstract-level PubTator. Our
pipeline reads PubTator and builds the context as `title + " " + abstract` (both
.strip()ed). To keep character offsets valid for full text we emit an EMPTY title
and put the whole reconstructed document text into the abstract field, so an
annotation's BioC document offset maps directly onto the pipeline's full_text.

We keep only span-level annotations (those with a <location>) whose identifier is a
MeSH id, emit type "Chemical", and normalise composite ids "MESH:D1,MESH:D2" -> "D1|D2".
Document ids that are PMCIDs ("PMC12345") are reduced to their digits so the numeric
PMID regex in the parser matches.

Usage:
  python3 src/bioc_to_pubtator.py NLM-Chem-test.BioC.xml Data/NLM-Chem/nlm_chem_test.pubtator
"""
import sys
import re
import xml.etree.ElementTree as ET


def convert(bioc_paths, out_path):
    if isinstance(bioc_paths, str):
        bioc_paths = [bioc_paths]
    lines, ndoc, nann = [], 0, 0
    for bioc_path in bioc_paths:
      root = ET.parse(bioc_path).getroot()
      for doc in root.findall(".//document"):
        did = doc.findtext("id") or ""
        pmid = re.sub(r"\D", "", did) or "0"          # PMC12345 -> 12345

        # ---- reconstruct full document text at BioC offsets ----
        passages = []
        maxend = 0
        for p in doc.findall("passage"):
            off = int(p.findtext("offset") or 0)
            txt = p.findtext("text") or ""
            passages.append((off, txt))
            maxend = max(maxend, off + len(txt))
        buf = [" "] * maxend
        for off, txt in passages:
            for i, ch in enumerate(txt):
                if 0 <= off + i < maxend:
                    buf[off + i] = ch
        fulltext = "".join(buf)
        # PubTator is one line per field; newlines/tabs would truncate the |a| line
        # and shift every later offset. Replace with spaces (length-preserving).
        for _ch in "\n\r\t":
            fulltext = fulltext.replace(_ch, " ")
        fulltext = fulltext.rstrip()

        # ---- span-level chemical annotations with a MeSH id ----
        anns = []
        for p in doc.findall("passage"):
            for a in p.findall("annotation"):
                loc = a.find("location")
                if loc is None:
                    continue
                ident, atype = None, "Chemical"
                for inf in a.findall("infon"):
                    if inf.get("key") == "identifier":
                        ident = inf.text
                    elif inf.get("key") in ("type", "Type"):
                        atype = inf.text or "Chemical"
                if not ident or ident.strip() in ("-", "MESH:-", "NONE", ""):
                    continue
                ident = (ident.replace("MESH:", "").replace("MeSH:", "")
                              .replace(";", "|").replace(",", "|").strip())
                start = int(loc.get("offset"))
                length = int(loc.get("length"))
                mention = (a.findtext("text") or fulltext[start:start + length]).replace("\n", " ")
                anns.append((start, start + length, mention, atype, ident))

        if not anns:
            continue
        lines.append(f"{pmid}|t|")
        lines.append(f"{pmid}|a|{fulltext}")
        for s, e, m, t, i in anns:
            lines.append(f"{pmid}\t{s}\t{e}\t{m}\t{t}\t{i}")
        lines.append("")
        ndoc += 1
        nann += len(anns)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"{ndoc} docs, {nann} chemical annotations -> {out_path}")


if __name__ == "__main__":
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("inputs", nargs="*", help="BioC XML file(s)")
    ap.add_argument("--dir", help="directory of per-document XML files (e.g. NLM-Chem ALL/)")
    ap.add_argument("--pmids", help="text file with one PMCID per line; expects <dir>/<pmid>_v1.xml")
    a = ap.parse_args()
    paths = list(a.inputs)
    if a.dir and a.pmids:
        for pid in open(a.pmids):
            pid = pid.strip()
            if not pid:
                continue
            cand = os.path.join(a.dir, f"{pid}_v1.xml")
            if os.path.exists(cand):
                paths.append(cand)
            else:
                # fall back to any file starting with the pmid
                hits = [os.path.join(a.dir, f) for f in os.listdir(a.dir)
                        if f.startswith(pid) and f.endswith(".xml")]
                paths.extend(hits[:1])
    if not paths:
        print("No input files. Use positional XML files, or --dir DIR --pmids LIST.")
        sys.exit(1)
    print(f"converting {len(paths)} file(s)...")
    convert(paths, a.out)
