#!/usr/bin/env python3
"""Assemble an anonymous artefact release and check it for identifying strings.

The paper claims that every number is recomputed from per-mention prediction
dumps and that those dumps and the scripts are released. This builds that
bundle and then greps it for the things that de-anonymise a submission:
usernames, the cluster hostname, local paths, e-mail addresses, institution and
author names.

    python3 scripts/make_release.py                 # build + check
    python3 scripts/make_release.py --check-only    # only re-run the scan
    python3 scripts/make_release.py --out release/  # different target

Nothing is uploaded. Review the report, fix what it finds, then upload the
directory to Anonymous GitHub or a anonymous Zenodo deposit and put the link in
the paper's footnote.
"""
import argparse
import hashlib
import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# What goes into the release. Globs are resolved relative to the repo root.
INCLUDE = [
    "preds_zs_*.jsonl",
    "preds_ft_*.jsonl",
    "preds_qwen3-4b-zeroshot_bc5cdr.jsonl",
    "preds_dose_p*.jsonl",
    "mentions_bc5cdr_test.jsonl",
    "cands_biosyn_bc5cdr.jsonl",
    "scripts/figures/*.py",
    "figures/*.pdf",
    "src/finding1_analysis.py",
    "src/finding2_crosstab.py",
    "src/analyze_decomposition.py",
    "src/analyze_dose.py",
    "src/mcnemar_compare.py",
    "scripts/degrade_candidates.py",
]

# Patterns that would identify the authors. Extend freely -- a false positive
# costs you ten seconds, a miss costs you a desk reject.
PATTERNS = [
    (r"s24mgilg", "cluster username"),
    (r"bender\.hpc\.uni-bonn\.de|bender\b", "cluster hostname"),
    (r"uni-bonn|Universit[aä]t Bonn|University of Bonn", "institution"),
    (r"/Users/[A-Za-z0-9._-]+|/home/(?!claude)[A-Za-z0-9._-]+", "local home path"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "e-mail address"),
    (r"\bMoritz\b|\bMohamed\b|\bSnehpreet\b|\bFrederik\b|\bGilges\b", "author or supervisor name"),
    (r"docs\.google\.com|drive\.google\.com", "internal document link"),
    (r"github\.com/[A-Za-z0-9_-]+", "GitHub account"),
]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".tex", ".sh", ".sbatch", ".bib", ".json", ".cfg", ".yml", ".yaml"}


def build(out: Path) -> list[Path]:
    if out.exists():
        shutil.rmtree(out)
    copied = []
    for pattern in INCLUDE:
        for src in sorted(ROOT.glob(pattern)):
            if not src.is_file():
                continue
            dst = out / src.relative_to(ROOT)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(dst)
    return copied


def scan(out: Path) -> list[tuple[Path, int, str, str]]:
    """Text files are read line by line; JSONL dumps are large, so they are
    scanned in chunks and reported without a line number."""
    hits = []
    for path in sorted(out.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix in TEXT_SUFFIXES:
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            for n, line in enumerate(lines, 1):
                for rx, what in PATTERNS:
                    if re.search(rx, line, re.I):
                        hits.append((path, n, what, line.strip()[:110]))
        elif path.suffix == ".jsonl":
            with open(path, errors="replace") as fh:
                for n, line in enumerate(fh, 1):
                    for rx, what in PATTERNS:
                        if re.search(rx, line, re.I):
                            hits.append((path, n, what, line.strip()[:110]))
                            break
    return hits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="release")
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    out = (ROOT / a.out).resolve()

    if not a.check_only:
        copied = build(out)
        total = sum(p.stat().st_size for p in copied)
        print(f"copied {len(copied)} files, {total/1e6:.1f} MB -> {out}")
        missing = [pat for pat in INCLUDE if not list(ROOT.glob(pat))]
        if missing:
            print("  nothing matched (fine if that job has not run yet): "
                  + ", ".join(missing))
        (out / "README.md").write_text(RELEASE_README)
        print(f"  wrote {out/'README.md'}")

    hits = scan(out)
    print(f"\nanonymity scan: {len(hits)} hit(s)")
    for path, n, what, line in hits[:60]:
        print(f"  {path.relative_to(out)}:{n}  [{what}]  {line}")
    if len(hits) > 60:
        print(f"  ... and {len(hits)-60} more")
    if not hits:
        print("  clean. Check the git history separately: a release directory copied\n"
              "  out of the repo carries no history, but do not upload the repo itself.")
    else:
        print("\n  Fix these before uploading. Cluster paths in .sbatch files are the\n"
              "  usual culprit; replace them with a placeholder such as $CLUSTER_HOME.")


RELEASE_README = """# Artefacts

Per-mention prediction dumps and the analysis and figure code for the paper.
No model inference is needed to reproduce any table or figure: every number is
recomputed from the dumps.

## Layout

    preds_*.jsonl              one row per mention (schema below)
    mentions_bc5cdr_test.jsonl the BC5CDR test mentions with gold concepts
    cands_biosyn_bc5cdr.jsonl  BioSyn's candidate lists for those mentions
    scripts/figures/           the figure scripts (output goes to figures/)
    src/                       the analysis scripts
    scripts/                   the retriever-degradation tool

## Prediction schema

Each line of a `preds_*.jsonl` is one gold-annotated mention:

    pmid            document id
    mention         the surface form
    gold_id         the gold concept
    entity_type     corpus-specific type label
    p3_correct      was the retriever's top-1 the gold concept
    p4_correct      was the pipeline's final answer the gold concept
    p3_top1         the retriever's top-1 concept
    p4_top1         the final concept
    llm_changed     did the LLM stage change the top-1
    p3_score_gap    retriever confidence: top1 minus top2 score
    p3_top1_score   the top-1 score
    gold_in_list    was the gold concept anywhere in the candidate list
    gold_rank       its rank when present
    surface_sim     string similarity of the mention to the top-1 label

Score scales differ between retrievers, so bin `p3_score_gap` within a run
(quantiles) rather than across runs. `preds_zs_bc5cdr_bm25.jsonl` uses `999.0`
as a "no second candidate" sentinel in that field.

## Reproducing

    python3 scripts/figures/make_fig1_value.py        # and the other make_fig*.py
    python3 src/finding1_analysis.py preds_qwen3-4b-zeroshot_bc5cdr.jsonl --labels BC5CDR
    python3 src/analyze_dose.py

Requires numpy, scipy and matplotlib.
"""


if __name__ == "__main__":
    main()
