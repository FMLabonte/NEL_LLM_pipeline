"""
PubTator format parser
======================
Parses a PubTator-formatted text file into three tidy DataFrames:

  metadata_df     – one row per paper     (pmid, title, abstract)
  annotations_df  – one row per entity    (pmid, start, end, mention, entity_type, mesh_id)
  relations_df    – one row per relation  (pmid, relation_type, id_1, id_2)

All three share the `pmid` column and can be joined with pd.merge(..., on="pmid").

Relation lines are detected generically — any 4-column tab-separated line where
the second column is not an integer offset is treated as a relation, regardless
of the relation type label (CID, caused_by, treats, inhibits, etc.).

Saving / loading
----------------
    save_dataframes(metadata_df, annotations_df, relations_df, prefix="pubtator")
    metadata_df, annotations_df, relations_df = load_dataframes(prefix="pubtator")
"""

import re
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------------
# Core parser
# ---------------------------------------------------------------------------

def parse_pubtator(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Parse a PubTator file and return (metadata_df, annotations_df, relations_df).

    Parameters
    ----------
    path : str or Path
        Path to the PubTator-format text file.

    Returns
    -------
    metadata_df : pd.DataFrame
        Columns: pmid (str), title (str), abstract (str)

    annotations_df : pd.DataFrame
        Columns: pmid (str), start (Int64), end (Int64),
                 mention (str), entity_type (str), mesh_id (str)

    relations_df : pd.DataFrame
        Columns: pmid (str), relation_type (str), id_1 (str), id_2 (str)
        Works for any relation type label (CID, causes, treats, etc.)
    """
    text = Path(path).read_text(encoding="utf-8")

    meta_rows = []
    ann_rows  = []
    rel_rows  = []

    # Split on blank lines — one block per paper
    blocks = re.split(r"\n{2,}", text.strip())

    for block in blocks:
        lines = block.strip().splitlines()
        pmid     = None
        title    = ""
        abstract = ""

        for line in lines:
            line = line.rstrip()
            if not line:
                continue

            # ----- title:  PMID|t|text -----
            m = re.match(r"^(\d+)\|t\|(.*)$", line)
            if m:
                pmid  = m.group(1)
                title = m.group(2).strip()
                continue

            # ----- abstract:  PMID|a|text -----
            m = re.match(r"^(\d+)\|a\|(.*)$", line)
            if m:
                pmid     = m.group(1)
                abstract = m.group(2).strip()
                continue

            parts = line.split("\t")

            # ----- relation: 4 or 5 columns where col[1] is NOT an integer -----
            # 4-col: PMID \t rel_type \t id_1 \t id_2
            # 5-col: PMID \t rel_type \t id_1 \t id_2 \t novelty   (e.g. BioRED)
            if len(parts) in (4, 5):
                try:
                    int(parts[1])           # if this succeeds it's NOT a relation
                except ValueError:
                    rel_rows.append({
                        "pmid":          parts[0],
                        "relation_type": parts[1],
                        "id_1":          parts[2],
                        "id_2":          parts[3],
                        "novelty":       parts[4] if len(parts) == 5 else pd.NA,
                    })
                    continue

            # ----- entity: 6 columns with integer offsets -----
            # PMID \t start \t end \t mention \t entity_type \t mesh_id
            if len(parts) == 6:
                try:
                    ann_rows.append({
                        "pmid":        parts[0],
                        "start":       int(parts[1]),
                        "end":         int(parts[2]),
                        "mention":     parts[3],
                        "entity_type": parts[4],
                        "mesh_id":     parts[5],
                    })
                except ValueError:
                    pass    # skip malformed lines
                continue

        if pmid:
            meta_rows.append({"pmid": pmid, "title": title, "abstract": abstract})

    # --- build DataFrames ---------------------------------------------------

    metadata_df = pd.DataFrame(meta_rows, columns=["pmid", "title", "abstract"])

    annotations_df = pd.DataFrame(ann_rows, columns=[
        "pmid", "start", "end", "mention", "entity_type", "mesh_id"
    ])
    annotations_df["start"] = annotations_df["start"].astype("Int64")
    annotations_df["end"]   = annotations_df["end"].astype("Int64")

    relations_df = pd.DataFrame(rel_rows, columns=[
        "pmid", "relation_type", "id_1", "id_2", "novelty"
    ])

    return metadata_df, annotations_df, relations_df


# ---------------------------------------------------------------------------
# Convenience: enrich relations with entity mention names
# ---------------------------------------------------------------------------

def enrich_relations(
    relations_df:   pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add human-readable mention names to a relations DataFrame by looking up
    id_1 and id_2 against the entity annotations.

    Returns a new DataFrame with extra columns: mention_1, mention_2.
    If a mesh_id has multiple distinct mentions (e.g. 'IDM' and 'indomethacin'),
    they are joined with ' / '.
    """
    lookup = (
        annotations_df[["pmid", "mesh_id", "mention"]]
        .drop_duplicates()
        .groupby(["pmid", "mesh_id"])["mention"]
        .apply(lambda x: " / ".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"mention": "mention_name"})
    )

    enriched = (
        relations_df
        .merge(lookup.rename(columns={"mesh_id": "id_1", "mention_name": "mention_1"}),
               on=["pmid", "id_1"], how="left")
        .merge(lookup.rename(columns={"mesh_id": "id_2", "mention_name": "mention_2"}),
               on=["pmid", "id_2"], how="left")
    )
    return enriched


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def save_dataframes(
    metadata_df:    pd.DataFrame,
    annotations_df: pd.DataFrame,
    relations_df:   pd.DataFrame,
    prefix:         str = "pubtator",
    output_dir:     str | Path = ".",
) -> tuple[Path, Path, Path]:
    """
    Save all three DataFrames to CSV.

    Files written:
        <output_dir>/<prefix>_metadata.csv
        <output_dir>/<prefix>_annotations.csv
        <output_dir>/<prefix>_relations.csv
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta_path = out / f"{prefix}_metadata.csv"
    ann_path  = out / f"{prefix}_annotations.csv"
    rel_path  = out / f"{prefix}_relations.csv"

    metadata_df.to_csv(meta_path,    index=False)
    annotations_df.to_csv(ann_path,  index=False)
    relations_df.to_csv(rel_path,    index=False)

    print(f"Saved metadata    → {meta_path}  ({len(metadata_df)} rows)")
    print(f"Saved annotations → {ann_path}  ({len(annotations_df)} rows)")
    print(f"Saved relations   → {rel_path}  ({len(relations_df)} rows)")

    return meta_path, ann_path, rel_path


def load_dataframes(
    prefix:    str = "pubtator",
    input_dir: str | Path = ".",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load the CSVs saved by save_dataframes() and restore correct dtypes.

    Returns (metadata_df, annotations_df, relations_df).
    """
    inp = Path(input_dir)

    metadata_df = pd.read_csv(
        inp / f"{prefix}_metadata.csv",
        dtype={"pmid": str},
    )

    annotations_df = pd.read_csv(
        inp / f"{prefix}_annotations.csv",
        dtype={"pmid": str, "mention": str, "entity_type": str, "mesh_id": str},
    )
    annotations_df["start"] = annotations_df["start"].astype("Int64")
    annotations_df["end"]   = annotations_df["end"].astype("Int64")

    relations_df = pd.read_csv(
        inp / f"{prefix}_relations.csv",
        dtype={"pmid": str, "relation_type": str, "id_1": str, "id_2": str, "novelty": str},
    )

    return metadata_df, annotations_df, relations_df


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python pubtator_parser.py <path_to_pubtator_file>")
        sys.exit(1)

    meta, anns, rels = parse_pubtator(sys.argv[1])

    print("\n=== metadata_df ===")
    print(meta.to_string(index=False))

    print("\n=== annotations_df (first 10 rows) ===")
    print(anns.head(10).to_string(index=False))

    print("\n=== relations_df ===")
    print(rels.to_string(index=False))

    print("\n=== relations enriched with mention names ===")
    print(enrich_relations(rels, anns).to_string(index=False))

    print(f"\nPapers      : {len(meta)}")
    print(f"Entities    : {len(anns)}")
    print(f"Relations   : {len(rels)}")
    print(f"Relation types found: {rels['relation_type'].unique().tolist()}")

    # Round-trip check
    save_dataframes(meta, anns, rels, prefix="demo_output")
    meta2, anns2, rels2 = load_dataframes(prefix="demo_output")
    print("\nRound-trip check passed:",
          meta.equals(meta2) and anns.equals(anns2) and rels.equals(rels2))