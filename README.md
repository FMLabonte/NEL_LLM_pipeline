Example usage of the custom data loader this allows you to load all 3 datasets that we are interested in uniformely. Making it easier to work with down the line
```python
from pubtator_parser import parse_pubtator, save_dataframes, load_dataframes, enrich_relations

# Parse — now returns 3 DataFrames
meta, anns, rels = parse_pubtator("Data/CDR_Data/CDR.Corpus.v010516/CDR_TestSet.PubTator.txt")

# Join annotations with metadata
combined = anns.merge(meta, on="pmid")

# Filter to just Chemical entities (example)
chemicals = anns[anns["entity_type"] == "Chemical"]

# Enrich relations with human-readable mention names
rels_named = enrich_relations(rels, anns)

# Save — pass rels as the third argument
save_dataframes(meta, anns, rels, prefix="CDR_test", output_dir="output/")

# Load returns 3 DataFrames metadata(abstract and PID), Annotations, Relations 
meta, anns, rels = load_dataframes(prefix="CDR_test", input_dir="output/")
``` 
