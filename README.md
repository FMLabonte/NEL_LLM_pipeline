## Goal of this repo

The goal of this repo is to reproduce and make easily available inspired by the following paper: https://link.springer.com/chapter/10.1007/978-981-96-0573-6_19 

### Project description:
A well-known problem in biomedical research is entity disambiguation, also known as entity linking. Multiple entities share the same name in different contexts, or have so many synonyms that no single database covers all of them. Therefore, most annotated datasets link their entities to an existing taxonomy to unambiguously identify what is meant. This is a challenging task that requires significant time from human annotators.
For quite a while, the state of the art was around 65% accuracy, but this was recently surpassed through smart use of LLMs in the disambiguation pipeline [https://link.springer.com/chapter/10.1007/978-981-96-0573-6_19]. Unfortunately, the code was not open-sourced and relied on closed-source LLMs.
Your goal is to build a basic open-source implementation using an existing dataset (BioRED) for testing and validation that uses MeSH identifiers, inspired by the paper above. The baseline implementation uses a simple search followed by candidate selection through an LLM. The goal is then to modularly add new methods and rules for database searching or model decision-making. This can include anything from embedding-based searches and semantic rules to model fine-tuning or testing different models—all to answer the question of how such pipelines should be built and what aspects deserve focus.
In short: You'll build an open-source baseline implementation of the AI-linker, then test different modules and modifications to see how they affect performance.

### Interesting sourcces
Review of entity linking tools: https://dl.acm.org/doi/pdf/10.1145/3796222
Inter-annotator agreement on these tasks: https://link.springer.com/article/10.1186/s12911-021-01395-z
MeSH database: https://www.ncbi.nlm.nih.gov/mesh
UMLS: https://www.nlm.nih.gov/research/umls/index.html

### General Research questions: 
- Which interventions help the models the most ?
   - Different searches
   - Different prompts
   - Pre filtering rules 
- Does this task scale with model size ? 
- How does the tool set translate between different Taxonomies ( test if it also works on the UMLS taxonomy and other datasets then BioRED) ?

## Example of Data Loading
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

