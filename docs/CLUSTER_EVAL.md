# Eval auf dem Cluster (vLLM)

Warum: vLLM serviert das HF-Modell direkt (OpenAI-kompatibel) — **kein GGUF,
keine Quantisierung, kein LMStudio, kein Tokenizer-Fix**. Entlastet den Mac und
erlaubt parallele + wiederholte Läufe (Varianz). Einmaliges Setup, dann pro Lauf
nur noch `sbatch cluster/bender_eval.sbatch`.

## 1. Was schon auf Bender ist / was fehlt

Im Git-Klon liegen bereits: der Code und die **Datensätze** (CDR_Data, BioRED,
MedMention — sind getrackt). **Nicht** im Git (gitignored) und daher zu
transferieren:

| Was | Größe | Zweck |
|---|---|---|
| `Data/MeSH/` | 1.1 G | MeSH-XML, um den Index zu bauen |
| `src/candidate-generation/cache/` | 211 M | Synonym-Caches (spart die 16 G UMLS) |
| `Data/UMLS/cui_to_mesh_cache.json` | 11 M | CUI→MeSH-Mapping |

**Nicht** transferieren: der FAISS-Embedding-Cache (9.8 G) — der **baut sich beim
ersten Lauf auf der GPU neu** (~15 Min) und wird gecached. Und die rohe 16-G-UMLS
`MRCONSO.RRF` brauchst du (hoffentlich) nicht, weil die Caches reichen — im
Smoke-Test (unten) verifizieren.

Transfer (lokales Terminal, ~1.3 G; am Campus schnell, per VPN langsam → ggf. `rsync -P`):

```bash
cd "/Users/moritz/Desktop/NLP Lab/NEL_LLM_pipeline"
rsync -avP Data/MeSH s24mgilg@bender.hpc.uni-bonn.de:~/NEL_LLM_pipeline/Data/
rsync -avP src/candidate-generation/cache \
    s24mgilg@bender.hpc.uni-bonn.de:~/NEL_LLM_pipeline/src/candidate-generation/
rsync -avP Data/UMLS/cui_to_mesh_cache.json \
    s24mgilg@bender.hpc.uni-bonn.de:~/NEL_LLM_pipeline/Data/UMLS/
```

(Code vorher per `git pull` auf Bender aktualisieren, siehe unten.)

## 2. eval_env einmalig aufsetzen (Login-Node)

Getrennt von `ft_env`, weil vLLM eigene Torch-Version mitbringt:

```bash
module load Python
cd ~/NEL_LLM_pipeline
python3 -m venv eval_env
source eval_env/bin/activate
pip install vllm rapidfuzz faiss-cpu openai pandas numpy
```

## 3. Smoke-Test (verifiziert Daten + Serving), interaktiv auf GPU

WICHTIG — nur A40 nutzen: eval_env ist auf dem Intel-Login-Node gebaut; die
A100-Nodes sind AMD und werfen "Illegal instruction". Alles auf A40 halten.

```bash
srun --pty --partition=A40devel --gpus=1 /bin/bash
module load Python && source ~/NEL_LLM_pipeline/eval_env/bin/activate
cd ~/NEL_LLM_pipeline
# nur die Kandidatengenerierung, ohne LLM, 5 Mentions -> lädt alle Daten/Caches:
python3 src/evaluate_pipeline.py --no-phase4 --limit 5 --dataset bc5cdr \
    --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding
```

Läuft das ohne „file not found"/UMLS-Fehler durch, sind alle Daten da. Wirft es
einen Fehler wegen `MRCONSO.RRF`, dann brauchst du doch die 16-G-Datei (oder wir
patchen den `--umls`-Pfad). `exit` beendet die Session.

## 4. Voller Lauf

Im `cluster/bender_eval.sbatch` oben `MODEL`, `SERVED_NAME`, `DATASET` anpassen, dann:

```bash
sbatch cluster/bender_eval.sbatch
squeue --me
tail -f logs/eval-*.out       # Eval-Fortschritt
tail -f logs/vllm-*.out       # vLLM-Server-Log (falls es hakt)
```

Der erste Lauf baut den FAISS-Index (~15 Min extra), danach schnell. Ergebnis:
`preds_<name>_<dataset>.jsonl` — enthält jetzt die **angereicherten Felder**
(Score-Gap, Surface-Sim, Gold-Rang) für die volle Zerlegung.

## 5. Auswerten (lokal oder auf Bender)

```bash
python3 src/analyze_decomposition.py preds_nel-4b_bc5cdr.jsonl \
    --train Data/finetune/bc5cdr_train_sft.jsonl
python3 src/mcnemar_compare.py preds_<zeroshot>.jsonl preds_<finetuned>.jsonl
```

## Hinweise

- **Zero-Shot vs. Finetuned:** zwei Läufe, nur `MODEL`/`SERVED_NAME` ändern.
  Für Zero-Shot `MODEL="Qwen/Qwen3-4B-2507"` (HF-ID, vLLM lädt es), für
  finetuned den Merge-Pfad. **Gleicher Base-Checkpoint** für beide!
- **Structured Output:** vLLM unterstützt `response_format` (guided decoding);
  falls eine vLLM-Version zickt, fällt die Pipeline automatisch auf den
  Text-Parser zurück.
- **Varianz (Frederik):** denselben `sbatch` 3× einreichen, Ausgabedateien
  durchnummerieren, Mittelwert ± Std über die FINAL-Acc@1.
- **GPU teilen:** `--gpu-memory-utilization 0.5` lässt Platz für SapBERT. Bei
  OOM runter auf 0.4, oder A100 statt A40.
