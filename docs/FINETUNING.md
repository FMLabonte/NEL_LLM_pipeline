# Finetuning – Ausführung

Begleitdoku zu `src/finetune_lora.py` und `cluster/bender_finetune.sbatch`.
Ziel: kleines offenes Modell per LoRA-SFT die BC5CDR-Annotationskonvention
lernen lassen (gegen die Überspezifikations-/Konventionsfehler). Gestaffelt
nach Frederik: erst Qwen3-0.6B als Machbarkeits-Probe, dann 4B, dann größer.

## 1. Trainingsdaten (falls noch nicht gebaut)

```bash
# Train
python3 src/evaluate_pipeline.py --dump-finetune Data/finetune/bc5cdr_train_sft.jsonl \
  --dataset bc5cdr --split train --keep-duplicates \
  --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
  --llm-abbreviation --llm-abbreviation-model qwen3-4b \
  --llm-top-k 10 --prompt-version v8 --finetune-target full --finetune-max-per-pair 5
# Dev (fürs Early Stopping) — dasselbe mit --split dev
```

Format: Chat-JSONL (system/user/assistant), Ziel = Gold-Kandidat, keine Few-Shots,
häufige Mentions auf 5 gedeckelt.

## 2. Installation auf Bender (einmalig)

Versionen gepinnt, weil TRLs API sich schnell ändert:

```bash
python3 -m venv ft_env && source ft_env/bin/activate
pip install "torch" "transformers>=4.46" "trl>=0.12,<0.14" "peft>=0.13" \
            "bitsandbytes>=0.43" "accelerate>=0.34" "datasets>=2.20"
```

Falls dein TRL abweicht: die 3 heiklen Zeilen im Skript sind mit `[TRL]` markiert
(`max_seq_length` vs `max_length`, `processing_class` vs `tokenizer`, das
Completion-Only-Template).

## 3. Trainieren

```bash
# lokaler Smoke-Test (läuft es durch?)
python3 src/finetune_lora.py --model Qwen/Qwen3-0.6B --epochs 1 --limit 200

# echter Lauf auf Bender
sbatch cluster/bender_finetune.sbatch        # squeue --me | tail -f logs/nel-lora-*.out
```

Im `.sbatch` vorher `--partition` und die `module load`-Namen an Bender anpassen
(HPC-Wiki). Ergebnis: LoRA-Adapter + gemergtes Modell in `out/…-merged`.

Zwei Dinge sind schon eingebaut: **Completion-Only-Loss** (trainiert nur auf der
Antwort, nicht dem langen Prompt) und ein **Truncation-Check** (warnt, wenn
Beispiele länger als `--max-seq-len` sind → dann Wert hoch oder Daten mit
`--max-definition-len 300` neu bauen).

## 4. Evaluieren — der eigentliche Test

Feinabgestimmtes 0.6B **gegen sein eigenes Zero-Shot-0.6B** (NICHT gegen 27B).
Beide als Phase 4 servieren, mit `--dump-predictions`, dann McNemar:

```bash
# Zero-Shot-Baseline
python3 src/evaluate_pipeline.py --model Qwen/Qwen3-0.6B \
  --base-url http://localhost:1234/v1 --api-key lm-studio \
  --wikidata --dbpedia --umls Data/UMLS/MRCONSO.RRF --embedding \
  --llm-top-k 10 --prompt-version v8 --dataset bc5cdr --temperature 0 \
  --structured-output --keep-duplicates --dump-predictions preds_0.6b_zeroshot.jsonl

# Feinabgestimmt (gemergtes Modell servieren, sonst identische Flags)
python3 src/evaluate_pipeline.py --model <merged-modell> ... \
  --dump-predictions preds_0.6b_finetuned.jsonl

python3 src/mcnemar_compare.py preds_0.6b_zeroshot.jsonl preds_0.6b_finetuned.jsonl
```

Erfolgssignal: nicht die absolute Accuracy (0.6B bleibt schwächer als 27B),
sondern **signifikanter Anstieg der Disambiguierungs-Accuracy** + **Rückgang der
Konventionsfehler**. Wenn ja → These belegt, hoch auf Qwe3-4B (im `.sbatch` schon
vorbereitet, mit `--load-in-4bit` QLoRA).
