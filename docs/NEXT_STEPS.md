# Nächste Schritte — Runbook (Stand 02.09.2026)

Die Abschnitte 1–4 sind **durch**: Prompt-Ablation, Retriever-Degradation,
zwei weitere Disambiguatoren (Qwen3-8B, Granite-3.1-8B) und drei
Fine-Tuning-Seeds liegen im Repo und sind im Paper eingearbeitet. Der Einwand
„ein Modell, ein Seed" ist damit erledigt und aus den Limitations gestrichen.

Offen ist noch **eine** inhaltliche Lücke: es gibt keinen Nicht-LLM-Re-Ranker
als Vergleich. Solange der fehlt, lautet die naheliegende Rückfrage „ihr habt
gezeigt, dass das LLM wenig bringt — aber verglichen womit?". Abschnitt 6
schließt das; das ist der nächste Job auf dem Cluster.

---

## 0. Dateien auf Bender kopieren

```bash
cd "/Users/moritz/Desktop/NLP Lab/NEL_LLM_pipeline"
B=s24mgilg@bender.hpc.uni-bonn.de

# Cross-Encoder-Baseline (Abschnitt 6) — das ist der aktuelle Stapel
rsync -avP cluster/bender_dump_candidates.sbatch cluster/bender_crossencoder.sbatch $B:NEL_LLM_pipeline/cluster/
rsync -avP scripts/rerank_crossencoder.py scripts/train_crossencoder.py \
                                                   $B:NEL_LLM_pipeline/scripts/
rsync -avP src/evaluate_pipeline.py src/analyze_robustness.py $B:NEL_LLM_pipeline/src/

# Historisch (Abschnitte 1–4, schon gelaufen)
rsync -avP cluster/bender_multimodel.sbatch cluster/bender_finetune_seeds.sbatch \
           cluster/bender_eval_ft_seeds.sbatch                    $B:NEL_LLM_pipeline/cluster/
rsync -avP src/finetune_lora.py                           $B:NEL_LLM_pipeline/src/
```

`evaluate_pipeline.py` **muss** mit — die alte Fassung kennt
`--dump-candidates` nicht, und `cluster/bender_dump_candidates.sbatch` bricht dann mit
`unrecognized arguments` ab.

`finetune_lora.py` ist nicht optional — die alte Fassung kennt `--seed` nicht,
und dann trainieren alle drei Array-Tasks dasselbe Modell. `cluster/bender_finetune_seeds.sbatch`
prüft das vor dem Start und bricht mit einer klaren Meldung ab, statt zehn
Stunden GPU zu verbrennen.

Zurückholen, wenn die Jobs durch sind:

```bash
rsync -avP $B:'NEL_LLM_pipeline/preds_zs_*_qwen3-8b.jsonl \
                NEL_LLM_pipeline/preds_zs_*_granite31-8b.jsonl \
                NEL_LLM_pipeline/preds_ft_*_s[123].jsonl' .
rsync -avP $B:NEL_LLM_pipeline/logs/ bender_logs/
```

---

## 1. ✅ Prompt-Ablation — durch

`preds_zs_{bc5cdr,biored}_v8-{notrust,noscore,gap}.jsonl` liegen im Repo,
`scripts/figures/fig7_prompt.pdf` und Section 5.3 sind gebaut. Nichts mehr zu tun.

## 2. ✅ Retriever-Degradation — durch

`preds_dose_p{0.1…0.5}.jsonl` liegen im Repo, `scripts/figures/fig6_dose.pdf` und
Section 4.1 sind gebaut. R@1 fällt von 84,8 auf 42,4 bei konstantem R@10 = 90,5 —
die kausale Dose-Response-Kurve, die dem Frame vorher fehlte.

---

## 3. ✅ Zweiter und dritter Disambiguator — durch

**Warum das der letzte wirklich harte Einwand ist.** Die Danger Zone, die
Deference-Verschiebung, das Value Profile — alles gemessen an Qwen3-4B. Ein
Reviewer fragt zu Recht, ob das eine Eigenschaft *von LLM-Disambiguatoren* ist
oder eine Eigenschaft *von diesem einen 4B-Modell*.

```bash
sbatch cluster/bender_multimodel.sbatch      # Array 1–2, je ~10 h, läuft parallel
```

Zwei Modelle, ein Array-Task pro Modell, alles andere identisch zu den
v8-Referenzläufen (`--wikidata --dbpedia --umls … --embedding`, `--prompt-version v8`,
`--temperature 0`, `--structured-output`, `--keep-duplicates`):

| Modell | Rolle |
|---|---|
| `Qwen/Qwen3-8B` | gleiche Familie, doppelte Größe → Artefakt kleiner Modelle? |
| `ibm-granite/granite-3.1-8b-instruct` | andere Familie, **gleiche** Größe → Artefakt von Qwen? |

Beide sind ungated, **kein HF-Login nötig**. Granite ist Apache-2.0 und mit 8B
exakt so groß wie Qwen3-8B — damit ist die Modellfamilie der einzige
Unterschied zwischen den beiden Vergleichsläufen, und das ist genau die Frage.

Vorher einmal auf dem Login-Knoten cachen, damit der Job keine GPU-Zeit mit
Downloaden verbrennt:

```bash
module load Python && source eval_env/bin/activate
df -h ~                                        # ~35 GB frei? je Modell ~16 GB
hf download Qwen/Qwen3-8B
hf download ibm-granite/granite-3.1-8b-instruct
```

**Wenn ihr ein anderes Modell eintragt:** vorher `max_position_embeddings` in
dessen `config.json` prüfen — es muss ≥ 8192 sein. Ein kürzeres Fenster lässt
vLLM zwar starten, aber die längsten Prompts werden abgelehnt, und dann rechnet
der Lauf auf einer *anderen Teilmenge* als die Referenz; die Danger-Zone-Bänder
wären nicht mehr vergleichbar. Der Job prüft das inzwischen selbst und bricht
vor dem vLLM-Start ab. Konkret ungeeignet: `allenai/OLMo-2-1124-*-Instruct`
(nur 4096 Positionen). Geeignet wären `microsoft/phi-4` (MIT, 16k, aber 14B →
`GPU_UTIL` auf 0.85) oder `tiiuae/Falcon3-7B-Instruct` (32k);
`analyze_robustness.py` kennt beide Tags schon.

Der Job läuft vier Korpora, kleine zuerst (BioRED → NCBI → NLM-Chem → BC5CDR),
damit bei Wall-Clock-Abbruch drei vollständige Dateien dastehen statt keiner.
Übersprungen wird nur, was **zeilengleich** zum Referenzlauf ist — ein
abgeschnittener Lauf wird neu erzeugt, nicht stillschweigend übernommen.

### NLM-Chem ist die Ausnahme — bewusst

`preds_zs_nlmchem.jsonl` ist **nicht** das volle Korpus: `cluster/bender_eval_nlmchem.sbatch`
hat es mit `--limit 1500 --limit-random --limit-seed 42` erzeugt, also aus einer
zufällig gemischten 1500er-Stichprobe von 11.729 Mentions. Der Multi-Modell-Job
fährt das volle Korpus (mehr Power für die Bänder), deshalb meldet er dort
`WARNUNG: … hat 11729 statt 1500 Zeilen` — das ist erwartet, kein Fehler.

`analyze_robustness.py` beschneidet vor jedem Vergleich auf die Mentions, die
**alle** beteiligten Läufe enthalten, und schreibt die Beschneidung in die
Kopfzeile. Weil `--limit-random` mischt, ist das eine mengenbasierte Zuordnung
über `(pmid, mention, gold_id)` und nicht `common_subset()` aus `aclfig.py` —
das setzt eine geordnete Teilmenge voraus und würde hier abbrechen.

Zusätzlich prüft das Skript nach dem Beschneiden, ob alle Läufe zeilenweise
dasselbe `p3_top1` und denselben `p3_score_gap` haben. Das muss 0 Abweichungen
geben; sonst stammen die Läufe aus verschiedenen Retriever-Konfigurationen und
dürfen gar nicht verglichen werden. Die Meldung sagt das dann explizit.

Für die Seed-Läufe ist es andersherum gelöst: `cluster/bender_eval_ft_seeds.sbatch`
schränkt NLM-Chem jetzt mit genau denselben Flags ein wie die Referenz. Da gibt
es keine bestehenden Ergebnisse zu retten, also ist die exakt gleiche Stichprobe
sauberer — und spart pro Seed über 10.000 Mentions Rechenzeit.

**Die Basislinie nachziehen** (`cluster/bender_eval_nlmchem_full.sbatch`, ~4–5 h):

```bash
sbatch cluster/bender_eval_nlmchem_full.sbatch    # -> preds_zs_nlmchem_full.jsonl
```

Qwen3-8B und Granite laufen auf dem vollen Korpus, nur Qwen3-4B nicht — und
solange das so ist, wird der Vergleich auf 1500 Mentions zurückgeschnitten. Im
Band [10,20) bleiben davon ~300 übrig, mit einem Bootstrap-Intervall von rund
±4,5. „Kein Effekt nachweisbar" lässt sich dort nicht von „zu wenig Daten"
unterscheiden. Mit dem vollen Lauf wächst das Band auf ~4000 Mentions und das
Intervall schrumpft auf etwa ±1,7 — dann steht fest, was auf NLM-Chem gilt.
Sobald die Datei existiert, benutzt `analyze_robustness.py --models` sie
automatisch als NLM-Chem-Referenz. Die Seed-Auswertung bleibt bewusst auf der
1500er Stichprobe, weil die Fine-Tuning-Läufe genau dagegen gemessen werden.

Auswerten:

```bash
python3 src/analyze_robustness.py --models
```

Die einzige Spalte, auf die es ankommt, ist `band [10,20)`. Bleibt das
Vorzeichen über die Modelle hinweg negativ, ist die Danger Zone keine
Qwen-Eigenschaft. Dreht es bei 8B ins Positive, ist das ebenfalls ein
publizierbares Ergebnis — dann skaliert das Problem weg, und das gehört genau
so ins Paper.

Darunter steht pro Korpus ein Dokument-Bootstrap-Block. Der ist **gepaart**:
ein Resample der Dokumente wird auf alle Modelle gleichzeitig angewandt, weil
sie nach dem Angleichen dieselben Mentions in derselben Reihenfolge haben.
Für die Aussage „Skalierung dämpft den Effekt" zählt deshalb ausschließlich das
Differenz-Intervall in der Spalte `vs Qwen3-4B` — **nicht**, ob sich die beiden
Einzelintervalle überschneiden. Bei stark korrelierten Größen überlappen die
Einzelintervalle fast immer, auch wenn die Differenz eindeutig ist; das ist der
Standardfehler beim Ablesen solcher Tabellen. Nur was in der Differenzspalte
`signifikant` heißt, darf im Paper als Unterschied behauptet werden.

---

## 4. ✅ Drei Fine-Tuning-Seeds — durch

Finding 2 (Deference) steht auf **einem** LoRA-Lauf. Ohne Seed-Varianz lässt
sich nicht sagen, wie viel vom Präzisionssprung 18 % → 64 % Effekt und wie viel
Rauschen ist.

```bash
sbatch cluster/bender_finetune_seeds.sbatch      # Array 1–3, je ~6–8 h
# wenn die drei -merged-Verzeichnisse da sind:
sbatch cluster/bender_eval_ft_seeds.sbatch       # Array 1–3, je ~10 h
python3 src/analyze_robustness.py --seeds
```

Hyperparameter sind byte-identisch zu `cluster/bender_finetune_4b.sbatch` — einzige
Änderung ist `--seed $SLURM_ARRAY_TASK_ID`, das in `SFTConfig` sowohl `seed`
(Gewichts-Init, Dropout) als auch `data_seed` (Reihenfolge, Shuffling) setzt.
Ein bereits existierendes `out/qwen3-4b-nel-lora-s<N>-merged` wird übersprungen,
Neu-Einreichen nach einem Abbruch ist also gefahrlos.

Der bisherige Einzellauf `preds_ft_bc5cdr.jsonl` erscheint im Report als Zeile
`ft (bisher)`, zählt aber **nicht** in die Spanne: er lief mit TRLs Default-Seed
und ohne gesetztes `data_seed`, ist also kein vergleichbarer vierter Zug.

Was ins Paper muss: Mittelwert und Spanne statt Punktschätzern, plus der Satz,
ob sich seen- und unseen-Spanne berühren. Berühren sie sich nicht, ist die
Deference-Aussage seed-robust und kann so stehen bleiben.

Platzbedarf: drei Merges à ~8 GB in `out/`. Vorher `df -h ~` prüfen; die
Adapter allein wären klein, aber `--merge` braucht die vollen Gewichte fürs
Servieren.

---

## 5. Vor dem Einreichen — anonymes Release

```bash
python3 scripts/make_release.py
```

Baut `release/` mit den preds-Dumps, den Figuren- und Analyse-Skripten und
einem README, und scannt danach alles auf Klarnamen, `s24mgilg`,
`bender.hpc.uni-bonn.de`, `/Users/moritz/…`, E-Mail-Adressen und den
Google-Docs-Link. Der Scanner ist gegen eure echten Dateien getestet — er
findet 23 Treffer allein in `README.md` und `RETRIEVER_CURVE_STEPS.md`. Diese
beiden Dateien gehören **nicht** ins Release.

Wichtig: nicht das Repo selbst hochladen, sondern nur das erzeugte Verzeichnis
— die Git-History enthält alles, was der Scanner findet, und mehr.

---

## 6. Cross-Encoder-Baseline — der letzte offene Punkt

Die Frage, die das Paper bisher nicht beantwortet: **hätte ein billiger
Re-Ranker dasselbe gekonnt?** Ohne diese Zahl steht „das LLM bringt hinter
einem starken Retriever wenig" ohne Vergleichsgröße da.

Der Aufbau ist bewusst *nicht* in die Pipeline eingebaut. Der Re-Ranker liest
die schon berechneten Kandidatenlisten und schreibt einen preds-Dump im
identischen Schema. Dadurch sind Mentions, Listen und alle P3-Felder
byte-gleich mit dem LLM-Lauf — der Dokument-Bootstrap ist exakt gepaart, ohne
Teilmengen und ohne Ausreden.

### Schritt A — Kandidatenlisten dumpen (8–11 h, GPU, kein vLLM)

```bash
sbatch cluster/bender_dump_candidates.sbatch
```

**Zeitbudget, gemessen statt geschätzt.** Der erste Anlauf (Job 248348) lief in
ein 3-Stunden-Limit, das ich zu knapp gesetzt hatte — A40short erlaubt 8 h,
A40medium einen Tag. Aus dem Lauf: BC5CDR-Test ~2 h 50 (9.661 Mentions, rund
1 s/Mention ohne LLM), BioRED 9 min. Der Job steht jetzt auf A40medium/16 h und
arbeitet die Korpora nach Wert ab — BC5CDR-Test zuerst, weil es sowohl den
Cross-Encoder als auch die GRPO-Vorstudie freischaltet, NLM-Chem zuletzt.

Ein Dump gilt nur als fertig, wenn die Begleitdatei `*.docs.jsonl` existiert
**und** die Kandidaten `synonyms` tragen. Die erste Fassung schrieb beides
nicht; ein blosses „Datei existiert" würde genau die unbrauchbaren Dumps
überspringen — derselbe Fehlertyp wie damals beim abgeschnittenen LoRA-Merge.
Alte Dumps werden deshalb automatisch verworfen und neu erzeugt.

Läuft mit `--no-phase4`, also ohne einen einzigen LLM-Call. Die GPU wird nur
für SapBERT/FAISS gebraucht, weil die Referenzläufe `--embedding` benutzen. Die
Retriever-Flags sind wörtlich aus den Eval-Jobs kopiert; wer einen davon
ändert, muss ihn an beiden Stellen ändern, sonst meldet
`analyze_robustness.py` einen P3-Mismatch (genau dafür ist die Prüfung da).

Erzeugt `cands_p3_{bc5cdr,biored,ncbi,nlmchem}.jsonl` plus
`cands_p3_bc5cdr_train.jsonl` für das Training, und zu jedem eine
`*.docs.jsonl` mit den Dokumenttexten. Jede Zeile enthält die Top-10-Liste nach
P3, den Satz um die Mention, die Definitionen und die Synonyme.

Warum die Dokumenttexte separat: **v8 benutzt gar nicht den Satz**, sondern ein
64-Wort-Fenster um die Mention im vollen Dokument (`context_words=64`,
`use_full_context=False`). Ohne die Texte lässt sich der Prompt offline nicht
rekonstruieren, und jede Aussage „gleicher Prompt wie im Paper" wäre falsch.
Einmal pro Dokument statt pro Mention geschrieben — bei NLM-Chem wären das
sonst rund 500 MB.

### Schritt B — Re-Ranker scoren (ca. 30 min)

```bash
sbatch cluster/bender_crossencoder.sbatch
```

Zwei Varianten, weil ein Reviewer nach beiden fragt:

| Tag | Was | Warum |
|---|---|---|
| `medcpt` | `ncbi/MedCPT-Cross-Encoder`, zero-shot, 109 M Parameter | ungetunt, ~1/40 der Größe des Disambiguators |
| `medcpt-ft` | dasselbe Modell, auf dem BC5CDR-Train-Split trainiert | fairer Gegenpart zum LoRA-Lauf in Section 6 |

Das Training benutzt **dieselbe Aufgabe** wie das LLM: listwise Softmax über
die Kandidatenliste, Ziel ist der Gold-Slot. Mentions ohne Gold in der Liste
fallen raus — es gibt nichts Richtiges zu wählen. Der Split geht über
**Dokumente**, nicht über Mentions, sonst leckt der Kontext.

Ausgabe: `preds_zs_<corpus>_medcpt.jsonl` und `…_medcpt-ft.jsonl`. Diese Namen
sind kein Zufall — `MODEL_TAGS` in `analyze_robustness.py` kennt sie schon,
also erscheinen beide ohne weiteres Zutun in der Modelltabelle, im gepaarten
Bootstrap und in den Kalibrierungs-Quotienten.

### Schritt C — auswerten

```bash
python3 src/analyze_robustness.py --models
```

### Was das Ergebnis bedeutet — beide Richtungen sind verwertbar

* **Der Cross-Encoder verliert deutlich** → das LLM kauft echtes
  Kontextverständnis, nicht nur einen zweiten Ranking-Durchgang. Das stärkt die
  Arbeit und macht die Danger Zone zu einer Aussage über *Kalibrierung*, nicht
  über Kompetenz.
* **Der Cross-Encoder ist gleichauf oder besser** → das ist das schärfere
  Ergebnis, nicht das schlechtere: dann ist die 4B-Stufe ein sehr teurer Weg zu
  etwas, das 109 M Parameter auch können, und das Paper bekommt eine konkrete
  Empfehlung statt einer Warnung.

Was nicht passieren darf, ist die Baseline zu verkrüppeln. Deshalb sieht sie
dieselben Top-10, dieselben Namen, dieselben Definitionen und denselben Satz —
und deshalb gibt es die trainierte Variante überhaupt.

### Kontrollen, die eingebaut sind

* Rank-1 des Dumps ≠ `p3_top1` der Referenz → Zeilenzahl wird gemeldet und der
  Lauf als „nicht dieselbe Retriever-Konfiguration" markiert.
* Referenz-Mentions ohne Kandidatenliste werden gezählt, nicht stillschweigend
  weggelassen (sonst ändert sich unbemerkt die ausgewertete Teilmenge).
* Beide sbatch-Dateien haben die `dd`-Quota-Probe und überspringen fertige
  Ausgaben, sind also gefahrlos wiederholbar.

---

## 7. GRPO-Vorstudie — bevor irgendwer RL-Code schreibt

```bash
sbatch cluster/bender_grpo_pilot.sbatch      # braucht nur cands_p3_bc5cdr* aus Abschnitt 6
```

Eine Frage: **hat GRPO auf dieser Aufgabe überhaupt einen Gradienten?** GRPO
normiert den Reward innerhalb der Gruppe,

    A_i = (r_i − mean(r)) / std(r)

Sind alle *G* Rollouts gleich gut, ist die Standardabweichung null, jeder
Advantage null, und die Gruppe trägt nichts zum Update bei. Bei
enum-beschränktem Decoding über zehn Kandidaten und einem Retriever, der auf
BC5CDR schon 82,7 % richtig liegt, ist das nicht der Randfall, sondern das
erwartbare Normalverhalten. Das ist die eine ungetestete Annahme, auf der der
ganze RL-Teil steht — und sie kostet hier zwei GPU-Stunden statt zwei Wochen.

Kein RL, keine Trainingsschleife: *G* Rollouts pro Mention bei mehreren
Temperaturen, durch denselben Endpunkt, denselben v8-Prompt und denselben
Enum-Decoder. Jede Gruppe fällt in eine von vier Klassen:

| Klasse | Bedeutung |
|---|---|
| alle richtig | kann das Modell schon |
| alle falsch, Gold in der Liste | schwer, aber lernbar |
| alle falsch, Gold nie retrieved | **unlernbar**, muss aus dem RL-Set raus |
| gemischt | nicht-degeneriert: das effektive Trainingsset |

`T = 0.0` läuft als Kontrolle mit: dort **muss** der gemischte Anteil exakt 0
sein. Ist er das nicht, sampelt vLLM trotz `temperature 0` und alle anderen
Zahlen messen etwas anderes als gedacht.

Die Aufschlüsselung nach Confidence-Band ist der inhaltlich interessante Teil.
Landen die gemischten Gruppen im mittleren Band, setzt GRPO seinen Gradienten
genau dort an, wo das Paper das Problem verortet — dann hängen die beiden
Hälften des Projekts inhaltlich zusammen und nicht nur zeitlich.

**Was ein schlechtes Ergebnis bedeutet.** Liegt der gemischte Anteil im
niedrigen einstelligen Bereich, ist naives GRPO hier tot, und es braucht einen
dichteren Reward (Rang des Golds statt binär), Curriculum-Filterung auf die
lernbaren Gruppen, oder mehr Temperatur mit dem entsprechenden Rauschen. Das
ist kein Rückschlag, sondern der Befund, der die Methodensektion des
RL-Papers trägt.

---

## Reihenfolge, wenn die Zeit knapp ist

Der Lab Report ist am **23.09.2026** fällig, also gut drei Wochen.

1. `cluster/bender_dump_candidates.sbatch` (Abschnitt 6, 8–11 h). Schaltet beides
   frei: den Cross-Encoder und die GRPO-Vorstudie.
2. `cluster/bender_crossencoder.sbatch` (~30 min) — der einzige inhaltlich offene Punkt
   am Paper. `cluster/bender_grpo_pilot.sbatch` (~2 h) kann parallel laufen, sobald
   `cands_p3_bc5cdr*` da ist; es gehört zum RL-Teil, nicht zum Lab Report.
3. Lab Report schreiben. Das Paper ist inhaltlich fertig; für den Report fehlt
   nur die Einordnung, nicht neue Empirie.
4. `scripts/make_release.py` (Abschnitt 5) — erst kurz vor einer echten
   Einreichung nötig, nicht für den Report.

Der Seitenumfang ist zu beachten: das Paper liegt bei ~9,7 Seiten Inhalt, das
\*ACL-Limit sind 8. Für den Report irrelevant, vor einer Einreichung der erste
Schnitt (Kandidat: Abschnitt 5.4 in den Appendix).

Historisch: `cluster/bender_multimodel.sbatch`, `cluster/bender_finetune_seeds.sbatch` und
`cluster/bender_eval_ft_seeds.sbatch` sind gelaufen. Alle Jobs schließen `node-02` aus
(12.676 nicht korrigierbare ECC-Fehler, gemeldet) und prüfen zur Laufzeit, ob
die zugeteilte GPU ECC-Fehler zählt.

---

## Was schon erledigt ist

* **Dedup-Kontrolle** — durchgerechnet, siehe unten, und im Paper eingearbeitet.
* **Multiplizität** — Absatz in Section 3: Band-Aufschlüsselungen sind
  deskriptiv, drei Analysen sind konfirmatorisch.
* **Die drei erklärungsbedürftigen Zahlen** — NCBI lexikalisch > SapBERT,
  die 0-%-Zelle, MedMentions 4.823 vs. 5.000: alle jetzt im Text erklärt.
* **Prompt-Ablation** und **Dose-Response** — gelaufen, ausgewertet, Figuren
  und Sections gebaut.

### Die Dedup-Kontrolle hat etwas gefunden

Alles unter drei Protokollen nachgerechnet: pro Mention-Vorkommen (euer
bisheriges), pro (Dokument, Mention, Konzept), und global eindeutige
(Mention, Konzept)-Paare.

**Hält:**

| Lauf | Band | Vorkommen | pro Dok. | eindeutig |
|---|---|---|---|---|
| BC5CDR Pipeline | [10,20) | −2,4 | −2,2 | −1,5 |
| BC5CDR BioSyn | [5,10) | −2,5 | −3,9 | **−5,0** |
| BC5CDR BioSyn | [10,20) | −1,7 | −1,4 | −1,5 |
| NCBI | [10,20) | −7,6 | −5,3 | −6,2 |

Die Danger Zone auf BioSyn wird ohne Duplikate **stärker**. Und die
seen/unseen-Aussage wird sauberer: 65,0 → 92,9 % auf gesehenen Konzepten gegen
71,0 → 69,6 % auf ungesehenen — die ungesehene Seite geht jetzt sogar leicht
nach unten.

**Hält nicht:**

| Lauf | Band | Vorkommen | pro Dok. | eindeutig |
|---|---|---|---|---|
| BioRED | [10,20) | −1,4 | **+0,8** | **+1,0** |
| NLM-Chem | [10,20) | −1,0 | **±0,0** | **±0,0** |

Auf BioRED und NLM-Chem ist die Danger Zone ein Duplikat-Artefakt — die Bänder
enthalten dort nur 205 bzw. 174 eindeutige Paare. Der Vorzeichentest des Frames
fällt von 12/14 (p = 0,013) auf 10/14 (p = 0,18), die Korrelation hält
(ρ = −0,67, p = 0,009).

Das Paper sagt das jetzt so: Replikation für BC5CDR (zwei Retriever) und NCBI,
protokollabhängig für BioRED und NLM-Chem. Section 5.3 plus Tabelle 3. Das
kostet etwas behauptete Stärke — aber ein Reviewer hätte es gefunden, und dann
wäre es teurer geworden.
