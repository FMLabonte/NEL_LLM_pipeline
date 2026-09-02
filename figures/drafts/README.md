# figures/drafts — Entwürfe für Figure 1 (Stand 23.08.2026)

Vier Kandidaten, alle aus den `preds_*.jsonl` im Repo-Root gerechnet.
Kein Modell, kein Cluster nötig — nur numpy + matplotlib.

    cd figures/drafts
    python3 make_v1_budget.py        # A · Budget-Kurve      -> V1_budget.{png,pdf}
    python3 make_v2_profile.py       # B · Value-Profile     -> V2_profile.{png,pdf}
    python3 make_v3_substitution.py  # C · Substitution      -> V3_substitution.{png,pdf}
    python3 make_v4_deference.py     # D · Deference         -> V4_deference.{png,pdf}

`common.py` hält alles Gemeinsame:

* `RUNS` — Lauf -> Datei -> Farbe/Marker (hier anpassen, um Zeilen hinzuzunehmen)
* `budget_curve(rows)` — Accuracy@1 als Funktion des Anteils an das LLM geschickter
  Mentions (Gate-Sweep über `p3_score_gap`, unsicherste zuerst)
* `dev_tuned_point(rows)` — Schwelle auf der Hälfte der *Dokumente* getunt, auf der
  anderen berichtet (★ in Entwurf A)
* `value_profile(rows)` — lokales Netto-Δ Accuracy@1 im gleitenden Fenster
* `DS` — CVD-validierte Palette pro Datensatz
* `style(ax)` — einheitliches Achsen-Styling

## Zwei Fallstricke, die im Code noch NICHT behandelt sind

1. `preds_zs_bc5cdr_bm25.jsonl` benutzt **999.0 als Sentinel** in `p3_score_gap`
   (10,9 % der Zeilen, kein Top-2 vorhanden). Vor jeder Perzentil-Bildung
   ausschließen oder separat führen — sonst landen sie geschlossen im obersten
   Konfidenz-Quintil. Betrifft auch `figures/make_fig1_heatmap.py`.
2. Diese Datei hat **9.391 statt 9.661** Zeilen. Für die Aussage „identische
   Mentions" in Entwurf A, Panel (a), müsste auf die Schnittmenge eingeschränkt
   werden.
