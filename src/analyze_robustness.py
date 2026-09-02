#!/usr/bin/env python3
"""Does the paper's story hold across disambiguators and across fine-tuning seeds?

Two checks, both run on prediction dumps only -- no model, no GPU.

  --models   Repeats the Finding-1 measurement for every extra disambiguator
             produced by bender_multimodel.sbatch. The question is narrow: does
             the mid-confidence band go net-harmful for this model too, on the
             corpora where it does for Qwen3-4B?

  --seeds    Reports the Finding-2 quantities for each LoRA seed produced by
             bender_finetune_seeds.sbatch, as mean and range, so the paper can
             say how much of the effect is run-to-run noise.

    python3 src/analyze_robustness.py --models
    python3 src/analyze_robustness.py --seeds
    python3 src/analyze_robustness.py --models --seeds --json robustness.json
"""
import argparse
import json
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
BANDS = [(0, 2), (2, 5), (5, 10), (10, 20), (20, 1e9)]
BLAB = ["[0,2)", "[2,5)", "[5,10)", "[10,20)", "20+"]

# corpus -> (zero-shot reference, suffix used by the corpus in the extra runs)
CORPORA = [("BC5CDR", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl", "bc5cdr"),
           ("BioRED", "preds_zs_biored.jsonl", "biored"),
           ("NCBI", "preds_zs_ncbi.jsonl", "ncbi"),
           ("NLM-Chem", "preds_zs_nlmchem.jsonl", "nlmchem")]
MODEL_TAGS = ["qwen3-8b", "granite31-8b", "llama31-8b", "falcon3-7b", "phi4",
              # Non-LLM re-rankers on the identical candidate lists
              # (scripts/rerank_crossencoder.py). They are not language models,
              # but every metric in this table -- gain, intervention rate,
              # precision, aggressiveness, selection skill -- is defined on
              # "the stage changed the retriever's top-1", which they do too.
              "medcpt", "medcpt-ft"]

# Full-corpus baselines that supersede a capped reference *for the model
# comparison only*. The seed report keeps the capped file, because the
# fine-tuned runs are measured against exactly that sample.
FULL_REF = {"NLM-Chem": "preds_zs_nlmchem_full.jsonl"}
SEEDS = [1, 2, 3]


def load(name):
    p = ROOT / name
    if not p.exists():
        return None
    rows = [json.loads(l) for l in open(p) if l.strip()]
    return [r for r in rows if r.get("p3_score_gap") is not None]


def acc(rows, f):
    return float(np.mean([bool(r.get(f)) for r in rows]) * 100) if rows else float("nan")


def band_of(g):
    for i, (lo, hi) in enumerate(BANDS):
        if lo <= g < hi:
            return i
    return None


def precision(rows):
    fx = sum(1 for r in rows if r.get("p4_correct") and not r.get("p3_correct"))
    bk = sum(1 for r in rows if r.get("p3_correct") and not r.get("p4_correct"))
    return (fx / (fx + bk) if fx + bk else None), fx, bk


def doc_ci(rows, stat, nb=600, seed=0):
    by = defaultdict(list)
    for r in rows:
        by[r.get("pmid")].append(r)
    docs = list(by)
    if not docs:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(nb):
        s = []
        for _ in range(len(docs)):
            s.extend(by[rng.choice(docs)])
        v = stat(s)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    vals.sort()
    return vals[int(.025 * len(vals))], vals[int(.975 * len(vals))]


def paired_band_bootstrap(runs, i=3, nb=500, seed=0):
    """Document-level bootstrap of the band-i delta, *paired* across runs.

    After align() the runs hold the same mentions in the same order, and the
    band is defined by `p3_score_gap`, which align() has verified is identical
    across them. So one document resample can be applied to every run at once,
    and the difference between two models can be read off the same resample.

    That pairing matters. Two independent CIs on two strongly correlated
    quantities overlap far more than the uncertainty of their difference
    warrants -- reading "the intervals overlap, so it is not significant" off
    unpaired intervals is the classic way to miss a real effect. The difference
    CI reported here is the one to quote.

    Resampling whole documents, not mentions: mentions inside one abstract share
    context and repeat surface forms, so they are not independent draws.
    """
    inband = (np.ones(len(runs[0]), bool) if i is None
              else np.array([band_of(r["p3_score_gap"]) == i for r in runs[0]]))
    if not inband.any():
        return None
    docs = {}
    for pos, r in enumerate(runs[0]):
        docs.setdefault(r.get("pmid"), []).append(pos)
    keys = list(docs)
    C = np.array([int(inband[docs[k]].sum()) for k in keys], float)
    S = np.array([[float(sum((1 if rows[p]["p4_correct"] else 0)
                             - (1 if rows[p]["p3_correct"] else 0)
                             for p in docs[k] if inband[p])) for k in keys]
                  for rows in runs])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(nb, len(keys)))
    den = C[idx].sum(axis=1)                       # (nb,)
    ok = den > 0
    if not ok.any():
        return None
    num = S[:, idx].sum(axis=2)                    # (nruns, nb)
    vals = num[:, ok] / den[ok] * 100.0

    def pct(v):
        return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

    # Point estimates come from the data, not from the mean of the resamples --
    # otherwise the printed number drifts from the one in the table above.
    obs = S.sum(axis=1) / C.sum() * 100.0
    out = []
    for k in range(len(runs)):
        d = vals[k] - vals[0]
        lo, hi = pct(d)
        out.append({"point": float(obs[k]), "ci": pct(vals[k]),
                    "diff": None if k == 0 else float(obs[k] - obs[0]),
                    "diff_ci": None if k == 0 else (lo, hi),
                    "diff_sig": None if k == 0 else not (lo <= 0 <= hi)})
    return {"n_band": int(inband.sum()), "n_docs": len(keys), "nb": int(ok.sum()),
            "runs": out}


def calibration(rows):
    """Intervention rate divided by the retriever's error rate, band by band.

    Reads only `llm_changed` (what the model decided) and `p3_correct` (whether
    the retriever was right). It never touches `p4_correct`, so it is
    independent of the band delta and cannot be a restatement of it.

    That independence is the point. The tempting "mechanism" analysis --
    regressing the band delta on intervention rate and intervention precision --
    is circular: up to neutral changes, delta = rate x (2 x precision - 1). It
    would look like evidence and be arithmetic. This quantity instead describes
    the model's decision *policy* against the state of the world it is reacting
    to, and the outcome plays no part in it.

    Reading: 1.0 means the model intervenes exactly as often as the retriever is
    wrong. Below 1 it is more restrained than the error rate. Above 1 it must be
    overriding correct top-1 predictions, because there are not enough wrong
    ones to account for the interventions. A policy that tracked retriever
    reliability would stay flat or fall across the bands; a rising profile means
    the model becomes *relatively* more interventionist exactly where the
    retriever gets more reliable.
    """
    rate, skill = [], []
    for i in range(len(BANDS)):
        sub = [r for r in rows if band_of(r["p3_score_gap"]) == i]
        err = 100.0 - acc(sub, "p3_correct") if sub else 0.0
        if not sub or err <= 0:
            rate.append(None); skill.append(None); continue
        rate.append(acc(sub, "llm_changed") / err)
        # Auswahlguete: haette das Modell innerhalb des Bandes blind ueberschrieben,
        # traefe es die falschen Top-1 mit genau der Fehlerrate des Bandes. Die
        # Praezision daran gemessen sagt, ob die Auswahl ueberhaupt Information
        # traegt -- 1.0 ist Zufall, darunter schlechter als Zufall. Das ist keine
        # Umformulierung des Deltas: die Fehlerrate kommt darin nicht vor.
        p, _, _ = precision(sub)
        skill.append(None if p is None else (p * 100.0) / err)
    return rate, skill


def confident_delta(rows, lo=20):
    g = np.array([float(r["p3_score_gap"]) for r in rows])
    o = np.argsort(g, kind="stable")
    d = np.array([(1 if rows[i]["p4_correct"] else 0) - (1 if rows[i]["p3_correct"] else 0)
                  for i in o], float)
    return float(d[int(lo / 100 * len(d)):].mean() * 100)


def band_row(rows, i):
    sub = [r for r in rows if band_of(r["p3_score_gap"]) == i]
    if not sub:
        return None
    p, fx, bk = precision(sub)
    return {"n": len(sub), "rate": acc(sub, "llm_changed"),
            "precision": None if p is None else p * 100,
            "delta": acc(sub, "p4_correct") - acc(sub, "p3_correct"),
            "fixes": fx, "breaks": bk}


def fmt_pct(v, w=19):
    """A band with zero interventions has no precision -- print '-', not a crash."""
    return f"{'-':>{w}} " if v is None else f"{v:{w}.0f}%"


def fmt_cell(p, ci, fx, bk):
    if p is None:
        return f"- (+{fx}/-{bk})"
    c = "" if ci is None else f" [{ci[0]*100:.0f},{ci[1]*100:.0f}]"
    return f"{p*100:.1f}%{c} (+{fx}/-{bk})"


def _key(r):
    return (r["pmid"], r["mention"], r.get("gold_id"))


def align(runs):
    """Restrict every run to the mentions that ALL runs contain.

    Runs of the same corpus are not automatically on the same mentions:
    `preds_zs_nlmchem.jsonl` was produced with `--limit 1500 --limit-random
    --limit-seed 42`, i.e. a *shuffled* random sample of the 11729 mentions,
    while the extra-model runs cover the corpus in full. Comparing two models
    across two different samples is exactly the confound this experiment is
    supposed to remove, so the comparison has to happen on the intersection.

    `aclfig.common_subset` cannot be reused here: it assumes the short run is an
    ordered prefix-style subset, which a shuffled sample is not.

    With --keep-duplicates a (pmid, mention, gold_id) key can repeat inside one
    document, so this matches by *count* and keeps the first k occurrences on
    each side. Which occurrence of a repeated key gets matched to which is then
    arbitrary -- but everything Phase 3 produces (top-1, score gap, and hence
    the confidence band) depends only on the mention string, not on where in the
    abstract it sits, so repeated keys carry identical P3 fields and the banding
    is unaffected. The returned `mismatch` count verifies exactly that: it is the
    number of aligned positions where the runs disagree on `p3_top1`. It must be
    0; anything else means the runs were not produced by the same retriever
    configuration and must not be compared at all.
    """
    if len(runs) < 2:
        return runs, 0
    # Occurrence queues per key, per run. Emitting in the ANCHOR run's order --
    # not each run's own file order -- is what makes the result row-aligned.
    # A capped reference produced with --limit-random is shuffled relative to a
    # full run, so keeping each run's own order would pair unrelated mentions.
    idx = [defaultdict(deque) for _ in runs]
    for j, rows in enumerate(runs):
        for pos, r in enumerate(rows):
            idx[j][_key(r)].append(pos)
    anchor = min(range(len(runs)), key=lambda j: len(runs[j]))
    out = [[] for _ in runs]
    for r in runs[anchor]:
        k = _key(r)
        if all(idx[j][k] for j in range(len(runs))):
            for j in range(len(runs)):
                out[j].append(runs[j][idx[j][k].popleft()])
    mismatch = sum(1 for tup in zip(*out)
                   if len({r.get("p3_top1") for r in tup}) > 1
                   or len({round(float(r["p3_score_gap"]), 4) for r in tup}) > 1)
    return out, mismatch


# ------------------------------------------------------------------ models
def models_report(out):
    print("#" * 96)
    print("# Does the mid-confidence band go net-harmful for other disambiguators too?")
    print("#" * 96)
    found = False
    for corpus, ref, tag in CORPORA:
        full = FULL_REF.get(corpus)
        if full and (ROOT / full).exists():
            ref = full
        variants = [("Qwen3-4B", ref)] + [
            (m, f"preds_zs_{tag}_{m}.jsonl") for m in MODEL_TAGS]
        avail = [(lab, load(f)) for lab, f in variants]
        avail = [(lab, r) for lab, r in avail if r]
        if len(avail) < 2:
            continue
        found = True
        raw = [len(r) for _, r in avail]
        aligned, mism = align([r for _, r in avail])
        avail = [(lab, rows) for (lab, _), rows in zip(avail, aligned)]
        note = ""
        if len(set(raw)) > 1:
            # Name every run's own row count. The intersection is over ALL runs,
            # so one short or half-finished dump silently shrinks the comparison
            # for every model in the table -- that has to be visible.
            per = ", ".join(f"{lab} {n}" for (lab, _), n in zip(avail, raw))
            note = f"\n  [gemeinsame Mentions: {per}  ->  {len(avail[0][1])}]"
        if mism:
            note += (f"\n  !! {mism} Zeilen mit abweichendem P3-Top1 -- die Laeufe stammen "
                     f"NICHT aus derselben Retriever-Konfiguration, nicht vergleichen.")
        print(f"\n{corpus}   (n={len(avail[0][1])}, Recall@1 {acc(avail[0][1],'p3_correct'):.1f})"
              f"{note}")
        print(f"  {'model':14}{'P4':>7}{'gain':>7}{'rate%':>8}{'prec':>7}"
              f"{'  band [10,20)':>18}{'confident 80 %':>17}")
        for lab, rows in avail:
            b = band_row(rows, 3)
            p, _, _ = precision(rows)
            bs = (f"{b['delta']:+6.1f} ({b['precision']:.0f}%)" if b and b["precision"] is not None
                  else "-")
            print(f"  {lab:14}{acc(rows,'p4_correct'):7.1f}"
                  f"{acc(rows,'p4_correct')-acc(rows,'p3_correct'):+7.1f}"
                  f"{acc(rows,'llm_changed'):8.1f}{(p or 0)*100:6.0f}%"
                  f"{bs:>18}{confident_delta(rows):+17.2f}")
            out.setdefault("models", {}).setdefault(corpus, {})[lab] = {
                "P4": acc(rows, "p4_correct"),
                "gain": acc(rows, "p4_correct") - acc(rows, "p3_correct"),
                "rate": acc(rows, "llm_changed"),
                "precision": None if p is None else p * 100,
                "band_10_20": b, "confident_delta": confident_delta(rows)}

        base = avail[0][0]
        runs_only = [r for _, r in avail]

        def block(title, bs):
            if not bs:
                return
            print(f"\n  {title}")
            for (lab, _), st in zip(avail, bs["runs"]):
                lo, hi = st["ci"]
                line = f"    {lab:14}{st['point']:+7.2f}  [{lo:+6.2f}, {hi:+6.2f}]"
                if st["diff"] is not None:
                    dlo, dhi = st["diff_ci"]
                    line += (f"   vs {base}: {st['diff']:+.2f} [{dlo:+.2f}, {dhi:+.2f}]"
                             f" {'signifikant' if st['diff_sig'] else 'n.s.'}")
                print(line)

        bs = paired_band_bootstrap(runs_only, i=3)
        if bs:
            block(f"Band [10,20), Dokument-Bootstrap (95 %, n={bs['n_band']} "
                  f"Mentions in {bs['n_docs']} Dokumenten):", bs)
            out.setdefault("band_ci", {})[corpus] = bs
        gs = paired_band_bootstrap(runs_only, i=None)
        if gs:
            # Der Gesamt-Gain braucht dasselbe Intervall: ein Modell, das netto
            # verliert, ist eine Behauptung ueber das ganze Korpus.
            block("Gesamt-Gain (P4 - P3), derselbe Bootstrap:", gs)
            out.setdefault("gain_ci", {})[corpus] = gs

        cal = {lab: calibration(rows) for lab, rows in avail}
        # Der Nenner beider Quotienten, einmal ausgeschrieben. Wird er klein
        # (rechte Spalte), sind die Quotienten numerisch instabil -- das soll man
        # sehen koennen, statt 10.18 fuer bare Muenze zu nehmen.
        base_rows = avail[0][1]
        errs = []
        for i in range(len(BANDS)):
            sub = [r for r in base_rows if band_of(r["p3_score_gap"]) == i]
            errs.append(None if not sub else 100.0 - acc(sub, "p3_correct"))
        print("\n  Retriever-Fehlerrate je Band (Nenner der beiden Quotienten):")
        print("    " + f"{'':14}" + "".join(f"{b:>9}" for b in BLAB))
        print("    " + f"{'':14}"
              + "".join("        -" if e is None else f"{e:8.1f}%" for e in errs))
        for which, title in ((0, "Interventionsrate / Retriever-Fehlerrate  (wie aggressiv)"),
                             (1, "Interventionspraezision / Retriever-Fehlerrate  "
                                 "(Auswahlguete, 1.00 = Zufall)")):
            print(f"\n  {title}")
            print("    " + f"{'Modell':14}" + "".join(f"{b:>9}" for b in BLAB))
            for lab, _ in avail:
                cells = "".join("        -" if v is None else f"{v:9.2f}"
                                for v in cal[lab][which])
                print(f"    {lab:14}{cells}")
        for lab, _ in avail:
            out.setdefault("calibration", {}).setdefault(corpus, {})[lab] = {
                "rate_over_error": cal[lab][0], "precision_over_error": cal[lab][1]}
    if not found:
        print("\n  Keine Modell-Laeufe gefunden. Erwartet z.B. preds_zs_bc5cdr_qwen3-8b.jsonl")
        print("  (bender_multimodel.sbatch).")
    else:
        print("\n  Lesart: das Vorzeichen der Spalte 'band [10,20)' ist die Aussage.")
        print("  Bleibt es ueber die Modelle hinweg negativ, ist die Danger Zone keine")
        print("  Eigenschaft von Qwen3-4B.")
        print("  Fuer 'Skalierung daempft den Effekt' zaehlt NICHT, ob sich die beiden")
        print("  Intervalle ueberschneiden, sondern nur das gepaarte Differenz-Intervall.")
        print()
        print("  Die beiden Kalibrierungsbloecke normieren auf die Fehlerrate des Retrievers")
        print("  im selben Band, und die kommt in der Delta-Formel nicht vor -- sie sind also")
        print("  keine Umformulierung der Spalte 'band [10,20)'. (Eine Regression des Deltas")
        print("  auf Rate und Praezision waere genau das: bis auf neutrale Aenderungen gilt")
        print("  Delta = Rate x (2 x Praezision - 1). Arithmetik, kein Befund.)")
        print()
        print("  Aggressivitaet ueber 1.00: das Modell greift oefter ein, als der Retriever")
        print("  ueberhaupt falsch liegt -- dann MUSS es richtige Top-1 ueberschreiben.")
        print("  Auswahlguete: haette es innerhalb des Bandes blind ueberschrieben, laege")
        print("  seine Praezision bei der Fehlerrate des Bandes, also bei 1.00. Werte um 1")
        print("  heissen, dass die Auswahl kaum Information traegt; unter 1 waehlt das")
        print("  Modell schlechter als der Zufall aus. Das ist die schaerfste Fassung der")
        print("  Aussage: nicht 'es greift zu oft ein', sondern 'es weiss nicht, welche")
        print("  Top-1 falsch sind'.")


# ------------------------------------------------------------------- seeds
def seeds_report(out):
    print("\n" + "#" * 96)
    print("# How much of the fine-tuning effect is seed noise?")
    print("#" * 96)
    zs_bc5 = load("preds_qwen3-4b-zeroshot_bc5cdr.jsonl")
    fts = [(s, load(f"preds_ft_bc5cdr_s{s}.jsonl")) for s in SEEDS]
    fts = [(s, r) for s, r in fts if r]
    if not fts or not zs_bc5:
        print("\n  Keine Seed-Laeufe gefunden. Erwartet preds_ft_bc5cdr_s1.jsonl usw.")
        print("  (bender_finetune_seeds.sbatch + bender_eval_ft_seeds.sbatch).")
        return

    # Der bisherige Einzellauf lief mit TRLs Default-Seed und ohne gesetztes
    # data_seed -- als Referenzzeile nuetzlich, aber nicht Teil der Spanne.
    old = load("preds_ft_bc5cdr.jsonl")
    group = [zs_bc5] + ([old] if old else []) + [r for _, r in fts]
    raw = [len(r) for r in group]
    group, _ = align(group)
    zs_bc5, group = group[0], group[1:]
    if old:
        old, group = group[0], group[1:]
    fts = [(s, rows) for (s, _), rows in zip(fts, group)]

    print("\nBC5CDR — das Band, das im Zero-shot netto schadet:")
    if len(set(raw)) > 1:
        print(f"  (auf gemeinsame Mentions beschnitten: {max(raw)} -> {len(zs_bc5)})")
    zb = band_row(zs_bc5, 3)
    print(f"  {'run':14}{'P4':>7}{'rate%':>8}{'band [10,20) prec':>20}{'delta':>9}")

    def line(lab, rows, b):
        print(f"  {lab:14}{acc(rows,'p4_correct'):7.1f}{acc(rows,'llm_changed'):8.1f}"
              f"{fmt_pct(b['precision'])}{b['delta']:+9.1f}")

    line("zero-shot", zs_bc5, zb)
    if old:
        ob = band_row(old, 3)
        if ob:
            line("ft (bisher)", old, ob)
    P, D, A4 = [], [], []
    for s, rows in fts:
        b = band_row(rows, 3)
        if not b:
            continue
        line(f"seed {s}", rows, b)
        A4.append(acc(rows, "p4_correct"))
        D.append(b["delta"])
        if b["precision"] is not None:
            P.append(b["precision"])

    def rng(v):
        return (f"{np.mean(v):.1f} (range {min(v):.1f}–{max(v):.1f})" if v else "-")
    print(f"\n  ueber {len(A4)} Seeds:  P4 {rng(A4)} | Bandpraezision {rng(P)} | Band-Delta {rng(D)}")
    zp = "-" if zb["precision"] is None else f"{zb['precision']:.1f}"
    print(f"  zero-shot zum Vergleich: P4 {acc(zs_bc5,'p4_correct'):.1f} | "
          f"Bandpraezision {zp} | Band-Delta {zb['delta']:+.1f}")
    out["seeds_bc5cdr"] = {"zero_shot": {"P4": acc(zs_bc5, "p4_correct"), "band": zb},
                           "per_seed": {s: {"P4": acc(r, "p4_correct"), "band": band_row(r, 3)}
                                        for s, r in fts}}

    # seen / unseen on the three held-out corpora
    tgp = ROOT / "train_gold_ids.txt"
    if not tgp.exists():
        print("\n  train_gold_ids.txt fehlt -- seen/unseen uebersprungen.")
        return
    tg = set(open(tgp).read().split())
    seen = lambda r: any(g in tg for g in str(r.get("gold_id", "")).split("|"))

    # Pool the three held-out corpora -- but align each corpus separately first.
    # The zero-shot NLM-Chem reference is a 1500-mention random sample while a
    # fresh run covers all 11729; pooling before aligning would compare the
    # fine-tuned model against a different set of mentions.
    labels = ["zero-shot"] + [f"seed {s}" for s in SEEDS]
    pools = {lab: [] for lab in labels}
    trimmed = []
    for corpus, ref, tag in CORPORA[1:]:
        group = [("zero-shot", load(ref))] + [
            (f"seed {s}", load(f"preds_ft_{tag}_s{s}.jsonl")) for s in SEEDS]
        group = [(lab, r) for lab, r in group if r]
        if len(group) < 2:
            continue
        raw = [len(r) for _, r in group]
        aligned, mism = align([r for _, r in group])
        if len(set(raw)) > 1:
            trimmed.append(f"{corpus} {max(raw)}->{len(aligned[0])}")
        if mism:
            trimmed.append(f"!! {corpus}: {mism} Zeilen mit abweichendem P3-Top1")
        for (lab, _), rows in zip(group, aligned):
            pools[lab].extend(rows)

    print("\nDrei gehaltene Korpora — Interventionspraezision, seen vs unseen:")
    if trimmed:
        print(f"  (auf gemeinsame Mentions beschnitten: {', '.join(trimmed)})")
    print(f"  {'run':14}{'seen':>26}{'unseen':>26}")
    S, U = [], []
    for lab in labels:
        rows = pools[lab]
        if not rows:
            continue
        cells = []
        for grp in (True, False):
            sub = [r for r in rows if seen(r) == grp]
            p, fx, bk = precision(sub)
            cells.append(fmt_cell(p, doc_ci(sub, lambda x: precision(x)[0]), fx, bk))
            if p is not None and lab != "zero-shot":
                (S if grp else U).append(p * 100)
        print(f"  {lab:14}{cells[0]:>26}{cells[1]:>26}")
    if S and U:
        print(f"\n  ueber {len(S)} Seeds:  seen {np.mean(S):.1f} (range {min(S):.1f}–{max(S):.1f})"
              f"   unseen {np.mean(U):.1f} (range {min(U):.1f}–{max(U):.1f})")
        print("  Die Aussage haelt, wenn die seen-Spanne die unseen-Spanne nicht beruehrt.")
        out["seeds_seen_unseen"] = {"seen": S, "unseen": U}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", action="store_true")
    ap.add_argument("--seeds", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if not (a.models or a.seeds):
        a.models = a.seeds = True
    out = {}
    if a.models:
        models_report(out)
    if a.seeds:
        seeds_report(out)
    if a.json:
        json.dump(out, open(ROOT / a.json, "w"), indent=2, default=lambda x: None)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
