# Project update — slide text

*NLP Lab · Moritz, Mohamed, Snehpreet · for Frederik, 27 Aug 2026*

Everything below is the state as of today. Three numbers are still moving; they are
marked and collected on slide 11.

---

## Slide 1 — Where we are

**Title:** From a failed reproduction to a measurement paper

- We could not reproduce BioLinkerAI's numbers, and we stopped trying.
- Instead we turned the pipeline into an instrument and asked a question nobody in this
  literature answers: **when, and how much, does the LLM disambiguation stage actually
  contribute?**
- **One frame and two findings** (the paper's three contributions), across five corpora,
  fourteen retriever configurations and three disambiguators.
- Draft paper is written (8 pages of content, ACL format). Today: what we did and why.

**What to say.** The last time we spoke the honest summary was "the reproduction did not
work". That is still true, and we now treat it as a starting point rather than a failure.
Every paper in this area reports that adding an LLM disambiguator improves accuracy, and
every paper reports it against one retriever on one or two corpora. We decided to measure
the thing they all assume.

---

## Slide 2 — Why we pivoted, in numbers

**Title:** The reproduction gap, stated plainly

- Our pipeline reaches **83.7** on BC5CDR against the **93.3** reported with GPT-4.
- On MedMentions **59.6** against **81.3**.
- No code, no prompts, closed-source disambiguator → we cannot attribute the gap.
- So we make **no cross-paper accuracy claims** anywhere in the paper.

**What to say.** We do not know where the gap comes from — candidate generation, the
knowledge-base snapshot, the disambiguator, or evaluation details. Because we cannot
attribute it, we stopped competing on absolute accuracy. What we can do is hold
everything fixed and vary one component at a time. A constant offset in absolute accuracy
does not affect a difference. That is the whole design idea: every result in the paper is
a **Δ within one controlled comparison**, never a number compared across papers.

---

## Slide 3 — The instrument

**Title:** What "controlled" means here

- Pipeline: candidate generation (P2) → rule-based re-ranking (P3) → LLM disambiguation (P4).
- **P3** = the retriever's top-1. **P4** = the pipeline's answer. **Δ = P4 − P3**.
- Fixed across every run: the prompt, temperature 0, the top-10 candidate list, and
  constrained decoding — the model can only *select* from the list, never invent an ID.
- Confidence signal: the retriever's **top1 − top2 score gap**. Score scales differ per
  retriever, so it is always binned *within* a run, never across runs.
- Every number is recomputed from per-mention prediction dumps. No model inference is
  needed to reproduce any table or figure.

**What to say.** This slide is the reason the rest holds together. When we compare two
retrievers, the mentions, the prompt and the decoder are identical, so the difference is
the retriever. When we compare zero-shot to fine-tuned, the candidate lists are identical
too. And because we dump per-mention records, we can re-slice everything afterwards
without re-running the models — which is what made the last two weeks of analysis
possible at all.

---

## Slide 4 — What we extended, and why

**Title:** Five corpora, fourteen retriever configurations

- **Corpora.** You knew BC5CDR and MedMentions. We added **BioRED**, **NCBI-Disease** and
  **NLM-Chem** — all MeSH; MedMentions stays as the UMLS case.
- **Retrievers.** Our rule pipeline, plus three we did not build: **BM25**, **SapBERT**,
  **BioSyn**.
- That gives a range of retriever strength from **R@1 49.6 to 84.8** on comparable tasks.

**Why each addition.**

- *BioRED* — MeSH, but different annotation conventions and multiple entity types.
- *NCBI-Disease* — diseases only, dominated by short abbreviations. It is the one corpus
  where **dense retrieval is worse than lexical** (R@1 60.7 vs 71.3), because the surface
  form is highly diagnostic and SapBERT's embedding space blurs it. Useful precisely
  because it breaks the "dense = stronger" assumption.
- *NLM-Chem* — full-text articles, chemicals, fully held out from fine-tuning.
- *MedMentions* — kept deliberately as the **weak-retriever** case. It turns out to be the
  control that confirms the mechanism (slide 7).
- *SapBERT* — the retriever used by the two papers we compare against, so our numbers sit
  in the same frame as theirs.
- *BM25 and BioSyn* — external candidate lists and external scores, so the effects cannot
  be artefacts of our own scoring.

**What to say.** The extension was not "more datasets for the sake of it". Each one buys a
specific contrast: a different knowledge base, a different entity type, a case where
lexical beats dense, and two retrievers we had no hand in building.

---

## Slide 5 — The frame: the retriever sets the LLM's value

**Title:** "+16 points from an LLM" is a statement about a *pair*, not about a model

- Same model, same prompt, same candidate-list size. The stage's contribution ranges from
  **+0.6** (BC5CDR / BioSyn) to **+10.1** (NLM-Chem / lexical) — a **seventeen-fold**
  spread, driven entirely by what sits in front of it.
- The LLM is not *worse* behind a strong retriever. Behind BioSyn the pipeline reaches the
  **highest final accuracy of all runs (85.4)**. There is simply less left to repair.
- **Pooled over all 14 runs the relation is not significant** (ρ = −0.50, *p* = 0.069) —
  corpus difficulty confounds it. MedMentions has a weak retriever *and* is intrinsically
  hard; BC5CDR has a strong retriever *and* is easy. The two move together.
- **Within** a corpus — same mentions, same knowledge base — the retriever is isolated and
  the relation is clean: **12 of 14** ordered pairs put the smaller gain behind the
  stronger retriever (sign test *p* = 0.013); ρ = −0.72 (*p* = 0.004), slope **−0.32**
  gain points per point of R@1.
- Practical consequence — the two stages are **partial substitutes**. On MedMentions,
  lexical → SapBERT buys **+7.3** points of retrieval but only **+2.7** of final accuracy.
  On BC5CDR, SapBERT → BioSyn buys **+8.9** and **+4.9**.

**What to say.** Think of the stage as a proofreader. Hand it a manuscript full of typos
and it fixes a lot; hand it a clean one and it fixes little — and starts "improving" things
that were already right. Its skill never changed; the amount of available error did. That
is why we call this the *frame* rather than a finding: it is the lens that makes the two
findings interpretable.

Two things here are ours rather than known. Ye and Mitchell do observe this dependence,
but qualitatively and over two retrievers; we measure it over fourteen. More importantly,
we show **where it is identifiable at all** — the pooled correlation fails, and we say so
in the paper rather than quietly reporting the within-corpus number. Corpus difficulty is
a confounder that pushes retriever strength and LLM headroom in the same direction, so the
across-corpora comparison cannot carry the claim. Only the within-corpus comparison can.

The substitution number is the part a practitioner should take away: investing in the
retriever returns less final accuracy than it returns retrieval accuracy, because the LLM
stage had been masking the retriever's weakness all along.

---

## Slide 6 — The frame, made causal

**Title:** We stopped inferring the relation and manipulated it

**The design.** Everything on the previous slide is observational: swapping retrievers
moves the candidate pool, the score scale and top-1 accuracy all at once. So we built a
setting where exactly one of them moves.

- Take BioSyn's BC5CDR candidate lists — ten scored concepts per mention, sorted.
- For a fraction *p* of mentions, swap the concept at **rank 1** with one at a uniformly
  drawn rank in [2,10]. **The score vector stays where it is; only the concept labels
  move.**
- Therefore: the list is still sorted, the top1−top2 gap distribution is **unchanged** (the
  LLM sees an identical confidence signal), and the gold concept is still in the list —
  **R@10 is 90.5 at every level, by construction.**
- The only thing that changes is *which concept sits on top*, i.e. R@1.
- The levels are **nested** (what is degraded at *p* = 0.2 is also degraded at *p* = 0.3),
  so the five points are directly comparable and differences between them are not sampling
  noise.

**The result.**

- Retriever R@1 falls **84.8 → 42.4** — 42 points. Final accuracy falls **4.8**.
- The stage absorbs **89 %** of what the retriever gives up, almost perfectly linearly:
  slope −0.89 points per point, **r = −1.0000**.
- On the confident 80 % of mentions the effect runs from **+40.6** at *p* = 0.5 to
  **−1.6** at *p* = 0 and **crosses zero at R@1 ≈ 83** (slope −0.99, r = −0.9999).

**What to say.** This is the experiment I would defend hardest. It converts a correlation
across fourteen heterogeneous runs into a dose–response curve with one manipulated
variable, on one fixed mention set, with the confidence signal held constant by
construction.

The number to remember is the crossing point. Above a retriever quality of roughly 83
R@1, the stage stops paying on the confident majority of mentions — and that is exactly
where current retrievers sit. It also tells us the danger zone on the next slide is not a
quirk of one retriever: it is a function of retriever accuracy itself.

One caveat we state in the limitations: this design varies **ranking only**. A genuinely
weaker retriever also retrieves the gold concept less often (BM25 reaches R@10 79.1
against BioSyn's 90.5). So −0.89 is an upper bound on what swapping real retrievers buys,
and the observational −0.32 is what you should expect when recall and ranking move
together.

**And what bounds the stage from above** (backup — use if asked how much headroom there
even is):

- The model can only *select* from the candidate list, so **R@10 is a hard ceiling**. On
  the BC5CDR rule pipeline R@10 = 91.6, so no disambiguator could ever exceed 91.6 there.
- On that run, **809 of the retriever's 1,667 errors have no gold concept in the list at
  all** — structurally unfixable by any disambiguator.
- Of the errors it *could* fix, the stage recovers **24 %** (MedMentions) to **66 %**
  (BC5CDR/BioSyn), 30 % on the rule pipeline.
- Decomposing every change it makes on BC5CDR (771 of 9,661 mentions): **33 % fixes, 21 %
  breaks, 46 % lateral** — but only 15 points of those 46 are genuine confusion; the rest
  are lateral by construction because the gold was never retrieved.
- Summary: **over-eager where there is nothing to gain, under-powered where it matters.**

---

## Slide 7 — Contribution 2: a mid-confidence danger zone

**Title:** The intervention rate is calibrated. The precision is not.

- The model's intervention **rate** behaves sensibly: on BC5CDR it falls monotonically
  from **54.3 %** of the most-uncertain mentions to **0.9 %** of the most-certain ones.
- Its **precision** does not follow. Near-perfect where the retriever is unsure (98 % at
  gap [0,2)) — but in the band **gap [10,20)** it still intervenes on 6.1 % of mentions
  and is right only **18 %** of the time (CI [10,30]; 20 fixes against 89 breaks).
- Net effect in that band: **−2.4 points**. The retriever was already correct on **86.5 %**
  of those mentions.
- The naive reading is wrong: the **topmost** band (gap 20+) is *not* harmful (+0.2). The
  harm sits in a **middle** band — confident enough to act, retriever already reliable.
- External check: with **BioSyn's own** candidates and scores the zone reappears (−2.5 and
  −1.7 in bands [5,10) and [10,20)). With the weaker **SapBERT**, every band is positive.
- **MedMentions has no harmful band under any protocol** — exactly what the mechanism
  predicts, because there the retriever is weak enough that overriding it usually pays.

**What to say.** The mechanism claim is the point, not the number. The model modulates
*how often* it acts by retriever confidence, and it does that well. What it cannot do is
tell *which* confident top-1 predictions are wrong. And MedMentions is the control: the
effect is absent exactly where the theory says it should be absent.

---

## Slide 8 — "Isn't that just your prompt?"

**Title:** No — and the ablation says something more interesting

- The prompt we used all along **already** says *"the top-ranked candidate is correct in
  roughly 80 % of cases; override it only when the context gives you clear evidence"* —
  and it **already** shows every candidate's retriever score. The zone appears anyway.
- Three single-factor variants on BC5CDR + BioRED:
  - **Remove the scores** → clearly worse. Rate 8.0 → 11.7 % and 13.0 → 18.2 %, precision
    61 → 56 and 75 → 63, band −2.4 → −3.2 and −1.4 → −6.0. BioRED loses 1.1 points
    (p = 0.041).
  - **Remove the verbal instruction** → almost nothing. +0.24 (p = 0.054) and −0.42
    (p = 0.21).
  - **State the margin as a number** with a rule for using it → moves the right way (rate
    down to 6.8 / 11.0 %, precision up to 65 / 80) and **halves** the band — but does not
    remove it.

**What to say.** The reading is narrow and I think it is the interesting one: the model
responds to a **quantitative** confidence signal and barely at all to a **verbal** one.
Even when we hand it the margin as a number *and* tell it to defer on large margins, it
still overrides a mostly-correct retriever often enough to lose accuracy. This is not a
prompt-engineering oversight.

---

## Slide 9 — The practical consequence, stated two-sided

**Title:** A confidence gate — and where it fails

- Policy: call the LLM only when the retriever's gap is below a threshold. Threshold tuned
  on half the **documents**, evaluated on the other half, 20 random splits.
- Where a strong retriever created a zone, it **wins**: BC5CDR/BioSyn **+1.21 ± 0.27** at
  **18 %** of the calls; BC5CDR/pipeline **+0.55 ± 0.27** at 14 %.
- Where there is no zone it is **neutral** and gates almost nothing: SapBERT +0.01 at 99 %
  of calls, MedMentions −0.03 at 92 %.
- On **BioRED it is worse** than the full LLM: **−1.11 ± 0.92**.

**What to say.** We deliberately do not sell this as "less is more". BioRED has a harmful
band, but it is small compared to what the stage gains elsewhere, and one scalar threshold
cannot excise it without discarding those gains too. The honest claim is: gating pays
exactly where a strong retriever has created an inversion zone, and it has to be validated
per deployment rather than assumed. Reporting the case where our own proposal loses is, I
think, what makes the rest credible.

---

## Slide 10 — Contribution 3: fine-tuning buys deference, not competence

**Title:** LoRA does not make it a better disambiguator. It makes it stop.

- Fine-tuning (LoRA/QLoRA, SFT on the gold answer, BC5CDR train split) helps everywhere:
  BC5CDR 83.7 → 86.4, BioRED 83.6 → 85.4, NCBI 72.5 → 76.6, NLM-Chem 78.0 → 81.4.
- But **where** it helps is the point. On the genuinely uncertain band [0,2), where the
  answer must come from context, intervention precision is **unchanged** (96 % → 97 %).
- What changes is the harmful band: BC5CDR [10,20) goes from precision **18 % → 64 %** and
  Δ **−2.4 → +0.7**, with the intervention rate there falling 6.1 % → 4.4 %.
- It learned the gate internally — and consistently, the external gate no longer helps the
  fine-tuned model (−0.14 ± 0.17).
- **The restraint is bound to the training distribution.** Pooling the three held-out
  corpora (4,024 mentions), precision on **seen** concepts rises **68.7 % → 96.2 %** (CIs
  non-overlapping, breaks collapse 42 → 5); on **unseen** concepts it is essentially flat,
  **72.2 % → 74.3 %** (CIs overlapping, breaks in fact rise 77 → 88).

**What to say.** This connects directly to your question about SFT versus GRPO. Your
intuition — SFT memorises, RL would teach a more generalisable skill — is exactly what
this finding measures, in this domain: +28 points of precision where the concept was in
the training data, and nothing where it was not. That is memorisation, quantified. The
reference behind that intuition is Chu et al., *SFT Memorizes, RL Generalizes* (ICML 2025)
— we will cite it. We would like to discuss whether GRPO belongs in this paper or in the
next one; our current view is on slide 12.

---

## Slide 11 — What is running right now

**Title:** Three open jobs, and exactly what each one can change

**Done since we last spoke — two more disambiguators.** Everything above rested on a
single 4B model, which is the objection that kills an analysis paper. We re-ran the core
measurement with **Qwen3-8B** (same family, double the size) and **IBM Granite-3.1-8B**
(different family, same size), on four corpora, with byte-identical flags.

- The harmful band stays negative. Across **12 cells** (4 corpora × 3 models):
  **5 significantly negative, 7 not distinguishable from zero, 0 significantly positive.**
- **Granite shows it more strongly than our 4B**: NCBI −9.24 (95 % CI [−14.08, −4.70]);
  BC5CDR −4.13 ([−5.81, −2.59]).
- Size is not the variable. Qwen3-8B *attenuates* the effect significantly (BC5CDR +1.18
  [+0.23, +2.20]); Granite, at the same size, is worse than the 4B everywhere.
- Striking single result: **on BC5CDR, Granite's LLM stage makes the pipeline worse
  overall — 81.3 against a retriever at 82.7.** The first real model on a real retriever
  where the stage is a net loss. *(Point estimate; the confidence interval is computed but
  we have not re-run it since adding it — that is a 2-second job.)*

**Running now.**

1. **Three fine-tuning seeds** (LoRA seeds 1–3, then evaluation). Affects **Finding 3
   only**: the numbers on slide 10 become mean and range instead of point estimates. We
   had to patch our training script first — it had no `--seed` argument at all, so three
   array tasks would have trained three identical models.
2. **NLM-Chem on the full corpus.** Our NLM-Chem baseline covers 1,500 of 11,729 mentions
   (it was capped for runtime); the two new models cover all of them. Affects **one row**:
   whether NLM-Chem lands in "the zone replicates" or "not detectable". With 1,500 mentions
   the band holds ~300 and the interval is too wide to decide.

**What to say.** Neither open job can overturn anything on slides 5–9. The seeds change how
precisely we state Finding 3; NLM-Chem changes one line of a robustness table.

---

## Slide 12 — What we checked on ourselves

**Title:** The things a reviewer would have found

- **De-duplication control.** Scoring every mention occurrence over-weights frequent
  strings. Recomputed under three protocols: the zone **survives on BC5CDR and
  NCBI-Disease** (on BioSyn it even deepens, −2.5 → −5.0) and **disappears on BioRED and
  NLM-Chem**, whose bands hold only 205 and 174 unique pairs. We now report those two as
  protocol-dependent rather than as replications.
- **Not a surface-form artefact.** The score gap correlates with mention–label string
  similarity only moderately (ρ = 0.38–0.50), and the band structure survives stratifying
  on it.
- **Multiplicity.** Band breakdowns are labelled descriptive; exactly three analyses are
  confirmatory and carry the claims.
- **A calibration analysis that is not circular.** Regressing the band delta on
  intervention rate and precision would be arithmetic, not evidence (Δ ≈ rate × (2·prec −
  1)). Instead we normalise against the retriever's **error rate** in the same band, which
  does not appear in that identity: had the model overridden blindly inside a band, its
  precision would equal that error rate. In the harmful band we measure roughly **1.0** —
  and for Granite **0.82**, i.e. *worse than chance*.

**What to say.** That last point sharpened the claim. It is not "the model intervenes too
often" — the rate is well calibrated. It is "**the model cannot tell which confident top-1
predictions are wrong**". The first would be fixable with a threshold. The second is not,
which is exactly why the gate is the right answer and better prompting is not.

---

## Slide 13 — Where we think this goes

**Title:** Target, and the open question for you

- Target: **BioNLP workshop paper** (the 2026 deadline has passed; realistically BioNLP
  2027, which gives us time to do this properly).
- Draft is complete at 8 pages of content with figures, tables and three appendices.
- Remaining: fold in the seeds and NLM-Chem, rewrite the limitations paragraph that says
  "one model, one fine-tuning run" — it is now false — and build the anonymous artefact
  release.
- **On GRPO.** We think it is the next paper, not a section of this one. Three reasons:
  (a) it changes the genre from analysis to method, and 8 pages cannot carry both;
  (b) rollouts make it 8–16× the generation cost of SFT, which on our cluster quota is
  weeks; (c) the obvious reward — exact match on the gold ID — sets *the same incentive as
  SFT*. The reward that would attack our actual finding is different: **+1 for repairing a
  wrong top-1, −1 for breaking a correct one, 0 for not intervening.** That directly
  optimises intervention precision, which is what the whole paper is about.
- We found no RL-based work on biomedical entity linking at all, so the niche looks open.

**Ask:** does the scope split make sense to you, and is there a fourth thing you would
want measured before we submit?
