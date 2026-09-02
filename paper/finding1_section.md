# Finding 1 — A mid-confidence "danger zone" in LLM disambiguation, and a confidence gate

## Setup

We ask *where*, along the retriever's own certainty, the LLM disambiguation stage
helps or hurts. For every mention we take the retriever's confidence as the score
gap between its top-1 and top-2 candidates after the rule stage (Phase 3); a small
gap means the retriever is unsure. Within each confidence band we report three
quantities: the **intervention rate** (fraction of mentions on which the LLM changes
the top-1 prediction), the **intervention precision** (fixes / (fixes + breaks),
i.e. how often a change is an improvement), and the **net effect** on Accuracy@1
(Δ = P4 − P3). Precision confidence intervals are 95% document-level bootstrap
(resampling PMIDs, since mentions within an abstract are correlated). All numbers in
this section use the zero-shot Qwen3-4B disambiguator; the datasets are BC5CDR and
BioRED (MeSH) and MedMentions (full UMLS).

## The mechanism: the intervention rate is well-calibrated, the precision is not

The LLM's intervention *rate* falls monotonically as the retriever grows more
confident — on BC5CDR from 54.3% of the most-uncertain mentions to 0.9% of the
most-certain ones, and likewise on every other dataset (Figure 1a, line). In this
sense the model behaves sensibly: it acts less when the retriever is sure. Its
*precision*, however, does not track its rate. It is high where the retriever is
unsure (BC5CDR 98% at gap [0,2)), but it **collapses in a mid-high band** rather
than at the extreme: at gap [10,20) the model still intervenes on 6.1% of mentions
and is right only 18% of the time (95% CI [10,30]; 20 fixes against 89 breaks),
turning a stage that is strongly positive elsewhere into a **net-harmful −2.4 points**
in that band. The retriever is already correct on 86.5% of those mentions, so most
interventions overturn a correct answer. The naïve reading — "the more confident the
retriever, the more harmful the LLM" — is wrong: the topmost band (gap 20+) is not
net-harmful, and its precision is too noisy to interpret (rate <1%, CI [14,84]). The
harm is localised to a *middle* band where the model is confident enough to act but
the retriever is already reliable.

## The danger zone replicates where the retriever is reliable, and vanishes where it is not

The net-harmful band reproduces on BioRED (gap [10,20): precision 39%, CI [13,76];
Δ −1.4), the second MeSH dataset. On MedMentions, linked against the full UMLS, the
same band shows a precision dip (89% → 53%) but **no net harm** (Δ +0.1), and with
SapBERT retrieval every band is net-positive (Figure 1a, right panel). This is exactly
what the mechanism predicts: an inversion only produces harm when the retriever's
mid-confidence answers are usually correct. On MedMentions the retriever is weak
(Recall@1 ≈ 48–55%), so even its moderately-confident top-1 is often wrong and the LLM
has room to help; on the two MeSH corpora the mid-confidence top-1 is mostly right, so
overriding it costs accuracy. The danger zone is therefore a property of *stronger*
retrievers, and it strengthens rather than weakens the frame: the LLM's value — and its
risk — are both set by the retriever.

## Relation to the Confidence Gate Theorem

This is precisely the failure mode characterised by the Confidence Gate Theorem
(Doku, 2026), which proves that confidence-based abstention improves a ranked system
monotonically iff the confidence signal contains no *inversion zones* (condition C2).
Our retriever-gap signal contains one, localised to the [10,20) band on the MeSH
corpora; the theorem's own empirical validation covers recommendation, e-commerce and
clinical triage, but not entity linking or an LLM intervener. We contribute the
entity-linking instantiation and, in Finding 2, a phenomenon outside the theorem's
static setup: fine-tuning the intervener reshapes its own inversion zone.

## The consequence: a confidence gate

Because the harm is confined to a predictable band, a simple policy recovers it: call
the LLM only when the retriever's gap is below a threshold, and otherwise keep the
retriever's answer. To avoid tuning on the reported set, we split documents into two
halves, choose the threshold that maximises accuracy on the first half, and report on
the second. On BC5CDR this gate sends only **19% of mentions to the LLM yet reaches
85.5% Accuracy@1 on held-out documents, above the 85.2% of running the LLM on
everything** (base retriever 84.4%) — a strict improvement at roughly one-fifth of the
LLM calls, because gating removes the net-harmful band. Figure 1b plots the full
trade-off as the fraction of gated mentions varies; the BC5CDR curve exceeds 100% of
the full-LLM gain. The gate's *efficiency*, like the danger zone itself, scales with
retriever strength: on BC5CDR and BioRED a gate at under 30% of calls retains ≥80% of
the benefit, whereas on weak-retriever MedMentions the LLM is broadly useful and must
be called on roughly 60% of mentions to match full accuracy. The actionable claim is
therefore two-sided: behind a strong retriever the LLM stage should be gated — it is
cheaper *and* more accurate; behind a weak retriever it should be run broadly.

## Robustness: two standalone public retrievers

To check that these effects are not artefacts of our hand-built pipeline, we repeat the
analysis on BC5CDR with candidates and confidence drawn purely from two public retrievers
that we did not build: a generic **SapBERT** bi-encoder (Recall@1 75.9%, the retriever
used by LLM4BioEL and Ye & Mitchell) and **BioSyn** (Sung et al., 2020; Recall@1 84.8%),
a BC5CDR-tuned dense+sparse system. In both cases the core value--confidence structure
replicates: the intervention rate falls monotonically with confidence, the LLM's value
concentrates in the low-confidence region, and precision is lowest in a mid band.

What differs between the two is exactly what the retriever-strength mechanism predicts.
On the *weaker* generic SapBERT retriever every confidence band is net-positive (the gate
merely matches the full LLM), because SapBERT's mid-confidence top-1 is often wrong and
the LLM still has room to help. On the *stronger* BioSyn retriever the full danger zone
reappears with a public signal: precision collapses to 34% and 23% in the [5,10) and
[10,20) bands, turning **net-harmful ($-2.5$ and $-1.7$ points)**, while the low-confidence
bands stay strongly positive ($+17.7$, $+7.2$). Consequently the confidence gate **beats
the full LLM** on BioSyn — a threshold tuned on held-out development documents reaches
87.0% Accuracy@1 at 15% of LLM calls, above the 85.4% of running the LLM everywhere. The
self-harm and the "gate beats full" effect are therefore *not* an artefact of our
rule-based score: they appear whenever the retriever is strong enough that its
mid-confidence answers are usually right, and they vanish when it is not. The danger zone
scales with retriever strength, which ties Finding 1 directly to the Frame and matches the
inversion-zone condition of the Confidence Gate Theorem.

## Limitations

The topmost-confidence band is estimated from very few interventions and its precision
is not reliably determined; we therefore make no claim about a "recovery" at gap 20+.
The gate threshold is selected on held-out development documents but from a single
random split; the reported operating point should be read together with the full
Pareto curve rather than as a tuned constant. All results use one open 4B model; the
danger zone should be confirmed with at least one further disambiguator (e.g. a
cross-encoder, or the LLM4BioEL system we do not build ourselves) before the mechanism
is claimed to be a property of LLM disambiguation rather than of our implementation.
