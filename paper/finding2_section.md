# Finding 2 — Fine-tuning buys deference, not competence

## Setup

To ask what LoRA fine-tuning changes about the disambiguation stage, we compare the
zero-shot and fine-tuned Qwen3-4B and split every mention two ways at once: by the
retriever's confidence band (the top1−top2 gap, as in Finding 1) and — for the
generalisation question — by whether the gold concept appears in the fine-tuning
training set (BC5CDR training concepts). Within each cell we report intervention
precision (fixes / (fixes + breaks)) with document-level bootstrap 95% CIs. The
mechanism question ("*when* vs *how well*") is answered by the band breakdown; the
generalisation question ("does the learned behaviour transfer?") is answered by the
seen/unseen split evaluated *in-domain* (BC5CDR test) and *out-of-domain* (BioRED, a
corpus the model never trained on).

## The mechanism: deference, not competence

Read by confidence band, fine-tuning does not make the model a better disambiguator
where disambiguation is hard. On the truly-uncertain band (gap [0,2)), where the
answer must come from context, intervention precision is unchanged — 98% → 98% on
BC5CDR and 93% → 94% on BioRED. What fine-tuning changes is the model's behaviour in
the bands where the retriever is already mostly right. On BC5CDR the net-harmful
mid-confidence band from Finding 1 (gap [10,20)) flips from precision 18% to 64% and
from Δ −2.4 to +0.7, while the model's intervention *rate* in that band falls from
6.1% to 4.4%; on BioRED the same band moves from 39% to 69%. In short, fine-tuning
leaves competence on the hard cases flat and instead learns to stop overriding a
confident retriever — a confidence-dependent restraint, i.e. the gate of Finding 1,
learned internally. Consistently, across the whole intervention profile fine-tuning
raises the fixes share from 33% to 51% and halves the breaks share from 21% to 8%
(Finding 3), removing harmful overrides rather than adding new correct ones.

## The learned deference generalises in-domain but breaks down out-of-domain

Whether this learned restraint *transfers* depends on whether the concept was in the
fine-tuning distribution. Pooling three held-out corpora the model never trained on
(BioRED, NCBI-Disease and NLM-Chem, 4,024 mentions), fine-tuning raises intervention
precision on seen concepts from 69% to 97% (95% CIs [49,83] → [92,99]; the intervals do
not overlap, and breaks collapse from 42 to 5), while on unseen concepts precision is
essentially unchanged — 72% → 74% (CIs [62,81] → [65,83]; overlapping), and breaks in
fact rise from 77 to 88. The reliability fine-tuning buys is therefore realised almost
entirely on concepts it has seen. The confidence-band breakdown localises the effect:
on seen concepts the mid/high bands improve sharply (gap [5,10) +26, gap [10,20) +71
points), whereas on unseen concepts they barely move (seen-vs-unseen difference of +24
and +53 points respectively). The direction of the small unseen change is not itself
robust — it drifts down on the disease corpora and up on NLM-Chem — so we make the
conservative claim the data support: fine-tuning installs a confidence-dependent
restraint whose *precision benefit is bound to the training distribution*, present on
seen concepts and absent on unseen ones. This is the boundary a practitioner crosses
when applying a fine-tuned disambiguator to concepts unlike its training data.

## Honest status of the evidence and the confirmatory experiment

The band-level mechanism (competence flat, restraint learned) is established on four
MeSH datasets (BC5CDR, BioRED, NCBI, NLM-Chem) with non-overlapping confidence intervals
on BC5CDR, and is the robust core of the claim. The distribution-boundedness is
supported by the three pooled held-out corpora above: the seen-concept improvement is
significant, and the seen-versus-unseen difference is large and consistent in both the
aggregate and the band breakdown, across disease and chemical entities. Two honest limitations remain. The seen/unseen split
is confounded with concept frequency — memorised concepts are also the more frequent
ones — so part of the seen improvement may reflect frequency rather than exposure; a
clean isolation stratifies within frequency bands. And a single fine-tuning run underlies
these numbers, so the point estimates should be read with the seed-variance caveat that
applies throughout. Additional held-out corpora (e.g. NLM-Chem for chemicals) would add
entity-type diversity but are not required for the claim, which the pooled disease
corpora already support.

## Why this is the contribution

That fine-tuning trades cross-domain generalisation for in-domain performance is known
for entity-linking retrievers (Logeswaran et al., 2019) and entity matching. What is
specific here is the *mechanism* for the LLM disambiguation stage: fine-tuning does not
raise competence on the cases the retriever cannot solve; it installs a
confidence-dependent restraint — deference — that generalises within the training
domain and breaks down across domains. This one cause subsumes the confidence
stratification of Finding 1 and the intervention anatomy of Finding 3.
