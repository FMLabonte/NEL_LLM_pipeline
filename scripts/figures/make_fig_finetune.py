"""Figure for Results 6.4: what LoRA fine-tuning on BC5CDR changes, per corpus.
Report-type version of presentation/figs/defer_acc + defer_seen.
(a) Accuracy@1: retriever alone, zero-shot LLM, fine-tuned LLM (mean, range over seeds)
(b) change in intervention precision after fine-tuning, seen vs unseen concepts
    python3 paper/report/make_fig_finetune.py
"""
import sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import aclfig as A
from aclfig import load, acc, recall_at_1, precision, seed_files, train_concepts, is_seen

CORPORA = [("BC5CDR\n(in domain)", "preds_qwen3-4b-zeroshot_bc5cdr.jsonl", "bc5cdr"),
           ("BioRED", "preds_zs_biored.jsonl", "biored"),
           ("NCBI-Disease", "preds_zs_ncbi.jsonl", "ncbi"),
           ("NLM-Chem", "preds_zs_nlmchem.jsonl", "nlmchem")]
zs = {t: load(f) for _, f, t in CORPORA}
ft = {t: seed_files(t) for _, _, t in CORPORA}
nseed = min(len(v) for v in ft.values())
train = train_concepts()

r1 = [recall_at_1(zs[t]) for _, _, t in CORPORA]
p4z = [acc(zs[t], "p4_correct") for _, _, t in CORPORA]
fa = [[acc(rows, "p4_correct") for _, rows in ft[t]] for _, _, t in CORPORA]
p4f = [np.mean(v) for v in fa]; lo = [min(v) for v in fa]; hi = [max(v) for v in fa]

def prec(rows, seen):
    return precision([r for r in rows if is_seen(r, train) == seen])[0]
SEEN, UNSEEN, SHARE = [], [], []
for name, _, t in CORPORA:
    for store, flag in ((SEEN, True), (UNSEEN, False)):
        z = prec(zs[t], flag); f = [prec(rows, flag) for _, rows in ft[t]]
        store.append(np.mean(f) - z)
    SHARE.append(100 * np.mean([is_seen(r, train) for r in zs[t]]))
    i = len(SHARE) - 1
    print(f"{name.splitlines()[0]:13s} R@1 {r1[i]:.1f} zs {p4z[i]:.1f} ft {p4f[i]:.1f} "
          f"[{lo[i]:.1f},{hi[i]:.1f}]  dPrec seen {SEEN[i]:+.1f} unseen {UNSEEN[i]:+.1f}  "
          f"seen share {SHARE[i]:.0f}%")

A.use_acl()
LG, ZS, FT = "#c9ced3", "#8a95a0", "#0072B2"
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.69, 2.35), dpi=400, layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1]})
fig.get_layout_engine().set(w_pad=.02, h_pad=.02, wspace=.06)
x = np.arange(4); names = [c[0] for c in CORPORA]

A.style(a1, "y"); w = .26
a1.bar(x - w, r1, w * .92, color=LG, label="retriever alone")
a1.bar(x, p4z, w * .92, color=ZS, label="+ LLM, zero-shot")
a1.bar(x + w, p4f, w * .92, color=FT, label=f"+ LLM, fine-tuned ({nseed} seeds)")
a1.errorbar(x + w, p4f, yerr=[np.array(p4f) - lo, np.array(hi) - p4f], fmt="none",
            ecolor=A.INK, elinewidth=.7, capsize=1.8)
for i in range(4):
    a1.text(x[i] + w, hi[i] + .4, f"{p4f[i]:.1f}", ha="center", va="bottom", fontsize=6.2, color=FT)
    a1.text(x[i], p4z[i] + .4, f"{p4z[i]:.1f}", ha="center", va="bottom", fontsize=6.2, color=A.INK)
a1.set_ylim(60, 95); a1.set_yticks([60, 70, 80, 90])
a1.set_xticks(x); a1.set_xticklabels(names, fontsize=6.8)
a1.set_ylabel("Accuracy@1 (%)", labelpad=2)
a1.set_title("(a) accuracy per corpus", loc="left", fontsize=7.6)
a1.legend(frameon=False, loc="upper right", fontsize=6.3, handlelength=1.0,
          labelspacing=.25, borderaxespad=.1)

A.style(a2, "y"); w = .36
a2.axhline(0, color="#8f8f8f", lw=.6)
a2.bar(x - w / 2, SEEN, w * .92, color=FT, label="concept seen in fine-tuning data")
a2.bar(x + w / 2, UNSEEN, w * .92, color=LG, label="concept never seen")
for i in range(4):
    a2.text(x[i] - w / 2, SEEN[i] + 1, f"{SEEN[i]:+.0f}", ha="center", va="bottom", fontsize=6.4, color=FT)
    v = UNSEEN[i]
    a2.text(x[i] + w / 2, v + (1 if v >= 0 else -1), f"{v:+.0f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=6.4, color=A.NEG if v < 0 else A.INK)
a2.set_ylim(-22, 64); a2.set_yticks([-20, 0, 20, 40, 60])
a2.set_xticks(x); a2.set_xticklabels(names, fontsize=6.8)
a2.set_ylabel("change in precision (points)", labelpad=2)
a2.set_title("(b) intervention precision, after minus before", loc="left", fontsize=7.6)
a2.legend(frameon=False, loc="upper right", fontsize=6.3, handlelength=1.0,
          labelspacing=.25, borderaxespad=.1)

A.FIGDIR.mkdir(exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(A.FIGDIR / f"fig_finetune.{ext}", dpi=400)
print("wrote fig_finetune.pdf / .png")
