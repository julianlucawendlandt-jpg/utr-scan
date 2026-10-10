# elong/ — elongation correction for the commitment target

**This is a self-contained experiment. It can be deleted without touching the model.**

## Why it exists

`train.py`'s commitment term compares the chain's `P(commit_main)` against
`80S / Input` measured over a window of each gene's coding region. That ratio
is ribosome *density*, and at steady state

```
density  =  initiation rate / elongation rate
```

so it is not the initiation rate the chain models — it is initiation divided
by elongation speed. Elongation speed is a property of the **coding
sequence** (codon usage, tRNA supply), which the model cannot see, because it
only reads the 5'UTR.

The consequence is not noise, it is systematically wrong labels. Two genes
with an identical leader and identical true initiation, but different codon
usage, get different `80S/Input` — and the commitment term instructs the model
to predict a difference from inputs that are the same. Measured: `80S/Input`
correlates only +0.19 with leader-driven protein output in the Akirtava MPRA,
and *worse* at higher read depth, which rules out shot noise.

Rearranging:

```
initiation  =  (80S / Input)  x  elongation rate
```

So this directory estimates the elongation rate per gene and hands `train.py`
a per-gene multiplicative offset.

## Why the estimator is deliberately weak

The elongation model is fitted to the same `80S/Input` data it corrects. A
high-capacity model (a transformer over the CDS, say) would explain *all* of
`80S/Init` — including the leader-driven initiation component we are trying to
isolate — because CDS optimality and leader quality are correlated across the
genome (highly expressed genes have both). The residual would come out as
noise and we would conclude the leader explains nothing.

So: ordinary least squares, few coefficients, and **no positional information
about the CDS**. Three tiers, each a strict superset of the last:

| tier | predictors | coefficients |
|---|---|---|
| 0 | tAI alone, no fitting at all | 0 |
| 1 | `log tAI`, `log CDS length` | 3 |
| 2 | 60 codon frequencies + `log CDS length` | 62 |

Tier 2 is still a bag of codons — no order, no context — which is what keeps
it incapable of encoding leader information.

## The guardrail

Capacity arguments are not evidence. Before a correction is believed:

> Does the corrected target still correlate with **leader** properties (Kozak
> score, uAUG count) at least as strongly as raw `80S/Input` did?

If those correlations drop, the CDS model has eaten leader-driven variance and
that tier is disqualified however well it fits. `fit.py` reports this for
every tier.

## tAI, and why not CAI

Both score "how fast do this gene's codons decode". They differ in where the
per-codon weights come from.

* **CAI** derives weights from which codons *highly expressed genes* prefer.
  Expression is what we are trying to explain, so this smuggles the answer
  into the feature. Not used here.
* **tAI** derives weights from **tRNA gene copy numbers** in the genome. No
  expression data anywhere in its construction, and it is the more mechanistic
  of the two: it is a model of tRNA supply, which is what sets the speed.

The tRNA gene copy numbers are read out of `ref/sacCer3.gtf`, which annotates
tRNA genes with the anticodon in the gene id (`tR(ACG)D` = tRNA-Arg, anticodon
ACG). So nothing external is downloaded and the whole thing is reproducible
from the reference already in the repo.

## Running it

```
python elong/extract_cds.py          # full CDS -> codon counts   -> cache/cds_codons.npz
python elong/fit.py                  # tiers + guardrail          -> cache/elong_offsets.tsv
python train.py --data archer --cache cache/archer_cds.npz \
    --elong-offsets cache/elong_offsets.tsv ...
```

## How to remove it

1. Delete this directory.
2. In `train.py`: drop the `--elong-offsets` flag, the `load_elong_offsets`
   call, and the `elong_factor` argument to `commit_nll` (it defaults to 1.0,
   so the term is unchanged without it).

Nothing else in the pipeline or the model refers to any of this.
