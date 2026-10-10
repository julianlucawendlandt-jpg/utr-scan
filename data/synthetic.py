"""Toy UTRs with planted features and occupancy drawn from the same chain.

The point is a zero-dependency path that exercises features -> rate net ->
chain -> loss end to end, and a later "can the model recover the planted
rates?" check.  Ground truth comes from GROUND_TRUTH_RATES below, which is a
hand-written function of the sequence, fed through the very same
model.chain.run_chain the model uses.  So the model is being asked to recover
a function that is genuinely inside its hypothesis class.

Planted signals, all readable from a +/-10 nt window of one-hot sequence:

  * strong-Kozak uORFs (A at -3, A at -1) commit much harder than weak ones
  * the 40S also pauses longer at those AUGs, so occupancy piles up there
  * GC-rich hairpin stems slow scanning (higher dwell, lower velocity) and
    raise the chance of dropping off

The structure tracks from ViennaRNA are *not* what carries the hairpin signal
here -- GC content in the window is -- so synthetic training behaves the same
whether or not ViennaRNA is installed.
"""

import numpy as np
import torch

from data.features import featurize
from model.chain import run_chain

# Hand-set ground truth.  A 40S that reaches a strong-Kozak AUG commits most of
# the time; one crossing a GC-rich stem often falls off.
# All of these are log-odds against dropping off, so they have to be large:
# a processive 40S falls off at ~1e-3 per nucleotide, and a strong AUG has to
# out-compete a step, not a coin flip.
BASE_STEP_LOGIT = 6.5        # ~0.0015 drop-off per nt on plain sequence
GC_STEP_PENALTY = 3.0        # a full-GC stem pushes that to ~0.02 per nt
GC_WINDOW = 5
BASE_COMMIT_LOGIT = 3.0      # weak-Kozak AUG: ~5% commit, mostly leaky
KOZAK_MINUS3_A = 3.0         # A at -3 is the dominant yeast Kozak position
KOZAK_MINUS1_A = 1.5         # strong context together: ~75% commit
DWELL_OVER_COMMIT = 2.0      # a 40S lingers longest at the AUGs it likes most:
                             # strong Kozak -> ~85% stay per step, a ~6x pile-up
BASE_DWELL_LOGIT = -2.0      # baseline velocity away from any start codon
GC_DWELL_BONUS = 2.0         # structured (GC-rich) stretches are scanned slowly
TRUE_RHO_LOAD = 2.0          # 80S per mRNA copy per unit P(commit_main); the
                             # value the model's rho_load should recover
CDS_LENGTH = 100

HAIRPIN_STEM = "GCGCGGCGCC"
HAIRPIN_LOOP = "TTTT"


def _rc(s):
    return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]


def ground_truth_rates(seq):
    """(w, p, d) as numpy arrays, straight from the sequence.  Pure function.

    Dwell tracks commit strength with a fixed offset, so the planted peak at a
    strong-Kozak AUG is the thing a "recover planted rates" check would look
    for.  The offset does not alter fates -- see model/chain.py -- only how
    tall the pile-up at each AUG is.
    """
    seq = seq.upper().replace("U", "T")
    n = len(seq)
    gc = np.array([1.0 if c in "GC" else 0.0 for c in seq], dtype=np.float64)

    # Mean GC over a +/-GC_WINDOW box, edge-padded.
    gc_frac = np.empty(n)
    for i in range(n):
        lo, hi = max(0, i - GC_WINDOW), min(n, i + GC_WINDOW + 1)
        gc_frac[i] = gc[lo:hi].mean()

    w = BASE_STEP_LOGIT - GC_STEP_PENALTY * gc_frac

    p = np.full(n, BASE_COMMIT_LOGIT)
    for i in range(n):
        if i >= 3 and seq[i - 3] == "A":
            p[i] += KOZAK_MINUS3_A
        if i >= 1 and seq[i - 1] == "A":
            p[i] += KOZAK_MINUS1_A
    # Dwell everywhere: a slow baseline that slows further in GC-rich
    # structure, overridden by a strong pause at good start codons.  This is
    # the velocity field the chain now models, so the planted truth lives
    # inside the model's hypothesis class.
    d = BASE_DWELL_LOGIT + GC_DWELL_BONUS * gc_frac
    at_start = np.zeros(n, dtype=bool)
    for i in range(n - 2):
        if seq[i:i + 3] == "ATG":
            at_start[i] = True
    d = np.where(at_start, p + DWELL_OVER_COMMIT, d)
    return w, p, d


def _random_seq(n, rng, gc=0.38):
    """Random sequence at roughly yeast 5'UTR base composition (AT-rich)."""
    probs = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(rng.choice(list("ACGT"), size=n, p=probs))


def make_utr(rng, length=120, n_uorfs=1, hairpin=True):
    """One toy UTR.  Returns (seq, main_start_index).

    The sequence ends with the main AUG, so the last three nucleotides are
    ATG and main_start_index == len(seq) - 3.  Keeping the AUG inside the
    array means the occupancy vector covers the position we care about most.
    """
    seq = list(_random_seq(length, rng))

    # Scrub any accidental ATGs so the only starts are the ones we plant.
    s = "".join(seq)
    for i in range(len(s) - 2):
        if s[i:i + 3] == "ATG":
            seq[i + 1] = "A"
        s = "".join(seq)

    if hairpin:
        stem = HAIRPIN_STEM + HAIRPIN_LOOP + _rc(HAIRPIN_STEM)
        pos = rng.integers(10, max(11, length - len(stem) - 30))
        seq[pos:pos + len(stem)] = list(stem)

    for _ in range(n_uorfs):
        strong = bool(rng.integers(0, 2))
        # -4..-1 context then ATG.  Strong = A at -3 and -1.
        context = "AAAA" if strong else "TCTC"
        pos = int(rng.integers(length // 2, length - 12))
        seq[pos:pos + 7] = list(context + "ATG")

    seq = "".join(seq[:length]) + "ATG"
    return seq, len(seq) - 3


def make_example(rng, length=120, total_counts=3000, near_cognate=False,
                 structure=False, transcript_id="synthetic", flank=10):
    """A toy UTR plus Poisson-noised occupancy from the ground-truth chain.

    `flank` nt of real (well, randomly generated) sequence is added either
    side so the rate net never sees zero-padding -- the same arrangement the
    Archer pipeline produces.  It must be at least the net's window.
    """
    seq, main_start_index = make_utr(rng, length=length)
    left = _random_seq(flank, rng)
    right = _random_seq(flank, rng)
    context = left + seq + right
    feats = featurize(seq, main_start_index, near_cognate=near_cognate,
                      structure=structure, context_seq=context,
                      context_offset=flank)

    # Ground truth is computed on the context and cropped the same way the
    # model's rates are, so the two see identical inputs at every position.
    w, p, d = (a[flank:flank + len(seq)] for a in ground_truth_rates(context))
    with torch.no_grad():
        profile, fates = run_chain(
            torch.tensor(w, dtype=torch.float32),
            torch.tensor(p, dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32),
            torch.tensor(feats["start_mask"]),
            torch.tensor(feats["is_main_start"]),
        )
    profile = profile.numpy()

    # One walker's profile is a shape; scale it to a sequencing depth, then
    # draw counts.  Input counts stand in for a flat abundance control.
    rate = profile / profile.sum() * total_counts
    ssu_counts = rng.poisson(rate).astype(np.int64)
    input_counts = rng.poisson(np.full(len(seq), total_counts / len(seq))).astype(np.int64)

    # Commitment readout.  Give the transcript an abundance, draw Input reads
    # over a CDS window at that abundance, and draw 80S reads at
    # TRUE_RHO_LOAD * P(commit_main) * abundance -- the relation the
    # commitment loss assumes, so recovery is a fair test.
    abundance = float(rng.lognormal(mean=np.log(0.5), sigma=0.6))
    input_cds_counts = rng.poisson(np.full(CDS_LENGTH, abundance)).astype(np.int64)
    p_commit_main = float(fates[0])
    rs_cds_counts = rng.poisson(
        np.full(CDS_LENGTH, TRUE_RHO_LOAD * p_commit_main * abundance)).astype(np.int64)

    return {
        "transcript_id": transcript_id,
        "utr_seq": seq,
        "main_start_index": main_start_index,
        "utr_context_seq": context,
        "context_offset": flank,
        "ssu_counts": ssu_counts,
        "input_counts": input_counts,
        "rs_cds_counts": rs_cds_counts,
        "input_cds_counts": input_cds_counts,
        "true_w": w,
        "true_p": p,
        "true_d": d,
        "true_fates": fates.numpy(),
    }


def make_dataset(n=64, seed=0, length=120, total_counts=3000, flank=10, **kwargs):
    rng = np.random.default_rng(seed)
    return [
        make_example(rng, length=length, total_counts=total_counts, flank=flank,
                     transcript_id=f"synthetic_{i:04d}", **kwargs)
        for i in range(n)
    ]
