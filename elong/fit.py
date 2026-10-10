"""Estimate elongation rate per gene and emit a commitment-term offset.

    python elong/fit.py                      # -> cache/elong_offsets.tsv

The commitment term compares the chain's initiation rate against 80S/Input,
which is ribosome DENSITY.  At steady state

    density = initiation / elongation        so     initiation = density x elongation

This script estimates the elongation rate from the coding sequence and hands
train.py a per-gene factor that puts it back.

THE OFFSET, and its sign.  Write f_g for the part of log density that the CDS
explains.  Since log density = log initiation - log elongation, and the CDS
predicts the elongation half,

    f_g ~ const - log elongation_g      =>      1 / elongation_g ~ exp(f_g)

train.py multiplies the predicted Poisson rate by this factor:

    rate = r_init * input_total * elong_factor,    elong_factor = exp(f - mean f)

Centring on the mean keeps the factor at 1 on average, so rho_load's scale and
its `auto` anchor are untouched and this change is orthogonal to everything
already calibrated.  A fast-codon gene gets a factor below 1 -- for the same
initiation rate its ribosomes clear the window quickly, so fewer are seen.

The correction stays in the likelihood as a known constant per gene.  Nothing
new is fitted inside the training loop.

THREE TIERS, deliberately weak.  The estimator is fitted to the same
80S/Input it corrects, so capacity is the enemy: a model expressive enough to
explain all of 80S/Input would absorb the leader-driven initiation component
too, and the residual would come out as noise.  Hence OLS, few coefficients,
and no positional information about the CDS.

THE GUARDRAIL decides between them, not the fit quality.  A correct
correction removes a confound, so the corrected target should correlate with
LEADER properties -- Kozak score, uAUG count -- at least as strongly as raw
80S/Input did, and ideally more.  If a tier's correlation drops, it has eaten
leader-driven variance and is disqualified however well it fits.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from elong.extract_cds import OUT as CODON_NPZ, load_codon_counts
from elong.tai import CODONS, STOP_CODONS, codon_weights, gene_tai
from pipeline._common import CACHE_DIR

OFFSETS = os.path.join(CACHE_DIR, "elong_offsets.tsv")
MPRA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "eval",
                    "akirtava_scer_wt.tsv")

SENSE = [c for c in CODONS if c not in STOP_CODONS]


def rho(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def ols(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ b
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return pred, r2


def load_training_genes(cache):
    """Per gene: log density, uAUG count, and the raw totals."""
    from data.archer import load_examples
    rows = []
    for e in load_examples(cache):
        if "rs_cds_counts" not in e or "input_cds_counts" not in e:
            continue
        rs = float(np.sum(e["rs_cds_counts"]))
        ip = float(np.sum(e["input_cds_counts"]))
        if ip <= 0:
            continue
        seq = e["utr_seq"][:e["main_start_index"]].upper()
        rows.append({
            "transcript_id": e["transcript_id"],
            "orf": e["transcript_id"].replace("_mRNA", ""),
            "rs": rs, "ip": ip,
            # +0.5 on both: these are small counts and a zero numerator would
            # take the log to -inf.
            "logdens": float(np.log((rs + 0.5) / (ip + 0.5))),
            "uaug": float(sum(seq[i:i + 3] == "ATG" for i in range(max(0, len(seq) - 2)))),
        })
    return rows


def load_kozak():
    """{orf: LSMKozakStart} from the MPRA table, for the guardrail only."""
    p = os.path.normpath(MPRA)
    if not os.path.exists(p):
        return {}
    out = {}
    with open(p) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            v = r.get("LSMKozakStart")
            if r.get("orf") and v not in (None, ""):
                try:
                    out.setdefault(r["orf"], []).append(float(v))
                except ValueError:
                    pass
    return {k: float(np.median(v)) for k, v in out.items()}


def guardrail(rows, resid, kozak, label):
    """Do leader properties survive the correction?  Higher is better."""
    ua = np.array([r["uaug"] for r in rows])
    out = ["  %-28s uAUG rho %+.3f" % (label, rho(resid, ua))]
    idx = [i for i, r in enumerate(rows) if r["orf"] in kozak]
    if len(idx) >= 100:
        kz = np.array([kozak[rows[i]["orf"]] for i in idx])
        out.append("Kozak rho %+.3f (n=%d)" % (rho(resid[idx], kz), len(idx)))
    return "   ".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(CACHE_DIR, "archer_cds.npz"))
    ap.add_argument("--tier", type=int, default=0, choices=(0, 1, 2),
                    help="which fit to write offsets from. Default 0: on this "
                         "data tiers 1 and 2 fail the guardrail below")
    ap.add_argument("--out", default=OFFSETS)
    args = ap.parse_args()

    if not os.path.exists(CODON_NPZ):
        sys.exit("\nNo {} -- run elong/extract_cds.py first\n".format(CODON_NPZ))

    rows = load_training_genes(args.cache)
    codons = load_codon_counts(CODON_NPZ)
    w = codon_weights()
    kozak = load_kozak()

    keep = []
    for r in rows:
        c = codons.get(r["transcript_id"])
        if c is None:
            continue
        t = gene_tai(c[0], w)
        if t is None or t <= 0:
            continue
        n = sum(c[0].values())
        r["tai"], r["ncod"] = t, n
        r["freq"] = np.array([c[0].get(k, 0) for k in SENSE], dtype=float) / n
        keep.append(r)
    rows = keep

    y = np.array([r["logdens"] for r in rows])
    ltai = np.log(np.array([r["tai"] for r in rows]))
    llen = np.log(np.array([r["ncod"] for r in rows]))
    ones = np.ones(len(rows))
    print("[elong] {} genes with both a commitment readout and a CDS".format(len(rows)))
    print("[elong] log density vs log tAI  rho = %+.3f   vs log length  rho = %+.3f"
          % (rho(y, ltai), rho(y, llen)))

    fits = {}
    # Tier 0: no fitting at all.  Assumes elongation rate is proportional to
    # tAI, so f = -log tAI.  The proportionality constant washes out in the
    # centring; the EXPONENT is what tier 1 relaxes.
    fits[0] = (-ltai, float("nan"), "tAI only, no fit (0 coefficients)")
    for tier, X, desc in (
        (1, np.column_stack([ones, ltai, llen]), "log tAI + log length (3 coefficients)"),
        # 60 of 61 sense codons: frequencies sum to 1, so keeping all of them
        # alongside the intercept makes the design matrix singular.  Still a
        # bag of codons -- no order, no context -- which is what keeps it
        # unable to encode leader information.
        (2, np.column_stack([ones, np.array([r["freq"] for r in rows])[:, :-1], llen]),
         "60 codon frequencies + log length (62 coefficients)"),
    ):
        pred, r2 = ols(X, y)
        fits[tier] = (pred, r2, desc)

    print("\n" + "=" * 74)
    print("FIT: how much of log density does each tier explain?")
    print("=" * 74)
    for t in (0, 1, 2):
        f, r2, desc = fits[t]
        print("  tier %d  %-46s R2 = %s" % (t, desc, "  --  " if r2 != r2 else "%.4f" % r2))

    print("\n" + "=" * 74)
    print("GUARDRAIL: do leader properties survive?  (higher = better)")
    print("=" * 74)
    print(guardrail(rows, y, kozak, "raw log(80S/Input)"))
    for t in (0, 1, 2):
        f, _r2, _d = fits[t]
        print(guardrail(rows, y - f, kozak, "tier %d corrected" % t))
    print("\n  A DROP versus the raw row means that tier absorbed leader-driven")
    print("  variance and is disqualified, however well it fits.")

    f = fits[args.tier][0]
    factor = np.exp(f - f.mean())
    with open(args.out, "w", newline="") as fh:
        wri = csv.writer(fh, delimiter="\t")
        wri.writerow(["transcript_id", "elong_factor", "tai", "n_codons"])
        for r, v in zip(rows, factor):
            wri.writerow([r["transcript_id"], "%.6g" % v, "%.4f" % r["tai"], r["ncod"]])
    print("\n[elong] tier %d offsets -> %s  (%d genes, factor %.3f .. %.3f)"
          % (args.tier, args.out, len(rows), factor.min(), factor.max()))
    print("[elong] genes without an offset fall back to 1.0 in train.py")


if __name__ == "__main__":
    main()
