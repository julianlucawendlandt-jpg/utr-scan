"""Effective spatial resolution: replicate agreement vs bin width.

    python -m analysis.recoverability.02_replicate_resolution [--datasets ...]

For every dataset with two independent replicates, leader counts (5' end of
the window through the main AUG codon) are summed into AUG-anchored bins of
1, 3, 5, 10, 20 and 50 nt -- the same anchoring as likelihood.py -- and the
two replicates are compared on transcripts with >= MIN_READS in BOTH.

Metrics (each with a transcript-bootstrap 95% CI), overall and per depth
stratum (the smaller of the two replicates' leader reads):

  pearson_mean      mean per-transcript Pearson r between the bin vectors
  spearman_mean     mean per-transcript Spearman rho
  pooled_centered_r Pearson r over all bins of all transcripts after turning
                    each transcript into proportions and subtracting its own
                    mean -- agreement on shape, not on depth
  pooled_posadj_r   as above, but also subtracting the dataset-wide mean
                    proportion at that (bin, leader-length-class): agreement
                    on transcript-SPECIFIC shape, beyond what position alone
                    gives every transcript.  Not in the original spec; added
                    because raw correlations are inflated by the shared AUG
                    peak and cap-to-AUG ramp.
  *_ceiling         the same metric for a parametric-bootstrap pair: both
                    replicates redrawn as pure multinomial samples from the
                    pooled profile at their real depths.  Observed below the
                    ceiling = extra between-library variance (overdispersion);
                    observed AT the ceiling = agreement limited by depth alone.
                    The pooled profile is itself noisy, so the ceiling is
                    conservative (slightly low) at shallow depth.

Transcripts with fewer than 3 bins at a width are skipped at that width
(a 2-point correlation is +/-1 by construction); the number used is reported.

Writes resolution/replicate_resolution.tsv.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability.bootstrap import mean_ci
from analysis.recoverability.likelihood import _ANCHOR, _BIG
from analysis.recoverability.panel import load_panel

MIN_READS = 20
LEN_CLASSES = (0, 50, 100, 200, 400, 10 ** 9)


def binned(v, msi, b):
    """Leader vector (through the AUG codon) -> AUG-anchored bin sums, cap->AUG."""
    v = np.asarray(v[:msi + 3], dtype=float)
    d = msi - np.arange(len(v))
    key = np.floor_divide(d + _ANCHOR + _BIG, b)
    _, inv = np.unique(-key, return_inverse=True)   # cap first
    return np.bincount(inv, weights=v)


def _rank(x):
    return np.argsort(np.argsort(x, kind="stable"), kind="stable").astype(float)


def corr(a, b):
    if a.std() == 0 or b.std() == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def pooled_r(xs, ys, n_boot, seed):
    """Pearson over concatenated per-transcript vectors, transcript bootstrap."""
    if not xs:
        return np.nan, np.nan, np.nan
    r = corr(np.concatenate(xs), np.concatenate(ys))
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(xs), len(xs))
        bs.append(corr(np.concatenate([xs[i] for i in idx]),
                       np.concatenate([ys[i] for i in idx])))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return r, float(lo), float(hi)


def ceiling_pairs(recs, keep, r1, r2, seed):
    """Multinomial redraws of both replicates from the pooled profile."""
    rng = np.random.default_rng(seed + 991)
    out = {}
    for i in keep:
        t = recs[i]
        a = np.asarray(t["counts"][r1][:t["msi"] + 3], float)
        b = np.asarray(t["counts"][r2][:t["msi"] + 3], float)
        p = (a + b) / (a + b).sum()
        out[i] = (rng.multinomial(int(a.sum()), p), rng.multinomial(int(b.sum()), p))
    return out


def agreement_rows(recs, keep, src, b, variant, strata, depth, tot, reps, key, seed, n_boot):
    """All agreement metrics at bin width b for one source of replicate pairs."""
    r1, r2 = reps
    per = []
    for i in keep:
        t = recs[i]
        a_, b_ = src(i)
        x, y = binned(a_, t["msi"], b), binned(b_, t["msi"], b)
        if len(x) < 3:
            continue
        lc = int(np.searchsorted(LEN_CLASSES, t["msi"], side="right"))
        per.append((i, x, y, lc))
    # Dataset-wide mean proportion per (bin index from the AUG, length
    # class), averaged over both replicates: the positional expectation.
    posmean = {}
    for _i, x, y, lc in per:
        px, py = x / x.sum(), y / y.sum()
        for k in range(len(x)):
            s = posmean.setdefault((len(x) - 1 - k, lc), [0.0, 0])
            s[0] += 0.5 * (px[k] + py[k])
            s[1] += 1
    rows = []
    for name, lo, hi in strata:
        sel = [p for p in per if depth[p[0]] >= lo and (hi is None or depth[p[0]] <= hi)]
        if len(sel) < 10:
            continue
        pr = np.array([corr(x, y) for _i, x, y, _ in sel])
        sr = np.array([corr(_rank(x), _rank(y)) for _i, x, y, _ in sel])
        cx, cy, ax, ay = [], [], [], []
        for _i, x, y, lc in sel:
            px, py = x / x.sum(), y / y.sum()
            cx.append(px - px.mean())
            cy.append(py - py.mean())
            pm = np.array([posmean[(len(x) - 1 - k, lc)][0] / posmean[(len(x) - 1 - k, lc)][1]
                           for k in range(len(x))])
            ax.append(px - pm)
            ay.append(py - pm)
        n_reads = int(sum(tot[r1][i] + tot[r2][i] for i, *_ in sel))
        base = {"dataset": key, "replicate": "{}_vs_{}".format(r1, r2), "split": "all",
                "seed": seed, "bin_width": b, "model": "", "n_reads": n_reads,
                "depth_stratum": name}
        for metric, vals in (("pearson_mean", pr), ("spearman_mean", sr)):
            m = mean_ci(vals, n_boot=n_boot, seed=seed)
            rows.append(dict(base, metric=metric + variant, value=m["mean"], ci_lo=m["lo"],
                             ci_hi=m["hi"], n_transcripts=m["n"]))
        for metric, (xs, ys) in (("pooled_centered_r", (cx, cy)),
                                 ("pooled_posadj_r", (ax, ay))):
            r, lo_, hi_ = pooled_r(xs, ys, min(n_boot, 200), seed)
            rows.append(dict(base, metric=metric + variant, value=r, ci_lo=lo_, ci_hi=hi_,
                             n_transcripts=len(sel)))
    return rows


def analyse(key, n_boot, seed):
    recs, meta = load_panel(key)
    reps = sorted(meta["replicates"])
    if len(reps) < 2:
        print("[res] {}: one replicate, skipped".format(key))
        return []
    r1, r2 = reps[:2]
    tot = {rep: np.array([t["counts"][rep][:t["msi"] + 3].sum() for t in recs]) for rep in (r1, r2)}
    keep = np.nonzero((tot[r1] >= MIN_READS) & (tot[r2] >= MIN_READS))[0]
    depth = np.minimum(tot[r1], tot[r2])
    print("[res] {}: {} transcripts with >= {} reads in both replicates".format(
        key, len(keep), MIN_READS))
    rows = []
    strata = [("all", MIN_READS, None)] + [
        ("{}-{}".format(lo, hi if hi else "inf"), lo, hi) for lo, hi in C.DEPTH_STRATA]
    ceil = ceiling_pairs(recs, keep, r1, r2, seed)
    for b in C.RESOLUTION_BINS:
        for variant in ("", "_ceiling"):
            src = (lambda i: ceil[i]) if variant else                 (lambda i: (recs[i]["counts"][r1], recs[i]["counts"][r2]))
            rows += agreement_rows(recs, keep, src, b, variant, strata, depth, tot,
                                   (r1, r2), key, seed, n_boot)
        allrow = [r for r in rows if r["bin_width"] == b and r["depth_stratum"] == "all"]
        print("  b={:>2}  ".format(b) + "  ".join(
            "{} {:.3f}".format(r["metric"], r["value"]) for r in allrow))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="gse174329_eif3b,gse139131_unsel")
    ap.add_argument("--n-boot", type=int, default=C.N_BOOT)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    C.ensure_layout()
    rows = []
    for key in [k for k in args.datasets.split(",") if k]:
        try:
            rows += analyse(key, args.n_boot, args.seed)
        except FileNotFoundError as e:
            print("[res] {}: {}".format(key, e))
    out = C.results_path("resolution", "replicate_resolution.tsv")
    done = {r["dataset"] for r in rows}
    C.merge_tsv(out, rows, lambda r: r["dataset"] not in done)
    print("[res] -> {}".format(out))


if __name__ == "__main__":
    main()
