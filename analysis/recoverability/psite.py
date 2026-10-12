"""Read-length-resolved P-site assignment for 40S footprints.

A single 5'-end offset assumes every footprint's 5' end sits a fixed distance
from the P site.  For scanning 40S that is not true: footprints run ~20-80 nt
and the start-codon metagene of raw 5' ends is a ~20 nt plateau, not a peak.
So the rule is measured, per footprint-length bin:

    for each length bin and each anchor (5' end, 3' end):
        metagene of anchor positions around annotated start codons,
        each gene scaled to sum 1
    pick the anchor with the sharper peak (peak / median over +/-75 nt),
    offset = distance from that peak to the A of the AUG

Bins with too few reads in the metagene fall back to the best rule over all
lengths.  The single-5'-offset rule is kept as a comparator, and the combined
metagene after shifting is scored the same way, so the choice is documented
by numbers rather than asserted.

Reads come from pipeline/02_align_count.py's <fraction>.reads.npz: sorted 5'
positions and aligned reference spans per (chrom, strand).
"""

import numpy as np

from data.coverage import CountTracks

UP, DOWN = 100, 100
NB = 75
LENGTH_BINS = [(18, 24), (25, 29), (30, 34), (35, 39), (40, 44), (45, 49),
               (50, 59), (60, 69), (70, 200)]
MIN_GENES = 200


def load_reads(path):
    with np.load(path) as z:
        keys = [k for k in z.files if k.endswith("|pos")]
        out = {}
        for k in keys:
            chrom, strand, _ = k.rsplit("|", 2)
            out[(chrom, strand)] = (z[k], z["{}|{}|span".format(chrom, strand)].astype(np.int64))
    return out


def anchor_positions(pos5, span, strand, anchor):
    if anchor == "5p":
        return pos5
    return pos5 + span - 1 if strand == "+" else pos5 - span + 1


def to_track(reads, select=None, anchor="5p", shift=None):
    """Reads -> CountTracks of anchor (or shifted P-site) positions.

    select(span) -> bool mask; shift(span) -> per-read offset towards the P
    site (downstream from a 5' anchor, upstream from a 3' anchor).
    """
    data = {}
    for (chrom, strand), (pos5, span) in reads.items():
        m = np.ones(len(pos5), bool) if select is None else select(span)
        if not m.any():
            continue
        a = anchor_positions(pos5[m], span[m], strand, anchor)
        if shift is not None:
            s = shift(span[m])
            sign = 1 if (anchor == "5p") == (strand == "+") else -1
            a = a + sign * s
        u, c = np.unique(a, return_counts=True)
        data[(chrom, strand)] = (u.astype(np.int64), c.astype(np.int32))
    return CountTracks(data)


def metagene(track, starts, min_counts=3):
    agg = np.zeros(UP + DOWN)
    used = 0
    for _tx, chrom, strand, cs, ce in starts:
        a, b = (cs - UP, cs + DOWN) if strand == "+" else (ce - DOWN, ce + UP)
        if a < 0:
            continue
        v = track.window(chrom, strand, a, b).astype(float)
        if strand == "-":
            v = v[::-1]
        s = v.sum()
        if s < min_counts:
            continue
        agg += v / s
        used += 1
    return agg / max(used, 1), used


POS = np.arange(-UP, DOWN)


def peak_stats(mg, search):
    win = (POS >= search[0]) & (POS <= search[1])
    peak = int(POS[win][np.argmax(mg[win])])
    nb = mg[(POS >= -NB) & (POS < NB)]
    return peak, float(mg[POS == peak][0] / max(np.median(nb), 1e-12))


SEARCH = {"5p": (-70, 5), "3p": (-5, 80)}


def measure(reads, starts):
    """-> (rule, report).  rule: [(lo, hi, anchor, offset)]."""
    report, rule = [], []
    best_all = None
    for anchor in ("5p", "3p"):
        mg, used = metagene(to_track(reads, anchor=anchor), starts)
        peak, sharp = peak_stats(mg, SEARCH[anchor])
        report.append({"bin": "all", "anchor": anchor, "peak": peak,
                       "sharpness": sharp, "genes": used, "metagene": mg})
        if best_all is None or sharp > best_all[2]:
            best_all = (anchor, -peak if anchor == "5p" else peak, sharp)
    for lo, hi in LENGTH_BINS:
        sel = lambda s, lo=lo, hi=hi: (s >= lo) & (s <= hi)
        cands = []
        for anchor in ("5p", "3p"):
            mg, used = metagene(to_track(reads, select=sel, anchor=anchor), starts)
            peak, sharp = peak_stats(mg, SEARCH[anchor])
            report.append({"bin": "{}-{}".format(lo, hi), "anchor": anchor, "peak": peak,
                           "sharpness": sharp, "genes": used, "metagene": mg})
            if used >= MIN_GENES:
                cands.append((sharp, anchor, -peak if anchor == "5p" else peak))
        if cands:
            sharp, anchor, off = max(cands)
            rule.append((lo, hi, anchor, int(off)))
        else:
            rule.append((lo, hi, best_all[0], int(best_all[1])))
    return rule, report, best_all


def apply_rule(reads, rule):
    """Reads -> CountTracks of P-site positions under a length-resolved rule.

    Reads whose span falls outside every bin are dropped (counted by caller).
    """
    merged = CountTracks()
    for lo, hi, anchor, off in rule:
        sel = lambda s, lo=lo, hi=hi: (s >= lo) & (s <= hi)
        t = to_track(reads, select=sel, anchor=anchor, shift=lambda s, o=off: np.full(len(s), o))
        merged.merge(t)
    return merged


def single_offset_rule(offset):
    return [(1, 65535, "5p", int(offset))]
