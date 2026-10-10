"""Metagene of 5'-end counts around annotated start codons.

    python scripts/metagene.py --dataset human-40s --restrict-to cache/human_utr_lengths.tsv

Two jobs, and the second is the more important one.

1. THE P-SITE OFFSET.  A footprint is reported by its 5' end, but the position
   that means something is where the ribosome's P site sits -- several
   nucleotides further in.  For yeast SSU that offset was measured at 12 nt,
   not assumed, and pipeline/02_align_count.py bakes it in with --offset.
   Footprint lengths differ between protocols and species, so the offset has
   to be re-measured for every dataset.  Counting at --offset 0 keeps the raw
   5' ends, and this script says what the shift should be.

2. WHETHER THE DATA LOOKS LIKE SCANNING AT ALL.  The entire model rests on 40S
   accumulating at start codons: that peak is what the dwell field fits and
   what makes the commit/drop split visible in the first place.  If the
   metagene is flat, or peaks somewhere else, that is a fact about the
   experiment and no amount of training will conjure the signal.  Better to
   see it here, on one plot, than to infer it later from fates that will not
   settle.

The window is in GENOMIC coordinates around the start codon, oriented
5' -> 3', so a minus-strand gene is flipped to read the same way as a plus-
strand one.  Position 0 is the A of the AUG.
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "pipeline"))

from pipeline._common import dataset, load_counts_npz, work_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--restrict-to", metavar="TSV",
                    help="transcript ids in column 0 to keep (the MANE table)")
    ap.add_argument("--up", type=int, default=100, help="nt upstream of the AUG")
    ap.add_argument("--down", type=int, default=100, help="nt downstream")
    ap.add_argument("--min-counts", type=int, default=1,
                    help="skip genes with fewer counts in the window; a gene "
                         "with two reads contributes noise, not shape")
    ap.add_argument("--normalise", action="store_true",
                    help="scale each gene to sum 1 before averaging, so a "
                         "handful of very deep genes cannot define the shape")
    ap.add_argument("--offset", type=int, default=0,
                    help="apply a 5'-end -> P-site shift before aggregating, "
                         "to confirm a measured offset lands the peak on 0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ds = dataset(args.dataset)
    import importlib
    m3 = importlib.import_module("03_build_examples")

    keep = None
    if args.restrict_to:
        with open(args.restrict_to) as fh:
            keep = {ln.split("\t")[0].strip() for ln in fh
                    if ln.strip() and not ln.startswith("#")}
        print("[meta] restricted to {} transcripts".format(len(keep)))

    counts = None
    for frac in ds["fractions"]:
        p = work_path(ds, frac, "5ends.npz")
        if not os.path.exists(p):
            continue
        c = load_counts_npz(p)
        print("[meta] {}: {} ends".format(frac, format(c.total(), ",")))
        counts = c if counts is None else counts.merge(c)
    if counts is None:
        sys.exit("\nNo counts found. Run pipeline/02_align_count.py first.\n")
    print("[meta] pooled {} ends".format(format(counts.total(), ",")))

    starts, _n, _cds = m3.parse_start_codons(ds["gtf"])
    if keep is not None:
        starts = [s for s in starts if s[0] in keep]
    print("[meta] {} start codons".format(len(starts)))

    width = args.up + args.down
    agg = np.zeros(width, dtype=np.float64)
    used = 0
    for tx, chrom, strand, cs, ce in starts:
        if strand == "+":
            a, b = cs - args.up, cs + args.down
        else:
            a, b = ce - args.down, ce + args.up
        if a < 0:
            continue
        sh = -args.offset if strand == "+" else args.offset
        v = counts.window(chrom, strand, a + sh, b + sh).astype(np.float64)
        if len(v) != width:
            continue
        if strand == "-":
            v = v[::-1]
        s = v.sum()
        if s < args.min_counts:
            continue
        agg += (v / s) if args.normalise else v
        used += 1

    print("[meta] {} genes contributed ({} skipped for low counts)".format(
        used, len(starts) - used))
    if used == 0:
        sys.exit("\nNo gene had counts in the window.\n")

    pos = np.arange(-args.up, args.down)
    peak = int(pos[int(np.argmax(agg))])
    print()
    print("[meta] PEAK at {:+d} nt relative to the A of the AUG".format(peak))
    print("[meta] => --offset {} would move it onto the start codon".format(-peak))

    # A compact text profile beats a plot we cannot look at.  Each row is one
    # nucleotide near the AUG; the bar is relative to the window maximum.
    lo, hi = max(0, args.up - 25), min(width, args.up + 20)
    top = agg.max()
    print("\n  pos    counts  (bar relative to the window peak)")
    for i in range(lo, hi):
        p_ = pos[i]
        bar = "#" * int(round(40 * agg[i] / top)) if top > 0 else ""
        mark = "  <- AUG" if p_ == 0 else ("  <- peak" if p_ == peak else "")
        print("  %+4d %9.0f  %s%s" % (p_, agg[i], bar, mark))

    out = args.out or os.path.join("work", "{}metagene.npy".format(ds["prefix"]))
    np.save(out, agg)
    print("\n[meta] profile -> {}".format(out))


if __name__ == "__main__":
    main()
