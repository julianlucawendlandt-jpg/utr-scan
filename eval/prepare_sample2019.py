"""Sample et al. 2019 MPRA -> a flat table eval/fit_head.py can fit.

    python eval/prepare_sample2019.py --in mpra/GSM4084997_varying_length_25to100.csv.gz

This is the human counterpart of eval/prepare_akirtava.py, and it is here for
the same reason: an MPRA measures what a leader does with EVERYTHING ELSE held
constant -- one promoter, one coding sequence, one terminator -- so the
between-leader variance is leader-driven by construction.  Native translational
efficiency is not a substitute.  We established that the hard way on yeast,
where 80S/Input turned out to be a reproducible measurement of whole-gene
efficiency in which leader features explain R^2 = 0.012.

WHAT IS IN THE FILE (GSM4084997, GEO series GSE114002):

    106,530 rows, 25-100 nt, with mean ribosome load already computed
      human        16,739   native human 5'UTRs    <- the set that matters
      random       87,000   designed random sequence
      with_uaugs      900   designed, uAUG-bearing
      no_uaugs        874   designed, uAUG-free
      controls      1,017

`rl` is mean ribosome load from polysome profiling: the average number of
ribosomes on that mRNA.  Physically the same quantity as Akirtava's TEff, read
off a different assay.

DOWNLOAD ONLY THE PROCESSED FILE.  GEO also lists the raw runs behind this
table -- 100+ GB across a dozen SRX accessions.  They reconstruct what this
15 MB CSV already contains.

    https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM4084nnn/GSM4084997/suppl/

TWO LIMITS WORTH KNOWING BEFORE READING ANY NUMBER OFF THIS.

Length.  The library is capped at 100 nt while native human leaders run to
~200 nt median, so the `human` set is native SEQUENCE but not a native length
distribution.  It will not test the long-leader regime, which is exactly where
the yeast model beat a context score most clearly.

Splits.  There are no genomic coordinates here, so a chromosome holdout is not
available out of the box and fit_head.py falls back to --split random.  That
is the weaker test: near-duplicate sequences can straddle it.  The stronger
version is recoverable by matching these leaders back to GENCODE MANE 5'UTRs
by sequence, which needs the genome on disk; until then the random split is
what the data supports and fit_head.py says so when it uses one.
"""

import argparse
import csv
import gzip
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample2019.tsv")

# Mean ribosome load is a ratio over polysome fractions, so a leader seen a
# handful of times has a very noisy one.  Sample et al. filter by depth for
# the same reason; this is the knob, not a hidden constant.
MIN_READS = 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True,
                    help="GSM4084997_varying_length_25to100.csv.gz")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--sets", default="human",
                    help="comma-separated sets to keep, or 'all'. Default "
                         "'human': the native leaders, the only set that is "
                         "real biology rather than designed sequence.")
    ap.add_argument("--min-reads", type=int, default=MIN_READS)
    args = ap.parse_args()

    if not os.path.exists(args.src):
        sys.exit("\nNot found: {}\nSee the module docstring for the URL.\n".format(args.src))
    keep = None if args.sets == "all" else {s.strip() for s in args.sets.split(",")}

    opener = gzip.open if args.src.endswith(".gz") else open
    kept, stats = [], {"rows": 0, "wrong_set": 0, "low_reads": 0, "bad": 0}
    with opener(args.src, "rt", newline="") as fh:
        for r in csv.DictReader(fh):
            stats["rows"] += 1
            st = (r.get("set") or "").strip()
            if keep is not None and st not in keep:
                stats["wrong_set"] += 1
                continue
            seq = (r.get("utr") or "").upper().replace("U", "T")
            try:
                rl = float(r["rl"])
                reads = float(r.get("total_reads") or r.get("total") or 0)
            except (KeyError, TypeError, ValueError):
                stats["bad"] += 1
                continue
            if rl != rl or not seq or set(seq) - set("ACGT"):
                stats["bad"] += 1
                continue
            if reads < args.min_reads:
                stats["low_reads"] += 1
                continue
            kept.append({"id": "{}:{}".format(st or "row", len(kept)),
                         "set": st, "utr_seq": seq, "rl": rl,
                         "length": len(seq), "total_reads": int(reads)})

    if not kept:
        sys.exit("\nNothing kept. Try --sets all or a lower --min-reads.\n")

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0]), delimiter="\t")
        w.writeheader()
        w.writerows(kept)

    import statistics as st_
    rls = sorted(k["rl"] for k in kept)
    lens = sorted(k["length"] for k in kept)
    print("[prep] read {rows} rows ({wrong_set} other sets, {low_reads} under "
          "{m} reads, {bad} unusable)".format(m=args.min_reads, **stats))
    print("[prep] kept {}".format(len(kept)))
    print("[prep] rl      median {:.2f}  p10 {:.2f}  p90 {:.2f}".format(
        rls[len(rls) // 2], rls[len(rls) // 10], rls[9 * len(rls) // 10]))
    print("[prep] length  median {}  p10 {}  p90 {}".format(
        lens[len(lens) // 2], lens[len(lens) // 10], lens[9 * len(lens) // 10]))
    print("[prep] -> {}".format(args.out))
    print("\n  fit a head with:\n"
          "    python eval/fit_head.py --checkpoint <chain.pt> \\\n"
          "        --table {} --target rl --seq-column utr_seq \\\n"
          "        --split random\n".format(os.path.relpath(args.out)))


if __name__ == "__main__":
    main()
