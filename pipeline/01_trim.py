"""Step 1: adapter + poly(A) trimming with cutadapt.

    python pipeline/01_trim.py [--fastq-dir DIR] [--max-reads N]

Read layout, read straight off the raw FASTQs (see DATA.md):

    <footprint> AAAAAAAAAA... TGGAATTCTCGGGTGCCAAGG AACTCCAGTCAC<index>ATCTCGTATG...
    |__insert__| |_poly(A)_| |__TruSeq Small RNA RA3__|

The library is poly(A)-tailed before the 3' adapter is ligated, so there are
two things to strip off the 3' end, in that order.  --times 2 lets cutadapt
make a second pass so the poly(A) exposed by removing the adapter also goes.

Trimming only ever touches the 3' end, which is what makes this safe for us:
the 5' end -- the position the whole model is built on -- is never moved.  The
cost is that genomic A's at a footprint's 3' end get eaten along with the
tail, so trimmed lengths are a slight underestimate.  That distorts footprint
length, not 5'-end position, so it does not propagate into the occupancy.
"""

import argparse
import os
import pathlib
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline._common import (WORK_DIR, cutadapt_cmd, dataset, ensure_dirs,
                              find_fastqs, require, run, up_to_date, work_path)


def _stamp_path(fastq_out):
    return fastq_out + ".maxreads"


def _stored_max_reads(fastq_out):
    """The --max-reads a trimmed file was built with, or None if unknown.

    Timestamps alone are not enough: a subsample trimmed yesterday is still
    newer than the raw FASTQ, so a later full run would "skip" it and train
    on 10 M reads while believing it had 51 M.  The stamp makes the
    subsample size part of the identity of the output.
    """
    try:
        return int(pathlib.Path(_stamp_path(fastq_out)).read_text().strip())
    except (OSError, ValueError):
        return None


def trim_one(cutadapt, fastq_in, fastq_out, max_reads=None, ds=None):
    want = int(max_reads) if max_reads else 0
    if up_to_date(fastq_out, fastq_in) and _stored_max_reads(fastq_out) == want:
        print("[trim] {} is up to date (max_reads {}), skipping".format(
            fastq_out, want or "all"))
        return fastq_out

    source = fastq_in
    if max_reads:
        # cutadapt has no read-limit flag, so cap the input first.  The full
        # SSU FASTQ is 23 GB, which is a slow way to find out the wiring works.
        source = fastq_out + ".head.fastq"
        print("[trim] subsampling first {} reads -> {}".format(max_reads, source))
        _head_fastq(fastq_in, source, max_reads)

    # cutadapt picks its output codec from the extension, so a .gz name is
    # all it takes to keep the intermediate compressed -- about a quarter of
    # the size.  02_align_count.py decompresses on the way into bowtie2.
    tmp_out = fastq_out + (".part.fastq.gz" if fastq_out.endswith(".gz")
                           else ".part")
    run(cutadapt_cmd(cutadapt, source, tmp_out, ds=ds))
    os.replace(tmp_out, fastq_out)
    pathlib.Path(_stamp_path(fastq_out)).write_text(str(want))
    if source != fastq_in:
        # The subsample is 4.5 GB of scratch per fraction and nothing reads it
        # again: the idempotency check keys on the raw FASTQ, not on this.
        os.remove(source)
    return fastq_out


def _head_fastq(src, dest, n_reads):
    """Copy the first n_reads records.  Plain text only; these dumps are plain."""
    with open(src, "r") as fin, open(dest, "w") as fout:
        for i, line in enumerate(fin):
            if i >= n_reads * 4:
                break
            fout.write(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fastq-dir")
    ap.add_argument("--max-reads", type=int, default=None,
                    help="only process the first N reads of each FASTQ")
    args = ap.parse_args()

    ensure_dirs()
    fastqs = find_fastqs(args.fastq_dir)

    # Resolve the binary lazily: a run whose outputs are all cached should
    # not need cutadapt on PATH at all.
    cutadapt = None
    want = int(args.max_reads) if args.max_reads else 0
    for frac, path in sorted(fastqs.items()):
        print("[trim] {} fraction: {}".format(frac, path))
        out = os.path.join(WORK_DIR, "{}.trimmed.fastq".format(frac))
        stale = not (up_to_date(out, path) and _stored_max_reads(out) == want)
        if stale and cutadapt is None:
            cutadapt = require("cutadapt", "  pip install cutadapt")
        trim_one(cutadapt, path, out, max_reads=args.max_reads)
        print("[trim] -> {}".format(out))


if __name__ == "__main__":
    main()
