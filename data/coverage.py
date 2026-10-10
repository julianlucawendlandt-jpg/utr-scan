"""Footprint 5'-end assignment and projection onto 5'UTRs.

Pure functions -- no subprocesses, no file formats beyond parsing SAM text.
This is the load-bearing step of the pipeline, so it is deliberately explicit
rather than delegated to a generic coverage tool: a coverage track smears each
read across its whole length, which is exactly what we must not do.  A TCP-seq
footprint tells us where one 40S *was*, and the 5' end is the anchor for that.

Coordinates here are 0-based half-open throughout.  SAM is 1-based, and the
conversion happens once, in parse_sam_line.
"""

import re
import array
from collections import defaultdict

import numpy as np

CIGAR_RE = re.compile(r"(\d+)([MIDNSHP=X])")

# CIGAR ops that advance along the reference.
REF_CONSUMING = frozenset("MDN=X")

FLAG_UNMAPPED = 0x4
FLAG_REVERSE = 0x10
FLAG_SECONDARY = 0x100
FLAG_SUPPLEMENTARY = 0x800


def cigar_reference_span(cigar):
    """How many reference bases the alignment covers."""
    if cigar == "*":
        return 0
    return sum(int(n) for n, op in CIGAR_RE.findall(cigar) if op in REF_CONSUMING)


def five_prime_position(flag, pos1, cigar, offset=0):
    """The footprint's assigned position, strand-aware.  Returns (pos0, strand).

    The 5'-most genomic position of the read:
      + strand -- the leftmost aligned base, which is POS itself
      - strand -- the rightmost aligned base, POS + reference span - 1

    Soft-clipped bases are NOT added back; POS already excludes them, and the
    clipped sequence is adapter or untrimmed poly(A) rather than footprint.

    `offset` shifts the assignment DOWNSTREAM, in the direction the ribosome
    scans (3' along the read's own strand), so the sign flips on the minus
    strand.  Default 0 = the raw 5' end.

    To move from the footprint 5' end to the SSU P site, pass offset=12: the
    P site sits ~12 nt inside the footprint.  (The same shift is often written
    as "-12" when the convention is subtracting from the P site to recover the
    5' end -- same operation, opposite bookkeeping.)  Changing the assignment
    is this one parameter and nothing else.
    """
    if flag & FLAG_REVERSE:
        pos0 = (pos1 - 1) + cigar_reference_span(cigar) - 1
        return pos0 - offset, "-"
    return (pos1 - 1) + offset, "+"


def parse_sam_line(line, min_mapq=10):
    """One SAM alignment -> (rname, flag, pos1, cigar), or None if filtered out.

    Dropped: headers, unmapped reads, secondary and supplementary alignments,
    and anything below min_mapq.  For bowtie2, MAPQ >= 10 is the conventional
    uniquely-mapped threshold -- a read with a near-tied second-best alignment
    scores below it.  Multireads are discarded, not distributed: assigning a
    fraction of a read to several positions would invent occupancy that no
    single 40S ever had.
    """
    if not line or line.startswith("@"):
        return None
    f = line.rstrip("\n").split("\t")
    if len(f) < 11:
        return None
    flag = int(f[1])
    if flag & (FLAG_UNMAPPED | FLAG_SECONDARY | FLAG_SUPPLEMENTARY):
        return None
    if int(f[4]) < min_mapq:
        return None
    rname = f[2]
    if rname == "*":
        return None
    return rname, flag, int(f[3]), f[5]


class CountTracks(object):
    """Per-(chrom, strand) 5'-end counts, stored sparsely.

    The obvious representation is one dense int32 array per chromosome per
    strand.  That is what this used to do, and it is fine for yeast -- 12 Mb
    times two strands times four bytes is 96 MB.  It does not survive a
    mammalian genome: GRCh38 is 3.1 Gb, so the same scheme asks for ~25 GB of
    RAM before a single read is counted, and it dies on the allocation.

    Footprint 5' ends touch a small fraction of a large genome, so the sparse
    form is both far smaller and no slower to use: positions are kept sorted
    per key and a window is cut with two binary searches.

    Stored as {(chrom, strand): (positions int64 ascending, values int32)}.
    """

    __slots__ = ("data",)

    def __init__(self, data=None):
        self.data = dict(data or {})

    def __contains__(self, key):
        return key in self.data

    def keys(self):
        return self.data.keys()

    def total(self):
        return int(sum(int(v.sum()) for _p, v in self.data.values()))

    def window(self, chrom, strand, start, end):
        """Dense int32 of length end-start, zero where nothing was counted."""
        n = max(0, end - start)
        out = np.zeros(n, dtype=np.int32)
        hit = self.data.get((chrom, strand))
        if hit is None or n == 0:
            return out
        pos, val = hit
        lo = np.searchsorted(pos, start, "left")
        hi = np.searchsorted(pos, end, "left")
        if hi > lo:
            out[pos[lo:hi] - start] = val[lo:hi]
        return out

    def merge(self, other):
        """Add another set of counts in place.  Replicates are one measurement
        of the same thing; summing them is the point of having two."""
        for key, (p2, v2) in other.data.items():
            cur = self.data.get(key)
            if cur is None:
                self.data[key] = (p2.copy(), v2.copy())
                continue
            p1, v1 = cur
            pos = np.concatenate([p1, p2])
            val = np.concatenate([v1, v2])
            order = np.argsort(pos, kind="stable")
            pos, val = pos[order], val[order]
            uniq, first = np.unique(pos, return_index=True)
            self.data[key] = (uniq, np.add.reduceat(val, first).astype(np.int32))
        return self

    def to_npz(self):
        """Flat dict for np.savez_compressed; npz keys must be strings."""
        out = {}
        for (chrom, strand), (pos, val) in self.data.items():
            out["{}|{}|pos".format(chrom, strand)] = pos
            out["{}|{}|val".format(chrom, strand)] = val
        return out

    @classmethod
    def from_npz(cls, z):
        """Read either the sparse layout or the old dense one.

        Caches written before this change store one dense array per
        'chrom|strand'; they are converted on load, which is cheap at yeast
        scale and keeps every existing cache valid.
        """
        data = {}
        keys = [k for k in z.files if not k.startswith("_")]
        if any(k.endswith("|pos") for k in keys):
            for k in keys:
                if not k.endswith("|pos"):
                    continue
                chrom, strand, _ = k.rsplit("|", 2)
                data[(chrom, strand)] = (
                    z[k], z["{}|{}|val".format(chrom, strand)])
        else:
            for k in keys:
                chrom, strand = k.rsplit("|", 1)
                arr = z[k]
                nz = np.nonzero(arr)[0]
                data[(chrom, strand)] = (nz.astype(np.int64),
                                         arr[nz].astype(np.int32))
        return cls(data)


def accumulate_tracks(lines, chrom_lengths, offset=0, min_mapq=10,
                      coverage=False):
    """Stream SAM text once -> 5'-end counts, and optionally read coverage.

    Both tracks are built in the SAME pass.  The SAM stream is a pipe from
    bowtie2 and cannot be replayed, and buffering tens of millions of lines to
    iterate twice is not an option, so anything that needs the alignments has
    to take its turn here.

    5'-end counts are what the model consumes: a footprint means one 40S at
    one place, and smearing it over the read would invent occupancy.  Read
    coverage -- every position a read overlaps -- is for finding transcript
    boundaries instead, where the question is only "does mRNA exist here".
    Coverage is several times denser, so the cliff at a transcription start
    is much sharper.

    Coverage is accumulated as a difference array (+1 at the start, -1 past
    the end, one cumulative sum at the end) rather than a slice-add per read,
    which turns tens of millions of numpy slice operations into two scalar
    ones each.

    Returns (counts, coverage_or_None, stats).  Reads whose assigned position
    falls off the end of the chromosome (possible once offset is non-zero) are
    counted as out_of_bounds and skipped rather than wrapped around.
    """
    # One compact growable buffer of positions per (chrom, strand).  Four
    # bytes per assigned read, aggregated into sorted unique positions at the
    # end -- see CountTracks for why this is not a dense genome array.
    raw = defaultdict(lambda: array.array("l"))
    # Coverage stays dense, because a difference array has to be. It is only
    # ever asked for on the abundance fraction, which a 40S-only experiment
    # does not have, and it is allocated per chromosome on first touch so an
    # unused chromosome costs nothing.
    diff = {} if coverage else None
    stats = defaultdict(int)

    for line in lines:
        stats["total"] += 1
        rec = parse_sam_line(line, min_mapq=min_mapq)
        if rec is None:
            stats["filtered"] += 1
            continue
        rname, flag, pos1, cigar = rec
        if rname not in chrom_lengths:
            stats["unknown_chrom"] += 1
            continue

        pos0, strand = five_prime_position(flag, pos1, cigar, offset=offset)
        if 0 <= pos0 < chrom_lengths[rname]:
            raw[(rname, strand)].append(pos0)
            stats["assigned"] += 1
        else:
            stats["out_of_bounds"] += 1

        if diff is not None:
            d = diff.get((rname, strand))
            if d is None:
                d = diff[(rname, strand)] = np.zeros(
                    chrom_lengths[rname] + 1, dtype=np.int32)
            lo = pos1 - 1
            hi = min(lo + cigar_reference_span(cigar), len(d) - 1)
            if 0 <= lo < hi:
                d[lo] += 1
                d[hi] -= 1
                stats["covered"] += 1

    counts = CountTracks()
    for key, buf in raw.items():
        pos = np.frombuffer(buf, dtype=np.int64 if buf.itemsize == 8 else np.int32)
        pos = np.sort(pos.astype(np.int64))
        uniq, cnt = np.unique(pos, return_counts=True)
        counts.data[key] = (uniq, cnt.astype(np.int32))

    cov = None
    if diff is not None:
        cov = CountTracks()
        for k, v in diff.items():
            dense = np.cumsum(v[:-1], dtype=np.int32)
            nz = np.nonzero(dense)[0]
            cov.data[k] = (nz.astype(np.int64), dense[nz])
    return counts, cov, dict(stats)


def accumulate_five_prime_ends(lines, chrom_lengths, offset=0, min_mapq=10):
    """5'-end counts only.  Thin wrapper over accumulate_tracks."""
    counts, _, stats = accumulate_tracks(
        lines, chrom_lengths, offset=offset, min_mapq=min_mapq, coverage=False)
    return counts, stats


def utr_interval(cds_start, cds_end, strand, utr_length, chrom_length,
                 include_start_codon=True):
    """Genomic interval for a fixed-window 5'UTR, 0-based half-open.

    cds_start/cds_end are the 0-based half-open CDS bounds on the genome, so
    for a minus-strand gene the start codon sits at the *right* edge.

    The returned interval includes the start codon (3 nt) at its 3' end unless
    told otherwise, which is why main_start_index ends up at N - 3.  Keeping
    the AUG inside the window means the occupancy vector covers the position
    the model cares about most.

    Returns (start, end) or None if the window would run off the chromosome
    or come out empty.
    """
    pad = 3 if include_start_codon else 0
    if strand == "+":
        start = cds_start - utr_length
        end = cds_start + pad
    else:
        start = cds_end - pad
        end = cds_end + utr_length

    start = max(0, start)
    end = min(chrom_length, end)
    if end - start < pad + 1:
        return None
    return start, end


def first_cds_blocks(cds_blocks, strand, length):
    """The first `length` coding nt as genomic intervals, in transcript order.

    cds_blocks are the transcript's CDS exons as 0-based half-open genomic
    intervals, any order.  Walking them in transcript order (left to right on
    +, right to left on -) and stopping after `length` nt gives the region an
    80S ribosome elongates through right after committing at the AUG -- exon
    aware, so an intron 20 nt into a ribosomal-protein gene does not eat the
    window.

    This region is not part of the chain's state space.  It is where 80S
    footprints are summed to read out how many ribosomes committed at this
    AUG: every one of them has to pass through here, so the sum measures
    commitment flux.

    Returns a list of (start, end) intervals in transcript order, total span
    <= length; empty if there is no CDS.
    """
    blocks = sorted(cds_blocks, reverse=(strand == "-"))
    out, need = [], length
    for start, end in blocks:
        if need <= 0:
            break
        span = end - start
        if span <= 0:
            continue
        take = min(span, need)
        if strand == "+":
            out.append((start, start + take))
        else:
            out.append((end - take, end))
        need -= take
    return out


def project_onto_blocks(counts, chrom, strand, blocks, offset=0):
    """Concatenate project_onto_utr over several intervals in transcript order."""
    parts = [project_onto_utr(counts, chrom, strand, s, e, offset=offset)
             for s, e in blocks]
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int32)


def project_onto_utr(counts, chrom, strand, start, end, offset=0):
    """Slice the genomic count track onto a UTR, oriented cap -> AUG.

    On the minus strand the genomic array runs the other way, so the slice is
    reversed.  The result is indexed 1:1 with the UTR sequence returned by
    utr_sequence for the same interval.

    `offset` is the footprint-5'-end-to-P-site shift, applied here rather than
    at counting time.  five_prime_position can bake it in, but then it is
    frozen into a counts file that took an hour to build, and the right value
    is not known until the metagene has been measured FROM that file.  Storing
    raw 5' ends and shifting here keeps the measurement and the decision in
    the right order, and costs nothing: a P site at genomic g was a 5' end at
    g - offset on the plus strand, g + offset on the minus, so the window
    simply moves.
    """
    shift = -offset if strand == "+" else offset
    vec = counts.window(chrom, strand, start + shift, end + shift)
    if strand == "-":
        vec = vec[::-1]
    return np.ascontiguousarray(vec)


COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


def revcomp(seq):
    return seq.translate(COMPLEMENT)[::-1]


def utr_sequence(genome, chrom, strand, start, end):
    """Genomic sequence for the interval, oriented cap -> AUG."""
    seq = genome[chrom][start:end].upper()
    return revcomp(seq) if strand == "-" else seq


# --- inline sanity checks -------------------------------------------------
# These run on import.  They are cheap and they pin the coordinate
# conventions, which are the easiest thing in this file to get quietly wrong.

assert cigar_reference_span("30M") == 30
assert cigar_reference_span("10M5I10M") == 20          # insertions skip the ref
assert cigar_reference_span("10M5D10M") == 25          # deletions consume it
assert cigar_reference_span("3S27M2S") == 27           # soft clips do not

# Plus strand, 1-based POS 101 -> 0-based 100.
assert five_prime_position(0, 101, "30M") == (100, "+")
# Minus strand: 5' end is the far edge, 100 + 30 - 1 = 129.
assert five_prime_position(FLAG_REVERSE, 101, "30M") == (129, "-")
# Offset shifts downstream on each strand's own terms.
assert five_prime_position(0, 101, "30M", offset=12) == (112, "+")
assert five_prime_position(FLAG_REVERSE, 101, "30M", offset=12) == (117, "-")

# A plus-strand UTR of 50 nt ends 3 nt past the CDS start; a minus-strand one
# begins 3 nt before the CDS end, and both are 53 nt long.
assert utr_interval(1000, 2000, "+", 50, 10000) == (950, 1003)
assert utr_interval(1000, 2000, "-", 50, 10000) == (1997, 2050)

# The CDS window starts at the AUG on either strand, follows exons, and is
# clipped to the CDS for short genes.
assert first_cds_blocks([(1000, 2000)], "+", 100) == [(1000, 1100)]
assert first_cds_blocks([(1000, 2000)], "-", 100) == [(1900, 2000)]
assert first_cds_blocks([(1000, 1030)], "+", 100) == [(1000, 1030)]
# Intron 20 nt in: the window continues in the next exon, on either strand.
assert first_cds_blocks([(1000, 1020), (1300, 2000)], "+", 100) == [(1000, 1020), (1300, 1380)]
assert first_cds_blocks([(1000, 1700), (1980, 2000)], "-", 100) == [(1980, 2000), (1620, 1700)]
# UTR and CDS windows abut: the UTR's last 3 nt are the CDS's first 3.
assert utr_interval(1000, 2000, "+", 50, 10000)[1] == first_cds_blocks([(1000, 2000)], "+", 100)[0][0] + 3
