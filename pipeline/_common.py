"""Shared helpers for the pipeline scripts: paths, binaries, idempotency."""

import os
import shutil
import subprocess
import sys

# Repo root, so the scripts work no matter where they are invoked from.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

REF_DIR = os.path.join(ROOT, "ref")
WORK_DIR = os.path.join(ROOT, "work")
CACHE_DIR = os.path.join(ROOT, "cache")

# DATASETS.  Everything that differs between experiments, in one place.  Not a
# config framework -- a dict of presets, chosen with --dataset, so that adding
# a species does not mean editing five files and silently getting one wrong.
#
# The adapter is the entry that matters most: trimming with the wrong one is
# not a crash, it is a quiet 95% loss, because nothing gets trimmed and almost
# nothing aligns.  Measured on the first 100k reads of each run, the human 40S
# libraries carry CTGTAGGCACCATCAAT in ~94.5% of reads and the yeast adapter
# in 0.0%.
DATASETS = {
    "archer-yeast": {
        "genome": "sacCer3",
        # Illumina TruSeq Small RNA 3' adapter (RA3).  This library is
        # poly(A)-tailed before the adapter is ligated, so there are two
        # things to strip off the 3' end, in that order.
        "adapter": "TGGAATTCTCGGGTGCCAAGG",
        "poly_a": True,
        # Below 18 nt a read stops being uniquely placeable in a 12 Mb genome.
        "min_length": 18,
        "fractions": {
            "ssu": "SRR3458591",    # SRX1734358, WT SSU (40S footprints)
            "rs": "SRR3458592",     # SRX1734359, WT RS  (80S footprints)
            "input": "SRR3458593",  # SRX1734360, WT Input (total RNA)
        },
    },
    "human-40s": {
        "genome": "GRCh38",
        # Illumina Small RNA 3' adapter v1.5, the usual one in ribosome- and
        # 40S-profiling protocols.  No poly(A) tailing in this protocol, so
        # the second pass is off -- leaving it on would eat genomic A-runs off
        # real footprint ends.
        "adapter": "CTGTAGGCACCATCAAT",
        "poly_a": False,
        # The human genome is ~250x larger than yeast, so short reads stop
        # placing uniquely much sooner.  Measured insert lengths are median
        # 27-34 nt with a 10th percentile near 17, so 25 keeps the bulk of the
        # footprints and drops the tail that would multimap anyway.
        "min_length": 25,
        # Unselected 40S, two runs.  GEO calls them "human unselected 40S 1"
        # and "2" (GSM4131410 / GSM4131411, SRX7026380 / SRX7026381).  The
        # SRR mapping was verified against SRA runinfo: SRR10315320 is
        # GSM4131410 (HiSeq X Ten, 100 nt), SRR10315321 is GSM4131411 (HiSeq
        # 2500, 101 nt).  They are not interchangeable: different
        # instruments, and median insert is 34 nt in one and 27 in the other.
        "fractions": {
            "ssu": "SRR10315320",
            "ssu_rep2": "SRR10315321",
        },
    },
    # The SELECTED fraction from the same series, and the one to prefer.
    # Sel-TCP-seq pulls down eIF3b, which rides with scanning 40S, so this is
    # the subpopulation the model is actually about.  The authors' own
    # per-region footprint counts (GSE139131 supplementary) make the case
    # better than any argument:
    #
    #                      5'UTR   start     CDS    3'UTR
    #   eIF3b::40S         0.507   0.180   0.172    0.136
    #   unselected 40S     0.110   0.085   0.318    0.473
    #
    # Unselected 40S spends 47% of its reads in 3'UTRs, where a scanning
    # ribosome has no business being, and only 11% in leaders.  Modelling
    # scanning with it means fitting an 11% minority of a largely
    # non-scanning population -- which is what the first human run did.
    "human-eif3b": {
        "genome": "GRCh38",
        "adapter": "CTGTAGGCACCATCAAT",
        "poly_a": False,
        "min_length": 25,
        # GSM4131412 / SRX7026382.  One replicate only; the selected
        # fractions were not repeated.
        "fractions": {"ssu": "SRR10315322"},
    },
    # Replicated eIF3B Sel-TCP-seq, HEK293T (GSE174329).  GSM5291862 and
    # GSM5291863, resolved to runs through SRA runinfo rather than by order.
    #
    # The deposited reads are ALREADY adapter-trimmed (GEO: cutadapt + FASTX
    # quality filtering): on 100k reads per run, neither the small-RNA
    # adapter, the TruSeq/linker adapter nor the yeast RA3 occurs once, and
    # lengths run 10-89 nt.  The adapter below is the McGlincy-Ingolia linker
    # this protocol ligates, kept only as a guard -- it matches nothing --
    # so the same cutadapt call still applies the quality and length filters.
    # min_length matches the other human presets so that selected and
    # unselected libraries are filtered identically.
    "human-eif3b-replicated": {
        "genome": "GRCh38",
        "adapter": "AGATCGGAAGAGCACACGTCTGAA",
        "poly_a": False,
        "min_length": 25,
        "fractions": {
            "ssu_rep1": "SRR14510018",   # GSM5291862, eIF3B Sel-TCP-seq rep 1
            "ssu_rep2": "SRR14510019",   # GSM5291863, eIF3B Sel-TCP-seq rep 2
        },
    },
}

DEFAULT_DATASET = "archer-yeast"

# The yeast paths stay as module constants because 03_build_examples.py,
# scripts/call_utr_boundaries.py and elong/ all import them directly and only
# ever run on yeast.  Anything that must work on both species takes a dataset
# dict instead -- see dataset().
BOWTIE2_INDEX = os.path.join(REF_DIR, "sacCer3")
GENOME_FA = os.path.join(REF_DIR, "sacCer3.fa")
GTF = os.path.join(REF_DIR, "sacCer3.gtf")
FRACTIONS = DATASETS[DEFAULT_DATASET]["fractions"]


def dataset(name=None):
    """Resolve a preset into concrete paths.  -> dict.

    `prefix` namespaces this dataset's files under work/, empty for the
    default so existing yeast outputs keep their names and stay valid.
    """
    name = name or DEFAULT_DATASET
    if name not in DATASETS:
        sys.exit("\nUnknown dataset {!r}. Known: {}\n".format(
            name, ", ".join(sorted(DATASETS))))
    d = dict(DATASETS[name])
    g = d["genome"]
    d.update(
        name=name,
        index=os.path.join(REF_DIR, g),
        genome_fa=os.path.join(REF_DIR, g + ".fa"),
        gtf=os.path.join(REF_DIR, g + ".gtf"),
        prefix="" if name == DEFAULT_DATASET else name + ".",
    )
    return d


def work_path(ds, frac, suffix):
    """work/<prefix><fraction>.<suffix>, so two datasets never collide."""
    return os.path.join(WORK_DIR, "{}{}.{}".format(ds["prefix"], frac, suffix))

# Default search locations for the raw FASTQs, relative to ROOT's parent.
FASTQ_SEARCH_DIRS = [
    os.path.join(os.path.dirname(ROOT), "fastq"),
    os.path.join(ROOT, "fastq"),
    os.path.dirname(ROOT),
]


# Illumina TruSeq Small RNA 3' adapter (RA3), and the floor below which a
# yeast read stops being uniquely placeable.  Shared so that 01_trim.py and
# 02_align_count.py --stream trim identically; a module whose name starts with
# a digit cannot be imported, so this is the only place they can agree.
ADAPTER = "TGGAATTCTCGGGTGCCAAGG"
MIN_LENGTH = 18


def find_cutadapt():
    """-> argv prefix for cutadapt, as a list.

    pip drops cutadapt.exe into a per-user Scripts directory that is usually
    NOT on PATH on Windows, so resolving it by name fails even though the
    package is installed.  Running it as a module with this interpreter works
    whenever the import works, which is the condition that actually matters.
    """
    exe = _which("cutadapt")
    if exe:
        return [exe]
    try:
        import cutadapt  # noqa: F401
    except ImportError:
        sys.exit("\ncutadapt not found on PATH, and not importable either.\n"
                 "  pip install cutadapt\n")
    return [sys.executable, "-m", "cutadapt"]


def cutadapt_cmd(cutadapt, src, dst, threads=0, ds=None):
    """The trimming invocation for one dataset.  dst of "-" writes to stdout.

    Only the 3' end is ever touched, which is what keeps the 5' end -- the
    position the whole model is built on -- exactly where the sequencer put
    it.  Where the library is poly(A)-tailed before the adapter is ligated
    there are two things to strip, and --times 2 makes the second pass that
    catches the tail the adapter was hiding.

    cutadapt sends its report to stderr whenever -o is stdout, so the stream
    stays clean FASTQ.
    """
    ds = ds or dataset()
    prefix = list(cutadapt) if isinstance(cutadapt, (list, tuple)) else [cutadapt]
    cmd = prefix + ["-a", ds["adapter"]]
    if ds.get("poly_a"):
        cmd += ["-a", "A{20}", "--times", "2"]
    cmd += [
        "-m", str(ds["min_length"]),
        "-q", "20",             # 3' quality trim; these runs degrade badly late
        "--max-n", "0",
        "-j", str(threads),
        "-o", dst,
        src,
    ]
    return cmd


def find_fastqs(extra_dir=None, fractions=None):
    """Locate the raw FASTQs by accession.  Returns {fraction: path}.

    Looks for <accession>.fastq(.gz) and, in case the runs were dumped as
    paired, <accession>_1.fastq(.gz).  These data are single-end, so a _1 file
    is treated as the single read file.
    """
    dirs = ([extra_dir] if extra_dir else []) + FASTQ_SEARCH_DIRS
    found = {}
    for frac, acc in (fractions or FRACTIONS).items():
        for d in dirs:
            if not d or not os.path.isdir(d):
                continue
            for name in (acc + ".fastq", acc + ".fastq.gz",
                         acc + "_1.fastq", acc + "_1.fastq.gz"):
                p = os.path.join(d, name)
                if os.path.exists(p):
                    found[frac] = p
                    break
            if frac in found:
                break
        if frac not in found:
            sys.exit(
                "\nCould not find the {} FASTQ ({}).\n"
                "Looked in:\n  {}\n"
                "Pass --fastq-dir to point at it.\n".format(
                    frac, acc, "\n  ".join(d for d in dirs if d))
            )
    return found


# Binaries dropped into <repo>/bin/ (any subfolder) are found before PATH.
# This is how bowtie2 runs on Windows, where bioconda has no packages: unzip
# the official mingw build under bin/ and nothing else needs configuring.
BIN_DIR = os.path.join(ROOT, "bin")

BOWTIE2_HINT = """  Windows: download bowtie2-<ver>-mingw-x86_64.zip from
           https://github.com/BenLangmead/bowtie2/releases and unzip it under
           utr-scan/bin/  (bioconda has no Windows packages)
  Linux/macOS/WSL: conda install -c bioconda bowtie2"""


def _which(name):
    """shutil.which, but also searching bin/ and its subfolders first."""
    if os.path.isdir(BIN_DIR):
        for d, _, _ in os.walk(BIN_DIR):
            hit = shutil.which(name, path=d)
            if hit:
                return hit
    return shutil.which(name)


def require(binary, install_hint, fallbacks=()):
    """Resolve a binary or exit with something actionable.

    `fallbacks` are alternative names tried in order.  bowtie2's `bowtie2`
    entry point is a Perl wrapper whose only jobs are picking the small/large
    index flavour and decompressing gz input; we need neither, so the bare
    `bowtie2-align-s` executable is a fine stand-in.
    """
    for name in (binary,) + tuple(fallbacks):
        exe = _which(name)
        if exe:
            return exe
    sys.exit(
        "\n{} not found on PATH or under {}.\n{}\n"
        "See README.md for the full list of required binaries.\n".format(
            binary, BIN_DIR, install_hint)
    )


def up_to_date(output, *inputs):
    """True if `output` exists and is newer than every input that exists.

    This is what makes each step cheap to re-run: the step checks for its own
    output first and returns immediately if nothing upstream has moved.
    """
    if not os.path.exists(output) or os.path.getsize(output) == 0:
        return False
    out_mtime = os.path.getmtime(output)
    return all(
        os.path.getmtime(i) <= out_mtime
        for i in inputs if i and os.path.exists(i)
    )


def run(cmd, **kwargs):
    print("[run] " + " ".join(str(c) for c in cmd))
    return subprocess.run(cmd, check=True, **kwargs)


def ensure_dirs():
    for d in (WORK_DIR, CACHE_DIR, REF_DIR):
        os.makedirs(d, exist_ok=True)


def load_counts_npz(path):
    """Read a 5'-end counts .npz back into a CountTracks.

    npz keys are flat strings, so a track is stored as "chrom|strand|pos" and
    "chrom|strand|val".  Entries starting with "_" are run metadata, not count
    tracks.  Caches written before counts went sparse hold one dense array per
    "chrom|strand"; CountTracks.from_npz reads either layout, so existing
    yeast caches stay valid.
    """
    import numpy as np
    from data.coverage import CountTracks
    with np.load(path, allow_pickle=True) as z:
        return CountTracks.from_npz(z)
