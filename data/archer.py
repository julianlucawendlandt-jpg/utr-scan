"""Loader for the cached Archer TCP-seq examples.

The pipeline writes one cache file; training reads it and never touches a
FASTQ.  Each example is a dict:

    transcript_id      str
    utr_seq            str, cap -> AUG, the last 3 nt are the main AUG
    main_start_index   int, == len(utr_seq) - 3
    ssu_counts         (N,) int64   40S footprint 5'-end counts (SRX1734358)
    input_counts       (N,) int64   total-RNA 5'-end counts     (SRX1734360)
    rs_cds_counts      (M,) int64   80S footprint 5'-end counts (SRX1734359)
                                    over the first M nt of the CDS
    input_cds_counts   (M,) int64   total-RNA counts over the same M nt
    utr_context_seq    str          utr_seq with real genomic flanks on both
                                    sides, so the rate net never sees padding
    context_offset     int          index of utr_seq[0] inside utr_context_seq

The last two are the commitment readout (see train.py: commit_nll).  They
are optional so that older caches still load; without them the loss falls
back to occupancy only and the fates are not identified.

Stored as a pickled object array inside an .npz, which keeps the ragged N.
"""

import os

import numpy as np

DEFAULT_CACHE = os.path.join("cache", "archer_examples.npz")

# What every example must carry.  input_counts is NOT here: it is the
# abundance control, and it only exists where the experiment had an Input
# fraction.  A 40S-only dataset has none, and the model never reads it unless
# train.py --normalize-input is passed.  Requiring it silently dropped every
# human example.
REQUIRED_KEYS = ("transcript_id", "utr_seq", "main_start_index", "ssu_counts")


def save_examples(examples, path=DEFAULT_CACHE):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    np.savez_compressed(path, examples=np.array(examples, dtype=object))
    return path


def load_examples(path=DEFAULT_CACHE, min_total_counts=1, min_length=10):
    """Read the cache and drop examples that cannot train.

    Degenerate cases -- no reads at all, a UTR too short to hold a window, a
    missing main AUG -- are skipped rather than raised, and the number dropped
    is printed so it stays visible.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            "No cached examples at {}.  Run the pipeline first:\n"
            "  python pipeline/run_all.py\n"
            "(see DATA.md for the required binaries)".format(path)
        )

    with np.load(path, allow_pickle=True) as z:
        raw = list(z["examples"])

    kept, dropped = [], 0
    for ex in raw:
        try:
            assert all(k in ex for k in REQUIRED_KEYS)
            n = len(ex["utr_seq"])
            if n < min_length:
                dropped += 1
                continue
            if len(ex["ssu_counts"]) != n:
                dropped += 1
                continue
            if "input_counts" in ex and len(ex["input_counts"]) != n:
                dropped += 1
                continue
            if int(np.sum(ex["ssu_counts"])) < min_total_counts:
                dropped += 1
                continue
            i = ex["main_start_index"]
            if not (0 <= i < n) or ex["utr_seq"][i:i + 3].upper().replace("U", "T") != "ATG":
                dropped += 1
                continue
            kept.append(ex)
        except Exception:
            dropped += 1

    print("[archer] loaded {} examples from {} ({} dropped)".format(len(kept), path, dropped))
    return kept
