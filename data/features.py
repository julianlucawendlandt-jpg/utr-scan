"""Sequence -> feature tracks.  Pure functions, numpy in / numpy out.

Every track is length N and indexed 1:1 with the UTR sequence, which runs
cap -> AUG.  The observed SSU counts use the same indexing, so anything that
changes the convention here has to change there too.
"""

import warnings

import numpy as np

NUCS = "ACGT"
NUC_INDEX = {c: i for i, c in enumerate(NUCS)}

# Near-cognates are the one-mismatch neighbours of AUG that yeast initiates at
# weakly.  Off by default -- switching them on widens start_mask, which opens
# extra commit edges in the chain.
NEAR_COGNATES = ("ATG", "TTG", "CTG", "GTG", "ACG", "AAG", "AGG", "ATA", "ATC", "ATT")

STOP_CODONS = ("TAA", "TAG", "TGA")

# Extra channels the reinitiation head reads on top of the rate net's trunk:
# (uORF length, reinitiation gap).  See uorf_returns.
N_UORF_GEOM = 2


def one_hot(seq):
    """(N, 4) one-hot over ACGT.  Anything else (N, IUPAC codes) is all-zero."""
    seq = seq.upper().replace("U", "T")
    x = np.zeros((len(seq), 4), dtype=np.float32)
    for i, c in enumerate(seq):
        j = NUC_INDEX.get(c)
        if j is not None:
            x[i, j] = 1.0
    return x


def start_positions(seq, near_cognate=False):
    """(N,) bool marking the FIRST nucleotide of each start codon.

    This is a separate track, never a substitution into the sequence -- the
    real nucleotides stay untouched and the rate net still sees them.  The
    mask only tells the chain where a commit transition is allowed to open.

    Flagging the A of ATG (rather than the codon centre or its 3' end) is the
    convention the whole codebase uses; the target counts are indexed the same
    way, so the model's commit position and the data's 5'-end pileup line up.
    """
    seq = seq.upper().replace("U", "T")
    codons = NEAR_COGNATES if near_cognate else ("ATG",)
    mask = np.zeros(len(seq), dtype=bool)
    for i in range(len(seq) - 2):
        if seq[i:i + 3] in codons:
            mask[i] = True
    return mask


def main_start_track(n, main_start_index):
    """(N,) bool with a single True at the annotated main AUG."""
    mask = np.zeros(n, dtype=bool)
    if main_start_index is not None and 0 <= main_start_index < n:
        mask[main_start_index] = True
    return mask


def uorf_returns(seq, start_mask, main_start_index=None):
    """Where a 40S resumes scanning after translating each uORF.

    A ribosome that initiates at an upstream AUG translates in frame to the
    first in-frame stop, terminates, and -- for a fraction of events -- stays
    on the mRNA and resumes scanning just past that stop.  model/chain.py
    turns the resumption into a transient edge Q[j, return_idx[j]], which is
    what stops a uORF being a dead end.

    Two cases get no return edge and stay absorbing at commit_uORF:

      * no in-frame stop inside the state space.  We cannot say where the
        ribosome terminates, so we do not invent a position.
      * the stop lands past the main AUG (an overlapping uORF).  The ribosome
        terminates DOWNSTREAM of the start it would have to reinitiate at, so
        it cannot come back to it.  These are the strongly repressive ones and
        "absorbed at commit_uORF" is the right fate for them.

    Returns three (N,) arrays, all zero / -1 away from a uORF start:

        return_idx  int32    index scanning resumes at, or -1 for no return
        uorf_len    float32  codons from the uAUG to its stop, stop excluded
        reinit_gap  float32  nucleotides from the return to the main AUG

    The two geometry tracks exist because reinitiation efficiency depends on
    them: a long uORF gives the 40S more time to shed initiation factors, and
    a long gap gives it more time to reacquire ternary complex before the next
    AUG.  Without them the reinitiation head would have nothing to condition
    on but the local sequence, and would almost certainly degenerate into a
    constant the way the uORF commit rate already has.

    They feed the reinitiation head ONLY, never the shared trunk.  reinit_gap
    is a distance to the main AUG -- exactly the positional cue that featurize
    goes to some trouble to keep out of the rate net (see the note there).
    Keeping it behind the trunk means w, p and d still cannot see where they
    are; see model/rate_net.py.
    """
    seq = seq.upper().replace("U", "T")
    n = len(seq)
    return_idx = np.full(n, -1, dtype=np.int32)
    uorf_len = np.zeros(n, dtype=np.float32)
    reinit_gap = np.zeros(n, dtype=np.float32)

    # A return may land anywhere up to and including the main AUG: landing on
    # it exactly is the best case, a walker delivered straight to the start.
    limit = (n - 1) if main_start_index is None else int(main_start_index)

    for j in np.nonzero(np.asarray(start_mask))[0]:
        j = int(j)
        if main_start_index is not None and j == int(main_start_index):
            continue                       # the main AUG is not a uORF
        stop = None
        k = j + 3
        while k + 3 <= n:
            if seq[k:k + 3] in STOP_CODONS:
                stop = k
                break
            k += 3
        if stop is None:
            continue
        ret = stop + 3
        if ret > limit or ret >= n:
            continue                       # oORF, or off the end of the chain
        return_idx[j] = ret
        uorf_len[j] = (stop - j) // 3
        reinit_gap[j] = limit - ret

    return return_idx, uorf_len, reinit_gap


def _fold_tracks(seq, window):
    """(paired, local_dG) from ViennaRNA, or zeros with a warning if absent."""
    n = len(seq)
    try:
        import RNA
    except ImportError:
        warnings.warn(
            "ViennaRNA (import RNA) not installed -- structure tracks are all "
            "zero.  pip install ViennaRNA to switch them on.",
            RuntimeWarning, stacklevel=3,
        )
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32)

    rna = seq.upper().replace("T", "U")

    structure, _ = RNA.fold(rna)
    paired = np.array([0.0 if c == "." else 1.0 for c in structure], dtype=np.float32)

    # Sliding-window local folding energy, assigned to the window centre and
    # edge-padded.  More local than the MFE of the whole UTR, which is what a
    # scanning 40S actually contends with.
    local_dG = np.zeros(n, dtype=np.float32)
    if n >= 1:
        half = window // 2
        for i in range(n):
            lo, hi = max(0, i - half), min(n, i + half + 1)
            local_dG[i] = RNA.fold(rna[lo:hi])[1]
    return paired, local_dG


def featurize(seq, main_start_index=None, near_cognate=False,
              structure=True, fold_window=40, context_seq=None,
              context_offset=0, paired=None, local_dG=None):
    """Build every track for one UTR.

    `seq` is the chain's state space: N nucleotides, cap -> AUG, indexed 1:1
    with the observed counts.  `context_seq` is the same stretch of genome
    with real flanking sequence added on both sides, and `context_offset` says
    where `seq` starts inside it (so context_seq[offset:offset+N] == seq).

    The two are separate on purpose.  A convolution cannot see array indices,
    only the values in its receptive field -- so the ONLY way position can
    leak into the rates is through padding, which is a value (zero) that no
    real nucleotide takes.  With the window ending 2 nt after the main AUG,
    "3 slots from the zeros" was a perfect main-AUG detector, and the net used
    it instead of learning any codon context.  Giving every position real
    sequence on both sides removes the cue entirely.  model/scanner.py crops
    the net's output back to the N chain positions and checks the flanks are
    at least as wide as the net's window.

    Returns a dict:
        one_hot        (N, 4) float32   chain-length, for reference
        start_mask     (N,)   bool      where a commit edge may open
        is_main_start  (N,)   bool      the single annotated main AUG
        uorf_return    (N,)   int32     where scanning resumes after a uORF
        uorf_geom      (N, 2) float32   uORF length + reinitiation gap
        paired         (M,)   float32   context-length structure tracks
        local_dG       (M,)   float32
        net_input      (M, C) float32   what the rate net consumes
        context_offset int              index of seq[0] within context_seq
        n_states       int              N

    The start tracks are deliberately NOT in net_input: the net has to read
    AUG context out of the raw sequence, while the masks are the chain's
    business.
    """
    n = len(seq)
    ctx = seq if context_seq is None else context_seq
    if context_seq is not None:
        assert ctx[context_offset:context_offset + n] == seq, \
            "context_seq[context_offset:context_offset+N] must equal seq"

    m = len(ctx)
    if structure and paired is not None and local_dG is not None:
        # Precomputed upstream.  Folding is by far the most expensive thing
        # here -- a sliding window over a 300 nt context is one RNAfold call
        # per position, ~29 minutes for the human set -- and featurize is
        # called afresh on every training run and every evaluation.  So
        # 03_build_examples.py --structure computes these once and stores
        # them in the cache; see data/archer.py REQUIRED_KEYS.
        paired = np.asarray(paired, dtype=np.float32)
        local_dG = np.asarray(local_dG, dtype=np.float32)
        assert len(paired) == len(local_dG) == m, (
            "precomputed structure tracks are {}/{} long but the context is "
            "{}".format(len(paired), len(local_dG), m))
    elif structure:
        paired, local_dG = _fold_tracks(ctx, fold_window)
    else:
        paired = np.zeros(m, dtype=np.float32)
        local_dG = np.zeros(m, dtype=np.float32)

    start_mask = start_positions(seq, near_cognate=near_cognate)
    ret_idx, uorf_len, reinit_gap = uorf_returns(seq, start_mask, main_start_index)

    feats = {
        "one_hot": one_hot(seq),
        "start_mask": start_mask,
        "is_main_start": main_start_track(n, main_start_index),
        "uorf_return": ret_idx,
        # log1p, then /5, purely to land both channels in roughly unit range:
        # a 300-nt gap and a 2-codon uORF are both plausible and a raw scale
        # would let the long tail dominate the head's single linear layer.
        "uorf_geom": (np.log1p(np.stack([uorf_len, reinit_gap], axis=1)) / 5.0
                      ).astype(np.float32),
        "paired": paired,
        # Scaled to roughly unit range so it does not swamp the one-hot inputs.
        "local_dG": local_dG / 10.0,
        "context_offset": int(context_offset),
        "n_states": n,
    }
    ctx_oh = one_hot(ctx)
    feats["net_input"] = np.concatenate(
        [ctx_oh, paired[:, None], feats["local_dG"][:, None]], axis=1
    ).astype(np.float32)
    return feats


# Number of channels in feats["net_input"], for wiring up the rate net.
NET_INPUT_CHANNELS = 6
