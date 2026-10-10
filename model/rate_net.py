"""f_theta: a window of local sequence + structure -> per-position rates.

Three scalars come out per nucleotide j:

    w_j  step tendency    -- log-odds of scanning on, against dropping off
    p_j  commit tendency  -- log-odds of committing, against dropping off
    d_j  dwell tendency   -- log-odds of pausing in place, against dropping off

and a fourth from a separate head:

    r_j  reinit tendency  -- logit of the fraction of walkers initiating at
                             uAUG j that terminate and resume scanning

STRUCTURE AS AN AUXILIARY TARGET, not as an input.  The trunk carries a
second head that predicts local secondary structure -- whether each base is
paired, and the local folding energy -- and train.py --struct-aux adds that
to the loss.  The INPUT stays one-hot sequence; nothing is folded at
inference.

The first attempt fed `paired` and `local_dG` in as extra input channels.
That was worse than no structure at all, and the reason is probably that it
hands the model a shortcut built on a fiction: the context we fold is the
last 5'UTR exon plus GENOMIC flanks, so for the ~45% of human leaders that
are spliced, part of what RNAfold sees is intron the ribosome never meets.
As an input channel that fiction is consumed directly.  As a target it only
has to be predicted, and a noisy target degrades gracefully.

This is the arrangement UTR-LM uses (Chu et al. 2024): secondary structure
and minimum free energy enter as pre-training objectives beside masked-
nucleotide reconstruction, never as input.  Worth noting their own ablation
dropped the structure objective for the translation-efficiency task while
keeping it for ribosome loading -- so the auxiliary weight is something to
measure, not assume.

r_j is produced from the uORF's GEOMETRY ALONE -- its length, and the gap
from its stop codon to the main AUG -- and never touches the shared stack.
That isolation is the whole design, and it was arrived at the hard way.

The first version fed the head [trunk state, geometry].  It looked safe: the
trunk's INPUT still had no positional channel, so w, p and d could not read
position forward.  But gradient flows the other way.  Backprop from the head
reached the trunk, which was then co-optimised to serve a consumer that does
know where it is, and the learned rates shifted everywhere.  Measured against
the MPRA, commit_main fell from rho 0.644 to 0.527 -- and crucially it fell
just as far on the 4156 leaders with no uAUG at all, where the reinitiation
edges are mathematically inert.  Sequences the mechanism cannot touch moved
by as much as the ones it can, which is what identified the gradient rather
than the edges as the cause.

So the head reads geometry and nothing else.  Two inputs, one hidden layer of
eight: too small to fit profile wiggles, and with no path into the trunk it
cannot reshape the rates even if it wanted to.  Reinitiation efficiency is
known to depend on uORF length and intercistronic distance, so this is also
the more honest parameterisation -- a mechanism rather than a free function.

p_j and d_j are only consulted where start_mask is set; elsewhere the chain
masks them out, so the net is free to output anything there.

The window is how a memoryless chain still gets to see a 3-nt codon and its
Kozak context.  We stack dilated convolutions whose receptive field is exactly
2 * window + 1, so position j's rates depend on [j - window, j + window] and
nothing else.
"""

import torch
import torch.nn as nn

from data.features import NET_INPUT_CHANNELS, N_UORF_GEOM


class RateNet(nn.Module):
    def __init__(self, in_channels=NET_INPUT_CHANNELS, hidden=32, window=10):
        super().__init__()
        self.window = window
        k = 2 * window + 1

        # One wide conv for the context, then two 1x1 layers that mix channels
        # without widening the receptive field any further.
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, hidden, kernel_size=k, padding=window),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(hidden, 3, kernel_size=1),
        )

        # Geometry in, one logit out, no connection to the trunk in either
        # direction -- see the note at the top of the file for what happened
        # when there was one.  Kept out of self.net so the trunk's parameter
        # names stay exactly as they were and a checkpoint from before
        # reinitiation still loads (model/scanner.py: load_state_compat).
        self.reinit_head = nn.Sequential(
            nn.Linear(N_UORF_GEOM, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

        # Auxiliary structure head: two numbers per position, read off the
        # SHARED trunk so that predicting structure shapes the same features
        # the rates are built from.  That sharing is the whole mechanism, and
        # also the risk -- the reinitiation head once backpropagated into this
        # trunk and cost 0.1 of held-out R2.  The difference is that head knew
        # where it was in the sequence; this one predicts a local property of
        # the sequence itself.  Weighted by train.py --struct-aux.
        self.struct_head = nn.Conv1d(hidden, 2, kernel_size=1)

        # Start life processive.  These are log-odds against drop-off, so a
        # step logit of 6 means ~0.0025 drop-off per nt and a particle that
        # actually reaches the far end of a 100-nt UTR.  Initialise near zero
        # and it falls off within a few nucleotides, leaving the solve with a
        # profile of all zeros and nothing to learn from.  Dwell starts level
        # with commit: a modest pause at each AUG, to be sharpened by the data.
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        with torch.no_grad():
            final.bias.copy_(torch.tensor([6.0, 2.0, 2.0]))

        # Start agnostic about reinitiation: sigmoid(0) = 0.5, so half the
        # walkers initiating at a uORF come back.  Initialising either way
        # would be asserting the answer to the question this head exists to
        # ask.
        nn.init.zeros_(self.reinit_head[-1].weight)
        nn.init.zeros_(self.reinit_head[-1].bias)

    def forward(self, x):
        """x: (N, C) feature tracks -> (w, p, d, struct).

        struct is (2, N): a paired/unpaired LOGIT and a local folding energy,
        over the full context.  It is a training signal only -- nothing in the
        chain reads it -- so it is computed before any cropping, which gives
        the auxiliary loss the flanks as extra supervision.
        """
        assert x.dim() == 2, "RateNet takes one UTR at a time, shaped (N, C)"
        z = x.t().unsqueeze(0)
        h = self.net[:-1](z)                          # (1, hidden, N)
        out = self.net[-1](h).squeeze(0)              # (3, N)
        struct = self.struct_head(h).squeeze(0)       # (2, N)
        return out[0], out[1], out[2], struct

    def reinit_logit(self, geom):
        """geom: (N, 2) uORF geometry -> (N,) reinitiation logit.

        Takes no trunk state, on purpose.  geom is indexed in chain
        coordinates because it is measured relative to the main AUG, which
        only exists there.
        """
        assert geom.dim() == 2 and geom.shape[1] == N_UORF_GEOM, \
            "geom must be (N, {}), got {}".format(N_UORF_GEOM, tuple(geom.shape))
        return self.reinit_head(geom).reshape(-1)
