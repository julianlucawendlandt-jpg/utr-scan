"""Absorbing Markov chain over 5'UTR positions.

A single 40S particle loads at the cap (position 0) and takes discrete steps
along the UTR.  Every nucleotide j = 0 .. N-1 is one transient state.  There
are four absorbing states:

    0  commit_main   -- committed at the annotated main AUG
    1  commit_uORF   -- initiated at an upstream AUG and did NOT come back
                       (see REINITIATION below)
    2  drop_off      -- left the UTR upstream of the main AUG, without committing
    3  read_through  -- scanned PAST the main AUG without committing

When the state space is extended into the coding region (--cds-states), the
positions past the main AUG hold leaked 40S that are still scanning, and the
SSU footprints observed there become direct evidence for read_through.
Without that extension read_through is a residual: the window stops 2 nt
past the AUG, so every leaked walker is absorbed immediately and the only
evidence is the handful of counts at +1 and +2.

drop_off and read_through are both "no protein from this gene", but they are
different events and the data distinguishes them.  A walker can only reach a
position downstream of the main AUG by leaking past it, so every exit from
there is a read-through; an exit upstream is a genuine scanning failure.
Collapsing the two (as an earlier version did) hides leaky scanning inside
the drop-off number.  Nothing downstream of the main AUG is ever a start
codon -- the window ends 2 nt later -- so the split is exact, not an
approximation.

REINITIATION.  A uORF is not a dead end.  A 40S that initiates at an
upstream AUG translates the uORF as an 80S, terminates at its stop codon, and
a fraction of those particles stay on the mRNA and resume scanning -- which is
how a leader carrying a uORF can still make protein.  So the mass arriving at
a uAUG splits two ways:

    P_commit * (1 - f)  ->  R[j, COMMIT_UORF]       absorbed, protein lost
    P_commit *      f   ->  Q[j, return_idx[j]]     back in the chain

with f = sigmoid(r_j) the reinitiation fraction and return_idx[j] the position
just past the uORF's in-frame stop (data/features.py: uorf_returns).  The
softmax below is untouched -- this only chooses where the commit mass LANDS,
so the rows still sum to 1 by construction.

Two consequences worth stating.  First, positions between the uAUG and its
stop are skipped, and that is correct: there the particle is an 80S
translating the uORF, so it must not contribute to predicted 40S occupancy.
Positions after the return ARE visited, and that is also correct -- a
post-termination 40S really is scanning there and really does appear in the
SSU data.  That is what makes reinitiation an observable rather than a
bookkeeping choice.

Second, until this existed the uORF commit rate had no consequence any second
observable could see: commit there and the walker vanished, so the only
gradient reaching p_j at a uAUG came from occupancy shape, and "an AUG is
present here" was enough to satisfy it.  With a return edge the uORF rate
feeds commit_main, which the 80S term constrains.  The branch has a gradient
from an independent observable for the first time.

One consequence to be aware of, because it is easy to mistake for a bug.  A
reinitiating walker SKIPS the uORF span, so it also skips the drop-off risk
of those nucleotides -- which means that at high reinitiation a uORF can leave
commit_main very slightly ABOVE what the same leader with no uORF would give.
That is the right behaviour for the mechanism as stated: while translating,
the particle is an elongating 80S, and an elongating ribosome does not abort.
The effect is bounded by the drop-off probability over the uORF's length, so
it is small (a percent or two for a typical uORF); it is a modelling choice
made visible, not an artefact to be clamped.

The simplification: a returning walker re-enters as an ordinary scanning
particle, with the same w, p, d as anyone else at that position.  A real
post-termination 40S is depleted of ternary complex and briefly incompetent
to initiate, which is why reinitiation favours long intercistronic gaps.  That
dependence enters here only through r_j -- the head is shown the gap -- not
through the walker's state after it returns.

Allowed moves out of a transient state j:

    non-start, j < N-1 :  step j -> j+1          |  stay  |  drop_off
    start,     j < N-1 :  step j -> j+1 (leaky)  |  commit  |  stay  |  drop_off
    uORF start with a return:  the commit branch splits, absorb / re-enter
    j == N-1, non-start:  stay  |  drop_off  (nowhere left to step)
    j == N-1, start    :  commit  |  stay  |  drop_off

A word on identifiability.  An occupancy profile records where particles
*are*; committing and dropping off both remove a particle from the chain, and
the profile alone cannot tell the two apart.  Something has to break the tie.

There are two ways, and they are not equally good.

(1) A SECOND OBSERVABLE that sees the consequence of committing -- 80S
footprints in the coding region (train.py: commit_nll).  This was the default
for most of the project's life and it was the wrong choice.  80S/Input is
ribosome density over a gene's CDS: a real, reproducible measurement (split-
half reliability 0.46) of WHOLE-GENE translational efficiency, of which the
leader is a small minority component.  Measured against leader-driven output
in an MPRA it correlates +0.06.  So the term anchors the fates to codon
usage, mRNA stability and expression level -- properties of the gene that a
model reading 50-150 nt of 5'UTR cannot see and has no business predicting.
The symptom was unmistakable in hindsight: the longer you trained, the worse
the model ranked leaders, because it was converging onto the contaminated
target.

(2) A STRUCTURAL ASSUMPTION, which is what drop_at_starts=False gives:
drop-off is legal everywhere EXCEPT a start codon, so a 40S that leaves at an
AUG has committed by definition.  Now the profile pins everything on its own.
Mass that vanishes at the AUG is commitment; mass that continues past it is
leaky scanning; mass that fades along the leader is drop-off.  One assay in,
fates out -- and no rho_load, because there is no second library to convert
between.

(2) is the honest default.  It trades a measurement for a stated assumption,
and the assumption is at least about scanning, which is what this models.
TCP-seq genuinely cannot distinguish a committed 40S from a dissociated one;
that is a limit of the assay, and the right response is to say so in one line
of code rather than to import a proxy that measures something else.

"stay" is a self-loop, Q[j, j] > 0, and it is available at EVERY position.
Expected visits at j are P(reach j) / (1 - P(stay at j)), so a self-loop of
probability q means an expected residence time of 1 / (1 - q).  That makes
the dwell field a per-position SCANNING VELOCITY: high q = slow.  Occupancy
then comes out as flux / velocity, which is the standard steady-state
relation and the reason a footprint pile-up means "the ribosome is slow
here", not "something discrete happened here".

Restricting the self-loop to start codons (as an earlier version did) pins
the velocity at exactly 1 nt per step everywhere else, and that is a physical
claim the data rejects: the observed metagene ramps up over ~6 nt approaching
the AUG, which is a 40S decelerating as it inspects the context.  A chain
with fixed velocity cannot draw a ramp.

It also mattered for the fates.  With no dwell downstream of the AUG,
visits(j+1) / visits(j) IS P_step(j), so the observed ratio mechanically
dictated the leak and drove read_through to ~0.9.  With a velocity field the
ratio becomes P_step(j) / (1 - q(j+1)), so the same data can mean "the
ribosome moves fast just past the AUG" instead of "90% of ribosomes skip the
start codon".  The observation no longer dictates the fate.

Dwell changes occupancy but not fate.  Staying just defers the choice among
the other moves, so the probability of eventually committing versus dropping
from j is the same with or without the self-loop.  That decoupling is
deliberate: the profile is fit to *where* particles are seen, the fates say
where they *end up*, and the pause connects the two without confounding them.

The chain is memoryless: the probabilities above depend only on j.  All the
sequence context (the 3-nt codon, the Kozak -3/+4 positions, local structure)
enters through the *rates* w_j, p_j, d_j, which a windowed network computes
once, up front.  Context shapes the rates at build time; the walk itself
never looks back.

Q is upper triangular: the self-loop sits on the diagonal and the only other
transient moves are j -> j+1 and the reinitiation edge j -> return_idx[j].
That edge is strictly forward -- the shortest possible uORF is AUG plus a
stop, so return_idx[j] >= j + 6 -- which is why adding it changes nothing
about invertibility, conditioning, or the absence of cycles.

Every diagonal entry is strictly below 1 because
drop_off is always available with a finite logit, so the spectral radius of
Q is below 1, (I - Q) is invertible, and the particle is absorbed with
probability 1.  That holds at every position now, not just at start codons,
because the drop-off reference category is what bounds the self-loop.

Everything here is differentiable end to end, including the linear solve.
"""

import torch

# Absorbing state indices.  Ordering is part of the public contract: the fate
# vector returned by `solve_chain` is read in this order.
COMMIT_MAIN = 0
COMMIT_UORF = 1
DROP_OFF = 2
READ_THROUGH = 3
N_ABSORBING = 4

FATE_NAMES = ("commit_main", "commit_uORF", "drop_off", "read_through")


def build_QR(w, p, d, start_mask, is_main_start, r=None, uorf_return=None,
             drop_at_starts=True):
    """Assemble the transition matrices from per-position rates.

    Args:
        drop_at_starts: if False, drop-off is illegal at start codons, so
                       leaving at an AUG IS committing.  This is what makes
                       the commit/drop split identifiable from occupancy
                       alone -- see the identifiability note above.
        w:             (N,) step logit.  Higher -> the particle keeps scanning.
        p:             (N,) commit logit.  Only consulted at start positions.
        d:             (N,) dwell logit, at EVERY position.  Sets the
                       self-loop Q[j, j], i.e. residence time 1 / (1 - q),
                       i.e. the inverse scanning velocity.
        start_mask:    (N,) bool, True where an AUG (or near-cognate) begins.
                       Flags the *first* nucleotide of the codon.
        is_main_start: (N,) bool, True at the single annotated main AUG.
        r:             (N,) reinitiation logit.  sigmoid(r_j) is the fraction
                       of walkers initiating at uAUG j that resume scanning.
                       None switches reinitiation off entirely.
        uorf_return:   (N,) int64, the position scanning resumes at, or -1
                       where there is no return edge.  None = off.

    Returns:
        Q:   (N, N)  transient -> transient
        R:   (N, 4)  transient -> absorbing
        aux: per-position uORF initiation and reinitiation mass, for the
             caller to integrate against the visit profile

    The drop-off logit is pinned at 0 and acts as the reference category, so w,
    p and d are read as log-odds against dropping off.  Each row is a softmax
    over exactly the moves that are legal at that position, which makes the
    row sums come out at 1 by construction rather than by renormalisation.

    Q and R are built in float64 whatever dtype the logits arrive in.  The
    quantity the solve actually depends on is 1 - P_stay, and a 40S dwelling
    hard at a start codon drives P_stay towards 1: at P_stay = 0.9999, float32
    holds only ~3 correct digits of the difference, so the fundamental matrix
    comes out ~0.1% wrong and the fates stop summing to 1.  Catastrophic
    cancellation, and it happens here rather than in the solve.  These
    matrices are one per transcript and a few hundred wide at most, so double
    precision is free.  model/scanner.py casts the outputs back.
    """
    N = w.shape[0]
    assert w.shape == p.shape == d.shape == (N,), "w, p, d must be per-position scalars"
    assert start_mask.shape == (N,) and is_main_start.shape == (N,)
    assert not bool((is_main_start & ~start_mask).any()), \
        "the main start must also be flagged in start_mask"

    w, p, d = w.double(), p.double(), d.double()
    dtype, device = w.dtype, w.device

    # Legality masks, as additive -inf terms on the logits.  Row j always has
    # drop_off available; step needs a j+1 to step into; commit and stay both
    # need an AUG.
    can_step = torch.ones(N, dtype=torch.bool, device=device)
    can_step[N - 1] = False

    # Positions strictly downstream of the main AUG.  The only way to be
    # there is to have leaked through it.
    after_main = (torch.cumsum(is_main_start.to(torch.int64), dim=0) > 0) & ~is_main_start

    # Commit is shut off downstream of the main AUG.  When the states run on
    # into the CDS those positions model a leaked 40S still scanning, and an
    # AUG there would be a downstream initiation event -- real biology (it
    # makes N-terminally truncated protein) but a different fate needing its
    # own absorbing state and its own evidence.  Leaving it open would route
    # that mass into commit_uORF and mislabel it.
    can_commit = start_mask & ~after_main

    neg_inf = torch.finfo(dtype).min
    step_logit = torch.where(can_step, w, torch.full_like(w, neg_inf))
    commit_logit = torch.where(can_commit, p, torch.full_like(p, neg_inf))
    stay_logit = d                       # a velocity is defined everywhere
    drop_logit = torch.zeros(N, dtype=dtype, device=device)
    if not drop_at_starts:
        # Leaving at an AUG is committing.  Closing this one edge is what
        # makes the fates readable off the profile alone; see above.  Note
        # w, p and d are log-odds against drop-off, so where drop is closed
        # they are read against each other instead -- the softmax handles it,
        # but it means those positions have one fewer reference point.
        drop_logit = torch.where(can_commit, torch.full_like(drop_logit, neg_inf),
                                 drop_logit)

    # (N, 4) softmax over [step, commit, stay, drop] -> row-stochastic by
    # construction.
    probs = torch.softmax(
        torch.stack([step_logit, commit_logit, stay_logit, drop_logit], dim=1), dim=1)
    P_step, P_commit, P_stay, P_drop = probs[:, 0], probs[:, 1], probs[:, 2], probs[:, 3]

    # Q: self-loops on the diagonal, j -> j+1 on the super-diagonal.
    idx = torch.arange(N, device=device)
    Q = torch.zeros(N, N, dtype=dtype, device=device)
    Q[idx, idx] = P_stay
    if N > 1:
        Q[idx[:-1], idx[1:]] = P_step[:-1]

    # R: commit mass is routed to main or uORF by is_main_start; a position
    # that is not a start contributes zero commit mass anyway.  Exit mass is
    # routed to drop_off or read_through by whether the walker has already
    # passed the main AUG.
    after = after_main.to(dtype)
    main = is_main_start.to(dtype)

    # P_commit at a uAUG is the probability of INITIATING there.  Splitting
    # that mass is what reinitiation means; see the note at the top of the
    # file.  With r or uorf_return absent the split is degenerate and every
    # uORF initiation is absorbed, which is the pre-reinitiation behaviour.
    P_init_uorf = P_commit * (1.0 - main)
    if r is not None and uorf_return is not None:
        has_return = uorf_return >= 0
        reinit_frac = torch.sigmoid(r.double()) * has_return.to(dtype)
    else:
        has_return = torch.zeros(N, dtype=torch.bool, device=device)
        reinit_frac = torch.zeros(N, dtype=dtype, device=device)
    P_reinit = P_init_uorf * reinit_frac

    R = torch.zeros(N, N_ABSORBING, dtype=dtype, device=device)
    R[:, COMMIT_MAIN] = P_commit * main
    R[:, COMMIT_UORF] = P_init_uorf - P_reinit
    R[:, DROP_OFF] = P_drop * (1.0 - after)
    R[:, READ_THROUGH] = P_drop * after

    # The re-entry edge.  This is the whole of reinitiation: the returning
    # walker goes back into Q rather than into R, so solve_chain counts its
    # visits downstream and it can go on to reach the main AUG.
    src = torch.nonzero(has_return).flatten()
    if src.numel():
        dst = uorf_return[src].long()
        assert bool((dst > src).all()), \
            "a uORF return must land downstream of its own start codon"
        assert bool((dst < N).all()), "uORF return index outside the state space"
        Q.index_put_((src, dst), P_reinit[src], accumulate=True)

    # Sanity asserts, not a test suite: every transient row is a distribution,
    # and no self-loop is certain (that is what keeps I - Q invertible).
    row_sums = Q.sum(dim=1) + R.sum(dim=1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), \
        f"transient rows must sum to 1, got range [{row_sums.min():.6f}, {row_sums.max():.6f}]"
    assert bool((P_stay < 1.0).all()), "a self-loop with probability 1 would trap the particle"
    assert bool((torch.triu(Q) == Q).all()), \
        "Q must stay upper triangular; a backward return edge would allow cycles"

    return Q, R, {"p_init_uorf": P_init_uorf, "p_reinit": P_reinit}


def solve_chain(Q, R, start_state=0):
    """Fundamental-matrix solve.

    The fundamental matrix N = (I - Q)^-1 has N[i, j] = expected number of
    visits to j before absorption, starting from i.  Starting the particle at
    the cap means reading row `start_state`.

    N @ R gives absorption probabilities: (N @ R)[i, a] = P(absorbed in a |
    started in i).

    We get N from torch.linalg.solve(I - Q, I) rather than torch.linalg.inv --
    same answer, better conditioned, and the gradient flows through the solve.

    Q and R arrive in float64 from build_QR, which is where the precision
    actually matters -- see the note there.  A self-loop of probability q also
    makes (I - Q) ill-conditioned by a factor 1 / (1 - q), so the solve wants
    the headroom too.

    Returns:
        profile: (N,) expected visits per position from the cap.  This is the
                 predicted occupancy *shape*; it is not a probability vector
                 and does not sum to 1.
        fates:   (4,) absorption probabilities, indexed by COMMIT_* /
                 DROP_OFF / READ_THROUGH.
    """
    n = Q.shape[0]
    I = torch.eye(n, dtype=Q.dtype, device=Q.device)

    fundamental = torch.linalg.solve(I - Q, I)

    profile = fundamental[start_state]
    fates = (fundamental @ R)[start_state]

    assert torch.all(profile >= -1e-6), "expected visit counts cannot be negative"
    assert torch.allclose(fates.sum(), torch.ones_like(fates.sum()), atol=1e-4), \
        f"fates must sum to 1, got {fates.detach().sum().item():.6f}"

    return profile, fates


def run_chain(w, p, d, start_mask, is_main_start, r=None, uorf_return=None,
              return_aux=False, drop_at_starts=True):
    """build_QR + solve_chain, the usual entry point.

    return_aux adds a third return value with two integrated diagnostics:

        uorf_init_flux  expected uORF INITIATION events per walker, whether or
                        not the walker came back.  commit_uORF counts only the
                        ones that did not, so this is the number to watch when
                        asking whether the uORF branch is doing anything at
                        all.
        reinit_flux     expected reinitiation events per walker.
    """
    Q, R, aux = build_QR(w, p, d, start_mask, is_main_start,
                         r=r, uorf_return=uorf_return,
                         drop_at_starts=drop_at_starts)
    profile, fates = solve_chain(Q, R)
    if not return_aux:
        return profile, fates
    return profile, fates, {
        "uorf_init_flux": (profile * aux["p_init_uorf"]).sum(),
        "reinit_flux": (profile * aux["p_reinit"]).sum(),
    }
