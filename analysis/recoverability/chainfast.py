"""Batched closed-form expected visits for the scanning chain without
reinitiation, for fitting many transcripts at once.

model/chain.py solves (I - Q)^-1 per transcript, which is exact for any Q
including reinitiation edges.  With reinitiation off, Q has only self-loops
and j -> j+1 steps, so the expected visits from the cap are

    V_j = prod_{k<j} P_step(k) / (1 - P_stay(k))  *  1 / (1 - P_stay(j))

i.e. a cumulative sum in log space -- O(N) and trivially batched over a flat
array of concatenated transcripts.  test_chainfast.py checks this against
model.chain.run_chain to float precision.

Moves per position are a softmax over the legal subset of
[step, commit, stay, drop] with drop as the zero-logit reference, exactly as
model/chain.build_QR: step is illegal at a transcript's last position, commit
only at start codons, drop illegal at start codons when drop_at_starts=False.
"""

import torch

NEG = -1e30


def log_visits(w, p, d, start, last, seg, drop_at_starts=True):
    """Flat per-position logits -> flat log expected visits.

    w, p, d   (P,) step / commit / dwell logits vs drop
    start     (P,) bool, commit legal here.  Must already be False downstream
              of the main AUG, as build_QR's can_commit is.
    last      (P,) bool, final position of its transcript (no step)
    seg       (P,) transcript index, positions contiguous and in order
    """
    neg = torch.full_like(w, NEG)
    step_l = torch.where(last, neg, w)
    commit_l = torch.where(start, p, neg)
    drop_l = torch.zeros_like(w)
    if not drop_at_starts:
        drop_l = torch.where(start, neg, drop_l)
    logits = torch.stack([step_l, commit_l, d, drop_l], dim=1)
    logp = torch.log_softmax(logits, dim=1)
    log_step, log_stay = logp[:, 0], logp[:, 2]
    log_leave = torch.log1p(-torch.exp(log_stay).clamp(max=1 - 1e-12))   # log(1 - P_stay)
    inc = torch.where(last, torch.zeros_like(w), log_step - log_leave)
    # exclusive cumulative sum within each transcript
    cs = torch.cumsum(inc, 0)
    first = torch.ones_like(seg, dtype=torch.bool)
    first[1:] = seg[1:] != seg[:-1]
    start_idx = torch.cummax(torch.where(first, torch.arange(len(seg), device=seg.device), torch.zeros_like(seg)), 0)[0]
    excl = cs - inc - (cs[start_idx] - inc[start_idx])
    return excl - log_leave
