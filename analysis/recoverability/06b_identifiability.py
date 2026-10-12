"""Mechanistic identifiability of the scanning chain (Step 11).

    python -m analysis.recoverability.06b_identifiability

Separate from biological-effect recoverability: these are properties of the
MODEL + OBSERVATION, not of any dataset.  For a latent parameter vector
theta of model/chain.py and an observation (binned 40S occupancy, optionally
80S counts), the expected Fisher information is

    occupancy:  J = N * sum_b (dq_b/dtheta)(dq_b/dtheta)^T / q_b
    80S count:  J = (dmu/dtheta)(dmu/dtheta)^T / mu        (Poisson)

with q the bin probabilities of the chain's normalised expected-visits
profile and N reads per transcript.  Derivatives come from autograd through
the exact chain solve.  J singular (smallest eigenvalue ~0 relative to the
largest) = structurally unidentifiable at ANY depth; otherwise J^-1 gives
standard errors and the correlation between estimates, as functions of
depth N and resolution b.

Experiments
  E1 commit vs drop-off   theta = share of the main-AUG exit mass that
                          commits.  Occupancy alone; + 80S with known scale
                          rho; + 80S with unknown rho (theta = (share, rho)).
  E2 blur vs leakage      theta = (P(step past the AUG), blur w).  Window
                          ending at AUG+2 (the original design) vs 30 CDS
                          states; resolution 1, 3, 5 nt.
  E3 reinit vs direct     theta = (uORF commit logit, reinit logit).  Full
                          occupancy profile vs main-AUG output alone.
  E4 global loading       theta = rho.  Conditional multinomial occupancy.

Also writes likelihood surfaces (expected log-likelihood, i.e. the KL
surface around the truth) for E1-E3 for Figure 5.

Writes chain/identifiability.tsv and chain/identifiability_surfaces.tsv.
"""

import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from model.chain import COMMIT_MAIN, run_chain

torch.set_default_dtype(torch.float64)

LEADER = 100
DEPTHS = (20, 50, 100, 200, 500, 2000, 10000)
BASE_STEP, BASE_STAY = 0.80, 0.199       # per nt away from starts (drop 0.001)


def logits_from_probs(step, commit, stay, drop):
    """Probabilities of [step, commit, stay, drop] -> (w, p, d) logits vs drop."""
    return torch.log(step / drop), torch.log(commit / drop), torch.log(stay / drop)


def template(n_cds=0, uorf=None):
    """Masks for a leader of LEADER nt + AUG (+ n_cds), optional uORF (start, stop)."""
    n = LEADER + 3 + n_cds
    start = torch.zeros(n, dtype=torch.bool)
    main = torch.zeros(n, dtype=torch.bool)
    start[LEADER] = main[LEADER] = True
    ret = torch.full((n,), -1, dtype=torch.int64)
    if uorf is not None:
        s, stop = uorf
        start[s] = True
        ret[s] = stop + 3
    return n, start, main, ret


def chain_profile(n, start, main, aug_probs, uorf_logits=None, ret=None,
                  leak_drop=None):
    """Profile + fates.  aug_probs = (step, commit, stay, drop) at the main AUG."""
    step = torch.full((n,), BASE_STEP)
    stay = torch.full((n,), BASE_STAY)
    drop = 1.0 - step - stay
    commit = torch.full((n,), 1e-12)
    w, p, d = logits_from_probs(step, commit, stay, drop)
    a_step, a_commit, a_stay, a_drop = aug_probs
    wa, pa, da = logits_from_probs(a_step, a_commit, a_stay, a_drop)
    idx = LEADER
    w = torch.cat([w[:idx], wa.reshape(1), w[idx + 1:]])
    p = torch.cat([p[:idx], pa.reshape(1), p[idx + 1:]])
    d = torch.cat([d[:idx], da.reshape(1), d[idx + 1:]])
    r = None
    if uorf_logits is not None:
        pu, ru = uorf_logits
        s = int(torch.nonzero(start & ~main)[0])
        p = torch.cat([p[:s], pu.reshape(1), p[s + 1:]])
        r = torch.zeros(n)
        r = torch.cat([r[:s], ru.reshape(1), r[s + 1:]])
    return run_chain(w, p, d, start, main, r=r, uorf_return=ret if uorf_logits is not None else None)


def blur(profile, wb):
    k = torch.stack([wb, 1 - 2 * wb, wb]).view(1, 1, 3)
    return torch.nn.functional.conv1d(profile.view(1, 1, -1), k, padding=1).view(-1)


def bins_of(q, b):
    """Nucleotide probabilities -> AUG-anchored bins (bin starts at dAUG = -2)."""
    n = q.shape[0]
    d = LEADER - torch.arange(n)
    key = torch.div(d + 2 + 60000, b, rounding_mode="floor")
    _, inv = torch.unique(-key, return_inverse=True)
    return torch.zeros(int(inv.max()) + 1).index_add(0, inv, q)


def fisher_occupancy(fn, theta, N, b):
    """fn(theta) -> nucleotide profile (unnormalised).  -> J (k x k)."""
    def q_of(t):
        prof = fn(t)
        return bins_of(prof / prof.sum(), b)
    q = q_of(theta)
    Jq = torch.autograd.functional.jacobian(q_of, theta)          # (bins, k)
    keep = q > 1e-15
    return N * (Jq[keep].T @ (Jq[keep] / q[keep, None]))


def summarise(J):
    J = J.detach().numpy()
    ev = np.linalg.eigvalsh(J)
    rel = ev[0] / ev[-1] if ev[-1] > 0 else 0.0
    out = {"min_eig": float(ev[0]), "max_eig": float(ev[-1]), "rel_min_eig": float(rel)}
    if rel > 1e-10:
        cov = np.linalg.inv(J)
        se = np.sqrt(np.diag(cov))
        out["se"] = se.tolist()
        if len(se) > 1:
            out["corr"] = float(cov[0, 1] / (se[0] * se[1]))
    else:
        out["se"] = [float("inf")] * J.shape[0]
        out["corr"] = float("nan")
    return out


def rows_from(exp, cond, N, b, s, names, base=None):
    r = []
    common = {"dataset": "theory", "replicate": "", "split": "", "seed": "",
              "bin_width": b, "model": exp, "condition": cond, "reads_per_transcript": N}
    r.append(dict(common, metric="rel_min_eigenvalue", value=s["rel_min_eig"]))
    # Information per read along the worst direction.  Exactly 0 in theory
    # for a structural non-identifiability; float noise (~1e-30) in practice.
    per_read = s["min_eig"] / float(N) if N else s["min_eig"]
    r.append(dict(common, metric="min_information_per_read", value=per_read))
    r.append(dict(common, metric="identifiable",
                  value=int(per_read > 1e-10 and s["rel_min_eig"] > 1e-10)))
    for nm, se in zip(names, s["se"]):
        r.append(dict(common, metric="se_" + nm, value=se))
    if "corr" in s:
        r.append(dict(common, metric="corr_" + "_".join(names), value=s["corr"]))
    return r


def e1_commit_drop(rows, surf):
    n, start, main, _ = template()
    exit_mass, a_step, a_stay = torch.tensor(0.4), torch.tensor(0.2), torch.tensor(0.4)

    def fn(t):
        share = torch.sigmoid(t[0])
        prof, _ = chain_profile(n, start, main, (a_step, share * exit_mass, a_stay,
                                                 (1 - share) * exit_mass))
        return prof

    def commit(t):
        share = torch.sigmoid(t[0])
        _, f = chain_profile(n, start, main, (a_step, share * exit_mass, a_stay,
                                              (1 - share) * exit_mass))
        return f[COMMIT_MAIN]

    theta = torch.tensor([0.5])
    # Numerical check of exact invariance of the profile.
    profs = [fn(torch.tensor([v])) for v in (-3.0, 0.0, 3.0)]
    maxdiff = max(float((a / a.sum() - profs[0] / profs[0].sum()).abs().max()) for a in profs)
    rows.append({"dataset": "theory", "model": "E1_commit_vs_drop", "condition": "check",
                 "metric": "max_abs_profile_difference_across_shares", "value": maxdiff})
    abundance = 1.0
    rho = torch.tensor(2.0)
    for N in DEPTHS:
        Jocc = fisher_occupancy(fn, theta, N, 1)
        rows += rows_from("E1_commit_vs_drop", "occupancy_only", N, 1, summarise(Jocc), ["share"])
        # + 80S counts, mu = rho * commit * abundance * N_80 (N_80 = N reads scale)
        g = torch.autograd.functional.jacobian(lambda t: rho * commit(t) * abundance * N, theta)
        mu = rho * commit(theta) * abundance * N
        J80 = g.reshape(1, -1).T @ g.reshape(1, -1) / mu
        rows += rows_from("E1_commit_vs_drop", "occupancy+80S_rho_known", N, 1,
                          summarise(Jocc + J80), ["share"])
        # unknown rho: theta2 = (share logit, log rho)
        th2 = torch.tensor([0.5, float(torch.log(rho))])

        def mu2(t):
            return torch.exp(t[1]) * commit(t[:1]) * abundance * N
        g2 = torch.autograd.functional.jacobian(mu2, th2).reshape(1, -1)
        J2 = torch.zeros(2, 2)
        J2[0, 0] = Jocc[0, 0]
        J2 = J2 + g2.T @ g2 / mu2(th2)
        rows += rows_from("E1_commit_vs_drop", "occupancy+80S_rho_unknown", N, 1,
                          summarise(J2), ["share", "log_rho"])
    # Surface: expected LL vs share, occupancy only and with 80S (N = 500).
    q0 = fn(theta)
    q0 = q0 / q0.sum()
    mu0 = rho * commit(theta) * 500
    for v in np.linspace(-3, 4, 36):
        t = torch.tensor([float(v)])
        q = fn(t)
        q = q / q.sum()
        ll_occ = float(500 * (q0 * torch.log(q)).sum())
        mu = rho * commit(t) * 500
        ll80 = float(mu0 * torch.log(mu) - mu)
        surf.append({"model": "E1_commit_vs_drop", "x_name": "share_logit", "x": v,
                     "y_name": "", "y": "", "exp_ll_occupancy": ll_occ,
                     "exp_ll_occupancy_80S": ll_occ + ll80})


def e2_blur_leak(rows, surf):
    for n_cds in (0, 30):
        n, start, main, _ = template(n_cds=n_cds)
        exit_mass = torch.tensor(0.3)

        def fn(t, n=n, start=start, main=main):
            leak = torch.sigmoid(t[0]) * 0.6           # P(step past AUG), stay fills the rest
            stay = 1 - exit_mass - leak
            prof, _ = chain_profile(n, start, main, (leak, exit_mass * 0.999, stay,
                                                     exit_mass * 0.001))
            return blur(prof, 0.33 * torch.sigmoid(t[1]))

        theta = torch.tensor([-1.0, -0.5])
        cond = "window_ends_AUG+2" if n_cds == 0 else "30_CDS_states"
        for b in (1, 3, 5):
            for N in DEPTHS:
                rows += rows_from("E2_blur_vs_leak", cond, N, b,
                                  summarise(fisher_occupancy(fn, theta, N, b)),
                                  ["leak_logit", "blur_logit"])
        q0 = fn(theta)
        q0 = q0 / q0.sum()
        for x in np.linspace(-3, 1.5, 19):
            for y in np.linspace(-3, 2, 19):
                q = fn(torch.tensor([float(x), float(y)]))
                q = q / q.sum()
                surf.append({"model": "E2_blur_vs_leak", "condition": cond,
                             "x_name": "leak_logit", "x": x, "y_name": "blur_logit", "y": y,
                             "exp_ll_occupancy": float(500 * (q0 * torch.log(q)).sum())})


def e3_reinit(rows, surf):
    n, start, main, ret = template(uorf=(30, 57))     # 9-codon uORF, stop at 57
    aug = (torch.tensor(0.1), torch.tensor(0.5), torch.tensor(0.3), torch.tensor(0.1))

    def run(t):
        return chain_profile(n, start, main, aug, uorf_logits=(t[0], t[1]), ret=ret)

    # Logits are against drop-off, and the step logit away from starts is
    # log(0.8/0.001) = 6.7, so a commit logit of 6.0 means ~27% of walkers
    # reaching the uAUG initiate there; reinit logit 0 = half come back.
    theta = torch.tensor([6.0, 0.0])
    for b in (1, 5, 10, 20):
        for N in DEPTHS:
            rows += rows_from("E3_reinit_vs_direct", "occupancy", N, b,
                              summarise(fisher_occupancy(lambda t: run(t)[0], theta, N, b)),
                              ["uorf_commit_logit", "reinit_logit"])
    # output only: one number (commit_main) for two parameters
    for N in DEPTHS:
        g = torch.autograd.functional.jacobian(lambda t: run(t)[1][COMMIT_MAIN] * N, theta).reshape(1, -1)
        mu = run(theta)[1][COMMIT_MAIN] * N
        rows += rows_from("E3_reinit_vs_direct", "main_AUG_output_only", N, "",
                          summarise(g.T @ g / mu), ["uorf_commit_logit", "reinit_logit"])
    q0 = run(theta)[0]
    q0 = q0 / q0.sum()
    for x in np.linspace(3, 9, 19):
        for y in np.linspace(-3, 3, 19):
            pr, f = run(torch.tensor([float(x), float(y)]))
            q = pr / pr.sum()
            surf.append({"model": "E3_reinit_vs_direct", "condition": "occupancy",
                         "x_name": "uorf_commit_logit", "x": x, "y_name": "reinit_logit", "y": y,
                         "exp_ll_occupancy": float(500 * (q0 * torch.log(q)).sum()),
                         "commit_main": float(f[COMMIT_MAIN])})


def e4_loading(rows):
    n, start, main, _ = template()
    aug = (torch.tensor(0.2), torch.tensor(0.3), torch.tensor(0.4), torch.tensor(0.1))

    def fn(t):
        prof, _ = chain_profile(n, start, main, aug)
        return torch.exp(t[0]) * prof         # loading scales every visit equally
    for N in DEPTHS:
        rows += rows_from("E4_global_loading", "conditional_multinomial", N, 1,
                          summarise(fisher_occupancy(fn, torch.tensor([0.0]), N, 1)), ["log_rho"])


def main():
    C.ensure_layout()
    rows, surf = [], []
    e1_commit_drop(rows, surf)
    e2_blur_leak(rows, surf)
    e3_reinit(rows, surf)
    e4_loading(rows)
    C.write_tsv(rows, C.results_path("chain", "identifiability.tsv"))
    C.write_tsv(surf, C.results_path("chain", "identifiability_surfaces.tsv"))
    for r in rows:
        if r.get("reads_per_transcript") in (500, None, "") and r["metric"] in (
                "rel_min_eigenvalue", "max_abs_profile_difference_across_shares") or \
                (r.get("reads_per_transcript") == 500 and r["metric"].startswith("corr")):
            print("{:<22} {:<28} b={:<3} {:<34} {}".format(
                r["model"], r.get("condition", ""), r.get("bin_width", ""), r["metric"],
                "{:.3g}".format(r["value"]) if isinstance(r["value"], float) else r["value"]))


if __name__ == "__main__":
    main()
