"""Does recovered 40S information explain reporter translation beyond sequence?

    python -m analysis.recoverability.08_downstream_utility [--targets MeanYFP,TEff]

RUN ONLY AFTER THE RECOVERABILITY RESULTS ARE FROZEN.  Nothing here feeds
back into model choice: the 40S model is picked from effects/nested_effects.tsv
(highest minimum cross-replicate held-out Delta-LL; within-replicate for
single-replicate yeast), never from an MPRA number.

40S REPRESENTATION.  The chosen multinomial model is refitted on 40S data with
the MPRA leader's chromosome fold held out (yeast), so neither labels nor
sequences leak.  It is applied to the reporter's actual sequence (leader +
AUG + reporter CDS start), folded the same way as the panel, and summarised:

    p_start        predicted share of 40S footprints in the main start window
    p_start_nopos  the same with positional terms removed (feature-only)
    p_uaug         share at upstream start windows
    p_first30, p_last30   share in the first / last 30 nt of the leader
    entropy        Shannon entropy of the predicted profile / log(length)

SEQUENCE BASELINES (no 40S information):
    simple   log length, uAUG count, main-AUG -3 and +4 one-hots, Kozak
             class, GC, MFE per nt
    kmer     simple + 3-mer frequencies (64), ridge

COMPARED, each by held-out R^2 with a leader-bootstrap CI on Delta-R^2:
    seq                     kmer baseline
    40S                     representation only
    seq+40S                 concatenated, ridge
    late_fusion             stacking of seq and 40S predictions (inner CV)
    residual                seq prediction + 40S model of the seq residual

Yeast Akirtava uses the yeast chromosome folds.  Human Sample 2019 (MRL) has
no genomic coordinates; it uses a random 5-fold split, the GSE174329 40S
model fitted on all chromosomes, and is reported separately as secondary.

Writes downstream/downstream_utility.tsv.
"""

import argparse
import csv
import importlib
import os
import sys

import numpy as np
from sklearn.linear_model import RidgeCV

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C
from analysis.recoverability import likelihood as L
from analysis.recoverability import splits
from analysis.recoverability.features import FeatureTable, kozak_class
from data.features import _fold_tracks

EFF = importlib.import_module("analysis.recoverability.03_nested_effects")

# Reporter context.  Yeast: Akirtava construct, YFP (yEVenus) N-terminus.
# Human: Sample et al. 2019 eGFP N-terminus.
CDS_START = {"yeast": "TCTAAAGGTGAAGAATTATTCACTGGTGTT",
             "human": "GTGAGCAAGGGCGAGGAGCTGTTCACCGGG"}
ALPHAS = np.logspace(-3, 4, 15)


def choose_model(key, single_rep):
    rows = [r for r in C.read_tsv(C.results_path("effects", "nested_effects.tsv"))
            if r.get("panel", r["dataset"]) == key and r["metric"] == "heldout_dll_read_weighted"
            and r["model"] not in ("S5",)]
    best = {}
    for r in rows:
        A, B = r["replicate"].split("->")
        if (A != B) == single_rep:
            continue
        k = (r["model"], int(r["bin_width"]))
        best[k] = min(best.get(k, np.inf), float(r["value"]))
    (model, b), v = max(best.items(), key=lambda kv: kv[1])
    return model, b, v


def _fold_reporter(args):
    seq, w = args
    return _fold_tracks(seq, w)


def reporter_panel(rows, species):
    """Reporter sequences folded exactly like the panel, in parallel."""
    import multiprocessing as mp
    seqs = [r["seq"] + "ATG" + CDS_START[species] for r in rows]
    with mp.get_context("spawn").Pool(C.default_workers(gb_per_worker=0.3)) as pool:
        folded = pool.map(_fold_reporter, [(s_, C.FOLD_WINDOW) for s_ in seqs], chunksize=50)
    return [{"transcript_id": r["id"], "chrom": r.get("chrom", "NA"), "seq": s_,
             "msi": len(r["seq"]), "paired": pr, "local_dG": dg, "counts": {}}
            for r, s_, (pr, dg) in zip(rows, seqs, folded)]


def represent(fit, model, b, ft_mpra, n):
    blocks = L.MODELS[model]
    out = np.zeros((n, 6))
    prob = L.Problem(ft_mpra, np.arange(n), b, blocks)
    beta = prob.init_beta(fit.as_dict())
    logp = L.nucleotide_logp(prob, beta)
    nopos = beta.copy()
    for k in ("pos_aug", "pos_five"):
        nopos[prob.slices[k]] = 0.0
    logp0 = L.nucleotide_logp(prob, nopos)
    p, p0 = np.exp(logp), np.exp(logp0)
    pos = prob.pos
    dA, d5 = ft_mpra.dAUG[pos], ft_mpra.d5[pos]
    startwin = ft_mpra.cat["start"][pos, 0] >= 0
    main = np.abs(dA) <= 2
    L_tx = np.bincount(prob.seg)
    for t in range(n):
        m = prob.seg == t
        out[t] = [p[m][main[m]].sum(), p0[m][main[m]].sum(),
                  p[m][startwin[m] & ~main[m]].sum(),
                  p[m][d5[m] < 30].sum(), p[m][(dA[m] < 30) & (dA[m] >= 0)].sum(),
                  float(-(p[m] * np.log(p[m] + 1e-300)).sum() / np.log(L_tx[t]))]
    return out


def seq_features(rows, species):
    """Handcrafted features and 3-mer frequencies.  No 40S information."""
    import RNA
    p4 = "ACGT".find(CDS_START[species][0])     # +4 is fixed by the reporter
    simple, kmer = [], []
    for r in rows:
        s = r["seq"].upper()
        L_ = len(s)
        uaug = sum(1 for i in range(L_ - 2) if s[i:i + 3] == "ATG")
        m3 = "ACGT".find(s[-3]) if L_ >= 3 else -1
        oh3 = [float(m3 == i) for i in range(4)]
        kz = kozak_class(m3, p4) if m3 >= 0 else 1
        mfe = RNA.fold(s.replace("T", "U"))[1]
        gc = (s.count("G") + s.count("C")) / max(L_, 1)
        f = [np.log(L_), uaug, gc, mfe / max(L_, 1)] + oh3 + [float(kz == i) for i in range(3)]
        simple.append(f)
        km = np.zeros(64)
        for i in range(L_ - 2):
            w = s[i:i + 3]
            if set(w) <= set("ACGT"):
                km[int("".join(str("ACGT".index(c)) for c in w), 4)] += 1
        kmer.append(f + list(km / max(L_ - 2, 1)))
    return np.array(simple), np.array(kmer)


def r2(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def evaluate(X_seq, X_40, y, groups, seed=0):
    """Out-of-fold predictions for every variant, then R^2 + bootstrap CI."""
    preds = {k: np.zeros(len(y)) for k in ("seq", "40S", "seq+40S", "late_fusion", "residual")}
    for g in np.unique(groups):
        te, tr = groups == g, groups != g
        mu, sd = X_seq[tr].mean(0), X_seq[tr].std(0) + 1e-9
        m4, s4 = X_40[tr].mean(0), X_40[tr].std(0) + 1e-9
        S, F = (X_seq - mu) / sd, (X_40 - m4) / s4
        ms = RidgeCV(alphas=ALPHAS).fit(S[tr], y[tr])
        mf = RidgeCV(alphas=ALPHAS).fit(F[tr], y[tr])
        mb = RidgeCV(alphas=ALPHAS).fit(np.hstack([S, F])[tr], y[tr])
        preds["seq"][te] = ms.predict(S[te])
        preds["40S"][te] = mf.predict(F[te])
        preds["seq+40S"][te] = mb.predict(np.hstack([S, F])[te])
        # late fusion: inner out-of-fold predictions on the training set
        inner = np.zeros((tr.sum(), 2))
        idx = np.nonzero(tr)[0]
        folds = np.random.default_rng(seed).integers(0, 5, len(idx))
        for f_ in range(5):
            a, bb = idx[folds != f_], idx[folds == f_]
            inner[folds == f_, 0] = RidgeCV(alphas=ALPHAS).fit(S[a], y[a]).predict(S[bb])
            inner[folds == f_, 1] = RidgeCV(alphas=ALPHAS).fit(F[a], y[a]).predict(F[bb])
        st = RidgeCV(alphas=ALPHAS).fit(inner, y[tr])
        preds["late_fusion"][te] = st.predict(np.column_stack([preds["seq"][te], preds["40S"][te]]))
        res = y[tr] - ms.predict(S[tr])
        mr = RidgeCV(alphas=ALPHAS).fit(F[tr], res)
        preds["residual"][te] = preds["seq"][te] + mr.predict(F[te])
    rng = np.random.default_rng(seed)
    boot = rng.integers(0, len(y), (500, len(y)))
    out = {}
    for k, p in preds.items():
        d = np.array([r2(y[b], p[b]) - r2(y[b], preds["seq"][b]) for b in boot])
        out[k] = {"r2": r2(y, p), "delta_r2_vs_seq": r2(y, p) - r2(y, preds["seq"]),
                  "ci_lo": float(np.percentile(d, 2.5)), "ci_hi": float(np.percentile(d, 97.5))}
    return out


def load_akirtava():
    rows = []
    with open(os.path.join(C.ROOT, "eval", "akirtava_scer_wt.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = (r["utr_seq"] or "").upper()
            if len(s) < C.MIN_LEADER_LEN or set(s) - set("ACGT") or len(s) > L.POS_MAX:
                continue
            rows.append({"id": r["id"], "chrom": r["chrom"], "seq": s,
                         "MeanYFP": r["MeanYFP"], "TEff": r["TEff"]})
    return rows


def load_sample2019():
    rows = []
    with open(os.path.join(C.ROOT, "eval", "sample2019.tsv")) as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            s = r["utr_seq"].upper()
            if len(s) >= C.MIN_LEADER_LEN:
                rows.append({"id": r["id"], "chrom": "NA", "seq": s, "rl": r["rl"]})
    return rows


def run_yeast(targets):
    key = "archer_yeast"
    model, b, v = choose_model(key, single_rep=True)
    print("[down] yeast 40S model {} at {} nt (held-out dLL {:.4f})".format(model, b, v))
    recs, meta, reps, reads, ft, y = EFF.prepare(key)
    rows = load_akirtava()
    panel = reporter_panel(rows, "yeast")
    fo = splits.fold_of("yeast")
    groups = np.array([fo.get(r["chrom"], -1) for r in rows])
    X40 = np.zeros((len(rows), 6))
    for k in range(splits.load("yeast")["n_folds"]):
        tr_mask, _ = splits.train_test(ft.chrom, "yeast", k)
        tr = np.nonzero(tr_mask)[0]
        prob = L.Problem(ft, tr, b, L.MODELS[model])
        fit = L.fit(prob, prob.bin_counts(y[reps[0]]), prior_sd=EFF.PRIOR_SD)
        sel = np.nonzero(groups == k)[0]
        if len(sel):
            sub = FeatureTable([panel[i] for i in sel])
            X40[sel] = represent(fit, model, b, sub, len(sel))
    _, Xk = seq_features(rows, "yeast")
    out = []
    for target in targets:
        yv = np.array([float(r[target]) if r[target] not in ("", None) else np.nan for r in rows])
        ok = np.isfinite(yv) & (groups >= 0)
        res = evaluate(Xk[ok], X40[ok], yv[ok], groups[ok])
        for k, m in res.items():
            out.append({"dataset": "akirtava_yeast", "replicate": "", "split": "chrom_folds",
                        "seed": 0, "bin_width": b, "model": k, "n_transcripts": int(ok.sum()),
                        "metric": "heldout_r2", "value": m["r2"], "target": target,
                        "delta_r2_vs_seq": m["delta_r2_vs_seq"], "ci_lo": m["ci_lo"],
                        "ci_hi": m["ci_hi"], "representation_model": model})
            print("  {:<8} {:<12} R2 {:.4f}  dR2 {:+.4f} [{:+.4f},{:+.4f}]".format(
                target, k, m["r2"], m["delta_r2_vs_seq"], m["ci_lo"], m["ci_hi"]))
    return out


def run_human():
    key = "gse174329_eif3b"
    model, b, v = choose_model(key, single_rep=False)
    print("[down] human 40S model {} at {} nt (cross-replicate dLL {:.4f})".format(model, b, v))
    recs, meta, reps, reads, ft, y = EFF.prepare(key)
    prob = L.Problem(ft, np.arange(len(recs)), b, L.MODELS[model])
    fit = L.fit(prob, prob.bin_counts(y[reps[0]]), prior_sd=EFF.PRIOR_SD)
    rows = load_sample2019()
    panel = reporter_panel(rows, "human")
    X40 = represent(fit, model, b, FeatureTable(panel), len(rows))
    _, Xk = seq_features(rows, "human")
    groups = np.random.default_rng(0).integers(0, 5, len(rows))
    yv = np.array([float(r["rl"]) for r in rows])
    res = evaluate(Xk, X40, yv, groups)
    out = []
    for k, m in res.items():
        out.append({"dataset": "sample2019_human", "replicate": "", "split": "random_5fold",
                    "seed": 0, "bin_width": b, "model": k, "n_transcripts": len(rows),
                    "metric": "heldout_r2", "value": m["r2"], "target": "rl",
                    "delta_r2_vs_seq": m["delta_r2_vs_seq"], "ci_lo": m["ci_lo"],
                    "ci_hi": m["ci_hi"], "representation_model": model, "secondary": True})
        print("  rl       {:<12} R2 {:.4f}  dR2 {:+.4f} [{:+.4f},{:+.4f}]".format(
            k, m["r2"], m["delta_r2_vs_seq"], m["ci_lo"], m["ci_hi"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="MeanYFP,TEff")
    ap.add_argument("--skip-human", action="store_true")
    args = ap.parse_args()
    C.ensure_layout()
    rows = run_yeast(args.targets.split(","))
    if not args.skip_human:
        rows += run_human()
    C.write_tsv(rows, C.results_path("downstream", "downstream_utility.tsv"))
    print("[down] done")


if __name__ == "__main__":
    main()
