"""Yeast TCP-seq: within-library evidence only (one SSU library).

    python -m analysis.recoverability.10_yeast_single_library_summary [--power-tag n50]

Yeast has ONE independent SSU library (SRR3458591).  Cross-replicate
recoverability is untestable, so this summary never uses the cross-replicate
statuses ("recoverable", "not_recoverable_at_depth", ...).  It reads only
frozen / completed outputs:

  effects/nested_effects.tsv     held-out chromosome rep1->rep1 rows (03)
  nulls/shuffle_nulls.tsv        matched positional nulls (04), 1/5/10/20 nt
  synthetic/power_curve_archer_yeast_multinomial_<tag>.tsv   (06)
  qc/qc_summary.tsv              observed median leader depth

Classification (held-out chromosome, within library):

  dataset_specific_support      CI > 0  AND  matched-null p <= 0.05
  fails_position_matched_null   CI > 0  BUT  matched-null p >  0.05
  no_within_library_support     CI includes 0
  held_out_support_no_null_defined   CI > 0, family has no matched null
                                     (combined models; reported, not classified)

Power qualifier: multinomial ("optimistic sampling-only") power of the SAME
within-library test (06's p_recovered_within) at the reference effect size,
at the simulated depth nearest the observed median.  It describes sampling
noise only; one library cannot estimate between-library overdispersion.

Writes missing_eval/yeast_single_library_summary.tsv.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis.recoverability import config as C

KEY = "archer_yeast"
FAMILIES = ("SW", "ST", "AUG", "UORF", "SW+ST+AUG")
SYNTH = {"SW": ("SW", "B"), "ST": ("ST", "C"), "AUG": ("AUG", "D")}
REFERENCE_SIZE = 1
POWER_OK = 0.8
STATEMENT = ("yeast has only one independent SSU library; cross-replicate recoverability "
             "is untestable. All evidence here is held-out-chromosome, within-library.")


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--power-tag", default="n50")
    args = ap.parse_args()
    C.ensure_layout()
    eff = [r for r in C.read_tsv(C.results_path("effects", "nested_effects.tsv"))
           if r.get("panel", r["dataset"]) == KEY]
    nul = [r for r in C.read_tsv(C.results_path("nulls", "shuffle_nulls.tsv")) if r["dataset"] == KEY]
    pw_path = C.results_path("synthetic", "power_curve_{}_multinomial{}.tsv".format(
        KEY, "_" + args.power_tag if args.power_tag else ""))
    pw = C.read_tsv(pw_path) if os.path.exists(pw_path) else []
    qc = [_f(r["value"]) for r in C.read_tsv(C.results_path("qc", "qc_summary.tsv"))
          if r["dataset"] == KEY and r["metric"] == "median_reads_per_leader_ge5"]
    obs_depth = float(np.median(qc)) if qc else np.nan

    def pick(model, b, metric, split="chrom_heldout", rep="rep1->rep1"):
        for r in eff:
            if (r["model"] == model and str(r["bin_width"]) == str(b) and r["metric"] == metric
                    and r["split"] == split and r["replicate"] == rep):
                return r
        return None

    def null(model, b, metric):
        for r in nul:
            if r["model"] == model and str(r["bin_width"]) == str(b) and r["metric"] == metric:
                return _f(r["value"])
        return np.nan

    def power(fam):
        if fam not in SYNTH or not pw:
            return np.nan, np.nan, np.nan, np.nan
        sf, truth = SYNTH[fam]
        pts = {}
        for r in pw:
            if (r["model"] == sf and r["truth"] == truth and r["metric"] == "p_recovered_within"
                    and int(_f(r["size_index"])) == REFERENCE_SIZE):
                pts[int(_f(r["target_median_depth"]))] = (_f(r["value"]), _f(r.get("p_lo")),
                                                         _f(r.get("p_hi")))
        if not pts or not np.isfinite(obs_depth):
            return np.nan, np.nan, np.nan, np.nan
        ds = np.array(sorted(pts))
        near = int(ds[np.argmin(np.abs(np.log(ds) - np.log(obs_depth)))])
        return pts[near] + (near,)

    rows = []
    for fam in FAMILIES:
        for b in C.MODEL_BINS:
            h = pick(fam, b, "heldout_dll_read_weighted")
            if h is None:
                sys.exit("\nMissing frozen nested-effect row for {} {} nt; stopping rather than "
                         "refitting.\n".format(fam, b))
            e = pick(fam, b, "heldout_dll_equal_transcript")
            tr = pick(fam, b, "train_dll_per_read", split="chrom_train_mean", rep="rep1")
            lo = _f(h["ci_lo"])
            p, z = null(fam, b, "empirical_p"), null(fam, b, "z")
            if lo > 0 and np.isfinite(p):
                status = "dataset_specific_support" if p <= 0.05 else "fails_position_matched_null"
            elif lo > 0:
                status = "held_out_support_no_null_defined"
            else:
                status = "no_within_library_support"
            pv, plo, phi, pdepth = power(fam)
            pstat = ("no_synthetic_counterpart" if not np.isfinite(pv) else
                     "adequate_sampling_power" if pv >= POWER_OK else "low_sampling_power")
            train = _f(tr["value"]) if tr else np.nan
            rows.append({
                "dataset": KEY, "replicate": "rep1->rep1", "split": "chrom_heldout",
                "bin_width": b, "model": fam, "family": fam,
                "heldout_dll_read_weighted": _f(h["value"]), "ci_lo": lo, "ci_hi": _f(h["ci_hi"]),
                "heldout_dll_equal_transcript": _f(e["value"]) if e else np.nan,
                "equal_transcript_ci_lo": _f(e["ci_lo"]) if e else np.nan,
                "equal_transcript_ci_hi": _f(e["ci_hi"]) if e else np.nan,
                "train_dll_per_read": train,
                "train_minus_heldout_gap": train - _f(h["value"]),
                "null_real": null(fam, b, "real_dll"), "null_mean": null(fam, b, "null_mean"),
                "null_sd": null(fam, b, "null_sd"), "null_p": p, "null_z": z,
                "status": status,
                "optimistic_power_at_observed_depth": pv, "power_ci_lo": plo, "power_ci_hi": phi,
                "power_depth_used": pdepth, "observed_median_depth": obs_depth,
                "power_status": pstat, "power_noise": "multinomial (sampling-only, optimistic)",
                "n_transcripts": h["n_transcripts"], "n_reads": h["n_reads"],
                "note": STATEMENT})
            print("{:<10} b={:<3} dLL {:+.4f} [{:+.4f},{:+.4f}]  null p {:.3f}  -> {:<34} power {}".format(
                fam, b, _f(h["value"]), lo, _f(h["ci_hi"]), p, status,
                "{:.2f}".format(pv) if np.isfinite(pv) else "n/a"))
    out = C.results_path("missing_eval", "yeast_single_library_summary.tsv")
    C.write_tsv(rows, out, extra_first=("dataset", "family", "bin_width", "status"))
    print(STATEMENT)
    print("-> {}".format(out))


if __name__ == "__main__":
    main()
