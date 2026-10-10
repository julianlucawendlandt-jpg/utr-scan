"""Did reinitiation fix the uORF branch, or is it still a uAUG detector?

    python eval/check_uorf.py            # reads eval/scored.tsv

Before reinitiation existed, commit_uORF was structurally degenerate.  A uORF
was an absorbing state, so the only thing the branch could affect was "some
mass disappears here", and the data it answered to -- occupancy shape alone --
was satisfied by noticing that an AUG is present.  Measured on the held-out
set, it was exactly that:

    commit_uORF = 0.000 for all 947 leaders with no uAUG
                  mean 0.276 for the 219 with one
    within uAUG-bearing leaders, vs protein output:
                  commit_uORF  rho -0.414
                  plain uAUG count  rho -0.469      <- the branch lost to
                                                       counting start codons

Reinitiation gives the branch a consequence a second observable can see: uORF
mass that comes back feeds commit_main, which the 80S term constrains.  So the
branch should stop being a presence flag and start ranking uORFs by strength.

Four questions, in the order that matters:

1. Is it still a step function?  commit_uORF should now vary among leaders
   that all have a uAUG, because reinitiation varies.
2. Does it beat counting?  Within uAUG-bearing leaders, commit_uORF vs plain
   uAUG count against protein output.  Losing here means the branch is still
   not measuring anything a start-codon count does not.
3. Is reinitiation conditioned on geometry, or constant?  If reinit_flux /
   uorf_init_flux is flat across leaders, the head learned a constant and the
   two geometry channels bought nothing.
4. Are the two routes to commit_main identifiable?  Direct scanning and
   reinitiation both feed it and the 80S term constrains only the total.  A
   flat reinitiation fraction is what that failure looks like, which is why
   question 3 is also the answer to this one.
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def have(r, keys):
    return all(r.get(k) not in (None, "") for k in keys)


def arr(rows, name):
    return np.array([float(r[name]) for r in rows])


def rho(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def r2_of(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    path = os.path.join(HERE, "scored.tsv")
    if not os.path.exists(path):
        sys.exit("\nNo %s -- run eval/score_akirtava.py first\n" % path)
    rows = load(path)
    has_reinit = "reinit_flux" in rows[0]
    target = "MeanYFP"
    rows = [r for r in rows if have(r, [target, "commit_uORF", "uAUG"])]
    print("[uorf] %d leaders, reinitiation columns %s\n"
          % (len(rows), "present" if has_reinit else "ABSENT (old scored.tsv)"))

    y = arr(rows, target)
    cu = arr(rows, "commit_uORF")
    cm = arr(rows, "commit_main")
    n_uaug = arr(rows, "uAUG")
    with_u = n_uaug > 0

    # --- 1 ---------------------------------------------------------------
    print("=" * 72)
    print("1. STEP FUNCTION?  commit_uORF conditioned on uAUG presence")
    print("=" * 72)
    for lbl, m in (("no uAUG", ~with_u), ("has uAUG", with_u)):
        v = cu[m]
        print("  %-9s n=%4d   mean %.5f   sd %.5f   range %.5f .. %.5f"
              % (lbl, m.sum(), v.mean(), v.std(), v.min(), v.max()))
    spread = cu[with_u].std() / max(cu[with_u].mean(), 1e-12)
    print("\n  coefficient of variation within uAUG-bearing leaders: %.3f" % spread)
    print("  (near zero = the branch only reports presence; the old model was"
          "\n   a step function with all its variance BETWEEN the two groups)")

    # --- 2 ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("2. DOES IT BEAT COUNTING?  within uAUG-bearing leaders only")
    print("=" * 72)
    sub = [r for r, m in zip(rows, with_u) if m]
    ys, cus, cms, ns = (arr(sub, c) for c in
                        (target, "commit_uORF", "commit_main", "uAUG"))
    print("  n = %d" % len(sub))
    print("    commit_uORF vs %-8s  rho = %+.3f" % (target, rho(cus, ys)))
    print("    uAUG count  vs %-8s  rho = %+.3f   <- the bar to clear"
          % (target, rho(ns, ys)))
    print("    commit_main vs %-8s  rho = %+.3f" % (target, rho(cms, ys)))
    if has_reinit:
        for c in ("uorf_init_flux", "reinit_flux"):
            print("    %-11s vs %-8s  rho = %+.3f" % (c, target, rho(arr(sub, c), ys)))
    verdict = "BEATS counting" if abs(rho(cus, ys)) > abs(rho(ns, ys)) else "still loses to counting"
    print("\n  verdict: commit_uORF %s" % verdict)

    # --- 3 ---------------------------------------------------------------
    if has_reinit:
        print("\n" + "=" * 72)
        print("3. IS REINITIATION CONDITIONED, OR CONSTANT?")
        print("=" * 72)
        init = arr(sub, "uorf_init_flux")
        re = arr(sub, "reinit_flux")
        live = init > 1e-9
        frac = re[live] / init[live]
        print("  reinitiated fraction over %d uAUG-bearing leaders:" % live.sum())
        print("    mean %.3f   sd %.3f   10th-90th pct %.3f .. %.3f"
              % (frac.mean(), frac.std(),
                 *np.percentile(frac, [10, 90])))
        if frac.std() < 0.02:
            print("\n  FLAT -- the head learned a constant.  The geometry channels"
                  "\n  bought nothing, and direct scanning vs reinitiation is"
                  "\n  unidentifiable as feared.")
        else:
            print("\n  varies -- the head is using something.  Whether it is the"
                  "\n  right something is question 3b.")
        print("\n  3b. what does it track?  (Spearman against the fraction)")
        for c in ("length", "uAUG"):
            if have(sub[0], [c]):
                print("      %-12s rho = %+.3f" % (c, rho(arr(sub, c)[live], frac)))
        print("      %-12s rho = %+.3f" % (target, rho(ys[live], frac)))
        print("  (reinitiation is known to rise with the gap from the uORF stop"
              "\n   to the main AUG, so a positive correlation with leader length"
              "\n   is the expected sign, not a confound)")

    # --- 4 ---------------------------------------------------------------
    print("\n" + "=" * 72)
    print("4. INCREMENTAL R2 over Kozak + uAUG  (all leaders)")
    print("=" * 72)
    full = [r for r in rows if have(r, ["LSMKozakStart"])]
    yf = arr(full, target)
    K, U = arr(full, "LSMKozakStart"), arr(full, "uAUG")
    base = r2_of(yf, [K, U])
    print("  Kozak + uAUG                 R2 = %.4f" % base)
    cand = ["commit_main", "commit_uORF"] + (
        ["uorf_init_flux", "reinit_flux"] if has_reinit else [])
    for c in cand:
        v = arr(full, c)
        print("    + %-16s         R2 = %.4f   (%+.4f)"
              % (c, r2_of(yf, [K, U, v]), r2_of(yf, [K, U, v]) - base))
    allc = [arr(full, c) for c in cand]
    print("    + all of them              R2 = %.4f   (%+.4f)"
          % (r2_of(yf, [K, U] + allc), r2_of(yf, [K, U] + allc) - base))


if __name__ == "__main__":
    main()
