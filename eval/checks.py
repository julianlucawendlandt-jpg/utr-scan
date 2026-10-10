"""Three checks on the Akirtava result, run against eval/scored.tsv.

    python eval/checks.py

1. LENGTH CONFOUND.  commit_main correlates -0.49 with leader length, and
   short leaders genuinely translate better.  The chain does encode length
   (fewer states, fewer chances to drop off), but that is a trivial mechanism
   and not what the result is claiming.  So: does the contribution survive
   once length is in the baseline?

2. SHORT-LEADER EXCLUSION.  498 leaders under 20 nt were skipped, and they
   are not a random subset -- they are the shortest, hence among the best
   translated.  Does the result hold on the length range actually scored, and
   does the skipped set differ?

3. SATURATION / RANGE.  A fate is bounded in [0,1] while TEff and YFP are
   not, so a high correlation could come from a few saturated leaders.
   Checks the result survives in the interior of the predicted range.

The chromosome holdout is a separate script (it needs retraining).
"""

import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(path):
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def arr(rows, name):
    return np.array([float(r[name]) for r in rows])


def have(r, keys):
    return all(r.get(k) not in (None, "") for k in keys)


def r2_of(y, cols):
    X = np.column_stack([np.ones(len(y))] + cols)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    p = X @ b
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def partial_F(y, base_cols, extra_cols):
    b = r2_of(y, base_cols)
    f = r2_of(y, base_cols + extra_cols)
    n = len(y)
    k1, k2 = len(base_cols) + 1, len(base_cols) + len(extra_cols) + 1
    F = ((f - b) / (k2 - k1)) / ((1 - f) / (n - k2))
    return b, f, F


def check_length(rows):
    print("=" * 72)
    print("1. LENGTH CONFOUND -- does commit_main survive with length in the baseline?")
    print("=" * 72)
    for target in ("TEff", "MeanYFP"):
        need = [target, "commit_main", "LSMKozakStart", "uAUG", "length"]
        rs = [r for r in rows if have(r, need)]
        y = arr(rs, target)
        cm, koz, ua, ln = (arr(rs, c) for c in ("commit_main", "LSMKozakStart", "uAUG", "length"))
        # log length: the chain's drop-off compounds per nucleotide, so its
        # effect on survival is multiplicative, not additive.
        lnl = np.log(ln)
        print("\n  %s (n=%d)" % (target, len(rs)))
        print("    length alone                      R2 = %.4f" % r2_of(y, [ln, lnl]))
        for label, base in (
            ("Kozak + uAUG", [koz, ua]),
            ("Kozak + uAUG + length + log(length)", [koz, ua, ln, lnl]),
        ):
            b, f, F = partial_F(y, base, [cm])
            print("    %-34s R2 = %.4f  -> +commit_main %.4f  (delta %+.4f, F=%.0f)"
                  % (label, b, f, f - b, F))


def check_short(rows, all_table):
    print("\n" + "=" * 72)
    print("2. SHORT-LEADER EXCLUSION -- are the 498 skipped leaders different?")
    print("=" * 72)
    scored_ids = {r["id"] for r in rows}
    skipped = [r for r in all_table
               if r["id"] not in scored_ids and r.get("utr_seq")]
    print("  scored %d, skipped %d" % (len(rows), len(skipped)))
    for target in ("TEff", "MeanYFP"):
        sv = [float(r[target]) for r in rows if have(r, [target])]
        kv = [float(r[target]) for r in skipped if have(r, [target])]
        if kv:
            print("    %-8s scored median %.3f (n=%d)   skipped median %.3f (n=%d)"
                  % (target, np.median(sv), len(sv), np.median(kv), len(kv)))

    print("\n  result within length bands (commit_main vs target, Spearman):")
    for target in ("TEff", "MeanYFP"):
        rs = [r for r in rows if have(r, [target, "commit_main", "length"])]
        ln = arr(rs, "length")
        print("    %s" % target)
        for lo, hi in ((20, 40), (40, 70), (70, 120), (120, 10000)):
            sub = [r for r, L in zip(rs, ln) if lo <= L < hi]
            if len(sub) < 50:
                continue
            y, cm = arr(sub, target), arr(sub, "commit_main")
            koz = arr(sub, "LSMKozakStart") if all(have(r, ["LSMKozakStart"]) for r in sub) else None
            ra = np.argsort(np.argsort(cm)).astype(float)
            rb = np.argsort(np.argsort(y)).astype(float)
            rho = np.corrcoef(ra, rb)[0, 1]
            extra = ""
            if koz is not None:
                rk = np.argsort(np.argsort(koz)).astype(float)
                extra = "   (Kozak rho %+.3f)" % np.corrcoef(rk, rb)[0, 1]
            print("      %4d-%-5s nt  n=%4d  rho = %+.3f%s" % (lo, hi if hi < 9999 else "+", len(sub), rho, extra))


def check_range(rows):
    print("\n" + "=" * 72)
    print("3. SATURATION -- is the correlation carried by the extremes?")
    print("=" * 72)
    for target in ("TEff", "MeanYFP"):
        rs = [r for r in rows if have(r, [target, "commit_main"])]
        y, cm = arr(rs, target), arr(rs, "commit_main")
        lo, hi = np.percentile(cm, [10, 90])
        mid = (cm > lo) & (cm < hi)
        def rho(a, b):
            return np.corrcoef(np.argsort(np.argsort(a)).astype(float),
                               np.argsort(np.argsort(b)).astype(float))[0, 1]
        print("  %-8s all n=%d rho=%+.3f   interior 10-90%% n=%d rho=%+.3f"
              % (target, len(y), rho(cm, y), mid.sum(), rho(cm[mid], y[mid])))
        print("           predicted commit_main range: %.4f .. %.4f (10th..90th pct %.4f..%.4f)"
              % (cm.min(), cm.max(), lo, hi))


def main():
    scored = os.path.join(HERE, "scored.tsv")
    table = os.path.join(HERE, "akirtava_scer_wt.tsv")
    if not os.path.exists(scored):
        sys.exit("\nNo %s -- run eval/score_akirtava.py first\n" % scored)
    rows = load(scored)
    all_table = load(table)
    # carry the measurements onto the skipped rows for check 2
    check_length(rows)
    check_short(rows, all_table)
    check_range(rows)


if __name__ == "__main__":
    main()
