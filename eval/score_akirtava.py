"""Score a trained model against the Akirtava MPRA.  First-pass sketch.

    python eval/score_akirtava.py --checkpoint work/model.pt

The model is run FORWARD only -- it never sees an MPRA measurement, and
nothing here is fitted.  So this is an out-of-sample test of a mechanism,
not a regression.

What is compared
----------------
  P(commit_main)  the chain's fate: of the 40S that load at the cap, the
                  fraction that commit at the main AUG
  r_init          rho_load * P(commit_main); proportional to the above, so it
                  ranks identically -- reported only for completeness

against

  TEff            PoLib-Seq ribosome loading (% translating).  The closest
                  match to what the model predicts, and the headline axis.
  MeanYFP         protein output.  One step further downstream (elongation
                  and protein stability sit in between), so a weaker test.

Baseline: LSMKozakStart, their own Kozak score, on the same rows.  The
comparison that matters is against their *mechanistic* leaky-scanning model
(R^2 ~ 0.49 in the paper), not their full feature model (~0.80) -- the latter
is supervised on this data with engineered features, which is a different
kind of object.

Scale does not matter here.  rho_load is unidentifiable in the model, but a
rank correlation is invariant to it, so this test is valid despite that.

Caveat to keep in view: the model was trained on TCP-seq from these same
yeast genes, so it has seen these SEQUENCES, though never this LABEL.  Not
label leakage, but not a clean held-out set either.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import NET_INPUT_CHANNELS, featurize
from model.chain import COMMIT_MAIN, COMMIT_UORF, DROP_OFF, FATE_NAMES, READ_THROUGH
from model.scanner import Scanner, load_state_compat, to_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "akirtava_scer_wt.tsv")

# The MPRA construct is the same on both sides of every leader, so the flanks
# are constant across rows and contribute nothing to between-TL variance.
# They exist only so the rate net never sees zero-padding.
UPSTREAM_FLANK = "AGTTTCTTTCATAACACC"          # vector, 5' of the leader
DOWNSTREAM_FLANK = "TCTAAAGGTGAAGAATTATTC"     # start of YFP, 3' of the AUG


def load_rows(path):
    if not os.path.exists(path):
        sys.exit("\nNo scored table at {}\nRun: python eval/prepare_akirtava.py\n".format(path))
    with open(path) as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def num(s):
    try:
        v = float(s)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def report(name, pred, obs):
    r = float(np.corrcoef(pred, obs)[0, 1])
    print("  %-26s n=%4d   Pearson r=%+.3f  R2=%.3f   Spearman rho=%+.3f"
          % (name, len(pred), r, r * r, spearman(pred, obs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--min-length", type=int, default=20,
                    help="leaders shorter than this cannot host a window")
    ap.add_argument("--max-length", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N rows")
    ap.add_argument("--only-chroms", default="",
                    help="score only leaders on these chromosomes, e.g. "
                         "'II,V,X,XIV'. Paired with train.py --exclude-chroms "
                         "this gives a genuinely held-out test.")
    args = ap.parse_args()

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = Scanner(window=cfg["window"], hidden=cfg["hidden"],
                    in_channels=cfg["in_channels"], rho_load=cfg.get("rho_load"),
                    # Absent means the checkpoint predates reinitiation, so
                    # it has no reinit head and must be scored with the edges
                    # off -- defaulting to True would silently score it with a
                    # zero-init head at sigmoid(0) = 0.5.
                    reinit=cfg.get("reinit", False),
                    drop_at_starts=cfg.get("drop_at_starts", True))
    load_state_compat(model, ck["state_dict"])
    model.eval()
    print("[eval] checkpoint {}".format(args.checkpoint))
    print("[eval] trained with: {}".format(ck.get("train_args")))

    rows = load_rows(args.table)
    if args.only_chroms:
        keep = {c.strip() for c in args.only_chroms.split(",") if c.strip()}
        before = len(rows)
        rows = [r for r in rows if r.get("chrom") in keep]
        print("[eval] held-out chromosomes {}: {} -> {} leaders".format(
            sorted(keep), before, len(rows)))
    if args.limit:
        rows = rows[:args.limit]

    out, skipped = [], {"short": 0, "long": 0, "bad_bases": 0, "error": 0}
    for r in rows:
        seq = (r["utr_seq"] or "").upper().replace("U", "T")
        if len(seq) < args.min_length:
            skipped["short"] += 1
            continue
        if len(seq) > args.max_length:
            skipped["long"] += 1
            continue
        if set(seq) - set("ACGT"):
            skipped["bad_bases"] += 1
            continue

        # Leader + the construct's own AUG, matching how the pipeline builds
        # examples: the AUG sits at the end, main_start_index = len - 3.
        full = seq + "ATG"
        context = UPSTREAM_FLANK + full + DOWNSTREAM_FLANK
        try:
            feats = featurize(full, len(full) - 3, structure=False,
                              context_seq=context, context_offset=len(UPSTREAM_FLANK))
            with torch.no_grad():
                o = model(to_tensors(feats))
            f = o["fates"].numpy()
        except AssertionError:
            skipped["error"] += 1
            continue
        out.append({
            "id": r["id"],
            "uorf_init_flux": float(o["uorf_init_flux"]),
            "reinit_flux": float(o["reinit_flux"]),
            "commit_main": float(f[COMMIT_MAIN]),
            "commit_uORF": float(f[COMMIT_UORF]),
            "drop_off": float(f[DROP_OFF]),
            "read_through": float(f[READ_THROUGH]),
            "r_init": float(o["r_init"]),
            "TEff": num(r.get("TEff")),
            "MeanYFP": num(r.get("MeanYFP")),
            "MeanRNA": num(r.get("MeanRNA")),
            "LSMKozakStart": num(r.get("LSMKozakStart")),
            "uAUG": num(r.get("uAUG")),
            "length": len(seq),
        })

    print("[eval] scored {} leaders  (skipped {})".format(len(out), skipped))
    if not out:
        sys.exit("nothing scored")

    for target in ("TEff", "MeanYFP"):
        sub = [d for d in out if d[target] is not None]
        if len(sub) < 30:
            print("\n== {} == too few rows ({})".format(target, len(sub)))
            continue
        y = np.array([d[target] for d in sub])
        print("\n== vs {}  ({} leaders) ==".format(target, len(sub)))
        for pred_name in ("commit_main", "read_through", "drop_off", "commit_uORF"):
            report("model: " + pred_name, np.array([d[pred_name] for d in sub]), y)
        base = [d for d in sub if d["LSMKozakStart"] is not None]
        if len(base) >= 30:
            report("baseline: LSMKozakStart",
                   np.array([d["LSMKozakStart"] for d in base]),
                   np.array([d[target] for d in base]))

    csv_out = os.path.join(HERE, "scored.tsv")
    with open(csv_out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(out)
    print("\n[eval] per-leader predictions -> {}".format(csv_out))


if __name__ == "__main__":
    main()
