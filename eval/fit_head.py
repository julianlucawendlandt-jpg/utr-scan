"""Stage 2: a downstream head on the chain's representation.

    python eval/fit_head.py --checkpoint work/model_clean.pt --target TEff

WHY THIS EXISTS.  Until now the evaluation read a single scalar out of the
chain -- P(commit_main) -- and correlated it with a measurement.  That throws
away the occupancy profile and the entire rate field, and it demands that one
unidentified number be directly comparable across genes.  Most of this
project's difficulty came from trying to force that.

The chain's job is the MECHANISM: given a leader, where do 40S sit and what
are the local step / commit / dwell tendencies.  It is fitted to TCP-seq
occupancy SHAPE, which is the one thing TCP-seq actually measures, and to
nothing else.  Per-transcript SCALE is then a separate problem, handled by a
head trained on labels that measure it.

That split is not ours.  seq2ribo (Kaynar & Kingsford 2026) makes the same
one, and states the reason plainly: their sTASEP simulator uses a single
initiation rate shared across transcripts, so "the simulator cannot capture
transcript-level differences in ribosome load; it can only shape the
positional distribution", and they therefore fit it against per-transcript
SCALED error.  Scale is left to a learned head on top of the simulated
profile.  Our occupancy term already normalises to the observed total, so the
chain was always a shape model; the commit term was the one place we broke
the rule, and it is where the trouble came from.

THE REPRESENTATION, per leader of N nucleotides:

    log1p(profile)   (N,)   expected 40S visits, on a log scale
    profile / sum    (N,)   the same as a shape, scale removed
    w, p, d          (N,3)  the rate field: step, commit and dwell tendency
    fates            (4,)   one global vector

THE HEAD mirrors seq2ribo's: a position-wise feed-forward network, masked
mean pooling over positions, then a linear layer.  It is small on purpose --
the representation is supposed to be doing the work, and a large head would
hide a bad one.

WHAT THIS DOES NOT DO.  Once a head is fitted on an MPRA, that MPRA is no
longer an untouched test set for that head.  The split below is what keeps the
reported number honest, and for a group split it only works if the CHAIN was
also trained with those groups excluded (train.py --exclude-chroms).  The
script checks and warns.

ANY TABLE, NOT JUST AKIRTAVA.  The column names are flags, so a human MPRA
(Sample et al. 2019 mean ribosome load, say) needs a parser into a flat table
and nothing here.  Two split modes:

  --split group   hold out whole chromosomes, or any other grouping column.
                  The honest one: related sequences stay on the same side of
                  the wall, and it can be matched to the chain's own holdout.
  --split random  a seeded row split, for libraries of designed or random
                  sequences that have no genomic location. Weaker -- near
                  duplicates can straddle the split -- so it warns.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import featurize
from model.scanner import Scanner, load_state_compat, to_tensors

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "akirtava_scer_wt.tsv")

# Same constant construct flanks as score_akirtava.py: identical on every row,
# so they contribute nothing between leaders and exist only to keep the rate
# net away from zero-padding.
UPSTREAM_FLANK = "AGTTTCTTTCATAACACC"
DOWNSTREAM_FLANK = "TCTAAAGGTGAAGAATTATTC"

N_CHANNELS = 5      # log1p(profile), profile/sum, w, p, d
N_FATES = 4


def rho(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


class Head(nn.Module):
    """Position-wise FFN -> masked mean pool -> linear.  ~2k parameters.

    Deliberately small.  If the chain's representation is good, this is enough
    to read a scalar off it; if it is not, a bigger head would paper over that
    and we would learn nothing about the representation.
    """

    def __init__(self, hidden=32):
        super().__init__()
        self.pos = nn.Sequential(
            nn.Linear(N_CHANNELS, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
        )
        self.out = nn.Linear(hidden + N_FATES, 1)

    def forward(self, x, mask, fates):
        """x: (B, L, C)  mask: (B, L)  fates: (B, 4) -> (B,)"""
        h = self.pos(x)                                   # (B, L, H)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)  # masked mean
        return self.out(torch.cat([pooled, fates], dim=1)).squeeze(-1)


def load_rows(path, target, seq_col="utr_seq", id_col="id",
              group_col="chrom", baseline_col="LSMKozakStart"):
    """Flat table -> rows.  Delimiter is taken from the extension (MPRA
    supplements are as often .csv as .tsv)."""
    delim = "," if path.lower().endswith(".csv") else "\t"
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter=delim))
    if rows and target not in rows[0]:
        sys.exit("\nNo column {!r} in {}.\nColumns: {}\n".format(
            target, path, ", ".join(rows[0])))
    out = []
    for i, r in enumerate(rows):
        seq = (r.get(seq_col) or "").upper().replace("U", "T")
        try:
            y = float(r[target])
        except (KeyError, TypeError, ValueError):
            continue
        if y != y or not seq or set(seq) - set("ACGT"):
            continue
        out.append({"id": r.get(id_col) or str(i),
                    "group": (r.get(group_col) or "").strip(),
                    "seq": seq, "y": y,
                    "kozak": r.get(baseline_col, ""),
                    "uaug": r.get("uAUG", "")})
    return out


def extract(model, rows, min_len, max_len):
    """Run the chain forward once per leader and keep the representation."""
    kept, skipped = [], 0
    for r in rows:
        if not (min_len <= len(r["seq"]) <= max_len):
            skipped += 1
            continue
        full = r["seq"] + "ATG"
        ctx = UPSTREAM_FLANK + full + DOWNSTREAM_FLANK
        try:
            f = featurize(full, len(full) - 3, structure=False,
                          context_seq=ctx, context_offset=len(UPSTREAM_FLANK))
            with torch.no_grad():
                o = model(to_tensors(f))
        except AssertionError:
            skipped += 1
            continue
        prof = o["profile"].numpy()
        chans = np.stack([
            np.log1p(prof),
            prof / max(prof.sum(), 1e-12),
            o["w"].numpy(), o["p"].numpy(), o["d"].numpy(),
        ], axis=1).astype(np.float32)            # (N, C)
        r = dict(r)
        r["x"] = chans
        r["fates"] = o["fates"].numpy().astype(np.float32)
        r["commit_main"] = float(o["fates"][0])
        kept.append(r)
    return kept, skipped


def pad_batch(rows, device=None):
    L = max(len(r["x"]) for r in rows)
    B, C = len(rows), N_CHANNELS
    x = np.zeros((B, L, C), dtype=np.float32)
    mask = np.zeros((B, L), dtype=bool)
    fates = np.zeros((B, N_FATES), dtype=np.float32)
    for i, r in enumerate(rows):
        n = len(r["x"])
        x[i, :n] = r["x"]
        mask[i, :n] = True
        fates[i] = r["fates"]
    y = np.array([r["y"] for r in rows], dtype=np.float32)
    return (torch.as_tensor(x, device=device), torch.as_tensor(mask, device=device),
            torch.as_tensor(fates, device=device), torch.as_tensor(y, device=device))


def report(name, pred, obs):
    print("  %-30s n=%4d   Pearson r=%+.3f  R2=%.3f   Spearman rho=%+.3f"
          % (name, len(pred), pearson(pred, obs), pearson(pred, obs) ** 2,
             rho(pred, obs)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--target", default="TEff",
                    help="column to predict. For Akirtava: TEff (polysome "
                         "loading) is closest to what the chain models, while "
                         "MeanYFP is protein output -- further downstream and "
                         "partly driven by mRNA level")
    ap.add_argument("--table", default=TABLE)
    ap.add_argument("--seq-column", default="utr_seq")
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--group-column", default="chrom",
                    help="column defining the holdout groups (--split group)")
    ap.add_argument("--baseline-column", default="LSMKozakStart",
                    help="an existing predictor to report alongside, if present")
    ap.add_argument("--split", choices=("group", "random"), default="group")
    ap.add_argument("--test-groups", "--test-chroms", dest="test_groups",
                    default="IV,VII")
    ap.add_argument("--val-groups", "--val-chroms", dest="val_groups",
                    default="II,XV")
    ap.add_argument("--test-frac", type=float, default=0.2,
                    help="--split random only")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="--split random only")
    ap.add_argument("--min-length", type=int, default=20)
    ap.add_argument("--max-length", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", metavar="PATH",
                    help="write the head plus everything needed to score a new "
                         "sequence: weights, the target's train-set mean and sd "
                         "(predictions are z-scored internally), and which chain "
                         "checkpoint the representation must come from")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    model = Scanner(window=cfg["window"], hidden=cfg["hidden"],
                    in_channels=cfg["in_channels"], rho_load=cfg.get("rho_load"),
                    reinit=cfg.get("reinit", False),
                    drop_at_starts=cfg.get("drop_at_starts", True))
    load_state_compat(model, ck["state_dict"])
    model.eval()
    print("[head] chain: {}".format(args.checkpoint))
    print("[head] chain trained with: {}".format(ck.get("train_args")))

    test_c = {c.strip() for c in args.test_groups.split(",") if c.strip()}
    val_c = {c.strip() for c in args.val_groups.split(",") if c.strip()}
    if args.split == "group":
        excl = (ck.get("train_args") or {}).get("exclude_chroms") or ""
        excl = {c.strip() for c in excl.split(",") if c.strip()}
        if not test_c <= excl:
            print("\n  WARNING: the chain saw groups {} during its own "
                  "training.\n  The head's test split is therefore only held "
                  "out at the LABEL level, not\n  the sequence level. Retrain "
                  "the chain with --exclude-chroms {} for a\n  clean "
                  "number.\n".format(
                      sorted(test_c - excl), ",".join(sorted(test_c))))

    rows = load_rows(args.table, args.target, seq_col=args.seq_column,
                     id_col=args.id_column, group_col=args.group_column,
                     baseline_col=args.baseline_column)
    rows, skipped = extract(model, rows, args.min_length, args.max_length)
    print("[head] {} leaders with a {} label ({} skipped)".format(
        len(rows), args.target, skipped))

    if args.split == "group":
        if not any(r["group"] for r in rows):
            sys.exit("\nNo values in group column {!r}. Use --group-column, or "
                     "--split random for a library with no genomic location.\n"
                     .format(args.group_column))
        tr = [r for r in rows if r["group"] not in test_c and r["group"] not in val_c]
        va = [r for r in rows if r["group"] in val_c]
        te = [r for r in rows if r["group"] in test_c]
        label = "groups {} / {}".format(",".join(sorted(val_c)), ",".join(sorted(test_c)))
    else:
        # A seeded row split.  Weaker than a group split: near-duplicate
        # sequences can land on both sides, which flatters the test number.
        # It is the only option for a library of designed sequences.
        order = np.random.default_rng(args.seed).permutation(len(rows))
        n_te = int(len(rows) * args.test_frac)
        n_va = int(len(rows) * args.val_frac)
        te = [rows[i] for i in order[:n_te]]
        va = [rows[i] for i in order[n_te:n_te + n_va]]
        tr = [rows[i] for i in order[n_te + n_va:]]
        label = "random, seed {}".format(args.seed)
        print("\n  NOTE: a random split does not separate sequence families. "
              "If these\n  sequences have genomic locations, --split group is "
              "the honest test.\n")
    assert tr and va and te, "an empty split: check the split settings"
    print("[head] split  train {}  val {}  test {}  ({})".format(
        len(tr), len(va), len(te), label))

    # Standardise the target on TRAIN statistics only.
    mu, sd = float(np.mean([r["y"] for r in tr])), float(np.std([r["y"] for r in tr]))
    for split in (tr, va, te):
        for r in split:
            r["y_raw"], r["y"] = r["y"], (r["y"] - mu) / max(sd, 1e-9)

    Xtr, Mtr, Ftr, Ytr = pad_batch(tr)
    Xva, Mva, Fva, Yva = pad_batch(va)
    Xte, Mte, Fte, Yte = pad_batch(te)

    head = Head(hidden=args.hidden)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    lossf = nn.MSELoss()
    print("[head] {} parameters".format(sum(p.numel() for p in head.parameters())))

    best, best_state, since = float("inf"), None, 0
    for ep in range(args.epochs):
        head.train()
        opt.zero_grad()
        loss = lossf(head(Xtr, Mtr, Ftr), Ytr)
        loss.backward()
        opt.step()
        head.eval()
        with torch.no_grad():
            vl = float(lossf(head(Xva, Mva, Fva), Yva))
        if vl < best - 1e-5:
            best, best_state, since = vl, {k: v.clone() for k, v in head.state_dict().items()}, 0
        else:
            since += 1
        if ep % 50 == 0:
            print("  epoch {:4d}  train {:.4f}  val {:.4f}".format(ep, float(loss), vl))
        if since > 80:
            print("  early stop at epoch {} (best val {:.4f})".format(ep, best))
            break
    head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        # The head predicts z-scores; put them back in the target's own units
        # so predictions and observations are directly comparable.  Omitting
        # this leaves the two columns on different scales -- correlations
        # survive that, calibration numbers do not.
        pred = head(Xte, Mte, Fte).numpy() * sd + mu
    obs = np.array([r["y_raw"] for r in te])

    print("\n== HELD OUT ({}), vs {} ==".format(
        label, args.target))
    report("head on representation", pred, obs)
    report("commit_main alone (old way)", np.array([r["commit_main"] for r in te]), obs)
    kz = [(float(r["kozak"]), r["y_raw"]) for r in te if r["kozak"] not in (None, "")]
    if len(kz) >= 30:
        report("baseline LSMKozakStart",
               np.array([a for a, _ in kz]), np.array([b for _, b in kz]))

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        torch.save({
            "state_dict": head.state_dict(),
            "config": {"hidden": args.hidden, "target": args.target},
            # The head predicts in z-scored units; these invert it.  They are
            # TRAIN statistics, so nothing from val or test leaks in.
            "target_mean": mu, "target_sd": sd,
            # A head is only meaningful on the representation it was fitted to,
            # so the chain is recorded and predict.py refuses a mismatch.
            "chain_checkpoint": os.path.abspath(args.checkpoint),
            "splits": {"mode": args.split,
                       "test": sorted(test_c), "val": sorted(val_c)},
            "held_out_pearson": pearson(pred, obs),
            "held_out_spearman": rho(pred, obs),
        }, args.save)
        print("[head] saved -> {}".format(args.save))

    out = os.path.join(HERE, "head_predictions.tsv")
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["id", "group", "length", "observed", "predicted", "commit_main"])
        for r, p in zip(te, pred):
            w.writerow([r["id"], r["group"], len(r["seq"]), "%.6g" % r["y_raw"],
                        "%.6g" % float(p), "%.6g" % r["commit_main"]])
    print("\n[head] held-out predictions -> {}".format(out))


if __name__ == "__main__":
    main()
