"""Score arbitrary 5'UTR sequences: chain -> representation -> head -> number.

    python eval/predict.py --head work/head_teff.pt --seq ACGTACGT...
    python eval/predict.py --head work/head_teff.pt --fasta leaders.fa
    python eval/predict.py --head work/head_teff.pt --tsv leaders.tsv --column utr_seq

This is the whole inference path in one place.  Neither TEff nor protein
output is an output of the chain -- the chain is target-agnostic.  It produces
a representation of how a 40S scans the leader, and a head trained on a
specific assay reads a number off it:

    sequence -> featurize -> chain -> profile, w, p, d, fates
                                   -> head -> z-scored value
                                   -> x sd + mean -> the target's own units

The chain is trained once, on TCP-seq occupancy.  To predict something new you
fit a new head (eval/fit_head.py --save) and the chain does not move.

THE CONSTRUCT FLANKS.  The rate net reads a +/- window around every position,
so the ends of a bare leader would sit against zero-padding, which is a value
no real nucleotide takes and which the net will happily use as a position cue.
Here we pad with the Akirtava construct's own flanks, the same ones the head
was fitted under.  For a leader from a different context those flanks are
wrong in principle; they are constant across sequences, so they shift every
prediction alike and leave the ranking intact, but an absolute number for a
non-Akirtava construct should be read with that in mind.
"""

import argparse
import csv
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.features import featurize
from eval.fit_head import (N_CHANNELS, UPSTREAM_FLANK, DOWNSTREAM_FLANK, Head)
from model.scanner import Scanner, load_state_compat, to_tensors


def load_chain(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    m = Scanner(window=cfg["window"], hidden=cfg["hidden"],
                in_channels=cfg["in_channels"], rho_load=cfg.get("rho_load"),
                reinit=cfg.get("reinit", False),
                drop_at_starts=cfg.get("drop_at_starts", True))
    load_state_compat(m, ck["state_dict"])
    m.eval()
    return m


def read_fasta(path):
    name, buf, out = None, [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if name is not None:
                    out.append((name, "".join(buf)))
                name, buf = line[1:].split()[0], []
            elif line:
                buf.append(line)
    if name is not None:
        out.append((name, "".join(buf)))
    return out


def represent(chain, seq):
    """One leader -> (channels (N, C), fates (4,)).  None if unusable."""
    seq = seq.upper().replace("U", "T")
    if len(seq) < 20 or set(seq) - set("ACGT"):
        return None
    full = seq + "ATG"
    ctx = UPSTREAM_FLANK + full + DOWNSTREAM_FLANK
    try:
        f = featurize(full, len(full) - 3, structure=False,
                      context_seq=ctx, context_offset=len(UPSTREAM_FLANK))
        with torch.no_grad():
            o = chain(to_tensors(f))
    except AssertionError:
        return None
    prof = o["profile"].numpy()
    x = np.stack([
        np.log1p(prof),
        prof / max(prof.sum(), 1e-12),
        o["w"].numpy(), o["p"].numpy(), o["d"].numpy(),
    ], axis=1).astype(np.float32)
    return x, o["fates"].numpy().astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", required=True, help="from fit_head.py --save")
    ap.add_argument("--chain", help="override the chain recorded in the head")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--seq", help="one leader, 5'->3', WITHOUT the AUG")
    src.add_argument("--fasta")
    src.add_argument("--tsv")
    ap.add_argument("--column", default="utr_seq", help="sequence column for --tsv")
    ap.add_argument("--id-column", default="id")
    ap.add_argument("--out", help="write a TSV instead of printing")
    args = ap.parse_args()

    hk = torch.load(args.head, map_location="cpu", weights_only=False)
    chain_path = args.chain or hk["chain_checkpoint"]
    if not os.path.exists(chain_path):
        sys.exit("\nThe head was fitted on chain {}, which is not on disk.\n"
                 "Pass --chain to point at it.\n".format(chain_path))
    if args.chain and os.path.abspath(args.chain) != hk["chain_checkpoint"]:
        print("[predict] WARNING: this head was fitted on {}. A head is only "
              "meaningful on\n  the representation it was trained against; a "
              "different chain gives different\n  features and the numbers "
              "below are not calibrated.".format(hk["chain_checkpoint"]))

    chain = load_chain(chain_path)
    head = Head(hidden=hk["config"]["hidden"])
    head.load_state_dict(hk["state_dict"])
    head.eval()
    target = hk["config"]["target"]
    mu, sd = hk["target_mean"], hk["target_sd"]
    print("[predict] target {}  (head held-out Pearson r={:+.3f}, rho={:+.3f} "
          "on chromosomes {})".format(
              target, hk.get("held_out_pearson", float("nan")),
              hk.get("held_out_spearman", float("nan")),
              ",".join(hk.get("splits", {}).get("test", []))))

    if args.seq:
        items = [("query", args.seq)]
    elif args.fasta:
        items = read_fasta(args.fasta)
    else:
        with open(args.tsv) as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        items = [(r.get(args.id_column, str(i)), r[args.column])
                 for i, r in enumerate(rows)]

    results, skipped = [], 0
    for name, seq in items:
        rep = represent(chain, seq)
        if rep is None:
            skipped += 1
            continue
        x, fates = rep
        with torch.no_grad():
            z = head(torch.as_tensor(x).unsqueeze(0),
                     torch.ones(1, len(x), dtype=torch.bool),
                     torch.as_tensor(fates).unsqueeze(0))
        results.append({
            "id": name, "length": len(seq),
            target: float(z) * sd + mu,
            "z": float(z),
            "commit_main": float(fates[0]), "commit_uORF": float(fates[1]),
            "drop_off": float(fates[2]), "read_through": float(fates[3]),
        })

    if skipped:
        print("[predict] skipped {} sequences (under 20 nt, or non-ACGT)".format(skipped))
    if not results:
        sys.exit("nothing scored")

    if args.out:
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()), delimiter="\t")
            w.writeheader()
            w.writerows(results)
        print("[predict] {} sequences -> {}".format(len(results), args.out))
    else:
        print("\n%-24s %6s %12s %8s   %s" % ("id", "len", target, "z", "fates (main/uORF/drop/read)"))
        for r in results:
            print("%-24s %6d %12.4f %8.2f   %.3f %.3f %.3f %.3f"
                  % (r["id"], r["length"], r[target], r["z"], r["commit_main"],
                     r["commit_uORF"], r["drop_off"], r["read_through"]))


if __name__ == "__main__":
    main()
