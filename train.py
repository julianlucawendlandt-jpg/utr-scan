"""Minimal training loop: Poisson NLL on occupancy, plus a commitment term.

    python train.py                      # synthetic, the default
    python train.py --data archer        # needs the pipeline cache

Two likelihood terms per transcript:

  occupancy   predicted SSU profile (scaled to the observed total) vs the
              observed per-nucleotide 40S counts -- the shape of scanning
  commitment  rho_load * P(commit_main) * (Input reads in the CDS window)
              vs the observed 80S reads in the same window -- how many
              walkers actually committed at the main AUG

The occupancy term compares a BLURRED prediction against the raw counts.  The
chain predicts where ribosomes are; the assay reports where footprint 5' ends
landed, and the nuclease does not cut at exactly the same offset every time,
so a hundred 40S on one AUG produce reads spread over a few nucleotides.  The
data is never touched -- the measurement model lives on the prediction side,
where it can be switched off and its width varied.  See blur_profile.

The second term is what makes the fates identifiable; without it commit and
drop-off are indistinguishable in the profile (see model/chain.py).

One example at a time on the FORWARD pass: UTR lengths vary and the chain
solve is per-transcript anyway, so a batched forward would buy padding bugs
and very little speed.  The BACKWARD pass is a different question, and for a
long time this conflated the two.  Gradients are accumulated over --batch
transcripts before Adam steps, because the point of a batch here is not
throughput, it is averaging: a step per transcript meant ~6.6k single-example
Adam steps per epoch and a result that depended on the order they arrived in.
Identical config and seed, on data differing by four examples out of 6601,
landed at held-out R2 0.245 and 0.408.  --batch 1 restores the old behaviour.
"""

import argparse
import time

import numpy as np
import torch

from data.features import NET_INPUT_CHANNELS, featurize
from model.chain import FATE_NAMES
from model.scanner import Scanner, to_tensors

EPS = 1e-8


def blur_profile(profile, w):
    """Spread the predicted profile over +/-1 nt: [w, 1-2w, w].

    The chain can only put elevated occupancy on a start codon, because that
    is the only place a self-loop exists.  The observed peak is ~5 nt wide, so
    without this the model has to explain the counts either side of the AUG
    some other way -- and the only route available is "the walker stepped
    past", which is read out as leaky scanning.  Blurring the prediction gives
    those counts their real explanation: the edge of one smeared peak.

    IMPORTANT, and the reason `w` is a flag rather than a fitted parameter:
    blur width and leak rate are confounded.  Counts one nucleotide past the
    AUG can be a smeared peak OR a genuine leaky-scanning walker, and the
    occupancy profile alone cannot separate them.  Fitting w would just move
    mass between the two with no way to tell which is right.  Treat it as a
    sensitivity knob: run a couple of widths and see how much the fates move.

    The kernel is symmetric because the nuclease has no idea where the AUG
    is.  Edge mass falls off the ends of the profile and is absorbed by the
    renormalisation in poisson_nll.
    """
    if w <= 0:
        return profile
    k = torch.tensor([w, 1.0 - 2.0 * w, w], dtype=profile.dtype,
                     device=profile.device).view(1, 1, 3)
    out = torch.nn.functional.conv1d(
        profile.view(1, 1, -1), k, padding=1).view(-1)
    return out.clamp(min=0.0)


def poisson_nll(profile, counts, total=None):
    """Poisson NLL, dropping the log(k!) term, which is constant in theta.

    The chain gives expected visits for ONE walker, so the profile is a shape
    with no sequencing depth attached to it.  We scale it to the observed
    total for the transcript before comparing; that is what makes this a fit
    to occupancy shape rather than to library size.
    """
    if total is None:
        total = counts.sum()
    rate = profile / (profile.sum() + EPS) * total
    return (rate - counts * torch.log(rate + EPS)).sum()


def load_elong_offsets(path):
    """{transcript_id: elongation factor} from elong/fit.py, or {} if unused.

    Optional and self-contained: see elong/README.md.  Genes absent from the
    file fall back to 1.0, which is the uncorrected term.
    """
    import csv
    out = {}
    with open(path) as fh:
        for r in csv.DictReader(fh, delimiter="	"):
            try:
                out[r["transcript_id"]] = float(r["elong_factor"])
            except (KeyError, TypeError, ValueError):
                continue
    assert out, "no offsets read from {}".format(path)
    return out


def commit_nll(r_init, rs_total, input_total, elong_factor=1.0):
    """Poisson NLL of the 80S readout given the chain's initiation rate.

    Expected 80S in the CDS window = r_init * (number of mRNA copies), with
    the Input reads over the same window standing in for copy number.  Every
    ribosome that commits at the AUG elongates through this window, so the
    80S sum measures commitment flux, and dividing by abundance turns it
    into commitment per mRNA -- the quantity P(commit_main) predicts.
    rho_load (inside r_init) absorbs the constant between the two libraries.

    The optional elong_factor corrects a confound: 80S/Input is ribosome
    DENSITY, which is initiation divided by elongation rate, and elongation is
    a property of the coding sequence that this model cannot see.  The factor
    multiplies it back in.  It is centred at 1, so rho_load's scale is
    untouched, and 1.0 leaves the term exactly as it was.  See elong/README.md.
    """
    rate = r_init * input_total * elong_factor
    return rate - rs_total * torch.log(rate + EPS)


def prepare(example, args):
    """Example dict -> (tensors, target counts).  Features cached in place."""
    if "_feats" not in example:
        example["_feats"] = featurize(
            example["utr_seq"],
            example["main_start_index"],
            near_cognate=args.near_cognate,
            structure=args.structure,
            context_seq=example.get("utr_context_seq"),
            context_offset=example.get("context_offset", 0),
            # Present when the cache was built with 03 --structure; folding
            # here instead would repeat ~29 minutes of work every run.
            paired=example.get("paired"),
            local_dG=example.get("local_dG"),
        )
    tensors = to_tensors(example["_feats"])

    counts = torch.as_tensor(np.asarray(example["ssu_counts"]), dtype=torch.float32)

    # Structure as a TARGET, from the cache built by 03 --structure.  Indexed
    # over the context, matching the rate net's output before cropping.
    #
    # local_dG is STANDARDISED over the cache, not just divided by 10 as it is
    # when the same track is used as an input.  Folding energy is strongly
    # negative -- around -10 kcal/mol -- while a freshly initialised head
    # outputs near zero, so the raw MSE starts about ten times the target's
    # own variance.  Weighted in at that scale the auxiliary term's early
    # gradient is dominated by learning an intercept, which teaches the trunk
    # nothing about structure and eats the budget meant for the chain.
    # Centred and scaled, a mean-predicting head scores ~1.0 and the two
    # auxiliary terms are commensurate with each other.
    if "paired" in example and "local_dG" in example:
        mu, sd = getattr(args, "_dG_stats", (0.0, 10.0))
        dg = torch.as_tensor(np.asarray(example["local_dG"]), dtype=torch.float32)
        tensors["struct_target"] = torch.stack([
            torch.as_tensor(np.asarray(example["paired"]), dtype=torch.float32),
            (dg - mu) / sd,
        ])

    # Commitment readout, if the cache carries it.  Genes with no Input reads
    # in the CDS window have no abundance estimate and contribute no term.
    readout = None
    if "rs_cds_counts" in example and "input_cds_counts" in example:
        rs_total = float(np.sum(example["rs_cds_counts"]))
        input_total = float(np.sum(example["input_cds_counts"]))
        if input_total > 0:
            factor = args._elong.get(example["transcript_id"], 1.0) if args._elong else 1.0
            readout = (torch.tensor(rs_total), torch.tensor(input_total),
                       torch.tensor(float(factor)))

    if args.normalize_input:
        # Divide out transcript abundance so what is left is occupancy per
        # molecule rather than occupancy times how much message there is.
        # Off by default: the choice stays visible instead of being forced.
        inp = torch.as_tensor(np.asarray(example["input_counts"]), dtype=torch.float32)
        scale = inp.mean().clamp(min=1.0)
        counts = counts / (inp / scale).clamp(min=0.1)
    return tensors, counts, readout


def anchor_rho_load(examples, pct=95.0, min_rs=10, min_input=5):
    """Pin rho_load from the data, using P(commit_main) <= 1.

    The 80S term says 80S ~ rho_load * P(commit_main) * abundance, so for
    every gene 80S/Input <= rho_load.  The most efficiently initiated gene in
    the dataset therefore sets a floor, and putting rho_load AT that floor is
    the assumption "the best gene commits on essentially every scan".

    A high percentile rather than the outright maximum, because the maximum of
    a ratio of small counts is noise.  This is a stated assumption, not a
    measurement -- it fixes the scale that the data cannot, and every
    P(commit_main) reported afterwards is relative to it.
    """
    r = []
    for e in examples:
        if "rs_cds_counts" not in e:
            continue
        rs = float(np.sum(e["rs_cds_counts"]))
        ip = float(np.sum(e["input_cds_counts"]))
        if rs >= min_rs and ip >= min_input:
            r.append(rs / ip)
    if not r:
        return None
    return float(np.percentile(r, pct))


def load_data(args):
    if args.data == "synthetic":
        from data.synthetic import make_dataset
        # Flanks must be at least the net's window, or scanner.py refuses.
        return make_dataset(n=args.n_synthetic, seed=args.seed,
                            length=args.utr_length, structure=args.structure,
                            near_cognate=args.near_cognate, flank=args.window)
    from data.archer import load_examples
    return load_examples(args.cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["synthetic", "archer"], default="synthetic")
    ap.add_argument("--cache", default="cache/archer_examples.npz")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--save-every", type=int, default=0, metavar="N",
                    help="also write <save>.ep<N>.pt every N epochs. Lets one "
                         "training run answer how many epochs the "
                         "REPRESENTATION needs, which the loss curve alone "
                         "does not say.")
    ap.add_argument("--batch", type=int, default=16, metavar="K",
                    help="accumulate gradients over K transcripts before "
                         "stepping. The forward pass stays per-transcript, so "
                         "this costs nothing and adds no padding. 1 reproduces "
                         "the pre-accumulation behaviour, whose run-to-run "
                         "spread in held-out R2 was about 0.16.")
    ap.add_argument("--window", type=int, default=10, help="rate net context, +/- nt")
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-synthetic", type=int, default=64)
    ap.add_argument("--utr-length", type=int, default=120)
    ap.add_argument("--structure", action="store_true",
                    help="fold UTRs with ViennaRNA (slow; zero tracks if absent)")
    ap.add_argument("--near-cognate", action="store_true",
                    help="let near-cognate codons open commit edges too")
    ap.add_argument("--normalize-input", action="store_true",
                    help="divide the SSU target by the input (abundance) track")
    ap.add_argument("--blur", type=float, default=0.0, metavar="W",
                    help="measurement blur on the PREDICTION before the loss: "
                         "kernel [W, 1-2W, W]. 0 = off. Blur width and leak "
                         "rate are confounded, so this is a sensitivity knob, "
                         "not a fitted parameter (see blur_profile)")
    ap.add_argument("--rho-load", default=None, metavar="V",
                    help="pin the global scale instead of learning it: a "
                         "number, or 'auto' for the 95th percentile of "
                         "80S/Input (= the best gene commits ~100%%). Learned "
                         "rho_load is unidentifiable -- see anchor_rho_load")
    ap.add_argument("--exclude-chroms", default="",
                    help="comma-separated chromosomes to drop from training, "
                         "e.g. 'II,V,X,XIV'. Their genes become a held-out set "
                         "for eval/score_akirtava.py --only-chroms, which turns "
                         "'the model saw these sequences' from a caveat into a "
                         "measured effect.")
    ap.add_argument("--save", metavar="PATH",
                    help="write the trained weights plus the config needed to "
                         "rebuild the model (window, hidden, rho_load). Without "
                         "this the run throws its weights away and nothing "
                         "downstream can evaluate the model.")
    ap.add_argument("--elong-offsets", metavar="PATH",
                    help="per-gene elongation correction for the commitment "
                         "term, from elong/fit.py. Optional and removable; "
                         "without it the term is unchanged. See elong/README.md")
    ap.add_argument("--struct-aux", type=float, default=0.0, metavar="W",
                    help="weight of an AUXILIARY loss that makes the rate "
                         "net predict local secondary structure from sequence "
                         "alone. The input stays one-hot -- nothing is folded "
                         "at inference. Needs a cache built with 03 "
                         "--structure. 0 = off.")
    ap.add_argument("--no-drop-at-starts", action="store_true",
                    help="drop-off is illegal at start codons, so leaving at "
                         "an AUG IS committing. Makes the commit/drop split "
                         "identifiable from the 40S occupancy profile alone, "
                         "which is what TCP-seq actually measures. Pair with "
                         "--commit-weight 0 to train on one assay and nothing "
                         "else. See model/chain.py.")
    ap.add_argument("--no-reinit", action="store_true",
                    help="switch off uORF reinitiation, restoring the chain "
                         "that treats every uORF as absorbing. The regression "
                         "baseline for the reinitiation change, not a knob.")
    ap.add_argument("--commit-weight", type=float, default=1.0,
                    help="weight of the 80S commitment term (0 = occupancy only; "
                         "fates are then not identified)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args._dG_stats = (0.0, 10.0)
    args._elong = None
    if args.elong_offsets:
        args._elong = load_elong_offsets(args.elong_offsets)
        print("[train] elongation offsets for {} genes from {}".format(
            len(args._elong), args.elong_offsets))
    examples = load_data(args)
    if args.exclude_chroms:
        drop = {c.strip() for c in args.exclude_chroms.split(",") if c.strip()}
        before = len(examples)
        examples = [e for e in examples if e.get("chrom") not in drop]
        print("[train] held out {}: {} -> {} examples".format(
            sorted(drop), before, len(examples)))
    assert examples, "no examples to train on"
    print("[train] {} examples, mean UTR length {:.0f}".format(
        len(examples), np.mean([len(e["utr_seq"]) for e in examples])))

    if args.struct_aux > 0 and "local_dG" in examples[0]:
        all_dg = np.concatenate([np.asarray(e["local_dG"]).ravel()
                                 for e in examples])
        mu, sd = float(all_dg.mean()), float(all_dg.std())
        args._dG_stats = (mu, sd if sd > 1e-6 else 1.0)
        print("[train] local dG standardised: mean {:.2f}, sd {:.2f} "
              "kcal/mol".format(mu, sd))

    rho = None
    if args.rho_load == "auto":
        rho = anchor_rho_load(examples)
        if rho is None:
            print("[train] no 80S readout in the cache; rho_load stays learned")
        else:
            print("[train] rho_load pinned at {:.3f} (95th pct of 80S/Input)".format(rho))
    elif args.rho_load is not None:
        rho = float(args.rho_load)
        print("[train] rho_load pinned at {:.3f}".format(rho))

    model = Scanner(window=args.window, hidden=args.hidden,
                    in_channels=NET_INPUT_CHANNELS, rho_load=rho,
                    reinit=not args.no_reinit,
                    drop_at_starts=not args.no_drop_at_starts)
    print("[train] reinitiation {}, drop-off at start codons {}".format(
        "off" if args.no_reinit else "on",
        "OFF (leaving at an AUG = commit)" if args.no_drop_at_starts else "on"))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    assert args.batch >= 1, "--batch must be at least 1"
    opt.zero_grad()
    n_accum = 0

    checked = False
    for epoch in range(args.epochs):
        t0 = time.time()
        total_loss, n_pos, skipped, n_ex = 0.0, 0, 0, 0
        commit_loss, n_commit = 0.0, 0
        order = np.random.default_rng(args.seed + epoch).permutation(len(examples))
        fate_sum = np.zeros(len(FATE_NAMES))
        r_init_sum = 0.0
        uorf_init_sum, reinit_sum = 0.0, 0.0
        struct_bce, struct_mse, n_struct = 0.0, 0.0, 0
        struct_floor, struct_var = 0.0, 0.0

        for idx in order:
            ex = examples[int(idx)]
            try:
                tensors, counts, readout = prepare(ex, args)
                out = model(tensors)
                loss = poisson_nll(blur_profile(out["profile"], args.blur), counts)
                if args.struct_aux > 0:
                    tgt = tensors.get("struct_target")
                    assert tgt is not None, (
                        "--struct-aux needs paired/local_dG in the cache; "
                        "rebuild with 03_build_examples.py --structure")
                    pred = out["struct_pred"]
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(
                        pred[0], tgt[0])
                    mse = torch.nn.functional.mse_loss(pred[1], tgt[1])
                    loss = loss + args.struct_aux * (bce + mse)
                    struct_bce += float(bce.detach())
                    struct_mse += float(mse.detach())
                    # The BCE a head that only learned this transcript's
                    # paired FRACTION would reach.  Without it the loss number
                    # is uninterpretable: 0.86 could mean the head reads
                    # structure off the sequence, or that it predicts a
                    # constant.  Only a BCE below this floor is the former,
                    # and only then is there anything for the trunk to carry.
                    with torch.no_grad():
                        q = tgt[0].mean().clamp(1e-6, 1 - 1e-6)
                        struct_floor += float(
                            -(q * q.log() + (1 - q) * (1 - q).log()))
                        # and the MSE of predicting this transcript's mean dG,
                        # which is the target's variance.  Same argument.
                        struct_var += float(tgt[1].var(unbiased=False))
                    n_struct += 1
                if readout is not None and args.commit_weight > 0:
                    c = commit_nll(out["r_init"], *readout)
                    loss = loss + args.commit_weight * c
                    commit_loss += float(c.detach())
                    n_commit += 1
            except AssertionError as e:
                # One malformed transcript must not take the whole run down.
                skipped += 1
                if skipped <= 3:
                    print("  [skip] {}: {}".format(ex["transcript_id"], e))
                continue

            # Scaled by 1/batch so the accumulated gradient is the MEAN over
            # the batch, which keeps --lr meaning the same thing at any K.
            (loss / args.batch).backward()
            n_accum += 1

            if not checked:
                # Inline sanity check, run once: gradients really do flow back
                # through torch.linalg.solve into f_theta.
                grads = [p.grad for p in model.rate_net.parameters() if p.grad is not None]
                assert grads, "no gradient reached the rate net"
                assert all(torch.isfinite(g).all() for g in grads), \
                    "non-finite gradient through the chain solve"
                assert any(g.abs().sum() > 0 for g in grads), "all gradients are zero"
                assert torch.all(out["profile"] >= 0), "negative occupancy"
                print("[train] grad check through the solve: ok")
                checked = True

            # Clipping applies to the averaged gradient, so the threshold is
            # per-batch and not per-transcript.
            if n_accum >= args.batch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
                opt.zero_grad()
                n_accum = 0

            total_loss += float(loss.detach())
            n_pos += len(counts)
            n_ex += 1
            fate_sum += out["fates"].detach().numpy()
            r_init_sum += float(out["r_init"].detach())
            uorf_init_sum += float(out["uorf_init_flux"].detach())
            reinit_sum += float(out["reinit_flux"].detach())

        # Flush a partial final batch rather than discarding it.  Dropping it
        # would quietly throw away up to batch-1 transcripts every epoch.
        if n_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            opt.zero_grad()
            n_accum = 0

        if n_ex == 0:
            print("epoch {:3d}  every example skipped".format(epoch))
            continue
        # Fates and r_init are averaged over the epoch; reading them off
        # whichever example happened to come last is pure noise.
        fates = fate_sum / n_ex
        print("epoch {:3d}  loss/nt {:9.4f}  commit/gene {:8.2f}  rho_load {:6.3f}  r_init {:7.4f}  {}  ({:.1f}s{})".format(
            epoch, total_loss / max(n_pos, 1),
            commit_loss / n_commit if n_commit else float("nan"),
            float(model.rho_load.detach()), r_init_sum / n_ex,
            "  ".join("{} {:.3f}".format(k, v) for k, v in zip(FATE_NAMES, fates)),
            time.time() - t0,
            ", {} skipped".format(skipped) if skipped else ""))
        if n_struct:
            bce_m = struct_bce / n_struct
            floor_m = struct_floor / n_struct
            mse_m, var_m = struct_mse / n_struct, struct_var / n_struct
            print("           structure aux  paired BCE {:.4f} (base rate "
                  "{:.4f}, {:+.1f}%)   dG MSE {:.4f} (variance {:.4f}, "
                  "{:+.1f}%)   weight {:g}".format(
                      bce_m, floor_m,
                      100.0 * (bce_m - floor_m) / floor_m if floor_m else 0.0,
                      mse_m, var_m,
                      100.0 * (mse_m - var_m) / var_m if var_m else 0.0,
                      args.struct_aux))
        if args.save and args.save_every and (epoch + 1) % args.save_every == 0:
            stem = args.save[:-3] if args.save.endswith(".pt") else args.save
            save_checkpoint("{}.ep{}.pt".format(stem, epoch + 1), model, args, rho)
        # uORF initiation events per walker, and how many of them came back.
        # commit_uORF is the difference; watching only the fate hides whether
        # the branch is firing at all.
        print("           uORF init {:.4f}/walker, of which reinitiated {:.4f} "
              "({:.0f}%)".format(
                  uorf_init_sum / n_ex, reinit_sum / n_ex,
                  100.0 * reinit_sum / uorf_init_sum if uorf_init_sum else 0.0))

    if args.save:
        save_checkpoint(args.save, model, args, rho)


def save_checkpoint(path, model, args, rho):
    """Weights + the config needed to rebuild the model, in one file."""
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": {"window": args.window, "hidden": args.hidden,
                   "in_channels": NET_INPUT_CHANNELS, "rho_load": rho,
                   "reinit": not args.no_reinit,
                   "drop_at_starts": not args.no_drop_at_starts},
        # Recorded so an evaluation can reproduce the training conditions
        # rather than guessing them.
        "train_args": {"data": args.data, "cache": args.cache,
                       "epochs": args.epochs, "blur": args.blur,
                       "commit_weight": args.commit_weight,
                       "near_cognate": args.near_cognate,
                       "structure": args.structure,
                       "exclude_chroms": args.exclude_chroms,
                       "elong_offsets": args.elong_offsets,
                       "struct_aux": args.struct_aux,
                       # seed and lr were missing, which made two checkpoints
                       # from different runs indistinguishable after the fact.
                       "seed": args.seed, "lr": args.lr, "batch": args.batch},
    }, path)
    print("[train] saved -> {}".format(path))


if __name__ == "__main__":
    main()
