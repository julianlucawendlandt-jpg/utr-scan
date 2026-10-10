"""Glue: features -> rate net -> absorbing chain -> outputs."""

import numpy as np
import torch
import torch.nn as nn

from model.chain import COMMIT_MAIN, run_chain
from model.rate_net import RateNet


def to_tensors(feats, device=None, dtype=torch.float32):
    """Feature dict (numpy) -> the tensors the model consumes."""
    t = {
        "net_input": torch.as_tensor(feats["net_input"], dtype=dtype, device=device),
        "start_mask": torch.as_tensor(np.asarray(feats["start_mask"]), dtype=torch.bool, device=device),
        "is_main_start": torch.as_tensor(np.asarray(feats["is_main_start"]), dtype=torch.bool, device=device),
        "context_offset": int(feats.get("context_offset", 0)),
        "n_states": int(feats.get("n_states", len(feats["start_mask"]))),
    }
    # Absent from features built before reinitiation existed; the model then
    # runs without return edges rather than failing.
    if "uorf_return" in feats and "uorf_geom" in feats:
        t["uorf_return"] = torch.as_tensor(
            np.asarray(feats["uorf_return"]), dtype=torch.int64, device=device)
        t["uorf_geom"] = torch.as_tensor(
            np.asarray(feats["uorf_geom"]), dtype=dtype, device=device)
    return t


def load_state_compat(model, state_dict):
    """Load weights, tolerating a checkpoint from before the reinit head.

    Nothing else is allowed to be missing.  strict=False on its own would
    silently accept a checkpoint with, say, the wrong hidden size half-loaded;
    this refuses everything except the one key set we know how to default.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    allowed = ("rate_net.reinit_head.", "rate_net.struct_head.")
    stale = [k for k in missing if not k.startswith(allowed)]
    assert not stale, "checkpoint is missing weights: {}".format(stale)
    assert not unexpected, "checkpoint has unknown weights: {}".format(unexpected)
    if missing:
        print("[model] checkpoint predates reinitiation; reinit head left at "
              "its init (sigmoid 0.5). Pass reinit=False to reproduce it.")
    return model


class Scanner(nn.Module):
    """One 40S particle scanning one UTR.

    rho_load is a single global scalar shared across every transcript: the
    conversion between "probability of committing per scan" and "80S
    ribosomes observed per mRNA copy".  r_init = rho_load * P(commit_main) is
    the initiation rate the chain implies, and train.py compares
    r_init * (mRNA abundance) against 80S footprints in the coding region.
    That comparison is what gives rho_load, and the commit/drop split, a
    gradient.
    """

    def __init__(self, window=10, hidden=32, in_channels=None, rho_load=None,
                 reinit=True, drop_at_starts=True):
        super().__init__()
        kwargs = {"hidden": hidden, "window": window}
        if in_channels is not None:
            kwargs["in_channels"] = in_channels
        self.rate_net = RateNet(**kwargs)

        # reinit=False restores the pre-reinitiation chain exactly: no return
        # edges, every uORF initiation absorbed.  It is the regression
        # baseline, not a tuning knob.
        self.reinit = bool(reinit)

        # False = a 40S that leaves at an AUG has committed.  This is the
        # structural alternative to anchoring the fates on 80S/Input, which
        # measures whole-gene efficiency rather than initiation; see the
        # identifiability note in model/chain.py.
        self.drop_at_starts = bool(drop_at_starts)

        # rho_load is either learned or pinned.  Learned, it is unidentifiable:
        # the 80S term constrains only the PRODUCT rho_load * P(commit_main),
        # so the optimiser slides freely along that curve and P(commit_main)
        # -- the number we actually want -- lands wherever rho_load drifted to.
        # Pinning it is one stated assumption that makes the fate readable.
        self.fixed_rho_load = rho_load
        if rho_load is None:
            # Raw parameter, softplus'd on read so it stays positive.
            self._rho_load_raw = nn.Parameter(torch.tensor(0.5413))  # -> ~1.0
        else:
            self.register_buffer("_rho_fixed", torch.tensor(float(rho_load)))

    @property
    def rho_load(self):
        if self.fixed_rho_load is not None:
            return self._rho_fixed
        return nn.functional.softplus(self._rho_load_raw)

    def forward(self, tensors):
        """-> dict with profile (N,), fates (4,), r_init (scalar), w, p, d, r."""
        w, p, d, struct = self.rate_net(tensors["net_input"])

        # Crop the rates from context coordinates back to the chain's states.
        off, n = tensors["context_offset"], tensors["n_states"]
        win = self.rate_net.window
        assert off >= win and (w.shape[0] - off - n) >= win, (
            "flanks too narrow: the rate net looks +/-{} nt but the context "
            "carries only {} nt before and {} after the chain's {} states. "
            "Zero-padding would reach the outputs and position would leak "
            "back into the rates.".format(
                win, off, w.shape[0] - off - n, n))
        w, p, d = w[off:off + n], p[off:off + n], d[off:off + n]

        # The reinitiation logit comes from uORF geometry alone and shares no
        # parameters with the rate net's trunk.  It is computed in chain
        # coordinates because the geometry is measured relative to the main
        # AUG, which only exists there.
        r, uorf_return = None, None
        if self.reinit and "uorf_return" in tensors:
            uorf_return = tensors["uorf_return"]
            r = self.rate_net.reinit_logit(tensors["uorf_geom"])

        profile, fates, aux = run_chain(
            w, p, d, tensors["start_mask"], tensors["is_main_start"],
            r=r, uorf_return=uorf_return, return_aux=True,
            drop_at_starts=self.drop_at_starts)
        # The chain works in float64 for conditioning; come back to the
        # model's dtype so the loss and the rest of the net stay float32.
        profile, fates = profile.to(w.dtype), fates.to(w.dtype)
        return {
            "profile": profile,
            "fates": fates,
            "r_init": self.rho_load * fates[COMMIT_MAIN],
            "w": w,
            "p": p,
            "d": d,
            "r": r,
            # commit_uORF counts only the walkers that never came back, so
            # these are what to watch when asking whether the uORF branch is
            # doing anything.
            "uorf_init_flux": aux["uorf_init_flux"].to(w.dtype),
            "reinit_flux": aux["reinit_flux"].to(w.dtype),
            # Auxiliary structure prediction over the FULL context, for
            # train.py --struct-aux.  Nothing in the chain consumes it.
            "struct_pred": struct,
        }
