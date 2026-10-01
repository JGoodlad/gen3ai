"""Effective-rank diagnostics for the shared-trunk representations (rank/* metrics).

A representation-rank probe found the model runs in far fewer effective dimensions than its width
(the "~3–5 of 128" trunk finding) — capacity isn't the constraint, so watching *how many* dims each
readout actually uses is the live signal for that. Measured at the three functional readouts:

  - **trunk**     — the shared TeamTransformer output tokens (the body both heads read; the
                    capacity-utilization signal).
  - **value_cls** — `value_pooled` (the value-dedicated CLS readout; known to run very low-rank —
                    ties to the critic tail-blindness).
  - **policy**    — `pi_features` (the actor's final rep; runs higher → the percentile counts
                    n90/n95/n99 are the informative view of how concentrated it is).

`effective_rank` is the NumPy reference (unit-tested; the offline capacity battery uses it).
`effective_rank_t` is the SAME spectrum on the representation's own DEVICE (K8, gen3_rank_device_v1):
float64 eigenvalues of the D x D centred Gram matrix (= the SVD's sigma^2), the five descriptors as one
tensor. The learner reads the four representations from the extractor's stashes of the micro-step's
OWN forward (`rank_probe_from_stash`, called right after region R1 on the first micro-batch) — no
second forward, no host copy of a [B*12, D] matrix, ONE host read for all twenty scalars. The
forward-based `rank_probe` stays for callers outside the learner. Returns {} for a non-Gen3 extractor
(no matching modules), like grad_balance.
"""
from typing import Any, Dict

import numpy as np
import torch as th


def effective_rank(Z: np.ndarray) -> dict:
    """Rank descriptors of the centered representation ``Z`` (shape [N, D]) via its covariance spectrum.

    Returns a dict of:
      pr       — participation ratio ``(Σσ²)² / Σσ⁴ = 1/Σpᵢ²`` (pᵢ = normalized eigenvalue): the
                 "effective number of equally-used dimensions" (1 = rank-1, D = isotropic).
      effrank  — entropy effective rank ``exp(-Σ pᵢ log pᵢ)`` (the spectral-entropy exponential).
      n90/n95/n99 — the number of leading dims needed to capture 90/95/99% of the variance
                 (the "relevant percentiles" view — how concentrated the representation is).
    """
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim != 2 or Z.shape[0] < 2 or Z.shape[1] < 1:
        return dict(pr=0.0, effrank=0.0, n90=0, n95=0, n99=0)
    Zc = Z - Z.mean(0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    var = s ** 2
    tot = float(var.sum())
    if tot <= 0.0:
        return dict(pr=0.0, effrank=0.0, n90=0, n95=0, n99=0)
    p = var / tot
    pr = float(1.0 / np.square(p).sum())
    effrank = float(np.exp(-(p * np.log(p + 1e-12)).sum()))
    cum = np.cumsum(p)
    n_at = lambda t: int((cum < t).sum()) + 1
    return dict(pr=pr, effrank=effrank, n90=n_at(0.90), n95=n_at(0.95), n99=n_at(0.99))


RANK_REPS = ("trunk", "value_cls", "policy", "vf_feat")
_KEYS = {"policy": ("pr", "effrank", "n90", "n95", "n99")}
_DEFAULT_KEYS = ("pr", "effrank", "n90", "n95")


def effective_rank_t(Z: th.Tensor) -> th.Tensor:
    """`effective_rank` on ``Z``'s own device, as one float64 tensor ``[pr, effrank, n90, n95, n99]``.

    The covariance spectrum is the eigenvalues of the centred Gram ``Zc^T Zc`` (D x D, float64) —
    the SVD's sigma^2 — sorted descending, clamped at 0 (eigvalsh's rounding can read a null
    direction as -1e-18). No host read. Same degenerate cases as the reference: < 2 rows or no
    variance -> all zeros."""
    Z = Z.detach().reshape(Z.shape[0], -1).to(th.float64)
    if Z.ndim != 2 or Z.shape[0] < 2 or Z.shape[1] < 1:
        return th.zeros(5, dtype=th.float64, device=Z.device)
    Zc = Z - Z.mean(0, keepdim=True)
    var = th.linalg.eigvalsh(Zc.T @ Zc).clamp(min=0.0).flip(0)
    tot = var.sum()
    p = var / th.where(tot > 0, tot, th.ones_like(tot))
    pr = 1.0 / th.where(tot > 0, th.square(p).sum(), th.ones_like(tot))
    effrank = th.exp(-(p * th.log(p + 1e-12)).sum())
    cum = th.cumsum(p, 0)
    ns = [(cum < t).sum().to(th.float64) + 1.0 for t in (0.90, 0.95, 0.99)]
    out = th.stack([pr, effrank, *ns])
    return th.where(tot > 0, out, th.zeros_like(out))


def rank_reps(fe: Any) -> Dict[str, th.Tensor]:
    """The four representations, DETACHED references, from the stashes of the forward that just
    ran (`trunk_tokens`, `value_cls`, `features_out`). {} for a non-Gen3 extractor; a missing stash
    is a missing key. Read them BEFORE any other forward of the same extractor (the fold's tail
    re-forwards and overwrites the stash)."""
    if not (hasattr(fe, "team_transformer") and hasattr(fe, "cls_pool")):
        return {}
    st: Any = getattr(fe, "stash", None)
    reps: Dict[str, th.Tensor] = {}
    trunk = getattr(st, "trunk_tokens", None)
    if trunk is not None:
        # TeamTransformer -> (our_team_out [B,6,D], their_team_out [B,6,D]); rank over all 12 team tokens.
        reps["trunk"] = th.cat([trunk[0], trunk[1]], dim=1).reshape(-1, trunk[0].shape[-1]).detach()
    vcls = getattr(st, "value_cls", None)
    if vcls is not None:
        reps["value_cls"] = vcls.detach()
    feats = getattr(st, "features_out", None)
    if feats is not None:
        reps["policy"] = feats[0].detach()
        # The POST-FiLM value features (what the critic MLP consumes): value_cls measures the pool
        # BEFORE the vf FiLM, so it cannot see whether the conditioning enriches or thins the
        # critic's actual input; this can (the crystallization-vs-division-of-labor read).
        reps["vf_feat"] = feats[1].detach()
    return reps


def rank_metrics_from_reps(reps: Dict[str, th.Tensor]) -> dict:
    """`rank/<rep>_<descriptor>` for every rep present — each spectrum on its own device, ALL of them
    read to the host in ONE transfer. value_cls / trunk / vf_feat report pr / effrank / n90 / n95;
    policy additionally n99 (the same keys the probe always logged)."""
    names = [n for n in RANK_REPS if n in reps]
    if not names:
        return {}
    vals = th.stack([effective_rank_t(reps[n]).to(reps[names[0]].device) for n in names])
    host = vals.cpu().numpy()                      # the ONE host read
    out: Dict[str, float] = {}
    for i, name in enumerate(names):
        for j, k in enumerate(("pr", "effrank", "n90", "n95", "n99")):
            if k in _KEYS.get(name, _DEFAULT_KEYS):
                out[f"rank/{name}_{k}"] = float(host[i, j])
    return out


def rank_probe_from_stash(fe: Any) -> dict:
    """The learner's probe (K8): the four reps from the micro-step's own forward, spectra on the
    device. A DIAGNOSTIC must never crash the run: any failure -> {} (nothing logged this call)."""
    try:
        return rank_metrics_from_reps(rank_reps(fe))
    except Exception:
        return {}


def rank_probe(features_extractor: Any, obs: Any, extract_features_fn: Any) -> dict:
    """The forward-based probe: ONE no_grad forward of ``obs`` through ``extract_features_fn``
    (``policy.extract_features`` -> (pi, vf)), then the reps from that forward's stashes. For callers
    outside the learner; the learner reads the micro-step's own forward (`rank_probe_from_stash`).

    HOOK-FREE (gen3_rank_probe_stash_v1, K6): the reps are the extractor's own stashes
    (`trunk_tokens`, `value_cls`, `features_out`) — a hook on a module dynamo compiled is a GUARD
    (`len(_forward_hooks) != 0`), so the old hooked probe was a SECOND train/no-grad signature: its
    first call recompiled the learner graph after the prewarm, and the `8fc297a2` lock absorbed it
    on iteration 1 (found 2026-09-30)."""
    fe = features_extractor
    if not (hasattr(fe, "team_transformer") and hasattr(fe, "cls_pool")):
        return {}
    try:
        with th.no_grad():
            extract_features_fn(obs)
        return rank_metrics_from_reps(rank_reps(fe))
    except Exception:
        return {}
