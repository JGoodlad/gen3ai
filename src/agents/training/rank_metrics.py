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

Pure NumPy math (unit-tested); the probe does ONE no_grad forward per train() call (cheap), mirroring
the grad-balance probe. Returns {} for a non-Gen3 extractor (no matching modules), like grad_balance.
"""
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


def rank_probe(features_extractor, obs, extract_features_fn) -> dict:
    """Capture the trunk / value_cls / policy reps on one minibatch and return their ``rank/*`` metrics.

    ``features_extractor`` = the Gen3 extractor (needs ``team_transformer`` + ``cls_pool``);
    ``obs`` = the minibatch obs dict; ``extract_features_fn`` = ``policy.extract_features`` (→ (pi, vf)).
    ONE no_grad forward. Returns {} for a non-Gen3 extractor.

    HOOK-FREE (gen3_rank_probe_stash_v1, K6): the trunk tokens and the value CLS are read from the
    extractor's own stashes (`last_trunk_tokens`, `last_value_cls`) — the same tensors the old forward
    hooks captured. A hook on a module dynamo compiled is a GUARD (`len(_forward_hooks) != 0`), so the
    hooked probe was a SECOND train/no-grad signature: its first call recompiled the learner graph
    after the prewarm, and the `8fc297a2` lock absorbed it on iteration 1 (found 2026-09-30). The
    probe's forward is now exactly the prewarm's declared `rank-probe train/no-grad` signature.
    """
    fe = features_extractor
    if not (hasattr(fe, "team_transformer") and hasattr(fe, "cls_pool")):
        return {}
    cap: dict = {}
    # A DIAGNOSTIC must never crash the run: any capture/forward failure → {} (nothing logged this call).
    try:
        with th.no_grad():
            pi_feat, vf_feat = extract_features_fn(obs)
            trunk = getattr(fe, "last_trunk_tokens", None)
            if trunk is not None:
                # TeamTransformer → (our_team_out [B,6,D], their_team_out [B,6,D]); rank over all 12 team tokens.
                cap["trunk"] = th.cat([trunk[0], trunk[1]], dim=1).reshape(-1, trunk[0].shape[-1]).detach()
            vcls = getattr(fe, "last_value_cls", None)
            if vcls is not None:
                cap["value_cls"] = vcls.detach()
            cap["policy"] = pi_feat.detach()
            # The POST-FiLM value features (what the critic MLP consumes) — the live version of
            # the offline vf-rank probe (tmp/vf_rank_probe.py): value_cls measures the pool
            # BEFORE the vf FiLM, so it cannot see whether the conditioning enriches or thins
            # the critic's actual input; this can (the crystallization-vs-division-of-labor read).
            cap["vf_feat"] = vf_feat.detach()
    except Exception:
        return {}

    out: dict = {}
    for name in ("trunk", "value_cls", "policy", "vf_feat"):
        Z = cap.get(name)
        if Z is None:
            continue
        r = effective_rank(Z.float().cpu().numpy())
        # value_cls/trunk/vf_feat report pr/effrank/n90/n95; policy additionally n99.
        keys = ("pr", "effrank", "n90", "n95", "n99") if name == "policy" else ("pr", "effrank", "n90", "n95")
        for k in keys:
            out[f"rank/{name}_{k}"] = float(r[k])
    return out
