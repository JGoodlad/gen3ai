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
    ONE no_grad forward; hooks are always removed. Returns {} for a non-Gen3 extractor.
    """
    fe = features_extractor
    if not (hasattr(fe, "team_transformer") and hasattr(fe, "cls_pool")):
        return {}
    cap: dict = {}

    def _trunk_hook(_m, _i, o):
        # TeamTransformer → (our_team_out [B,6,D], their_team_out [B,6,D]); rank over all 12 team tokens.
        cap["trunk"] = th.cat([o[0], o[1]], dim=1).reshape(-1, o[0].shape[-1]).detach()

    def _cls_hook(_m, _i, o):
        # CLSPool → (our_pooled, their_pooled, our_active_refined, value_pooled); index 3 = value CLS.
        cap["value_cls"] = o[3].detach()

    # A DIAGNOSTIC must never crash the run: any capture/forward failure → {} (nothing logged this call).
    try:
        h1 = fe.team_transformer.register_forward_hook(_trunk_hook)
        h2 = fe.cls_pool.register_forward_hook(_cls_hook)
        try:
            with th.no_grad():
                pi_feat, _vf_feat = extract_features_fn(obs)
                cap["policy"] = pi_feat.detach()
        finally:
            h1.remove()
            h2.remove()
    except Exception:
        return {}

    out: dict = {}
    for name in ("trunk", "value_cls", "policy"):
        Z = cap.get(name)
        if Z is None:
            continue
        r = effective_rank(Z.float().cpu().numpy())
        # value_cls/trunk report pr/effrank/n90/n95; policy additionally n99 (the "relevant percentiles").
        keys = ("pr", "effrank", "n90", "n95", "n99") if name == "policy" else ("pr", "effrank", "n90", "n95")
        for k in keys:
            out[f"rank/{name}_{k}"] = float(r[k])
    return out
