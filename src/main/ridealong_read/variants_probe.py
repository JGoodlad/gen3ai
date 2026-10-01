"""The TORCH half of the RND variants' read: per-key errors, the identification probes, the
predictor-reset floor, and feat's drift scoring. The statistics are :mod:`.variants` (pure NumPy).

* :func:`novelty_errors` — every key's RAW error on given rows: the observation keys (``rnd`` and the
  obs variants) through BASE's normaliser and frozen target (shared, so paired), ``rndv_feat``
  through its own feature normalisation and target over ``value_pooled``.
* :func:`build_probes` — (e)'s two FIXED probe sets: (i) ``variants.N_PROBE`` bank rows chosen with a
  fixed seed in a seeded shuffled order, (ii) ``block_chimera`` of (i) in that order (``chimera_v1``,
  block edges from ``obs_block_edges`` over the encoder's layout — never hardcoded). Chimera rows go
  through every key's normaliser / target as any row does (per-dimension normalisation and the ±5
  clip commute with block copying). For ``feat`` the extractor is run on the chimera observations;
  their action mask is the mask of the row that donated the CONTEXT block (the block whose start is
  ``layout['parts']['context']['start']``).
* :func:`floor_heads` — the same heads with every PREDICTOR reset to its declared init: fresh heads
  are built with the SAME spec (``build_ridealong``, private RNG) and only predictor weights are
  copied over a deep copy of the read heads, so the checkpoint's own normalisers and targets stay.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.ridealong_read import variants as VA
from main.ridealong_read.boot import BOOT_SEED, N_BOOT

ERR_BATCH = 2048


def key_modules(heads: Any) -> Dict[str, Any]:
    """``{"rnd": base, "rndv_<name>": variant}`` for every RND module the heads carry."""
    out: Dict[str, Any] = {}
    if getattr(heads, "rnd", None) is not None:
        out[VA.BASE] = heads.rnd
        rv = getattr(heads, "rnd_variants", None)
        for n, m in (rv.items() if rv is not None else ()):
            out[VA.variant_key(n)] = m
    return out


def is_feat(module: Any) -> bool:
    decl = getattr(module, "decl", None)
    return decl is not None and getattr(decl, "input", None) == "feat"


def novelty_errors(heads: Any, obs: Optional[np.ndarray], pooled: Optional[np.ndarray],
                   batch: int = ERR_BATCH) -> Dict[str, np.ndarray]:
    """Per key the raw error [N] (float64). Observation keys need ``obs``; ``rndv_feat`` needs
    ``pooled`` (a key whose input is absent is left out)."""
    import torch

    mods = key_modules(heads)
    out: Dict[str, list] = {}
    with torch.no_grad():
        if obs is not None and VA.BASE in mods:
            for i in range(0, len(obs), batch):
                x, tgt = heads.rnd.inputs(torch.from_numpy(np.ascontiguousarray(obs[i:i + batch],
                                                                                dtype=np.float32)))
                for k, m in mods.items():
                    if not is_feat(m):
                        out.setdefault(k, []).append(m.error_from(x, tgt).numpy())
        if pooled is not None:
            for k, m in mods.items():
                if is_feat(m):
                    out[k] = [feat_err(m, pooled, batch)]
    return {k: np.concatenate(v).astype(np.float64) for k, v in out.items()}


def feat_err(feat: Any, pooled: np.ndarray, batch: int = ERR_BATCH) -> np.ndarray:
    """The feat head's raw error over ``pooled`` [N, D_MODEL] (float64)."""
    import torch

    res = []
    with torch.no_grad():
        for i in range(0, len(pooled), batch):
            res.append(feat.error(torch.from_numpy(np.ascontiguousarray(pooled[i:i + batch],
                                                                        dtype=np.float32))).numpy())
    return np.concatenate(res).astype(np.float64) if res else np.zeros(0)


def fit_fresh_stats(heads: Any, cols: Any, fit_mask: np.ndarray) -> None:
    """FRESH heads only: ``feat``'s feature statistics fitted on the TRAIN rows' ``value_pooled``
    (AFTER the forward), feat re-scored, then every variant's error z-statistics fitted on the TRAIN
    rows and its z recomputed (base's are fitted by ``forward_columns`` as before)."""
    import torch

    for k, m in key_modules(heads).items():
        if k == VA.BASE:
            continue
        if is_feat(m):
            m.update_obs_stats(torch.from_numpy(np.ascontiguousarray(cols.pooled[fit_mask],
                                                                     dtype=np.float32)))
            cols.rndv[f"{k}_err"] = feat_err(m, cols.pooled).astype(np.float32)
        e = cols.rndv[f"{k}_err"]
        m.update_err_stats(torch.from_numpy(np.ascontiguousarray(e[fit_mask], dtype=np.float32)))
        cols.rndv[f"{k}_z"] = m.zscore(torch.from_numpy(np.ascontiguousarray(e))).numpy()


# ---------------------------------------------------------------------------------------------
# the probes
# ---------------------------------------------------------------------------------------------

def probe_layout(fe: Any) -> Tuple[dict, str]:
    """The observation layout: the extractor's own when it carries ``parts``, else the encoder's."""
    lay = getattr(fe, "layout", None)
    if isinstance(lay, dict) and isinstance(lay.get("parts"), dict):
        return lay, "features_extractor.layout"
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    return Gen3ObservationEncoder(load_mappings()).get_layout(), "Gen3ObservationEncoder.get_layout()"


def context_block_index(layout: Mapping[str, Any], edges: Sequence[int]) -> int:
    start = int(layout["parts"]["context"]["start"])
    if start not in edges:
        raise ValueError(f"the context block's start {start} is not a block edge {list(edges)}")
    return list(edges).index(start)


def build_probes(rows: np.ndarray, masks: np.ndarray, battles: np.ndarray, ids: Sequence[str],
                 layout: Mapping[str, Any], layout_source: str) -> dict:
    """(i) and (ii) with their masks, clusters and provenance (deterministic; draws no torch RNG)."""
    import hashlib

    import torch

    from agents.model.ridealong_heads import block_chimera, obs_block_edges

    obs_dim = int(rows.shape[1])
    edges = obs_block_edges(layout, obs_dim)
    idx = VA.probe_indices(len(rows))
    rows_i = np.ascontiguousarray(rows[idx], dtype=np.float32)
    rows_ii = block_chimera(torch.from_numpy(rows_i), edges).numpy()
    donors = VA.chimera_donors(len(idx), edges)
    kc = context_block_index(layout, edges)
    masks_ii = np.asarray(masks)[idx][donors[kc]]
    return {"idx": idx, "rows_i": rows_i, "rows_ii": rows_ii, "masks_i": np.asarray(masks)[idx],
            "masks_ii": masks_ii, "clusters": np.asarray(battles)[idx], "edges": edges,
            "provenance": {
                "n": int(len(idx)), "n_requested": VA.N_PROBE, "seed": VA.PROBE_SEED,
                "order": "rows drawn without replacement with the seed, then a seeded permutation",
                "probe_ids_sha256": hashlib.sha256("\n".join(ids[i] for i in idx).encode()).hexdigest(),
                "generator": VA.CHIMERA_GENERATOR, "edges": list(edges), "layout_source": layout_source,
                "context_block_index": kc,
                "chimera_action_mask": f"the mask of the row that donated the context block (block "
                                       f"{kc}, starting at layout['parts']['context']['start'] = "
                                       f"{edges[kc]})",
                "cluster": "probe row j (both (i) and (ii)) = the battle of (i)'s row j"}}


def floor_heads(heads: Any, fe: Any, obs_dim: int) -> Tuple[Any, dict]:
    """``(heads0, checks)``: a deep copy of ``heads`` with every RND PREDICTOR reset to its declared
    init (fresh heads of the same spec, private RNG); normalisers, targets and error statistics are
    the read heads' own. ``checks``: whether `decay`'s frozen anchor equals the rebuilt init (it is
    the declared init by construction, so a mismatch means the rebuild did not reproduce it)."""
    import torch

    from agents.model.ridealong_heads import build_ridealong

    fresh = build_ridealong(fe, obs_dim=int(obs_dim), spec=heads.spec)
    if fresh is None:
        raise RuntimeError("build_ridealong returned None for the read heads' spec")
    h0 = copy.deepcopy(heads)
    fm, hm = key_modules(fresh), key_modules(h0)
    for k, m in hm.items():
        m.predictor.load_state_dict(fm[k].predictor.state_dict())
    checks: Dict[str, Any] = {"decay_anchor_matches_rebuilt_init": None}
    dec = hm.get(VA.variant_key("decay"))
    if dec is not None and getattr(dec, "anchor", None) is not None:
        anc = dict(heads.rnd_variants["decay"].anchor.named_buffers())
        checks["decay_anchor_matches_rebuilt_init"] = all(
            torch.equal(anc[n], p) for n, p in fm[VA.variant_key("decay")].predictor.named_parameters())
    return h0, checks


def identification(heads: Any, fe: Any, probes: Mapping[str, Any], pooled_i: Optional[np.ndarray],
                   pooled_ii: Optional[np.ndarray]) -> dict:
    """(e) for one checkpoint: every key's raw error on (i) and (ii), with the read heads and with
    the predictor-reset floor, then :func:`variants.identification_read`."""
    h0, checks = floor_heads(heads, fe, int(probes["rows_i"].shape[1]))
    e_i = novelty_errors(heads, probes["rows_i"], pooled_i)
    e_ii = novelty_errors(heads, probes["rows_ii"], pooled_ii)
    e_i0 = novelty_errors(h0, probes["rows_i"], pooled_i)
    e_ii0 = novelty_errors(h0, probes["rows_ii"], pooled_ii)
    archs = {k: v["predictor_vs_target"]
             for k, v in VA.declarations(list(getattr(heads, "variant_names", lambda: ())())).items()}
    errs = {k: {"i": e_i[k], "ii": e_ii[k], "i0": e_i0[k], "ii0": e_ii0[k]} for k in e_i}
    out = VA.identification_read(errs, probes["clusters"], archs)
    out["probe"] = probes["provenance"]
    out["floor"] = {"rule": "the same heads with every RND predictor reset to its declared init "
                            "(build_ridealong with the same spec, private RNG; predictor weights "
                            "copied over a deep copy, so normalisers and targets are the "
                            "checkpoint's own)", **checks}
    return out


# ---------------------------------------------------------------------------------------------
# one checkpoint
# ---------------------------------------------------------------------------------------------

def read_variants(heads: Any, fe: Any, decisions: Sequence[dict], cols: Any, rows: np.ndarray,
                  masks: np.ndarray, label: str, path: str, heads_prov: str,
                  pooled_fn: Callable[[np.ndarray, np.ndarray], np.ndarray]) -> Tuple[dict, VA.Carry]:
    """The per-checkpoint JSON block (a, b, c, e) and the :class:`variants.Carry` the
    cross-checkpoint reads (c's ratios, d, e's series) use."""
    import torch

    keys = [VA.BASE] + [f"rndv_{n}" for n in getattr(heads, "variant_names", lambda: ())()] \
        if getattr(heads, "rnd", None) is not None else []
    carry = VA.Carry(label=label, path=str(path), heads=heads_prov, keys=keys)
    if not keys:
        return {"skipped": "the heads carry no RND"}, carry
    errs = {k: cols.err(k).astype(np.float64) for k in keys}
    zs = {k: cols.z(k).astype(np.float64) for k in keys}
    body, carry.spread = VA.per_checkpoint(decisions, cols.v, errs, zs)
    carry.errs = errs
    battles = np.array([d["battle"] for d in decisions])
    layout, src = probe_layout(fe)
    probes = build_probes(rows, masks, battles, [d["id"] for d in decisions], layout, src)
    he = tuple(int(e) for e in getattr(heads, "block_edges", ()) or ())
    probes["provenance"]["heads_block_edges_match"] = he == tuple(probes["edges"])
    pooled_ii = pooled_fn(probes["rows_ii"], probes["masks_ii"]) if "rndv_feat" in keys else None
    with torch.no_grad():
        ident = identification(heads, fe, probes,
                               cols.pooled[probes["idx"]] if pooled_ii is not None else None,
                               pooled_ii)
    body["e_identification"] = ident
    carry.ident = ident
    if "rndv_feat" in keys:
        carry.feat = copy.deepcopy(heads.rnd_variants["feat"]).cpu().eval()
    return {"keys": keys, "declarations": VA.declarations([k[5:] for k in keys if k != VA.BASE]),
            "bootstrap": {"B": N_BOOT, "seed": BOOT_SEED, "paired": "every key on the SAME battle "
                          "resamples (boot.weight_chunks)"},
            "alpha": VA.ALPHA, **body}, carry


def feat_through(feat: Any, pooled_b: np.ndarray, train_mask: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """(d): A's feat head over B's pooled — with A's frozen feature statistics, and with them
    re-fitted on B's TRAIN-row pooled (``rnd_choice.renormalised``; all rows if no TRAIN rows)."""
    from main.ridealong_read.rnd_choice import renormalised

    m = train_mask if train_mask is not None and bool(np.asarray(train_mask).any()) else \
        np.ones(len(pooled_b), bool)
    return feat_err(feat, pooled_b), feat_err(renormalised(feat, pooled_b[m]), pooled_b)


__all__ = ["key_modules", "novelty_errors", "feat_err", "fit_fresh_stats", "probe_layout",
           "context_block_index", "build_probes", "floor_heads", "identification", "read_variants",
           "feat_through"]
