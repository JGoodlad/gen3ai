"""Unit tests for the effective-rank probe (rank_metrics.py)."""
import numpy as np

from agents.training.rank_metrics import effective_rank, rank_probe


def test_full_rank_isotropic_is_high():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((2000, 16))          # isotropic → all 16 dims equally used
    r = effective_rank(Z)
    assert r["pr"] > 13.0                          # participation ratio ≈ D (16)
    assert r["effrank"] > 13.0
    assert r["n90"] >= 13 and r["n95"] >= 14 and r["n99"] >= 15


def test_rank_one_collapses_to_one():
    rng = np.random.default_rng(1)
    u = rng.standard_normal((2000, 1))
    v = rng.standard_normal((1, 16))
    Z = u @ v                                      # exact rank-1 (plus a constant offset)
    Z += rng.standard_normal((1, 16))              # a mean shift — centering removes it
    r = effective_rank(Z)
    assert r["pr"] < 1.5                            # one dominant direction
    assert r["effrank"] < 1.5
    assert r["n90"] == 1 and r["n95"] == 1 and r["n99"] == 1


def test_low_rank_between():
    rng = np.random.default_rng(2)
    # 3 strong directions + tiny noise → pr ≈ 3, most variance in 3 dims.
    base = rng.standard_normal((2000, 3)) * np.array([5.0, 4.0, 3.0])
    mix = rng.standard_normal((3, 16))
    Z = base @ mix + 0.01 * rng.standard_normal((2000, 16))
    r = effective_rank(Z)
    assert 2.0 < r["pr"] < 4.5
    assert r["n90"] <= 3


def test_degenerate_returns_zeros():
    assert effective_rank(np.zeros((10, 8)))["pr"] == 0.0        # zero variance
    assert effective_rank(np.ones((1, 8)))["n90"] == 0          # <2 rows


class _NonGen3Extractor:
    """No team_transformer / cls_pool → the probe must no-op gracefully."""


def test_rank_probe_non_gen3_returns_empty():
    assert rank_probe(_NonGen3Extractor(), obs={}, extract_features_fn=lambda o: (None, None)) == {}


def test_effective_rank_keys_present():
    r = effective_rank(np.random.default_rng(3).standard_normal((100, 8)))
    assert set(r) == {"pr", "effrank", "n90", "n95", "n99"}


# ------------------------------------------------------------- gen3_rank_probe_stash_v1 (K6)
def _hooked_reference(fe, obs):
    """The pre-K6 probe's capture (forward hooks), kept here as the equivalence reference."""
    import torch as th
    cap = {}
    h1 = fe.team_transformer.register_forward_hook(
        lambda _m, _i, o: cap.__setitem__("trunk", th.cat([o[0], o[1]], 1).reshape(-1, o[0].shape[-1])))
    h2 = fe.cls_pool.register_forward_hook(lambda _m, _i, o: cap.__setitem__("value_cls", o[3]))
    try:
        with th.no_grad():
            pi, vf = fe(obs)
    finally:
        h1.remove()
        h2.remove()
    cap["policy"], cap["vf_feat"] = pi, vf
    return cap


def _production_fe_and_obs(b=16):
    import torch as th
    from agents.model.compile_trainer import _parity_obs
    from agents.model.extra_obs_keys import zero_extra_obs
    from agents.model.extractor_compiles_test import _build_production_extractor
    fe, layout = _build_production_extractor()
    obs, _ = _parity_obs(layout["total_dim"], b, th.device("cpu"))
    obs.update(zero_extra_obs(fe, batch=b, device=th.device("cpu")))
    return fe.train(), obs


def test_the_hook_free_probe_reads_the_same_tensors_the_hooks_did():
    import torch as th
    fe, obs = _production_fe_and_obs()
    ref = _hooked_reference(fe, obs)
    got = rank_probe(fe, obs, fe)
    for name in ("trunk", "value_cls", "policy", "vf_feat"):
        r = effective_rank(ref[name].float().numpy())
        for k, v in r.items():
            if f"rank/{name}_{k}" in got:
                assert got[f"rank/{name}_{k}"] == float(v), (name, k)
    assert any(k.startswith("rank/trunk_") for k in got) and any(k.startswith("rank/value_cls_") for k in got)
    assert not fe.team_transformer._forward_hooks and not fe.cls_pool._forward_hooks
    del th


def test_the_probe_reaches_no_new_compiled_signature_after_the_lock():
    """THE iteration-1 signature `8fc297a2` absorbed: the hooked probe recompiled the learner's
    train/no-grad graph on torch 2.5.1 (`len(_forward_hooks) != 0`, on the resume frame after the
    `forward_guard` break). With the prewarm's declared train/no-grad signature compiled and the
    sentinel LOCKED, the probe must run clean AND measure every readout. Fails on a revert to hooks on
    BOTH torches: 2.5.1 rejects the recompile (the probe swallows it and returns {}); 2.8 silently
    skips the hooks inside the compiled frame, so the trunk / value_cls readings are missing."""
    import torch as th
    from agents.model import compile_control as cc
    fe, obs = _production_fe_and_obs()
    cc._reset_control_for_tests()
    th._dynamo.reset()
    try:
        ctl = cc.control()
        ctl.install()
        fe.forward = ctl.wrap_compiled(th.compile(fe.forward, backend="eager"))
        with th.no_grad():
            fe(obs)                                  # the declared rank-probe train/no-grad signature
        ctl.lock("prewarm (test)")
        got = rank_probe(fe, obs, fe)
        assert got, "the probe returned nothing — it hit a rejected recompile"
        assert ctl.compiles_after_lock == 0 and ctl.rejected_after_lock == 0, ctl.after_lock_frames
        # ...and it MEASURED the trunk and the value CLS through the compiled forward. On torch 2.8
        # the hooked probe did not recompile — dynamo does not guard (or run) a hook registered on a
        # submodule of an already-compiled frame (`skip_nnmodule_hook_guards`) — so its trunk /
        # value_cls readings silently VANISHED on every compiled run.
        assert any(k.startswith("rank/trunk_") for k in got), sorted(got)
        assert any(k.startswith("rank/value_cls_") for k in got), sorted(got)
    finally:
        cc._reset_control_for_tests()
        th._dynamo.config.error_on_recompile = False
        th._dynamo.reset()
