"""`--diagnostics-every N` (gen3_diagnostics_cadence_v1, M5 Lane K2) — the guarantee and the regime.

THE GUARANTEE (the decisive test here): on a REAL Gen3 extractor + dual-head policy, one PPO update
with every optional probe ON and the same update with them all SKIPPED leave bit-identical
parameters, optimizer state, loss scalars and RNG state. It fails the day a diagnostic draws a
random number, writes `.grad`, touches a parameter or steps the optimizer. A third arm (the ON arm
again) is the reproducibility control, so an ON/OFF equality cannot be two arms that are simply
never different; and the ON arm must actually EMIT every gated family, so the equality is not
vacuous either.

Then the regime: skipped updates write NONE of the gated tags; the plan (first update of a process,
the rollout-index phase, the two load-bearing exemptions); the flag's resolve / inherit / migrate
surface; the latch never rides a checkpoint; `checkargs` reports it.
"""
from __future__ import annotations

import inspect
import random
import types

import numpy as np
import pytest
import torch as th

from agents.training import learner_benchmark as lb
from agents.training.instrumented_ppo import diagnostics_cadence as dc
from main.train.parser import build_parser

GATED_PREFIXES = ("grad/", "rank/", "edge/", "cell/")
PER_TERM_TAGS = ("train/noise_scale_ratio_policy", "train/noise_scale_policy",
                 "train/noise_scale_share_policy", "train/noise_per_term_ms")
EVERY_UPDATE_TAGS = ("train/loss", "train/value_loss", "train/policy_gradient_loss",
                     "train/approx_kl", "train/clip_fraction", "train/grad_norm",
                     "train/approx_kl_epoch_0", "train/clip_fraction_epoch_0", "train/train_ms")


def _gated(keys):
    return sorted(k for k in keys if k.startswith(GATED_PREFIXES) or k in PER_TERM_TAGS)


# ── the plan (pure) ───────────────────────────────────────────────────────────────────────────
def _m(every=10, ran=True, steps=0, rank=False, noise=False):
    m = types.SimpleNamespace(diagnostics_every=every, n_steps=8, n_envs=2, num_timesteps=steps,
                              rank_probe_every_update=rank, noise_terms_every_update=noise)
    if ran:
        m._diagnostics_ran_in_process = True
    return m


def test_the_first_update_of_a_process_is_always_a_diagnostic_update():
    m = _m(every=10, ran=False, steps=16 * 3)        # rollout index 3: not on the cadence
    plan = dc.plan_for(m)
    assert plan.due and plan.noise_terms and plan.grad_balance and plan.rank and plan.liveness
    dc.mark_ran(m, plan)
    assert not dc.plan_for(m).due                    # the latch is what makes the next one skip


def test_the_phase_is_the_restored_rollout_index():
    due = [dc.plan_for(_m(every=4, steps=16 * k)).due for k in range(1, 13)]
    assert due == [k % 4 == 0 for k in range(1, 13)]
    assert all(dc.plan_for(_m(every=1, steps=16 * k)).due for k in range(1, 6))


def test_a_skipped_update_runs_no_gated_probe_unless_a_consumer_declared_it_load_bearing():
    off = dc.plan_for(_m(every=10, steps=16 * 3))
    assert off == dc.DiagnosticsPlan(False, False, False, False, False)
    rank = dc.plan_for(_m(every=10, steps=16 * 3, rank=True))
    assert rank.rank and not (rank.due or rank.grad_balance or rank.liveness or rank.noise_terms)
    noise = dc.plan_for(_m(every=10, steps=16 * 3, noise=True))
    assert noise.noise_terms and not (noise.due or noise.rank or noise.grad_balance)


def test_the_exemptions_are_derived_from_the_predicates_that_register_the_consumers():
    import main.train.callbacks as cb
    import main.train.model_build as mb
    cb_src, mb_src = inspect.getsource(cb), inspect.getsource(mb)
    assert 'getattr(args, "rank_tripwire", "warn") != "off"' in cb_src
    assert 'model.rank_probe_every_update = getattr(args, "rank_tripwire", "warn") != "off"' in mb_src
    assert 'getattr(args, "adaptive_batch", "off") != "off"' in cb_src
    assert ('model.noise_terms_every_update = getattr(args, "adaptive_batch", "off") '
            'not in ("off", "total")') in mb_src
    assert 'model.diagnostics_every = int(getattr(args, "diagnostics_every", None) or 1)' in mb_src


# ── the flag surface: resolve, inherit, migrate, record ───────────────────────────────────────
def _resolved(argv):
    from main.train.config import resolve_config
    p = build_parser()
    args = p.parse_args(argv)
    resolve_config(args, p)
    return args


def test_the_argparse_default_is_None_and_a_fresh_run_resolves_to_the_production_default():
    assert build_parser().parse_args([]).diagnostics_every is None
    assert _resolved([]).diagnostics_every == dc.DIAGNOSTICS_EVERY_DEFAULT == 10
    assert _resolved(["--diagnostics-every", "1"]).diagnostics_every == 1
    with pytest.raises(SystemExit):
        _resolved(["--diagnostics-every", "0"])


def test_a_flagless_resume_INHERITS_the_recorded_cadence_and_a_typed_one_overrides_it():
    from agents.model.model_version import ModelVersion
    from main.train.config import inherit_saved_flag
    saved = ModelVersion.__new__(ModelVersion)
    saved.diagnostics_every = 1
    args = build_parser().parse_args([])
    assert inherit_saved_flag(args, saved, "diagnostics_every", 10) is True
    assert args.diagnostics_every == 1
    args = build_parser().parse_args(["--diagnostics-every", "25"])
    assert inherit_saved_flag(args, saved, "diagnostics_every", 10) is False
    assert args.diagnostics_every == 25
    import main.train.config as cfg
    assert '_resolve("diagnostics_every", DIAGNOSTICS_EVERY_DEFAULT)' in inspect.getsource(cfg)


def test_it_is_a_RECORDED_field_and_a_pre_v124_config_migrates_to_every_update():
    from agents.model.model_version import ModelVersion
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    from agents.model.model_version.construct import ModelVersionConstruction
    from agents.model.model_version.migrations import _migrate_config
    assert ModelVersion.__dataclass_fields__["diagnostics_every"].default == 1
    fn = ModelVersionConstruction.from_layout_and_policy_kwargs
    assert "diagnostics_every=int(diagnostics_every)" in inspect.getsource(fn)
    assert MODEL_CONFIG_VERSION >= 124
    out = _migrate_config({"config_version": 123})
    assert out["diagnostics_every"] == dc.DIAGNOSTICS_EVERY_PRE_V124 == 1
    assert out["config_version"] >= 124
    assert _migrate_config({"config_version": 123, "diagnostics_every": 10})["diagnostics_every"] == 10
    import main.train.model_build as mb
    assert inspect.getsource(mb).count("diagnostics_every=args.diagnostics_every,") == 2


def test_checkargs_reports_the_inherited_cadence(tmp_path):
    """`main.checkargs` resolves an argv the way the launch does: a flagless resume of a pre-v124
    parent (the production mirror, a real recorded config) INHERITS 1; a typed value wins."""
    import json

    from agents.model.model_version import ModelVersion
    from main.checkargs import resolve_against_parent
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    # The mirror's config FIELDS (K10(a): its `recipe` block is not a model_config.json field).
    from agents.training.baselines import production_config
    (run / "model_config.json").write_text(json.dumps(production_config(), indent=2))
    assert ModelVersion.from_json_file(str(run / "model_config.json")).diagnostics_every == 1
    ckpt = run / "checkpoints" / "ckpt.zip"
    ckpt.write_bytes(b"")
    out = resolve_against_parent(["--model", str(ckpt)])
    assert out["inherited"].get("diagnostics_every") == 1, out.get("read_error") or out["tried"]
    typed = resolve_against_parent(["--model", str(ckpt), "--diagnostics-every", "10"])
    assert "diagnostics_every" not in typed["inherited"] and typed["ns"].diagnostics_every == 10


def test_the_process_latch_never_rides_a_checkpoint():
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    m = InstrumentedMaskablePPO.__new__(InstrumentedMaskablePPO)
    assert "_diagnostics_ran_in_process" in m._excluded_save_params()


# ── THE GUARANTEE: bit-identical learning, on a real Gen3 policy ─────────────────────────────
def _real_gen3_ppo(device: str = "cpu"):
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.action.constants import ACTION_SPACE_SIZE
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    class _Env(gym.Env):
        def __init__(self, dim, seed):
            self.observation_space = spaces.Dict({
                "observation": spaces.Box(0.0, 1.0, (dim,), np.float32),
                "action_mask": spaces.Box(0, 1, (ACTION_SPACE_SIZE,), np.int8)})
            self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
            self._d, self._t, self._seed = dim, 0, seed

        def _row(self):
            return (self._t * 5 + self._seed * 17) % len(_ROWS)

        def _o(self):
            # gen3_fresh_parity_probe_v1: the committed REAL-obs fixture rows and their masks, not
            # an all-zero obs — a zero obs has no valid move seat, so the seat projections, edge
            # maps and everything behind them never received a gradient (measured below).
            return {"observation": _ROWS[self._row()].astype(np.float32),
                    "action_mask": _MASKS[self._row()].astype(np.int8)}

        def reset(self, **kw):
            self._t = 0
            return self._o(), {}

        def step(self, a):
            # A deterministic, action-dependent reward and episode length: non-zero advantages,
            # so the update genuinely moves the weights and the probes see real gradients.
            self._t += 1
            r = float(((int(a) + self._t + self._seed) % 5) - 2)
            return self._o(), r, self._t >= 3 + self._seed, False, {}

        def action_masks(self):
            return _MASKS[self._row()].astype(bool)

    # THE PRODUCTION SURFACE (`main.fresh_checkpoint`'s kwargs — every edge family and pointer
    # cell production builds), so every gated probe has its real modules to read.
    from main.fresh_checkpoint import _production_policy_kwargs
    _args, layout, pk = _production_policy_kwargs()
    from agents.model.compile_parity_fixture import load_parity_rows
    _ROWS, _MASKS = load_parity_rows(layout["total_dim"])
    # The SCALAR critic: the win-prob critic's BCE needs outcome labels that only the full
    # callback stack attaches, and without them the grad-balance probe (correctly) waits for a
    # minibatch that has a win-prob term — for ever, on this toy. The probes' read-only-ness does
    # not depend on which critic term they measure.
    pk = {**pk, "critic": "shaped"}
    th.manual_seed(0)
    model = InstrumentedMaskablePPO(
        Gen3DualHeadMaskablePolicy,
        DummyVecEnv([(lambda s=s: _Env(layout["total_dim"], s)) for s in range(2)]),
        n_steps=8, batch_size=4, n_epochs=2, device=device, seed=0, policy_kwargs=pk)
    from agents.training.rust_rollout.testkit import attach_vec_collector
    attach_vec_collector(model)         # the rollout comes from the toy VecEnv (the Rust collector is production's)
    # gen3_fresh_parity_probe_v1: and off the fresh zero-init weights — the shared seeded
    # perturbation (private RNG, so the seeded stream below is unchanged) opens the zero-init
    # projections' paths from the first minibatch on. `test_learning_is_BIT_IDENTICAL…` asserts
    # the coverage both buy.
    from agents.model.parity_probe import perturb_
    perturb_(model.policy)
    # …and no win-prob BCE term at all, for the same reason. (A `--win-prob-coef 0` once switched
    # the term off here; that flag is deleted, so the head's MODE is set to "none" AFTER the build —
    # the head keeps its modules, the fold simply does not score it.)
    model.policy.features_extractor.win_prob_mode = "none"
    model.grad_accum_steps = 2          # the accumulation branch: the noise probes need it
    orig = InstrumentedMaskablePPO.train
    InstrumentedMaskablePPO.train = lambda self: None    # stop at the first update's buffer
    try:
        model.learn(total_timesteps=8 * 2)
    finally:
        InstrumentedMaskablePPO.train = orig
    return model


def _rng_state():
    return th.get_rng_state().clone(), np.random.get_state(), random.getstate()


def _one_update(model, pristine, state0, *, skip: bool):
    lb.restore_buffer_state(model.rollout_buffer, pristine)
    lb.restore_model_state(model, state0)
    lb.seed_all(7)
    model.logger.name_to_value.clear()
    model.rank_probe_every_update = False
    model.noise_terms_every_update = False
    if skip:
        model.diagnostics_every = 2 ** 31 - 1
        model._diagnostics_ran_in_process = True
    else:
        model.diagnostics_every = 1
    model.train()
    params = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    opt = model.policy.optimizer.state_dict()
    opt_tensors = {(i, k): v.detach().clone() for i, st in opt["state"].items()
                   for k, v in st.items() if th.is_tensor(v)}
    return params, opt_tensors, dict(model.logger.name_to_value), _rng_state()


@pytest.fixture(scope="module")
def three_arms():
    from agents.training.instrumented_ppo.noise_scale_terms import NOISE_TERM_GROUPS
    model = _real_gen3_ppo()
    # Prime both noise-scale EMA families to a post-warm-up state (value AND count), as the
    # per-term test does: one sample on a 16-row toy can read negative, which the emit gate
    # correctly withholds — and then the ON arm could not show that the per-term probe ran.
    model._noise_ema_s, model._noise_ema_g2, model._noise_ema_n = 50.0, 2.0, 500
    model._noise_ema_terms = {g: [50.0, 2.0, 500] for g in NOISE_TERM_GROUPS}
    pristine = lb.capture_buffer_state(model.rollout_buffer)
    state0 = lb.capture_model_state(model)
    on = _one_update(model, pristine, state0, skip=False)
    on_again = _one_update(model, pristine, state0, skip=False)
    off = _one_update(model, pristine, state0, skip=True)
    state0 = {**state0, "param_names": sorted(n for n, _ in model.policy.named_parameters())}
    return state0, on, on_again, off


def test_the_ON_arm_really_runs_every_gated_probe(three_arms):
    _, on, _, _ = three_arms
    logged = on[2]
    for prefix in ("grad/", "rank/", "edge/", "cell/"):
        assert any(k.startswith(prefix) for k in logged), f"no {prefix}* tag: the ON arm is vacuous"
    for tag in PER_TERM_TAGS:
        assert tag in logged, tag


def test_learning_is_BIT_IDENTICAL_with_the_diagnostics_on_and_skipped(three_arms):
    state0, on, on_again, off = three_arms
    p_on, o_on, l_on, r_on = on
    # The control: two ON arms from the restored state agree bit for bit …
    for k in p_on:
        assert th.equal(p_on[k], on_again[0][k]), f"restore does not reproduce the update ({k})"
    # … and the update really moved the weights, so the equalities below are not vacuous — not
    # just SOME weight: nearly every parameter (gen3_fresh_parity_probe_v1). Measured 2026-09-29,
    # parameters that never moved in the ON update: all-zero obs + fresh weights 106/254 (so "a
    # diagnostic changed parameter k" could not fail for them), zero obs + perturbed 107, real
    # rows + fresh 22, real rows + perturbed 19 (the belief/win heads, whose loss terms this toy
    # does not train, and two edge maps).
    assert any(not th.equal(p_on[k], state0["policy"][k]) for k in p_on)
    names = state0["param_names"]
    unmoved = [k for k in names if th.equal(p_on[k], state0["policy"][k])]
    assert len(unmoved) <= 0.1 * len(names), (
        f"{len(unmoved)}/{len(names)} parameters never moved in the update — the bit-identity "
        f"check is VACUOUS for them (fresh zero-init weights?): {unmoved[:8]}")
    p_off, o_off, l_off, r_off = off
    for k in p_on:
        assert th.equal(p_on[k], p_off[k]), f"a diagnostic changed parameter {k}"
    assert o_on.keys() == o_off.keys() and o_on
    for k in o_on:
        assert th.equal(o_on[k], o_off[k]), f"a diagnostic changed optimizer state {k}"
    for tag in EVERY_UPDATE_TAGS:
        if tag == "train/train_ms":
            assert tag in l_off
            continue
        assert l_on[tag] == l_off[tag], f"a diagnostic perturbed {tag}"
    # THE RNG: a probe that drew one random number leaves a different generator behind.
    assert th.equal(r_on[0], r_off[0]), "a diagnostic consumed the torch RNG"
    assert r_on[1][0] == r_off[1][0] and np.array_equal(r_on[1][1], r_off[1][1]) \
        and r_on[1][2:] == r_off[1][2:], "a diagnostic consumed the numpy RNG"
    assert r_on[2] == r_off[2], "a diagnostic consumed python's random"


def test_a_skipped_update_writes_NONE_of_the_gated_tags_and_every_per_update_one(three_arms):
    _, on, _, off = three_arms
    assert _gated(on[2]), "the ON arm logged no gated tag"
    assert _gated(off[2]) == [], f"a skipped update wrote {_gated(off[2])}"
    for tag in EVERY_UPDATE_TAGS:
        assert tag in off[2], tag
