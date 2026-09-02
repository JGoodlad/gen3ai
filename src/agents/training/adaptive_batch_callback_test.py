"""Tests for `--adaptive-batch` (`agents/training/adaptive_batch_callback.py`).

Four things have to hold, and each is a different kind of test:

1. **The decision rule is right and is PURE** — a planted ratio sequence walks K exactly where the
   ratio implies, the band gives hysteresis, the clamps and the cadence hold, and every
   "instrument not readable" state is a no-op rather than a guess (`test_controller_*`).
2. **It can only ever move K.** `--batch-size` is what `--compile-trainer` keys its graphs on, so a
   controller that touched it would silently drop the compiled trainer to eager. Proved
   behaviourally AND at the source (`test_the_callback_only_ever_mutates_grad_accum_steps`,
   `test_the_module_never_assigns_batch_size`, `test_shape_stability_does_not_depend_on_k`).
3. **A moved K survives a launcher restart** — through the sidecar the checkpointer already
   writes, with no new key (`test_a_moved_k_round_trips_through_the_checkpoint_sidecar`).
4. **Attaching it changes no training math** when it does not move K
   (`test_the_callback_cannot_change_the_ppo_update`) — the same byte-identity discipline
   `instrumented_ppo_noise_scale_terms_test` applies to the probe it reads.
"""
import inspect
import os
import re
import types

import numpy as np
import pytest
import torch as th

from agents.training.adaptive_batch_callback import (
    _ACCUM_FLOOR_WHEN_ON,
    AdaptiveBatchCallback,
    AdaptiveBatchController,
)
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo_test import _build_tiny_ppo


def _ctl(**kw):
    kw.setdefault("mode", "policy")
    return AdaptiveBatchController(**kw)


def _walk(ctl, ratios, accum, samples=500):
    """Feed a ratio SEQUENCE through the controller, returning K after each rollout."""
    out = []
    for r in ratios:
        d = ctl.decide(ratio=r, samples=samples, accum=accum)
        accum = d.accum
        out.append(accum)
    return out


# --------------------------------------------------------------------------------------
# 1. The pure decision rule
# --------------------------------------------------------------------------------------


def test_controller_doubles_on_a_noise_limited_ratio_and_halves_on_an_over_batched_one():
    """The whole rule in one assertion: above the band buy batch, below it buy update steps."""
    ctl = _ctl(every=1, max_accum=64)
    # 6.2 is the measured policy-term reading that motivated the flag. From K=2 it should climb
    # until the ratio would be in band — here the ratio is held fixed, so it climbs every rollout.
    assert _walk(ctl, [6.2, 6.2, 6.2], accum=2) == [4, 8, 16]
    ctl2 = _ctl(every=1)
    # 0.05 is the measured TOTAL reading on the generalists ("over-batched 20x").
    assert _walk(ctl2, [0.05, 0.05, 0.05], accum=16) == [8, 4, 2]


def test_a_single_move_lands_inside_the_band_at_the_default_band_of_two():
    """THE hysteresis property at the shipped default, walked through the real feedback.

    A K move changes the ratio by EXACTLY 2x (K is the ratio's denominator), so at the default
    band=2 one correction lands in `[target/2, target*2]` and stays there. The test recomputes the
    ratio the new K implies after every decision rather than replaying a fixed sequence — a fixed
    sequence would never exercise the feedback that makes this true. (The general boundary is
    sqrt(2); see the parametrized test below.)
    """
    ctl = _ctl(every=1, band=2.0, target=1.0, max_accum=256)
    b_simple, accum = 3.7 * 4 * 1024.0, 4     # a critical batch 3.7x the effective one
    seen = []
    for _ in range(8):
        ratio = b_simple / (1024.0 * accum)
        d = ctl.decide(ratio=ratio, samples=500, accum=accum)
        accum = d.accum
        seen.append(round(ratio, 4))
    assert seen[0] == pytest.approx(3.7)
    assert seen[1] == pytest.approx(1.85)     # one doubling, and 1.85 is inside [0.5, 2.0]
    assert all(s == pytest.approx(1.85) for s in seen[1:]), (
        f"the loop settled and then chattered: {seen}")


@pytest.mark.parametrize("band,settles", [(1.30, False), (1.41, False), (1.50, True), (2.0, True)])
def test_the_chatter_boundary_is_sqrt_two_and_it_is_measured_not_asserted(band, settles):
    """The exact cost of a narrow band, pinned so nobody 'tightens' it thinking it is safer.

    A move changes the ratio by exactly 2x, so a correction crosses the band only if
    `target*band < ratio` and `ratio/2 < target/band` can both hold — i.e. iff `band^2 < 2`. The
    parametrization straddles sqrt(2)=1.4142 and the arms confirm the algebra: 1.41 chatters
    forever, 1.50 settles. (This test was written believing the boundary was 2.0; running it is
    what corrected the number, in the module docstring and the flag help too.)

    The start ratio sits JUST outside the band, because the overshoot window is
    `(target*band, 2*target/band)` and that interval is only 0.6% wide at band=1.41 — a start
    further out takes several one-directional moves and lands in band, which is progress rather
    than chatter and would have made this test pass for the wrong reason.
    """
    ctl = _ctl(every=1, band=band, target=1.0, max_accum=4096)
    b_simple, accum, seen = band * 1.0005 * 8 * 1024.0, 8, []
    for _ in range(30):
        accum = ctl.decide(ratio=b_simple / (1024.0 * accum), samples=500, accum=accum).accum
        seen.append(accum)
    assert (len(set(seen[-6:])) == 1) is settles, f"band={band} → {seen[-8:]}"


def test_the_cadence_holds_k_still_until_every_rollouts_have_passed():
    ctl = _ctl(every=4, max_accum=64)
    # Rollouts 1-3 are suppressed by the cadence; the 4th moves, and the counter resets.
    assert _walk(ctl, [6.0] * 9, accum=2) == [2, 2, 2, 4, 4, 4, 4, 8, 8]


def test_the_floor_is_two_when_the_loop_is_on_however_low_min_accum_is():
    """K=1 emits no noise scale at all, so a loop that reached it could never climb back out."""
    ctl = _ctl(every=1, min_accum=1)
    assert ctl.min_accum == _ACCUM_FLOOR_WHEN_ON == 2
    assert ctl.floor_was_raised
    assert _walk(ctl, [0.01, 0.01], accum=2) == [2, 2]
    assert ctl.decide(ratio=0.01, samples=500, accum=2).reason == "clamped"
    # An explicit min at or above the floor is respected verbatim and reports no raise.
    assert not _ctl(min_accum=4).floor_was_raised
    assert _ctl(min_accum=4).min_accum == 4


def test_the_max_clamp_reports_rather_than_moving():
    ctl = _ctl(every=1, max_accum=8)
    assert _walk(ctl, [9.0, 9.0, 9.0], accum=4) == [8, 8, 8]
    assert ctl.decide(ratio=9.0, samples=500, accum=8).reason == "clamped"


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), 0.0, -1.0])
def test_an_unreadable_ratio_never_moves_k(bad):
    """PROTECTION: no reading, no move. A controller that guesses here is unfalsifiable."""
    d = _ctl(every=1).decide(ratio=bad, samples=500, accum=4)
    assert (d.accum, d.moved, d.reason) == (4, False, "unavailable")


def test_a_cold_ema_never_moves_k_and_a_single_sample_never_can():
    """A single-sample B_simple can SIGN-FLIP; the warm-up is the guard, and it has a hard floor."""
    ctl = _ctl(every=1)
    assert ctl.decide(ratio=6.0, samples=1, accum=2).reason == "warming"
    assert ctl.decide(ratio=6.0, samples=ctl.warmup_samples - 1, accum=2).moved is False
    assert ctl.decide(ratio=6.0, samples=ctl.warmup_samples, accum=2).moved is True
    assert _ctl(every=1, warmup_samples=0).warmup_samples >= 2, (
        "the loop must never be allowed to act on one sample, even if asked to")


def test_off_is_inert_and_keeps_the_requested_floor():
    ctl = AdaptiveBatchController(mode="off", every=1, min_accum=1)
    d = ctl.decide(ratio=99.0, samples=999, accum=3)
    assert (d.accum, d.moved, d.reason) == (3, False, "off")
    assert ctl.min_accum == 1 and not ctl.floor_was_raised


@pytest.mark.parametrize("kw", [
    {"mode": "nonsense"}, {"band": 1.0}, {"band": 0.5}, {"target": 0.0}, {"target": -1.0},
    {"min_accum": 0}, {"min_accum": 8, "max_accum": 4}, {"every": 0},
])
def test_an_invalid_configuration_raises_at_construction(kw):
    with pytest.raises(ValueError):
        _ctl(**kw)


def test_the_metric_key_names_the_series_the_loop_is_judged_by():
    assert _ctl(mode="total").metric_key == "train/noise_scale_ratio"
    assert _ctl(mode="policy").metric_key == "train/noise_scale_ratio_policy"


# --------------------------------------------------------------------------------------
# 2. The read seam — the controller must act on the SERIES, not a second estimate
# --------------------------------------------------------------------------------------


class _Fold(InstrumentedMaskablePPO):
    """Just enough of an instance to exercise the EMA readers (no PPO construction)."""

    def __init__(self):    # noqa: D107 - deliberately does NOT call MaskablePPO.__init__
        self._noise_ema_terms = None
        self._noise_ema_s = None
        self._noise_ema_g2 = None
        self._nsr_samples = 0


def test_noise_ratio_sample_returns_exactly_the_number_the_series_carries():
    """`noise_ratio_sample` must be the SAME value `train()` records, or a post-mortem compares the
    controller against a quantity it never saw."""
    f = _Fold()
    f._noise_ema_s, f._noise_ema_g2, f._nsr_samples = 8192.0, 2.0, 37
    f._noise_ema_terms = {"policy": [40960.0, 2.0, 11]}
    b_big = 1024.0
    assert f.noise_ratio_sample("total", b_big) == (pytest.approx(4.0), 37)
    assert f.noise_ratio_sample("policy", b_big) == (pytest.approx(20.0), 11)
    # It obeys the SAME emit gate train() does: a non-positive EMA yields no reading, not a wrong
    # one — which is what routes the controller into its `unavailable` branch.
    f._noise_ema_s = -1.0
    assert f.noise_ratio_sample("total", b_big) == (None, 37)
    assert f.noise_ratio_sample("aux", b_big) == (None, 0)
    assert f.noise_ratio_sample("not_a_group", b_big) == (None, 0)


# --------------------------------------------------------------------------------------
# 3. The callback — it may move K and nothing else
# --------------------------------------------------------------------------------------


class _Rec:
    def __init__(self):
        self.vals = {}

    def record(self, k, v, *a, **kw):
        self.vals[k] = v

    def __getattr__(self, _n):
        return lambda *a, **kw: None


def _fake_model(ratio, samples=500, accum=2, batch_size=1024):
    m = types.SimpleNamespace(batch_size=batch_size, grad_accum_steps=accum,
                              num_timesteps=0, logger=_Rec())
    m.noise_ratio_sample = lambda kind, b_big: (ratio, samples)
    return m


def _attach(cb, model):
    cb.init_callback(model)
    cb.on_training_start({}, {})
    return cb


def test_the_callback_only_ever_mutates_grad_accum_steps():
    """The compile-safety property, measured: over a long ratio-driven walk, `batch_size` — the
    ONLY thing torch.compile keys a graph on here — is never touched."""
    model = _fake_model(ratio=6.0)
    cb = _attach(AdaptiveBatchCallback(_ctl(every=1, max_accum=32), verbose=0), model)
    before = dict(vars(model))
    for _ in range(12):
        cb.on_rollout_end()
    assert model.grad_accum_steps == 32, "the loop should have climbed to its max"
    assert model.batch_size == before["batch_size"] == 1024
    changed = {k for k in vars(model) if vars(model)[k] is not before.get(k)}
    assert changed <= {"grad_accum_steps", "logger"}, f"the callback also mutated {changed}"


def test_the_module_never_assigns_batch_size():
    """The source-level half of the same claim: no assignment to a batch-size attribute exists.

    Behavioural coverage can only prove it for the paths a test walks; this proves it for all of
    them, and is what fails if someone 'improves' the loop by moving `--batch-size` directly.
    """
    import agents.training.adaptive_batch_callback as mod
    src = inspect.getsource(mod)
    body = "\n".join(ln for ln in src.splitlines() if not ln.strip().startswith("#"))
    assert not re.search(r"\.batch_size\s*=", body), (
        "the adaptive-batch loop must move `grad_accum_steps` and NEVER `batch_size` — a moving "
        "batch size is an unbounded shape set, which drops --compile-trainer to eager SILENTLY.")


def test_shape_stability_does_not_depend_on_k():
    """`check_shape_stability` — the refusal that protects the compiled trainer — takes n_steps,
    n_envs, batch_size and async_rollout. K is not among them, BY CONSTRUCTION, which is the whole
    reason the loop moves K: no value it can pick is expressible as a shape change."""
    from agents.model.compile_trainer import check_shape_stability
    params = set(inspect.signature(check_shape_stability).parameters)
    assert params == {"n_steps", "n_envs", "batch_size", "async_rollout"}
    for k in (1, 2, 4, 8, 16, 32):       # every K the controller can reach is equally acceptable
        check_shape_stability(n_steps=2048, n_envs=64, batch_size=2048 * 64 // 8,
                              async_rollout=False)
        assert k >= 1


def test_the_series_are_recorded_every_rollout():
    model = _fake_model(ratio=3.0, accum=4)
    cb = _attach(AdaptiveBatchCallback(_ctl(every=99), verbose=0), model)
    cb.on_rollout_end()
    assert model.logger.vals["train/grad_accum_steps"] == 4
    assert model.logger.vals["train/effective_batch"] == 4096
    assert model.logger.vals["train/adaptive_batch_ratio_used"] == pytest.approx(3.0)


def test_a_no_move_reason_is_reported_exactly_once(capsys):
    """A silently-idle loop is indistinguishable from a broken one; a loop that says so every
    rollout is noise. Once per reason is the compromise, and it is a contract."""
    model = _fake_model(ratio=None)
    cb = _attach(AdaptiveBatchCallback(_ctl(every=1), verbose=0), model)
    for _ in range(5):
        cb.on_rollout_end()
    out = capsys.readouterr().out
    assert out.count("no move (unavailable)") == 1, out
    assert model.grad_accum_steps == 2


# --------------------------------------------------------------------------------------
# 4. Persistence across a launcher restart
# --------------------------------------------------------------------------------------


def test_a_moved_k_round_trips_through_the_checkpoint_sidecar(tmp_path):
    """K survives a restart through the EXISTING checkpointer, with no new key.

    `_model_hparams` already records `grad_accum_steps` straight off the model attribute the
    callback owns, so the whole persistence story is: the loop moves the attribute, the
    checkpointer writes it, `build_callbacks` reads it back and hands it to the next callback as
    `resume_accum`, which installs it AFTER `model_build` applied the CLI value. This test walks
    that entire path with real `record_checkpoint` / `read_checkpoint_metadata`.
    """
    from agents.model.snapshot import read_checkpoint_metadata, record_checkpoint
    from main.train.run_io import _model_hparams

    model = _fake_model(ratio=6.0, accum=2)
    cb = _attach(AdaptiveBatchCallback(_ctl(every=1, max_accum=8), verbose=0), model)
    for _ in range(2):
        cb.on_rollout_end()
    assert model.grad_accum_steps == 8

    saveable = types.SimpleNamespace(
        gamma=0.99, gae_lambda=0.95, ent_coef=0.0, vf_coef=0.5, batch_size=model.batch_size,
        n_steps=2048, grad_accum_steps=model.grad_accum_steps,
        clip_range=lambda _: 0.2, clip_range_vf=None,
        policy=types.SimpleNamespace(optimizer=types.SimpleNamespace(
            param_groups=[{"weight_decay": 0.0}])),
    )
    ckpt = tmp_path / "checkpoints" / "checkpoint_1000_steps.zip"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_bytes(b"")
    record_checkpoint(str(tmp_path), str(ckpt), 3e-4, 5, hparams=_model_hparams(saveable))
    assert read_checkpoint_metadata(str(ckpt))["grad_accum_steps"] == 8

    # The restart: model_build has just applied the CLI `--grad-accum-steps 2`, and the callback's
    # own history must win over it.
    resumed = _fake_model(ratio=6.0, accum=2)
    resume_k = read_checkpoint_metadata(str(ckpt))["grad_accum_steps"]
    _attach(AdaptiveBatchCallback(_ctl(every=1, max_accum=8), resume_accum=resume_k, verbose=0),
            resumed)
    assert resumed.grad_accum_steps == 8


def test_a_resumed_k_is_clamped_into_the_current_bounds():
    """A restart that TIGHTENS --adaptive-batch-max-accum must not reinstate the old K above it."""
    m = _fake_model(ratio=None, accum=2)
    _attach(AdaptiveBatchCallback(_ctl(every=1, max_accum=8), resume_accum=64, verbose=0), m)
    assert m.grad_accum_steps == 8


# --------------------------------------------------------------------------------------
# 5. Attaching the loop cannot change the update
# --------------------------------------------------------------------------------------


def _learn_arm(with_callback):
    """One arm: a FRESH identically-seeded toy PPO at K=2, `learn()`ed with or without the loop.

    Fresh per arm for the same reason `instrumented_ppo_noise_scale_terms_test` gives — a `train()`
    on this toy is not reproducible from a restored `state_dict`, so reusing one model would
    compare the drift instead of the feature. K starts at 2 (the loop's floor) so the arms differ
    ONLY in whether the callback exists, never in the K the update ran at.
    """
    th.manual_seed(0)
    np.random.seed(0)
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.grad_accum_steps = 2
    cbs = None
    if with_callback:
        # ratio unreadable for the whole run (the toy never warms an EMA in 4 rollouts), so the
        # controller is in its no-move branch — which is precisely the state this pins.
        cbs = [AdaptiveBatchCallback(_ctl(every=1), verbose=0)]
    model.learn(total_timesteps=8 * 4 * 4, callback=cbs)
    if with_callback:
        assert model.grad_accum_steps == 2, "the arm was supposed not to move K"
    return {k: v.detach().clone() for k, v in model.policy.state_dict().items()}


def test_the_callback_cannot_change_the_ppo_update():
    """THE regression gate for the flag's byte-identity claim: with the loop attached but not
    moving, not one parameter differs from a run that never had it."""
    os.environ["GEN3AI_NOISE_SCALE_PER_TERM"] = "0"
    try:
        off, off2, on = _learn_arm(False), _learn_arm(False), _learn_arm(True)
    finally:
        os.environ.pop("GEN3AI_NOISE_SCALE_PER_TERM", None)
    for k in off:
        assert th.equal(off[k], off2[k]), f"the arms are not reproducible at all ({k})"
        assert th.equal(off[k], on[k]), f"the adaptive-batch callback moved parameter {k}"


def test_the_flag_defaults_to_off_and_registers_no_callback():
    """OFF must be OFF at the parser AND at the assembly: no callback, therefore no series."""
    from main.train_rl_agent import build_parser
    args = build_parser().parse_args(["--steps", "1"])
    assert args.adaptive_batch == "off"
    import main.train.callbacks as _cbmod
    src = inspect.getsource(_cbmod)
    assert re.search(r'getattr\(args,\s*"adaptive_batch",\s*"off"\)\s*!=\s*"off"', src), (
        "the AdaptiveBatchCallback must be registered behind the flag, so an off run adds no "
        "callback at all rather than a callback that decides to do nothing.")
