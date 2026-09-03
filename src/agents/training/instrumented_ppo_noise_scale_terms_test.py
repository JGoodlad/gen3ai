"""Tests for the PER-LOSS-TERM gradient noise scale (`instrumented_ppo/noise_scale_terms.py`).

Three things have to hold, and each is a different kind of test:

1. **The math is the same math.** The per-group fold must recover a PLANTED `tr(Σ)` / `|G|²` per
   group exactly, through the same two-point solve the total uses — `test_per_term_fold_*`.
2. **The probe cannot change training.** With the feature OFF, one real `train()` from a captured
   init must produce a BIT-IDENTICAL parameter update and the identical total scalars; and even
   ON, `add()` must hand back the very object it was given — `test_off_is_byte_identical*`,
   `test_add_returns_the_same_tensor_object`.
3. **The advisor says the interesting thing.** A total that reads over-batched while the policy
   term does not is the finding; it has to become its own warning naming the deflation — the
   `test_advice_*` family.
"""
import copy
import os
import re

import numpy as np
import pytest
import torch as th

from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo.constants import _NOISE_SCALE_EMA_DECAY
from agents.training.instrumented_ppo.noise_scale import debiased_ema
from agents.training.instrumented_ppo.noise_scale_terms import (
    NOISE_TERM_GROUPS,
    NULL_TAGGER,
    PerTermNoiseSampler,
    per_term_enabled,
)
from agents.training.instrumented_ppo_test import _build_tiny_ppo, _train_from_init


# --------------------------------------------------------------------------------------
# 1. The estimator, per group — a planted (tr(Σ), |G|²) must come back out
# --------------------------------------------------------------------------------------


class _Fold(InstrumentedMaskablePPO):
    """Just enough of an instance to call the pure EMA fold (no PPO construction)."""

    def __init__(self):    # noqa: D107 - deliberately does NOT call MaskablePPO.__init__
        self._noise_ema_terms = None


@pytest.mark.parametrize("b_small,b_big", [(64, 256), (512, 1024)])
def test_per_term_fold_recovers_a_planted_ratio_per_group(b_small, b_big):
    """Each group's `(g_small_sq, g_big_sq)` is built from EXACT expectations
    `E‖Ĝ_B‖² = |G|² + tr(Σ)/B` with a DIFFERENT planted `B_simple` per group. The first fold has no
    prior EMA, so the recovered ratio is the planted one exactly — which is the point: the per-term
    numbers must be the same estimator on a different gradient, never a different estimator."""
    planted = {"policy": (2.0, 4000.0), "value": (5.0, 50.0), "aux": (10.0, 5.0)}
    per_term = {g: (g2 + s / b_small, g2 + s / b_big) for g, (g2, s) in planted.items()}

    out = _Fold()._fold_per_term_noise(per_term, float(b_small), float(b_big), total_g2=None)

    for group, (g2, s) in planted.items():
        assert out[f"train/noise_scale_{group}"] == pytest.approx(s / g2, rel=1e-6)
        assert out[f"train/noise_scale_ratio_{group}"] == pytest.approx(s / g2 / b_big, rel=1e-6)
    # aux plants the LOWEST critical batch (5/10 = 0.5) and policy the highest (4000/2 = 2000) —
    # i.e. the exact confound this module exists to expose, and the fold keeps them apart.
    assert out["train/noise_scale_aux"] < out["train/noise_scale_policy"]


def test_per_term_share_is_over_the_total_and_is_not_a_partition():
    """`share` = |G_g|² / |G_total|². It is NOT normalized to sum to 1 — `|G_total|²` carries the
    cross terms — and the test pins that on purpose, so nobody 'fixes' it into a pie chart."""
    b_small, b_big = 100.0, 400.0
    planted = {"policy": (3.0, 30.0), "aux": (9.0, 9.0)}
    per_term = {g: (g2 + s / b_small, g2 + s / b_big) for g, (g2, s) in planted.items()}

    out = _Fold()._fold_per_term_noise(per_term, b_small, b_big, total_g2=4.0)

    assert out["train/noise_scale_share_policy"] == pytest.approx(3.0 / 4.0, rel=1e-6)
    assert out["train/noise_scale_share_aux"] == pytest.approx(9.0 / 4.0, rel=1e-6)   # > 1, legal
    assert sum(v for k, v in out.items() if "_share_" in k) > 1.0


def test_per_term_fold_withholds_a_group_whose_ema_is_not_yet_positive():
    """A single-sample estimate can come out negative under noise (the reason the EMAs exist). A
    group in that state must emit NOTHING rather than a garbage `B_simple` — the same gate the
    total applies."""
    # g_big_sq > g_small_sq ⇒ tr(Σ) < 0: a bigger batch measured NOISIER, which only sampling
    # noise produces.
    out = _Fold()._fold_per_term_noise({"policy": (1.0, 2.0)}, 10.0, 40.0, total_g2=1.0)
    assert "train/noise_scale_policy" not in out
    assert "train/noise_scale_ratio_policy" not in out
    assert "train/noise_scale_share_policy" in out    # the share is still readable


def test_per_term_ratio_reads_the_ema_state_not_the_last_fold():
    fold = _Fold()
    assert fold._per_term_ratio("policy", 1000.0) is None            # nothing folded yet
    fold._noise_ema_terms = {"policy": [500.0, 2.0, 7]}              # B_simple = 250
    assert fold._per_term_ratio("policy", 1000.0) == pytest.approx(0.25)
    assert fold._per_term_ratio("aux", 1000.0) is None               # group never sampled
    fold._noise_ema_terms = {"policy": [-1.0, 2.0, 7]}               # unwarmed / negative tr(Σ)
    assert fold._per_term_ratio("policy", 1000.0) is None


def test_the_warm_up_is_a_running_mean_not_an_anchor_on_the_first_sample():
    """The per-group EMA must not be hostage to sample 1. A strongly noise-limited policy term has
    |G|² ≈ 0 at production batch sizes, so its single-sample estimate SIGN-FLIPS — a plain EMA that
    starts on one negative sample suppresses `train/noise_scale_ratio_policy` for hundreds of calls,
    which reads as "the probe is broken" rather than "the term is noisy". Two samples that average
    positive must therefore report positive."""
    fold = _Fold()
    b_small, b_big = 100.0, 200.0
    # sample 1: |G|² = 2*g_big - g_small = -1 (negative); sample 2: +3. Mean = +1.
    fold._fold_per_term_noise({"policy": (5.0, 2.0)}, b_small, b_big, None)
    assert fold._noise_ema_terms["policy"][1] == pytest.approx(-1.0)
    out = fold._fold_per_term_noise({"policy": (5.0, 4.0)}, b_small, b_big, None)
    assert fold._noise_ema_terms["policy"][1] == pytest.approx(1.0)     # a MEAN, not 0.99*(-1)+...
    assert fold._noise_ema_terms["policy"][2] == 2
    assert "train/noise_scale_ratio_policy" in out


# --------------------------------------------------------------------------------------
# 2. The sampler — real gradients, known answers
# --------------------------------------------------------------------------------------


def _tiny_net():
    th.manual_seed(0)
    return th.nn.Linear(3, 1, bias=False)


def test_sampler_recovers_the_two_batch_sizes_from_real_gradients():
    """Two micro-batches, two groups. `small_sq` is micro 0's own gradient; `big_sq` is
    ‖(g0+g1)/accum‖² — exactly what `.grad` holds for the total after `(loss/accum).backward()`
    twice. Both are checked against gradients computed independently."""
    net = _tiny_net()
    params = list(net.parameters())
    xs = [th.tensor([[1.0, 2.0, 3.0]]), th.tensor([[-4.0, 1.0, 0.5]])]

    def terms(x):
        y = net(x).sum()
        return {"policy": y, "value": 2.0 * (y ** 2)}

    sampler = PerTermNoiseSampler(params)
    manual = {"policy": [], "value": []}
    for x in xs:
        t = terms(x)
        for g, term in t.items():
            manual[g].append(th.autograd.grad(term, params, retain_graph=True)[0].clone())
            sampler.add(g, term)
        sampler.flush_micro()

    res = sampler.result(accum=2)
    assert set(res) == {"policy", "value"}
    for g, (small_sq, big_sq) in res.items():
        assert small_sq == pytest.approx(float(manual[g][0].pow(2).sum()), rel=1e-5)
        expect_big = float(((manual[g][0] + manual[g][1]) / 2.0).pow(2).sum())
        assert big_sq == pytest.approx(expect_big, rel=1e-5)


def test_sampler_yields_nothing_for_a_partial_group():
    """A KL early-stop discards the accumulation group; without its second point the estimate has
    no `B = batch_size·accum` reading, so the sampler must return {} rather than a wrong one."""
    net = _tiny_net()
    sampler = PerTermNoiseSampler(list(net.parameters()))
    sampler.add("policy", net(th.ones(1, 3)).sum())
    sampler.flush_micro()
    assert sampler.micros == 1
    assert sampler.result(accum=2) == {}          # one micro of a two-micro group
    assert sampler.result(accum=1) == {}          # accum<2 has no second batch size at all


def test_sampler_drops_a_group_that_first_appears_on_a_later_micro_batch():
    """A term folded on micro 1 but not micro 0 has no matched small-batch point. Reporting it
    would silently compare two different batch sizes' worth of terms."""
    net = _tiny_net()
    sampler = PerTermNoiseSampler(list(net.parameters()))
    sampler.add("policy", net(th.ones(1, 3)).sum())
    sampler.flush_micro()
    sampler.add("policy", net(th.ones(1, 3)).sum())
    sampler.add("distill", net(th.full((1, 3), 2.0)).sum())
    sampler.flush_micro()
    res = sampler.result(accum=2)
    assert "policy" in res and "distill" not in res


def test_sampler_never_writes_dot_grad():
    """`autograd.grad` is a READ. If this ever regressed to `.backward()`, the accumulated
    gradient the optimizer steps on would be doubled — silently, and only for the sampled call."""
    net = _tiny_net()
    sampler = PerTermNoiseSampler(list(net.parameters()))
    sampler.add("policy", net(th.ones(1, 3)).sum())
    sampler.flush_micro()
    assert all(p.grad is None for p in net.parameters())


def test_add_returns_the_same_tensor_object():
    """The fold reads `loss = loss + _ntg.add("aux", term)`. `add` returning anything but the very
    object it was handed would change the loss expression — the one thing this must never do."""
    net = _tiny_net()
    term = net(th.ones(1, 3)).sum()
    assert PerTermNoiseSampler(list(net.parameters())).add("aux", term) is term
    assert NULL_TAGGER.add("aux", term) is term
    assert NULL_TAGGER.add("aux", 0.0) == 0.0          # the `_vf_term = 0.0` case (value_from_dist)


def test_a_raising_probe_disables_itself_and_reports_nothing(monkeypatch, capsys):
    """A diagnostic must never take a run down. Whatever `autograd.grad` raises (a compiled
    backward that refuses a second traversal is the realistic one), the probe retires itself for
    the call, prints once, and leaves the training step to proceed untouched."""
    net = _tiny_net()
    sampler = PerTermNoiseSampler(list(net.parameters()))
    sampler.add("policy", net(th.ones(1, 3)).sum())

    def _boom(*a, **kw):
        raise RuntimeError("Trying to backward through the graph a second time")

    monkeypatch.setattr(th.autograd, "grad", _boom)
    sampler.flush_micro()
    assert sampler.failed and sampler.result(accum=2) == {}
    assert "per-term noise-scale probe disabled" in capsys.readouterr().out
    monkeypatch.undo()
    sampler.add("policy", net(th.ones(1, 3)).sum())
    sampler.flush_micro()                       # stays retired, and still takes no gradient
    assert sampler.result(accum=2) == {} and all(p.grad is None for p in net.parameters())


def test_null_tagger_is_inert():
    assert NULL_TAGGER.collecting is False
    assert NULL_TAGGER.result(2) == {} and NULL_TAGGER.flush_micro() is None
    assert NULL_TAGGER.release() is None


def test_env_var_wins_over_the_class_default(monkeypatch):
    class _M:
        noise_scale_per_term = True

    m = _M()
    monkeypatch.delenv("GEN3AI_NOISE_SCALE_PER_TERM", raising=False)
    assert per_term_enabled(m) is True
    m.noise_scale_per_term = False
    assert per_term_enabled(m) is False
    for off in ("0", "false", "off", "no", ""):
        monkeypatch.setenv("GEN3AI_NOISE_SCALE_PER_TERM", off)
        assert per_term_enabled(m) is False
    monkeypatch.setenv("GEN3AI_NOISE_SCALE_PER_TERM", "1")
    assert per_term_enabled(m) is True     # env wins even though the attribute says False


def test_the_group_names_are_the_ones_the_fold_tags():
    """The tags in `ppo.train()` and this tuple must agree, or a group is silently never reported."""
    from agents.training.instrumented_ppo import ppo as _ppo_mod
    import inspect
    src = inspect.getsource(_ppo_mod.InstrumentedMaskablePPO.train)
    tagged = set(re.findall(r'_ntg\.add\(\s*"([a-z_]+)"', src))
    assert tagged == set(NOISE_TERM_GROUPS), (
        f"train() tags {sorted(tagged)} but NOISE_TERM_GROUPS is {sorted(NOISE_TERM_GROUPS)} — a "
        f"group tagged but not listed is never measured; one listed but not tagged never appears.")


# --------------------------------------------------------------------------------------
# 3. End to end on a real train(): OFF is byte-identical, ON emits the tags
# --------------------------------------------------------------------------------------


class _Rec:
    def __init__(self):
        self.vals = {}

    def record(self, k, v, *a, **kw):
        self.vals[k] = v

    def __getattr__(self, _n):
        return lambda *a, **kw: None


def _arm(per_term):
    """One arm: a FRESH model, `learn()`, then exactly one `train()` at accum=2.

    Fresh per arm deliberately. A `train()` on this toy is NOT reproducible from a restored
    state_dict — three consecutive `_train_from_init` calls from one captured init drift by ~5e-4
    on the same weights, with the probe entirely absent — so reusing one model would have made
    this test compare the drift instead of the feature, in whichever direction the arm order
    happened to fall. Two fresh, identically-seeded models ARE bit-identical, which is what makes
    the equality below mean what it says.
    """
    os.environ["GEN3AI_NOISE_SCALE_PER_TERM"] = "1" if per_term else "0"
    th.manual_seed(0)
    np.random.seed(0)
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    # Prime both EMA families positive — the post-warmup state a real run is in. A single sample on
    # a 32-row toy can put either estimate negative, which the emit gate correctly withholds; that
    # gate has its own unit test above and is not what this one is measuring.
    model._noise_ema_s, model._noise_ema_g2 = 50.0, 2.0
    model._noise_ema_n = 500          # post-warmup means the COUNT is primed too: both families
    # warm up through `debiased_ema`, whose effective decay is `1 - 1/(n+1)`, so an EMA primed with
    # a value but not a count is still on sample 1 and would take the next sample whole.
    model._noise_ema_terms = {g: [50.0, 2.0, 500] for g in NOISE_TERM_GROUPS}
    model._noise_per_term_calls = 0
    model._logger = _Rec()
    sd = _train_from_init(model, init_sd, init_opt, batch_size=4, accum=2)
    return sd, model.logger.vals


def test_off_is_byte_identical_and_on_adds_the_new_tags():
    """THE regression gate. Two identically-seeded runs, one with the probe and one without: the
    probe must not move a single parameter, must not perturb the TOTAL noise-scale scalars, and
    must add its own."""
    try:
        sd_off, log_off = _arm(per_term=False)
        sd_off2, _ = _arm(per_term=False)
        sd_on, log_on = _arm(per_term=True)
    finally:
        os.environ.pop("GEN3AI_NOISE_SCALE_PER_TERM", None)

    for k in sd_off:      # the control: two OFF arms agree, so an ON/OFF difference would be real
        assert th.equal(sd_off[k], sd_off2[k]), f"the arms are not reproducible at all ({k})"
        assert th.equal(sd_off[k], sd_on[k]), f"the probe moved parameter {k}"
    for k in ("train/noise_scale", "train/noise_scale_ratio", "train/loss", "train/value_loss",
              "train/grad_norm", "train/policy_gradient_loss"):
        assert log_off[k] == log_on[k], f"the probe perturbed {k}"

    assert not [k for k in log_off if "noise_scale_ratio_" in k], "OFF must emit no per-term tag"
    assert "train/noise_per_term_ms" not in log_off
    for group in ("policy", "value"):     # entropy is coef-0 here; aux/distill are off on this env
        assert f"train/noise_scale_{group}" in log_on
        assert f"train/noise_scale_ratio_{group}" in log_on
        assert f"train/noise_scale_share_{group}" in log_on
    assert log_on["train/noise_per_term_ms"] >= 0.0
    assert log_on["train/train_ms"] > 0.0 and log_off["train/train_ms"] > 0.0


def test_per_term_tags_are_absent_without_accumulation():
    """Same rule the total obeys: no second batch size ⇒ no estimate, per term either."""
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model._noise_ema_terms = None
    model._logger = _Rec()
    _train_from_init(model, init_sd, init_opt, batch_size=8, accum=1)
    assert not [k for k in model.logger.vals if "noise_scale_ratio_" in k]
    assert model._noise_ema_terms is None


# --------------------------------------------------------------------------------------
# 4. The advisor — the DISAGREEMENT is the finding
# --------------------------------------------------------------------------------------


def test_advice_quotes_the_policy_term_ratio_in_both_bands():
    advise = InstrumentedMaskablePPO._noise_scale_advice
    high = dict(advise(6.0, 16384.0, policy_ratio=6.4))
    assert "PPO-policy-term ratio 6.4" in high["global_high"]
    low = dict(advise(0.1, 16384.0, policy_ratio=0.12))
    assert "PPO-policy-term ratio 0.12" in low["global_low"]
    # …and stays silent about it when there is no per-term reading yet (the old two-arg call).
    assert "PPO-policy-term" not in dict(advise(6.0, 16384.0))["global_high"]


def test_advice_fires_a_disagreement_when_the_bands_differ():
    """The owner's question in one assertion: the total says over-batched, the policy term says
    noise-limited. That must be its own warning, and it must name the aux deflation."""
    keys = [k for k, _ in InstrumentedMaskablePPO._noise_scale_advice(0.05, 16384.0, 3.0)]
    assert keys == ["global_low", "total_vs_policy_disagree"]
    msg = dict(InstrumentedMaskablePPO._noise_scale_advice(0.05, 16384.0, 3.0))[
        "total_vs_policy_disagree"]
    assert "noise_scale_share_" in msg and "DEFLATION" in msg.upper()


def test_advice_fires_a_disagreement_on_a_wide_gap_inside_one_band():
    """Both in the over-batched band, but 10x apart — still the finding, because the fix ("shrink
    the batch") is sized off a number that is wrong by 10x for the term you care about."""
    keys = [k for k, _ in InstrumentedMaskablePPO._noise_scale_advice(0.01, 16384.0, 0.1)]
    assert "total_vs_policy_disagree" in keys


def test_advice_is_silent_when_the_two_agree():
    assert InstrumentedMaskablePPO._noise_scale_advice(1.0, 16384.0, 1.2) == []
    assert InstrumentedMaskablePPO._noise_scale_advice(1.0, 16384.0, None) == []


def test_bands_are_named_not_repeated():
    """The band edges must be read from ONE place; a second literal is how the total half and the
    per-term half drift into disagreeing about what 'over-batched' means."""
    cls = InstrumentedMaskablePPO
    assert cls._nsr_band(None) is None
    assert cls._nsr_band(cls._NSR_HIGH) == "in band" and cls._nsr_band(cls._NSR_LOW) == "in band"
    assert cls._nsr_band(cls._NSR_HIGH * 1.01) == "noise-limited"
    assert cls._nsr_band(cls._NSR_LOW * 0.99) == "over-batched"


def test_emit_passes_the_policy_ratio_through(monkeypatch):
    """The advisor's rate-limited emitter must forward the third argument — a warm-up guard that
    quietly dropped it would leave the Events panel showing the total-only message forever."""
    seen = {}

    class _P(InstrumentedMaskablePPO):
        def __init__(self):
            self._nsr_samples = 100
            self._nsr_warn_last = {}

        @staticmethod
        def _noise_scale_advice(g, b, p=None):
            seen["args"] = (g, b, p)
            return []

    _P()._emit_noise_scale_warnings(0.05, 16384.0, 3.0)
    assert seen["args"] == (0.05, 16384.0, 3.0)


# --------------------------------------------------------------------------------------
# 5. THE EMA WARM-UP — one helper, one behaviour, for the total AND the per-term readings
#
# `gen3_noise_scale_warmup_v1`. The TOTAL's EMA used to ANCHOR ON ITS FIRST SAMPLE at a fixed
# decay 0.99, so `train/noise_scale` reported that one sample for its first few hundred calls (the
# R5F15 reading that had to be published as "provisional, n=2"), and a single NEGATIVE first
# `tr(Σ)` — which this estimator's single-call two-point solve produces routinely under noise —
# withheld the scalar entirely for hundreds of calls. The per-term half already warmed up as a
# running MEAN; both now go through the one `debiased_ema`.
# --------------------------------------------------------------------------------------


def test_debiased_ema_is_a_running_mean_while_it_warms_and_decays_afterwards():
    """The helper's whole contract, on a stream whose analytic mean is known."""
    decay = 0.99
    ema, n = None, 0
    for i, x in enumerate([4.0, 6.0, 2.0, 8.0], start=1):
        ema = debiased_ema(ema, n, x, decay)
        n += 1
        assert ema == pytest.approx(np.mean([4.0, 6.0, 2.0, 8.0][:i]))   # unbiased at every step
    # past the 1/(1-decay) window the effective decay saturates at `decay` — an EMA again, not a
    # mean that would keep the whole run's history at equal weight forever.
    assert debiased_ema(1.0, 10_000, 2.0, decay) == pytest.approx(0.99 * 1.0 + 0.01 * 2.0)
    assert debiased_ema(None, 0, 7.0, decay) == 7.0                      # first sample is itself


def test_debiased_ema_reads_a_constant_stream_as_that_constant_from_step_one():
    """The property a zero-init-without-correction fold fails: `(1-d)·c`, `(1-d²)·c`, … creeping up
    to `c`. It is also why a CONSTANT stream alone cannot separate the two warm-ups being fixed
    here — see the outlier test below for the revert-catcher."""
    ema, n = None, 0
    for _ in range(5):
        ema = debiased_ema(ema, n, 3.25, 0.99)
        n += 1
        assert ema == pytest.approx(3.25)


class _ConstantEstimate:
    """A CONSTANT synthetic gradient stream: every call returns the same `(tr(Σ), |G|²)` sample, so
    the analytic reading is `tr(Σ)/|G|²` at EVERY step with no start-up transient."""

    def __init__(self, tr_sigma, g2):
        self.sample = (float(tr_sigma), float(g2))
        self.calls = 0

    def __call__(self, *_args):
        self.calls += 1
        return self.sample


class _SampleStream:
    """A scripted sample sequence — the last entry repeats once exhausted."""

    def __init__(self, samples):
        self.samples = [(float(s), float(g)) for s, g in samples]
        self.calls = 0

    def __call__(self, *_args):
        s = self.samples[min(self.calls, len(self.samples) - 1)]
        self.calls += 1
        return s


def _total_only_model():
    """A tiny real PPO with the per-term probe OFF, so `_noise_scale_estimate` is called EXACTLY
    once per `train()` — by the TOTAL fold — and a scripted stream maps one sample to one step."""
    os.environ["GEN3AI_NOISE_SCALE_PER_TERM"] = "0"
    th.manual_seed(0)
    np.random.seed(0)
    model, _venv = _build_tiny_ppo(n_steps=8, n_envs=4)
    model.learn(total_timesteps=8 * 4)
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model._noise_ema_s = None          # a FRESH process: nothing folded yet
    model._noise_ema_g2 = None
    model._noise_ema_n = 0
    return model, init_sd, init_opt


def test_the_total_reading_has_no_startup_bias_on_a_constant_gradient_stream():
    """Steps 1, 2 and 3 must each read the analytic `B_simple = tr(Σ)/|G|²` exactly — no warm-up
    transient at all. Driven through the REAL `train()`, with only the two-point solve replaced by
    a constant, so this exercises the fold as it ships."""
    try:
        model, init_sd, init_opt = _total_only_model()
        est = _ConstantEstimate(tr_sigma=50.0, g2=2.0)
        model._noise_scale_estimate = est
        b_big = 4 * 2                                    # batch_size · accum, the ratio's divisor
        for step in (1, 2, 3):
            model._logger = _Rec()
            _train_from_init(model, init_sd, init_opt, batch_size=4, accum=2)
            assert est.calls == step, "the total must take exactly one sample per train()"
            assert model.logger.vals["train/noise_scale"] == pytest.approx(25.0), f"step {step}"
            assert model.logger.vals["train/noise_scale_ratio"] == pytest.approx(25.0 / b_big)
            assert model._noise_ema_n == step
    finally:
        os.environ.pop("GEN3AI_NOISE_SCALE_PER_TERM", None)


def test_a_negative_first_sample_no_longer_suppresses_the_total_for_hundreds_of_calls():
    """THE revert-catcher, and the production failure it reproduces (documented in
    `src/agents/training/CLAUDE.md`: "the total's tr(Σ) EMA started negative and
    `train/noise_scale{,_ratio}` therefore never emitted at all across 11 calls").

    Sample 1 is `tr(Σ) = -10` — the sign flip a single-call two-point solve produces routinely.
    Sample 2 is `+30`, so the MEAN is `+10` and step 2 must report `B_simple = 10/2 = 5`. Under the
    old anchor-on-first-sample fold the EMA at step 2 is `0.99·(-10) + 0.01·(+30) = -9.6`, still
    negative, so the emit gate withholds the scalar — for hundreds of calls."""
    try:
        model, init_sd, init_opt = _total_only_model()
        model._noise_scale_estimate = _SampleStream([(-10.0, 1.0), (30.0, 3.0)])

        model._logger = _Rec()
        _train_from_init(model, init_sd, init_opt, batch_size=4, accum=2)
        assert "train/noise_scale" not in model.logger.vals, "a negative tr(Σ) must withhold"
        assert model._noise_ema_s == pytest.approx(-10.0)

        model._logger = _Rec()
        _train_from_init(model, init_sd, init_opt, batch_size=4, accum=2)
        assert model._noise_ema_s == pytest.approx(10.0)    # the MEAN, not 0.99·(-10) + 0.01·30
        assert model._noise_ema_g2 == pytest.approx(2.0)
        assert model.logger.vals["train/noise_scale"] == pytest.approx(5.0)
        assert model.logger.vals["train/noise_scale_ratio"] == pytest.approx(5.0 / 8.0)
    finally:
        os.environ.pop("GEN3AI_NOISE_SCALE_PER_TERM", None)


def test_the_total_and_the_per_term_readings_warm_up_identically():
    """Same warm-up, same samples, same answer — the property that lets a reader compare
    `train/noise_scale` against `train/noise_scale_ratio_policy` on a young run at all.

    Fed the IDENTICAL `(tr(Σ), |G|²)` sequence, the total's fold (as `ppo.train()` performs it,
    via the shared helper) and `_fold_per_term_noise` must hold the same EMA state at every step —
    including the first, where an anchored fold and a mean disagree."""
    decay = _NOISE_SCALE_EMA_DECAY
    b_small, b_big = 100.0, 200.0
    samples = [(-1.0, 4.0), (7.0, 2.0), (3.0, 3.0), (5.0, 1.0)]

    fold = _Fold()
    total_s, total_g2, total_n = None, None, 0
    for tr_sigma, g2 in samples:
        total_s = debiased_ema(total_s, total_n, tr_sigma, decay)
        total_g2 = debiased_ema(total_g2, total_n, g2, decay)
        total_n += 1
        # the same sample, spelled as the (g_small_sq, g_big_sq) pair the per-term fold takes
        small_sq = g2 + tr_sigma / b_small
        big_sq = g2 + tr_sigma / b_big
        fold._fold_per_term_noise({"policy": (small_sq, big_sq)}, b_small, b_big, None)
        ema_s, ema_g2, n = fold._noise_ema_terms["policy"]
        assert ema_s == pytest.approx(total_s), f"tr(Σ) warm-up diverged at step {total_n}"
        assert ema_g2 == pytest.approx(total_g2), f"|G|² warm-up diverged at step {total_n}"
        assert n == total_n


def test_both_fold_sites_go_through_the_one_shared_helper():
    """A source scan, because the two folds live in two modules and the failure mode is that one of
    them is 'tidied' back into a bare fixed-decay line — which reads perfectly healthy and silently
    re-introduces the start-up bias on exactly one of the two readings."""
    import inspect

    from agents.training.instrumented_ppo import noise_scale as ns_mod
    from agents.training.instrumented_ppo import ppo as ppo_mod

    total_src = inspect.getsource(ppo_mod.InstrumentedMaskablePPO.train)
    per_term_src = inspect.getsource(ns_mod.NoiseScaleDiagnostics._fold_per_term_noise)
    for name, src in (("the total fold in train()", total_src),
                      ("_fold_per_term_noise", per_term_src)):
        assert "debiased_ema(" in src, f"{name} no longer folds through the shared helper"
    # and neither site keeps a hand-rolled `d * prev + (1 - d) * x` beside it
    for name, src in (("the total fold in train()", total_src),
                      ("_fold_per_term_noise", per_term_src)):
        assert not re.search(r"d\s*\*\s*self\._noise_ema", src), f"{name} kept a raw EMA fold"
        assert not re.search(r"d\s*\*\s*prev\[", src), f"{name} kept a raw EMA fold"
    assert "self._noise_ema_n" in total_src, "the total must carry its own sample count"
