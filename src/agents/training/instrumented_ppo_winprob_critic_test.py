"""The WIN-PROB CRITIC's half of the PPO step — `gen3_winprob_critic_mode_v1`.

`critic_mode_test.py` pins the ROUTE (`_critic_value` reads the head, in [0,1]); this file pins
what `train()` does about it, and the OFF path.

**The OFF claim is the load-bearing one, and it is asserted on the PARAMETER UPDATE rather than on
a source read.** `train()` gained four `critic_winprob` conditionals — the value-loss branch, the
`_vf_term` gate, the win-prob fold's tag and weight, and the grad-balance value term — and on the
`shaped` path every one of them must reduce to the expression that was there. A source-level check
("the flag is False so the branch is not taken") cannot see an operator-precedence slip inside the
rewritten condition, which is exactly the shape of change that was made. Two identically-seeded
models trained from one captured init is the instrument for that (the `_build_tiny_ppo` /
`_train_from_init` harness this file borrows, for the reason its own docstring gives: a `train()`
on the toy is not reproducible from a restored `state_dict` alone).

**A tiny PPO cannot exercise the ON path end to end** — `MultiInputPolicy` has no
`Gen3FeaturesExtractor`, so there is no `win_head` to route to. The ON path is therefore covered
where its pieces are: the route in `critic_mode_test.py`, the fold's tag/weight in the pure
predicates below, and the whole composition by the `--debug --critic winprob` smoke recorded in
`designs/CHANGELOG.md`.
"""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch as th

from agents.model.critic_mode import is_winprob
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo.calibration import critic_reliability
from agents.training.instrumented_ppo_test import _build_tiny_ppo, _train_from_init


# --------------------------------------------------------------------------------------------
# OFF is byte-identical, measured on the update
# --------------------------------------------------------------------------------------------

def _arms(*, clip_range_vf=None):
    """One model, one init, one filled buffer — the `_train_from_init` protocol the byte-identity
    tests in `instrumented_ppo_test` use: capture the init BEFORE `learn()`, then replay `train()`
    from it once per arm. `atol=1e-7` matches those tests: two replays of one `train()` agree to
    the float32 noise floor, not bit-for-bit, because the optimizer state is restored rather than
    re-derived."""
    model, _ = _build_tiny_ppo(n_steps=8, n_envs=4)
    if clip_range_vf is not None:
        model.clip_range_vf = clip_range_vf     # a SCHEDULE — SB3 calls it with progress
    init_sd = copy.deepcopy(model.policy.state_dict())
    init_opt = copy.deepcopy(model.policy.optimizer.state_dict())
    model.learn(total_timesteps=8 * 4)          # one rollout to fill the buffer
    return model, init_sd, init_opt


def _same(a, b, why):
    for k in a:
        assert th.allclose(a[k], b[k], atol=1e-7), f"{why}: {k}"


def test_the_harness_is_reproducible_at_all():
    """The ANTI-VACUITY control for everything below. Two replays of the same seeded `train()`
    from the same init must agree — otherwise a byte-identity assertion between two arms would
    pass by measuring nothing, or fail for a reason that has nothing to do with the critic."""
    model, sd, opt = _arms()
    a = _train_from_init(model, sd, opt, batch_size=4, accum=1)
    b = _train_from_init(model, sd, opt, batch_size=4, accum=1)
    _same(a, b, "the harness itself is not reproducible")


def test_the_shaped_update_is_unchanged_by_the_critic_mode_attribute():
    """`_critic_mode` set to 'shaped' must produce the SAME update as a policy that never heard of
    the attribute — i.e. every new conditional in `train()` reduces to the expression that was
    there. The bogus-value arm is the second half: `is_winprob` is a strict equality, so anything
    that is not 'winprob' must take the historical branch, and a future `startswith` / truthiness
    "cleanup" fails here."""
    model, sd, opt = _arms()
    baseline = _train_from_init(model, sd, opt, batch_size=4, accum=1)

    model.policy._critic_mode = "shaped"
    _same(baseline, _train_from_init(model, sd, opt, batch_size=4, accum=1),
          "'shaped' moved the update")

    model.policy._critic_mode = "not-a-mode"
    _same(baseline, _train_from_init(model, sd, opt, batch_size=4, accum=1),
          "a non-'winprob' mode took the winprob branch — is_winprob must be a strict equality")


def test_the_clip_branch_condition_still_reduces_to_the_original():
    """The value-loss branch became `elif critic_winprob or self.clip_range_vf is None:`. On the
    shaped path that must be `elif self.clip_range_vf is None:` — which holds only if the
    disjunction is in that order and `critic_winprob` is a plain bool. Exercised with a LIVE clip,
    the one arm the change could have broken: at `clip_range_vf=None` both readings agree and the
    test would be vacuous."""
    model, sd, opt = _arms(clip_range_vf=lambda _progress: 0.5)
    a = _train_from_init(model, sd, opt, batch_size=4, accum=1)
    # ...and the clip was genuinely ACTIVE on that update, or the arm above tested nothing.
    assert model.logger.name_to_value.get("train/clip_range_vf") == pytest.approx(0.5)
    model.policy._critic_mode = "shaped"
    b = _train_from_init(model, sd, opt, batch_size=4, accum=1)
    _same(a, b, "the clipped value branch moved")


# --------------------------------------------------------------------------------------------
# the ON path's decisions, as pure predicates over the source
# --------------------------------------------------------------------------------------------

def _train_source() -> str:
    import inspect
    return inspect.getsource(InstrumentedMaskablePPO.train)


def test_the_win_prob_term_is_tagged_value_under_the_winprob_critic():
    """§1.4 fact 4 of the design, not repeated: under the distributional critic the REAL critic
    loss was tagged "aux", so `train/noise_scale_value` spent that whole era describing a
    zero-weighted term. The promoted BCE must be tagged "value"."""
    src = _train_source()
    assert 'if critic_winprob:' in src
    assert '_ntg.add("value", win_prob_term)' in src, (
        "the promoted BCE must join the `value` noise-scale group — an `aux` tag here reproduces "
        "the defect the design's §1.4 records")
    assert '_ntg.add("aux", win_prob_term)' in src, "the shaped path must keep its `aux` tag"


def test_the_promoted_bce_is_weighted_by_vf_coef_not_win_prob_coef():
    """One critic, one coefficient. Two on one loss is the ambiguity `_ce_w` existed to resolve."""
    src = _train_source()
    assert 'win_prob_term = self.vf_coef * wp_loss' in src
    assert 'win_prob_term = self.win_prob_coef * wp_loss' in src, "the aux path keeps its own coef"


def test_the_scalar_value_term_is_dropped_under_the_winprob_critic():
    """`value_net` is in no loss graph here, so a `vf_coef * value_loss` term would train a readout
    nothing reads — the Phase-B treatment, for the same reason."""
    assert '_vf_term = 0.0 if (value_from_dist or critic_winprob) else self.vf_coef * value_loss' \
        in _train_source()


def test_the_bce_is_forced_on_even_at_win_prob_coef_zero():
    """`--win-prob-coef 0` must not be able to switch the critic's own loss off."""
    assert '(self.win_prob_coef != 0.0 or critic_winprob)' in _train_source()


def test_the_grad_balance_value_term_follows_the_critic():
    """`grad/value_share` has to measure the term that is actually weighted, or it reports the
    frozen scalar head's pull as if it were the critic's (the 2026-07-22 catch, one critic over)."""
    assert 'win_prob_term if (critic_winprob and win_prob_term is not None)' in _train_source()


# --------------------------------------------------------------------------------------------
# the P(win)-currency reliability read
# --------------------------------------------------------------------------------------------

class _Buf:
    def __init__(self, values, target, mask):
        self.values = np.asarray(values, dtype=np.float32).reshape(-1, 1)
        self.observations = {
            "win_target": np.asarray(target, dtype=np.float32).reshape(-1, 1),
            "win_mask": np.asarray(mask, dtype=np.float32).reshape(-1, 1),
        }


def test_the_reliability_read_uses_the_shared_scaffolding_function():
    """The live number and the offline calibration gate must be the SAME statistic, or a run's
    `win_prob/critic_resolution` cannot be compared with `main.scaffolding_gauge --reliability`."""
    from agents.training.scaffolding import reliability_table
    rng = np.random.default_rng(0)
    p = rng.random(400)
    y = (rng.random(400) < p).astype(np.float32)
    got = critic_reliability(_Buf(p, y, np.ones(400)))
    want = reliability_table(p, y)
    # `rollout_buffer.values` is float32, so the read casts a narrowed copy — the tolerance is
    # that cast, not a second implementation. A parallel implementation would miss by far more
    # than this on `resolution`, which is a sum of squared BIN deviations.
    for k in ("brier", "skill", "ece", "reliability", "resolution", "uncertainty"):
        assert got[k] == pytest.approx(float(want[k]), rel=1e-6), f"{k} is not the shared stat"


def test_only_the_MASKED_rows_are_scored():
    """`win_mask` is 0 for the trailing in-progress episode, whose `win_target` is a placeholder
    0.0 — scoring it would report a fabricated loss for every unfinished trajectory."""
    p = np.array([0.9, 0.9, 0.9, 0.9])
    y = np.array([1.0, 1.0, 0.0, 0.0])          # the two masked-out rows disagree with the head
    m = np.array([1.0, 1.0, 0.0, 0.0])
    got = critic_reliability(_Buf(p, y, m))
    assert got["n"] == 2
    assert got["base_rate"] == pytest.approx(1.0)


@pytest.mark.parametrize("case", ["no_keys", "no_known_label"])
def test_an_unmeasurable_rollout_publishes_NOTHING(case):
    """`{}` — never zeros. A calibration of nothing and a perfect calibration must not render the
    same, and TensorBoard draws a hole for an absent point."""
    buf = _Buf([0.5, 0.5], [1.0, 0.0], [0.0, 0.0])
    if case == "no_keys":
        buf.observations = {}
    assert critic_reliability(buf) == {}


def test_the_read_lives_in_calibration_and_is_gated_on_the_critic_mode():
    """It reads `rollout_buffer.values` AS probabilities. Under `shaped` those are shaped returns
    in PopArt units, so publishing it there would be a calibration curve of the wrong quantity."""
    src = _train_source()
    assert 'if critic_winprob:' in src
    assert 'win_prob/critic_' in src


def test_is_winprob_is_what_train_reads():
    """One predicate, imported — never a second string compare inside the loop."""
    src = _train_source()
    assert 'is_winprob(getattr(self.policy, "_critic_mode", "shaped"))' in src
    assert is_winprob("winprob") and not is_winprob("shaped")
