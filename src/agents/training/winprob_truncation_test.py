"""THE 250-TURN CAP AT GAMMA 1 — `gen3_winprob_critic_mode_v1`, design gap **B6**.

The question the design left open: when the trainee hits `MAX_TURNS` and forfeits, does the
episode reach SB3 as a TERMINAL (the design's contract — a timeout is a not-win, indicator 0, and
it is COUNTED) or as a TRUNCATION (which SB3 bootstraps, `r += gamma * V(s_last)`; at `gamma = 1`
under `--critic winprob` that is `V(s_last)` against a target of `V(s_last)` — a tautology whose TD
error is identically zero, removing the timeout from the loss and letting a stalling policy think
it is fine)?

**MEASURED ANSWER: it was a TRUNCATION.** On a real bridge battle with `StallConfig.threshold`
lowered to 6, the cap forfeit produced `battle.finished=True`, `battle.won=False`, six mons alive a
side, and `terminated=False, truncated=True` at the wrapper boundary — so `SubprocVecEnv` would
have written `TimeLimit.truncated=True` and `MaskablePPO.collect_rollouts` would have bootstrapped.
`wrappers.resolve_episode_end` is the fix; this file is its specification, and every claim in the
chain is asserted here rather than described:

1. the poke-env RULE that produces it (`PokeEnv.calc_term_trunc`) — including that our env never
   truncates in the SB3 sense at all, since either flag requires `battle.finished`;
2. the pure re-labelling (`resolve_episode_end`), on both critics;
3. the wrapper actually applying it, on the tri-state outcome branch beside it;
4. **the composition, through the REAL SB3 code path** — a scripted env that ends exactly the way
   the cap does, driven by a real `InstrumentedMaskablePPO.collect_rollouts`, asserting the ROLLOUT
   BUFFER's last-step reward, its `episode_starts`, and the presence/absence of the bootstrap.

(4) is the one that matters: (1)-(3) could all be right while SB3 still did something else, and the
bootstrap lives in a library file no test in this tree had ever exercised.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from agents.model.critic_mode import CRITIC_SHAPED, CRITIC_WINPROB
from agents.training.wrappers import resolve_episode_end

# The constant V(s) the scripted critic below returns. Deliberately NOT 0.0 and not the terminal
# reward, so a bootstrap is visible as an exact arithmetic identity rather than a coincidence.
_V_CONST = 0.37


# ---------------------------------------------------------------------------------------------
# 1. the poke-env rule that produced the truncation
# ---------------------------------------------------------------------------------------------

class _FakeMon:
    def __init__(self, fainted: bool):
        self.fainted = fainted


class _FakeBattle:
    """The three fields `PokeEnv.calc_term_trunc` reads, and nothing else."""

    def __init__(self, *, finished: bool, ours_fainted: int, theirs_fainted: int, size: int = 6):
        self.finished = finished
        self.team_size = size
        self.team = {i: _FakeMon(i < ours_fainted) for i in range(size)}
        self.opponent_team = {i: _FakeMon(i < theirs_fainted) for i in range(size)}


def _term_trunc(**kw):
    from poke_env.environment.env import PokeEnv
    return PokeEnv.calc_term_trunc(_FakeBattle(**kw))


def test_a_wipe_terminates_and_a_cap_forfeit_truncates():
    """The rule, stated as the four cases that reach it.

    A forfeit at the 250-turn cap finishes the battle with mons alive on BOTH sides, which is the
    `else` branch of `calc_term_trunc`'s `(ours == 0) != (theirs == 0)` — the same branch a genuine
    tie takes. That is the whole origin of B6: the sim has no "timeout" concept, so the cap arrives
    as "finished, but nobody was wiped"."""
    assert _term_trunc(finished=True, ours_fainted=6, theirs_fainted=2) == (True, False)   # we lost
    assert _term_trunc(finished=True, ours_fainted=1, theirs_fainted=6) == (True, False)   # we won
    # THE CAP FORFEIT — measured live: 6 alive a side, `won=False`, `finished=True`.
    assert _term_trunc(finished=True, ours_fainted=0, theirs_fainted=0) == (False, True)
    # A genuine tie takes the same branch.
    assert _term_trunc(finished=True, ours_fainted=6, theirs_fainted=6) == (False, True)


def test_an_unfinished_battle_is_neither_which_is_why_no_truncation_here_is_a_time_limit():
    """Either flag REQUIRES `battle.finished`. So a `truncated=True` out of this env never means
    "an episode was cut off mid-flight" — the only thing `TimeLimit.truncated` is supposed to mean —
    it means "finished, and not by a wipe". Bootstrapping it is wrong on the semantics before it is
    wrong on the arithmetic."""
    assert _term_trunc(finished=False, ours_fainted=0, theirs_fainted=0) == (False, False)
    assert _term_trunc(finished=False, ours_fainted=5, theirs_fainted=5) == (False, False)


# ---------------------------------------------------------------------------------------------
# 2. the pure re-labelling
# ---------------------------------------------------------------------------------------------

@pytest.mark.parametrize("term,trunc", [(True, False), (False, True), (False, False)])
def test_shaped_is_the_identity(term, trunc):
    """`shaped` must be byte-identical — this is the guard on that, not a comment about it."""
    assert resolve_episode_end(term, trunc, CRITIC_SHAPED) == (term, trunc)
    assert resolve_episode_end(term, trunc) == (term, trunc)      # the DEFAULT is shaped


def test_winprob_relabels_a_truncation_as_terminal_and_touches_nothing_else():
    assert resolve_episode_end(False, True, CRITIC_WINPROB) == (True, False)
    assert resolve_episode_end(True, False, CRITIC_WINPROB) == (True, False)   # a wipe: unchanged
    assert resolve_episode_end(False, False, CRITIC_WINPROB) == (False, False)  # mid-episode


# ---------------------------------------------------------------------------------------------
# 3. the wrapper applies it, and the outcome branch beside it is unmoved
# ---------------------------------------------------------------------------------------------

class _StubInner:
    """The minimum `MaskableAgentWrapper.step` reads out of its parent: a `super().step` result and
    an `env.battle1`. Bypasses `SingleAgentWrapper.step` entirely (it needs two live poke-env
    players); the subject under test is the wrapper's own tail, which is where B6's fix lives."""

    def __init__(self, out):
        self._out = out
        self.agent1_to_move = True
        self.battle1 = None


def _wrapper_step(term, trunc, critic, *, won):
    from agents.training import wrappers as W

    w = object.__new__(W.MaskableAgentWrapper)
    w._critic = critic
    w._exploiter_player = None
    w._pool_player = None
    w.opponent = object()
    w._team_wr_tracking = False
    w._opponent_class = 0

    class _B:
        pass

    b = _B()
    b.won = won
    inner = _StubInner(None)
    inner.battle1 = b
    w.env = inner
    # `super().step` is `SingleAgentWrapper.step`; stub it at the class the MRO reaches.
    out = ({}, 1.0, term, trunc, {})
    orig = W.SingleAgentWrapper.step
    W.SingleAgentWrapper.step = lambda self, action: out                     # noqa: ARG005
    try:
        return w.step(0)
    finally:
        W.SingleAgentWrapper.step = orig


def test_the_wrapper_relabels_the_cap_only_under_winprob():
    _, _, term, trunc, info = _wrapper_step(False, True, CRITIC_WINPROB, won=False)
    assert (term, trunc) == (True, False)
    _, _, term, trunc, info2 = _wrapper_step(False, True, CRITIC_SHAPED, won=False)
    assert (term, trunc) == (False, True), "shaped must be untouched"
    # And the outcome branch beside it is unmoved by the fix: both label the cap a NOT-WIN.
    assert info["win_outcome"] == 0.0 and info2["win_outcome"] == 0.0


def test_the_cap_forfeit_is_a_loss_not_a_draw_in_the_label_stream():
    """MEASURED CORRECTION (2026-09-06, real bridge battle): the cap forfeits, Showdown answers
    `|win|<opponent>`, and `won_by` sets `_won = False`. So `win_draw` — which feeds
    `signal/draw_rate` — counts TIES (`|tie|` ⇒ `won is None`) and NOT timeouts. The label is 0.0
    either way, so nothing downstream moved; the meter's MEANING did, and a reader watching
    `draw_rate` for the stall rate would be watching the wrong series."""
    _, _, _, _, cap = _wrapper_step(False, True, CRITIC_WINPROB, won=False)
    assert (cap["win_outcome"], cap["win_draw"]) == (0.0, 0.0)
    _, _, _, _, tie = _wrapper_step(False, True, CRITIC_WINPROB, won=None)
    assert (tie["win_outcome"], tie["win_draw"]) == (0.0, 1.0)
    _, _, _, _, win = _wrapper_step(True, False, CRITIC_WINPROB, won=True)
    assert (win["win_outcome"], win["win_draw"]) == (1.0, 0.0)


# ---------------------------------------------------------------------------------------------
# 4. THE COMPOSITION — a scripted episode to the cap, through the REAL SB3 rollout loop
# ---------------------------------------------------------------------------------------------

class _CapEnv(gym.Env):
    """A scripted episode that ends the way the 250-turn cap ends.

    Every step pays 0 (the win indicator's value for a not-win, which is what `--terminal-indicator`
    pays on a cap loss), and at `cap` the episode ends with whatever `(terminated, truncated)`
    `resolve_episode_end` says for this critic — so the arm under test is the REAL function, not a
    hand-written flag pair that could drift from it."""

    def __init__(self, cap: int, critic: str):
        super().__init__()
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(2)
        self._cap, self._critic, self._t = cap, critic, 0

    def action_masks(self):
        return np.ones(2, dtype=np.int8)

    def _obs(self):
        return {"observation": np.array([float(self._t)], dtype=np.float32),
                "action_mask": np.ones(2, dtype=np.int8)}

    def reset(self, *, seed=None, options=None):
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        if self._t < self._cap:
            return self._obs(), 0.0, False, False, {}
        # The sim says (terminated=False, truncated=True) here — measured. What the LEARNER sees is
        # whatever `resolve_episode_end` makes of that, which is the arm under test.
        term, trunc = resolve_episode_end(False, True, self._critic)
        return self._obs(), 0.0, term, trunc, {}


def _collect(critic: str, cap: int = 3, n_steps: int = 6):
    """One real `collect_rollouts` over a single scripted env, with the critic pinned to a constant
    so a bootstrap is an exact arithmetic identity. Returns the filled rollout buffer."""
    from stable_baselines3.common.callbacks import ConvertCallback
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.training.instrumented_ppo import InstrumentedMaskablePPO

    venv = DummyVecEnv([lambda: _CapEnv(cap, critic)])
    model = InstrumentedMaskablePPO(
        "MultiInputPolicy", venv, n_steps=n_steps, batch_size=2, n_epochs=1,
        gamma=1.0,                      # THE REGIME UNDER TEST — the design's gamma for winprob
        ent_coef=0.0, vf_coef=0.5, device="cpu", seed=0,
    )
    model.policy.predict_values = lambda obs: th.full(                       # noqa: ARG005
        (int(np.asarray(obs["observation"]).shape[0]), 1), _V_CONST)
    model._setup_learn(n_steps, callback=None)
    cb = ConvertCallback(None)
    cb.init_callback(model)
    model.collect_rollouts(model.env, cb, model.rollout_buffer, n_rollout_steps=n_steps)
    return model.rollout_buffer, cap


def test_under_winprob_the_cap_is_a_terminal_zero_with_no_bootstrap():
    """THE B6 CONTRACT. The last step of the capped episode must carry reward EXACTLY 0 — the win
    indicator for a not-win — and the next row must open a new episode.

    If the truncation ever comes back, this row reads `0 + 1.0 * 0.37`, its TD error collapses to
    zero, and the stall rate G7 kills the arm on becomes a signal the critic was never shown."""
    buf, cap = _collect(CRITIC_WINPROB)
    last = cap - 1                                   # the row whose step ENDED the episode
    assert buf.rewards[last, 0] == pytest.approx(0.0), (
        f"the cap bootstrapped: reward {buf.rewards[last, 0]} != 0 (V={_V_CONST})")
    assert buf.episode_starts[last, 0] == 0.0        # this row is inside the episode …
    assert buf.episode_starts[last + 1, 0] == 1.0    # … and the next one opens a new one


def test_under_shaped_the_cap_still_bootstraps_and_that_is_todays_behaviour():
    """The ANTI-VACUITY control, and the honest record of what `shaped` does.

    Without it the test above could pass because SB3 never bootstraps anything — it must be shown
    that the same scripted ending, under the untouched critic, DOES pick up `gamma * V(s_last)`.
    That is also the finding for `shaped`, stated rather than implied: a cap forfeit (and a tie)
    have always bootstrapped `0.9999 * V(s_last)` on top of a terminal reward that already paid
    `--draw-penalty`. This change does not touch it."""
    buf, cap = _collect(CRITIC_SHAPED)
    last = cap - 1
    assert buf.rewards[last, 0] == pytest.approx(_V_CONST), (
        "shaped must still bootstrap — otherwise the winprob assertion above measures nothing")
