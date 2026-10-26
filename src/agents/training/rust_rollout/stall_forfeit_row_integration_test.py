"""The stall-forfeit decision is NOT a training row (`gen3_stall_forfeit_row_v1`, GIGO audit 2026-10-26 B-C2).

At a p1 decision whose turn has reached the turn limit the core presents the decision (obs, mask), the host samples an
action and writes the row, and the core then FORFEITS instead of feeding it (`rust_env/src/episode.rs`
`stall_forfeit_due`). That row's action never ran, yet PPO credited it with the loss (`0 - V(s)`) — one row per stall game
(executed in the `--debug` smoke at a 6-turn limit: all 297 ended games carried `rows_written = p1_CHOOSE_fed + 1`).

This runs the REAL collector (the self-check core, T2 eager on CPU) at a 5-turn limit, so almost every game stall-forfeits,
and asserts per ended forfeit game that the host was presented exactly ONE more decision than it trained on, that decision
sitting at the limit and every kept one below it. Fails on revert: `GameLog.end` keeps the unplayed row.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import pytest

from agents.training.rust_rollout import testkit as TK
from agents.training.rust_rollout.build import RustEnvDecl
from agents.training.rust_vec_env import RustVecEnv

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N, N_STEPS, LIMIT = 3, 24, 5


class Decisions:
    """The collector hook: every trainee decision the host was PRESENTED, as (env, episode, turn)."""

    def __init__(self) -> None:
        self.seen: List[Tuple[int, int, int]] = []

    def episode_started(self, col: Any, moved: np.ndarray) -> None:
        pass

    def p2(self, col: Any) -> None:
        pass

    def trainee(self, col, envs, logp, act, blogp, value, u, slots) -> None:
        c = col.cols
        for e in envs.tolist():
            self.seen.append((int(e), int(c["episode"][e]), int(c["turn"][e])))


@pytest.fixture(scope="module")
def spaces():
    TK.build_selfcheck()
    return TK.production_spaces()


@pytest.mark.slow  # ship gate (2026-10-26): 9 s beside a live run, over the 8 s routine bar (a real collector on the self-check core)
def test_the_decision_the_core_forfeited_instead_of_playing_is_not_a_training_row(spaces):
    _args, obs, act = spaces
    decl = RustEnvDecl(n_envs=N, threads=2, front="ffi", profile="selfcheck", n_steps=N_STEPS, micro_batch=N_STEPS,
                       device="cpu", backend="eager", run_seed=11, gamma=1.0, gae_lambda=0.8, turn_limit=LIMIT)
    env = RustVecEnv(n_envs=N, observation_space=obs, action_space=act,
                     build=lambda m: TK.collector_for(m, obs, decl=decl))
    model = TK.fresh_model(env, n_steps=N_STEPS, batch_size=N_STEPS)
    col = env.startup(model)
    seen = Decisions()
    col.hooks.append(seen)
    cb = TK.NullCallback()
    try:
        assert col.collect(model, cb, model.rollout_buffer)
        forfeits = [i for i in cb.infos if i["rust_env"]["forfeit"]]
        assert forfeits, "fixture: at a 5-turn limit the games stall-forfeit (else this proves nothing)"
        for info in forfeits:
            key = (info["rust_env"]["env"], info["rust_env"]["episode"])
            turns = sorted(t for e, ep, t in seen.seen if (e, ep) == key)
            kept = info["episode"]["l"]
            assert len(turns) == kept + 1, (
                f"game {key}: the host was presented {len(turns)} decisions but trained on {kept} — the forfeited "
                "decision (its action never ran) must be released, not trained")
            assert turns[-1] >= LIMIT > turns[-2], f"game {key}: the unplayed decision is the one at the limit: {turns}"
        assert col.log.forfeit_rows_dropped == len([i for i in cb.infos if i["rust_env"]["forfeit"]])
        # the accounting closes with the dropped rows named
        log = col.log
        consumed = int(model.rollout_buffer.buffer_size) * int(model.rollout_buffer.n_envs)
        assert len(seen.seen) == (consumed + log.completed_rows + log.in_progress_rows() + log.rows_cut
                                  + log.forfeit_rows_dropped)
    finally:
        env.close()
