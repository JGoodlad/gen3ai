"""The Rust eval executor's CAPTURE QUOTA, per game (``RustEvalCore._finish``; ``gen3_trace_result_v2``).

The draw bucket's guards, held on the executor that plays every eval cycle. (They were ``EvalRLPlayer``'s, the
poke-env eval trainee, until the Python eval worker was deleted — poke-env retirement P6 slice 6c.) A finished
game is booked into its unit by ``_finish``: a CAPTURED game's trace is KEPT iff its outcome's bucket is not yet
full, and every drawn game PLAYED is counted whether or not it is kept. The trace writer and the core's replay are
stubbed: what is asserted is which bucket a game lands in and which file name it is written under.
"""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

import agents.training.rust_eval.executor as EX
from agents.training.eval_quota import ForensicQuota, trace_filename_stem
from agents.training.eval_sharding import BOT

TURN_LIMIT = 250


@pytest.fixture
def written(monkeypatch):
    out = []
    monkeypatch.setattr(EX, "write_core_trace", lambda prefix, **_k: out.append(prefix))
    return out


def _core():
    core = EX.RustEvalCore.__new__(EX.RustEvalCore)
    core.turn_limit, core.lib, core.commit, core.names = TURN_LIMIT, None, "", ("p1", "p2")
    return core


def _unit(win=5, loss=10, draw=5):
    unit = SimpleNamespace(item=SimpleNamespace(key="heuristic", kind=BOT), shard_index=0, n_games=10_000,
                           unit_id="heuristic__0")
    return EX._Unit(unit=unit, games=[], quota=ForensicQuota(win=win, loss=loss, draw=draw))


def _row():
    return (np.zeros(4, np.float32), np.ones(11, bool), np.zeros(11, np.float32), 0.5, 0)


def _finish(core, u, g, *, winner, turn, tmp_path):
    gm = EX._Game(u=u, g=g, capturing=u.quota_open, rows=[_row(), _row()])
    f = {"env": 0, "winner": winner, "end_turn": turn, "forfeit": 0, "script": ""}
    ffi = SimpleNamespace(trace=lambda _lib, _script, commit, label: {"winner": winner, "turn": turn})
    return core._finish(gm, f, 0.0, step=1000, gamma=1.0, forensic_root=str(tmp_path), result_dir=str(tmp_path),
                        pool=None, st=EX.CycleStats(step=1000), game_log=None, trace_stem=trace_filename_stem,
                        ffi=ffi)


def _prefixes(written):
    return [os.path.basename(w).split("_")[0] for w in written]


def test_a_pre_cap_TIE_is_kept_in_the_DRAW_bucket(written, tmp_path):
    """A tie (no winner, before the cap) is a DRAW and is persisted under ``draw_`` — never dropped."""
    core, u = _core(), _unit()
    _finish(core, u, 0, winner=0, turn=37, tmp_path=tmp_path)
    assert (u.kept[EX.DRAW], u.draws_seen) == (1, 1)
    assert _prefixes(written) == ["draw"]


def test_a_game_at_the_TURN_LIMIT_lands_in_the_DRAW_bucket_not_the_LOSS_bucket(written, tmp_path):
    """A timeout arrives wearing a loss's flags (the trainee forfeits at the cap); it is a DRAW."""
    core, u = _core(), _unit()
    _finish(core, u, 0, winner=2, turn=TURN_LIMIT, tmp_path=tmp_path)
    assert (u.kept[EX.DRAW], u.kept[EX.LOSS]) == (1, 0)
    assert _prefixes(written) == ["draw"]


def test_the_draw_bucket_is_INDEPENDENT_so_a_stall_storm_cannot_evict_a_loss(written, tmp_path):
    """20 timeouts arrive first and fill the draw quota; the loss quota is untouched, so every decisive loss is
    still captured — and every draw PLAYED is still counted."""
    core, u = _core(), _unit(win=5, loss=10, draw=5)
    for g in range(20):
        _finish(core, u, g, winner=2, turn=TURN_LIMIT, tmp_path=tmp_path)
    for g in range(20, 30):
        _finish(core, u, g, winner=2, turn=60, tmp_path=tmp_path)
    assert u.kept[EX.DRAW] == 5 and u.draws_seen == 20 and u.kept[EX.LOSS] == 10
    p = _prefixes(written)
    assert p.count("draw") == 5 and p.count("loss") == 10


def test_wins_and_decisive_losses_are_bucketed_by_outcome(written, tmp_path):
    core, u = _core(), _unit()
    _finish(core, u, 0, winner=1, turn=20, tmp_path=tmp_path)
    _finish(core, u, 1, winner=2, turn=20, tmp_path=tmp_path)
    assert (u.kept[EX.WIN], u.kept[EX.LOSS], u.kept[EX.DRAW], u.draws_seen) == (1, 1, 0, 0)
    assert _prefixes(written) == ["win", "loss"]


def test_the_quota_stays_OPEN_while_only_the_draw_bucket_is_unfilled():
    """``quota_open`` decides whether a game is CAPTURED at all; if it ignored draws, a unit whose wins and
    losses filled first would capture no draw ever again."""
    u = _unit()
    u.kept[EX.WIN], u.kept[EX.LOSS] = u.quota.win, u.quota.loss
    assert u.quota_open is True
    u.kept[EX.DRAW] = u.quota.draw
    assert u.quota_open is False
