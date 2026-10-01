"""E2 — the owned Python-core collect on REAL GAMES equals upstream's, byte for byte
(`design_own_ppo_loop.md` §4).

`rust_rollout.parity` already records games on the Rust collector and REPLAYS them through today's
Python path (a `DummyVecEnv` of production `Gen3Env`s on the rust bridge, `WinProbLabelCallback`
registered, the trainee's keyed draw). Here that replay runs TWICE over the same recording — once with
the learner pinned to upstream's loop (the `sb3_reference` seam), once with the owned loop — in one
process, on the same weights and the same batch composition, so the bar is EXACT on every field
(parity's own float bars exist only because it compares two different collectors). This is the
check stage 1 must pass BEFORE the deletion pass removes the Python env core it runs through.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pytest

from agents.training.instrumented_ppo import loop as L
from agents.training.rust_rollout import parity as PA
from agents.training.rust_rollout import testkit as TK

pytestmark = [pytest.mark.sim, pytest.mark.integration]


@pytest.fixture(scope="module")
def recording() -> Dict[str, Any]:
    TK.build_selfcheck()
    return PA.record(n_envs=4, n_steps=48, windows=2)


def _replay(rec: Dict[str, Any], mode: str, monkeypatch: Any) -> Dict[str, Any]:
    orig = TK.fresh_model
    seen = []

    def fresh_model(*a: Any, **k: Any) -> Any:
        m = orig(*a, **k)
        m._ppo_loop_mode = mode                   # what `learn()` would resolve; replay drives collect directly
        real = m._collect_python

        def counted(*aa: Any, **kk: Any) -> Any:
            seen.append(mode)
            return real(*aa, **kk)

        m._collect_python = counted
        return m

    monkeypatch.setattr(TK, "fresh_model", fresh_model)
    try:
        out = PA.replay(rec)
    finally:
        monkeypatch.setattr(TK, "fresh_model", orig)
    assert seen == [mode] * int(rec["windows"]), seen
    return out


def test_owned_python_collect_equals_upstream_on_the_same_games(recording: Any, monkeypatch: Any) -> None:
    ref = _replay(recording, L.LOOP_REFERENCE, monkeypatch)
    own = _replay(recording, L.LOOP_OWNED, monkeypatch)
    assert len(ref["buffers"]) == len(own["buffers"]) == int(recording["windows"])
    for w, (a, b) in enumerate(zip(ref["buffers"], own["buffers"])):
        assert a.keys() == b.keys()
        bad = sorted(k for k in a if a[k].dtype != b[k].dtype or a[k].tobytes() != b[k].tobytes())
        assert not bad, f"window {w}: fields differ between upstream and the owned collect: {bad}"
    assert ref["flips"] == own["flips"]
    # non-vacuity: the windows carried finished games (win labels) and the replay matched the recording
    assert any(float(np.asarray(b["obs:win_mask"]).sum()) > 0 for b in own["buffers"]), \
        "no game ended inside a window: the win-label path was not compared"
    cmp = PA.compare(recording["buffers"], own["buffers"])
    assert not cmp["divergences"], cmp["divergences"]
