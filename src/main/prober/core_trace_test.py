"""`main.prober.core_trace` without a bridge: the cross-check's refusal, the live masked softmax, and
the session's meta-only read (a core trace is never expanded to answer `run_summary`).

The end-to-end expansion — the Rust core's walk (`gen3_core_walk_v1`) — against the stored rows
is `core_trace_integration_test.py` (`sim`; its equality with the live Python recorder was retired, P6 slice 6d-1); the walk's own pure helpers are
`core_walk_test.py`. Milliseconds; unmarked.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pytest

from agents.training.reward_config import terminal_breakdown
from main.prober.core_trace import (
    CoreTraceError, CoreTraceMismatch, _check_protocol, _masked_probs, expand, protocol_lines,
    reward_config_for,
)
from main.train.parser.objective import OBJECTIVE_CONSTANTS


def test_the_cross_check_ignores_timestamps_and_names_the_first_differing_line():
    rec = ["|t:|<NORMALIZED>", "|turn|1", "|move|p1a: Jirachi|Wish|p1a: Jirachi", "|upkeep"]
    rep = ["|t:|1727", "", "|turn|1", "|move|p1a: Jirachi|Wish|p1a: Jirachi", "|upkeep"]
    _check_protocol(rep, rec, "r")                        # equal modulo |t:| and blanks
    assert protocol_lines(rep) == protocol_lines(rec)
    with pytest.raises(CoreTraceMismatch, match=r"protocol line 1: record='\|move\|p1a: Jirachi\|Wish"):
        _check_protocol(["|turn|1", "|move|p1a: Jirachi|Protect|p1a: Jirachi"], rec, "r")
    with pytest.raises(CoreTraceMismatch, match="record has 1 more, first '\\|upkeep'"):
        _check_protocol(rep[:-1], rec, "r")


def test_probs_are_the_live_masked_softmax():
    logits = np.array([-1e9, -0.5, -1.2] + [-1e9] * 8, dtype=np.float32)
    mask = np.array([0, 1, 1] + [0] * 8, dtype=np.int8)
    p = _masked_probs(logits, mask)
    assert p.dtype == np.float32 and p[mask == 0].sum() == 0
    assert np.isclose(p.sum(), 1.0) and np.isclose(p[1] / p[2], np.exp(0.7), rtol=1e-5)


def test_a_non_core_summary_passes_through_untouched():
    s = {"meta": {"step": 1}, "invocations": []}
    assert expand("/nowhere_summary.json", s) is s


def _core_meta_only(tmp_path) -> str:
    d = tmp_path / "eval_traces" / "step_1000" / "heuristic"
    d.mkdir(parents=True)
    sp = d / "loss_s0_001_summary.json"
    sp.write_text(json.dumps({"meta": {
        "step": 1000, "battle_id": "core-1000-heuristic-g0", "result": "LOSS", "turns": 9,
        "invocations": 7, "result_vocabulary": "gen3_trace_result_v2",
        "trace_source": {"schema": "gen3_core_trace_v1", "kind": "gen3_core_event_v1",
                         "env_core": "rust", "trainee_side": "p1",
                         "records": {"p1": "loss_s0_001.p1.jsonl.gz", "p2": "loss_s0_001.p2.jsonl.gz"},
                         "reconstruction": "loss_s0_001_reconstruction.json",
                         "states": "loss_s0_001_states.npz"}}}))
    return str(sp)


def test_the_session_reads_a_core_meta_without_expanding_and_refuses_a_broken_trace(tmp_path):
    from main.prober.session import ProbeSession

    sp = _core_meta_only(tmp_path)
    sess = ProbeSession(str(tmp_path))
    (b,) = sess.tree.all_battles()
    assert sess._meta(b)["battle_id"] == "core-1000-heuristic-g0"
    summ = sess.run_summary()                       # meta-only: never touches the missing siblings
    assert summ["result_vocabulary"] == ["gen3_trace_result_v2"]
    with pytest.raises(CoreTraceError, match="missing record, reconstruction, states"):
        sess._summary(b)
    assert os.listdir(os.path.dirname(sp)) == [os.path.basename(sp)]   # nothing written


# -- the reward a core trace is scored with: the PRODUCTION terminal, never the retired signed one ----

_BOARDS = {                       # (won, lost, finished, turn) — a win, a loss, a pre-cap tie, the 250-turn stall
    "win": (True, False, True, 40), "loss": (False, True, True, 40),
    "tie": (False, False, True, 40), "timeout": (False, True, True, 250),
}


def _fold(cfg) -> dict:
    return {k: terminal_breakdown(cfg, won=w, lost=l, finished=f, turn=t).win_loss
            for k, (w, l, f, t) in _BOARDS.items()}


def test_a_run_that_records_no_terminal_is_scored_with_the_production_terminal():
    """No `model_config.json` (`{}`), or one with no terminal fields, used to fall to RewardConfig's
    dataclass defaults — the RETIRED signed terminal — and print +30 / -30 / -35 on a trace the Rust core
    played under the win indicator. It is 1 / 0 / 0 / 0, the namespace constants."""
    production = {"win": 1.0, "loss": 0.0, "tie": 0.0, "timeout": 0.0}
    for absent in (None, {}, {"arch": "gen3_x"}):
        cfg = reward_config_for(absent)
        assert (cfg.terminal_indicator, cfg.victory_value, cfg.draw_penalty) == (
            OBJECTIVE_CONSTANTS["terminal_indicator"], OBJECTIVE_CONSTANTS["victory_value"],
            OBJECTIVE_CONSTANTS["draw_penalty"])
        assert _fold(cfg) == production


def test_a_recorded_terminal_still_wins_over_the_default():
    """The run's RECORD decides when it states a terminal — the production one, or a retired signed one a
    pre-win-prob run recorded (read faithfully, not silently rewritten to the production values)."""
    signed = reward_config_for({"terminal_indicator": False, "victory_value": 30.0, "draw_penalty": -35.0})
    assert _fold(signed) == {"win": 30.0, "loss": -30.0, "tie": -30.0, "timeout": -35.0}
    assert _fold(reward_config_for({"terminal_indicator": True, "victory_value": 1.0, "draw_penalty": 0.0})) == {
        "win": 1.0, "loss": 0.0, "tie": 0.0, "timeout": 0.0}


def test_the_indicator_terminal_never_reads_draw_penalty():
    """Under the win indicator every non-win pays 0.0 whatever the (inert, recorded) `draw_penalty` says."""
    for dp in (0.0, -35.0, 12.5):
        cfg = reward_config_for({"terminal_indicator": True, "victory_value": 1.0, "draw_penalty": dp})
        assert _fold(cfg) == {"win": 1.0, "loss": 0.0, "tie": 0.0, "timeout": 0.0}, dp
