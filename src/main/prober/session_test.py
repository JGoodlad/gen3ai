"""Tests for the agent-facing ProbeSession + query CLI (no real checkpoint)."""

import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
import pytest

from agents.action.constants import MOVE_START
from main.prober.model import ObsOffsets
from main.prober.session import ProbeSession

_OFF = ObsOffsets(mm_off=10, om_off=20, active_block_dim=5,
                  turn_history_offset=200, turn_history_dim=10)
_OBS_LEN = 256


class _FakeModel:
    offsets = _OFF

    def action_dist(self, obs, mask):
        w = np.ones(len(mask), dtype=np.float64)
        for s in range(4):
            w[MOVE_START + s] = 1.0 + float(obs[_OFF.mm_off + s]) * 10.0
        w = w * mask.astype(np.float64)
        return w / w.sum(), np.arange(len(mask), dtype=np.float64)

    def logit_grad(self, obs, mask, idx):
        return np.arange(len(obs), dtype=np.float64)

    def value(self, obs, mask):
        return 1.23


def _build_run(tmp_path):
    run = tmp_path / "run"
    bd = run / "eval_traces" / "step_2000000" / "staller"
    os.makedirs(bd, exist_ok=True)
    actions = {f"switch:m{i}": {"prob": "1.0%", "valid": True} for i in range(6)}
    actions.update({"thunderbolt": {"prob": "92.1%", "valid": True},
                    "earthquake": {"prob": "2.8%", "valid": True},
                    "move2": {"prob": "0.0%", "valid": False},
                    "move3": {"prob": "0.0%", "valid": False},
                    "struggle": {"prob": "0.0%", "valid": False}})
    invs = [
        {"i": 1, "turn": 1, "phase": "move_selection", "chosen": "earthquake",
         "our": {"species": "zapdos"}, "opp": {"species": "jynx"}, "actions": actions,
         "outcome": {"reward": {"total": -1.4, "base": "hp=-1.2"}, "events": []}},
        {"i": 2, "turn": 2, "phase": "move_selection", "chosen": "switch:m0",
         "our": {"species": "zapdos"}, "opp": {"species": "jynx"}, "actions": actions,
         "outcome": {"reward": {"total": 0.5}, "events": ["opp:jynx:fainted"]}},
    ]
    summary = {"meta": {"step": 2000000, "result": "WIN", "turns": 4, "invocations": 2},
               "invocations": invs}
    with open(bd / "win_001_summary.json", "w") as f:
        json.dump(summary, f)
    obs = np.zeros((2, _OBS_LEN), dtype=np.float32)
    obs[:, _OFF.mm_off:_OFF.mm_off + 4] = [0.5, 0.25, 0.0, 0.125]
    np.savez(bd / "win_001_states.npz", obs=obs,
             has_state=np.array([1, 1], dtype=np.int8),
             values=np.array([2.0, 5.0], dtype=np.float32))
    # manifest (no retained snapshot → resolution stays 'nearest'); identity for run_summary
    with open(bd.parent / "eval_manifest.json", "w") as f:
        json.dump({"step": 2000000, "git_hash": "abc123",
                   "arch_signature": "gen3_x", "snapshot": None}, f)
    (run / "checkpoint_3200000_steps.zip").write_text("")  # gives the ladder a path
    with open(run / "metadata.json", "w") as f:
        json.dump({"gamma": 0.95}, f)
    return str(run), str(bd / "win_001_summary.json")


def test_battles_filter(tmp_path):
    run, _ = _build_run(tmp_path)
    sess = ProbeSession(run)
    assert len(sess.battles()) == 1
    assert sess.battles(outcome="loss") == []
    assert sess.battles(opponent="staller")[0]["step"] == 2000000


def test_battle_overview_is_model_free(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run)  # no model_loader → must not load a model
    ov = sess.battle_overview(summ)
    assert ov["meta"]["result"] == "WIN"
    assert ov["model_resolution"]["tier"] == "nearest"
    r0, r1 = ov["invocations"]
    assert r0["chosen"] == "earthquake" and r0["value"] == 2.0 and r0["reward_total"] == -1.4
    assert "switch" in r1["flags"] and "faint" in r1["flags"]


def test_find_model_free(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run)
    assert sess.find(summ, "switch") == [1]
    assert sess.find(summ, "faint") == [1]


def test_analyze_loads_resolved_model(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    d = sess.analyze(summ, 0)
    assert d["chosen"] == "earthquake"
    assert d["model_resolution"]["tier"] == "nearest"
    assert d["value"]["recorded"] == 2.0 and d["value"]["rerun"] == 1.23
    assert d["value"]["next_recorded"] == 5.0 and abs(d["value"]["delta"] - 3.0) < 1e-9
    # JSON-serializable
    json.dumps(d)


def test_find_disagree_uses_model(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    # inv0 chose earthquake but the fake favors thunderbolt → disagree
    assert 0 in sess.find(summ, "disagree")


def test_run_summary(tmp_path):
    run, _ = _build_run(tmp_path)
    s = ProbeSession(run).run_summary()
    assert s["n_steps"] == 1 and s["gamma"] == 0.95
    assert s["totals"] == {"win": 1, "loss": 0, "battles": 1}
    step = s["steps"][0]
    assert step["step"] == 2000000
    assert step["identity"]["git_hash"] == "abc123"
    assert step["identity"]["snapshot_available"] is False
    assert step["opponents"][0] == {"name": "staller", "win": 1, "loss": 0, "battles": 1}
    assert s["checkpoints"][0]["step"] == 3200000


def test_overview_delta_td_and_notable(tmp_path):
    run, summ = _build_run(tmp_path)
    ov = ProbeSession(run).battle_overview(summ)
    r0, r1 = ov["invocations"]
    assert r0["delta_v"] == 3.0                       # 5.0 - 2.0
    assert abs(r0["td_residual"] - (-1.4 + 0.95 * 5.0 - 2.0)) < 1e-9   # γ from metadata
    assert r1["delta_v"] is None and r1["td_residual"] is None         # no next decision
    nt = ov["notable"]
    assert nt["faints"] == [1] and nt["switches"] == [1] and nt["uncertain_count"] == 2
    assert nt["biggest_value_drops"] == [{"inv": 0, "delta_v": 3.0}]


def test_overview_has_active_board(tmp_path):
    run, summ = _build_run(tmp_path)
    ov = ProbeSession(run).battle_overview(summ)
    assert ov["invocations"][0]["our_active"].startswith("zapdos")
    assert ov["invocations"][0]["opp_active"].startswith("jynx")


def test_analyze_carries_board(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run, model_loader=lambda path: _FakeModel())
    d = sess.analyze(summ, 0)
    assert d["board"]["ours"]["active_species"] == "zapdos"


def test_find_value_criteria_and_limit(tmp_path):
    run, summ = _build_run(tmp_path)
    sess = ProbeSession(run)
    assert sess.find(summ, "value_drop") == [0]       # only inv0 has a next V
    assert sess.find(summ, "low_value") == [0, 1]     # V=2.0 then 5.0
    assert sess.find(summ, "high_value") == [1, 0]
    assert sess.find(summ, "low_value", limit=1) == [0]


def test_short_id_resolves(tmp_path):
    run, _ = _build_run(tmp_path)
    ov = ProbeSession(run).battle_overview("step_2000000/staller/win_001")
    assert ov["meta"]["step"] == 2000000


def test_cli_error_envelope(tmp_path):
    import sys
    import main.prober.query as q
    argv = sys.argv
    sys.argv = ["query", "analyze", str(tmp_path / "nope_summary.json"), "0"]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            with pytest.raises(SystemExit) as ei:
                q.main()
        assert ei.value.code == 1
        assert "error" in json.loads(buf.getvalue())
    finally:
        sys.argv = argv


def test_cli_overview_emits_json(tmp_path):
    import main.prober.query as q
    import sys
    run, summ = _build_run(tmp_path)
    argv = sys.argv
    sys.argv = ["query", "overview", summ]
    try:
        buf = io.StringIO()
        with redirect_stdout(buf):
            q.main()
    finally:
        sys.argv = argv
    parsed = json.loads(buf.getvalue())
    assert parsed["meta"]["step"] == 2000000 and len(parsed["invocations"]) == 2
