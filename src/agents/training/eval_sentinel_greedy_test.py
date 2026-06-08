"""Unit guard for the --eval-sentinel-greedy thread: `_eval_sentinel` builds the pool sentinel
STOCHASTIC by default (mirroring its training-opponent regime) and GREEDY (argmax) when
`sentinel_greedy` is set, while the measured trainee is always greedy. The heavy bits
(model load, players, the battle gather) are stubbed so this stays a pure construction test."""

from unittest.mock import MagicMock

import main.eval_worker as ew


def _run_eval_sentinel(monkeypatch, sentinel_greedy):
    captured = {}

    monkeypatch.setattr(ew, "load_model_snapshot", lambda *a, **k: MagicMock(name="sentinel_model"))

    def fake_trainee(*a, **k):
        captured["trainee_stochastic"] = k.get("stochastic")
        return MagicMock(name="trainee")
    monkeypatch.setattr(ew, "EvalRLPlayer", fake_trainee)

    def fake_opponent(*a, **k):
        captured["sentinel_stochastic"] = k.get("stochastic")
        captured["sentinel_temperature"] = k.get("temperature")
        return MagicMock(name="opponent")
    monkeypatch.setattr(ew, "RLPlayer", fake_opponent)

    async def fake_matchup(*a, **k):
        return {"win_rate": 0.5, "reward_mean": 0.0, "ep_len": 1.0,
                "duration_sec": 0.0, "td_resid_tail": None}
    monkeypatch.setattr(ew, "eval_one_matchup", fake_matchup)

    spec = {"label": "sentinel_0", "path": "snap.zip", "step": 1000}
    out = ew._eval_sentinel(
        MagicMock(name="model"), spec, current_version=None,
        trainee_tb=MagicMock(), opp_tb=MagicMock(), mappings=None,
        server_config=MagicMock(), concurrency=1, device="cpu", temperature=0.7,
        n_games=2, model_dir=None, step=1000, tag="t", use_bridge=False, gamma=0.99,
        sentinel_greedy=sentinel_greedy, reward_factory=MagicMock(name="reward_factory"),
    )
    return captured, out


def test_sentinel_stochastic_by_default(monkeypatch):
    cap, out = _run_eval_sentinel(monkeypatch, sentinel_greedy=False)
    assert cap["sentinel_stochastic"] is True          # default: samples at temperature
    assert cap["sentinel_temperature"] == 0.7
    assert cap["trainee_stochastic"] is False           # measured trainee always greedy
    assert out["sentinel_step"] == 1000


def test_sentinel_greedy_when_flagged(monkeypatch):
    cap, _ = _run_eval_sentinel(monkeypatch, sentinel_greedy=True)
    assert cap["sentinel_stochastic"] is False          # best-vs-best: sentinel plays argmax
    assert cap["trainee_stochastic"] is False           # trainee still greedy


def test_eval_sentinel_greedy_defaults_off(monkeypatch):
    # Positional/older callers that don't pass sentinel_greedy keep the stochastic sentinel.
    captured = {}
    monkeypatch.setattr(ew, "load_model_snapshot", lambda *a, **k: MagicMock())
    monkeypatch.setattr(ew, "EvalRLPlayer", lambda *a, **k: MagicMock())
    monkeypatch.setattr(ew, "RLPlayer",
                        lambda *a, **k: captured.setdefault("s", k.get("stochastic")) or MagicMock())

    async def fake_matchup(*a, **k):
        return {"win_rate": 0.5, "reward_mean": 0.0, "ep_len": 1.0,
                "duration_sec": 0.0, "td_resid_tail": None}
    monkeypatch.setattr(ew, "eval_one_matchup", fake_matchup)

    ew._eval_sentinel(
        MagicMock(), {"label": "s", "path": "p", "step": 1}, None,
        MagicMock(), MagicMock(), None, MagicMock(), 1, "cpu", 1.0,
        2, None, 1, "t",   # no use_bridge/gamma/sentinel_greedy → all default
        reward_factory=MagicMock(name="reward_factory"),
    )
    assert captured["s"] is True
