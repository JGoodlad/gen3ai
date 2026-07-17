"""Unit tests for TeamPFSPCallback — the centralized cross-worker aggregation + GIGO guard.

The weight MATH (compute_team_pfsp_weights) is pinned in utils/teambuilder_test.py; here we pin the
callback's aggregation (sum across workers → EMA → push), the update_every throttle, the None-worker
filter, and — critically — the pool-size-mismatch GIGO guard (a silent per-index win-rate corruption
if a worker's pool order ever diverged)."""
import json
import types
import pytest
from agents.training.team_pfsp_callback import TeamPFSPCallback


class _FakeLogger:
    def __init__(self):
        self.records = {}
    def record(self, k, v):
        self.records[k] = v


class _FakeVecEnv:
    """Stub VecEnv: ``drain_team_pfsp_counts`` → preset per-worker counts; ``get_team_pfsp_keys`` →
    per-worker fingerprint lists (default = deterministic ``k####`` per pool slot, identical across
    workers); ``set_team_pfsp_weights`` captures the pushed weight vector."""
    def __init__(self, drain_results, keys=None):
        self._drain = drain_results
        if keys is None:
            keys = [None if r is None else [f"k{i:04d}" for i in range(r[2])] for r in drain_results]
        self._keys = keys
        self.pushed = None
    def env_method(self, name, *args):
        if name == "drain_team_pfsp_counts":
            return self._drain
        if name == "get_team_pfsp_keys":
            return self._keys
        if name == "set_team_pfsp_weights":
            self.pushed = args[0]
            return [None] * len(self._drain)
        raise AssertionError(f"unexpected env_method {name!r}")


def _mk(env, update_every=1, cap=3.0, floor=0.05, beta=0.7, mode="var", persist_dir=None):
    cb = TeamPFSPCallback(cap=cap, floor=floor, ema_beta=beta, update_every=update_every,
                          mode=mode, persist_dir=persist_dir)
    # BaseCallback.training_env / .logger are getter-only properties over self.model.
    cb.model = types.SimpleNamespace(logger=_FakeLogger(), get_env=lambda: env)
    return cb


def test_aggregates_across_workers_and_up_weights_the_competitive_team():
    # 3 pool teams, 2 workers. Team 0 always wins, team 1 always loses, team 2 ~50% (max variance).
    drain = [
        ([2.0, 0.0, 1.0], [2.0, 2.0, 2.0], 3),
        ([2.0, 0.0, 1.0], [2.0, 2.0, 2.0], 3),
    ]
    env = _FakeVecEnv(drain)
    cb = _mk(env)
    assert cb._on_rollout_end() is True
    # summed W=[4,0,2] G=[4,4,4] → p=[1,0,.5]; EMA(seed .5,β.7): [.65,.35,.5]
    w = env.pushed
    assert w is not None and len(w) == 3
    assert w[2] > w[0] and w[2] > w[1]     # the ~50% (most-to-learn) team is sampled most


def test_identity_guard_raises_on_pool_order_mismatch():
    # SAME pool size but a DIFFERENT team per index (order diverged) → the strong guard fires.
    drain = [([1.0, 0.0], [1.0, 1.0], 2), ([1.0, 0.0], [1.0, 1.0], 2)]
    keys = [["aaa", "bbb"], ["bbb", "aaa"]]     # index 0/1 name different teams across workers
    cb = _mk(_FakeVecEnv(drain, keys=keys))
    with pytest.raises(RuntimeError, match="IDENTITY mismatch"):
        cb._on_rollout_end()


def test_size_belt_raises_on_pool_size_mismatch():
    # keys=None (not PFSP-capable → identity check skipped) isolates the cheap per-cycle SIZE belt.
    drain = [
        ([1.0, 0.0], [1.0, 1.0], 2),
        ([1.0, 0.0, 0.0], [1.0, 1.0, 1.0], 3),
    ]
    cb = _mk(_FakeVecEnv(drain, keys=[None, None]))
    with pytest.raises(RuntimeError, match="pool-size mismatch"):
        cb._on_rollout_end()


def test_update_every_throttles_pull_and_push():
    env = _FakeVecEnv([([1.0], [1.0], 1)])
    cb = _mk(env, update_every=3)
    cb._on_rollout_end(); assert env.pushed is None      # rollout 1 — skipped
    cb._on_rollout_end(); assert env.pushed is None      # rollout 2 — skipped
    cb._on_rollout_end(); assert env.pushed is not None  # rollout 3 — fires


def test_none_workers_are_filtered_not_fatal():
    # Workers with no self-play battles / not team-pfsp-capable report None → filtered out.
    env = _FakeVecEnv([None, ([2.0], [2.0], 1), None])
    cb = _mk(env)
    assert cb._on_rollout_end() is True
    assert env.pushed is not None and len(env.pushed) == 1


def test_all_none_is_a_noop():
    env = _FakeVecEnv([None, None])
    cb = _mk(env)
    assert cb._on_rollout_end() is True
    assert env.pushed is None


def test_measure_mode_persists_snapshot_but_never_pushes(tmp_path):
    # 3 pool teams: team 0 always wins, team 1 always loses, team 2 ~50%.
    drain = [([2.0, 0.0, 1.0], [2.0, 2.0, 2.0], 3), ([2.0, 0.0, 1.0], [2.0, 2.0, 2.0], 3)]
    env = _FakeVecEnv(drain)
    cb = _mk(env, mode="measure", persist_dir=str(tmp_path))
    assert cb._on_rollout_end() is True
    assert env.pushed is None                       # measure NEVER biases sampling
    snap = json.load(open(tmp_path / "team_winrates.json"))
    assert snap["mode"] == "measure"
    assert snap["n_teams_measured"] == 3
    wrs = [t["win_rate"] for t in snap["teams"]]
    assert wrs == sorted(wrs)                        # weakest-first (ascending win-rate)
    assert snap["teams"][0]["win_rate"] < snap["teams"][-1]["win_rate"]
    assert all(("sha" in t and "games" in t) for t in snap["teams"])
    # A step-tagged history row is appended for offline trend/noise tracking over time.
    hist_lines = (tmp_path / "team_winrates_history.jsonl").read_text().strip().splitlines()
    assert len(hist_lines) == 1
    row = json.loads(hist_lines[0])
    assert "step" in row and len(row["wr"]) == 3
