"""Unit tests for the frozen-snapshot ELO ladder — the durable store + the anchored fit.
Pure (no bridge / no model load): exercises games.jsonl accumulation, pair-key symmetry, the
measure-once idempotence contract, and the BT fit recovering a known ordering."""
import json
import os

from agents.training import snapshot_ladder as sl


def _append_pair(run_dir, a, b, wins_a, games):
    sl._append_game(run_dir, a, b, wins_a, games)


def test_games_log_accumulates_and_is_symmetric(tmp_path):
    run = str(tmp_path)
    _append_pair(run, 200, 100, 30, 40)       # 200 beat 100 30/40
    _append_pair(run, 100, 200, 5, 10)        # reversed order, 100 beat 200 5/10 → 200 won 5/10
    g = sl.load_games(run)
    lo, hi = (100, 200)
    assert (lo, hi) in g
    wins_lo, total = g[(lo, hi)]
    # 100 (lo) wins: from row1 it lost 30/40 → 10 wins; from row2 it won 5/10 → 5 wins ⇒ 15/50
    assert total == 50 and wins_lo == 15


def test_pair_key_orders(tmp_path):
    assert sl._pair_key(248, 100) == (100, 248)
    assert sl._pair_key(100, 248) == (100, 248)


def test_measure_once_contract(tmp_path, monkeypatch):
    """_measure_missing must SKIP pairs already in games.jsonl (frozen ⇒ stationary ⇒ measure
    once). We stub the bridge play so no model/battle is needed, and assert it only fires for the
    unmeasured pair."""
    run = str(tmp_path)
    _append_pair(run, 208, 224, 55, 100)      # already measured

    played = []

    def fake_play(run_dir, a, b, n, *a_, **k_):
        played.append((a, b))
        return 60, 100  # wins_a, finished

    monkeypatch.setattr(sl, "_play_pair", fake_play)
    # Stub the heavy loaders `_measure_missing` imports INSIDE its body. Patch ATTRIBUTES on the
    # real modules — never `sys.modules[...]` with a SimpleNamespace, which is what made this test
    # order-dependent: replacing `agents.observation.state_encoder` wholesale broke the very next
    # `agents.model.snapshot` import (`from ...state_encoder import Gen3ObservationEncoder`), so
    # the test passed only when some EARLIER test in the session had already imported snapshot.
    # A function-local `from X import y` re-reads the attribute at call time, so this is enough.
    import agents.model.snapshot            # noqa: F401 — import before patching, not after
    import agents.observation.state_encoder  # noqa: F401
    import utils.team_loader                 # noqa: F401

    monkeypatch.setattr("agents.observation.state_encoder.load_mappings", lambda: {})
    monkeypatch.setattr("agents.model.snapshot.current_model_version", lambda m: None)

    class _Loader:
        def get_all_teams(self): return ["t"]
        def get_sample_teams(self): return ["t"]
    monkeypatch.setattr("utils.team_loader.TeamLoader", _Loader)

    n = sl._measure_missing(run, [(208, 224), (208, 240)], n_games=100, concurrency=1, impl="node")
    assert (208, 224) not in played      # already measured → skipped
    assert (208, 240) in played          # new → played
    assert n == 1


def test_fit_recovers_ordering(tmp_path, monkeypatch):
    """With a clean transitive ladder (later snapshots beat earlier ones ~65%), the anchored fit
    must rank them monotonically. No bot anchors present → gauge-pinned, ordering still holds."""
    run = str(tmp_path)
    steps = [100, 200, 300]
    # transitive: higher beats lower 65%
    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            _append_pair(run, b, a, 65, 100)   # b (higher) beat a 65/100
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl.elo_mod, "load_rows", lambda run_dir, source="log": [])
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda run_dir: steps)
    ladder = sl.fit_ladder(run)
    r = ladder["ratings"]
    assert r["300"] > r["200"] > r["100"]
    assert os.path.exists(sl.ladder_json_path(run))
    assert ladder["fit_quality"]["n_frozen_pairs"] == 3


def test_latest_promoted_elo_reads_sidecar(tmp_path):
    run = str(tmp_path)
    os.makedirs(os.path.join(run, "snapshot_ladder"))
    json.dump({"ratings": {"100": 1900.0, "248": 2020.0}, "se": {"248": 12.0}},
              open(sl.ladder_json_path(run), "w"))
    step, elo, se = sl.latest_promoted_elo(run)
    assert step == 248 and elo == 2020.0 and se == 12.0
