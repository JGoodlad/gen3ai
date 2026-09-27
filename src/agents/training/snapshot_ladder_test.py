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


# ── the eval-cycle SENTINEL edges must NOT enter the ladder fit ───────────────────────────────
# `elo._rows_to_results` yields BOTH families off one eval row: trainee-vs-bot (`bot:`) and
# trainee-vs-sentinel (`snap:` vs `snap:`). The second is a DIFFERENT measurement of the same
# frozen pair `games.jsonl` already holds — eval plays greedy trainee vs STOCHASTIC sentinel with
# an asymmetric teambuilder, the ladder plays greedy-vs-greedy with symmetric builders. Measured
# 2026-09-07 over the 60 shared pairs on `ai_v12_02_winprob_critic`: the eval edge favours the
# newer snapshot by +8.9 pp [+7.0, +10.7], and mixing them inflated the newest ladder nodes by
# +21..+29 Elo. So `fit_ladder` keeps the BOT edges (the anchor) and drops the sentinel ones.

def _plant_eval_log(run_dir, rows):
    """Write an `eval_results.jsonl` in the flat shape `_rows_from_log` reads."""
    with open(os.path.join(run_dir, "eval_results.jsonl"), "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def test_fit_ladder_keeps_bot_edges_and_drops_eval_sentinel_edges(tmp_path, monkeypatch):
    """One planted eval row carrying ONE bot edge and ONE sentinel edge.

    It bites twice: the sentinel edge (200 beats 100 at 0.95) disagrees violently with the dense
    edge for that same pair (an even 50/100), so had it entered the fit it would have opened a
    large gap — and the results list handed to the fit is inspected directly for any snap-vs-snap
    pair coming from the log source.
    """
    run = str(tmp_path)
    steps = [100, 200]
    _append_pair(run, 200, 100, 50, 100)      # the DENSE edge: dead even
    _plant_eval_log(run, [{"step": 200, "n_games": 100,
                           "bots": {"random": 0.9},
                           "sentinels": [{"step": 100, "win_rate": 0.95}]}])

    seen: list[list[tuple[str, str, int, int]]] = []
    real_fit = sl.elo_mod.fit_pairwise

    def spy(results, *a, **k):
        seen.append(list(results))
        return real_fit(results, *a, **k)

    monkeypatch.setattr(sl.elo_mod, "fit_pairwise", spy)
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda run_dir: steps)

    ladder = sl.fit_ladder(run, write=False)

    # (a) the drop is COUNTED and reported
    assert ladder["eval_sentinel_edges_dropped"] == 1

    results = seen[0]
    snap_pairs = [r for r in results
                  if sl.elo_mod.is_snapshot(r[0]) and sl.elo_mod.is_snapshot(r[1])]
    # (b) exactly ONE snapshot-vs-snapshot edge reached the fit — the dense one, 50/100.
    #     The 95/100 sentinel version of the SAME pair is absent.
    assert len(snap_pairs) == 1
    assert snap_pairs[0][2:] == (50, 100)
    # (c) the BOT edge (the anchor connection) DID enter
    bot_pairs = [r for r in results
                 if r[1].startswith("bot:") or r[0].startswith("bot:")]
    assert len(bot_pairs) == 1 and bot_pairs[0][2:] == (90, 100)

    # (d) it BITES: with only the even dense edge the two nodes are level; the dropped sentinel
    #     edge would have pushed 200 far above 100.
    r = ladder["ratings"]
    assert abs(r["200"] - r["100"]) < 5.0


def test_fit_ladder_sentinel_drop_changes_the_rating(tmp_path, monkeypatch):
    """The same planted log, fit WITH the sentinel edge re-admitted, must differ — otherwise the
    test above would pass on a no-op filter."""
    run = str(tmp_path)
    steps = [100, 200]
    _append_pair(run, 200, 100, 50, 100)
    _plant_eval_log(run, [{"step": 200, "n_games": 100,
                           "bots": {"random": 0.9},
                           "sentinels": [{"step": 100, "win_rate": 0.95}]}])
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda run_dir: steps)
    fixed = sl.fit_ladder(run, write=False)["ratings"]

    # the pre-fix behaviour, reconstructed by hand from the same two sources
    results = [(sl.elo_mod.snap_key(100), sl.elo_mod.snap_key(200), 50, 100)]
    results += [(a, b, w, g) for a, b, w, g in
                sl.elo_mod._rows_to_results(sl.elo_mod.load_rows(run, source="log"))]
    ratings, _se, _c = sl.elo_mod.fit_pairwise(results)
    old_gap = ratings[sl.elo_mod.snap_key(200)] - ratings[sl.elo_mod.snap_key(100)]
    new_gap = fixed["200"] - fixed["100"]
    assert old_gap > 100.0          # the sentinel edge alone opens a large gap
    assert abs(new_gap) < 5.0       # dropping it leaves the honest, even dense edge


def test_fit_ladder_first_n_and_steps_kwargs_still_work(tmp_path, monkeypatch):
    """`first_n` / `steps` / `write` are unchanged by the sentinel filter."""
    run = str(tmp_path)
    steps = [100, 200, 300]
    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            _append_pair(run, b, a, 65, 100)
    _plant_eval_log(run, [{"step": 300, "n_games": 100, "bots": {"random": 0.9},
                           "sentinels": [{"step": 100, "win_rate": 0.95}]}])
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda run_dir: steps)

    pref = sl.fit_ladder(run, first_n=2)
    assert set(pref["ratings"]) == {"100", "200"}
    assert pref["first_n"] == 2
    assert not os.path.exists(sl.ladder_json_path(run))   # first_n never writes
    # the 300-row's sentinel edge (300 vs 100) has an endpoint OUTSIDE the prefix, so `_kept`
    # already excluded it — the drop counter only counts edges that survived the prefix filter
    assert pref["eval_sentinel_edges_dropped"] == 0

    sliced = sl.fit_ladder(run, steps=[100, 200], write=False)
    assert set(sliced["ratings"]) == {"100", "200"}
    assert not os.path.exists(sl.ladder_json_path(run))

    full = sl.fit_ladder(run)                              # write=True default
    assert os.path.exists(sl.ladder_json_path(run))
    assert full["eval_sentinel_edges_dropped"] == 1
    assert full["ratings"]["300"] > full["ratings"]["200"] > full["ratings"]["100"]


def test_the_per_cycle_eval_elo_star_fit_still_counts_sentinel_edges(tmp_path):
    """🚨 THE EXCLUSION IS THE LADDER'S ALONE. `elo.fit_from_run` (the per-cycle `eval/elo`
    written by `record_elo`) DELIBERATELY uses the sentinel edges — it has no dense matrix to
    prefer, and those edges are the only resolution the saturated bots cannot give. This change
    must not reach it."""
    from agents.training import elo as elo_mod
    run = str(tmp_path)
    _plant_eval_log(run, [
        {"step": 100, "n_games": 100, "bots": {"random": 0.9}, "sentinels": []},
        {"step": 200, "n_games": 100, "bots": {"random": 0.9},
         "sentinels": [{"step": 100, "win_rate": 0.95}]},
    ])
    rows = elo_mod.load_rows(run, source="log")
    pairs = list(elo_mod._rows_to_results(rows))
    snap_pairs = [p for p in pairs
                  if elo_mod.is_snapshot(p[0]) and elo_mod.is_snapshot(p[1])]
    assert snap_pairs == [(elo_mod.snap_key(200), elo_mod.snap_key(100), 95, 100)]

    fit = elo_mod.fit_from_run(run, source="log", anchors_path="/nonexistent-anchors.json")
    # both snapshots are rated, and 200 sits far above 100 — which is only possible through the
    # sentinel edge (their bot win rates are identical).
    assert fit.ratings[elo_mod.snap_key(200)] - fit.ratings[elo_mod.snap_key(100)] > 100.0


# ── RECIPE v3 (owner decision 2026-09-27): the games that SELECTED a snapshot never RATE it ─────
# Promotion is decided by the eval cycle's games vs the sentinels; v2 then REUSED those games as
# ladder edges (`source: "eval_cycle"`) — a winner's curse of ~+15..+40 Elo at n = 100. v3 plays a
# FRESH 200-game baseline vs each sentinel the eval used, plus the usual 100 vs every other node.

def _symmetric_row(trainee, sentinels, *, greedy=True, symmetric=True, n_games=100):
    return {"step": trainee, "n_games": n_games,
            "bots": {"random": 0.9},
            "sentinel_regime": {"greedy": greedy, "symmetric_teams": symmetric},
            "sentinels": sentinels}


def test_eval_measured_pairs_reads_only_fully_symmetric_rows(tmp_path):
    """The READ of what an eval cycle measured under the ladder's protocol survives (the
    winner's-curse comparator and the writer→row contract) — it just never becomes an edge."""
    run = str(tmp_path)
    _plant_eval_log(run, [
        _symmetric_row(300, [{"step": 100, "win_rate": 0.6, "counts": [60, 100]}]),
        _symmetric_row(300, [{"step": 200, "win_rate": 0.6, "counts": [60, 100]}], symmetric=False),
        _symmetric_row(400, [{"step": 100, "win_rate": 0.6, "counts": [60, 100]}], greedy=False),
        {"step": 400, "n_games": 100, "sentinels": [{"step": 200, "win_rate": 0.6}]},
    ])
    assert sl.eval_measured_pairs(run) == {(100, 300): [40, 100]}


def test_eval_measured_pairs_falls_back_to_win_rate_when_a_row_has_no_counts(tmp_path):
    run = str(tmp_path)
    _plant_eval_log(run, [_symmetric_row(300, [{"step": 100, "win_rate": 0.63}], n_games=200)])
    assert sl.eval_measured_pairs(run) == {(100, 300): [74, 200]}   # 200 - round(0.63*200)


def test_promotion_sentinel_steps_reads_the_promoting_cycles_sentinels_in_ANY_regime(tmp_path):
    """Selection is what causes the curse, not the protocol — so a stochastic-regime row's
    sentinels get the fresh 200-game baseline too."""
    run = str(tmp_path)
    _plant_eval_log(run, [
        _symmetric_row(300, [{"step": 100, "win_rate": 0.6}], greedy=False, symmetric=False),
        {"step": 300, "n_games": 100, "sentinels": [{"step": 200, "win_rate": 0.6}]},
        _symmetric_row(400, [{"step": 250, "win_rate": 0.6}]),     # a different cycle
    ])
    assert sl.promotion_sentinel_steps(run, 300) == [100, 200]
    assert sl.promotion_sentinel_steps(run, 999) == []


def _stub_players(monkeypatch, played):
    """Stub the bridge + the heavy loaders `_measure_missing` imports in its body (attributes on
    the real modules — see `test_measure_once_contract`)."""
    import agents.model.snapshot            # noqa: F401
    import agents.observation.state_encoder  # noqa: F401
    import utils.team_loader                 # noqa: F401

    def fake_play(run_dir, a, b, n, *a_, **k_):
        played.append((a, b, n))
        return n // 2 + 1, n

    class _Loader:
        def get_all_teams(self): return ["t"]
        def get_sample_teams(self): return ["t"]

    monkeypatch.setattr(sl, "_play_pair", fake_play)
    monkeypatch.setattr("agents.observation.state_encoder.load_mappings", lambda: {})
    monkeypatch.setattr("agents.model.snapshot.current_model_version", lambda m: None)
    monkeypatch.setattr("utils.team_loader.TeamLoader", _Loader)


def test_v3_promotion_ingests_NO_eval_cycle_edge_and_plays_200_FRESH_vs_each_sentinel(
        tmp_path, monkeypatch, capsys):
    """🚨 Fails on revert: v2 folded the greedy+symmetric eval edges in as `eval_cycle` rows and
    then skipped those pairs. v3 writes no eval_cycle row, plays each sentinel pair FRESH at 200
    (tagged `promotion_baseline`), plays the rest at 100, and prints the fresh-game cost."""
    run = str(tmp_path)
    _plant_eval_log(run, [_symmetric_row(400, [
        {"step": 100, "win_rate": 0.60, "counts": [60, 100]},
        {"step": 200, "win_rate": 0.55, "counts": [55, 100]},
    ])])
    played: list = []
    _stub_players(monkeypatch, played)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200, 300, 400])
    monkeypatch.setattr(sl, "fit_ladder", lambda d, *a, **k: {"ratings": {}})

    sl.update_for_promotion(run, 400, n_games=100)

    assert sorted(played) == [(400, 100, 200), (400, 200, 200), (400, 300, 100)]
    rows = [json.loads(x) for x in open(sl.games_log_path(run))]
    assert all(r.get("source") != sl.EVAL_CYCLE_SOURCE for r in rows)
    by_pair = {(r["a"], r["b"]): r for r in rows}
    assert by_pair[(400, 100)]["source"] == sl.PROMOTION_BASELINE_SOURCE
    assert by_pair[(400, 100)]["games"] == 200
    assert "source" not in by_pair[(400, 300)]
    assert all(r["recipe_version"] == sl.LADDER_FITTER_VERSION == 3 for r in rows)
    out = capsys.readouterr().out
    assert "FRESH GAMES THIS PROMOTION: 500" in out            # 2 x 200 + 1 x 100


def test_promotion_plan_is_the_cost_line(tmp_path, monkeypatch):
    run = str(tmp_path)
    _plant_eval_log(run, [_symmetric_row(400, [{"step": 100, "win_rate": 0.6},
                                                {"step": 50, "win_rate": 0.6}])])   # 50 groomed
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200, 300, 400])
    _append_pair(run, 400, 300, 50, 100)                       # already measured: never replayed
    plan = sl.promotion_plan(run, 400)
    assert plan["baseline"] == [(400, 100)] and plan["round_robin"] == [(400, 200)]
    assert plan["already_measured"] == 1 and plan["sentinels_not_in_pool"] == [50]
    assert plan["fresh_games"] == 200 + 100


def test_a_v2_eval_cycle_row_is_NOT_a_rating_edge_and_reads_as_missing(tmp_path, monkeypatch):
    """A games.jsonl a v2 tree wrote: the eval_cycle row is ignored by the fit, the pair reads
    unmeasured, and it is exactly `--backfill-fresh`'s work list."""
    run = str(tmp_path)
    sl._append_game(run, 100, 300, 40, 100, source=sl.EVAL_CYCLE_SOURCE)
    _append_pair(run, 100, 200, 50, 100)
    _append_pair(run, 200, 300, 50, 100)
    assert sl.load_games(run) == {(100, 200): [50, 100], (200, 300): [50, 100]}
    assert sl.load_games(run, include_eval_cycle=True)[(100, 300)] == [40, 100]
    assert sl.missing_pairs(run, [(300, 100)]) == [(300, 100)]
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200, 300])
    assert sl.eval_cycle_only_pairs(run) == [(100, 300)]
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    ladder = sl.fit_ladder(run, write=False)
    assert ladder["n_frozen_pairs_measured"] == 2
    assert ladder["recipe"]["eval_cycle_pair_edges_used"] is False
    assert ladder["recipe"]["eval_cycle_pair_edges_dropped_count"] == 1
    assert ladder["pairs_by_source"] == {"eval_cycle": 1, "ladder": 2}   # provenance still shown


def test_backfill_fresh_replaces_the_eval_cycle_pairs_at_200_and_is_idempotent(
        tmp_path, monkeypatch):
    run = str(tmp_path)
    sl._append_game(run, 100, 300, 40, 100, source=sl.EVAL_CYCLE_SOURCE)
    _append_pair(run, 100, 200, 50, 100)
    played: list = []
    _stub_players(monkeypatch, played)
    monkeypatch.setattr(sl, "pool_snapshot_steps", lambda d: [100, 200, 300])
    monkeypatch.setattr(sl, "fit_ladder", lambda d, *a, **k: {"ratings": {}})
    assert sl.backfill_fresh(run, dry_run=True) == {} and played == []
    sl.backfill_fresh(run)
    assert played == [(100, 300, 200)]
    sl.backfill_fresh(run)
    assert played == [(100, 300, 200)]                       # nothing left to replace
    assert sl.load_games(run)[(100, 300)] == [101, 200]


# ── THE RELATIVE COLUMN — Elo above a pinned frozen reference node ─────────────────────────────

def _bt_consistent(run, strengths, n=10_000):
    """Append every pair with counts EXACTLY consistent with Bradley-Terry at `strengths`."""
    steps = sorted(strengths)
    for i, a in enumerate(steps):
        for b in steps[i + 1:]:
            p_a = sl.elo_mod.win_prob(strengths[a], strengths[b])
            _append_pair(run, a, b, int(round(p_a * n)), n)


def test_the_relative_column_is_INVARIANT_to_adding_a_stronger_node(tmp_path, monkeypatch):
    """🚨 Fails on revert (no column). Frozen pairs never change, so adding a node — however
    strong — must not move the existing nodes' Elo above the reference, while the reference sits
    at exactly 0."""
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl.elo_mod, "load_rows", lambda run_dir, source="log": [])
    strengths = {100: 1500.0, 200: 1560.0, 300: 1610.0, 400: 1640.0}
    before_dir, after_dir = str(tmp_path / "before"), str(tmp_path / "after")
    os.makedirs(before_dir), os.makedirs(after_dir)
    _bt_consistent(before_dir, strengths)
    _bt_consistent(after_dir, {**strengths, 500: 1900.0})     # a much STRONGER newcomer
    before = sl.fit_ladder(before_dir, write=False, steps=[100, 200, 300, 400])
    after = sl.fit_ladder(after_dir, write=False, steps=[100, 200, 300, 400, 500])

    assert before["reference"]["step"] == after["reference"]["step"] == 100   # first snapshot
    for s in ("100", "200", "300", "400"):
        assert abs(before["ratings_relative"][s] - after["ratings_relative"][s]) < 0.5, s
    assert before["ratings_relative"]["100"] == 0.0
    assert abs(before["ratings_relative"]["300"] - 110.0) < 1.0      # the true gap is recovered
    assert abs(after["ratings_relative"]["500"] - 400.0) < 2.0


def test_an_explicit_reference_and_a_disconnected_node(tmp_path, monkeypatch):
    monkeypatch.setattr(sl.elo_mod, "load_bot_anchors", lambda: None)
    monkeypatch.setattr(sl.elo_mod, "load_rows", lambda run_dir, source="log": [])
    run = str(tmp_path)
    _bt_consistent(run, {100: 1500.0, 200: 1560.0})
    lad = sl.fit_ladder(run, write=False, steps=[100, 200], reference=200)
    assert lad["reference"]["step"] == 200 and lad["ratings_relative"]["200"] == 0.0
    assert abs(lad["ratings_relative"]["100"] + 60.0) < 1.0
    # 300 has no frozen edge to the reference's component: omitted, never a prior-only number
    rel, _ = sl.fit_relative([(sl.elo_mod.snap_key(100), sl.elo_mod.snap_key(200), 50, 100),
                              (sl.elo_mod.snap_key(300), sl.elo_mod.snap_key(400), 50, 100)], 100)
    assert set(rel) == {"100", "200"}


def test_the_default_reference_is_the_registry_baseline_ONLY_when_this_ladder_holds_its_file(
        tmp_path):
    from agents.training import baselines
    b = baselines.get(sl.DEFAULT_REFERENCE_BASELINE)
    run = str(tmp_path / b.run)
    os.makedirs(run)
    ref, why = sl.resolve_reference(run, [16_000_032, b.num_timesteps, 30_000_000])
    assert ref == b.num_timesteps and sl.DEFAULT_REFERENCE_BASELINE in why
    # same step number in a DIFFERENT run with no matching zip → the first snapshot instead
    other = str(tmp_path / "some_other_run")
    os.makedirs(other)
    ref, why = sl.resolve_reference(other, [16_000_032, b.num_timesteps])
    assert ref == 16_000_032 and "first snapshot" in why
