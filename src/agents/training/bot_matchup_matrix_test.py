"""Unit tests for the raw bot-matchup accumulator.

PURE by construction — no bridge, no battles, no server. Every test that "plays" replaces the
``play_fn`` seam with a deterministic fake, which is the whole reason ``run_chunk`` takes one:
the accumulator's correctness (merge, atomicity, balanced fill, draw accounting, SE math) must
be provable in milliseconds, or it will only ever be checked by an hour-long real chunk.
"""
from __future__ import annotations

import os
import json
import math
import itertools

import pytest

from agents.training import bot_matchup_matrix as bmm

BOTS = ["alpha", "bravo", "charlie"]


def _pairs():
    return bmm.all_pairs(BOTS)


def _fake_play(script=None, default=(6, 3, 1)):
    """A play seam that records its calls. Returns (wins_a, wins_b, draws, finished) sized to the
    requested battle count so `n` accounting stays honest."""
    calls = []

    def play(bots, a, b, n, concurrency):
        calls.append((a, b, n, concurrency))
        wa, wb, dr = (script or {}).get(bmm.pair_key(a, b), default)
        # scale the fixed shape to exactly n finished battles
        total = wa + wb + dr
        k, rem = divmod(n, total)
        return wa * k + rem, wb * k, dr * k, n

    play.calls = calls
    return play


# ── keys / accumulate / draws ────────────────────────────────────────────────


def test_pair_key_is_canonical_and_order_free():
    assert bmm.pair_key("bravo", "alpha") == bmm.pair_key("alpha", "bravo") == "alpha|bravo"


def test_accumulate_normalizes_win_ownership_to_the_sorted_name():
    store = bmm.new_store(target=100, bots=BOTS)
    bmm.accumulate(store, "alpha", "bravo", 5, 2, 1)
    # same pair, passed the OTHER way round: bravo won 3, alpha won 1
    bmm.accumulate(store, "bravo", "alpha", 3, 1, 0)
    e = bmm.entry(store, "alpha", "bravo")
    assert (e["a"], e["b"]) == ("alpha", "bravo")
    assert e["wins_a"] == 6 and e["wins_b"] == 5 and e["draws"] == 1
    assert e["n"] == 12


def test_draws_are_stored_separately_and_never_folded_into_a_win():
    store = bmm.new_store(target=100, bots=BOTS)
    bmm.accumulate(store, "alpha", "bravo", 10, 10, 30)
    e = bmm.entry(store, "alpha", "bravo")
    assert e["draws"] == 30
    assert e["wins_a"] + e["wins_b"] == 20
    assert e["n"] == 50
    st = bmm.edge_stats(e)
    # p is over DECISIVE games only — the 30 draws must not move it off 0.5, nor count as n
    assert st["decisive"] == 20 and st["n"] == 50
    assert st["p"] == pytest.approx(0.5)


def test_accumulate_refuses_negative_counts():
    store = bmm.new_store(target=10, bots=BOTS)
    with pytest.raises(ValueError):
        bmm.accumulate(store, "alpha", "bravo", 5, -1, 0)


def test_unplayed_pair_reads_as_zeros_not_a_keyerror():
    store = bmm.new_store(target=10, bots=BOTS)
    assert bmm.n_games(store, "alpha", "charlie") == 0
    assert bmm.entry(store, "charlie", "alpha")["wins_a"] == 0


# ── IO: atomic write, merge-on-load, crash recovery ──────────────────────────


def test_atomic_write_leaves_no_tmp_and_reloads_identically(tmp_path):
    path = str(tmp_path / "m.json")
    store = bmm.new_store(target=50, bots=BOTS)
    bmm.accumulate(store, "alpha", "bravo", 3, 4, 0)
    bmm.save(path, store)
    assert not os.path.exists(path + ".tmp")
    back = bmm.load(path)
    assert bmm.entry(back, "alpha", "bravo")["wins_b"] == 4
    assert back["target_per_pair"] == 50


def test_load_of_a_missing_file_is_an_empty_store(tmp_path):
    store = bmm.load(str(tmp_path / "nope.json"), target=77, bots=BOTS)
    assert store["pairs"] == {} and store["target_per_pair"] == 77
    assert store["schema_version"] == bmm.SCHEMA_VERSION


def test_a_crash_mid_write_loses_at_most_the_in_flight_visit(tmp_path):
    """Simulate the kill: commit one visit, then leave a TRUNCATED .tmp behind (what a process
    killed inside atomic_write_json leaves). The next load must return the last COMMITTED state
    and ignore the tmp entirely — that is what tmp+rename buys."""
    path = str(tmp_path / "m.json")
    bmm.commit(path, [("alpha", "bravo", 10, 5, 0)], target=100, bots=BOTS)
    with open(path + ".tmp", "w") as f:
        f.write('{"pairs": {"alpha|bra')  # half-written, as a SIGKILL would leave it
    back = bmm.load(path)
    assert bmm.n_games(back, "alpha", "bravo") == 15
    assert bmm.entry(back, "alpha", "bravo")["wins_a"] == 10


def test_an_unreadable_artifact_starts_fresh_rather_than_raising(tmp_path):
    path = str(tmp_path / "m.json")
    with open(path, "w") as f:
        f.write("{not json at all")
    store = bmm.load(path, target=10, bots=BOTS)
    assert store["pairs"] == {}


def test_commit_merges_a_concurrent_writers_counts_instead_of_clobbering(tmp_path):
    """Two accumulators pointed at one artifact. `commit` re-reads before writing, so B's chunk
    survives A's next commit — the property that makes a fleet (or a resume) sum."""
    path = str(tmp_path / "m.json")
    a_store = bmm.commit(path, [("alpha", "bravo", 4, 1, 0)], target=100, bots=BOTS)
    assert bmm.n_games(a_store, "alpha", "bravo") == 5
    # "process B" writes a different pair while A holds a stale in-memory copy
    bmm.commit(path, [("alpha", "charlie", 7, 2, 1)], target=100, bots=BOTS)
    merged = bmm.commit(path, [("alpha", "bravo", 3, 2, 0)], target=100, bots=BOTS)
    assert bmm.n_games(merged, "alpha", "bravo") == 10
    assert bmm.n_games(merged, "alpha", "charlie") == 10


def test_merge_stores_sums_pairs_and_dedups_history_by_chunk_id():
    a = bmm.new_store(target=10, bots=BOTS)
    bmm.accumulate(a, "alpha", "bravo", 2, 1, 0)
    a["history"].append({"chunk_id": "c1", "games_added": 3})
    b = bmm.new_store(target=10, bots=BOTS)
    bmm.accumulate(b, "bravo", "alpha", 1, 2, 1)   # reversed order on purpose
    b["history"] += [{"chunk_id": "c1", "games_added": 4}, {"chunk_id": "c2", "games_added": 9}]
    m = bmm.merge_stores(a, b)
    e = bmm.entry(m, "alpha", "bravo")
    assert (e["wins_a"], e["wins_b"], e["draws"], e["n"]) == (4, 2, 1, 7)
    assert [h["chunk_id"] for h in m["history"]] == ["c1", "c2"]
    assert m["history"][0]["games_added"] == 4          # the LATER record wins


def test_history_record_is_upserted_not_appended_per_visit(tmp_path):
    path = str(tmp_path / "m.json")
    for i in range(3):
        bmm.commit(path, [("alpha", "bravo", 1, 1, 0)], target=100, bots=BOTS,
                   history={"chunk_id": "ck", "games_added": 2 * (i + 1)})
    store = bmm.load(path)
    assert len(store["history"]) == 1
    assert store["history"][0]["games_added"] == 6


def test_artifact_carries_schema_protocol_target_and_history(tmp_path):
    path = str(tmp_path / "m.json")
    bmm.commit(path, [("alpha", "bravo", 1, 0, 0)], target=10_000, bots=BOTS,
               history={"chunk_id": "x", "wall_seconds": 1.0})
    with open(path) as f:
        raw = json.load(f)
    assert raw["schema_version"] == bmm.SCHEMA_VERSION
    assert raw["target_per_pair"] == 10_000
    assert raw["bots"] == BOTS
    for key in ("team_sampling", "bridge_impl", "counting", "turn_caps", "forfeit", "draws"):
        assert key in raw["protocol"]
    assert raw["history"][0]["chunk_id"] == "x"
    assert "updated_at" in raw


# ── balanced scheduler ───────────────────────────────────────────────────────


def test_scheduler_picks_the_lowest_n_pair_first():
    store = bmm.new_store(target=100, bots=BOTS)
    bmm.accumulate(store, "alpha", "bravo", 30, 0, 0)
    bmm.accumulate(store, "alpha", "charlie", 10, 0, 0)
    # bravo|charlie has 0
    assert bmm.next_pair(store, _pairs(), 100) == ("bravo", "charlie")
    bmm.accumulate(store, "bravo", "charlie", 20, 0, 0)
    assert bmm.next_pair(store, _pairs(), 100) == ("alpha", "charlie")


def test_scheduler_tie_breaks_on_the_sorted_name_tuple_not_the_pipe_joined_key():
    """`|` sorts AFTER `_`, so keying the tie-break on pair_key would put
    `aggressive_v2|heuristic` ahead of `aggressive|aggressive_v2` — deterministic but confusing
    on the real roster, where 4 of the 9 bot names carry a `_v2` suffix."""
    store = bmm.new_store(target=100, bots=BOTS)
    assert bmm.next_pair(store, _pairs(), 100) == ("alpha", "bravo")
    real = ["aggressive", "aggressive_v2", "heuristic"]
    assert bmm.next_pair(bmm.new_store(target=100, bots=real), bmm.all_pairs(real), 100) == \
        ("aggressive", "aggressive_v2")


def test_scheduler_returns_none_once_every_pair_hits_target():
    store = bmm.new_store(target=5, bots=BOTS)
    for a, b in _pairs():
        bmm.accumulate(store, a, b, 5, 0, 0)
    assert bmm.next_pair(store, _pairs(), 5) is None


def test_scheduler_ignores_pairs_at_target_even_when_they_are_the_lowest():
    store = bmm.new_store(target=5, bots=BOTS)
    bmm.accumulate(store, "alpha", "bravo", 5, 0, 0)      # exactly at target
    bmm.accumulate(store, "alpha", "charlie", 9, 0, 0)    # over
    bmm.accumulate(store, "bravo", "charlie", 7, 0, 0)
    assert bmm.next_pair(store, _pairs(), 5) is None


# ── SE / ELO resolution math ─────────────────────────────────────────────────


def test_se_elo_matches_the_closed_form():
    e = {"a": "alpha", "b": "bravo", "wins_a": 500, "wins_b": 500, "draws": 0, "n": 1000}
    st = bmm.edge_stats(e)
    p = (500 + 0.5) / 1001.0
    expect_logit = 1.0 / math.sqrt(1000 * p * (1 - p))
    assert st["se_logit"] == pytest.approx(expect_logit)
    assert st["se_elo"] == pytest.approx(expect_logit * 400.0 / math.log(10.0))
    # sanity on the scale: ~1000 even games ≈ ±11 ELO
    assert 10.0 < st["se_elo"] < 12.0


def test_se_elo_falls_as_one_over_sqrt_n():
    small = bmm.edge_stats({"wins_a": 50, "wins_b": 50, "draws": 0, "n": 100})
    big = bmm.edge_stats({"wins_a": 5000, "wins_b": 5000, "draws": 0, "n": 10_000})
    assert small["se_elo"] / big["se_elo"] == pytest.approx(10.0, rel=0.01)


def test_a_lopsided_edge_still_reports_a_finite_bound():
    """0-of-N would divide by zero without the Haldane-Anscombe correction; an ∞ here would make
    the headline 'worst pair' line unreadable exactly when an edge is most decided."""
    st = bmm.edge_stats({"wins_a": 1000, "wins_b": 0, "draws": 0, "n": 1000})
    assert st["p"] == 1.0                      # p itself is reported UNcorrected
    assert math.isfinite(st["se_elo"]) and st["se_elo"] > 0


def test_an_unplayed_edge_reports_infinite_resolution():
    st = bmm.edge_stats({"wins_a": 0, "wins_b": 0, "draws": 0, "n": 0})
    assert math.isinf(st["se_elo"]) and math.isnan(st["p"])


def test_all_draws_edge_is_infinite_not_a_crash():
    st = bmm.edge_stats({"wins_a": 0, "wins_b": 0, "draws": 40, "n": 40})
    assert st["decisive"] == 0 and math.isinf(st["se_elo"])


def test_a_saturated_edge_is_flagged_and_a_contested_one_is_not():
    assert bmm.edge_stats({"wins_a": 500, "wins_b": 0, "draws": 0, "n": 500})["saturated"]
    assert bmm.edge_stats({"wins_a": 0, "wins_b": 500, "draws": 0, "n": 500})["saturated"]
    assert not bmm.edge_stats({"wins_a": 499, "wins_b": 1, "draws": 0, "n": 500})["saturated"]


def test_summary_separates_saturated_edges_from_the_sample_size_reading():
    """A 100%-win edge has an unbounded ELO gap, so its SE stays huge forever and would own the
    'worst pair' headline permanently. The second reading must exclude it."""
    store = bmm.new_store(target=100, bots=BOTS)
    bmm.accumulate(store, "alpha", "bravo", 200, 0, 0)      # saturated
    bmm.accumulate(store, "alpha", "charlie", 100, 100, 0)
    bmm.accumulate(store, "bravo", "charlie", 25, 25, 0)    # the least-resolved CONTESTED edge
    text = bmm.format_summary(store, _pairs(), 100)
    assert "SATURATED" in text
    assert "excluding those: worst" in text
    assert "bravo vs charlie" in text.splitlines()[-1]


def test_summary_reports_the_worst_pair_as_the_headline():
    store = bmm.new_store(target=100, bots=BOTS)
    bmm.accumulate(store, "alpha", "bravo", 500, 500, 0)
    bmm.accumulate(store, "alpha", "charlie", 50, 50, 0)
    bmm.accumulate(store, "bravo", "charlie", 5, 5, 0)
    text = bmm.format_summary(store, _pairs(), 100)
    assert "resolution now ±" in text
    assert "bravo vs charlie" in text.splitlines()[-1]   # the least-resolved edge
    assert "min n 10" in text and "max n 1000" in text


# ── run_chunk: budgets, balance, resumability ────────────────────────────────


def test_run_chunk_stops_at_the_battle_budget(tmp_path):
    path = str(tmp_path / "m.json")
    play = _fake_play()
    store = bmm.run_chunk(path, target=1000, chunk_battles=60, games_per_visit=10,
                          bot_names=BOTS, build_bots_fn=lambda n: {}, play_fn=play,
                          verbose=False)
    assert sum(c[2] for c in play.calls) == 60
    assert sum(bmm.n_games(store, a, b) for a, b in _pairs()) == 60


def test_run_chunk_stops_at_the_deadline(tmp_path):
    path = str(tmp_path / "m.json")
    clock = itertools.count(0, 30)   # 30 simulated seconds per `now()` call

    play = _fake_play()
    bmm.run_chunk(path, target=1000, max_minutes=1.0, games_per_visit=10, bot_names=BOTS,
                  build_bots_fn=lambda n: {}, play_fn=play, now=lambda: next(clock),
                  verbose=False)
    assert play.calls, "should have played at least one visit before the deadline"
    assert sum(c[2] for c in play.calls) < 200   # stopped early, not a full fill


def test_run_chunk_fills_pairs_uniformly(tmp_path):
    path = str(tmp_path / "m.json")
    play = _fake_play()
    store = bmm.run_chunk(path, target=1000, chunk_battles=90, games_per_visit=10,
                          bot_names=BOTS, build_bots_fn=lambda n: {}, play_fn=play,
                          verbose=False)
    counts = [bmm.n_games(store, a, b) for a, b in _pairs()]
    assert counts == [30, 30, 30], counts
    # ...and every prefix stays balanced: no pair is ever more than one visit ahead of another
    seen = {bmm.pair_key(a, b): 0 for a, b in _pairs()}
    for a, b, n, _c in play.calls:
        seen[bmm.pair_key(a, b)] += n
        assert max(seen.values()) - min(seen.values()) <= 10, seen


def test_run_chunk_is_resumable_and_never_overshoots_target(tmp_path):
    path = str(tmp_path / "m.json")
    kw = dict(target=40, games_per_visit=25, bot_names=BOTS,
              build_bots_fn=lambda n: {}, verbose=False)
    bmm.run_chunk(path, chunk_battles=45, play_fn=_fake_play(), **kw)
    store = bmm.run_chunk(path, play_fn=_fake_play(), **kw)      # run to completion
    counts = [bmm.n_games(store, a, b) for a, b in _pairs()]
    assert counts == [40, 40, 40], counts
    # a third run has nothing to do and must not play a single battle
    play = _fake_play()
    bmm.run_chunk(path, play_fn=play, build_bots_fn=_no_bots, target=40, games_per_visit=25,
                  bot_names=BOTS, verbose=False)
    assert play.calls == []


def _no_bots(names):
    raise AssertionError("bots must not be built when there is nothing to play")


def test_run_chunk_records_a_history_row_with_wall_time_and_games(tmp_path):
    path = str(tmp_path / "m.json")
    clock = itertools.count(1000, 5)
    bmm.run_chunk(path, target=100, chunk_battles=30, games_per_visit=10, bot_names=BOTS,
                  build_bots_fn=lambda n: {}, play_fn=_fake_play(), now=lambda: next(clock),
                  verbose=False)
    store = bmm.load(path)
    assert len(store["history"]) == 1
    rec = store["history"][0]
    assert rec["games_added"] == 30 and rec["games_scheduled"] == 30
    assert rec["wall_seconds"] > 0 and rec["pairs_touched"] == 3
    assert "started_at" in rec and "updated_at" in rec


def test_run_chunk_survives_a_crash_mid_chunk_with_the_committed_visits_intact(tmp_path):
    path = str(tmp_path / "m.json")
    calls = {"n": 0}

    def flaky(bots, a, b, n, concurrency):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("bridge child died")
        return n, 0, 0, n

    with pytest.raises(RuntimeError):
        bmm.run_chunk(path, target=100, chunk_battles=100, games_per_visit=10, bot_names=BOTS,
                      build_bots_fn=lambda n: {}, play_fn=flaky, verbose=False)
    store = bmm.load(path)
    assert sum(bmm.n_games(store, a, b) for a, b in _pairs()) == 20   # the 2 committed visits
    assert store["history"][0]["games_added"] == 20                   # honest, not the target


def test_run_chunk_budgets_on_scheduled_games_so_a_timing_out_pair_cannot_spin(tmp_path):
    """Every battle times out (finished=0). The loop must still terminate on the budget rather
    than re-scheduling a pair whose n never rises."""
    path = str(tmp_path / "m.json")

    def all_timeouts(bots, a, b, n, concurrency):
        return 0, 0, 0, 0

    store = bmm.run_chunk(path, target=100, chunk_battles=50, games_per_visit=10,
                          bot_names=BOTS, build_bots_fn=lambda n: {}, play_fn=all_timeouts,
                          verbose=False)
    assert sum(bmm.n_games(store, a, b) for a, b in _pairs()) == 0


def test_run_chunk_passes_the_requested_concurrency_to_the_play_seam(tmp_path):
    path = str(tmp_path / "m.json")
    play = _fake_play()
    bmm.run_chunk(path, target=100, chunk_battles=10, games_per_visit=10, concurrency=4,
                  bot_names=BOTS, build_bots_fn=lambda n: {}, play_fn=play, verbose=False)
    assert play.calls[0][3] == 4


# ── contract guards against drift in the inherited protocol ──────────────────


def test_recorded_bridge_impl_matches_the_runner_default():
    """The artifact claims `impl="node"` because `bot_elo_calibration._play_chunk` passes no
    `impl`. If `run_local_battles`' default ever changes, this artifact's protocol block would
    silently lie about which sim produced the counts."""
    import inspect
    from utils.bridge.local_battle_runner import run_local_battles
    default = inspect.signature(run_local_battles).parameters["impl"].default
    assert bmm.PROTOCOL["bridge_impl"] == default


def test_the_inherited_calibration_seams_still_exist():
    """`play_pair` / `build_bots` call these by name — a rename must fail here, loudly, rather
    than at hour 3 of an accumulation run."""
    from agents.training import bot_elo_calibration as cal
    assert callable(cal._build_bot) and callable(cal._play_chunk)


def test_the_anchor_path_is_not_this_modules_default_out():
    """Belt and braces on the one file this module must never write."""
    assert "elo_anchors" not in bmm.DEFAULT_OUT
    assert bmm.DEFAULT_OUT == os.path.join("data", "gen3_bot_matchups.json")
    assert not os.path.isabs(bmm.DEFAULT_OUT)   # relative ⇒ a worktree run stays in-worktree
