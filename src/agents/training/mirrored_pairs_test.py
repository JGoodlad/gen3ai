"""T17 MIRRORED TEAM PAIRS (`gen3_mirrored_pairs_v1`) — the pairing rule, the plan, the pair statistics,
and every reader's refusal to cross the regime boundary.

Each test names the defect it would let through if the code it pins were reverted.
"""
from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from agents.training import mirrored_pairs as MP
from agents.training.eval_sharding import BOT, EvalItem, ShardedEvalPool, ShardResult
from agents.training.eval_sharding.results import aggregate, to_merged, write_shard_result
from agents.training.eval_sharding.units import _split_games, game_range, plan_units
from agents.training.rust_eval import seeds as SD


# ---------------------------------------------------------------- the pentanomial + its interval

def test_pair_counts_score_each_pair_in_half_points_and_void_an_unfinished_pair():
    # pairs: (W,W)=4, (W,L)=2, (D,L)=1, (None,W) voided, (L,L)=0
    pts = [2, 2, 2, 0, 1, 0, None, 2, 0, 0]
    assert MP.pair_counts(pts) == [1, 1, 1, 0, 1]
    with pytest.raises(ValueError):
        MP.pair_counts([2, 0, 1])                      # half a pair is not a measurement


def test_the_interval_is_over_PAIRS_not_games():
    """Perfect anti-correlation inside every pair (team luck decides each game) has ZERO pair variance:
    a per-game binomial interval would say ±0.1 at n=100 games; the pair interval says the score is
    exactly 0.5 — which is what mirroring buys."""
    counts = [0, 0, 50, 0, 0]
    mu, lo, hi = MP.pair_score_ci(counts)
    assert mu == pytest.approx(0.5) and lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    # and a spread pentanomial gets a non-zero width that shrinks like 1/sqrt(PAIRS)
    w1 = MP.pair_score_ci([10, 10, 10, 10, 10])
    w4 = MP.pair_score_ci([40, 40, 40, 40, 40])
    assert (w1[2] - w1[1]) == pytest.approx(2 * (w4[2] - w4[1]), rel=1e-9)


def test_the_bootstrap_is_deterministic_and_pair_resampling():
    c = [3, 5, 9, 6, 2]
    assert MP.pair_bootstrap_ci(c, seed=7) == MP.pair_bootstrap_ci(c, seed=7)
    mu, lo, hi = MP.pair_bootstrap_ci(c, seed=7)
    assert lo <= mu <= hi and mu == pytest.approx(MP.pair_score(c))


def test_a_draw_is_half_a_point_and_result_points_reads_the_trace_vocabulary():
    from agents.training.trace_result import DRAW, LOSS, WIN
    assert [MP.result_points(r) for r in (WIN, DRAW, LOSS)] == [2, 1, 0]
    assert MP.game_points(False, False) == 1                     # a |tie| (neither won nor lost)


# ---------------------------------------------------------------- the seed rule

def test_a_pairs_two_games_share_ONE_key_and_the_second_is_swapped():
    k0, s0 = SD.pair_game(11, "heuristic", 6, True)
    k1, s1 = SD.pair_game(11, "heuristic", 7, True)
    assert k0 == k1 == SD.game_key(11, "heuristic", 6)
    assert (s0, s1) == (False, True)
    assert SD.battle_seed(k0) == SD.battle_seed(k1)
    # the next pair is a different pairing
    assert SD.pair_game(11, "heuristic", 8, True)[0] != k0


def test_unmirrored_is_exactly_the_old_per_game_key():
    for g in range(6):
        assert SD.pair_game(5, "staller", g, False) == (SD.game_key(5, "staller", g), False)


# ---------------------------------------------------------------- the plan

def test_a_mirrored_split_keeps_every_pair_inside_one_unit():
    for n, shard in [(100, 25), (8, 3), (4, 1), (10, 0), (200, 200)]:
        sizes = _split_games(n, shard, paired=True)
        assert sum(sizes) == n and all(s % 2 == 0 and s > 0 for s in sizes), (n, shard, sizes)
    with pytest.raises(ValueError):
        _split_games(5, 2, paired=True)
    item = EvalItem("heuristic", BOT, 100)
    for u in plan_units([item], 25, paired=True):
        r = game_range(u, 25, paired=True)
        assert r.start % 2 == 0 and len(r) % 2 == 0


def test_the_plan_carries_the_regime_and_an_unmirrored_plan_is_byte_identical(tmp_path):
    items = [EvalItem("random", BOT, 4)]
    ShardedEvalPool(items, 2, step=1).write_plan(str(tmp_path / "a"))
    plain = json.loads((tmp_path / "a" / "plan.json").read_text())
    assert "mirrored" not in plain
    ShardedEvalPool(items, 2, step=1, mirrored=True).write_plan(str(tmp_path / "b"))
    back = ShardedEvalPool.from_plan(str(tmp_path / "b"))
    assert back.mirrored is True and all(len(back.game_range(u)) % 2 == 0 for u in back.units)


def test_shards_pool_their_pentanomials_exactly_and_unmirrored_stays_None(tmp_path):
    pool = ShardedEvalPool([EvalItem("random", BOT, 8), EvalItem("heuristic", BOT, 4)], 4, step=0,
                           mirrored=True)
    for u in pool.units:
        pc = [0, 1, 0, 0, 1] if u.item_key == "random" else None
        write_shard_result(str(tmp_path), ShardResult(u.unit_id, u.item_key, 0, 1, 2, 0.0, 2, 10.0, 1.0,
                                                      pair_counts=pc))
    merged = to_merged(aggregate(pool.units, str(tmp_path)))
    assert merged["pairs"] == {"random": [0, 2, 0, 0, 2]}        # heuristic: None never becomes zeros


# ---------------------------------------------------------------- both eval paths play the pairing

class _SeededBuilder:
    """``draw_team`` re-seeds ``_rng`` and calls ``yield_team`` — a stand-in pool of named teams."""

    def __init__(self, prefix):
        self.prefix = prefix
        self._rng = random.Random(0)

    def yield_team(self):
        return f"{self.prefix}{self._rng.randrange(10_000)}"


def test_the_python_worker_swaps_the_teams_of_the_second_game():
    from main.eval_worker import _per_game_teams

    pool = ShardedEvalPool([EvalItem("heuristic", BOT, 4)], 4, step=0, mirrored=True)
    unit = pool.units[0]
    ours, theirs = _per_game_teams(unit, pool, unit.item, 99, _SeededBuilder("T"), _SeededBuilder("O"))
    o = [ours.yield_team() for _ in range(4)]
    t = [theirs.yield_team() for _ in range(4)]
    assert o[0].startswith("T") and t[0].startswith("O")
    assert (o[1], t[1]) == (t[0], o[0])                          # pair 0, from the other side
    assert (o[3], t[3]) == (t[2], o[2])
    assert o[2] != o[0]                                          # the next pair is a new pairing


def test_the_rust_core_swaps_the_teams_and_keeps_the_seed():
    from agents.training.rust_eval.executor import EvalTable, RustEvalCore, _Unit

    core = RustEvalCore.__new__(RustEvalCore)
    core.team_table = SimpleNamespace(index=lambda team, label: team)
    core.trainee_builder, core.opp_builder = _SeededBuilder("T"), _SeededBuilder("O")
    core.fixed_builders = {}
    core.table = EvalTable(bots=("heuristic",), trainee_slot=0)
    unit = ShardedEvalPool([EvalItem("heuristic", BOT, 4)], 4, step=0, mirrored=True).units[0]
    u = _Unit(unit=unit, games=[0, 1, 2, 3], quota=None)
    g0, g1 = (core._make_game(u, g, 5, True, {}, mirrored=True) for g in (0, 1))
    assert g1.teams == (g0.teams[1], g0.teams[0]) and (g0.swapped, g1.swapped) == (False, True)
    assert g0.seed == g1.seed and g0.sample_seed == g1.sample_seed
    plain = core._make_game(u, 1, 5, True, {}, mirrored=False)
    assert plain.seed != g0.seed and not plain.swapped


# ---------------------------------------------------------------- the regime on disk, and the readers

def _row(step, mirrored):
    r = {"step": step, "n_games": 4, "bots": {"random": 0.75}, "sentinels": []}
    if mirrored:
        r["mirrored_pairs"] = {"schema": MP.SCHEMA, "bots": {"random": [0, 0, 1, 0, 1]}}
    return r


def test_the_elo_reader_REFUSES_a_run_whose_rows_cross_the_boundary(tmp_path):
    from agents.training import elo

    (tmp_path / "eval_results.jsonl").write_text(
        "\n".join(json.dumps(_row(s, s >= 300)) for s in (100, 200, 300, 400)) + "\n")
    with pytest.raises(elo.MixedEvalRegimeError, match="300"):
        elo.load_rows(str(tmp_path), "log")
    assert [r.step for r in elo.load_rows(str(tmp_path), "log", mirrored=True)] == [300, 400]
    assert [r.step for r in elo.load_rows(str(tmp_path), "log", mirrored=False)] == [100, 200]


def test_the_row_writer_stamps_the_regime_only_when_mirrored(tmp_path):
    from agents.model.snapshot import append_eval_result_row

    append_eval_result_row(str(tmp_path), 1, 4, {"random": 1.0})
    append_eval_result_row(str(tmp_path), 2, 4, {"random": 1.0},
                           mirrored_pairs={"bots": {"random": [0, 0, 0, 0, 2]}})
    rows = [json.loads(x) for x in (tmp_path / "eval_results.jsonl").read_text().splitlines()]
    assert "mirrored_pairs" not in rows[0]
    assert rows[1]["mirrored_pairs"]["bots"] == {"random": [0, 0, 0, 0, 2]}


def test_the_ladder_never_reads_a_mirrored_row_as_a_same_protocol_comparator(tmp_path):
    from agents.training.snapshot_ladder import eval_measured_pairs

    base = {"step": 200, "n_games": 4, "bots": {},
            "sentinel_regime": {"greedy": True, "symmetric_teams": True},
            "sentinels": [{"step": 100, "win_rate": 0.5, "counts": [2, 4]}]}
    (tmp_path / "eval_results.jsonl").write_text(
        json.dumps(base) + "\n" + json.dumps({**base, "step": 300, "mirrored_pairs": {"sentinels": {}}}) + "\n")
    assert eval_measured_pairs(str(tmp_path)) == {(100, 200): [2, 4]}


def test_record_pair_scores_records_the_pooled_pair_level_score():
    from agents.training.eval_record import mirrored_pairs_block, record_pair_scores

    rec = {}
    logger = SimpleNamespace(record=lambda k, v: rec.__setitem__(k, v))
    tui = {}
    merged = {"pairs": {"sentinel_0": [0, 0, 2, 0, 2], "sentinel_1": [0, 0, 0, 0, 4]}}
    blk = record_pair_scores(logger, tui, merged, ["sentinel_0", "sentinel_1"], "pool")
    assert blk["n_pairs"] == 8 and rec["eval/pairs_vs_pool"] == 8.0
    assert rec["eval/pair_score_vs_pool"] == pytest.approx((2 * 0.5 + 6 * 1.0) / 8)
    assert record_pair_scores(logger, tui, {"pairs": {}}, ["x"], "bots") is None
    mp = mirrored_pairs_block(merged, bots=[], sentinels=[("sentinel_0", 4_000_000)])
    assert mp["sentinels"] == {"4000000": [0, 0, 2, 0, 2]}


def test_the_manifest_states_the_pairing_regime(tmp_path):
    from agents.training.eval_launch import write_eval_manifest

    m = write_eval_manifest(str(tmp_path), 7, opponents=["random"], n_games=4, mirrored_pairs=True)
    assert m["mirrored_pairs"] is True
    assert write_eval_manifest(str(tmp_path), 8, opponents=["random"], n_games=4)["mirrored_pairs"] is False


def test_the_eval_count_is_even_by_construction():
    from agents.training.eval_launch import mirrored_eval_games

    assert [mirrored_eval_games(n) for n in (3, 4, 99, 100)] == [4, 4, 100, 100]


def test_the_untaught_meter_refuses_to_pool_the_two_regimes():
    from agents.training import untaught_meter as um

    a = {"R": {"T0": um.Cell(wins=1, finished=2, attempted=2, pairs=[0, 0, 1, 0, 0])}}
    b = {"R2": {"T0": um.Cell(wins=1, finished=2, attempted=2)}}
    with pytest.raises(um.MeterError, match="MIRRORED"):
        um.cells_regime({**a, **b})
    with pytest.raises(um.MeterError, match="MIRRORED"):
        um.merge_cells([{"R": {"T0": a["R"]["T0"].to_json()}}, {"R": {"T1": b["R2"]["T0"].to_json()}}])
    back = um.cell_from_json(a["R"]["T0"].to_json())
    assert back.pairs == [0, 0, 1, 0, 0] and back.mirrored
    res = um.aggregate(a, ["T0"], ref_labels=["R"], baseline_label=None, draws=50)
    assert res["mirrored_pairs"] is True and res["levels"]["R"]["pairs"]["n_pairs"] == 1


def test_the_exploiter_gap_treats_the_pairing_regime_as_part_of_the_regime():
    from agents.training.best_response_gap import _regime

    old = _regime({"cli_args": {}}, [], None)
    new = _regime({"cli_args": {"eval_mirrored_pairs": True}}, [], None)
    assert old["eval_mirrored_pairs"] is False and new["eval_mirrored_pairs"] is True
    assert old != new                       # check_matched compares the tuple: unmatched ⇒ refused


def test_the_PINNED_TEAM_meters_REFUSE_mirrored_pairs(tmp_path, monkeypatch, capsys):
    """P13 (owner 2026-10-02): the fixed-team meters measure the pilot ON its pinned team; mirroring
    swaps the pinned team inside the pair, so one game measures piloting and the other the response to
    it — pooled into one number. `main.untaught_meter --mirrored-pairs` and `main.best_response_gap
    --play --mirrored-pairs` REFUSE with a typed error naming why, at the CLI and at the library choke
    point (`untaught_meter.play_cells`), before anything is played. Fails on revert of any of them."""
    from agents.training import best_response_gap as brg
    from agents.training import untaught_meter as um
    from main import best_response_gap as brg_cli
    from main import untaught_meter as um_cli

    with pytest.raises(um.MirroredPinnedTeamError, match="PINNED-TEAM") as ei:
        um.play_cells([], [], None, games_per_team=4, mirrored=True)
    assert isinstance(ei.value, um.MeterError) and "eval_and_rating.md" in str(ei.value)

    target = tmp_path / "target.zip"
    target.write_text("x")
    run = brg.ExploiterRun(run="X", run_dir=str(tmp_path), target_run="T", target_file=str(target),
                           target_step=1, target_pins_own_teams=False, fork_step=0, num_timesteps=1,
                           budget=1, dose_rate=None, lr_median=None, teams=["a.txt"], archetype=None,
                           membership="", regime={}, series=[], lineage_derived=False)
    with pytest.raises(brg.MirroredPinnedTeamError, match="PILOTING") as ei2:
        brg.play_head_to_head(run, games=4, mirrored=True)
    assert isinstance(ei2.value, brg.BestResponseGapError) and ei2.value.cause == "mirrored_pinned_team"

    assert brg_cli.main([str(tmp_path / "no_such_run"), "--play", "4", "--mirrored-pairs"]) == 2
    assert "PINNED-TEAM" in capsys.readouterr().err
    assert um_cli.main([str(tmp_path / "no_such_ref"), "--mirrored-pairs"]) == 1
    assert "PINNED-TEAM" in capsys.readouterr().err