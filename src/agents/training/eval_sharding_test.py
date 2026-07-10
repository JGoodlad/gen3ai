"""Unit tests for battle-level work-stealing (``agents.training.eval_sharding``).

The load-bearing test is aggregation EXACTNESS: pooling an opponent's shards must reproduce the
unsharded win_rate / reward / ep_len / td_resid_tail / counts bit-for-bit. Plus the partition
invariants (Σshards == n_games), the plan round-trip, claim-exactly-once, and partial coverage.
"""
import random

import pytest

from agents.training.eval_sharding import (
    EvalItem, ShardResult, ShardedEvalPool, BOT, SENTINEL, FIXED, td_tail,
)
from agents.training.eval_sharding.units import _split_games, plan_units, shards_per_item


# ── partition ─────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [1, 7, 8, 25, 99, 100, 250])
@pytest.mark.parametrize("sz", [0, 1, 10, 25, 30, 1000])
def test_split_sums_to_n_and_is_even(n, sz):
    parts = _split_games(n, sz)
    assert sum(parts) == n                         # Σshards == n_games (exactness invariant)
    assert all(p >= 1 for p in parts)
    assert max(parts) - min(parts) <= 1            # as even as possible
    if sz <= 0 or sz >= n:
        assert parts == [n]                        # disabled / coarse → single shard
    else:
        assert all(p <= sz for p in parts)


def test_plan_units_count_and_round_robin_lpt_order():
    items = [EvalItem("random", BOT, 100), EvalItem("staller", BOT, 100),
             EvalItem("sentinel_0", SENTINEL, 100, path="/s.zip", step=5),
             EvalItem("heuristic", BOT, 100)]
    pool = ShardedEvalPool(items, shard_games=25)
    assert pool.n_units == 16                       # 4 items x 4 shards
    # round 0 = shard-0 of each item in cost-descending order; sentinel(2.0) leads, random(0.5) trails
    r0 = [u.item_key for u in pool.units[:4]]
    assert r0[0] == "sentinel_0" and r0[-1] == "random"
    # every unit_id is unique and shard indices cover 0..3 per item
    assert len({u.unit_id for u in pool.units}) == 16
    for key in ("random", "staller", "sentinel_0", "heuristic"):
        idxs = sorted(u.shard_index for u in pool.units if u.item_key == key)
        assert idxs == [0, 1, 2, 3]


def test_plan_units_is_pure_deterministic():
    items = [EvalItem("a", BOT, 50), EvalItem("b", BOT, 100)]
    assert ([u.unit_id for u in plan_units(items, 25)]
            == [u.unit_id for u in plan_units(items, 25)])


def test_ragged_shard_counts_round_robin():
    # b splits into 4 shards, a into 2 — the round-robin must not index past a's shards.
    items = [EvalItem("a", BOT, 50), EvalItem("b", BOT, 100)]
    pool = ShardedEvalPool(items, shard_games=25)
    assert pool.n_units == 6
    assert shards_per_item(items, 25) == {"a": 2, "b": 4}


def test_shard_games_ge_n_games_is_one_shard_per_item():
    items = [EvalItem("a", BOT, 100), EvalItem("b", BOT, 100)]
    pool = ShardedEvalPool(items, shard_games=100)
    assert pool.n_units == 2
    assert all(u.shard_index == 0 and u.n_games == 100 for u in pool.units)


def test_evalitem_validation():
    with pytest.raises(ValueError):
        EvalItem("x", "bogus", 10)
    with pytest.raises(ValueError):
        EvalItem("x", BOT, 0)
    with pytest.raises(ValueError):
        EvalItem("x", SENTINEL, 10)  # sentinel needs a path


# ── plan manifest round-trip ────────────────────────────────────────────────────

def test_plan_round_trip_identical_units(tmp_path):
    items = [EvalItem("random", BOT, 100),
             EvalItem("sentinel_0", SENTINEL, 100, path="/s.zip", step=7),
             EvalItem("ext_run", FIXED, 100, path="/e.zip", config_path="/c.json")]
    pool = ShardedEvalPool(items, shard_games=30, step=42)
    pool.write_plan(str(tmp_path))
    back = ShardedEvalPool.from_plan(str(tmp_path))
    assert [u.unit_id for u in back.units] == [u.unit_id for u in pool.units]
    assert back.shard_games == 30 and back.step == 42
    s = next(i for i in back.items if i.kind == SENTINEL)
    assert s.path == "/s.zip" and s.step == 7
    e = next(i for i in back.items if i.kind == FIXED)
    assert e.config_path == "/c.json"


# ── claim exactly-once ──────────────────────────────────────────────────────────

def test_claim_each_unit_exactly_once_across_workers(tmp_path):
    items = [EvalItem("a", BOT, 100), EvalItem("b", BOT, 100), EvalItem("c", BOT, 60)]
    p = ShardedEvalPool(items, shard_games=25)
    p.write_plan(str(tmp_path))
    cd = str(tmp_path / "claims")
    import os
    os.makedirs(cd, exist_ok=True)
    # three independent "workers" interleave their claims
    workers = [ShardedEvalPool.from_plan(str(tmp_path)) for _ in range(3)]
    claimed = []
    while True:
        got = [w.claim_next(cd) for w in workers]
        live = [u.unit_id for u in got if u is not None]
        claimed += live
        if not live:
            break
    assert sorted(claimed) == sorted(u.unit_id for u in p.units)
    assert len(claimed) == len(set(claimed))             # none twice
    assert all(w.claim_next(cd) is None for w in workers)  # pool drained


# ── AGGREGATION EXACTNESS (the load-bearing property) ───────────────────────────

def _publish_split(pool, result_dir, per_battle, *, skip_units=()):
    """Distribute a list of per-battle (won, reward, turns, [deltas]) records across the pool's
    planned shards by their sizes and publish each as a ShardResult."""
    idx = 0
    for wi, unit in enumerate(pool.units):
        sz = unit.n_games
        chunk = per_battle[idx:idx + sz]
        idx += sz
        if unit.unit_id in skip_units:
            continue
        pool.publish(result_dir, ShardResult(
            unit_id=unit.unit_id, item_key=unit.item_key, worker_id=wi,
            n_won=sum(b[0] for b in chunk), n_finished=len(chunk),
            sum_reward=sum(b[1] for b in chunk), n_episodes=len(chunk),
            sum_ep_len=sum(b[2] for b in chunk), duration_sec=1.0,
            td_residuals=[d for b in chunk for d in b[3]]))


def test_sharded_aggregation_equals_unsharded(tmp_path):
    rng = random.Random(0)
    for trial in range(150):
        n = rng.randint(20, 130)
        per_battle = []
        deltas_all = []
        for _ in range(n):
            won = 1 if rng.random() < 0.42 else 0
            reward = rng.gauss(0, 5)
            turns = rng.randint(5, 90)
            ds = [rng.gauss(-1, 3) for _ in range(rng.randint(0, 4))]
            per_battle.append((won, reward, turns, ds))
            deltas_all += ds
        # unsharded truth
        u_wr = sum(b[0] for b in per_battle) / n
        u_rew = sum(b[1] for b in per_battle) / n
        u_len = sum(b[2] for b in per_battle) / n
        u_td = td_tail(deltas_all)
        # sharded
        item = EvalItem("x", BOT, n)
        pool = ShardedEvalPool([item], shard_games=max(1, n // rng.randint(1, 6)))
        rd = tmp_path / f"t{trial}"
        rd.mkdir()
        _publish_split(pool, str(rd), per_battle)
        merged, missing = pool.collect(str(rd))
        assert not missing
        assert merged["win_rates"]["x"] == pytest.approx(u_wr, abs=1e-12)
        assert merged["reward_means"]["x"] == pytest.approx(u_rew, abs=1e-9)
        assert merged["ep_lens"]["x"] == pytest.approx(u_len, abs=1e-9)
        assert merged["counts"]["x"] == (sum(b[0] for b in per_battle), n)
        assert merged["coverage"]["x"] == 1.0
        if u_td is None:
            assert "x" not in merged["td_resid_tails"]
        else:
            assert merged["td_resid_tails"]["x"] == pytest.approx(u_td, abs=1e-12)


def test_sharded_aggregation_diverging_denominators(tmp_path):
    """reward pools over n_episodes (reward-tracked battles) while ep_len/win_rate pool over
    n_finished — they can differ (a battle finishes without a finalized reward). Pooling must STILL
    equal the unsharded computation, because each numerator is summed against its own denominator."""
    rng = random.Random(7)
    for trial in range(60):
        n = rng.randint(20, 80)
        # each battle: (won, turns, has_reward, reward)
        bs = [(1 if rng.random() < 0.4 else 0, rng.randint(5, 60),
               rng.random() < 0.85, rng.gauss(0, 4)) for _ in range(n)]
        n_fin = n
        n_rew = sum(1 for b in bs if b[2])
        sum_rew = sum(b[3] for b in bs if b[2])
        sum_turns = sum(b[1] for b in bs)
        n_won = sum(b[0] for b in bs)
        # unsharded targets (the two distinct denominators)
        u_wr = n_won / n_fin
        u_rew = (sum_rew / n_rew) if n_rew else 0.0          # mean_episode_reward denom
        u_len = sum_turns / n_fin                             # ep_len denom = n_finished
        # shard it; each shard reports n_finished=chunk size, n_episodes=reward count in chunk
        pool = ShardedEvalPool([EvalItem("x", BOT, n)], shard_games=max(1, n // rng.randint(1, 5)))
        rd = tmp_path / f"d{trial}"; rd.mkdir()
        idx = 0
        for wi, unit in enumerate(pool.units):
            chunk = bs[idx:idx + unit.n_games]; idx += unit.n_games
            pool.publish(str(rd), ShardResult(
                unit.unit_id, "x", wi,
                n_won=sum(b[0] for b in chunk), n_finished=len(chunk),
                sum_reward=sum(b[3] for b in chunk if b[2]),
                n_episodes=sum(1 for b in chunk if b[2]),
                sum_ep_len=sum(b[1] for b in chunk), duration_sec=1.0, td_residuals=[]))
        merged, _ = pool.collect(str(rd))
        assert merged["win_rates"]["x"] == pytest.approx(u_wr, abs=1e-12)
        assert merged["reward_means"]["x"] == pytest.approx(u_rew, abs=1e-9)
        assert merged["ep_lens"]["x"] == pytest.approx(u_len, abs=1e-9)


# ── partial coverage / missing ──────────────────────────────────────────────────

def test_partial_coverage_measures_over_actual_games(tmp_path):
    item = EvalItem("x", BOT, 100)
    pool = ShardedEvalPool([item], shard_games=25)  # 4 shards of 25
    per_battle = [(1 if i % 2 == 0 else 0, 0.0, 10, []) for i in range(100)]
    # drop the last shard (a crashed worker)
    skip = pool.units[-1].unit_id
    _publish_split(pool, str(tmp_path), per_battle, skip_units=(skip,))
    merged, missing = pool.collect(str(tmp_path))
    assert missing == []                                  # some shards present → not fully missing
    assert merged["coverage"]["x"] == pytest.approx(3 / 4)
    # win_rate is over the 75 games that ran (50% alternating), not the nominal 100
    assert merged["counts"]["x"][1] == 75


def test_fully_missing_opponent_in_missing_list(tmp_path):
    items = [EvalItem("a", BOT, 50), EvalItem("b", BOT, 50)]
    pool = ShardedEvalPool(items, shard_games=25)
    # publish only a's shards
    for wi, unit in enumerate(pool.units):
        if unit.item_key != "a":
            continue
        pool.publish(str(tmp_path), ShardResult(
            unit.unit_id, "a", wi, n_won=10, n_finished=unit.n_games, sum_reward=0.0,
            n_episodes=unit.n_games, sum_ep_len=10.0 * unit.n_games, duration_sec=1.0,
            td_residuals=[]))
    merged, missing = pool.collect(str(tmp_path))
    assert missing == ["b"]
    assert "a" in merged["win_rates"] and "b" not in merged["win_rates"]


def test_eval_item_team_str_round_trips_through_plan():
    # the fold-back pin must survive plan.json (parent writes, worker reads)
    from agents.training.eval_sharding.units import EvalItem, FIXED
    it = EvalItem("ext_spec", FIXED, 10, path="/m/x.zip", config_path="/m/c.json",
                  team_str="Skarmory @ Leftovers\n")
    rt = EvalItem.from_dict(it.to_dict())
    assert rt.team_str == it.team_str and rt == it
    bare = EvalItem("ext_gen", FIXED, 10, path="/m/y.zip")
    assert "team_str" not in bare.to_dict() and EvalItem.from_dict(bare.to_dict()).team_str is None
