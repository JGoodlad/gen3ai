"""Unit tests for PFSP (prioritized fictitious self-play): the SnapshotPool hardness-weighted
sampling + spread retention, plus the SelfPlayCallback win-rate EMA/push logic.

PFSP oversamples the pool selves the trainee is LOSING to. Off (``pfsp_scale=0`` / no
``--pool-spread``) it must be byte-identical to the legacy recency-only sliding window — that
invariant is the spine of these tests."""

import random
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.training.snapshot_pool import SnapshotPool, SnapshotEntry
from agents.training.selfplay_callback import SelfPlayCallback, _PFSP_WR_EMA_BETA
from agents.training.wrappers import MaskableAgentWrapper


# ── helpers (mirrors snapshot_pool_test) ─────────────────────────────────────

def _make_pool(tmp_path, **kwargs) -> SnapshotPool:
    version = MagicMock()
    version.to_json.return_value = "{}"
    with patch("agents.training.snapshot_pool.load_model_snapshot"):
        return SnapshotPool(pool_dir=tmp_path, current_version=version, **kwargs)


def _fake_model(tmp_path) -> MagicMock:
    m = MagicMock()

    def _save(path):
        p = Path(path)
        if not str(p).endswith(".zip"):
            p = Path(str(p) + ".zip")
        p.touch()

    m.save.side_effect = _save
    return m


def _fill(pool, steps):
    model = _fake_model(pool.pool_dir)
    for s in steps:
        pool.add(model, step=s)


# ── OFF = byte-identical to the legacy recency-only window ────────────────────

def test_pfsp_off_weight_is_pure_recency(tmp_path):
    pool = _make_pool(tmp_path, recency_weight=0.3)            # pfsp_scale defaults to 0.0
    _fill(pool, [1_000_000, 2_000_000, 3_000_000])
    pool.set_win_rates({1_000_000: 0.0, 2_000_000: 0.5, 3_000_000: 1.0})  # ignored while off
    steps = [e.step for e in pool._entries]
    lo, span = steps[0], steps[-1] - steps[0]
    for e in pool._entries:
        expected = 1.0 + 0.3 * (e.step - lo) / span
        assert pool.entry_weight(e) == pytest.approx(expected)


def test_pfsp_off_sample_matches_recency_choices(tmp_path):
    """sample() with pfsp_scale=0 draws EXACTLY as random.choices over the legacy recency weights."""
    pool = _make_pool(tmp_path, recency_weight=0.3)
    _fill(pool, [1_000_000, 2_000_000, 3_000_000, 4_000_000])
    pool.set_win_rates({1_000_000: 0.0})  # would skew sampling if PFSP leaked while off
    steps = [e.step for e in pool._entries]
    lo, span = steps[0], steps[-1] - steps[0]
    recency = [1.0 + 0.3 * (e.step - lo) / span for e in pool._entries]

    random.seed(12345)
    got = [pool.sample().step for _ in range(200)]
    random.seed(12345)
    want = [c.step for c in (random.choices(pool._entries, weights=recency, k=1)[0] for _ in range(200))]
    assert got == want


# ── ON: hardness weighting ───────────────────────────────────────────────────

def test_pfsp_weight_formula(tmp_path):
    pool = _make_pool(tmp_path, recency_weight=0.3, pfsp_scale=2.0)
    _fill(pool, [1_000_000, 2_000_000, 3_000_000])
    pool.set_win_rates({1_000_000: 0.1, 2_000_000: 0.5, 3_000_000: 0.9})
    steps = [e.step for e in pool._entries]
    lo, span = steps[0], steps[-1] - steps[0]
    for e, p in zip(pool._entries, [0.1, 0.5, 0.9]):
        recency = 1.0 + 0.3 * (e.step - lo) / span
        assert pool.entry_weight(e) == pytest.approx(recency * (1.0 + 2.0 * (1.0 - p)))


def test_pfsp_oversamples_the_self_we_lose_to(tmp_path):
    # recency_weight=0 isolates the PFSP term: lowest win-rate must be sampled most.
    pool = _make_pool(tmp_path, recency_weight=0.0, pfsp_scale=2.0)
    _fill(pool, [1_000_000, 2_000_000, 3_000_000])
    pool.set_win_rates({1_000_000: 0.1, 2_000_000: 0.5, 3_000_000: 0.9})  # lose to 1M, beat 3M
    random.seed(7)
    counts = {1_000_000: 0, 2_000_000: 0, 3_000_000: 0}
    for _ in range(6000):
        counts[pool.sample().step] += 1
    assert counts[1_000_000] > counts[2_000_000] > counts[3_000_000]
    # never starved: the dominated self still gets sampled (coverage preserved).
    assert counts[3_000_000] > 0


def test_pfsp_unmeasured_uses_mean_of_known(tmp_path):
    pool = _make_pool(tmp_path, recency_weight=0.0, pfsp_scale=2.0)
    _fill(pool, [1_000_000, 2_000_000, 3_000_000])
    pool.set_win_rates({1_000_000: 0.2, 3_000_000: 0.8})   # 2M unmeasured → default = mean(0.2,0.8)=0.5
    mid = next(e for e in pool._entries if e.step == 2_000_000)
    assert pool.entry_weight(mid) == pytest.approx(1.0 + 2.0 * (1.0 - 0.5))


def test_pfsp_no_known_rates_is_pure_recency(tmp_path):
    # PFSP on but nothing measured yet (cold start) → every factor collapses to 1 ⇒ recency only.
    pool = _make_pool(tmp_path, recency_weight=0.3, pfsp_scale=2.0)
    _fill(pool, [1_000_000, 2_000_000, 3_000_000])
    steps = [e.step for e in pool._entries]
    lo, span = steps[0], steps[-1] - steps[0]
    for e in pool._entries:
        assert pool.entry_weight(e) == pytest.approx(1.0 + 0.3 * (e.step - lo) / span)


def test_set_win_rates_coerces_json_round_trip(tmp_path):
    # On resume the map arrives with string keys (JSON). They must be accepted transparently.
    pool = _make_pool(tmp_path, recency_weight=0.0, pfsp_scale=1.0)
    _fill(pool, [1_000_000])
    pool.set_win_rates({"1000000": "0.25"})
    e = pool._entries[0]
    assert pool.entry_weight(e) == pytest.approx(1.0 + 1.0 * (1.0 - 0.25))


def test_steps_accessor(tmp_path):
    pool = _make_pool(tmp_path)
    _fill(pool, [3_000_000, 1_000_000, 2_000_000])
    assert pool.steps() == [1_000_000, 2_000_000, 3_000_000]   # sorted ascending


# ── spread retention ─────────────────────────────────────────────────────────

def test_spread_keeps_newest_and_oldest(tmp_path):
    pool = _make_pool(tmp_path, max_snapshots=3, pool_spread=True)
    _fill(pool, [0, 1_000_000, 2_000_000, 3_000_000, 4_000_000])  # 5 added, cap 3
    steps = pool.steps()
    assert len(steps) == 3
    assert steps[0] == 0 and steps[-1] == 4_000_000   # oldest + newest always retained


def test_spread_thins_the_most_redundant_interior(tmp_path):
    # Steps 0,1,2,9 (×1e6). Adding 10 (cap 4) must drop the most-clustered interior point.
    pool = _make_pool(tmp_path, max_snapshots=4, pool_spread=True)
    _fill(pool, [0, 1_000_000, 2_000_000, 9_000_000])
    _fill(pool, [10_000_000])   # now 0,1,2,9,10 over cap 4
    steps = pool.steps()
    assert steps[0] == 0 and steps[-1] == 10_000_000
    # 1M and 2M are the cluster; one of them is the redundant victim (smallest neighbour gap).
    assert (1_000_000 in steps) ^ (2_000_000 in steps)
    assert 9_000_000 in steps   # the lone late anchor (wide gaps) is kept for diversity


def test_spread_off_is_legacy_sliding_window(tmp_path):
    pool = _make_pool(tmp_path, max_snapshots=3)   # pool_spread defaults False
    _fill(pool, [0, 1_000_000, 2_000_000, 3_000_000])
    assert pool.steps() == [1_000_000, 2_000_000, 3_000_000]   # oldest (the seed) evicted


def test_spread_evicted_file_unlinked(tmp_path):
    pool = _make_pool(tmp_path, max_snapshots=3, pool_spread=True)
    _fill(pool, [0, 1_000_000, 1_100_000, 5_000_000])   # 1M/1.1M cluster → one is dropped
    on_disk = {int(p.stem.split("_")[1]) for p in tmp_path.glob("snapshot_*.zip")}
    assert on_disk == set(pool.steps())   # disk and pool agree (the victim file is gone)


def test_spread_keeps_endpoints_and_pinned_interior(tmp_path):
    # Defensive (pinning is never set at runtime today): the +inf endpoint redundancy keeps the
    # newest/oldest anchors, and pinned entries are exempt — so when a thinnable interior exists
    # it's the victim, never an endpoint or a pinned interior.
    pool = _make_pool(tmp_path, max_snapshots=3, pool_spread=True)
    _fill(pool, [0, 1_000_000, 2_000_000])              # [0, 1M, 2M] at cap, no evict yet
    for e in pool._entries:
        if e.step == 1_000_000:
            e.pinned = True                             # pin the 1M interior
    _fill(pool, [3_000_000])                            # [0, 1M(pin), 2M, 3M] over cap 3
    steps = pool.steps()
    assert 0 in steps and 3_000_000 in steps            # endpoints preserved (+inf)
    assert 1_000_000 in steps                           # pinned interior exempt
    assert 2_000_000 not in steps                       # the only thinnable interior is the victim
    assert len(steps) == 3


# ── SelfPlayCallback EMA + push (unbound-method style, no heavy construction) ──

def _sentinel(step, win):
    return (SnapshotEntry(path=Path(f"snapshot_{step:012d}.zip"), step=step), f"sentinel_{step}",
            win, 0.0, 0.0)


def _fake_cb(scale, ema=None):
    return types.SimpleNamespace(
        _pfsp_scale=scale,
        _pfsp_winrate_ema=dict(ema or {}),
        _pool=MagicMock(),
        logger=MagicMock(),
    )


def test_update_pfsp_ema_off_is_noop():
    cb = _fake_cb(0.0)
    SelfPlayCallback._update_pfsp_ema(cb, [_sentinel(1_000_000, 0.2)], {})
    assert cb._pfsp_winrate_ema == {}
    cb.logger.record.assert_not_called()
    cb._pool.set_win_rates.assert_not_called()        # off → trainer pool untouched


def test_update_pfsp_ema_refreshes_trainer_pool_same_cycle():
    # The metadata sentinel "weight" reads pool.entry_weight, so the trainer pool must see this
    # cycle's measurement BEFORE the block is built — not lag a cycle.
    cb = _fake_cb(2.0)
    SelfPlayCallback._update_pfsp_ema(cb, [_sentinel(1_000_000, 0.3)], {})
    cb._pool.set_win_rates.assert_called_once_with({1_000_000: 0.3})


def test_update_pfsp_ema_first_sight_seeds_then_blends():
    cb = _fake_cb(2.0)
    SelfPlayCallback._update_pfsp_ema(cb, [_sentinel(1_000_000, 0.20)], {})
    assert cb._pfsp_winrate_ema[1_000_000] == pytest.approx(0.20)        # first sight = measured
    SelfPlayCallback._update_pfsp_ema(cb, [_sentinel(1_000_000, 0.60)], {})
    blended = _PFSP_WR_EMA_BETA * 0.20 + (1 - _PFSP_WR_EMA_BETA) * 0.60  # EMA, not overwrite
    assert cb._pfsp_winrate_ema[1_000_000] == pytest.approx(blended)


def test_update_pfsp_ema_records_hardest():
    cb = _fake_cb(2.0)
    tui = {}
    SelfPlayCallback._update_pfsp_ema(
        cb, [_sentinel(1_000_000, 0.8), _sentinel(2_000_000, 0.3)], tui)
    assert tui["eval/pfsp_hardest_win_rate"] == pytest.approx(0.3)       # the self we lose to most
    assert tui["eval/pfsp_tracked_snapshots"] == pytest.approx(2.0)


def _fake_push_cb(scale, ema, live_steps):
    pool = MagicMock()
    pool.steps.return_value = list(live_steps)
    return types.SimpleNamespace(
        _pfsp_scale=scale, _pfsp_winrate_ema=dict(ema), _pool=pool, training_env=MagicMock())


def test_prune_and_push_off_makes_no_ipc():
    cb = _fake_push_cb(0.0, {1_000_000: 0.3}, [1_000_000])
    SelfPlayCallback._prune_and_push_pfsp(cb)
    cb.training_env.env_method.assert_not_called()
    cb._pool.set_win_rates.assert_not_called()


def test_prune_and_push_drops_evicted_and_pushes():
    # 2M slid out of the pool → its stale EMA entry is pruned before the push.
    cb = _fake_push_cb(2.0, {1_000_000: 0.3, 2_000_000: 0.9}, [1_000_000, 3_000_000])
    SelfPlayCallback._prune_and_push_pfsp(cb)
    assert cb._pfsp_winrate_ema == {1_000_000: 0.3}
    cb._pool.set_win_rates.assert_called_once_with({1_000_000: 0.3})
    cb.training_env.env_method.assert_called_once_with("set_opponent_win_rates", {1_000_000: 0.3})


def test_prune_and_push_survives_env_method_failure():
    cb = _fake_push_cb(2.0, {1_000_000: 0.3}, [1_000_000])
    cb.training_env.env_method.side_effect = RuntimeError("worker gone")
    SelfPlayCallback._prune_and_push_pfsp(cb)   # non-fatal — must not raise


# ── wrapper forwarding ────────────────────────────────────────────────────────

def test_wrapper_forwards_win_rates_to_pool():
    pool = MagicMock()
    w = MaskableAgentWrapper.__new__(MaskableAgentWrapper)   # skip SingleAgentWrapper.__init__
    w._pool = pool
    w.set_opponent_win_rates({1_000_000: 0.4})
    pool.set_win_rates.assert_called_once_with({1_000_000: 0.4})


def test_wrapper_win_rates_noop_without_pool():
    w = MaskableAgentWrapper.__new__(MaskableAgentWrapper)
    w._pool = None
    w.set_opponent_win_rates({1_000_000: 0.4})   # heuristic-only env → no pool, must not raise
