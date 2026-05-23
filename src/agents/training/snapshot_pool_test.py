"""Unit tests for SnapshotPool and heuristic_fraction."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from agents.training.snapshot_pool import SnapshotPool, SnapshotEntry, heuristic_fraction


# ── heuristic_fraction ──────────────────────────────────────────────────────

def test_heuristic_fraction_at_floor():
    assert heuristic_fraction(0.0) == pytest.approx(0.80)


def test_heuristic_fraction_below_ramp():
    # Win rate at or below ramp start (0.50) returns ceiling (0.80)
    assert heuristic_fraction(0.50) == pytest.approx(0.80)


def test_heuristic_fraction_at_ceiling():
    # Win rate at or above ramp end (0.85) returns floor (0.10)
    assert heuristic_fraction(0.85) == pytest.approx(0.10)
    assert heuristic_fraction(1.0) == pytest.approx(0.10)


def test_heuristic_fraction_midpoint():
    # Smoothstep midpoint at t=0.5 → 3(0.5²) - 2(0.5³) = 0.5, so 0.80*0.5 + 0.10*0.5 = 0.45
    mid_wr = 0.50 + (0.85 - 0.50) * 0.5
    assert heuristic_fraction(mid_wr) == pytest.approx(0.45)


def test_heuristic_fraction_is_monotone_decreasing():
    prev = 1.0
    for wr in [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.85, 1.0]:
        val = heuristic_fraction(wr)
        assert val <= prev + 1e-9
        prev = val


def test_heuristic_fraction_stays_in_bounds():
    for wr in [x / 100 for x in range(0, 101)]:
        val = heuristic_fraction(wr)
        assert 0.10 <= val <= 0.80


# ── SnapshotPool helpers ────────────────────────────────────────────────────

def _make_pool(tmp_path, **kwargs) -> SnapshotPool:
    """Create a SnapshotPool with a mock ModelVersion (no real model loading)."""
    version = MagicMock()
    with patch("agents.training.snapshot_pool.load_model_snapshot"):
        pool = SnapshotPool(pool_dir=tmp_path, current_version=version, **kwargs)
    return pool


def _fake_model(tmp_path, name="model") -> MagicMock:
    """Mock model whose .save() writes a real zip so glob finds it."""
    m = MagicMock()

    def _save(path):
        p = Path(path) if not str(path).endswith(".zip") else Path(path)
        if not str(p).endswith(".zip"):
            p = Path(str(p) + ".zip")
        p.touch()

    m.save.side_effect = _save
    return m


# ── seed ────────────────────────────────────────────────────────────────────

def test_seed_creates_step0_entry(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    entry = pool.seed(model)
    assert entry.step == 0
    assert entry.pinned is True
    assert len(pool) == 1


def test_seed_is_idempotent(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    e1 = pool.seed(model)
    e2 = pool.seed(model)
    assert e1.path == e2.path
    assert len(pool) == 1
    assert model.save.call_count == 1  # written only once


def test_seed_skips_write_if_file_exists(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    # Second pool instance (simulates restart) — seed should not re-write
    pool2 = _make_pool(tmp_path)
    model2 = _fake_model(tmp_path)
    pool2.seed(model2)
    assert model2.save.call_count == 0


# ── add / evict ─────────────────────────────────────────────────────────────

def test_add_creates_entry(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    pool.add(model, step=1_000_000)
    assert len(pool) == 2


def test_evict_removes_oldest_unpinned(tmp_path):
    pool = _make_pool(tmp_path, max_snapshots=3)
    model = _fake_model(tmp_path)
    pool.seed(model)                    # step 0 (pinned)
    pool.add(model, step=1_000_000)     # step 1M
    pool.add(model, step=2_000_000)     # step 2M
    # Pool is at max=3 now. Adding one more should evict step 1M (oldest unpinned).
    pool.add(model, step=3_000_000)
    steps = [e.step for e in pool._entries]
    assert 1_000_000 not in steps
    assert 0 in steps          # seed still present
    assert 2_000_000 in steps
    assert 3_000_000 in steps


def test_evict_never_removes_pinned(tmp_path):
    pool = _make_pool(tmp_path, max_snapshots=2)
    model = _fake_model(tmp_path)
    pool.seed(model)                # step 0 (pinned)
    pool.add(model, step=1_000_000)
    pool.add(model, step=2_000_000)  # triggers eviction of step 1M
    assert any(e.step == 0 for e in pool._entries)


def test_add_replaces_same_step(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.add(model, step=500_000)
    pool.add(model, step=500_000)
    assert len(pool) == 1
    assert pool._entries[0].step == 500_000


# ── _scan (directory reconstruction) ────────────────────────────────────────

def test_scan_reconstructs_from_disk(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    pool.add(model, step=1_000_000)
    # New pool instance reads same directory
    pool2 = _make_pool(tmp_path)
    assert len(pool2) == 2
    steps = [e.step for e in pool2._entries]
    assert 0 in steps
    assert 1_000_000 in steps


def test_scan_pins_step_zero(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    pool.add(model, step=1_000_000)
    pool2 = _make_pool(tmp_path)
    seed = next(e for e in pool2._entries if e.step == 0)
    assert seed.pinned is True
    non_seed = next(e for e in pool2._entries if e.step != 0)
    assert non_seed.pinned is False


def test_scan_ignores_malformed_filenames(tmp_path):
    # Write a file that looks like a snapshot but has a non-integer step
    (tmp_path / "snapshot_badname.zip").touch()
    pool = _make_pool(tmp_path)
    assert len(pool) == 0


def test_scan_entries_sorted_by_step(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.add(model, step=3_000_000)
    pool.add(model, step=1_000_000)
    pool.add(model, step=2_000_000)
    steps = [e.step for e in pool._entries]
    assert steps == sorted(steps)


# ── sample ──────────────────────────────────────────────────────────────────

def test_sample_raises_on_empty_pool(tmp_path):
    pool = _make_pool(tmp_path)
    with pytest.raises(RuntimeError):
        pool.sample()


def test_sample_returns_single_entry(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    entry = pool.sample()
    assert isinstance(entry, SnapshotEntry)


def test_sample_uniform_with_zero_recency_weight(tmp_path):
    pool = _make_pool(tmp_path, recency_weight=0.0)
    model = _fake_model(tmp_path)
    pool.seed(model)
    for s in range(1, 6):
        pool.add(model, step=s * 1_000_000)
    counts = {e.step: 0 for e in pool._entries}
    for _ in range(2000):
        counts[pool.sample().step] += 1
    # With uniform weights all steps should appear. No step should be zero.
    assert all(c > 0 for c in counts.values())


def test_sample_single_entry_always_returns_it(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    for _ in range(20):
        assert pool.sample().step == 0


# ── sentinel_entries ─────────────────────────────────────────────────────────

def test_sentinel_entries_returns_all_when_pool_small(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    pool.add(model, step=1_000_000)
    sentinels = pool.sentinel_entries(n=5)
    assert len(sentinels) == 2


def test_sentinel_entries_newest_first(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    for s in range(5):
        pool.add(model, step=s * 1_000_000)
    sentinels = pool.sentinel_entries(n=5)
    steps = [e.step for e in sentinels]
    assert steps == sorted(steps, reverse=True)


def test_sentinel_entries_exact_n_when_pool_large(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    for s in range(10):
        pool.add(model, step=s * 1_000_000)
    sentinels = pool.sentinel_entries(n=5)
    assert len(sentinels) == 5


# ── win-rate persistence ─────────────────────────────────────────────────────

def test_persist_and_load_win_rate(tmp_path):
    pool = _make_pool(tmp_path)
    pool.persist_win_rate(0.734)
    assert pool.load_persisted_win_rate() == pytest.approx(0.734, abs=1e-5)


def test_load_win_rate_returns_zero_if_missing(tmp_path):
    pool = _make_pool(tmp_path)
    assert pool.load_persisted_win_rate() == pytest.approx(0.0)


def test_load_win_rate_returns_zero_on_corrupt_file(tmp_path):
    pool = _make_pool(tmp_path)
    (tmp_path / pool._WIN_RATE_FILE).write_text("not_a_number\n")
    assert pool.load_persisted_win_rate() == pytest.approx(0.0)


# ── bot-peak persistence ─────────────────────────────────────────────────────

def test_persist_and_load_bot_peaks(tmp_path):
    pool = _make_pool(tmp_path)
    peaks = {"Heuristic": 0.75, "Staller": 0.82}
    pool.persist_bot_peaks(peaks)
    loaded = pool.load_persisted_bot_peaks()
    assert loaded == pytest.approx(peaks)


def test_load_bot_peaks_returns_empty_if_missing(tmp_path):
    pool = _make_pool(tmp_path)
    assert pool.load_persisted_bot_peaks() == {}


def test_load_bot_peaks_returns_empty_on_corrupt_file(tmp_path):
    pool = _make_pool(tmp_path)
    (tmp_path / pool._BOT_PEAKS_FILE).write_text("{invalid json")
    assert pool.load_persisted_bot_peaks() == {}
