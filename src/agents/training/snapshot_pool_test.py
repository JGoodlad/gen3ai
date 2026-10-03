"""Unit tests for SnapshotPool and heuristic_fraction."""

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest

from agents.training.snapshot_pool import SnapshotPool, SnapshotEntry, heuristic_fraction


# ── heuristic_fraction ──────────────────────────────────────────────────────

def test_heuristic_fraction_no_self_play_below_threshold():
    # Below SELF_PLAY_START (0.55) → 100% heuristics (0% self-play): don't burn cycles
    # on a weak self-opponent before the model can reliably beat the bots.
    assert heuristic_fraction(0.0) == pytest.approx(1.0)
    assert heuristic_fraction(0.40) == pytest.approx(1.0)
    assert heuristic_fraction(0.55) == pytest.approx(1.0)


def test_heuristic_fraction_at_ceiling():
    # At/above SELF_PLAY_FULL (0.80) → HEURISTIC_FLOOR 0.10 (90% self-play).
    assert heuristic_fraction(0.80) == pytest.approx(0.10)
    assert heuristic_fraction(0.85) == pytest.approx(0.10)
    assert heuristic_fraction(1.0) == pytest.approx(0.10)


def test_heuristic_fraction_midpoint():
    # Smoothstep midpoint (t=0.5 → 0.5): 1.0*0.5 + 0.10*0.5 = 0.55.
    mid_wr = 0.55 + (0.80 - 0.55) * 0.5
    assert heuristic_fraction(mid_wr) == pytest.approx(0.55)


def test_heuristic_fraction_ramps_between_threshold_and_full():
    # Strictly between START and FULL it's strictly inside (1.0, 0.10).
    v = heuristic_fraction(0.65)
    assert 0.10 < v < 1.0


def test_heuristic_fraction_is_monotone_decreasing():
    prev = 1.01
    for wr in [0.0, 0.3, 0.55, 0.6, 0.7, 0.75, 0.80, 0.85, 1.0]:
        val = heuristic_fraction(wr)
        assert val <= prev + 1e-9
        prev = val


def test_heuristic_fraction_stays_in_bounds():
    for wr in [x / 100 for x in range(0, 101)]:
        val = heuristic_fraction(wr)
        assert 0.10 <= val <= 1.0


# ── configurable transition + floor (#2) ─────────────────────────────────────

def test_heuristic_fraction_custom_floor():
    # Raised floor → more bot episodes forever once saturated; default unchanged.
    assert heuristic_fraction(1.0, floor=0.25) == pytest.approx(0.25)
    assert heuristic_fraction(0.90, floor=0.25) == pytest.approx(0.25)
    assert heuristic_fraction(1.0) == pytest.approx(0.10)   # default untouched


def test_heuristic_fraction_custom_start_full_shift_ramp():
    # A later `full` means the ramp hasn't bottomed out yet at the old ceiling (0.80):
    # the curve stays bot-heavier for longer.
    later = heuristic_fraction(0.80, start=0.60, full=0.95, floor=0.25)
    assert later > 0.25                       # not yet saturated to the floor
    assert heuristic_fraction(0.95, start=0.60, full=0.95, floor=0.25) == pytest.approx(0.25)
    assert heuristic_fraction(0.55, start=0.60, full=0.95) == pytest.approx(1.0)  # below start


def test_heuristic_fraction_defaults_match_constants():
    # Passing the module constants explicitly reproduces the no-arg curve exactly.
    from agents.training.snapshot_pool import HEURISTIC_FLOOR, SELF_PLAY_START, SELF_PLAY_FULL
    for wr in [0.0, 0.5, 0.55, 0.675, 0.8, 1.0]:
        assert heuristic_fraction(wr) == pytest.approx(
            heuristic_fraction(wr, floor=HEURISTIC_FLOOR, start=SELF_PLAY_START, full=SELF_PLAY_FULL))


def test_heuristic_fraction_degenerate_start_equals_full():
    # start==full must not divide by zero; it becomes a step at the threshold.
    assert heuristic_fraction(0.50, start=0.70, full=0.70, floor=0.10) == pytest.approx(1.0)
    assert heuristic_fraction(0.80, start=0.70, full=0.70, floor=0.10) == pytest.approx(0.10)


# ── SnapshotPool helpers ────────────────────────────────────────────────────

def _make_pool(tmp_path, **kwargs) -> SnapshotPool:
    """Create a SnapshotPool with a mock ModelVersion (no real model loading)."""
    version = MagicMock()
    # _write() now drops a shared model_config.json via version.to_json(); give the
    # mock a real JSON string so write_text() gets a str, not a MagicMock.
    version.to_json.return_value = "{}"
    with patch("agents.training.snapshot_pool.load_opponent_snapshot"):
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

def test_seed_creates_unpinned_step0_entry(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    entry = pool.seed(model)
    assert entry.step == 0
    assert entry.pinned is False   # sliding window — the seed ages out, never pinned
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


def test_evict_removes_oldest_sliding_window(tmp_path):
    pool = _make_pool(tmp_path, max_snapshots=3)
    model = _fake_model(tmp_path)
    pool.seed(model)                    # step 0 (the seed — NOT pinned)
    pool.add(model, step=1_000_000)
    pool.add(model, step=2_000_000)
    # Pool is at max=3. Adding one more evicts the OLDEST (the step-0 seed) — sliding window.
    pool.add(model, step=3_000_000)
    steps = [e.step for e in pool._entries]
    assert 0 not in steps          # the old seed ages out
    assert steps == [1_000_000, 2_000_000, 3_000_000]


def test_seed_ages_out_under_eviction(tmp_path):
    # Nothing is pinned, so a low-cap pool drops the seed once newer snapshots arrive.
    pool = _make_pool(tmp_path, max_snapshots=2)
    model = _fake_model(tmp_path)
    pool.seed(model)                # step 0 (the seed)
    pool.add(model, step=1_000_000)
    pool.add(model, step=2_000_000)  # over cap → evict the oldest (the seed)
    assert all(e.step != 0 for e in pool._entries)


def test_add_replaces_same_step(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.add(model, step=500_000)
    pool.add(model, step=500_000)
    assert len(pool) == 1
    assert pool._entries[0].step == 500_000


# ── add_from_path (promote a frozen snapshot file, not a live model) ─────────

def _frozen_zip(tmp_path, name="frozen.zip", content=b"frozen-weights") -> Path:
    p = tmp_path / name
    p.write_bytes(content)
    return p


def test_add_from_path_copies_and_registers(tmp_path):
    pool = _make_pool(tmp_path)
    src = _frozen_zip(tmp_path)
    entry = pool.add_from_path(src, step=1_000_000)
    dst = tmp_path / "snapshot_000001000000.zip"
    assert dst.exists()
    assert dst.read_bytes() == b"frozen-weights"   # bit-exact copy of the frozen file
    assert entry.step == 1_000_000 and entry.pinned is False
    assert len(pool) == 1
    # Shared arch tag dropped so load_model_snapshot does a REAL compatibility check.
    assert (tmp_path / "model_config.json").exists()


def test_add_from_path_does_not_mutate_source(tmp_path):
    pool = _make_pool(tmp_path)
    src = _frozen_zip(tmp_path)
    pool.add_from_path(src, step=2_000_000)
    assert src.exists()  # copy, not move — the trainer cleans up the scratch itself


def test_add_from_path_evicts_oldest_sliding_window(tmp_path):
    pool = _make_pool(tmp_path, max_snapshots=3)
    model = _fake_model(tmp_path)
    pool.seed(model)  # step 0 (the seed — not pinned)
    pool.add_from_path(_frozen_zip(tmp_path, "a.zip"), step=1_000_000)
    pool.add_from_path(_frozen_zip(tmp_path, "b.zip"), step=2_000_000)
    pool.add_from_path(_frozen_zip(tmp_path, "c.zip"), step=3_000_000)  # over cap → evict oldest (seed)
    steps = [e.step for e in pool._entries]
    assert 0 not in steps                       # the seed ages out
    assert steps == [1_000_000, 2_000_000, 3_000_000]
    # The evicted snapshot file is unlinked from disk.
    assert not (tmp_path / "snapshot_000000000000.zip").exists()


def test_add_from_path_idempotent_replaces_same_step(tmp_path):
    pool = _make_pool(tmp_path)
    pool.add_from_path(_frozen_zip(tmp_path, "a.zip", b"first"), step=500_000)
    pool.add_from_path(_frozen_zip(tmp_path, "b.zip", b"second"), step=500_000)
    assert len(pool) == 1
    assert pool._entries[0].step == 500_000
    assert (tmp_path / "snapshot_000000500000.zip").read_bytes() == b"second"


def test_add_from_path_handles_src_equal_dst(tmp_path):
    # Promoting a file that is already at the destination path must not error.
    pool = _make_pool(tmp_path)
    dst = tmp_path / "snapshot_000000750000.zip"
    dst.write_bytes(b"already-here")
    entry = pool.add_from_path(dst, step=750_000)
    assert entry.path == dst and dst.read_bytes() == b"already-here"
    assert len(pool) == 1


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


def test_scan_pins_nothing(tmp_path):
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    pool.seed(model)
    pool.add(model, step=1_000_000)
    pool2 = _make_pool(tmp_path)
    assert all(not e.pinned for e in pool2._entries)   # sliding window — nothing pinned


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


def test_sentinel_entries_n_one_on_deep_pool_no_zerodiv(tmp_path):
    # --n-sentinels 1 against a multi-snapshot pool must NOT hit the (len-1)/(n-1) divide-by-zero
    # (it would crash the trainer on the eval boundary and re-crash every resume). n<=1 → the newest.
    pool = _make_pool(tmp_path)
    model = _fake_model(tmp_path)
    for s in range(10):
        pool.add(model, step=s * 1_000_000)
    for n in (1, 0, -3):                       # callback clamps 0/neg to 1, but guard the pool too
        sentinels = pool.sentinel_entries(n=n)
        assert len(sentinels) == 1
        assert sentinels[0].step == 9_000_000  # newest first


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


# ── summary.json (resume state) ──────────────────────────────────────────────

def test_persist_summary_merges_and_loads(tmp_path):
    pool = _make_pool(tmp_path)
    pool.persist_summary(win_rate_vs_bots=0.71, self_play_fraction=0.45, last_eval_step=49_000_000)
    pool.persist_summary(pool_generation=3)            # merge, not overwrite
    s = pool.load_summary()
    assert s["win_rate_vs_bots"] == pytest.approx(0.71)
    assert s["self_play_fraction"] == pytest.approx(0.45)
    assert s["last_eval_step"] == 49_000_000
    assert s["pool_generation"] == 3


def test_summary_is_preferred_source_for_win_rate(tmp_path):
    pool = _make_pool(tmp_path)
    pool.persist_summary(win_rate_vs_bots=0.66)
    assert pool.load_persisted_win_rate() == pytest.approx(0.66)


def test_load_persisted_win_rate_falls_back_to_legacy_txt(tmp_path):
    # A pool predating summary.json (only the .txt) still resumes at the right level.
    pool = _make_pool(tmp_path)
    (tmp_path / pool._WIN_RATE_FILE).write_text("0.620000\n")
    assert pool.load_persisted_win_rate() == pytest.approx(0.62)


def test_load_summary_missing_returns_empty(tmp_path):
    pool = _make_pool(tmp_path)
    assert pool.load_summary() == {}




# ── the pool's ARCH RECORD is write-once (gen3_pool_arch_record_v1, P10 follow-up F1) ──────────

def _real_versions():
    """A real `ModelVersion` and a STRUCTURALLY different one (opp_belief_cls_k is a weight-shape field)."""
    import dataclasses

    from agents.model.snapshot import current_model_version
    from agents.observation.state_encoder import load_mappings

    a = current_model_version(load_mappings())
    return a, dataclasses.replace(a, opp_belief_cls_k=a.opp_belief_cls_k + 1)


def test_the_pool_arch_record_is_written_once_so_a_resumed_pool_stays_checked(tmp_path):
    """The record (`<pool>/model_config.json`) is what `load_opponent_snapshot` gates the live version
    against. It was REWRITTEN with the live version on every add, so after the first promotion of a
    session every load compared the run with its own copy (vacuous for the snapshots already in the
    directory — a resume's, a fork's seeded parent pool). Revert `_record_arch` to the rewrite and the
    drifted pool's `add` succeeds, then `load_model` passes the arch check and fails for another reason."""
    from agents.model.model_version import ModelVersionError

    a, b = _real_versions()
    pool_a = SnapshotPool(pool_dir=tmp_path, current_version=a)
    pool_a.add(_fake_model(tmp_path), step=100)
    record = (tmp_path / "model_config.json").read_text()
    assert record == a.to_json()

    pool_b = SnapshotPool(pool_dir=tmp_path, current_version=b)      # the dir resumed under drifted code
    old = pool_b._entries[0]
    with pytest.raises(ModelVersionError):
        pool_b.load_model(old)                                        # a REAL check: the record is A's
    with pytest.raises(ModelVersionError, match="NOT compatible"):
        pool_b.add(_fake_model(tmp_path), step=200)                   # refused BEFORE anything is written
    assert (tmp_path / "model_config.json").read_text() == record     # never rewritten with B
    assert not (tmp_path / "snapshot_000000000200.zip").exists()
    with pytest.raises(ModelVersionError):
        pool_b.load_model(old)                                        # and the next load is STILL a real check


def test_a_compatible_version_adds_without_rewriting_the_record(tmp_path):
    import dataclasses

    a, _ = _real_versions()
    pool_a = SnapshotPool(pool_dir=tmp_path, current_version=a)
    pool_a.add(_fake_model(tmp_path), step=100)
    record = (tmp_path / "model_config.json").read_text()
    a2 = dataclasses.replace(a, vf_coef=a.vf_coef + 0.25)             # not a structural field
    assert a2.to_json() != record
    pool_2 = SnapshotPool(pool_dir=tmp_path, current_version=a2)
    pool_2.add(_fake_model(tmp_path), step=200)                       # compatible: the add goes through
    assert (tmp_path / "model_config.json").read_text() == record     # the record keeps what it was first written under
    assert sorted(e.step for e in pool_2._entries) == [100, 200]


def test_add_from_path_is_refused_under_the_same_record(tmp_path):
    from agents.model.model_version import ModelVersionError

    a, b = _real_versions()
    SnapshotPool(pool_dir=tmp_path, current_version=a).add(_fake_model(tmp_path), step=100)
    src = tmp_path / "frozen.zip"
    src.write_bytes(b"frozen-weights")
    with pytest.raises(ModelVersionError, match="NOT compatible"):
        SnapshotPool(pool_dir=tmp_path, current_version=b).add_from_path(src, step=300)
    assert not (tmp_path / "snapshot_000000000300.zip").exists()


def test_a_pool_holding_snapshots_but_no_record_is_announced_then_recorded(tmp_path, capsys):
    a, _ = _real_versions()
    (tmp_path / "snapshot_000000000050.zip").write_bytes(b"old")      # a pre-record run's snapshot
    pool = SnapshotPool(pool_dir=tmp_path, current_version=a)
    assert not (tmp_path / "model_config.json").exists()
    pool.add(_fake_model(tmp_path), step=100)
    assert (tmp_path / "model_config.json").read_text() == a.to_json()
