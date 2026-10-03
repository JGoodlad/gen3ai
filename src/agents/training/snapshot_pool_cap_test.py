"""gen3_pool_cap_every_path_v1 — ``max_snapshots`` holds on EVERY path that populates a SnapshotPool.

Before it, the window was applied only when a snapshot was ADDED; a pool built by a directory SCAN
(every env worker's, every resume's, the Rust env core's, M5 Lane G's fewer-snapshots harness) held
whatever the directory held. Each test here fails on a revert of the scan-side cap."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.training.snapshot_pool import (
    DEFAULT_MAX_SNAPSHOTS,
    PoolOverCapError,
    SnapshotEntry,
    SnapshotPool,
)


def _version() -> MagicMock:
    v = MagicMock()
    v.to_json.return_value = "{}"
    return v


def _pool(pool_dir: Path, **kw) -> SnapshotPool:
    return SnapshotPool(pool_dir=pool_dir, current_version=_version(), **kw)


def _zips(d: Path, steps) -> list[Path]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for s in steps:
        p = d / f"snapshot_{s:012d}.zip"
        p.write_bytes(b"zip")
        out.append(p)
    return out


def _on_disk(d: Path) -> list[int]:
    return sorted(int(p.stem.split("_")[1]) for p in d.glob("snapshot_*.zip"))


STEPS25 = [1_000 * (i + 1) for i in range(25)]


def test_reader_scan_holds_the_default_window_and_leaves_disk_alone(tmp_path):
    _zips(tmp_path, STEPS25)
    pool = _pool(tmp_path)                          # a reader (env worker / Rust core / harness)
    assert pool.max_snapshots == DEFAULT_MAX_SNAPSHOTS == 20
    assert len(pool) == 20
    assert pool.steps() == STEPS25[-20:]            # the add path's order: oldest evicted first
    assert _on_disk(tmp_path) == STEPS25            # a reader NEVER deletes (models/ is read-only)


def test_declared_smaller_window_over_a_real_dir(tmp_path):
    # The Lane G shape: a caller declares K < what the directory holds.
    _zips(tmp_path, STEPS25)
    pool = _pool(tmp_path, max_snapshots=4)
    assert pool.steps() == STEPS25[-4:]
    assert {pool.sample().step for _ in range(200)} <= set(STEPS25[-4:])
    assert {e.step for e in pool.sentinel_entries(n=9)} <= set(STEPS25[-4:])


def test_reader_scan_over_symlinks_keeps_the_targets(tmp_path):
    src = tmp_path / "run" / "snapshots"
    targets = _zips(src, STEPS25[:6])
    wd = tmp_path / "links"
    wd.mkdir()
    for t in targets:
        (wd / t.name).symlink_to(t)
    pool = _pool(wd, max_snapshots=2)
    assert pool.steps() == STEPS25[4:6]
    assert all(t.exists() for t in targets) and len(list(wd.iterdir())) == 6


def test_owner_scan_deletes_the_surplus(tmp_path):
    _zips(tmp_path, STEPS25)
    pool = _pool(tmp_path, owns_dir=True)           # the trainer's writer pool
    assert pool.steps() == STEPS25[-20:]
    assert _on_disk(tmp_path) == STEPS25[-20:]
    # so every reader that scans after it agrees with it
    assert _pool(tmp_path).steps() == pool.steps()


def test_rescan_applies_the_window_again(tmp_path):
    # RustEnvOpponents (and the deleted Python wrapper before it) re-scan the SAME pool object on each generation bump.
    _zips(tmp_path, STEPS25[:3])
    pool = _pool(tmp_path, max_snapshots=3)
    _zips(tmp_path, STEPS25[3:5])
    pool._scan()
    assert pool.steps() == STEPS25[2:5]


@pytest.mark.parametrize("spread", [False, True])
def test_reader_mid_promotion_scan_matches_the_writers_eviction(tmp_path, spread):
    """A reader that scans a dir one over the cap (a crash between a promotion's copy and its
    eviction) keeps EXACTLY the set the writer's add path keeps — one declared order, both paths."""
    writer_dir, reader_dir = tmp_path / "w", tmp_path / "r"
    base = [0, 1_000, 5_000, 6_000]
    _zips(writer_dir, base)
    writer = _pool(writer_dir, max_snapshots=4, pool_spread=spread, owns_dir=True)
    src = _zips(tmp_path / "frozen", [9_000])[0]
    writer.add_from_path(src, 9_000)
    _zips(reader_dir, base + [9_000])
    reader = _pool(reader_dir, max_snapshots=4, pool_spread=spread)
    assert reader.steps() == writer.steps()
    assert len(reader) == 4


def test_seed_on_a_full_pool_keeps_the_cap(tmp_path):
    _zips(tmp_path, STEPS25[:3])
    pool = _pool(tmp_path, max_snapshots=3)
    m = MagicMock()
    m.save.side_effect = lambda path: Path(path).write_bytes(b"zip")
    pool.seed(m)
    assert len(pool) == 3


@pytest.mark.parametrize("spread", [False, True])
def test_over_cap_after_eviction_raises(tmp_path, spread):
    # Everything pinned: the declared eviction cannot reach the window, so the pool must refuse to
    # exist rather than quietly run oversize (today's `_evict` loop just stopped).
    _zips(tmp_path, STEPS25[:2])
    pool = _pool(tmp_path, max_snapshots=2, pool_spread=spread)
    for e in pool._entries:
        e.pinned = True
    pool.max_snapshots = 1
    with pytest.raises(PoolOverCapError):
        pool._enforce_cap(unlink=False)
    assert _on_disk(tmp_path) == STEPS25[:2]


def test_sample_refuses_an_over_cap_pool(tmp_path):
    _zips(tmp_path, STEPS25[:2])
    pool = _pool(tmp_path, max_snapshots=2)
    pool._entries.append(SnapshotEntry(path=tmp_path / "x.zip", step=99_000))  # bypassing every path
    with pytest.raises(PoolOverCapError):
        pool.sample()


def test_a_trimmed_scan_is_reported(tmp_path):
    _zips(tmp_path, STEPS25[:5])
    with patch("agents.training.snapshot_pool.emit") as em:
        _pool(tmp_path, max_snapshots=3)
    lines = [c.args[0] for c in em.call_args_list]
    assert any("over its declared max_snapshots=3" in ln and "left on disk" in ln for ln in lines), lines


def test_window_below_one_is_refused(tmp_path):
    with pytest.raises(ValueError):
        _pool(tmp_path, max_snapshots=0)


def test_rust_route_roster_never_exceeds_the_window(tmp_path):
    # Lane E: the Rust env core's pool slots are max_snapshots + a small spare; a roster built from an
    # over-cap dir used to SILENTLY spend the spare (or raise SlotCapacityExceeded past it).
    from agents.training.rust_env_opponents import PoolRoster

    _zips(tmp_path, STEPS25)
    roster = PoolRoster(_pool(tmp_path))
    assert len(roster.model_ids()) == DEFAULT_MAX_SNAPSHOTS
