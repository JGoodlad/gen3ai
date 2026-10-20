"""B-F1 (core review 2026-10-20): a pool snapshot RE-promoted at a step the pool already holds (the replay after
a crash restart: checkpoint C < S, replay to S, promote again) must serve the NEW weights under a name that
says so - never the old ones.

DECISION: replace, do not refuse. A crash replay reaches S again BECAUSE the run restarted; refusing the
promotion would crash-loop an auto-restarting run, and the replayed timeline is the run's real one (the old
file belongs to the abandoned timeline). So the file is replaced (as it always was) and every copy of the old
weights is dropped: the pool's LRU model cache, the step's win rate, the PFSP EMA, and the host's T2 slot
(``pool:<S>`` becomes ``pool:<S>@<n>``, which `SlotFamily.admit` loads into a free slot).

Each test fails on a revert of the identity check (`SnapshotPool._observe`)."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.training import rust_env_opponents as E
from agents.training.selfplay_callback import SelfPlayCallback
from agents.training.snapshot_pool import SnapshotPool

STEP = 1000


def _digest_loader(path, **_kw):
    """Stands in for the opponent load: the 'model' IS the file's content hash."""
    return SimpleNamespace(content=hashlib.sha256(Path(path).read_bytes()).hexdigest()[:8],
                           policy=SimpleNamespace(eval=lambda: None))


def _pool(d: Path, **kw) -> SnapshotPool:
    v = MagicMock()
    v.to_json.return_value = "{}"
    return SnapshotPool(pool_dir=d, current_version=v, **kw)


def _src(d: Path, name: str, payload: bytes) -> Path:
    p = d / name
    p.write_bytes(payload)
    return p


def test_a_repromotion_serves_the_new_weights_to_a_separate_reader_pool(tmp_path):
    """The production shape: the trainer's writer pool promotes, the host's reader pool (another object over
    the same directory) rescans at the next generation and loads."""
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    with patch("agents.training.snapshot_pool.load_opponent_snapshot", _digest_loader):
        writer = _pool(pool_dir, owns_dir=True)
        reader = _pool(pool_dir)
        writer.add_from_path(_src(src, "a.zip", b"OLD weights"), STEP)
        reader._scan()
        old = reader.load_model(reader._entries[0]).content
        assert reader.model_id(STEP) == f"pool:{STEP}"

        writer.add_from_path(_src(src, "b.zip", b"NEW weights, after the crash replay"), STEP)   # the replay
        assert writer.revision(STEP) == 1 and writer.model_id(STEP) == f"pool:{STEP}@1"
        reader._scan()                                     # what RustEnvOpponents._refresh_pool does
        new = reader.load_model(reader._entries[0]).content
        assert new != old, "the reader kept serving the OLD weights under the same name"
        assert new == hashlib.sha256(b"NEW weights, after the crash replay").hexdigest()[:8]
        assert reader.model_id(STEP) == f"pool:{STEP}@1"
        # the writer's own cache is dropped too
        assert writer.load_model(writer._entries[0]).content == new


def test_an_unchanged_rescan_is_not_a_replacement(tmp_path):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    with patch("agents.training.snapshot_pool.load_opponent_snapshot", _digest_loader):
        writer = _pool(pool_dir, owns_dir=True)
        writer.add_from_path(_src(src, "a.zip", b"W"), STEP)
        reader = _pool(pool_dir)
        reader.load_model(reader._entries[0])
        for _ in range(3):
            reader._scan()
        assert reader.revision(STEP) == 0 and reader.model_id(STEP) == f"pool:{STEP}"
        assert len(reader._model_cache) == 1               # still cached: no spurious reload


def test_the_win_rate_measured_on_the_old_weights_is_dropped(tmp_path):
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    with patch("agents.training.snapshot_pool.load_opponent_snapshot", _digest_loader):
        pool = _pool(pool_dir, owns_dir=True)
        pool.add_from_path(_src(src, "a.zip", b"OLD"), STEP)
        pool.set_win_rates({STEP: 0.1})
        pool.add_from_path(_src(src, "b.zip", b"NEW"), STEP)
        assert STEP not in pool._win_rates
    # ... and the callback's EMA, which is what the pool's rates are pushed FROM and persisted as
    cb = SimpleNamespace(_pool=pool, _pfsp_winrate_ema={STEP: 0.1, 2 * STEP: 0.9})
    SelfPlayCallback._forget_replaced_step(cb, STEP)
    assert cb._pfsp_winrate_ema == {2 * STEP: 0.9}
    SelfPlayCallback._forget_replaced_step(cb, 2 * STEP)    # never replaced: untouched
    assert cb._pfsp_winrate_ema == {2 * STEP: 0.9}


def test_the_host_reloads_its_t2_slot_when_the_snapshot_is_replaced(tmp_path):
    """The whole chain through the REAL host: the slot that held ``pool:<S>`` is not mistaken for the new
    weights - the new id is LOADED (with the new weights) into a free slot."""
    pool_dir = tmp_path / "pool"
    pool_dir.mkdir()
    src = tmp_path / "src"
    src.mkdir()
    loaded: list[tuple[int, str, str]] = []
    with patch("agents.training.snapshot_pool.load_opponent_snapshot", _digest_loader):
        writer = _pool(pool_dir, owns_dir=True)
        writer.add_from_path(_src(src, "a.zip", b"OLD"), STEP)
        host_pool = _pool(pool_dir)
        plan = E.OpponentPlan(pool_slots=3, bots=("heuristic",))

        def load(route: int, mid: str) -> None:
            step = int(mid.split(":")[1].partition("@")[0])
            entry = next(e for e in host_pool._entries if e.step == step)
            loaded.append((route, mid, host_pool.load_model(entry).content))

        host = E.RustEnvOpponents(plan, 1, load=load, pool=host_pool)
        assert [m for _, m, _ in loaded] == [f"pool:{STEP}"]
        old_content = loaded[0][2]

        writer.add_from_path(_src(src, "b.zip", b"NEW"), STEP)
        reloaded = host.set_self_play_target(0.5, 1)         # a new generation: rescan + admit
        assert len(reloaded) == 1, "the replaced snapshot was not reloaded into T2"
        assert [m for _, m, _ in loaded] == [f"pool:{STEP}", f"pool:{STEP}@1"]
        assert loaded[1][2] != old_content
        assert loaded[1][2] == hashlib.sha256(b"NEW").hexdigest()[:8]
        assert host.families["pool"].roster == [f"pool:{STEP}@1"]
