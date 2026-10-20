"""The checkpoint COMMIT protocol (T46 review F8): a save that fails or is killed mid-write never leaves a torn file
at a name a reader loads, and `latest.txt` moves only after the zip AND its sidecars are complete.

The defect: `lifecycle._commit_abort` wrote `model.save("final_model_interrupted")` straight to its fixed name. A
failed save left a TORN zip; `latest.txt` (which named that file after any earlier interrupt) still pointed at it, and
its fresh mtime passed the GPU window's freshness check, so a resume loaded a torn file. Every test below fails on a
revert to the in-place save / the old order (zip, pointer, THEN sidecars).
"""
from __future__ import annotations

import ast
import errno
import os
import threading
import zipfile
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from main.train import lifecycle, run_io
from main.train.run_io import _TrackingCheckpointCallback, commit_checkpoint
from utils import atomic_io
from utils.paths import src_path

OLD_MTIME = 1_000_000_000          # 2001: far older than any SIGTERM the GPU window compares against


def _write_zip(path, marker: str) -> None:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("data", marker)


def _marker(path) -> str:
    with zipfile.ZipFile(path) as z:
        return z.read("data").decode()


class _Model:
    """A learner stub whose `save(path)` writes a real zip at the path it is handed (SB3 appends `.zip` to a
    suffix-less name, nothing else) and can die HALF-WAY through, leaving a partial file."""

    def __init__(self, marker: str = "new", fail: "BaseException | None" = None):
        self.marker, self.fail = marker, fail
        self.num_timesteps = 150
        self.saves: List[str] = []
        self.dumps = 0

    def dump_logs(self) -> None:
        self.dumps += 1

    def save(self, path: str) -> None:
        self.saves.append(path)
        if not os.path.splitext(path)[1]:
            path += ".zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data", self.marker)
            if self.fail is not None:
                z.fp.flush()
                raise self.fail


def _previous_run(tmp_path, *, pointer: str = "final_model_interrupted.zip"):
    """A run whose last save was an interrupt: a valid `final_model_interrupted.zip` that `latest.txt` names."""
    prev = tmp_path / "final_model_interrupted.zip"
    _write_zip(prev, "previous")
    os.utime(prev, (OLD_MTIME, OLD_MTIME))
    (tmp_path / "latest.txt").write_text(pointer + "\n")
    return prev


def _assert_pointer_names_a_complete_target(run_dir, *, expect_marker: str) -> None:
    name = (run_dir / "latest.txt").read_text().strip()
    target = run_dir / name
    assert target.is_file(), f"latest.txt names a MISSING file: {name}"
    assert zipfile.is_zipfile(target), f"latest.txt names a PARTIAL file: {name}"
    assert _marker(target) == expect_marker


def _no_temp_files(directory) -> None:
    assert [n for n in os.listdir(directory) if atomic_io.is_temp_name(n)] == []


class _Exit:
    def __init__(self) -> None:
        self.calls: List[int] = []

    def __call__(self, code: int) -> None:
        self.calls.append(code)


@pytest.fixture
def hooks(monkeypatch):
    """`lifecycle`'s collaborators, recorded: the sidecar writers (`save_model_snapshot`, `record_checkpoint`) and
    the pointer write, each noting what `latest.txt` and the zip said at the instant it ran."""
    events: List[tuple] = []
    state = {"run": None, "fail_in": None}

    def view():
        run = state["run"]
        pointer = (run / "latest.txt").read_text().strip() if (run / "latest.txt").exists() else None
        zp = run / "final_model_interrupted.zip"
        return pointer, (_marker(zp) if zp.exists() and zipfile.is_zipfile(zp) else None)

    def sidecar(name):
        def fn(*a, **k):
            events.append((name, *view()))
            if state["fail_in"] == name:
                raise RuntimeError(f"{name} failed")
        return fn

    real_pointer = lifecycle._write_latest_txt

    def pointer(model_dir, name):
        events.append(("pointer", *view()))
        real_pointer(model_dir, name)

    monkeypatch.setattr(lifecycle, "save_model_snapshot", sidecar("save_model_snapshot"))
    monkeypatch.setattr(lifecycle, "record_checkpoint", sidecar("record_checkpoint"))
    monkeypatch.setattr(lifecycle, "_write_latest_txt", pointer)
    monkeypatch.setattr(lifecycle, "_model_hparams", lambda m: {})
    return events, state


def _abort(model, run_dir, hooks_state) -> _Exit:
    hooks_state["run"] = run_dir
    ex = _Exit()
    ab = lifecycle._setup_signal_handlers(model, str(run_dir), threading.Event(), None, lambda: 1e-3, lambda: 1,
                                          exit_fn=ex, start_watchdog=False)
    try:
        ab.abort("SIGTERM received")
    finally:
        ab.close()
    return ex


# ------------------------------------------------------------------------------ the defect, end to end

def test_a_failed_interrupted_save_leaves_the_previous_zip_the_pointer_and_the_mtime_alone(tmp_path, hooks, capsys):
    events, state = hooks
    prev = _previous_run(tmp_path)
    before = prev.read_bytes()
    ex = _abort(_Model("new", fail=OSError(errno.ENOSPC, "No space left on device")), tmp_path, state)
    assert ex.calls == [15] and "[ABORT] Save failed" in capsys.readouterr().out
    assert prev.read_bytes() == before and os.stat(prev).st_mtime == OLD_MTIME, \
        "the failed save rewrote final_model_interrupted.zip in place (a TORN zip with a FRESH mtime)"
    _assert_pointer_names_a_complete_target(tmp_path, expect_marker="previous")
    assert events == [], f"a sidecar / pointer ran for a save that failed: {events}"
    _no_temp_files(tmp_path)
    assert sorted(os.listdir(tmp_path)) == ["final_model_interrupted.zip", "latest.txt"]


def test_a_killed_first_interrupt_leaves_no_file_for_latest_txt_to_name(tmp_path, hooks):
    events, state = hooks
    (tmp_path / "latest.txt").write_text("checkpoints/checkpoint_100_steps.zip\n")
    (tmp_path / "checkpoints").mkdir()
    _write_zip(tmp_path / "checkpoints" / "checkpoint_100_steps.zip", "periodic")
    _abort(_Model("new", fail=RuntimeError("killed mid-write")), tmp_path, state)
    assert not (tmp_path / "final_model_interrupted.zip").exists(), "a partial zip sits at the final name"
    _assert_pointer_names_a_complete_target(tmp_path, expect_marker="periodic")
    _no_temp_files(tmp_path)


def test_the_interrupted_save_commits_in_order_zip_then_sidecars_then_the_pointer(tmp_path, hooks, capsys):
    events, state = hooks
    _previous_run(tmp_path)
    ex = _abort(_Model("new"), tmp_path, state)
    assert ex.calls == [15] and "Checkpoint saved" in capsys.readouterr().out
    # each sidecar already sees the NEW, complete zip while the pointer is still on the previous target; the
    # pointer write itself (the last step) also sees a complete new zip.
    assert [e[0] for e in events] == ["save_model_snapshot", "record_checkpoint", "pointer"], events
    for _name, pointer, marker in events:
        assert marker == "new", "a step ran before the new zip was complete at its final name"
    _assert_pointer_names_a_complete_target(tmp_path, expect_marker="new")
    _no_temp_files(tmp_path)


@pytest.mark.parametrize("failing", ["save_model_snapshot", "record_checkpoint"])
def test_latest_txt_does_not_move_when_a_sidecar_write_fails(tmp_path, hooks, failing):
    events, state = hooks
    state["fail_in"] = failing
    _previous_run(tmp_path, pointer="final_model_interrupted.zip")
    _abort(_Model("new"), tmp_path, state)
    assert "pointer" not in [e[0] for e in events], "the pointer moved although a sidecar was not written"
    assert (tmp_path / "latest.txt").read_text().strip() == "final_model_interrupted.zip"
    assert zipfile.is_zipfile(tmp_path / "final_model_interrupted.zip")        # (the new zip, complete)
    _no_temp_files(tmp_path)


# ------------------------------------------------------------------------------ the protocol itself

def test_commit_checkpoint_runs_zip_then_sidecars_then_pointer_and_returns_the_final_zip(tmp_path):
    seen = []

    def sidecars(zip_path):
        seen.append(("sidecars", zip_path, zipfile.is_zipfile(zip_path), (tmp_path / "latest.txt").exists()))

    def pointer(model_dir, name):
        seen.append(("pointer", model_dir, name))

    (tmp_path / "checkpoints").mkdir()
    final = commit_checkpoint(_Model(), str(tmp_path), str(tmp_path / "checkpoints" / "checkpoint_150_steps.zip"),
                              sidecars=sidecars, pointer=pointer)
    zp = str(tmp_path / "checkpoints" / "checkpoint_150_steps.zip")
    assert final == zp
    assert seen == [("sidecars", zp, True, False), ("pointer", str(tmp_path), "checkpoints/checkpoint_150_steps.zip")]


def test_commit_checkpoint_stops_before_the_pointer_on_any_failure(tmp_path):
    pointer = MagicMock()
    with pytest.raises(OSError):
        commit_checkpoint(_Model(fail=OSError(errno.EIO, "io")), str(tmp_path), str(tmp_path / "final_model"),
                          pointer=pointer)
    pointer.assert_not_called()
    with pytest.raises(RuntimeError):
        commit_checkpoint(_Model(), str(tmp_path), str(tmp_path / "final_model"),
                          sidecars=MagicMock(side_effect=RuntimeError("sidecar")), pointer=pointer)
    pointer.assert_not_called()


def test_the_default_pointer_writes_latest_txt_run_relative_and_atomically(tmp_path):
    (tmp_path / "checkpoints").mkdir()
    commit_checkpoint(_Model(), str(tmp_path), str(tmp_path / "checkpoints" / "checkpoint_7_steps.zip"))
    commit_checkpoint(_Model(), str(tmp_path), str(tmp_path / "final_model"))
    assert (tmp_path / "latest.txt").read_text() == "final_model.zip\n"
    _no_temp_files(tmp_path)
    _no_temp_files(tmp_path / "checkpoints")


# ------------------------------------------------------------------------------ the forced checkpoint

def test_a_failed_forced_checkpoint_is_reported_and_leaves_the_pointer_where_it_was(tmp_path, hooks, capsys):
    events, state = hooks
    state["run"] = tmp_path
    prev = _previous_run(tmp_path)
    ex = _Exit()
    ab = lifecycle._setup_signal_handlers(_Model("new", fail=OSError(errno.ENOSPC, "full")), str(tmp_path),
                                          threading.Event(), None, lambda: 1e-3, lambda: 1, exit_fn=ex,
                                          start_watchdog=False)
    try:
        ab.request_checkpoint("SIGUSR1 received")
        ab.safe_point("test")
    finally:
        ab.close()
    assert "Forced save FAILED" in capsys.readouterr().out and ex.calls == []
    assert (tmp_path / "latest.txt").read_text().strip() == "final_model_interrupted.zip" and prev.exists()
    assert events == []
    assert os.listdir(tmp_path / "checkpoints") == [], "a fragment / temp was left in checkpoints/"


def test_a_forced_checkpoint_commits_its_sidecar_before_the_pointer(tmp_path, hooks):
    events, state = hooks
    state["run"] = tmp_path
    ab = lifecycle._setup_signal_handlers(_Model("forced"), str(tmp_path), threading.Event(), None,
                                          lambda: 1e-3, lambda: 1, exit_fn=_Exit(), start_watchdog=False)
    try:
        ab.request_checkpoint("SIGUSR1 received")
        ab.safe_point("test")
    finally:
        ab.close()
    assert [e[0] for e in events] == ["record_checkpoint", "pointer"], events
    name = (tmp_path / "latest.txt").read_text().strip()
    assert name.startswith("checkpoints/checkpoint_forced_") and _marker(tmp_path / name) == "forced"
    _no_temp_files(tmp_path / "checkpoints")


# ------------------------------------------------------------------------------ the periodic checkpointer

def _periodic(tmp_path, model, monkeypatch, *, sidecar_fails=False):
    ckpts = tmp_path / "run" / "checkpoints"
    cb = _TrackingCheckpointCallback(interval_env_steps=100, save_path=str(ckpts))
    model.num_timesteps = 0                  # (the anchor of the first boundary test; the caller steps past 100)
    cb.model = model
    cb._init_callback()
    cb._on_training_start()
    cb._current_lr_fn, cb._current_epochs_fn = (lambda: 1e-3), (lambda: 1)
    order: List[str] = []
    monkeypatch.setattr(run_io, "_model_hparams", lambda m: {})

    def record(*a, **k):
        order.append("record_checkpoint")
        if sidecar_fails:
            raise RuntimeError("sidecar failed")

    real_pointer = run_io._write_latest_txt
    monkeypatch.setattr(run_io, "record_checkpoint", record)
    monkeypatch.setattr(run_io, "_write_latest_txt", lambda d, n: (order.append("pointer"), real_pointer(d, n)))
    return cb, ckpts, order


def test_a_failed_periodic_save_at_a_re_reached_step_keeps_the_old_checkpoint_and_the_pointer(tmp_path, monkeypatch):
    """A restart that resumed from an earlier checkpoint re-reaches step 150: the old `checkpoint_150_steps.zip` is a
    complete file that `latest.txt` may name. A save that dies half-way must leave it, not tear it."""
    model = _Model("new", fail=OSError(errno.EACCES, "denied"))
    cb, ckpts, order = _periodic(tmp_path, model, monkeypatch)
    old = ckpts / "checkpoint_150_steps.zip"
    _write_zip(old, "old-150")
    (tmp_path / "run" / "latest.txt").write_text("checkpoints/checkpoint_150_steps.zip\n")
    model.num_timesteps = 150
    with pytest.raises(OSError):
        cb._on_step()
    assert _marker(old) == "old-150" and order == []
    _assert_pointer_names_a_complete_target(tmp_path / "run", expect_marker="old-150")
    _no_temp_files(ckpts)


def test_the_periodic_pointer_moves_last_and_not_at_all_when_the_sidecar_fails(tmp_path, monkeypatch):
    cb, ckpts, order = _periodic(tmp_path, _Model("new"), monkeypatch)
    cb.model.num_timesteps = 150
    cb._on_step()
    assert order == ["record_checkpoint", "pointer"]
    _assert_pointer_names_a_complete_target(tmp_path / "run", expect_marker="new")

    cb, ckpts, order = _periodic(tmp_path / "second", _Model("new"), monkeypatch, sidecar_fails=True)
    cb.model.num_timesteps = 150
    with pytest.raises(RuntimeError):
        cb._on_step()
    assert order == ["record_checkpoint"] and not (tmp_path / "second" / "run" / "latest.txt").exists()


# ------------------------------------------------------------------------------ the JSON sidecars

def test_a_failed_metadata_write_keeps_the_previous_metadata_json_loadable(tmp_path):
    """`metadata.json` is read-modify-written by every save and every eval; a truncated one crashes the NEXT save."""
    import json

    from agents.model.snapshot import record_eval_results, write_checkpoint_metadata

    record_eval_results(str(tmp_path), 100, {"win_rate": 0.5})
    meta = (tmp_path / "metadata.json").read_bytes()
    with pytest.raises(TypeError):
        record_eval_results(str(tmp_path), 200, {"win_rate": 0.6, "bad": object()})
    assert (tmp_path / "metadata.json").read_bytes() == meta and json.loads(meta)["latest_eval"]["step"] == 100
    record_eval_results(str(tmp_path), 300, {"win_rate": 0.7})                  # ...and the next save still works
    assert json.loads((tmp_path / "metadata.json").read_text())["latest_eval"]["step"] == 300

    zp = str(tmp_path / "checkpoint_150_steps.zip")
    write_checkpoint_metadata(zp, 1e-3, 1, hparams={"a": 1}, git_hash="abc", num_timesteps=150)
    side = (tmp_path / "checkpoint_150_steps.json").read_bytes()
    with pytest.raises(TypeError):
        write_checkpoint_metadata(zp, 1e-3, 1, hparams={"a": object()}, git_hash="abc", num_timesteps=150)
    assert (tmp_path / "checkpoint_150_steps.json").read_bytes() == side
    _no_temp_files(tmp_path)


# ------------------------------------------------------------------------------ the self-play pool

def _pool(tmp_path):
    from agents.training.snapshot_pool import SnapshotPool

    version = MagicMock()
    version.to_json.return_value = "{}"
    with patch("agents.training.snapshot_pool.load_opponent_snapshot"):
        return SnapshotPool(pool_dir=tmp_path / "snapshots", current_version=version)


def test_a_failed_pool_snapshot_write_keeps_the_previous_snapshot_and_a_stray_temp_is_not_an_entry(tmp_path):
    pool = _pool(tmp_path)
    entry = pool.add(_Model("first"), 100)
    assert _marker(entry.path) == "first"
    with pytest.raises(OSError):
        pool.add(_Model("second", fail=OSError(errno.ENOSPC, "full")), 100)
    assert _marker(entry.path) == "first", "a worker re-scanning the pool would load a torn snapshot"
    assert sorted(os.listdir(tmp_path / "snapshots")) == ["model_config.json", "snapshot_000000000100.zip"]
    # a SIGKILLed save's debris is hidden, not a `snapshot_*.zip`: the pool's scan never sees it
    (tmp_path / "snapshots" / ".snapshot_000000000200.zip.4000001.deadbeef.tmp").write_bytes(b"PK-torn")
    with patch("agents.training.snapshot_pool.load_opponent_snapshot"):
        again = _pool(tmp_path)
    assert again.steps() == [100]


# ------------------------------------------------------------------------------ nothing saves in place any more

_SAVERS = ("main/train/run_io.py", "main/train/lifecycle.py", "main/train/model_build.py", "main/train/grow_fork.py",
           "agents/training/snapshot_pool.py", "agents/training/eval_callback.py",
           "agents/training/selfplay_callback.py")
#: `lifecycle._run_roundtrip_test` saves into its own throwaway tmpdir, never a run dir.
_EXEMPT_FUNCS = {"_run_roundtrip_test"}


def _direct_model_saves(path) -> List[str]:
    tree = ast.parse(src_path(*path.split("/")).read_text())
    found: List[str] = []

    def visit(node, func):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else func
            if (isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr == "save"
                    and ast.unparse(child.func.value) in ("model", "self.model") and f not in _EXEMPT_FUNCS):
                found.append(f"{path}:{child.lineno} {ast.unparse(child)}")
            visit(child, f)

    visit(tree, None)
    return found


@pytest.mark.parametrize("path", _SAVERS)
def test_no_training_process_module_saves_a_model_in_place(path):
    """A direct `model.save(path)` writes the final name in place (the F8 torn zip). Save through
    `utils.atomic_io.atomic_save_model` (or `run_io.commit_checkpoint`, which adds the sidecars and the pointer)."""
    assert _direct_model_saves(path) == []


def test_every_final_save_in_model_build_goes_through_the_commit_protocol():
    text = src_path("main", "train", "model_build.py").read_text()
    assert text.count("commit_checkpoint(") == 3, "final_model (fresh), final_model (resume), final_model_exception"
