"""`utils.atomic_io`: the one atomic writer behind every checkpoint, sidecar, pointer and pool file (T46 review F8).

The contract pinned here, each test failing on a revert of the helper to a plain `open(path, "w")` / in-place
`model.save(path)`:

* a writer that raises mid-write leaves the PREVIOUS file byte-identical and nothing partial at the final name;
* a PROCESS killed mid-write (SIGKILL, a real subprocess) does the same, and leaves only a hidden stray temp;
* a stray temp is never taken for a file by a reader (`*.zip` / `snapshot_*` globs, the launcher's checkpoint finder)
  and is removed by the next atomic write into its directory when its owner is dead - never when it is alive;
* the bytes written are exactly the bytes the bare `open` / `json.dump` / `model.save` always wrote.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import textwrap
import zipfile

import pytest

from utils import atomic_io
from utils import liveness_fakes as LF
from utils.atomic_io import (
    AtomicWriteError, atomic_save_model, atomic_write, atomic_write_bytes, atomic_write_json, atomic_write_text,
    is_temp_name, model_zip_path, sweep_stale_temps, temp_path,
)
from utils.paths import src_path

OLD = b"previous complete file"


def _only(directory) -> list:
    return sorted(os.listdir(directory))


def _partial_then(exc: BaseException):
    """A writer that writes HALF a file at the temp path it is handed, then fails."""

    def write(tmp: str) -> None:
        with open(tmp, "wb") as f:
            f.write(b"PK-torn-half")
            f.flush()
            raise exc

    return write


# ---------------------------------------------------------------------------------------------- the invariant

@pytest.mark.parametrize("exc", [OSError(28, "No space left on device"), RuntimeError("boom"), KeyboardInterrupt()],
                         ids=["enospc", "runtime", "keyboard-interrupt"])
def test_a_writer_that_raises_leaves_the_previous_file_intact_and_no_partial_file(tmp_path, exc):
    final = tmp_path / "final_model_interrupted.zip"
    final.write_bytes(OLD)
    mtime = os.stat(final).st_mtime_ns
    with pytest.raises(type(exc)):
        atomic_write(final, _partial_then(exc))
    assert final.read_bytes() == OLD and os.stat(final).st_mtime_ns == mtime, "the previous file was touched"
    assert _only(tmp_path) == ["final_model_interrupted.zip"], "a temp / partial file was left behind"


def test_a_writer_that_raises_for_a_new_name_creates_nothing_at_that_name(tmp_path):
    final = tmp_path / "checkpoints" / "checkpoint_100_steps.zip"
    with pytest.raises(RuntimeError):
        atomic_write(final, _partial_then(RuntimeError("boom")))
    assert not final.exists() and os.listdir(final.parent) == []


def test_a_writer_that_produces_no_file_is_an_error_not_a_save(tmp_path):
    final = tmp_path / "x.zip"
    final.write_bytes(OLD)
    with pytest.raises(AtomicWriteError):
        atomic_write(final, lambda tmp: None)
    assert final.read_bytes() == OLD and _only(tmp_path) == ["x.zip"]


def test_a_json_dump_that_fails_midway_keeps_the_previous_metadata(tmp_path):
    """`json.dump` streams chunks, so a value that cannot be serialised fails AFTER the first bytes are written:
    the exact torn-metadata shape a plain `open(path, "w")` leaves (and the next save then crashes reading)."""
    path = tmp_path / "metadata.json"
    atomic_write_json(path, {"snapshot_history": {"a": 1}}, indent=2)
    before = path.read_bytes()
    with pytest.raises(TypeError):
        atomic_write_json(path, {"snapshot_history": {"a": 1}, "bad": object()}, indent=2)
    assert path.read_bytes() == before and json.loads(before) == {"snapshot_history": {"a": 1}}
    assert _only(tmp_path) == ["metadata.json"]


# ---------------------------------------------------------------------------------------------- a killed process

def test_a_process_killed_mid_write_leaves_the_previous_file_and_only_a_hidden_stray_temp(tmp_path, monkeypatch):
    """A REAL SIGKILL inside the writer (no `except`, no `finally` can run): the final name still holds the previous
    file, the debris is a hidden `*.tmp`, and the next atomic write removes it once its owner is dead."""
    final = tmp_path / "final_model_interrupted.zip"
    final.write_bytes(OLD)
    child = textwrap.dedent(f"""
        import os, signal, sys
        sys.path.insert(0, {str(src_path())!r})
        from utils.atomic_io import atomic_write

        def writer(tmp):
            with open(tmp, "wb") as f:
                f.write(b"PK-torn-half")
                f.flush()
                os.kill(os.getpid(), signal.SIGKILL)

        atomic_write({str(final)!r}, writer)
    """)
    proc = subprocess.run([sys.executable, "-I", "-c", child], capture_output=True)
    assert proc.returncode == -signal.SIGKILL, proc
    assert final.read_bytes() == OLD, "the final name was torn by the killed save"
    strays = [n for n in os.listdir(tmp_path) if n != final.name]
    assert len(strays) == 1 and is_temp_name(strays[0]) and strays[0].startswith("."), strays
    assert (tmp_path / strays[0]).read_bytes() == b"PK-torn-half"
    assert [p.name for p in tmp_path.glob("*.zip")] == [final.name], "a stray temp matched a *.zip glob"
    # ...and the next save into the directory cleans it (the owner is dead; asserted, not left to pid reuse)
    owner = int(strays[0].rsplit(".", 3)[1])
    monkeypatch.setattr(atomic_io, "_pid_alive", lambda pid: pid != owner)
    atomic_write_bytes(tmp_path / "other.zip", b"new")
    assert _only(tmp_path) == ["final_model_interrupted.zip", "other.zip"]
    assert final.read_bytes() == OLD


# ---------------------------------------------------------------------------------------------- stray temps

def _stray(directory, name: str, pid: int, data: bytes = b"PK-torn-half") -> str:
    p = directory / f".{name}.{pid}.deadbeef.tmp"
    p.write_bytes(data)
    return p.name


def test_a_stray_temp_is_never_a_file_to_a_reader(tmp_path):
    """The launcher's checkpoint finder (`**/*.zip`), the pool's `snapshot_*.zip` and `endswith(".zip")` tests all
    skip it, even when it is the NEWEST file in the directory and a later step than every real checkpoint."""
    from main.launcher.checkpoint import find_latest_checkpoint

    ckpts = tmp_path / "checkpoints"
    ckpts.mkdir()
    real = ckpts / "checkpoint_100_steps.zip"
    real.write_bytes(b"PK-real")
    os.utime(real, (1, 1))
    for name in ("checkpoint_900_steps.zip", "final_model_interrupted.zip", "snapshot_000000000900.zip"):
        stray = _stray(ckpts, name, 999_999)
        assert is_temp_name(stray) and not stray.endswith((".zip", ".json"))
    (tmp_path / "latest.txt").write_text("checkpoints/checkpoint_900_steps.zip\n")      # points at a file that was never completed
    assert find_latest_checkpoint(run_dir=str(tmp_path)) == str(real), "a stray temp was taken for a checkpoint"
    assert list(ckpts.glob("snapshot_*.zip")) == [] and [n for n in os.listdir(ckpts) if n.endswith(".zip")] == [real.name]


def test_the_next_atomic_write_removes_a_dead_owners_temp_and_spares_a_live_one(tmp_path, monkeypatch):
    dead = _stray(tmp_path, "checkpoint_100_steps.zip", 4_000_001)
    alive = _stray(tmp_path, "checkpoint_200_steps.zip", 4_000_002)
    not_ours = ["checkpoint_300_steps.zip.tmp", ".hidden.tmp", "latest.txt.tmp", "notes.txt"]
    for n in not_ours:
        (tmp_path / n).write_text("x")
    monkeypatch.setattr(atomic_io, "_pid_alive", lambda pid: pid == 4_000_002)
    atomic_write_text(tmp_path / "latest.txt", "checkpoint_100_steps.zip\n")
    left = set(os.listdir(tmp_path))
    assert dead not in left, "a dead owner's temp survived the next save"
    assert alive in left, "a LIVE owner's temp (another process mid-save) was removed"
    assert set(not_ours) <= left, "a file that is not a temp of ours was removed"


def test_sweep_stale_temps_reports_what_it_removed_and_survives_a_missing_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(atomic_io, "_pid_alive", lambda pid: False)
    name = _stray(tmp_path, "x.zip", 7)
    assert sweep_stale_temps(tmp_path) == [name] and os.listdir(tmp_path) == []
    assert sweep_stale_temps(tmp_path / "missing") == []


def test_the_pid_probe_reads_proc_not_kill_so_a_sweep_deletes_only_on_proof_of_death(tmp_path, monkeypatch):
    """A zombie owner's temp is debris (``kill`` says alive); a live owner's temp is never swept even when ``kill``
    says ESRCH (the sandbox, acf2193f) or EPERM. Each cell fails on a revert to ``os.kill``."""
    LF.assert_pid_liveness(atomic_io._pid_alive, monkeypatch)
    mine = _stray(tmp_path, "checkpoint_200_steps.zip", os.getpid())             # a live writer mid-save
    with LF.zombie() as z:
        dead = _stray(tmp_path, "checkpoint_100_steps.zip", z)
        with monkeypatch.context() as m:
            LF.rig_kill(m, "esrch")
            assert sweep_stale_temps(tmp_path) == [dead]
    assert os.listdir(tmp_path) == [mine]


def test_temp_names_are_hidden_unique_and_in_the_final_names_directory(tmp_path):
    a, b = temp_path(tmp_path / "x" / "final_model.zip"), temp_path(tmp_path / "x" / "final_model.zip")
    assert a != b and os.path.dirname(a) == str(tmp_path / "x")
    assert is_temp_name(a) and os.path.basename(a).startswith(".final_model.zip.") and a.endswith(".tmp")
    assert not is_temp_name("checkpoint_100_steps.zip") and not is_temp_name("latest.txt.tmp")


# ---------------------------------------------------------------------------------------------- same bytes as before

def test_the_bytes_are_exactly_what_the_bare_writers_wrote(tmp_path):
    obj = {"b": [1, 2, {"z": None}], "a": 1.5, "name": "é"}
    atomic_write_json(tmp_path / "a.json", obj, indent=2)
    atomic_write_json(tmp_path / "b.json", obj, indent=2, sort_keys=True)
    atomic_write_json(tmp_path / "c.json", obj)
    with open(tmp_path / "ref.json", "w") as f:
        json.dump(obj, f, indent=2)
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "ref.json").read_bytes()
    assert (tmp_path / "b.json").read_text() == json.dumps(obj, indent=2, sort_keys=True)
    assert (tmp_path / "c.json").read_text() == json.dumps(obj)
    atomic_write_text(tmp_path / "latest.txt", "checkpoints/checkpoint_1_steps.zip\n")
    assert (tmp_path / "latest.txt").read_bytes() == b"checkpoints/checkpoint_1_steps.zip\n"
    atomic_write_text(tmp_path / "u.txt", "é\n", encoding="utf-8")
    assert (tmp_path / "u.txt").read_bytes() == "é\n".encode()


def test_an_existing_file_is_replaced_and_the_directory_is_created(tmp_path):
    p = tmp_path / "deep" / "er" / "x.txt"
    atomic_write_text(p, "one")
    atomic_write_text(p, "two")
    assert p.read_text() == "two" and _only(p.parent) == ["x.txt"]


# ---------------------------------------------------------------------------------------------- the model save

class _ZipModel:
    """`save(path)` as SB3's writes it: it appends nothing when the name has a suffix, and writes a real zip."""

    def __init__(self, payload: bytes = b"weights", fail: "BaseException | None" = None):
        self.payload, self.fail, self.paths = payload, fail, []

    def save(self, path: str) -> None:
        self.paths.append(path)
        if not os.path.splitext(path)[1]:
            path += ".zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("policy.pth", self.payload[: len(self.payload) // 2])
            if self.fail is not None:
                raise self.fail
            z.writestr("rest.pth", self.payload[len(self.payload) // 2:])


def test_model_zip_path_is_what_sb3_would_have_written():
    assert model_zip_path("/r/final_model") == "/r/final_model.zip"
    assert model_zip_path("/r/checkpoints/checkpoint_5_steps.zip") == "/r/checkpoints/checkpoint_5_steps.zip"
    assert model_zip_path("/r/snap.v2") == "/r/snap.v2", "SB3 appends .zip only to a suffix-less name"


def test_atomic_save_model_writes_a_complete_zip_at_the_final_name_from_a_temp_name(tmp_path):
    model = _ZipModel(b"0123456789")
    final = atomic_save_model(model, str(tmp_path / "final_model"))
    assert final == str(tmp_path / "final_model.zip") and zipfile.is_zipfile(final)
    (handed,) = model.paths
    assert is_temp_name(handed) and os.path.splitext(handed)[1] == ".tmp", "the model must be handed a suffixed temp"
    assert _only(tmp_path) == ["final_model.zip"]


def test_a_failed_model_save_keeps_the_previous_zip_and_every_hard_link_to_it(tmp_path):
    final = tmp_path / "checkpoint_150_steps.zip"
    atomic_save_model(_ZipModel(b"OLD-weights"), str(final))
    twin = tmp_path / "best_model.zip"
    os.link(final, twin)
    old_bytes = final.read_bytes()
    with pytest.raises(OSError):
        atomic_save_model(_ZipModel(b"NEW-weights", fail=OSError(28, "No space left on device")), str(final))
    assert final.read_bytes() == old_bytes and zipfile.is_zipfile(final) and twin.read_bytes() == old_bytes
    atomic_save_model(_ZipModel(b"NEW-weights"), str(final))
    assert final.read_bytes() != old_bytes and twin.read_bytes() == old_bytes, "the save wrote through the link"
    assert not os.path.samefile(final, twin)


def test_a_real_sb3_model_saves_atomically_and_loads_back(tmp_path):
    """The real writer: SB3 must put the zip exactly where the helper's temp name says (no `.zip` appended to
    `.tmp`), and the replaced file must be a loadable checkpoint."""
    from stable_baselines3.common.save_util import load_from_zip_file

    from main.fresh_checkpoint import build_fresh_model

    model, _args, _pk = build_fresh_model(0)
    final = atomic_save_model(model, str(tmp_path / "final_model"))
    assert _only(tmp_path) == ["final_model.zip"]
    data, params, _vars = load_from_zip_file(final, device="cpu")
    assert "policy" in params and data is not None
