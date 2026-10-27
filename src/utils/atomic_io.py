"""Atomic file replacement: the ONE writer for everything a resume, a restart or a reader may load.

A write goes to a temp file in the SAME directory, is fsync'd, and is then ``os.replace``d onto its final name
(and the directory is fsync'd), so the final name only ever holds the previous complete file or the new complete
one. A writer that raises, or a process killed mid-write, leaves the previous file untouched and nothing partial
at the final name (T46 review F8: ``model.save("final_model_interrupted")`` wrote straight to the fixed name, so a
failed save left a TORN zip that ``latest.txt`` still named and whose fresh mtime passed the GPU window's check).

File names and formats are the callers' own and unchanged: ``atomic_write_text`` / ``atomic_write_json`` write the
bytes ``open(path, "w")`` + ``json.dump`` always wrote, and ``atomic_save_model`` the zip ``model.save(path)`` always
wrote (SB3 appends ``.zip`` to a suffix-less name; so does this).

The temp name is ``.<final name>.<pid>.<8 hex>.tmp``: hidden, ending in ``.tmp``, so no ``*.zip`` / ``*.json`` /
``snapshot_*`` glob, no ``endswith(".zip")`` test and no step-number regex can load it. A SIGKILL leaves one behind;
the next atomic write into that directory removes every stray whose owner pid is dead (a live writer's temp, in this
or another process, is never touched).
"""
from __future__ import annotations

import contextlib
import json
import os
import pathlib
import re
import secrets
from typing import Any, Callable, List

from utils import procfs

#: ``.<final name>.<pid>.<token>.tmp`` (see the module docstring).
_TEMP_RE = re.compile(r"^\..+\.(?P<pid>\d+)\.[0-9a-f]{8}\.tmp$")


class AtomicWriteError(RuntimeError):
    """The writer returned without producing the file it was handed (a silent no-op is not a save)."""


def temp_path(path: "str | os.PathLike[str]") -> str:
    """A fresh temp name for ``path``, in its directory (so ``os.replace`` stays on one filesystem)."""
    d, base = os.path.split(os.fspath(path))
    return os.path.join(d, f".{base}.{os.getpid()}.{secrets.token_hex(4)}.tmp")


def is_temp_name(name: str) -> bool:
    """True for a name :func:`temp_path` produces (a reader that lists a directory skips these)."""
    return _TEMP_RE.match(os.path.basename(name)) is not None


#: Is ``pid`` a RUNNING process? Read from ``/proc`` (``utils.procfs``), never ``kill(pid, 0)``: inside a sandboxed shell
#: ``kill`` read a live process as dead (acf2193f), it reads an exited-but-unreaped zombie as alive, and it answers
#: EPERM for a process we may not signal where ``/proc`` simply lists it. A zombie's temp is debris (its writer has
#: exited); a process that ``/proc`` lists is never swept, whoever owns it.
_pid_alive = procfs.pid_alive


def sweep_stale_temps(directory: "str | os.PathLike[str]") -> List[str]:
    """Remove the temp files in ``directory`` whose owner process is dead (a crashed save's debris, up to a
    checkpoint in size). Returns the removed names. Never raises; never touches a live owner's temp."""
    removed: List[str] = []
    try:
        names = os.listdir(directory)
    except OSError:
        return removed
    for name in names:
        m = _TEMP_RE.match(name)
        if m is None or _pid_alive(int(m.group("pid"))):
            continue
        with contextlib.suppress(OSError):
            os.remove(os.path.join(directory, name))
            removed.append(name)
    return removed


def fsync_path(path: "str | os.PathLike[str]") -> None:
    """Flush a file's data to disk by name (whatever wrote it: a library's ``save(path)`` included)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(directory: str) -> None:
    """Make the rename itself durable. Best effort: a filesystem that cannot fsync a directory loses nothing
    a process kill (the hazard here) could cost."""
    with contextlib.suppress(OSError):
        fd = os.open(directory or ".", os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path: "str | os.PathLike[str]", write_fn: Callable[[str], Any], *, sweep: bool = True) -> str:
    """Write ``path`` atomically: ``write_fn(tmp)`` writes the whole file at the temp path it is given (any
    writer that takes a path: ``model.save``, ``shutil.copy2``, an ``open``); then fsync, ``os.replace``,
    directory fsync. If ``write_fn`` raises (or produces nothing) the temp is removed and the exception
    propagates, with the previous ``path`` untouched. Returns ``path``."""
    final = os.fspath(path)
    directory = os.path.dirname(final)
    os.makedirs(directory or ".", exist_ok=True)
    if sweep:
        sweep_stale_temps(directory or ".")
    tmp = temp_path(final)
    try:
        write_fn(tmp)
        if not os.path.isfile(tmp):
            raise AtomicWriteError(f"the writer for {final} produced no file at {tmp}")
        fsync_path(tmp)
        os.replace(tmp, final)
        # POSIX rename() of one hard link onto ANOTHER NAME OF THE SAME INODE does nothing and leaves the source
        # (a re-link of a name that is already the twin's): drop the leftover name.
        if os.path.lexists(tmp):
            os.remove(tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp)
        raise
    _fsync_dir(directory)
    return final


def atomic_write_bytes(path: "str | os.PathLike[str]", data: bytes) -> str:
    def _w(tmp: str) -> None:
        with open(tmp, "wb") as f:
            f.write(data)

    return atomic_write(path, _w)


def atomic_write_text(path: "str | os.PathLike[str]", text: str, *, encoding: "str | None" = None) -> str:
    def _w(tmp: str) -> None:
        with open(tmp, "w", encoding=encoding) as f:
            f.write(text)

    return atomic_write(path, _w)


def atomic_write_json(path: "str | os.PathLike[str]", obj: Any, **dump_kwargs: Any) -> str:
    """``json.dump(obj, f, **dump_kwargs)`` into ``path``, atomically (the same bytes the bare ``open`` wrote)."""
    def _w(tmp: str) -> None:
        with open(tmp, "w") as f:
            json.dump(obj, f, **dump_kwargs)

    return atomic_write(path, _w)


def model_zip_path(path: "str | os.PathLike[str]") -> str:
    """The file ``model.save(path)`` writes: SB3 appends ``.zip`` to a suffix-less name, nothing else."""
    p = pathlib.Path(os.fspath(path))
    return str(p.with_suffix(".zip")) if p.suffix == "" else str(p)


def atomic_save_model(model: Any, path: "str | os.PathLike[str]") -> str:
    """``model.save(path)`` atomically; returns the final ``.zip`` path (see :func:`model_zip_path`).

    The model is handed the TEMP name, which carries a suffix (``.tmp``), so SB3 writes exactly there; the
    weights appear at the final name only once complete, and the name's previous file (and every other hard
    link to it) is never written through."""
    return atomic_write(model_zip_path(path), lambda tmp: model.save(tmp))
