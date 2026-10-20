"""The GPU WINDOW's on-disk protocol: the files, their records, and the atomic writes (T46).

Everything a window says lives under ``<run_dir>/gpu_window/``, written by one of three parties:

==========================  ================  ==========================================================
file                        written by        meaning
==========================  ================  ==========================================================
``launcher.json``           the launcher      "this launcher (pid + start ticks) honours windows for this
                                              run" — written at start, only with ``--gpu-window``
``request.json``            ``arm``           a window ADDRESSED to that launcher: its id, its trigger
                                              epoch, its bounds, the precheck that passed
``current.json``            the launcher      the window in flight: its phase and why (the status line)
``w_<id>/request.json``     the launcher      the request, CONSUMED (renamed) at the trigger
``w_<id>/GO.json``          the launcher      the run has checkpointed, exited and left the GPU
``w_<id>/claim.json``       guest / launcher  created ``O_EXCL`` by exactly ONE side: the guest (it runs)
                                              or the launcher (the guest MISSED its start bound)
``w_<id>/DONE.json``        the guest         the guest command's exit code
``w_<id>/RESUME.json``      ``resume``        an operator ends a HOLD
==========================  ================  ==========================================================

Every write is atomic (a temp file + ``os.replace``, or ``O_EXCL`` where exactly one writer may win), and
every record carries the window id, so a file left by an EARLIER window can never satisfy a later one.
"""
from __future__ import annotations

import json
import os
import secrets
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from utils import procfs

WINDOW_DIRNAME = "gpu_window"
LAUNCHER_FILE = "launcher.json"
REQUEST_FILE = "request.json"
CURRENT_FILE = "current.json"
GO_FILE = "GO.json"
CLAIM_FILE = "claim.json"
DONE_FILE = "DONE.json"
RESUME_FILE = "RESUME.json"

#: Phases of ``current.json``. OPEN phases block a new ``arm``; the rest are terminal.
PAUSING, GO, CLAIMED, HOLD = "pausing", "go", "claimed", "hold"
DONE, MISSED, RESUMED, ABANDONED, REFUSED = "done", "missed", "resumed", "abandoned", "refused"
OPEN_PHASES = frozenset({PAUSING, GO, CLAIMED, HOLD})

#: Bounds an ``arm`` and the launcher both enforce (rule: a time trigger is an absolute epoch with a
#: sanity bound, `designs/ops/TRAINING_RUN_SOP.md` §3 rule 1).
PAST_SLACK_S = 60.0                     # an --at-epoch this far in the past is a typo, not "now"
DEFAULT_HORIZON_H = 48.0                # an --at-epoch further ahead than this is refused
HORIZON_CEILING_H = 7 * 24.0            # the largest --horizon-hours a request may carry
DEFAULT_GUEST_START_TIMEOUT_S = 180.0   # GO → claim; past it the launcher claims MISSED and resumes
DEFAULT_MAX_WINDOW_MIN = 60.0           # GO → DONE; past it the window HOLDS (the guest may hold the GPU)
MAX_WINDOW_CEILING_MIN = 24 * 60.0
GUEST_START_CEILING_S = 3600.0


class RequestInvalid(ValueError):
    """A ``request.json`` that cannot be honoured (schema, bounds, or addressed to another launcher)."""


def window_root(run_dir: "str | os.PathLike") -> Path:
    return Path(run_dir) / WINDOW_DIRNAME


def window_dir(run_dir: "str | os.PathLike", window_id: str) -> Path:
    return window_root(run_dir) / f"w_{window_id}"


def new_window_id(now: Optional[float] = None) -> str:
    t = time.time() if now is None else now
    return time.strftime("%Y%m%dT%H%M%S", time.localtime(t)) + "_" + secrets.token_hex(3)


def write_json_atomic(path: Path, obj: dict) -> None:
    """Write ``obj`` so a reader sees the old file or the new one, never half of either."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def create_json_exclusive(path: Path, obj: dict) -> bool:
    """Create ``path`` holding ``obj`` iff it does not exist; True iff THIS call created it.

    The record is written to a temp file first and LINKED into place, so the winner's file is complete
    the moment it exists (a reader never sees an empty claim)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(3)}.tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        return False
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path: Path) -> Optional[dict]:
    """The record at ``path``, or None (absent, or not a JSON object)."""
    try:
        with open(path) as f:
            obj = json.load(f)
    except (OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def process_identity(pid: int) -> Optional[dict]:
    """``{"pid", "start_ticks"}`` of a live process — the pair that survives pid reuse."""
    st = procfs.read_stat(pid)
    return None if st is None else {"pid": pid, "start_ticks": st.starttime}


def identity_alive(ident: Optional[dict]) -> bool:
    """True iff the process named by a ``process_identity`` record is still THAT process."""
    if not ident:
        return False
    try:
        st = procfs.read_stat(int(ident["pid"]))
        return st is not None and st.starttime == int(ident["start_ticks"]) and st.state != "Z"
    except (KeyError, TypeError, ValueError):
        return False


@dataclass(frozen=True)
class Request:
    """One armed window. ``launcher`` is the identity of the launcher it is addressed to."""

    window_id: str
    launcher: dict
    armed_epoch: float
    not_before_epoch: float
    guest_start_timeout_s: float
    max_window_s: float
    note: str = ""
    precheck: dict = field(default_factory=dict)
    lease: str = ""
    horizon_h: float = DEFAULT_HORIZON_H

    def to_json(self) -> dict:
        return {"version": 1, **asdict(self)}

    @classmethod
    def from_json(cls, d: Optional[dict]) -> "Request":
        if not d:
            raise RequestInvalid("request.json is missing or not a JSON object")
        try:
            req = cls(window_id=str(d["window_id"]), launcher=dict(d["launcher"]),
                      armed_epoch=float(d["armed_epoch"]), not_before_epoch=float(d["not_before_epoch"]),
                      guest_start_timeout_s=float(d["guest_start_timeout_s"]),
                      max_window_s=float(d["max_window_s"]), note=str(d.get("note", "")),
                      precheck=dict(d.get("precheck") or {}), lease=str(d.get("lease", "")),
                      horizon_h=float(d.get("horizon_h", DEFAULT_HORIZON_H)))
        except (KeyError, TypeError, ValueError) as e:
            raise RequestInvalid(f"request.json lacks or mistypes a field: {e!r}") from e
        problems = req.problems()
        if problems:
            raise RequestInvalid("; ".join(problems))
        return req

    def problems(self) -> list:
        """The bound violations of this request, judged against its OWN arming time and horizon (so the
        launcher, reading it later, reaches the verdict ``arm`` reached)."""
        out = []
        horizon_h = self.horizon_h
        if not (0 < horizon_h <= HORIZON_CEILING_H):
            out.append(f"horizon {horizon_h:g} h is outside (0, {HORIZON_CEILING_H:.0f}]")
        if not self.window_id or "/" in self.window_id or self.window_id.startswith("."):
            out.append(f"window_id {self.window_id!r} is not a plain name")
        if self.not_before_epoch < self.armed_epoch - PAST_SLACK_S:
            out.append(f"trigger epoch {self.not_before_epoch:.0f} is {self.armed_epoch - self.not_before_epoch:.0f} s "
                       f"BEFORE the arming (> {PAST_SLACK_S:.0f} s): an absolute epoch in the past is a typo")
        if self.not_before_epoch > self.armed_epoch + horizon_h * 3600:
            out.append(f"trigger epoch {self.not_before_epoch:.0f} is more than {horizon_h:g} h after the arming")
        if not (0 < self.guest_start_timeout_s <= GUEST_START_CEILING_S):
            out.append(f"guest start timeout {self.guest_start_timeout_s:g} s is outside (0, {GUEST_START_CEILING_S:.0f}]")
        if not (0 < self.max_window_s <= MAX_WINDOW_CEILING_MIN * 60):
            out.append(f"max window {self.max_window_s / 60:g} min is outside (0, {MAX_WINDOW_CEILING_MIN:.0f}]")
        if not ({"pid", "start_ticks"} <= set(self.launcher)):
            out.append("the request names no launcher identity (pid + start_ticks)")
        return out


def read_current(run_dir: "str | os.PathLike") -> Optional[dict]:
    return read_json(window_root(run_dir) / CURRENT_FILE)


def describe_epoch(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(t))
