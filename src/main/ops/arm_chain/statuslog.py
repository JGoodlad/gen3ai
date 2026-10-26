"""The status file: ``[<iso ts>] <TAG>: <message>`` lines, appended by several writers, read by triggers.

Three class bugs live here, and each is closed in code rather than by care:

* **A trigger matched a STALE line** (two watchers on 2026-10-15 fired on lines already in the log): a
  :class:`Trigger` carries the line OFFSET taken when it was armed and reads only lines past it. The offset is
  persisted by the chain, so a resumed supervisor never re-reads history; a file that SHRANK below an offset
  raises :class:`StatusError` instead of silently re-reading from the top, and so does a file REPLACED (rotated)
  under the trigger (its inode is part of the trigger).
* **A trigger matched ITS OWN line** (2026-10-19: an "armed" line quoted ``X26STOP: X26 STOPPED`` and a whole-file
  grep fired on it): every trigger is anchored (``^[<ts>] <exact tag>: <exact body>$``) on a tag the chain never
  writes, AND :meth:`StatusLog.append` refuses a line that any trigger armed through it would match.
* **An unanchored run name matched another run** (``END rb_x35_c0_s350`` is a prefix of ``…_s3501``): a run name is
  ``re.escape``-d and followed by a fixed delimiter in every pattern built here.

Writes are one ``O_APPEND`` ``write`` per line, so concurrent writers never interleave inside a line.
"""
from __future__ import annotations

import datetime as _dt
import os
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Pattern

_TS = r"\[[^\]]+\] "


class StatusError(RuntimeError):
    pass


class SelfMatchError(StatusError):
    """The chain was about to write a line that one of its own armed triggers would match."""


def _now() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


@dataclass(frozen=True)
class Trigger:
    """An anchored line pattern, live only past ``offset`` (the file's line count when it was armed) in the file
    whose inode was ``inode`` then (0 = the file did not exist yet)."""

    name: str
    pattern: Pattern[str]
    offset: int
    inode: int = 0

    def to_json(self) -> dict:
        return {"name": self.name, "pattern": self.pattern.pattern, "offset": self.offset, "inode": self.inode}

    @staticmethod
    def from_json(d: dict) -> "Trigger":
        return Trigger(name=d["name"], pattern=re.compile(d["pattern"]), offset=int(d["offset"]),
                       inode=int(d.get("inode", 0)))


def line_pattern(tag: str, body: str) -> Pattern[str]:
    """``^[ts] <tag>: <body>$`` with ``tag`` taken LITERALLY and ``body`` a regex the caller anchors with care."""
    return re.compile("^" + _TS + re.escape(tag) + ": " + body + "$")


def launch_tag(arm: str, seed: int) -> str:
    """The tag the backend's launch wrapper writes its lines under (``launch_x35.sh``'s ``log()``)."""
    return f"{arm} s{seed}"


def end_body(run: str) -> str:
    return "END " + re.escape(run) + r" launcher_rc=(-?\d+)"


def start_body(run: str) -> str:
    return "START " + re.escape(run) + ": .*"


class StatusLog:
    def __init__(self, path: str, own_tag: str, clock: Callable[[], str] = _now) -> None:
        self.path = path
        self.own_tag = own_tag
        self._clock = clock
        self._armed: List[Trigger] = []

    # -- reading -------------------------------------------------------------------------------------------------
    def lines(self) -> List[str]:
        try:
            with open(self.path, encoding="utf-8", errors="replace") as f:
                return f.read().splitlines()
        except FileNotFoundError:
            return []

    def line_count(self) -> int:
        return len(self.lines())

    def inode(self) -> int:
        try:
            return os.stat(self.path).st_ino
        except FileNotFoundError:
            return 0

    def lines_after(self, offset: int) -> List[str]:
        ls = self.lines()
        if len(ls) < offset:
            raise StatusError(f"{self.path} has {len(ls)} lines, fewer than the armed offset {offset}: it was "
                              "truncated or rotated — refusing to re-read it from the top")
        return ls[offset:]

    # -- triggers ------------------------------------------------------------------------------------------------
    def arm(self, name: str, tag: str, body: str) -> Trigger:
        """A trigger on ``tag`` lines written from NOW on. ``tag`` must not be this writer's own tag."""
        if tag == self.own_tag:
            raise SelfMatchError(f"trigger '{name}' listens on the chain's own tag '{tag}'")
        t = Trigger(name=name, pattern=line_pattern(tag, body), offset=self.line_count(), inode=self.inode())
        self._armed.append(t)
        return t

    def adopt(self, t: Trigger) -> Trigger:
        """Re-arm a persisted trigger (a resumed supervisor) at its ORIGINAL offset (idempotent)."""
        if t not in self._armed:
            self._armed.append(t)
        return t

    def disarm(self, t: Trigger) -> None:
        self._armed = [x for x in self._armed if x != t]

    def find(self, t: Trigger) -> Optional[re.Match]:
        ino = self.inode()
        if t.inode and ino and ino != t.inode:
            raise StatusError(f"{self.path} was REPLACED since trigger '{t.name}' was armed (inode {t.inode} -> "
                              f"{ino}): refusing to read a different file at the old offset")
        for line in self.lines_after(t.offset):
            m = t.pattern.match(line)
            if m:
                return m
        return None

    # -- writing -------------------------------------------------------------------------------------------------
    def append(self, msg: str, tag: Optional[str] = None) -> str:
        line = f"[{self._clock()}] {tag or self.own_tag}: {msg}"
        if "\n" in line:
            raise StatusError("a status line may not contain a newline")
        for t in self._armed:
            if t.pattern.match(line):
                raise SelfMatchError(f"refusing to write a line that the armed trigger '{t.name}' would match: {line}")
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, (line + "\n").encode())
        finally:
            os.close(fd)
        return line
