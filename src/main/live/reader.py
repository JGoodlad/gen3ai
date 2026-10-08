"""The reader SESSION — protocol lines in, the TRAINING core's row / mask / choice tokens out (P4).

A thin pipe to the ``live_reader`` Rust binary (``src/rust_sim/src/bin/live_reader.rs``), which reads ONE side of
ONE battle through :rust:`pokesim::side_reader::SideReader` — the very type ``sim_bridge``'s core observation mode
ships training's rows through (``designs/rust_sim/encoder.md`` §5a). The live reader is therefore the training
reader by construction, and P4's gate (a) ("two roads, one row") holds the two byte-equal on real battles.

What the caller owes the session (``main.live.client`` does all of it):

* the BATTLE PROTOCOL only — the room framing (``>battle-…``) and every declared non-battle line
  (:data:`ROOM_SKIP`) are stripped first; every other line is fed, and the reader REFUSES an unknown keyword;
* the side's writes in ORDER, a write being what one server message carried; a decision opens only when a
  write ENDS at the side's ``|request|`` (``encoder.md`` §5a's alignment), which is how a real server frames it
  (the sim's ``sendUpdates``: the battle chunk, then the request as its own ``sideupdate``);
* every choice it sends, through :meth:`LiveReader.choose`, BEFORE the next feed.

EVERY failure is a :class:`ReaderRefusal` (the binary's ``__ERR__``, a dead process, a malformed reply), and the
reader is dead for the rest of the battle: the caller turns it into a T28 halt (``main.live.halt``), never a skip.
"""
from __future__ import annotations

import base64
import json
import subprocess
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The observation width the frame must carry (the training row; read from the wire, checked here).
FRAME_DTYPE = "<f4"

#: Message types a server sends INSIDE a battle room that are NOT battle protocol: SKIPPED (never fed), by this
#: DECLARED list. Every other line is fed to the reader, which refuses an unknown keyword (T28) — an unknown
#: battle keyword is never silently skipped. Server-originated room framing (``init`` / ``title`` / ``deinit`` /
#: ``noinit`` / ``expire``), presence (``j`` / ``l`` / ``n`` and their long and capital forms), chat (``c`` /
#: ``c:`` / ``chat``), server HTML and announcements (``raw`` / ``html`` / ``uhtml`` / ``uhtmlchange`` /
#: ``notify`` / ``tempnotify`` / ``tempnotifyoff`` / ``hidelines`` / ``unlink`` / ``badge``), and the battle
#: TIMER's notices (``inactive`` / ``inactiveoff``). None of them is ever emitted by the simulator on the gen-3
#: path — the stream training reads never holds one (``designs/rust_sim/ws_frontend.md``; the server adds them).
ROOM_SKIP = frozenset({
    "init", "title", "deinit", "noinit", "expire",
    "j", "J", "join", "l", "L", "leave", "n", "N", "name",
    "c", "c:", "chat",
    "raw", "html", "uhtml", "uhtmlchange", "notify", "tempnotify", "tempnotifyoff", "hidelines", "unlink", "badge",
    "inactive", "inactiveoff",
})


def keyword(line: str) -> Optional[str]:
    """The ``|kw|`` of a protocol line (``""`` for the bare ``|``), ``None`` for a line not starting with ``|``."""
    if not line.startswith("|"):
        return None
    end = line.find("|", 1)
    return line[1:] if end < 0 else line[1:end]


def is_room_line(line: str) -> bool:
    """A declared non-battle line (:data:`ROOM_SKIP`)."""
    return keyword(line) in ROOM_SKIP


class ReaderRefusal(RuntimeError):
    """The reader refused its input (or died). Carries the reader's own kind and message."""

    def __init__(self, message: str, kind: str = "reader") -> None:
        super().__init__(message)
        self.kind = kind


@dataclass
class Frame:
    """One decision's training frame — exactly the ``__OBS__`` frame ``sim_bridge``'s core_obs mode ships."""
    side: str
    row: np.ndarray            # float32 (OBS_DIM,), read-only view
    mask: np.ndarray           # int8 (11,)
    tokens: Dict[int, str]     # action index -> the choice string the server accepts
    turn: int
    line: int
    rqid: Optional[int]
    n: int
    raw: str = field(repr=False, default="")  # the frame's JSON exactly as the reader wrote it

    @classmethod
    def parse(cls, side: str, text: str) -> "Frame":
        obj = json.loads(text)
        fr = obj["frame"]
        if fr.get("dtype") != FRAME_DTYPE or len(fr.get("shape", [])) != 1:
            raise ReaderRefusal(f"a frame with dtype {fr.get('dtype')!r} shape {fr.get('shape')!r}", "frame")
        buf = base64.b64decode(fr["b64"])
        row = np.frombuffer(buf, dtype=np.float32)
        if row.shape != (int(fr["shape"][0]),):
            raise ReaderRefusal(f"a frame of {len(buf)} bytes for shape {fr['shape']}", "frame")
        mask = np.asarray(obj["mask"], dtype=np.int8)
        tokens = {int(k): str(v) for k, v in obj["tokens"].items()}
        return cls(side=side, row=row, mask=mask, tokens=tokens, turn=int(obj["turn"]), line=int(obj["line"]),
                   rqid=obj.get("rqid"), n=int(obj["n"]), raw=text)


def live_reader_argv() -> List[str]:
    from utils.bridge.sim_bridge_bin import resolve_live_reader_bin
    return [resolve_live_reader_bin()]


class LiveReader:
    """One ``live_reader`` child; one side of one battle at a time (``open`` starts a new one)."""

    def __init__(self, argv: Optional[Sequence[str]] = None) -> None:
        self._argv = list(argv) if argv is not None else live_reader_argv()
        self._proc: Optional[subprocess.Popen] = None
        self.side: Optional[str] = None
        self.folded = 0
        self.decided = 0
        self.turn = 0
        self._dead: Optional[str] = None

    # -- process ------------------------------------------------------------------------------
    def _spawn(self) -> subprocess.Popen:
        if self._proc is None or self._proc.poll() is not None:
            self._proc = subprocess.Popen(self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        return self._proc

    def _send(self, line: str) -> None:
        proc = self._spawn()
        try:
            assert proc.stdin is not None
            proc.stdin.write(line + "\n")
            proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise self._died(f"write failed: {exc}") from exc

    def _died(self, what: str) -> ReaderRefusal:
        err = ""
        if self._proc is not None and self._proc.poll() is not None and self._proc.stderr is not None:
            err = self._proc.stderr.read()[-2000:]
        self._dead = f"the live_reader process died ({what}); stderr: {err!r}"
        return ReaderRefusal(self._dead, "process")

    def _read(self) -> str:
        assert self._proc is not None and self._proc.stdout is not None
        line = self._proc.stdout.readline()
        if not line:
            raise self._died("EOF")
        return line.rstrip("\n")

    def _expect(self, want: str) -> List[str]:
        """Read reply lines up to and including the one starting ``want``; ``__ERR__`` raises."""
        got: List[str] = []
        while True:
            ln = self._read()
            if ln.startswith("__ERR__ "):
                obj = json.loads(ln[len("__ERR__ "):])
                self._dead = obj.get("message", ln)
                raise ReaderRefusal(obj.get("message", ln), obj.get("kind", "reader"))
            got.append(ln)
            if ln.startswith(want):
                return got
            if not ln.startswith("__OBS__ "):
                raise ReaderRefusal(f"unexpected reply line {ln[:200]!r}", "protocol")

    def _alive(self) -> None:
        if self._dead is not None:
            raise ReaderRefusal(f"refused after an earlier failure: {self._dead}", "dead")

    # -- the session --------------------------------------------------------------------------
    def open(self, side: str, name: str, packed_team: Optional[str]) -> None:
        self._dead = None
        self.side, self.folded, self.decided, self.turn = side, 0, 0, 0
        self._send("OPEN " + json.dumps({"side": side, "name": name, "team": packed_team}, ensure_ascii=False))
        self._expect("__OK__")

    def feed(self, lines: Sequence[str]) -> Optional[Frame]:
        """Fold one write; the decision's :class:`Frame` when the write ended at one."""
        self._alive()
        for ln in lines:
            if "\n" in ln or "\r" in ln:
                raise ReaderRefusal(f"a line holding a newline: {ln[:120]!r}", "input")
        self._send("FEED " + json.dumps(list(lines), ensure_ascii=False))
        frame: Optional[Frame] = None
        for ln in self._expect("__FED__ "):
            if ln.startswith("__OBS__ "):
                _, side, payload = ln.split(" ", 2)
                frame = Frame.parse(side, payload)
            else:
                st = json.loads(ln[len("__FED__ "):])
                self.folded, self.decided, self.turn = int(st["folded"]), int(st["decided"]), int(st["turn"])
        return frame

    def choose(self, token: str) -> None:
        """Note the choice sent for the open decision (BEFORE the next feed)."""
        self._alive()
        if not token or "\n" in token:
            raise ReaderRefusal(f"a bad choice token {token!r}", "input")
        self._send("CHOOSE " + token)
        self._expect("__OK__")

    def probe(self) -> Optional[str]:
        """The DRIFT SCAN's encoder check (never the live path): encode the current reading with or without an
        open decision. ``None`` = it encoded (finite); else the encoder's message. A probe failure does not fail
        the reader."""
        self._alive()
        self._send("PROBE")
        ln = self._read()
        if ln == "__OK__":
            return None
        if ln.startswith("__ERR__ "):
            obj = json.loads(ln[len("__ERR__ "):])
            if obj.get("kind") != "probe":
                self._dead = obj.get("message", ln)
                raise ReaderRefusal(obj.get("message", ln), obj.get("kind", "reader"))
            return str(obj.get("message"))
        raise ReaderRefusal(f"unexpected PROBE reply {ln[:200]!r}", "protocol")

    def close(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._send("END")
                self._proc.wait(timeout=10)
            except (ReaderRefusal, subprocess.TimeoutExpired):
                self._proc.kill()
                self._proc.wait()
        self._proc = None

    def __enter__(self) -> "LiveReader":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def split_room_message(text: str) -> Tuple[Optional[str], List[str]]:
    """A server websocket message → (room id or None for the global room, its lines)."""
    lines = text.split("\n")
    if lines and lines[0].startswith(">"):
        return lines[0][1:].strip(), lines[1:]
    return None, lines
