"""The Showdown server this tool starts, and the PID it is obliged to stop.

An external-anchor read needs a server both clients can reach, and it must NEVER be one that was
already running: **8001 carries the live training run** (dropping it crashes every poke-env
websocket at once) and **8000 is the shared dev server**. Both are refused in CODE here, the same
way :func:`main.play.resolve_server` refuses them, because a warning in a document is not a guard.

Three properties, each of which has cost someone a session somewhere:

* **We start it, we stop it, BY PID.** :class:`ShowdownServer` is a context manager that records
  ``proc.pid`` and terminates exactly that process on exit — success, failure or exception. Never
  ``npm run stop``, which kills :8000, and never a blanket ``pkill``.
* **The port is ours alone.** :func:`pick_port` binds a candidate before returning it, so two
  anchor reads on the same box cannot collide. The search range comes from the config
  (``9500-9599`` by default); the two reserved ports are excluded from every path.
* **"Started" means ANSWERING, not "the process exists".** :meth:`ShowdownServer.wait_ready`
  polls the port with a real TCP connect until the deadline and raises naming the port and the
  server's own last log lines, because a client that connects to a half-started server hangs
  rather than failing.

``--server-uri`` bypasses all of this: pass a URI and the tool starts nothing, which is how it will
point at the in-repo websocket front end over the Rust bridge once that exists.
"""
from __future__ import annotations

import re
import socket
import subprocess
import time
from pathlib import Path
from typing import IO, List, Optional

from utils.paths import repo_path

#: Ports this process must never touch, with the reason stated for the message it prints.
RESERVED_PORTS = {8000: "the shared DEV server", 8001: "the live TRAINING server"}

#: The websocket path every Showdown server serves.
WS_PATH = "/showdown/websocket"


class ServerError(RuntimeError):
    """The server could not be started, or did not come up inside its deadline."""


def server_uri(port: int, host: str = "localhost") -> str:
    return f"ws://{host}:{port}{WS_PATH}"


def port_of(uri: str) -> Optional[int]:
    """The port in a ``ws://host:port/path`` URI, or None when it names none."""
    m = re.match(r"^wss?://[^/:]+:(\d+)(/|$)", uri)
    return int(m.group(1)) if m else None


def refuse_reserved(port: Optional[int]) -> None:
    """Raise if ``port`` is 8000 or 8001. Called on every path that names a port, including the
    one where the caller supplied a whole URI — a reserved port reached through a URI is the same
    mistake with better camouflage."""
    if port is None:
        return
    reason = RESERVED_PORTS.get(port)
    if reason is not None:
        raise ServerError(
            f"refusing port {port}: that is {reason}. An anchor read has no business on either; "
            "start your own on a 9XXX port (this tool picks one for you if you pass neither "
            "--port nor --server-uri)."
        )


def _free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def pick_port(port_range: Optional[List[int]] = None) -> int:
    """The first free port in the configured range, with the two reserved ports excluded.

    Bound-then-released rather than merely "nothing is listening": the race window is small and
    the alternative — two anchor reads sharing a server — silently mixes two measurements.
    """
    lo, hi = (port_range or [9500, 9599])[:2]
    for port in range(int(lo), int(hi) + 1):
        if port in RESERVED_PORTS:
            continue
        if _free(port):
            return port
    raise ServerError(f"no free port in {lo}-{hi}; another anchor read may still be running")


class ShowdownServer:
    """A Showdown server from ``deps/pokemon-showdown``, started and stopped by its own PID.

    Use as a context manager. ``uri`` is what the two clients connect to; ``pid`` is what gets
    terminated, and nothing else on the box is touched.
    """

    def __init__(self, port: int, node: str = "node", log_path: Optional[Path] = None,
                 start_timeout_s: float = 120.0) -> None:
        refuse_reserved(port)
        self.port = port
        self.node = node
        self.log_path = log_path
        self.start_timeout_s = start_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        #: The PID we started, kept after :meth:`stop` so the teardown line names a real process.
        self.last_pid: Optional[int] = None
        self._log: Optional[IO[str]] = None

    # -- lifecycle ----------------------------------------------------------------------------
    @property
    def uri(self) -> str:
        return server_uri(self.port)

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else self.last_pid

    def argv(self) -> List[str]:
        """The exact command. ``--no-security`` is what lets guest accounts battle locally; the
        port is POSITIONAL (Showdown has no ``--port`` flag)."""
        entry = repo_path("deps", "pokemon-showdown", "pokemon-showdown")
        return [self.node, str(entry), "start", "--no-security", str(self.port)]

    def start(self) -> "ShowdownServer":
        entry = repo_path("deps", "pokemon-showdown", "pokemon-showdown")
        if not entry.exists():
            raise ServerError(
                f"{entry} is missing — run `git submodule update --init`. In a worktree the "
                "dist/ and node_modules/ symlinks are needed too (root CLAUDE.md § Git Worktree "
                "Setup)."
            )
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "w", buffering=1)
        self.proc = subprocess.Popen(
            self.argv(),
            cwd=str(repo_path()),
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.wait_ready()
        return self

    def wait_ready(self) -> None:
        """Block until the port ANSWERS a TCP connect, or raise naming the port and the log tail.

        "The process exists" is not readiness: a client that dials a half-started server sits in
        the connect and the whole series looks dead with nothing to read.
        """
        deadline = time.time() + self.start_timeout_s
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(
                    f"the Showdown server exited immediately (rc={self.proc.returncode}). "
                    + self._log_tail()
                )
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1.0)
                if s.connect_ex(("127.0.0.1", self.port)) == 0:
                    return
            time.sleep(0.5)
        self.stop()
        raise ServerError(
            f"the Showdown server on :{self.port} never answered within {self.start_timeout_s:g}s. "
            + self._log_tail()
        )

    def _log_tail(self, n: int = 15) -> str:
        if self.log_path is None or not self.log_path.exists():
            return "(no server log captured)"
        lines = self.log_path.read_text(errors="replace").splitlines()[-n:]
        return "Last lines of " + str(self.log_path) + ":\n" + "\n".join(lines)

    def stop(self, grace_s: float = 10.0) -> None:
        """Terminate EXACTLY the PID we started. Idempotent; never raises.

        ``last_pid`` survives the stop so a caller can still SAY which PID it reaped — "stopped
        (pid was None)" is not a record of anything.
        """
        proc, self.proc = self.proc, None
        if proc is not None:
            self.last_pid = proc.pid
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=grace_s)
                except subprocess.TimeoutExpired:
                    pass
        if self._log is not None:
            self._log.close()
            self._log = None

    def __enter__(self) -> "ShowdownServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()
