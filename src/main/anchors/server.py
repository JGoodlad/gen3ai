"""The server this tool starts, and the PID it is obliged to stop.

An external-anchor read needs a server both clients can reach, and it must NEVER be one that was
already running: **8001 carries the live training run** (dropping it crashes every poke-env
websocket at once) and **8000 is the shared dev server**. Both are refused in CODE here, the same
way :func:`main.play.resolve_server` refuses them, because a warning in a document is not a guard.

🚨 **THE DEFAULT SERVER IS NOT NODE.** ``--server rust`` starts :class:`FrontEndServer` — the
in-repo websocket front end (:mod:`utils.bridge.ws_frontend`) over the Rust ``sim_bridge`` — and no
Node server exists for the life of the read. ``--server node`` starts the pinned
``deps/pokemon-showdown`` (:class:`ShowdownServer`) and is kept as the explicit opt-out, which is
what a transport differential is taken against. The promotion's evidence is
``designs/research_state/measurements/anchors_rust_frontend_2026-09-20/README.md``; the earlier
side-by-side that licensed it is ``foulplay_axes_and_frontend_validation_2026-09-16``.

Three properties, each of which has cost someone a session somewhere, and both classes inherit
them from :class:`ManagedServer` rather than restating them:

* **We start it, we stop it, BY PID.** A :class:`ManagedServer` is a context manager that records
  ``proc.pid`` and terminates exactly that process on exit — success, failure or exception. Never
  ``npm run stop``, which kills :8000, and never a blanket ``pkill``.
* **The port is ours alone.** :func:`pick_port` binds a candidate before returning it, so two
  anchor reads on the same box cannot collide. The search range comes from the config
  (``9500-9599`` by default); the two reserved ports are excluded from every path.
* **"Started" means SERVING, not "the process exists".** :meth:`ManagedServer.wait_ready` waits
  for the child's own readiness LINE when it prints one (the front end does) and polls the port
  with a real TCP connect when it does not (the Node server), then raises naming the port and the
  server's own last log lines, because a client that connects to a half-started server hangs
  rather than failing.

``--server-uri`` bypasses all of this: pass a URI and the tool starts nothing. It is then an
EXTERNAL server and the rows say so (``server_impl = "external"``) — this tool cannot vouch for a
transport it did not start.
"""
from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import IO, Dict, List, Optional

from utils.paths import repo_path, src_root

#: Ports this process must never touch, with the reason stated for the message it prints.
RESERVED_PORTS = {8000: "the shared DEV server", 8001: "the live TRAINING server"}

#: The websocket path every Showdown server serves.
WS_PATH = "/showdown/websocket"

#: ``--server`` values. ``rust`` is the DEFAULT: no Node process is started at all.
SERVER_KINDS = ("rust", "node")


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


class ManagedServer:
    """A server this tool started, stopped by its own PID. The half both transports share.

    Subclasses supply :meth:`argv` (the exact command), :attr:`label` (what the messages call it),
    optionally :meth:`preflight` (a refusal BEFORE anything is spawned, naming the fix) and
    :attr:`ready_marker` (a line the child prints when it is serving).
    """

    #: What the log lines and errors call this server.
    label = "server"
    #: The child's own readiness announcement, or None when a TCP answer is all there is.
    ready_marker: Optional[str] = None
    #: The default basename for the log, so a caller need not know which server it asked for.
    log_name = "server.log"

    def __init__(self, port: int, log_path: Optional[Path] = None,
                 start_timeout_s: float = 120.0) -> None:
        refuse_reserved(port)
        self.port = port
        self.log_path = log_path
        self.start_timeout_s = start_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        #: The PID we started, kept after :meth:`stop` so the teardown line names a real process.
        self.last_pid: Optional[int] = None
        self._log: Optional[IO[str]] = None

    # -- what a subclass supplies -------------------------------------------------------------
    def argv(self) -> List[str]:
        raise NotImplementedError

    def env(self) -> Optional[Dict[str, str]]:
        """The child's environment, or None to inherit this process's."""
        return None

    def preflight(self) -> None:
        """Refuse BEFORE spawning, naming the fix. Default: nothing to check."""

    def version(self) -> str:
        """What is stamped on every row as ``server_version`` — the transport's identity."""
        return ""

    # -- lifecycle ----------------------------------------------------------------------------
    @property
    def uri(self) -> str:
        return server_uri(self.port)

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc is not None else self.last_pid

    def start(self) -> "ManagedServer":
        self.preflight()
        if self.log_path is not None:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log = open(self.log_path, "w", buffering=1)
        self.proc = subprocess.Popen(
            self.argv(),
            cwd=str(repo_path()),
            env=self.env(),
            stdout=self._log or subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.wait_ready()
        return self

    def _announced_ready(self) -> bool:
        """Has the child printed its own readiness line into a log we can read?"""
        if self.ready_marker is None or self.log_path is None or not self.log_path.exists():
            return False
        return self.ready_marker in self.log_path.read_text(errors="replace")

    def _answers_tcp(self) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            return s.connect_ex(("127.0.0.1", self.port)) == 0

    def wait_ready(self) -> None:
        """Block until the server is SERVING, or raise naming the port and the log tail.

        "The process exists" is not readiness: a client that dials a half-started server sits in
        the connect and the whole series looks dead with nothing to read. Two pieces of evidence,
        and the child's own announcement is preferred where there is one:

        * **a readiness LINE in the log** (``ready_marker``) — printed after the listener is
          bound, so it is at least as strong as a connect, and it costs the server nothing;
        * **a TCP connect** otherwise, which is all a server that announces nothing offers.

        🚨 **The probe is not free on a websocket server.** A bare TCP connect that opens and
        closes without an HTTP request makes `websockets` log `opening handshake failed` with a
        three-deep traceback — measured 2026-09-20, one per start. That is a fabricated ERROR in
        the transport's own log, and "0 ERRORs in the server log" is exactly the criterion the
        front end's validation reads. So a server that announces itself is never TCP-probed.
        """
        deadline = time.time() + self.start_timeout_s
        announces = self.ready_marker is not None and self.log_path is not None
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise ServerError(
                    f"the {self.label} exited immediately (rc={self.proc.returncode}). "
                    + self._log_tail()
                )
            if self._announced_ready() if announces else self._answers_tcp():
                return
            time.sleep(0.5)
        self.stop()
        raise ServerError(
            f"the {self.label} on :{self.port} never answered within {self.start_timeout_s:g}s. "
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

    def __enter__(self) -> "ManagedServer":
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()


class ShowdownServer(ManagedServer):
    """A Showdown server from ``deps/pokemon-showdown``, started and stopped by its own PID.

    The ``--server node`` opt-out: a whole Node process, kept because a differential needs a
    reference transport that is not ours.
    """

    label = "Showdown server"
    log_name = "showdown.log"

    def __init__(self, port: int, node: str = "node", log_path: Optional[Path] = None,
                 start_timeout_s: float = 120.0) -> None:
        super().__init__(port, log_path=log_path, start_timeout_s=start_timeout_s)
        self.node = node

    def argv(self) -> List[str]:
        """The exact command. ``--no-security`` is what lets guest accounts battle locally; the
        port is POSITIONAL (Showdown has no ``--port`` flag)."""
        entry = repo_path("deps", "pokemon-showdown", "pokemon-showdown")
        return [self.node, str(entry), "start", "--no-security", str(self.port)]

    def preflight(self) -> None:
        entry = repo_path("deps", "pokemon-showdown", "pokemon-showdown")
        if not entry.exists():
            raise ServerError(
                f"{entry} is missing — run `git submodule update --init`. In a worktree the "
                "dist/ and node_modules/ symlinks are needed too (root CLAUDE.md § Git Worktree "
                "Setup)."
            )

    def version(self) -> str:
        return f"showdown:{showdown_pin()}"


class FrontEndServer(ManagedServer):
    """The in-repo websocket front end (:mod:`utils.bridge.ws_frontend`) — **no Node server**.

    Each battle is backed by ONE ``sim_bridge`` child instead of a Showdown server, so the read
    costs one Python process plus one short-lived bridge per battle. The front end is NOT a
    Showdown server and must never be described as one: its deferral list (no ladder, no timer, no
    reconnection) is ``designs/rust_sim/ws_frontend.md``.

    It runs as its own SUBPROCESS rather than a thread of this one, for the reason that made the
    front end exist at all: the opponent owns ``import poke_env``, and the server half must stay
    poke-env-free.

    ``seed_base``/``capture_dir`` are the reproducibility pair — a capture without a seed base is
    NOT replayable, and the front end warns about exactly that at startup.
    """

    label = "ws_frontend"
    log_name = "ws_frontend.log"
    ready_marker = "[ws_frontend] READY"

    def __init__(self, port: int, python: Optional[str] = None, impl: str = "rust",
                 battle_format: str = "gen3ou", seed_base: Optional[int] = None,
                 capture_dir: Optional[Path] = None, log_path: Optional[Path] = None,
                 start_timeout_s: float = 180.0) -> None:
        super().__init__(port, log_path=log_path, start_timeout_s=start_timeout_s)
        self.python = python or sys.executable
        self.impl = impl
        self.battle_format = battle_format
        self.seed_base = seed_base
        self.capture_dir = Path(capture_dir) if capture_dir is not None else None

    def argv(self) -> List[str]:
        argv = [self.python, "-m", "utils.bridge.ws_frontend",
                "--port", str(self.port), "--impl", self.impl,
                "--format", self.battle_format]
        if self.seed_base is not None:
            argv += ["--seed-base", str(self.seed_base)]
        if self.capture_dir is not None:
            argv += ["--capture-dir", str(self.capture_dir)]
        return argv

    def env(self) -> Dict[str, str]:
        """``src`` on ``PYTHONPATH``, PREPENDED even when it is already there further down — the
        child is ``python -m``, and in a worktree an
        editable install would otherwise resolve the module out of the MAIN checkout (root
        CLAUDE.md § Python Environment). Everything else is inherited, ``POKESIM_SIM_BRIDGE_BIN``
        included: the bridge binary the child spawns is the one this process was pointed at."""
        env = dict(os.environ)
        src = str(src_root())
        rest = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != src]
        env["PYTHONPATH"] = os.pathsep.join([src, *rest])
        return env

    def preflight(self) -> None:
        if self.capture_dir is not None:
            self.capture_dir.mkdir(parents=True, exist_ok=True)

    def version(self) -> str:
        """``ws_frontend@<gen3ai short HEAD>+<the bridge binary that backs each battle>``.

        The binary is named rather than assumed: ``POKESIM_SIM_BRIDGE_BIN`` is how a worktree
        avoids building into the main checkout's ``target/``, and a row that did not say which
        binary played could not tell two ports apart.
        """
        from utils.git import get_git_hash

        try:
            head = get_git_hash(short=True)
        except Exception:                            # noqa: BLE001 - provenance, never a blocker
            head = "unknown"
        binary = os.environ.get("POKESIM_SIM_BRIDGE_BIN") or "cargo:src/rust_sim"
        return f"ws_frontend@{head}+{self.impl}:{binary}"


def showdown_pin() -> str:
    """The pinned submodule commit. Recorded on BOTH transports: the Rust port was ported from
    this tree, and the front end still validates a ``/utm`` team through its ``validate_team.js``.
    """
    try:
        out = subprocess.run(
            ["git", "-C", str(repo_path("deps", "pokemon-showdown")), "rev-parse", "--short",
             "HEAD"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def build_server(kind: str, port: int, *, node: str = "node", python: Optional[str] = None,
                 battle_format: str = "gen3ou", seed_base: Optional[int] = None,
                 capture_dir: Optional[Path] = None,
                 out_dir: Optional[Path] = None) -> ManagedServer:
    """The ONE place ``--server`` turns into a process. The log is named after the server, so a
    read's output directory says which transport served it before anything is parsed."""
    if kind not in SERVER_KINDS:
        raise ServerError(f"unknown --server {kind!r} (known: {', '.join(SERVER_KINDS)})")
    if kind == "node":
        cls_log = ShowdownServer.log_name
        return ShowdownServer(
            port, node=node,
            log_path=(out_dir / cls_log) if out_dir is not None else None)
    return FrontEndServer(
        port, python=python, impl="rust", battle_format=battle_format, seed_base=seed_base,
        capture_dir=capture_dir,
        log_path=(out_dir / FrontEndServer.log_name) if out_dir is not None else None)
