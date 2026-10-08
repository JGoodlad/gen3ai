"""Resolve (building if needed) the env core's ``bot_reader`` binary — a scripted bot over one side's
protocol stream (``src/rust_env/src/bin/bot_reader.rs``; poke-env retirement P6, ``main.live.bot_reader``).

The ``live_reader`` precedent (``utils.bridge.sim_bridge_bin.resolve_live_reader_bin``) for the OTHER crate:
``$POKESIM_BOT_READER_BIN`` (an absolute path) wins; else an incremental ``cargo build --bin bot_reader`` of
THIS checkout's ``src/rust_env`` into that crate's own ``target/`` (``CARGO_TARGET_DIR`` forced there — never
another checkout's, the 09-09 rust-target incident). The EMISSION SELF-CHECK switch
(``$POKESIM_EMISSION_SELFCHECK == "1"``: every pytest session and fuzz script) builds the ``selfcheck`` profile,
production builds ``release``. A failure raises :class:`BotReaderBinaryError` naming the command; never a
fallback to a Python bot.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Dict

from utils.paths import src_path

BIN_NAME = "bot_reader"
ENV_OVERRIDE = "POKESIM_BOT_READER_BIN"

_cache: Dict[str, str] = {}
_lock = threading.Lock()


class BotReaderBinaryError(RuntimeError):
    """The ``bot_reader`` binary could not be resolved or built — the message names the fix."""


def profile() -> str:
    from utils.bridge.sim_bridge_bin import selfcheck_requested

    return "selfcheck" if selfcheck_requested() else "release"


def expected_path(prof: str = "") -> Path:
    return src_path("rust_env") / "target" / (prof or profile()) / BIN_NAME


def build_argv(prof: str) -> "list[str]":
    from utils.rust_env.build import PROFILE_FLAGS

    return ["build", *PROFILE_FLAGS[prof], "--bin", BIN_NAME,
            "--manifest-path", str(src_path("rust_env") / "Cargo.toml")]


def resolve_bot_reader_bin() -> str:
    """An absolute path to ``bot_reader`` (see the module docs)."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        p = Path(override)
        if not p.is_file():
            raise BotReaderBinaryError(f"{ENV_OVERRIDE}={override!r} is not a file; point it at a built "
                                       f"`{BIN_NAME}` or unset it to build from src/rust_env")
        return str(p.resolve())
    prof = profile()
    with _lock:
        if prof in _cache:
            return _cache[prof]
        crate = src_path("rust_env")
        if not (crate / "Cargo.toml").is_file():
            raise BotReaderBinaryError(f"no Rust env crate at {crate} (no Cargo.toml); set {ENV_OVERRIDE}")
        cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
        if not os.path.exists(cargo):
            raise BotReaderBinaryError(f"cargo not found (PATH or ~/.cargo/bin); install rustup or set {ENV_OVERRIDE}")
        argv = build_argv(prof)
        r = subprocess.run([cargo, *argv], cwd=str(crate), capture_output=True, text=True,
                           env=dict(os.environ, CARGO_TARGET_DIR=str(crate / "target")))
        if r.returncode != 0:
            tail = (r.stderr or r.stdout or "").strip()[-2000:]
            raise BotReaderBinaryError(f"`cargo {' '.join(argv)}` failed (exit {r.returncode}):\n{tail}")
        path = expected_path(prof)
        if not path.is_file():
            raise BotReaderBinaryError(f"cargo succeeded but {path} is missing")
        _cache[prof] = str(path.resolve())
        return _cache[prof]
