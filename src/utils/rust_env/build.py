"""Build THIS checkout's Rust env core (``src/rust_env``) at a trainer's STARTUP (the only env core).

Both front ends load the env core from a fixed place in the checkout the code was IMPORTED from —
``proc.default_path(profile)`` (``src/rust_env/target/<profile>/rust_env_proc``) and
``ffi.default_path(profile)`` (``…/libpokesim_env.so``, the eval core) — and nothing else built it:
``bootstrap.sh`` does not, and the launcher's pinned worktree (a fresh ``git worktree add`` in
``/tmp``) never has a ``target/``. So every Rust-core run through ``python -m main.launcher``
died ~10 s in with ``ProcLoadError: …/rust_env_proc does not exist`` (F-LG-6, 2026-09-30).

:func:`ensure_built` is the fix, and it follows the Rust bridge precedent
(``utils.bridge.sim_bridge_bin``): the trainer runs an incremental ``cargo build`` of its OWN checkout
once, at startup (the declared lifecycle — acquired before the model exists, never lazily later), into
that crate's own ``target/`` (``CARGO_TARGET_DIR`` is forced there, because that is where the loaders
look; never another checkout's — the 09-09 rust-target incident). A cold build is ~10 s (measured
2026-09-30); a warm one is a no-op. The loaders' stamp check (``stamp.check_stamp``) still refuses a
build that is not this tree's, so this can only ever turn "missing / stale" into "current".
"""
from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from utils.paths import src_path

#: ``--rust-env-profile`` → the cargo flags that build it into ``target/<profile>/``.
PROFILE_FLAGS = {
    "release": ("--release",),
    "selfcheck": ("--profile", "selfcheck", "--features", "emission-selfcheck"),
}

BUILD_TIMEOUT_S = 1800.0


class EnvCoreBuildError(RuntimeError):
    """The env core could not be built — the message names the command and the fix."""


def crate_dir() -> Path:
    return src_path("rust_env")


def build_argv(profile: str) -> List[str]:
    """The ``cargo`` arguments that build the process child AND the cdylib for ``profile``."""
    from utils.rust_env import proc

    if profile not in PROFILE_FLAGS:
        raise EnvCoreBuildError(f"unknown rust env profile {profile!r} (known: {sorted(PROFILE_FLAGS)})")
    return ["build", *PROFILE_FLAGS[profile], "--lib", "--bin", proc.BIN_NAME,
            "--manifest-path", str(crate_dir() / "Cargo.toml")]


def build_command(profile: str) -> str:
    """The shell command a person runs to build ``profile`` where the loaders look."""
    flags = " ".join(PROFILE_FLAGS.get(profile, PROFILE_FLAGS["selfcheck"]))
    return f"(cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo build {flags} --lib --bin rust_env_proc)"


def artifacts(profile: str) -> Tuple[Path, Path]:
    """``(process child, cdylib)`` exactly where ``proc.default_path`` / ``ffi.default_path`` look."""
    from utils.rust_env import ffi, proc

    return proc.default_path(profile), ffi.default_path(profile)


def _cargo() -> Optional[str]:
    c = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    return c if os.path.exists(c) else None


def ensure_built(profile: str, *, emit: Optional[Callable[[str], None]] = None,
                 run: Callable[..., "subprocess.CompletedProcess"] = subprocess.run) -> Tuple[Path, Path]:
    """Build (incrementally) this checkout's env core for ``profile``; return ``(child, cdylib)``.

    Raises :class:`EnvCoreBuildError` — naming the command — when the crate, cargo, the build or an
    artifact is missing. Never falls back to another checkout's build or to `the deleted Python env core`."""
    crate = crate_dir()
    argv = build_argv(profile)
    if not (crate / "Cargo.toml").is_file():
        raise EnvCoreBuildError(f"the Rust env core: no Rust env crate at {crate} (no Cargo.toml)")
    cargo = _cargo()
    if cargo is None:
        raise EnvCoreBuildError("the Rust env core: cargo not found (PATH or ~/.cargo/bin) — install rustup, "
                                f"or build it yourself: {build_command(profile)}")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    t0 = time.perf_counter()
    try:
        r = run([cargo, *argv], cwd=str(crate), env=env, capture_output=True, text=True, timeout=BUILD_TIMEOUT_S)
    except subprocess.TimeoutExpired as e:
        raise EnvCoreBuildError(f"the Rust env core: `cargo {' '.join(argv)}` did not finish in "
                                f"{BUILD_TIMEOUT_S:.0f} s") from e
    if r.returncode != 0:
        tail = (r.stderr or r.stdout or "").strip()[-3000:]
        raise EnvCoreBuildError(f"the Rust env core: `cargo {' '.join(argv)}` failed (exit {r.returncode}) in "
                                f"{crate}:\n{tail}")
    child, lib = artifacts(profile)
    missing = [str(p) for p in (child, lib) if not p.is_file()]
    if missing:
        raise EnvCoreBuildError(f"the Rust env core: cargo succeeded but {missing} is missing "
                                f"(CARGO_TARGET_DIR={env['CARGO_TARGET_DIR']})")
    if emit is not None:
        emit(f"🦀 [ENV CORE BUILD] {profile} — this checkout's src/rust_env built in "
             f"{time.perf_counter() - t0:.1f} s ({child.parent})")
    return child, lib
