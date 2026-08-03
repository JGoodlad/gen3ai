"""Resolve the spawn command for the in-process sim bridge — Node or Rust.

The two transport seams (``bridge_session.py`` for training, ``local_battle_runner.py``
for eval) both spawn a subprocess that speaks the ``local_sim_bridge.js`` stdin/stdout
protocol. By default that subprocess is ``node local_sim_bridge.js``; ``--use-bridge=rust``
swaps in the std-only Rust binary ``src/rust_sim/src/bin/sim_bridge.rs``, which is
byte-for-byte protocol-compatible (validated by ``harness/gen_sim_bridge_diff.js``).

This module owns the ONE place that turns an impl name (``"node"`` / ``"rust"``) into the
argv list the spawners exec, and — for ``rust`` — the binary-resolution + build logic:

- ``POKESIM_SIM_BRIDGE_BIN`` (absolute path) is honored FIRST — no build, just use it. This
  is the escape hatch for a pre-built binary (and how the wiring verification points at an
  isolated ``CARGO_TARGET_DIR`` without touching the shared ``src/rust_sim/target/``).
- else ``cargo build --release --bin sim_bridge`` is run in ``src/rust_sim`` and the
  resulting ``target/release/sim_bridge`` is returned.
- the resolved path is cached process-wide (the build is idempotent + not free), and any
  failure raises a CLEAR, actionable error naming the fix — we NEVER silently fall back to
  Node (a rust run that quietly became a node run would corrupt an A/B).

Scope note: **both** ``__RECON__`` (``gen3_bridge_recon_record_v1``) and ``resumeReseed``
(``gen3_bridge_resume_reseed_v1``) are supported, including on a SEEDLESS ``START`` — the
production case — since ``gen3_bridge_seedless_fixed_seed_v1`` made the child MINT and report a
real seed instead of silently reusing a constant (before that fix a rust eval wrote no
``*_reconstruction.json`` at all). The record's ``>start``/``>player`` lines are exact; its
committed-choice lines are replay-EQUIVALENT rather than byte-identical to the sim's own
``inputLog``, so the search TEACHER still needs ``--use-bridge=node``. The CLI warns once at
startup when ``rust`` is selected (see ``train_rl_agent``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

_BRIDGE_JS = str(Path(__file__).parent / "local_sim_bridge.js")

# The Rust crate root — resolved relative to THIS file, so it is correct inside a git
# worktree (where the crate lives at <worktree>/src/rust_sim, not the main checkout).
# __file__ = <root>/src/utils/bridge/sim_bridge_bin.py → parents[2] = <root>/src.
_RUST_CRATE_DIR = Path(__file__).resolve().parents[2] / "rust_sim"
_ENV_OVERRIDE = "POKESIM_SIM_BRIDGE_BIN"

# Cache the resolved rust binary path across the process (the build is idempotent, but not
# free, and both spawners hit this on every child spawn). Guarded so concurrent env workers
# racing the first spawn don't launch parallel cargo builds.
_rust_bin_cache: Optional[str] = None
_rust_bin_lock = threading.Lock()

VALID_IMPLS = ("node", "rust")


class SimBridgeBinaryError(RuntimeError):
    """The Rust sim bridge binary could not be resolved/built — actionable message attached."""


def resolve_sim_bridge_bin() -> str:
    """Return an absolute path to the Rust ``sim_bridge`` binary, building it if needed.

    Resolution order: ``$POKESIM_SIM_BRIDGE_BIN`` (absolute, must exist) → cached previous
    resolution → ``cargo build --release --bin sim_bridge`` in ``src/rust_sim``. Raises
    ``SimBridgeBinaryError`` with a clear fix instruction on any failure (missing cargo,
    missing crate, build error, missing artifact). NEVER falls back to Node.
    """
    global _rust_bin_cache

    override = os.environ.get(_ENV_OVERRIDE)
    if override:
        p = Path(override)
        if not p.is_file():
            raise SimBridgeBinaryError(
                f"{_ENV_OVERRIDE}={override!r} does not point at an existing file. "
                f"Set it to the absolute path of a built `sim_bridge` binary, or unset it "
                f"to build from {_RUST_CRATE_DIR}."
            )
        return str(p.resolve())

    with _rust_bin_lock:
        if _rust_bin_cache is not None:
            return _rust_bin_cache

        if not (_RUST_CRATE_DIR / "Cargo.toml").is_file():
            raise SimBridgeBinaryError(
                f"Rust sim crate not found at {_RUST_CRATE_DIR} (no Cargo.toml). "
                f"--use-bridge=rust needs the src/rust_sim crate. Either check it out, or "
                f"set {_ENV_OVERRIDE} to a pre-built sim_bridge binary."
            )

        cargo = _which_cargo()
        if cargo is None:
            raise SimBridgeBinaryError(
                "cargo not found on PATH — --use-bridge=rust needs the Rust toolchain to "
                "build src/rust_sim. Install rustup (https://rustup.rs) and ensure "
                "~/.cargo/bin is on PATH (e.g. `export PATH=\"$HOME/.cargo/bin:$PATH\"`), or "
                f"set {_ENV_OVERRIDE} to a pre-built sim_bridge binary."
            )

        try:
            proc = subprocess.run(
                [cargo, "build", "--release", "--bin", "sim_bridge"],
                cwd=str(_RUST_CRATE_DIR),
                capture_output=True,
                text=True,
            )
        except OSError as e:  # pragma: no cover - cargo present but unexecutable
            raise SimBridgeBinaryError(
                f"failed to invoke `{cargo} build --release --bin sim_bridge` in "
                f"{_RUST_CRATE_DIR}: {e}"
            ) from e
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise SimBridgeBinaryError(
                f"`cargo build --release --bin sim_bridge` failed in {_RUST_CRATE_DIR} "
                f"(exit {proc.returncode}). Fix the build (or set {_ENV_OVERRIDE} to a "
                f"pre-built binary). cargo output tail:\n{tail}"
            )

        # Honor a custom CARGO_TARGET_DIR (the verification path points it at an isolated
        # dir so it never rebuilds the shared src/rust_sim/target/ a live A/B fuzzer execs).
        target_dir = os.environ.get("CARGO_TARGET_DIR")
        bin_path = (
            Path(target_dir) / "release" / "sim_bridge"
            if target_dir
            else _RUST_CRATE_DIR / "target" / "release" / "sim_bridge"
        )
        if not bin_path.is_file():
            raise SimBridgeBinaryError(
                f"cargo build succeeded but the sim_bridge binary is missing at {bin_path}. "
                f"(CARGO_TARGET_DIR={target_dir!r}.)"
            )
        _rust_bin_cache = str(bin_path.resolve())
        return _rust_bin_cache


def resolve_and_publish_sim_bridge_bin() -> str:
    """Resolve the Rust binary ONCE and publish it to the env for every child process.

    The cache in ``resolve_sim_bridge_bin`` is per-PROCESS, but the bridge spawners run in
    the ``SubprocVecEnv`` env workers and the eval-worker subprocesses — each a fresh
    process with a cold cache. Without this, every one of them independently runs
    ``cargo build`` on its first spawn (at ``--n-envs 64`` that is 64 builds contending on
    cargo's target-dir lock, turning startup into a thundering herd).

    Publishing the resolved path into ``POKESIM_SIM_BRIDGE_BIN`` — which
    ``resolve_sim_bridge_bin`` honors FIRST, with no build — makes every inheriting child a
    pure path lookup. Idempotent: if the var was already set, we re-publish the same value.
    """
    path = resolve_sim_bridge_bin()
    os.environ[_ENV_OVERRIDE] = path
    return path


def _which_cargo() -> Optional[str]:
    """Find cargo on PATH, also probing the standard ~/.cargo/bin location."""
    from shutil import which

    found = which("cargo")
    if found:
        return found
    fallback = Path.home() / ".cargo" / "bin" / "cargo"
    return str(fallback) if fallback.is_file() else None


def bridge_spawn_argv(impl: str) -> List[str]:
    """Return the argv list the bridge spawners exec for ``impl`` (``"node"`` | ``"rust"``).

    ``node`` → ``["node", <local_sim_bridge.js>]`` (the current default behavior, unchanged).
    ``rust`` → ``[<resolved sim_bridge binary>]`` (built/resolved via ``resolve_sim_bridge_bin``).
    Both children speak the identical stdin/stdout protocol, so the callers' framing/demux is
    unchanged — only the executable differs.
    """
    if impl == "node":
        return ["node", _BRIDGE_JS]
    if impl == "rust":
        return [resolve_sim_bridge_bin()]
    raise ValueError(f"unknown bridge impl {impl!r}; expected one of {VALID_IMPLS}")


def rust_deferral_warning() -> str:
    """The one-time startup warning naming the Rust bridge's honest deferrals."""
    return (
        "ℹ️  [BRIDGE=rust] __RECON__ and resumeReseed are both SUPPORTED "
        "(gen3_bridge_recon_record_v1 / gen3_bridge_resume_reseed_v1), so the forensic "
        "reconstruction and counterfactual Monte-Carlo paths work on rust. Honest scope: the "
        "record's >start/>player lines are exact (the only part any consumer reads), while its "
        "COMMITTED-CHOICE lines are rendered from the engine's own script — replay-equivalent, "
        "not guaranteed byte-identical to the sim's inputLog normalization."
    )


def warn_rust_deferrals(emit=None) -> None:
    """Emit the deferral warning once (via ``emit`` if given, else stderr)."""
    msg = rust_deferral_warning()
    if emit is not None:
        emit(msg)
    else:  # pragma: no cover - trivial
        sys.stderr.write(msg + "\n")
