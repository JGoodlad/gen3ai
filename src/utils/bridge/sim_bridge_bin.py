"""Resolve the spawn command for the in-process sim bridge AND the offline search/replay
drivers — Node or Rust.

**Two child families, one seam.** Both are stdin/stdout JSON-line subprocesses, and both are
impl-selectable through this module:

1. **The live battle transport.** The two transport seams (``bridge_session.py`` for training,
   ``local_battle_runner.py`` for eval) spawn a child speaking the ``local_sim_bridge.js``
   protocol. ``node`` → ``node local_sim_bridge.js``; ``rust`` → the std-only Rust binary
   ``src/rust_sim/src/bin/sim_bridge.rs``, byte-for-byte protocol-compatible (validated by
   ``harness/gen_sim_bridge_diff.js``). Selected by ``--use-bridge={node,rust}``.
2. **The offline search / replay drivers.** ``search_session.py`` (the warm clone-and-branch
   search server) and ``reconstruction.py`` (``replay_battle`` / ``reroll_turn`` /
   ``reroll_many``) drive a child over a recorded ``ReconstructionRecord``. On **node** those
   are TWO scripts — ``search_driver.js`` and ``replay_driver.js``; on **rust** ONE binary,
   ``src/rust_sim/src/bin/search_driver.rs``, serves both verb families. Selected by the
   ``impl=`` parameter those callers thread (CLI: ``--impl``).

This module owns the ONE place that turns an impl name (``"node"`` / ``"rust"``) into the argv
list a spawner execs, and — for ``rust`` — the binary-resolution + build logic (shared by both
families via ``_resolve_rust_bin``):

- an env override (``POKESIM_SIM_BRIDGE_BIN`` / ``POKESIM_SEARCH_DRIVER_BIN``, absolute path)
  is honored FIRST — no build, just use it. This is the escape hatch for a pre-built binary
  (and how the wiring verification points at an isolated ``CARGO_TARGET_DIR`` without touching
  the shared ``src/rust_sim/target/``).
- else ``cargo build --release --bin <name>`` is run in ``src/rust_sim`` and the resulting
  ``target/release/<name>`` is returned.
- the resolved path is cached process-wide per binary (the build is idempotent + not free, and
  the spawners hit this on every child spawn), and any failure raises a CLEAR, actionable error
  naming the fix — we NEVER silently fall back to Node (a rust run that quietly became a node
  run would corrupt an A/B).

Scope note on the rust bridge: **both** ``__RECON__`` (``gen3_bridge_recon_record_v1``) and
``resumeReseed`` (``gen3_bridge_resume_reseed_v1``) are supported, including on a SEEDLESS
``START`` — the production case — since ``gen3_bridge_seedless_fixed_seed_v1`` made the child
MINT and report a real seed instead of silently reusing a constant (before that fix a rust eval
wrote no ``*_reconstruction.json`` at all). The offline drivers are on rust as well now
(``gen3_rust_search_driver_v1`` + ``gen3_rust_replay_driver_v1``), so the SEARCH TEACHER no longer
requires node either.

A note worth keeping, because it cost an investigation: the search-teacher's old node requirement
was long attributed to the rust record's ``input_log`` being replay-equivalent rather than
byte-identical. That was **false**. No consumer reads the record's committed-choice lines at all —
``replay_kernels.js::writeStart`` reads only ``>start`` + ``>player``, exactly as
``ReconstructionRecord.start_options()``/``players()`` do, and the rust record renders those
exactly. The real blocker was always the missing DRIVER. Do not re-derive a plan from the old
reason. The CLI warns once at startup when ``rust`` is selected (see ``train_rl_agent``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

_BRIDGE_JS = str(Path(__file__).parent / "local_sim_bridge.js")
_SEARCH_DRIVER_JS = str(Path(__file__).parent / "search_driver.js")

# The Rust crate root — resolved relative to THIS file, so it is correct inside a git
# worktree (where the crate lives at <worktree>/src/rust_sim, not the main checkout).
# __file__ = <root>/src/utils/bridge/sim_bridge_bin.py → parents[2] = <root>/src.
_RUST_CRATE_DIR = Path(__file__).resolve().parents[2] / "rust_sim"
_ENV_OVERRIDE = "POKESIM_SIM_BRIDGE_BIN"
_SEARCH_ENV_OVERRIDE = "POKESIM_SEARCH_DRIVER_BIN"

# Cache the resolved rust binary paths across the process, keyed by cargo bin name (the build
# is idempotent, but not free, and the spawners hit this on every child spawn). Guarded so
# concurrent env workers racing the first spawn don't launch parallel cargo builds.
_rust_bin_cache: Dict[str, str] = {}
_rust_bin_lock = threading.Lock()

VALID_IMPLS = ("node", "rust")


class SimBridgeBinaryError(RuntimeError):
    """A Rust binary (sim bridge / search driver) could not be resolved/built — actionable
    message attached."""


def _resolve_rust_bin(bin_name: str, env_var: str, selector: str) -> str:
    """Resolve (building if needed) ``src/rust_sim``'s ``bin_name`` release binary.

    The ONE cargo-invocation path, shared by every rust child family — ``sim_bridge`` (the live
    transport) and ``search_driver`` (the offline search/replay driver). ``env_var`` is the
    absolute-path override honored first; ``selector`` is the user-facing flag quoted in every
    error message so the fix instruction names the thing the caller actually typed.

    Resolution order: ``$<env_var>`` (absolute, must exist) → cached previous resolution →
    ``cargo build --release --bin <bin_name>``. Raises ``SimBridgeBinaryError`` with a clear fix
    instruction on any failure (missing cargo, missing crate, build error, missing artifact).
    NEVER falls back to Node.
    """
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if not p.is_file():
            raise SimBridgeBinaryError(
                f"{env_var}={override!r} does not point at an existing file. "
                f"Set it to the absolute path of a built `{bin_name}` binary, or unset it "
                f"to build from {_RUST_CRATE_DIR}."
            )
        return str(p.resolve())

    with _rust_bin_lock:
        cached = _rust_bin_cache.get(bin_name)
        if cached is not None:
            return cached

        if not (_RUST_CRATE_DIR / "Cargo.toml").is_file():
            raise SimBridgeBinaryError(
                f"Rust sim crate not found at {_RUST_CRATE_DIR} (no Cargo.toml). "
                f"{selector} needs the src/rust_sim crate. Either check it out, or "
                f"set {env_var} to a pre-built {bin_name} binary."
            )

        cargo = _which_cargo()
        if cargo is None:
            raise SimBridgeBinaryError(
                f"cargo not found on PATH — {selector} needs the Rust toolchain to "
                "build src/rust_sim. Install rustup (https://rustup.rs) and ensure "
                "~/.cargo/bin is on PATH (e.g. `export PATH=\"$HOME/.cargo/bin:$PATH\"`), or "
                f"set {env_var} to a pre-built {bin_name} binary."
            )

        try:
            proc = subprocess.run(
                [cargo, "build", "--release", "--bin", bin_name],
                cwd=str(_RUST_CRATE_DIR),
                capture_output=True,
                text=True,
            )
        except OSError as e:  # pragma: no cover - cargo present but unexecutable
            raise SimBridgeBinaryError(
                f"failed to invoke `{cargo} build --release --bin {bin_name}` in "
                f"{_RUST_CRATE_DIR}: {e}"
            ) from e
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            raise SimBridgeBinaryError(
                f"`cargo build --release --bin {bin_name}` failed in {_RUST_CRATE_DIR} "
                f"(exit {proc.returncode}). Fix the build (or set {env_var} to a "
                f"pre-built binary). cargo output tail:\n{tail}"
            )

        # Honor a custom CARGO_TARGET_DIR (the verification path points it at an isolated
        # dir so it never rebuilds the shared src/rust_sim/target/ a live A/B fuzzer execs).
        target_dir = os.environ.get("CARGO_TARGET_DIR")
        bin_path = (
            Path(target_dir) / "release" / bin_name
            if target_dir
            else _RUST_CRATE_DIR / "target" / "release" / bin_name
        )
        if not bin_path.is_file():
            raise SimBridgeBinaryError(
                f"cargo build succeeded but the {bin_name} binary is missing at {bin_path}. "
                f"(CARGO_TARGET_DIR={target_dir!r}.)"
            )
        resolved = str(bin_path.resolve())
        _rust_bin_cache[bin_name] = resolved
        return resolved


def resolve_sim_bridge_bin() -> str:
    """Return an absolute path to the Rust ``sim_bridge`` binary, building it if needed.

    Resolution order: ``$POKESIM_SIM_BRIDGE_BIN`` (absolute, must exist) → cached previous
    resolution → ``cargo build --release --bin sim_bridge`` in ``src/rust_sim``. Raises
    ``SimBridgeBinaryError`` with a clear fix instruction on any failure (missing cargo,
    missing crate, build error, missing artifact). NEVER falls back to Node.
    """
    return _resolve_rust_bin("sim_bridge", _ENV_OVERRIDE, "--use-bridge=rust")


def resolve_search_driver_bin() -> str:
    """Return an absolute path to the Rust ``search_driver`` binary, building it if needed.

    The exact mirror of :func:`resolve_sim_bridge_bin` for the OFFLINE driver family: the
    clone-and-branch search server (``search_session.py``) and the replay/re-roll primitives
    (``reconstruction.py``). Node splits those across two scripts (``search_driver.js`` +
    ``replay_driver.js``); the Rust port serves both verb families from this ONE binary.

    Resolution order: ``$POKESIM_SEARCH_DRIVER_BIN`` (absolute, must exist) → cached previous
    resolution → ``cargo build --release --bin search_driver`` in ``src/rust_sim``. Raises
    ``SimBridgeBinaryError`` with a clear fix instruction on any failure. NEVER falls back to
    Node — an ``impl="rust"`` search that quietly ran on node would silently answer a different
    question than the one asked.
    """
    return _resolve_rust_bin("search_driver", _SEARCH_ENV_OVERRIDE, "--impl rust")


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


def search_driver_spawn_argv(impl: str) -> List[str]:
    """Return the argv the OFFLINE search/replay drivers exec for ``impl`` (``"node"`` | ``"rust"``).

    ``node`` → ``["node", <search_driver.js>]`` (the current default behavior, unchanged).
    ``rust`` → ``[<resolved search_driver binary>]`` (built/resolved via
    :func:`resolve_search_driver_bin`).

    NOTE the asymmetry with node: node splits the offline verbs across ``search_driver.js``
    (open_root / expand_many) and ``replay_driver.js`` (replay / reroll / reroll_many), while the
    Rust port serves BOTH from the single ``search_driver`` binary — so ``reconstruction.py``
    routes through here too rather than growing a second resolver.
    """
    if impl == "node":
        return ["node", _SEARCH_DRIVER_JS]
    if impl == "rust":
        return [resolve_search_driver_bin()]
    raise ValueError(f"unknown search-driver impl {impl!r}; expected one of {VALID_IMPLS}")


def rust_deferral_warning() -> str:
    """The one-time startup warning naming the Rust bridge's honest deferrals."""
    return (
        "ℹ️  [BRIDGE=rust] __RECON__ and resumeReseed are SUPPORTED "
        "(gen3_bridge_recon_record_v1 / gen3_bridge_resume_reseed_v1), and so are the OFFLINE "
        "search/replay drivers (gen3_rust_search_driver_v1 / gen3_rust_replay_driver_v1 — one "
        "search_driver binary serves open_root/expand_many AND replay/reroll/reroll_many), so "
        "better-line / lookahead / falsify / the SEARCH TEACHER all run on rust. Select the "
        "driver impl with --impl / POKESIM_SEARCH_DRIVER_BIN. Honest scope, two known gaps: "
        "(1) a CHOICE-REJECT emits no |error| frame and re-opens the boundary to BOTH sides "
        "(node re-asks only the offending side) — a pre-existing bridge.rs gap on a path poke-env "
        "never takes, allowlisted in the parity harnesses and reconciled only when every log "
        "chunk is byte-equal; (2) pre_state volatile NAMES are reconstructed from typed fields "
        "and only the duration-1 exclusion is gated — pre_state has no consumer today."
    )


def warn_rust_deferrals(emit=None) -> None:
    """Emit the deferral warning once (via ``emit`` if given, else stderr)."""
    msg = rust_deferral_warning()
    if emit is not None:
        emit(msg)
    else:  # pragma: no cover - trivial
        sys.stderr.write(msg + "\n")
