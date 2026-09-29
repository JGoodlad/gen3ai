"""Run the Rust env core's own test suite from pytest, so its gates ride the routine tier (M5 Lane 0).

``cargo test`` over ``src/rust_env`` in the EMISSION SELF-CHECK build (optimized, the port's
per-emission check on, NaN-prefilled rows) — the build every pytest session and fuzzer runs. It
carries gate ② (``tests/determinism_test.rs``: seed → bytes, thread-count-invariant), gate ④ and the
declared lifecycle (``tests/lifecycle_refusal_test.rs``), gate ① (``tests/sim_bridge_parity_test.rs``:
the core's rows == THIS checkout's ``sim_bridge`` ``__OBS__`` rows byte for byte on the same input
logs — the port's self-check ``sim_bridge`` is built here first and handed over as
``POKESIM_SIM_BRIDGE_BIN``) and the unit tests (the spec, the bank, the generated contract). Gate ⑤'s
cross-language half: the build prints its STAMP and ``stamp.check_stamp`` must accept it.

The build goes to THIS checkout's ``src/rust_env/target`` (never main's; the 09-09 incident), so a
fresh worktree pays one compile of the port on the first run.
"""
import os
import shutil
import subprocess

import pytest

from utils.paths import src_path
from utils.rust_env import stamp

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def test_the_rust_env_core_suite_passes():
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the rust env core's gates cannot run (install rustup)")
    port = src_path("rust_sim")
    # The port's self-check sim_bridge, built into THIS checkout's port target (never main's).
    b = subprocess.run(
        [cargo, "build", "--profile", "selfcheck", "--features", "emission-selfcheck", "--bin", "sim_bridge",
         "--manifest-path", str(port / "Cargo.toml")],
        env=dict(os.environ, CARGO_TARGET_DIR=str(port / "target")), capture_output=True, text=True, timeout=1800,
    )
    assert b.returncode == 0, f"building sim_bridge failed:\n{b.stderr[-4000:]}"
    bridge = port / "target" / "selfcheck" / "sim_bridge"
    assert bridge.exists(), bridge
    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"), POKESIM_SIM_BRIDGE_BIN=str(bridge))
    r = subprocess.run(
        [cargo, "test", "--profile", "selfcheck", "--features", "emission-selfcheck",
         "--manifest-path", str(crate / "Cargo.toml"), "--", "--show-output"],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    tail = (r.stdout + r.stderr)[-6000:]
    assert r.returncode == 0, f"cargo test failed:\n{tail}"
    # Non-vacuity: the gates actually ran (a filtered or empty run passes too).
    for name in ("seed_to_bytes_is_invariant_under_thread_count_and_rerun",
                 "a_refused_battle_is_quarantined_banked_and_replayable",
                 "the_lifecycle_is_enforced_and_typed",
                 "the_cores_rows_are_sim_bridges_obs_rows_byte_for_byte",
                 "the_byte_comparison_has_teeth",
                 "the_stamp_is_well_formed_and_names_this_schema"):
        assert f"{name} ... ok" in r.stdout, f"{name} did not run:\n{tail}"
    # Gate ⑤, cross-language: the Rust build's stamp is what this tree recomputes (a self-check build).
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("POKESIM_ENV_STAMP=")]
    assert lines, f"the build did not print its stamp:\n{tail}"
    stamp.check_stamp(lines[0].split("=", 1)[1], "the rust env self-check build", nan_poison=True)
