"""Run the Rust env core's own test suite from pytest, so its gates ride the routine tier (M5 Lane 0).

``cargo test`` over ``src/rust_env`` in the EMISSION SELF-CHECK build (optimized, the port's
per-emission check on, NaN-prefilled rows) — the build every pytest session and fuzzer runs. It
carries gate ② (``tests/determinism_test.rs``: seed → bytes, thread-count-invariant), gate ④ and the
declared lifecycle (``tests/lifecycle_refusal_test.rs``) and the unit tests (the spec, the bank, the
generated contract).

The build goes to THIS checkout's ``src/rust_env/target`` (never main's; the 09-09 incident), so a
fresh worktree pays one compile of the port on the first run.
"""
import os
import shutil
import subprocess

import pytest

from utils.paths import src_path

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def test_the_rust_env_core_suite_passes():
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the rust env core's gates cannot run (install rustup)")
    crate = src_path("rust_env")
    env = dict(os.environ, CARGO_TARGET_DIR=str(crate / "target"))
    r = subprocess.run(
        [cargo, "test", "--profile", "selfcheck", "--features", "emission-selfcheck",
         "--manifest-path", str(crate / "Cargo.toml")],
        env=env, capture_output=True, text=True, timeout=1800,
    )
    tail = (r.stdout + r.stderr)[-6000:]
    assert r.returncode == 0, f"cargo test failed:\n{tail}"
    # Non-vacuity: the gates actually ran (a filtered or empty run passes too).
    for name in ("seed_to_bytes_is_invariant_under_thread_count_and_rerun",
                 "a_refused_battle_is_quarantined_banked_and_replayable",
                 "the_lifecycle_is_enforced_and_typed"):
        assert f"test {name} ... ok" in r.stdout, f"{name} did not run:\n{tail}"
