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


def _cargo() -> str:
    cargo = shutil.which("cargo") or os.path.expanduser("~/.cargo/bin/cargo")
    if not os.path.exists(cargo):
        pytest.fail("cargo is not installed — the rust env core's gates cannot run (install rustup)")
    return cargo


def _cargo_test(extra_env=None, args=()):
    """Build THIS checkout's self-check ``sim_bridge`` (into the port's own target, never main's),
    then ``cargo test`` the env crate against it. Returns the completed process."""
    cargo = _cargo()
    port = src_path("rust_sim")
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
    env.update(extra_env or {})
    return subprocess.run(
        [cargo, "test", "--profile", "selfcheck", "--features", "emission-selfcheck",
         "--manifest-path", str(crate / "Cargo.toml"), *args, "--", "--show-output"],
        env=env, capture_output=True, text=True, timeout=3600,
    )


def test_the_rust_env_core_suite_passes():
    r = _cargo_test()
    tail = (r.stdout + r.stderr)[-6000:]
    assert r.returncode == 0, f"cargo test failed:\n{tail}"
    # Non-vacuity: the gates actually ran (a filtered or empty run passes too).
    for name in ("seed_to_bytes_is_invariant_under_thread_count_and_rerun",
                 "a_refused_battle_is_quarantined_banked_and_replayable",
                 "the_lifecycle_is_enforced_and_typed",
                 "the_cores_rows_are_sim_bridges_obs_rows_byte_for_byte",
                 "the_byte_comparison_has_teeth",
                 "quarantines_are_thread_count_invariant",
                 "the_stamp_is_well_formed_and_names_this_schema",
                 # M5 Lane I (hand-off): the playout's battle is the core's row path; playouts are CRN-deterministic
                 "a_game_is_the_env_cores_row_path_and_its_branch_is_exact",
                 "playouts_are_deterministic_with_common_random_numbers_and_the_recorded_continuation_is_reproduced"):
        assert f"{name} ... ok" in r.stdout, f"{name} did not run:\n{tail}"
    # Gate ⑤, cross-language: the Rust build's stamp is what this tree recomputes (a self-check build).
    lines = [ln for ln in r.stdout.splitlines() if ln.startswith("POKESIM_ENV_STAMP=")]
    assert lines, f"the build did not print its stamp:\n{tail}"
    stamp.check_stamp(lines[0].split("=", 1)[1], "the rust env self-check build", nan_poison=True)


def _gate_1(tmp_path, name: str, teams, n: int, steps: int, seed: int) -> str:
    """Gate ① over ``teams`` (packed strings): the core's rows == ``sim_bridge``'s ``__OBS__``."""
    f = tmp_path / f"{name}.txt"
    f.write_text("\n".join(teams) + "\n")
    r = _cargo_test(
        {"RUST_ENV_PARITY_TEAMS": str(f), "RUST_ENV_PARITY_N": str(n), "RUST_ENV_PARITY_STEPS": str(steps),
         "RUST_ENV_PARITY_SEED": str(seed)},
        ("--test", "sim_bridge_parity_test", "the_cores_rows_are_sim_bridges_obs_rows_byte_for_byte"),
    )
    out = r.stdout + r.stderr
    assert r.returncode == 0, f"gate 1 on {name} failed:\n{out[-6000:]}"
    assert "the_cores_rows_are_sim_bridges_obs_rows_byte_for_byte ... ok" in r.stdout, out[-3000:]
    line = next(ln for ln in out.splitlines() if ln.startswith("gate 1:"))
    assert f"{len(teams)} teams" in line, line
    return line


def test_gate_1_on_the_ladder_commit_tier(tmp_path):
    """COMMIT tier: the 16-team ladder slice (owner 2026-09-24: the Metamon ladder-usage corpus joins
    every parity gate), 8 envs x 600 steps (~5 s warm)."""
    from utils.ladder_corpus import teams

    _gate_1(tmp_path, "ladder_commit", teams("commit"), 8, 600, 11)


@pytest.mark.slow
def test_gate_1_on_the_ladder_milestone_tier(tmp_path):
    """MILESTONE tier: the 800-team ladder tier, 16 envs x 4000 steps (~90 s warm; 2026-09-28:
    121,463 frames byte-equal over 717 logs)."""
    from utils.ladder_corpus import teams

    _gate_1(tmp_path, "ladder_milestone", teams("milestone"), 16, 4000, 24)


@pytest.mark.slow
def test_gate_1_on_the_training_pool(tmp_path):
    """MILESTONE tier: the 719 training-pool teams, packed exactly as training packs them
    (``Gen3Teambuilder``), 16 envs x 3000 steps (~40 s warm; 2026-09-28: 90,556 frames over 641 logs)."""
    from utils.team_sources import team_list
    from utils.teambuilder import Gen3Teambuilder

    _gate_1(tmp_path, "pool", Gen3Teambuilder(team_list("pool")).packed_teams, 16, 3000, 7)


@pytest.mark.slow
def test_gate_1_on_procedural_teams(tmp_path):
    """MILESTONE tier: 200 PROCEDURAL teams (``ou_random_teams.js``, Smogon-derived, TeamValidator-legal,
    seeded), 16 envs x 3000 steps (~35 s warm; 2026-09-28: 90,844 frames over 557 logs)."""
    from utils.team_sources import team_list

    _gate_1(tmp_path, "procedural", team_list("procedural", n=200), 16, 3000, 7)
