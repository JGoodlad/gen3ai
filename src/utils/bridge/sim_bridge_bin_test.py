"""Unit tests for the sim-bridge / search-driver binary resolvers (pure — no cargo build,
no subprocess)."""

import os
from pathlib import Path

import pytest

from utils.contention import scale_timeout

from utils.bridge import sim_bridge_bin
from utils.bridge.sim_bridge_bin import (
    SimBridgeBinaryError,
    VALID_IMPLS,
    bridge_spawn_argv,
    rust_deferral_warning,
    search_driver_spawn_argv,
)


@pytest.fixture(autouse=True)
def _cold_bin_cache(monkeypatch):
    """Every test starts with a COLD resolver cache. The cache is process-global and keyed by
    cargo bin name, so a real resolution earlier in the session would otherwise mask an env
    override (and the rust-binary integration test below populates it for real)."""
    monkeypatch.setattr(sim_bridge_bin, "_rust_bin_cache", {})


def test_node_argv_is_node_plus_the_js():
    argv = bridge_spawn_argv("node")
    assert argv[0] == "node"
    assert argv[1].endswith("local_sim_bridge.js")


def test_rust_argv_is_the_resolved_binary_via_env_override(tmp_path, monkeypatch):
    # A pre-built binary via POKESIM_SIM_BRIDGE_BIN is honored first — no cargo build.
    fake = tmp_path / "sim_bridge"
    fake.write_text("#!/bin/true\n")
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", str(fake))
    argv = bridge_spawn_argv("rust")
    assert argv == [str(fake.resolve())]  # NO "node" — just the binary


def test_env_override_missing_file_raises_clear_error(monkeypatch):
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", "/nonexistent/sim_bridge_xyz")
    with pytest.raises(SimBridgeBinaryError) as ei:
        bridge_spawn_argv("rust")
    assert "POKESIM_SIM_BRIDGE_BIN" in str(ei.value)


def test_unknown_impl_rejected():
    with pytest.raises(ValueError) as ei:
        bridge_spawn_argv("bogus")
    assert "bogus" in str(ei.value)
    assert VALID_IMPLS == ("node", "rust")


# ---------------------------------------------------------------------------
# The OFFLINE search/replay driver family — the exact mirror of the four above.
# ---------------------------------------------------------------------------


def test_search_driver_node_argv_is_node_plus_the_js():
    """`impl="node"` must keep producing the EXACT argv the drivers used before the seam
    existed — this is the byte-identical-default guarantee every current caller relies on."""
    argv = search_driver_spawn_argv("node")
    assert argv[0] == "node"
    assert argv[1].endswith("search_driver.js")
    assert Path(argv[1]).is_file(), "the node driver script must actually exist"
    assert len(argv) == 2


def test_search_driver_rust_argv_is_the_resolved_binary_via_env_override(tmp_path, monkeypatch):
    fake = tmp_path / "search_driver"
    fake.write_text("#!/bin/true\n")
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", str(fake))
    argv = search_driver_spawn_argv("rust")
    assert argv == [str(fake.resolve())]  # NO "node" — just the binary


def test_search_driver_env_override_missing_file_raises_clear_error(monkeypatch):
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", "/nonexistent/search_driver_xyz")
    with pytest.raises(SimBridgeBinaryError) as ei:
        search_driver_spawn_argv("rust")
    msg = str(ei.value)
    assert "POKESIM_SEARCH_DRIVER_BIN" in msg
    assert "search_driver" in msg


def test_search_driver_unknown_impl_rejected():
    with pytest.raises(ValueError) as ei:
        search_driver_spawn_argv("bogus")
    assert "bogus" in str(ei.value)


def test_search_driver_env_overrides_are_independent(tmp_path, monkeypatch):
    """The two families must not share an override. A single env var (or a single cache slot)
    would silently hand the search driver the sim_bridge binary — a wrong-protocol child that
    fails far from the cause."""
    bridge = tmp_path / "sim_bridge"
    bridge.write_text("#!/bin/true\n")
    driver = tmp_path / "search_driver"
    driver.write_text("#!/bin/true\n")
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", str(bridge))
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", str(driver))
    assert bridge_spawn_argv("rust") == [str(bridge.resolve())]
    assert search_driver_spawn_argv("rust") == [str(driver.resolve())]


def test_missing_rust_search_driver_raises_instead_of_falling_back_to_node(tmp_path, monkeypatch):
    """THE no-silent-fallback contract. A rust search that quietly ran on node would answer a
    different question than the one asked (and would make an engine A/B meaningless), so an
    unresolvable binary must RAISE — never return a node argv.

    Simulated by pointing the crate dir at an empty tmp dir (no Cargo.toml), the same shape as a
    checkout without src/rust_sim; the assertion that matters is 'not a node argv'.
    """
    monkeypatch.delenv("POKESIM_SEARCH_DRIVER_BIN", raising=False)
    monkeypatch.setattr(sim_bridge_bin, "_RUST_CRATE_DIR", tmp_path / "no_crate_here")
    with pytest.raises(SimBridgeBinaryError) as ei:
        search_driver_spawn_argv("rust")
    msg = str(ei.value)
    assert "Cargo.toml" in msg                  # says WHAT is missing
    assert "POKESIM_SEARCH_DRIVER_BIN" in msg   # ...and the escape hatch
    # And nothing was cached, so a later call re-raises rather than serving a half-resolution.
    assert "search_driver" not in sim_bridge_bin._rust_bin_cache


def test_deferral_warning_states_the_current_rust_scope():
    """The startup warning must describe the scope that is TRUE NOW, and no wider.

    Four capabilities have shipped in sequence and the warning has trailed each one, so this
    test pins the text against the two failure modes that actually hurt an operator:

    * **Under-claiming** sends someone to ``--use-bridge=node`` for work that runs on rust.
      ``__RECON__`` (``gen3_bridge_recon_record_v1``), ``resumeReseed``
      (``gen3_bridge_resume_reseed_v1``), and now BOTH offline driver families
      (``gen3_rust_search_driver_v1`` = ``open_root``/``expand_many``,
      ``gen3_rust_replay_driver_v1`` = ``replay``/``reroll``/``reroll_many``) are supported, so
      better-line / lookahead / falsify / the search-teacher all work.
    * **Over-claiming** hides a real divergence. ONE gap remains and must stay named: the
      reconstructed ``pre_state`` volatile names.
    * **Stale-gap claiming** is the third failure mode, and it is the one that actually bit.
      The warning named the CHOICE-REJECT framing as an open gap "on a path poke-env never
      takes" long after ``gen3_choice_reject_framing_v1`` closed the framing — and that
      "never takes" was itself false: poke-env takes it, and
      ``gen3_locked_choice_never_rejected_v1`` killed two production launches there at ~8
      minutes. A warning that names a fixed gap teaches an operator to discount the warning.

    It must ALSO not resurrect the retracted reason. The warning once blamed the record's
    ``input_log`` byte-identity for the search-teacher's node requirement; that was false —
    nothing reads the committed-choice lines (``replay_kernels.js::writeStart`` and
    ``ReconstructionRecord.start_options()``/``players()`` read only ``>start``/``>player``).
    """
    msg = rust_deferral_warning()
    assert "__RECON__" in msg and "resumeReseed" in msg
    assert "SUPPORTED" in msg, "must not still claim the forensic paths are deferred"
    assert "search_driver" in msg, "must name the binary an operator selects with --impl"
    assert "gen3_rust_search_driver_v1" in msg and "gen3_rust_replay_driver_v1" in msg, (
        "both offline verb families must be named — an operator reading only 'search' would "
        "still think reroll/falsify needs node")
    # The honest-scope half: the ONE remaining divergence stays visible.
    assert "pre_state" in msg, "the reconstructed volatile-name gap must stay named"
    # ...and the retracted input_log reason must never come back as the cause.
    assert "byte-identical" not in msg
    # The CHOICE-REJECT framing gap is CLOSED (`gen3_choice_reject_framing_v1`), and this
    # warning named it as OPEN long after the fix landed — the same "an allowlist entry
    # outlives its own fix" failure the root CLAUDE.md warns about, except printed to every
    # operator at startup. Pin the retraction from both sides so it cannot come back:
    assert "never takes" not in msg, (
        "the 'a path poke-env never takes' claim is FALSIFIED — poke-env takes it, and "
        "gen3_locked_choice_never_rejected_v1 killed two production launches there")
    assert "emits no |error|" not in msg, (
        "the port DOES emit the |error| frame — bridge_choice_reject_test::"
        "a_disabled_move_is_unavailable_and_re_requests_that_side_only pins the "
        "re-request-that-side-only behaviour this text claimed was missing")


def test_rust_bridge_emits_a_parseable_recon_record(tmp_path):
    """`gen3_bridge_recon_record_v1` — the Rust bridge emits `__RECON__` and the record
    round-trips through the REAL consumer.

    The binary used to emit no record at all, so every forensic-reconstruction path forced
    ``--use-bridge=node``. This drives the actual binary end-to-end and feeds the frame to
    :class:`ReconstructionRecord`, because the only claim worth pinning is that the real
    consumer can use it — a self-consistent JSON blob that ``start_options()`` chokes on would
    pass a shallower test.

    Skipped (not failed) when the binary is unavailable: the Rust toolchain is not a hard
    dependency of the Python unit suite.
    """
    import base64
    import json
    import subprocess

    from utils.bridge.reconstruction import ReconstructionRecord
    from utils.bridge.sim_bridge_bin import resolve_sim_bridge_bin

    try:
        binary = resolve_sim_bridge_bin()
    except Exception as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"sim_bridge binary unavailable: {exc}")

    team = "Aipom||Leftovers|RunAway|return,splash|Hardy|85,85,85,85,85,85|M||||"
    start = {
        "formatid": "gen3customgame",
        "seed": [7, 11, 13, 17],
        "p1": {"name": "P1", "team": team},
        "p2": {"name": "P2", "team": team},
    }
    stdin = "\n".join(
        ["START " + json.dumps(start)]
        + ["CHOOSE p1 move 1", "CHOOSE p2 move 1"] * 6
        + ["END"]
    ) + "\n"
    proc = subprocess.run(
        [str(binary)], input=stdin, capture_output=True, text=True, timeout=scale_timeout(120)
    )
    frames = [l for l in proc.stdout.splitlines() if l.startswith("__RECON__ ")]
    assert frames, f"no __RECON__ frame emitted; stderr={proc.stderr[:400]}"

    raw = json.loads(base64.b64decode(frames[0][len("__RECON__ ") :]).decode())
    record = ReconstructionRecord.from_dict(raw)

    # The two parts every consumer actually reads. The seed is the STRING form, matching the
    # sim's own `inputLog[0]` (`JSON.stringify({formatid, seed: battle.prngSeed})`, where
    # `prngSeed` is `PRNG.startingSeed` — the constructor `join(",")`s an array). The rust
    # record used to render a bare `[7,11,13,17]` array here, an unnoticed node/rust record
    # divergence; `gen3_bridge_seedless_fixed_seed_v1` had to fix it because a MINTED
    # `sodium,<hex>` seed is not a number list and the array spelling was invalid JSON.
    assert record.start_options()["seed"] == "7,11,13,17"
    assert record.prng_seed == "7,11,13,17"
    assert record.start_options()["formatid"] == "gen3customgame"
    assert record.username("p1") == "P1"
    assert record.packed_team("p1") == team
    # ...and the decoded referee view the prober/report path uses.
    assert record.team_details("p1")[0]["moves"] == ["return", "splash"]
    # `commands` is protocol-faithful: every CHOOSE processed, in order.
    assert record.commands, "commands must record the CHOOSE stream"
    assert all(side in ("p1", "p2") for side, _ in record.commands)
