"""Unit tests for the sim-bridge binary resolver (pure — no cargo build, no subprocess)."""

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
)


def test_node_argv_is_node_plus_the_js():
    argv = bridge_spawn_argv("node")
    assert argv[0] == "node"
    assert argv[1].endswith("local_sim_bridge.js")


def test_rust_argv_is_the_resolved_binary_via_env_override(tmp_path, monkeypatch):
    # A pre-built binary via POKESIM_SIM_BRIDGE_BIN is honored first — no cargo build.
    fake = tmp_path / "sim_bridge"
    fake.write_text("#!/bin/true\n")
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", str(fake))
    # The cache is process-global; clear it so the override is (re)read.
    monkeypatch.setattr(sim_bridge_bin, "_rust_bin_cache", None)
    argv = bridge_spawn_argv("rust")
    assert argv == [str(fake.resolve())]  # NO "node" — just the binary


def test_env_override_missing_file_raises_clear_error(monkeypatch):
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", "/nonexistent/sim_bridge_xyz")
    monkeypatch.setattr(sim_bridge_bin, "_rust_bin_cache", None)
    with pytest.raises(SimBridgeBinaryError) as ei:
        bridge_spawn_argv("rust")
    assert "POKESIM_SIM_BRIDGE_BIN" in str(ei.value)


def test_unknown_impl_rejected():
    with pytest.raises(ValueError) as ei:
        bridge_spawn_argv("bogus")
    assert "bogus" in str(ei.value)
    assert VALID_IMPLS == ("node", "rust")


def test_deferral_warning_names_recon_and_reseed_as_supported():
    """Both `__RECON__` (gen3_bridge_recon_record_v1) and `resumeReseed`
    (gen3_bridge_resume_reseed_v1) SHIPPED — and, since
    ``gen3_bridge_seedless_fixed_seed_v1``, the record is emitted on a SEEDLESS battle too
    (the child mints + reports a real seed), which is the production case.

    The warning must keep naming BOTH so an operator can tell which paths still need
    ``--use-bridge=node`` — but it must not claim either is deferred/ignored, which would
    send someone to the node bridge for forensics that now work on rust. What DOES remain
    node-only is the search TEACHER, which needs the sim's byte-identical ``input_log``.
    """
    msg = rust_deferral_warning()
    assert "__RECON__" in msg
    assert "resumeReseed" in msg
    assert "SUPPORTED" in msg, "the warning must not still claim these are deferred"
    assert "byte-identical" in msg, "the warning must keep the honest input_log scope"


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
