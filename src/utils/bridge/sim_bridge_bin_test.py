"""Unit tests for the sim-bridge binary resolver (pure — no cargo build, no subprocess)."""

import os
from pathlib import Path

import pytest

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


def test_deferral_warning_names_the_two_deferrals():
    msg = rust_deferral_warning()
    assert "__RECON__" in msg
    assert "resumeReseed" in msg
