"""Unit tests for the launcher's default --showdown-port injection.

Long launcher sessions must default to the dedicated training server (8001), never
the shared dev server on 8000 — pointing training at 8000 means routine dev-server
churn drops every worker's connection at once and crashes the whole run. An explicit
--showdown-port (any spelling) must always win.
"""

from main.launcher.checkpoint import (
    DEFAULT_TRAINING_SHOWDOWN_PORT,
    _apply_default_showdown_port,
    _peek_arg,
)


def test_injects_default_when_absent():
    args = ["--steps", "1000000", "--device", "cuda"]
    out = _apply_default_showdown_port(args)
    assert _peek_arg(out, "--showdown-port", type_=int) == DEFAULT_TRAINING_SHOWDOWN_PORT
    # original args preserved
    assert "--steps" in out and "1000000" in out


def test_default_is_8001():
    assert DEFAULT_TRAINING_SHOWDOWN_PORT == 8001


def test_explicit_space_form_wins():
    args = ["--showdown-port", "8000", "--steps", "100"]
    out = _apply_default_showdown_port(args)
    assert _peek_arg(out, "--showdown-port", type_=int) == 8000
    # not duplicated
    assert out.count("--showdown-port") == 1


def test_explicit_equals_form_wins():
    args = ["--showdown-port=8123", "--steps", "100"]
    out = _apply_default_showdown_port(args)
    assert _peek_arg(out, "--showdown-port", type_=int) == 8123


def test_respects_custom_default():
    out = _apply_default_showdown_port(["--steps", "100"], default_port=9009)
    assert _peek_arg(out, "--showdown-port", type_=int) == 9009


def test_does_not_mutate_input():
    args = ["--steps", "100"]
    _apply_default_showdown_port(args)
    assert args == ["--steps", "100"]


def test_bridge_mode_injects_no_port():
    # --use-showdown-bridge connects to no server, so no default port is injected (a phantom
    # :8001 would mislead the TUI / suggest a server is in use when none is).
    out = _apply_default_showdown_port(["--use-showdown-bridge", "--steps", "100"])
    assert _peek_arg(out, "--showdown-port", type_=int) is None
    assert "--showdown-port" not in out


def test_bridge_mode_keeps_explicit_port_but_it_is_inert():
    # An explicit port alongside the bridge is left as-is (it's simply ignored at runtime —
    # the bridge never connects), but we don't inject the default on top of it.
    out = _apply_default_showdown_port(["--use-showdown-bridge", "--showdown-port", "9001"])
    assert _peek_arg(out, "--showdown-port", type_=int) == 9001
    assert out.count("--showdown-port") == 1
