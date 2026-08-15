"""Unit tests for the launcher's default --showdown-port injection.

Long launcher sessions must default to the dedicated training server (8001), never
the shared dev server on 8000 — pointing training at 8000 means routine dev-server
churn drops every worker's connection at once and crashes the whole run. An explicit
--showdown-port (any spelling) must always win.

⚠️ THE DEFAULT INVERTED. `--use-bridge` now defaults to `rust`, so a run with NO transport flag
uses the in-process bridge and connects to no server at all — injecting a port there would be a
phantom. Only an explicit `--use-bridge off` reaches the injection path, and every test below that
exercises it says so. `child_uses_bridge` must agree with `train_rl_agent`'s own default; a drift
between the two is exactly what these tests exist to catch.
"""

from main.launcher.checkpoint import (
    DEFAULT_TRAINING_SHOWDOWN_PORT,
    _apply_default_showdown_port,
    _peek_arg,
    child_uses_bridge,
)

_WS = ["--use-bridge", "off"]        # the websocket transport, now the non-default one


def test_injects_default_when_absent():
    args = _WS + ["--steps", "1000000", "--device", "cuda"]
    out = _apply_default_showdown_port(args)
    assert _peek_arg(out, "--showdown-port", type_=int) == DEFAULT_TRAINING_SHOWDOWN_PORT
    # original args preserved
    assert "--steps" in out and "1000000" in out


def test_default_is_8001():
    assert DEFAULT_TRAINING_SHOWDOWN_PORT == 8001


def test_explicit_space_form_wins():
    args = _WS + ["--showdown-port", "8000", "--steps", "100"]
    out = _apply_default_showdown_port(args)
    assert _peek_arg(out, "--showdown-port", type_=int) == 8000
    # not duplicated
    assert out.count("--showdown-port") == 1


def test_explicit_equals_form_wins():
    args = _WS + ["--showdown-port=8123", "--steps", "100"]
    out = _apply_default_showdown_port(args)
    assert _peek_arg(out, "--showdown-port", type_=int) == 8123


def test_respects_custom_default():
    out = _apply_default_showdown_port(_WS + ["--steps", "100"], default_port=9009)
    assert _peek_arg(out, "--showdown-port", type_=int) == 9009


def test_does_not_mutate_input():
    args = _WS + ["--steps", "100"]
    before = list(args)
    _apply_default_showdown_port(args)
    assert args == before


def test_no_transport_flag_is_a_bridge_run_and_injects_no_port():
    """THE default-inversion pin. An absent --use-bridge means the trainer's `rust` default, so the
    launcher must NOT inject a port — the old behavior (absent => websocket => inject :8001) would
    now advertise a server that nothing connects to."""
    assert child_uses_bridge(["--steps", "100"]) is True
    out = _apply_default_showdown_port(["--steps", "100"])
    assert _peek_arg(out, "--showdown-port", type_=int) is None
    assert "--showdown-port" not in out


def test_explicit_bridge_impls_inject_no_port():
    for impl in ("node", "rust"):
        for form in ([f"--use-bridge={impl}"], ["--use-bridge", impl]):
            args = form + ["--steps", "100"]
            assert child_uses_bridge(args) is True, args
            out = _apply_default_showdown_port(args)
            assert "--showdown-port" not in out, args


def test_explicit_off_is_the_only_websocket_spelling():
    for form in (["--use-bridge=off"], ["--use-bridge", "off"]):
        assert child_uses_bridge(form + ["--steps", "100"]) is False, form


def test_bridge_mode_keeps_explicit_port_but_it_is_inert():
    # An explicit port alongside the bridge is left as-is (it's simply ignored at runtime —
    # the bridge never connects), but we don't inject the default on top of it.
    out = _apply_default_showdown_port(["--use-bridge=rust", "--showdown-port", "9001"])
    assert _peek_arg(out, "--showdown-port", type_=int) == 9001
    assert out.count("--showdown-port") == 1
