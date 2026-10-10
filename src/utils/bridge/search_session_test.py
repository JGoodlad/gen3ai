"""Unit tests for ``SearchSession``'s driver child (pure — the child is a fake ``Popen``, so nothing is
spawned and no cargo is needed). The driver is the Rust ``search_driver`` binary only: the Node driver and
the ``impl=`` selector were deleted 2026-10-10 (owner: "delete node").
"""

import pytest

from utils.bridge import search_session as ss_mod
from utils.bridge.search_session import SearchSession
from utils.bridge.sim_bridge_bin import SimBridgeBinaryError


class _FakeProc:
    """Enough of ``Popen`` for ``__init__`` + ``close()``: the reader threads iterate the streams
    (empty → they exit immediately) and teardown writes/flushes/waits."""

    def __init__(self, argv, **kw):
        self.argv = argv
        self.returncode = 0
        self.stdin = _FakeStream()
        self.stdout = _FakeStream()
        self.stderr = _FakeStream()

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class _FakeStream:
    def __iter__(self):
        return iter(())

    def write(self, _s):
        pass

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture
def spawned(monkeypatch):
    """Capture the argv every ``SearchSession`` would exec, spawning nothing."""
    calls = []

    def _fake_popen(argv, **kw):
        calls.append(list(argv))
        return _FakeProc(argv, **kw)

    monkeypatch.setattr(ss_mod.subprocess, "Popen", _fake_popen)
    return calls


def test_the_session_execs_the_resolved_rust_binary_and_never_node(spawned, tmp_path, monkeypatch):
    """The ONLY driver is the Rust ``search_driver`` binary (the Node driver and the ``impl=`` selector were
    deleted 2026-10-10): the session execs the resolved binary, and it no longer accepts an engine choice."""
    fake = tmp_path / "search_driver"
    fake.write_text("#!/bin/true\n")
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", str(fake))
    s = SearchSession()
    try:
        assert spawned == [[str(fake.resolve())]]
    finally:
        s.close()
    with pytest.raises(TypeError):
        SearchSession(impl="node")  # type: ignore[call-arg]
    assert len(spawned) == 1, "a refused construction must spawn nothing"


def test_unresolvable_rust_binary_raises_before_spawning(spawned, tmp_path, monkeypatch):
    """A missing rust binary must fail at construction with the actionable resolver error — not
    surface later as a mysterious dead child."""
    from utils.bridge import sim_bridge_bin

    monkeypatch.delenv("POKESIM_SEARCH_DRIVER_BIN", raising=False)
    monkeypatch.setattr(sim_bridge_bin, "_rust_bin_cache", {})
    monkeypatch.setattr(sim_bridge_bin, "_RUST_CRATE_DIR", tmp_path / "no_crate_here")
    with pytest.raises(SimBridgeBinaryError):
        SearchSession()
    assert spawned == [], "nothing may be spawned when the binary can't be resolved"


def test_child_errors_name_the_binary(spawned, tmp_path, monkeypatch):
    """A failure must self-diagnose: the stream-closed / timeout / desync messages name the binary that
    ran."""
    fake = tmp_path / "search_driver"
    fake.write_text("#!/bin/true\n")
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", str(fake))
    s = SearchSession()
    try:
        assert str(fake.resolve()) in s._who()
    finally:
        s.close()


# --- the CLOSE reap bound: scaled, not hardcoded -----------------------------------------------
#
# gen3_contention_robust_timeouts_v1. ``close()`` waits for the search-driver child to exit after
# the cooperative ``close`` command. That wait was a hardcoded ``timeout=5`` until 2026-09-01 — a
# wall-clock bound on a subprocess, so on a loaded box it measured the box rather than the child
# (the same defect that killed a measurement arm on ``local_battle_runner`` at load ~50 on 16
# cores, 2026-08-31). These two pin that the bound is read at CALL time through ``scale_timeout``,
# and that an idle box is unchanged.


def test_close_reap_timeout_is_the_base_value_on_an_idle_box(monkeypatch):
    """Factor 1.0 => still exactly 5.0 s. The fix must be a no-op when the box is quiet."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "1")
    assert ss_mod._CLOSE_REAP_TIMEOUT == 5.0
    assert ss_mod._close_reap_timeout() == 5.0


def test_close_reap_timeout_stretches_with_contention(monkeypatch):
    """The whole point: a loaded box gets proportionally longer to reap the child."""
    monkeypatch.setenv("GEN3AI_TIMEOUT_SCALE", "6")
    assert ss_mod._close_reap_timeout() == 30.0


# --------------------------------------------------------------------------- side elision
# Moved here from `main/search_dividend/side_elision_test.py` (P6 slice 6d-1, 2026-10-08, with the battery that
# held it): the subject is this module's `ElidedSide` sentinel and `expand_many`'s `side` argument, both KEPT.
# `gen3_expand_many_side_elision_v1`: `expand_many` takes a `side` and omits the other side's payload (the
# discarded copy measured 43.0% of the reply bytes on 864 banked arms,
# `designs/research_state/measurements/expand_many_2026-09-22/README.md`); the wire-level byte identity is pinned
# in rust (`tests/search_side_elision_test.rs`). What these pin is the sentinel: an elided side is falsy but RAISES
# on every read, so nothing downstream can encode a board nobody played.

def test_an_elided_side_refuses_to_be_read_rather_than_reading_empty():
    """The sentinel's contract, which is the whole reason elision is safe to default ON.

    Falsy — so the existing ``payload or {}`` guards take their COUNTED fallback — but raising on
    every way of actually getting a value out. An empty dict would ENCODE, into a well-formed
    observation of a battle nobody played."""
    el = ss_mod.ElidedSide("p2_chunks")

    assert not el, "an elided side must be FALSY so `payload or {}` still works"
    assert len(el) == 0

    for read in (lambda: el["species"],
                 lambda: el.get("species"),
                 lambda: list(el),
                 lambda: dict(el.items()),
                 lambda: list(el.keys()),
                 lambda: list(el.values()),
                 lambda: "species" in el):
        with pytest.raises(ss_mod.SearchError, match="ELIDED"):
            read()
    assert "p2_chunks" in repr(el)


def test_an_unknown_side_is_refused_in_python_before_the_driver_sees_it():
    """A typo must fail HERE, not become a driver error two layers away — and never both sides."""
    class _Never:
        def _call(self, payload):
            raise AssertionError("the driver must not be reached")

    with pytest.raises(ss_mod.SearchError, match="side must be"):
        SearchSession.expand_many(_Never(), [], side="p3")
