"""Unit tests for ``SearchSession``'s transport selection (pure — the driver child is a fake
``Popen``, so nothing is spawned and no Node/cargo is needed).

The point of these is the **byte-identical-default** bar: making the driver impl-selectable must
not change what any existing caller spawns. So the node case asserts the EXACT argv the session
used before the seam existed, and the rust case asserts it fails loudly rather than quietly
serving node.
"""

from pathlib import Path

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


def test_default_is_node_and_spawns_the_exact_historical_argv(spawned):
    """No ``impl`` argument ⇒ ``["node", <…/search_driver.js>]``, exactly what the session
    hard-coded before this was selectable. Any drift here is a behaviour change for every
    existing caller (the prober, both search-teacher workers, teacher/produce)."""
    s = SearchSession()
    try:
        assert s.impl == "node"
        assert len(spawned) == 1
        argv = spawned[0]
        assert argv[0] == "node"
        assert argv[1] == str(Path(ss_mod.__file__).parent / "search_driver.js")
        assert len(argv) == 2
    finally:
        s.close()


def test_explicit_node_matches_the_default(spawned):
    a = SearchSession()
    a.close()
    b = SearchSession(impl="node")
    b.close()
    assert spawned[0] == spawned[1]


def test_rust_execs_the_resolved_binary_and_never_node(spawned, tmp_path, monkeypatch):
    fake = tmp_path / "search_driver"
    fake.write_text("#!/bin/true\n")
    monkeypatch.setenv("POKESIM_SEARCH_DRIVER_BIN", str(fake))
    s = SearchSession(impl="rust")
    try:
        assert s.impl == "rust"
        assert spawned == [[str(fake.resolve())]]
    finally:
        s.close()


def test_unresolvable_rust_binary_raises_before_spawning(spawned, tmp_path, monkeypatch):
    """A missing rust binary must fail at construction with the actionable resolver error — NOT
    fall back to node, and not surface later as a mysterious dead child."""
    from utils.bridge import sim_bridge_bin

    monkeypatch.delenv("POKESIM_SEARCH_DRIVER_BIN", raising=False)
    monkeypatch.setattr(sim_bridge_bin, "_rust_bin_cache", {})
    monkeypatch.setattr(sim_bridge_bin, "_RUST_CRATE_DIR", tmp_path / "no_crate_here")
    with pytest.raises(SimBridgeBinaryError):
        SearchSession(impl="rust")
    assert spawned == [], "nothing may be spawned when the impl can't be resolved"


def test_unknown_impl_rejected_before_spawning(spawned):
    with pytest.raises(ValueError):
        SearchSession(impl="bogus")
    assert spawned == []


def test_child_errors_name_the_impl_and_the_binary(spawned):
    """A rust failure must self-diagnose. The stream-closed / timeout / desync messages used to
    say 'search_driver.js died' unconditionally, which would read as a Node crash on a run that
    never launched Node."""
    s = SearchSession(impl="node")
    try:
        who = s._who()
        assert "node" in who
        assert "search_driver.js" in who
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
