"""ELIDING the side the search never reads must be LOUD, never an empty read.

``gen3_expand_many_side_elision_v1``. ``expand_many`` takes a ``side`` and omits the other side's
payload (the discarded copy measured **43.0% of the reply bytes** on 864 banked arms,
``designs/research_state/measurements/expand_many_2026-09-22/README.md``). The wire-level byte
identity is pinned in rust (``tests/search_side_elision_test.rs``). What this file pins is the
sentinel: an elided side is falsy but RAISES on every read, so nothing downstream of
``SearchEngine._expand_ply`` can encode a board nobody played. (The decision-level A/B this file
used to carry ran the VIEW road, deleted in the Rust Core deletion pass, program §4 M2; the core
road reads only its own side's ``core_pN``.)
"""

from __future__ import annotations

import pytest

from utils.bridge.search_session import ElidedSide, SearchError, SearchSession

def test_an_elided_side_refuses_to_be_read_rather_than_reading_empty():
    """The sentinel's contract, which is the whole reason elision is safe to default ON.

    Falsy — so the existing ``payload or {}`` guards take their COUNTED fallback — but raising on
    every way of actually getting a value out. An empty dict would ENCODE, into a well-formed
    observation of a battle nobody played."""
    el = ElidedSide("view_p2")

    assert not el, "an elided side must be FALSY so `payload or {}` still works"
    assert len(el) == 0

    for read in (lambda: el["species"],
                 lambda: el.get("species"),
                 lambda: list(el),
                 lambda: dict(el.items()),
                 lambda: list(el.keys()),
                 lambda: list(el.values()),
                 lambda: "species" in el):
        with pytest.raises(SearchError, match="ELIDED"):
            read()
    assert "view_p2" in repr(el)


def test_an_unknown_side_is_refused_in_python_before_the_driver_sees_it():
    """A typo must fail HERE, not become a driver error two layers away — and never both sides."""
    class _Never:
        def _call(self, payload):
            raise AssertionError("the driver must not be reached")

    with pytest.raises(SearchError, match="side must be"):
        SearchSession.expand_many(_Never(), [], side="p3")
