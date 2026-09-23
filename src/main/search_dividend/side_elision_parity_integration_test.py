"""ELIDING the side the search never reads must change NOTHING the search sees.

``gen3_expand_many_side_elision_v1``. ``expand_many`` rendered both sides' one-sided payload
(``view_pN`` + ``pN_chunks``) on every arm and :meth:`SearchEngine._expand_ply` read exactly one
of them; the discarded copy measured **43.0% of the reply bytes** on 864 banked arms
(``designs/research_state/measurements/expand_many_2026-09-22/README.md``). The driver now takes a
``side`` on the request and omits the other.

🚨 **What could go wrong is not "the wrong side comes back" — that would be loud.** It is that
something downstream of ``_expand_ply`` reaches for the OTHER side and gets an empty payload
instead of a raise, in which case the arm silently falls back (or encodes a board nobody played)
and the scores stay plausible. The wire-level byte identity is pinned in rust
(``tests/search_side_elision_test.rs``); what this gate pins is the DECISION: the real
:class:`~main.search_dividend.search.SearchEngine` over one seeded decision, with elision on
(production) and off (the control), compared on every successor's observation BYTES in order plus
the per-action scores, the chosen action and the realized widths — including the FALLBACK
counters, because an elision that quietly pushed arms onto the protocol road would otherwise read
as a pass.

The scorer is the pure ``obs.sum`` for the reason the sibling gates give: a trained net is a
contraction and can map two different vectors onto one score, while a sum cannot.
"""

from __future__ import annotations

import tempfile
from typing import List

import numpy as np
import pytest

from main.search_dividend.budget import WidthCaps
from main.search_dividend.search import SearchConfig, SearchEngine
from utils.bridge.search_session import ElidedSide, SearchError, SearchSession

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: 🚨 PINNED — a collected test takes the SAME battle every run (the project's fuzz rule,
#: ``designs/rust_sim/one_sided_view.md`` §4). The sweeps are what run fresh draws.
_KEY = 5


def _engine(pool) -> SearchEngine:
    cfg = SearchConfig(
        arm="honest", budget_s=1e9, seed=7, max_depth=1, search_impl="rust",
        materializer="view", caps=WidthCaps(m_opp=2, k_worlds=3, r_dice=1))
    eng = SearchEngine(model=None, mappings=None, cfg=cfg, pool_packed=list(pool))
    eng._score_batch = lambda obs, masks: (                      # type: ignore[assignment]
        np.asarray(obs, dtype=np.float64).sum(axis=1), "fake")
    return eng


def _run(pool, record, side, turn, our_history, tokens, observed, *, elide: bool):
    """``(scores, chosen action, widths, [successor obs bytes, in order], elided_seen)``.

    The CONTROL drops the ``side`` kwarg on its way to the driver — the very code path the
    experiment runs, differing only in whether the driver is told it may skip a side. Stubbing
    ``_expand_ply`` instead would have compared two different call graphs."""
    import agents.training.view_successor as VS

    seen: List[bytes] = []
    orig_succ = VS.ViewSuccessorFactory.successor
    orig_expand = SearchSession.expand_many
    elided_seen = [0]

    def recording(self, payload, chunks, action):
        got = orig_succ(self, payload, chunks, action)
        if got is not None:
            seen.append(np.asarray(got.obs, dtype=np.float32).tobytes())
        return got

    def expand(self, arms, *, side=None):
        nodes = orig_expand(self, arms, side=(side if elide else None))
        for n in nodes:
            other = "p2" if side == "p1" else "p1"
            if isinstance(getattr(n, f"view_{other}"), ElidedSide):
                elided_seen[0] += 1
        return nodes

    VS.ViewSuccessorFactory.successor = recording
    SearchSession.expand_many = expand
    eng = _engine(pool)
    try:
        res = eng.choose(record=record, side=side, turn=turn, our_history=list(our_history),
                         our_tokens=dict(tokens), observed_our_lines=list(observed),
                         pub=None, policy_action=next(iter(tokens)), opp_true_packed=None)
    finally:
        VS.ViewSuccessorFactory.successor = orig_succ
        SearchSession.expand_many = orig_expand
        eng.close()
    return dict(res.scores or {}), int(res.action), res.widths, seen, elided_seen[0]


def test_eliding_the_unread_side_leaves_every_successor_byte_identical():
    import agents.battle.one_sided_view_parity_fuzz_test as G
    from main.search_dividend import determinize as dz
    from main.search_dividend.__main__ import _pool

    with tempfile.TemporaryDirectory() as td:
        record, summary, npz = G._record_one_battle(td, "rust", _KEY)
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary["invocations"]
    cand = [i for i, iv in enumerate(invs)
            if iv.get("phase") == "move_selection" and int(iv["turn"]) > 1]
    assert cand, "the fixture battle produced no mid-battle move selection"
    side = record.side_of(record.trainee_username)
    pool = _pool(0)

    compared = 0
    for frac in (0.0, 0.15, 0.3):
        anchor = cand[int(len(cand) * frac)]
        turn = int(invs[anchor]["turn"])
        our_history = [int(x) for x in actions[:anchor]]
        with SearchSession(record, impl="rust") as ss:
            pfx = getattr(ss.open_root(turn), f"prefix_{side}_chunks")
        tokens = G._choice_map(record, side, our_history, pfx, anchor)
        if not tokens:
            continue
        observed = dz.chunks_to_lines(pfx)

        ctl_s, ctl_a, ctl_w, ctl_obs, ctl_el = _run(
            pool, record, side, turn, our_history, tokens, observed, elide=False)
        exp_s, exp_a, exp_w, exp_obs, exp_el = _run(
            pool, record, side, turn, our_history, tokens, observed, elide=True)

        if not exp_obs:
            continue                       # no view arm at this anchor — nothing to compare
        compared += 1

        # NON-VACUITY, both directions: the experiment must really have elided, the control
        # must really not have. Without these the whole comparison passes on two identical runs.
        assert exp_el > 0, (
            f"turn {turn}: {len(exp_obs)} successors and NOT ONE elided side — the driver "
            f"returned both, so this comparison proves nothing")
        assert ctl_el == 0, f"turn {turn}: the control elided a side"

        assert len(exp_obs) == len(ctl_obs), (
            f"turn {turn}: {len(exp_obs)} elided successors vs {len(ctl_obs)} un-elided")
        for i, (a, b) in enumerate(zip(ctl_obs, exp_obs)):
            assert a == b, (
                f"turn {turn}: successor #{i} differs with the other side elided — the search "
                f"was reading a side it told the driver to skip")
        assert exp_s == ctl_s, f"turn {turn}: per-action scores differ"
        assert exp_a == ctl_a, f"turn {turn}: chose {exp_a} elided vs {ctl_a} un-elided"
        assert exp_w.arms_scored == ctl_w.arms_scored
        assert exp_w.view_arms == ctl_w.view_arms
        # 🚨 THE ROW THAT CATCHES A QUIET DEGRADATION. An elided payload read as empty would
        # simply push the arm onto the protocol road, where it still produces a well-formed obs.
        assert exp_w.view_fallback_no_payload == ctl_w.view_fallback_no_payload, (
            f"turn {turn}: elision pushed {exp_w.view_fallback_no_payload} arms onto the "
            f"protocol road against the control's {ctl_w.view_fallback_no_payload}")
        assert exp_w.view_fallback_intermediate == ctl_w.view_fallback_intermediate
        print(f"  turn {turn}: successors={len(exp_obs)} view_arms={exp_w.view_arms} "
              f"elided_arms={exp_el} fallbacks={exp_w.view_fallback_no_payload}/"
              f"{exp_w.view_fallback_intermediate}")

    assert compared >= 1, "no decision produced a view arm — the gate is vacuous"


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
