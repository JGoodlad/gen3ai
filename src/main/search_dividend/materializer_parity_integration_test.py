"""``--materializer view`` and ``--materializer core`` must decide EXACTLY what ``protocol`` decides.

The CORE road (`gen3_core_search_v1`, the Rust Core Program's M2 adoption) is the third road: a
successor is a Rust-core ``BattleVersion``, and this gate runs it with its INTEGRITY check on every
arm (each successor built typed AND from the side's text, view and obs bytes asserted equal).

`gen3_view_successor_v1`. ``agents/battle/one_sided_view_parity_fuzz_test.py`` proves the two
roads build the same 2501-dim vector for a successor. That is the hard half, and it is not the
whole claim: a search turns vectors into ARM SCORES and arm scores into an ACTION, through a
batch forward, a weighted marginalization over opponent candidates and dice, and a per-action
backup. This gate closes that gap by running the real :class:`~main.search_dividend.search.
SearchEngine` over the same seeded decision twice, once on each road, and comparing:

1. every arm's per-action score (``DecisionResult.scores``) — the values the leaf produced,
   not merely the obs it was produced from;
2. the chosen action and the fallback reason;
3. the realized WIDTHS, so a road that quietly materialized fewer arms fails here rather than
   reading as a tie.

🚨 **The scorer is a PURE FUNCTION OF THE OBS, deliberately.** A trained model would make this a
test of the checkpoint on disk; ``obs.sum(axis=1)`` makes it a test of the road, and it is
strictly SHARPER — a network is a contraction and can map two different vectors onto the same
score, while a sum cannot hide a single changed dimension.

⚠️ **The per-arm FALLBACK is what makes a green here meaningful, and it is asserted.** The view
road hands three classes of arm back to the protocol road (no ``view_pN``, an intermediate
decision, a parent that had itself fallen back), so a run in which every arm fell back would
compare the protocol road with itself and pass. The test asserts the view road actually
materialized arms.
"""

from __future__ import annotations

import tempfile
from typing import Tuple

import numpy as np
import pytest

from main.search_dividend.search import SearchConfig, SearchEngine, WidthCaps

pytestmark = [pytest.mark.sim, pytest.mark.integration]

_TURN_FRACTIONS = (0.3, 0.55, 0.8)


def _fixture(key: int = 0, impl: str = "rust"):
    """The REPRODUCIBLE fixture battle, through the one-sided gate's own recorder — one builder,
    so this test and that gate cannot drift onto different boards."""
    import agents.battle.one_sided_view_parity_fuzz_test as G

    with tempfile.TemporaryDirectory() as td:
        return G._record_one_battle(td, impl, key)


def _cfg(materializer: str) -> SearchConfig:
    # PINNED widths. A cell's realized width is a function of the wall clock, so an un-pinned
    # comparison would be measuring which road was faster — which is a different (and here,
    # meaningless) question. `--max-*` at 1 also makes the arm set small enough to be a test.
    # The CORE road runs with its INTEGRITY check on every arm (typed + text, view + obs bytes).
    return SearchConfig(
        arm="oracle", budget_s=60.0, seed=7, max_depth=1, search_impl="rust",
        materializer=materializer, integrity=1 if materializer == "core" else 0,
        caps=WidthCaps(m_opp=2, k_worlds=1, r_dice=1))


def _decide(materializer: str, record, side: str, turn: int, our_history, our_tokens,
            observed, opp_true_packed) -> Tuple[dict, int, str, object]:
    engine = SearchEngine(model=None, mappings=None, cfg=_cfg(materializer), pool_packed=[])
    # The pure-function scorer — see the module header.
    engine._score_batch = lambda obs, masks: (                      # type: ignore[assignment]
        np.asarray(obs, dtype=np.float64).sum(axis=1), "fake")
    try:
        res = engine.choose(
            record=record, side=side, turn=turn, our_history=list(our_history),
            our_tokens=dict(our_tokens), observed_our_lines=list(observed),
            pub=None, policy_action=next(iter(our_tokens)),
            opp_true_packed=opp_true_packed)
        return (dict(res.scores or {}), int(res.action), str(res.fallback), res.widths)
    finally:
        engine.close()


def test_the_two_materializers_decide_identically_on_a_seeded_decision():
    record, summary, npz = _fixture()
    actions = np.asarray(npz["actions"], dtype=int)
    invs = summary["invocations"]
    cand = [i for i, inv in enumerate(invs)
            if inv.get("phase") == "move_selection" and int(inv["turn"]) > 1]
    assert cand, "the fixture battle produced no mid-battle move selection"
    side = record.side_of(record.trainee_username)
    opp_true = record.packed_team("p2" if side == "p1" else "p1")

    compared = 0
    view_arms = 0
    core_arms = 0
    for frac in _TURN_FRACTIONS:
        anchor = cand[int(len(cand) * frac)]
        turn = int(invs[anchor]["turn"])
        our_history = [int(x) for x in actions[:anchor]]
        # The decision's own legal surface, through the REAL mapper — the same primitive the
        # live player hands `choose`.
        observed, tokens = _surface(record, side, our_history, anchor, turn)
        if not tokens:
            continue
        a = _decide("protocol", record, side, turn, our_history, tokens, observed, opp_true)
        pa_a, act_a, why_a, w_a = a
        for road in ("view", "core"):
            pa_b, act_b, why_b, w_b = _decide(road, record, side, turn, our_history, tokens,
                                              observed, opp_true)
            assert why_a == why_b, f"turn {turn}: fallback {why_a!r} vs {why_b!r} on {road}"
            assert set(pa_a) == set(pa_b), f"turn {turn}: scored actions differ on {road}"
            for k in pa_a:
                assert pa_a[k] == pytest.approx(pa_b[k], abs=0.0, rel=0.0), (
                    f"turn {turn}: action {k} scored {pa_a[k]!r} on protocol and {pa_b[k]!r} on {road}")
            assert act_a == act_b, f"turn {turn}: chose {act_a} vs {act_b} on {road}"
            assert w_a.arms_scored == w_b.arms_scored, (
                f"turn {turn}: {w_a.arms_scored} arms scored on protocol, {w_b.arms_scored} on {road}")
            print(f"  turn {turn} [{road}]: fallback={why_b!r} arms_scored={w_b.arms_scored} "
                  f"view_arms={w_b.view_arms} fb_no_payload={w_b.view_fallback_no_payload} "
                  f"fb_mid={w_b.view_fallback_intermediate} core_arms={w_b.core_arms} "
                  f"core_mid={w_b.core_arms_intermediate} integrity={w_b.integrity_checked} "
                  f"worlds_open_failed={w_b.worlds_open_failed} gate_failed={w_b.worlds_gate_failed}")
            if road == "view":
                view_arms += int(w_b.view_arms)
            else:
                core_arms += int(w_b.core_arms)
                assert w_b.integrity_checked == w_b.core_arms, "the core road ran un-checked arms"
        compared += 1

    assert compared >= 2, f"only {compared} decisions compared — the gate is vacuous"
    assert view_arms > 0, (
        "the VIEW road materialized NO arm — every one fell back to protocol, so this run "
        "compared the protocol road with itself. Is the search driver the rust one, and is the "
        "binary built from this tree?")
    assert core_arms > 0, "the CORE road answered no arm — the three-road comparison is vacuous"


def _surface(record, side: str, our_history, anchor: int, turn: int):
    """``(observed one-sided lines, {action index: choice string})`` at the decision — both read
    off a ``SearchSession`` root, which is where the live player gets them too."""
    from utils.bridge.search_session import SearchSession

    with SearchSession(record, impl="rust") as ss:
        root = ss.open_root(turn)
        pfx = root.prefix_p1_chunks if side == "p1" else root.prefix_p2_chunks
    import agents.battle.one_sided_view_parity_fuzz_test as G
    from main.search_dividend import determinize as dz

    # 🚨 LINES, not chunks. `prefix_matches` applies `chunks_to_lines` to the REPLAYED side only,
    # so handing it chunks fails the determinization gate on every decision — which reads as
    # `prefix_gate_failed` and silently turns the arm into the control.
    return dz.chunks_to_lines(pfx), G._choice_map(record, side, list(our_history), pfx, anchor)
