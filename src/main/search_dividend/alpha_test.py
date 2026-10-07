"""The alpha-candidate extraction rules — the alpha-CONSUMER contract, made testable.

Each test below names the clause it defends. They are worth having as tests rather than as
comments because every one of these failures is SILENT: a mis-aligned seat, a renormalized move
slice or a smeared belief-miss all produce a perfectly well-formed candidate set that weights the
wrong opponent action.
"""

from __future__ import annotations

import pytest

from main.search_dividend.alpha import (UNNAMED_FLOOR, AlphaPublication, alpha_publication,
                                        build_candidates, legal_choices_from_request)

# move id -> num, for tests: no dex load, so the rules are tested and not the data.
NUMS = {"earthquake": 89, "rockslide": 157, "surf": 57, "recover": 105}


def _request(moves, bench=(), trapped=False):
    return {
        "active": [{"moves": [{"id": m, "move": m.title(), "pp": 10} for m in moves],
                    "trapped": trapped}],
        "side": {"pokemon": ([{"ident": "p2a: Lead", "details": "Skarmory, M",
                               "condition": "100/100", "active": True}]
                             + [{"ident": f"p2: {n}", "details": f"{n}, M",
                                 "condition": c, "active": False} for (n, c) in bench])},
    }


# -- the legal surface --------------------------------------------------------


def test_legal_choices_read_the_servers_own_request():
    legal = legal_choices_from_request(_request(["earthquake", "surf"],
                                                bench=[("Blissey", "100/100")]))
    assert [c["token"] for c in legal] == ["move earthquake", "move surf", "switch 2"]


def test_a_disabled_or_pp_less_move_is_not_a_candidate():
    req = _request(["earthquake", "surf"])
    req["active"][0]["moves"][0]["disabled"] = True
    req["active"][0]["moves"][1]["pp"] = 0
    assert legal_choices_from_request(req) == []


def test_a_fainted_bench_mon_is_not_a_switch_target():
    legal = legal_choices_from_request(
        _request(["surf"], bench=[("Blissey", "0 fnt"), ("Celebi", "50/100")]))
    assert [c["token"] for c in legal] == ["move surf", "switch 3"]


def test_a_trapped_active_offers_no_switches():
    legal = legal_choices_from_request(
        _request(["surf"], bench=[("Blissey", "100/100")], trapped=True))
    assert [c["kind"] for c in legal] == ["move"]


@pytest.mark.parametrize("req", [None, {}, {"wait": True}, {"forceSwitch": [True]},
                                 {"teamPreview": True}])
def test_a_non_move_request_has_nothing_to_marginalize_over(req):
    """A forced switch has no branchable opponent action surface. Returning [] is what makes the
    caller record a `not_move_selection` fallback instead of inventing candidates."""
    assert legal_choices_from_request(req) == []


# -- the publication ----------------------------------------------------------


class _Stash:
    """An extractor stub carrying X5's FLAT opponent pointer stash (the only intent readout since the version
    break, config v144): K move seats, OTHER_move, six switch slots, OTHER_species — one softmax."""

    def __init__(self, k=2, seats=None, seat_p=None, other_move_p=0.0, slot_p=None, other_species_p=0.0,
                 built=True, stashed=True, width=None):
        import math
        from types import SimpleNamespace

        import torch
        self.flat_intent_head = object() if built else None
        seats = list(seats or [89, 157])[:k]
        seat_p = list(seat_p or [0.0] * k)
        slot_p = list(slot_p) if slot_p is not None else [None] * 6
        probs = seat_p + [other_move_p] + [0.0 if q is None else q for q in slot_p] + [other_species_p]
        live = [q > 0 for q in seat_p] + [other_move_p > 0] + [q is not None for q in slot_p] + [other_species_p > 0]
        logits = [math.log(q) if (lv and q > 0) else float("-inf") for q, lv in zip(probs, live)]
        if width is not None:
            logits = logits[:width]
        ids = seats + [0] + [100 + j for j in range(6)] + [0]
        self.last_flat_intent_logits = torch.tensor([logits]) if stashed else None
        self.last_flat_intent = (SimpleNamespace(k=k, live=torch.tensor([live]), cand_ids=torch.tensor([ids]),
                                                 slot_species=torch.tensor([[0] * 6]))
                                 if stashed else None)


def test_an_alpha_off_checkpoint_publishes_nothing():
    assert alpha_publication(None) is None
    assert alpha_publication(_Stash(built=False)) is None


def test_a_built_pointer_that_stashed_nothing_RAISES_rather_than_reading_as_no_opinion():
    """A silent None here would run the uniform ABSENCE fallback as if the model had no intent readout."""
    with pytest.raises(RuntimeError, match="stashed no flat_intent"):
        alpha_publication(_Stash(stashed=False))


def test_a_flat_width_mismatch_FAILS_LOUD():
    """Clause 3 — align by CONSTRUCTION. A list that is not K seats + OTHER_move + six slots + OTHER_species
    would pair each probability with the wrong candidate while every shape check still passed."""
    stash = _Stash(k=2, seat_p=[0.5, 0.5], width=7)
    with pytest.raises(ValueError, match="flat pointer width"):
        alpha_publication(stash)


def test_an_unfilled_seat_is_dropped_not_shown_as_an_anonymous_index():
    stash = _Stash(k=2, seats=[89, 0], seat_p=[0.4, 0.1], other_move_p=0.2, slot_p=[0.3, None, None, None, None, None])
    pub = alpha_publication(stash)
    assert set(pub.move_p) == {89}
    assert pub.move_p[89] == pytest.approx(0.4, abs=1e-5)
    assert pub.switch_p == pytest.approx(0.3, abs=1e-5)
    assert pub.other_move_p == pytest.approx(0.2, abs=1e-5)   # REPORTED, never a named branch


def test_beta_masks_are_respected():
    stash = _Stash(k=1, seats=[89], seat_p=[0.5], slot_p=[None, 0.3, 0.1, None, None, None], other_species_p=0.1)
    pub = alpha_publication(stash)
    assert set(pub.beta_p) == {1, 2}          # an illegal switch-in is UNREPRESENTABLE
    assert pub.switch_p == pytest.approx(0.5, abs=1e-5)       # OTHER_species is switch mass, but names no slot
    assert pub.beta_p[1] == pytest.approx(0.6, abs=1e-5) and pub.beta_p[2] == pytest.approx(0.2, abs=1e-5)


# -- the candidate set --------------------------------------------------------


def test_the_move_slice_is_NOT_renormalized_and_switch_mass_stays_switch():
    """Clause 2 — the missing mass is SWITCH; renormalizing the move slice asserts they attacked.

    Here alpha says 0.3/0.2 on two moves and 0.5 on SWITCH. The final weights must keep switch at
    half, not redistribute it across the attacks."""
    pub = AlphaPublication(move_p={89: 0.3, 157: 0.2}, switch_p=0.5, beta_p={})
    legal = legal_choices_from_request(_request(["earthquake", "rockslide"],
                                                bench=[("Blissey", "100/100")]))
    cands, diag = build_candidates(legal, pub, m_opp=3, num_by_id=NUMS)
    by_token = {c.token: c.weight for c in cands}
    assert by_token["switch 2"] == pytest.approx(0.5)
    assert by_token["move earthquake"] == pytest.approx(0.3)
    assert diag["alpha_used"] is True
    assert sum(by_token.values()) == pytest.approx(1.0)


def test_switch_mass_is_split_across_targets_by_beta():
    pub = AlphaPublication(move_p={89: 0.4}, switch_p=0.6, beta_p={1: 0.75, 2: 0.25})
    legal = legal_choices_from_request(
        _request(["earthquake"], bench=[("Blissey", "100/100"), ("Celebi", "100/100")]))
    cands, _ = build_candidates(legal, pub, m_opp=3, num_by_id=NUMS)
    by_token = {c.token: c.weight for c in cands}
    assert by_token["switch 2"] == pytest.approx(0.45)
    assert by_token["switch 3"] == pytest.approx(0.15)


def test_switch_mass_is_UNIFORM_over_targets_when_there_is_no_beta():
    pub = AlphaPublication(move_p={89: 0.4}, switch_p=0.6, beta_p={})
    legal = legal_choices_from_request(
        _request(["earthquake"], bench=[("Blissey", "100/100"), ("Celebi", "100/100")]))
    cands, _ = build_candidates(legal, pub, m_opp=3, num_by_id=NUMS)
    by_token = {c.token: c.weight for c in cands}
    assert by_token["switch 2"] == pytest.approx(0.3)
    assert by_token["switch 3"] == pytest.approx(0.3)


def test_a_belief_MISS_is_reported_not_smeared():
    """Clause 3's other half: alpha mass on a move the opponent does not have is the BELIEF's
    coverage failure. Folding it into the weights would hide which component to fix."""
    pub = AlphaPublication(move_p={89: 0.3, 999: 0.5}, switch_p=0.2, beta_p={})
    legal = legal_choices_from_request(_request(["earthquake"]))
    cands, diag = build_candidates(legal, pub, m_opp=4, num_by_id=NUMS)
    assert diag["unmatched_move_mass"] == pytest.approx(0.5)
    assert [c.token for c in cands] == ["move earthquake"]


def test_a_move_alpha_never_names_still_gets_a_floor():
    """A surprise must be reachable, or the search structurally cannot consider an action the
    opponent can legally take."""
    pub = AlphaPublication(move_p={89: 0.9}, switch_p=0.0, beta_p={})
    legal = legal_choices_from_request(_request(["earthquake", "recover"]))
    cands, _ = build_candidates(legal, pub, m_opp=4, num_by_id=NUMS)
    by = {c.token: c for c in cands}
    assert by["move recover"].source == "floor"
    assert 0 < by["move recover"].weight < by["move earthquake"].weight
    assert by["move recover"].weight == pytest.approx(UNNAMED_FLOOR / (0.9 + UNNAMED_FLOOR))


def test_pruning_keeps_the_TOP_m_and_reports_the_mass_it_dropped():
    pub = AlphaPublication(move_p={89: 0.5, 157: 0.3, 57: 0.1}, switch_p=0.1, beta_p={})
    legal = legal_choices_from_request(_request(["earthquake", "rockslide", "surf"]))
    cands, diag = build_candidates(legal, pub, m_opp=2, num_by_id=NUMS)
    assert [c.token for c in cands] == ["move earthquake", "move rockslide"]
    assert diag["retained_mass"] == pytest.approx(0.8)
    assert sum(c.weight for c in cands) == pytest.approx(1.0)


def test_the_no_head_fallback_is_UNIFORM_an_absence_not_a_claim():
    """The rejected alternative — the R1 `belief_mean` rung — has no switch class, so it would set
    alpha_SWITCH == 0 and thereby assert *they never switch*. A fallback that silently states
    something false is worse than a fallback that says it has no opinion (the v94 lesson)."""
    legal = legal_choices_from_request(
        _request(["earthquake", "surf"], bench=[("Blissey", "100/100")]))
    cands, diag = build_candidates(legal, None, m_opp=3)
    assert diag["alpha_used"] is False
    assert {c.source for c in cands} == {"uniform"}
    assert [c.weight for c in cands] == pytest.approx([1 / 3, 1 / 3, 1 / 3])
    assert any(c.kind == "switch" for c in cands), "SWITCH must remain representable"


def test_no_legal_choices_yields_no_candidates():
    cands, diag = build_candidates([], None, m_opp=3)
    assert cands == []
    assert diag["n_legal"] == 0
