"""`successors_parity.legal_choices_from_request` — the legal surface the in-process search tree's gates and its
benchmark read off Showdown's own request JSON.

Moved here with the function from `main/search_dividend/alpha.py` and `alpha_test.py` when P6 slice 6d-1
(2026-10-08) deleted the search-dividend battery (the function's other half of that module — the alpha-candidate
publication — went with it). Each test names the clause it defends; every one of these failures is SILENT (a
mis-listed legal surface hands the tree an illegal or missing arm, and the gate still compares what it was given).
Milliseconds, unmarked.
"""
from __future__ import annotations

import pytest

from utils.rust_env.successors_parity import legal_choices_from_request


def _request(moves, bench=(), trapped=False):
    return {
        "active": [{"moves": [{"id": m, "move": m.title(), "pp": 10} for m in moves],
                    "trapped": trapped}],
        "side": {"pokemon": ([{"ident": "p2a: Lead", "details": "Skarmory, M",
                               "condition": "100/100", "active": True}]
                             + [{"ident": f"p2: {n}", "details": f"{n}, M",
                                 "condition": c, "active": False} for (n, c) in bench])},
    }


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
