"""The move-category rules (unit)."""

from __future__ import annotations

import re

import pytest

from main.policy_spectrum import categories as C
from utils.paths import repo_path


@pytest.mark.parametrize("token,cat,sub", [
    ("switch Dugtrio", "switch", None),
    ("move thunderbolt", "attack", None),
    ("move hiddenpowergrass", "attack", None),
    ("move seismictoss", "attack", None),
    ("move struggle", "attack", None),
    ("move explosion", "attack", None),
    ("move rapidspin", "attack", None),
    ("move dragondance", "setup", None),
    ("move bellydrum", "setup", None),
    ("move curse", "setup", None),
    ("move spikes", "hazard", None),
    ("move recover", "recovery", None),
    ("move rest", "recovery", None),
    ("move wish", "recovery", None),
    ("move painsplit", "recovery", None),
    ("move toxic", "status", "inflict"),
    ("move willowisp", "status", "inflict"),
    ("move protect", "status", "protect"),
    ("move roar", "status", "phaze"),
    ("move healbell", "status", "cure"),
    ("move substitute", "status", "other"),
    ("move batonpass", "status", "other"),
    ("move leechseed", "status", "other"),
])
def test_token_categories(token, cat, sub):
    assert C.token_category(token) == cat
    assert C.token_subtype(token) == sub


def test_unknown_move_and_token_refused():
    with pytest.raises(KeyError):
        C.move_category("notamove")
    with pytest.raises(ValueError):
        C.token_category("team 123456")


def test_zero_bp_damaging_set_is_showdowns():
    """Every gen-3 dex move with base power 0 is an attack iff Showdown's own `category` says
    Physical/Special — the set cannot drift from the source."""
    from agents.gen3_data import moves as M

    txt = repo_path("deps", "pokemon-showdown", "data", "moves.ts").read_text()
    cat = {}
    for m in re.finditer(r"\n\t(\w+): \{(.*?)\n\t\},", txt, re.S):
        c = re.search(r'category: "(\w+)"', m.group(2))
        if c:
            cat[m.group(1)] = c.group(1)
    assert len(cat) > 500
    import json
    ids = json.loads(repo_path("data", "pokemon", "gen3_moves.json").read_text()).keys()
    zero_bp_damaging = {i for i in ids if M.get(i) and M.get(i).base_power == 0
                        and cat.get(i) not in (None, "Status")}
    assert zero_bp_damaging == set(C.ZERO_BP_DAMAGING)
