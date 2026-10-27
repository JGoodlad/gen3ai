"""The front end PLAYS the validator's re-packed team (a real server's `valResult.slice(1)`), and that is no play change
for OUR side (TECH_DEBT (d), found 2026-10-27 in SOP H22's sweep).

`utils.bridge.ws_frontend` used to validate a `/utm` pack and then store the RAW one; a real Showdown server battles
with the validator's `Teams.pack` of the sets it mutated. For our side that must be a no-op: the packs `Gen3Teambuilder`
builds for every team source we play (the pool, `away`, the registered panels) are the validator's sets. They are NOT
byte-identical strings (the validator spells ids in their display form, writes a fixed-gender species' gender and writes
the IV-derived `hpType`), so the proof is over what the engine reads: `showdown_pack.js`'s field-by-field comparison of
the set each string plays (ids, EVs, IVs, moves with aliases resolved, happiness, the IV-derived Hidden Power, the gender
as the sim resolves it, the nickname). A pack where it finds a difference is a PLAY CHANGE for that team, and this fails.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from main.anchors.config import AnchorConfigError, load_config
from main.anchors.peer_scripts.team_pack import showdown_packs
from main.anchors.session import build_team_source
from utils.bridge.team_validator import validate_teams_locally
from utils.bridge.ws_frontend import ShowdownFrontEnd, _Conn
from utils.paths import src_path

pytestmark = pytest.mark.integration

_BRIDGE_DIR = str(src_path("utils", "bridge"))


def _sources():
    """``(label, our-side team source spec)``: the pool, `away` and every registered panel, where the box has them."""
    from main.x35.kakuna import PANEL_CELLS

    out = [("home-pool", {"kind": "pool"})]
    try:
        out.append(("away", load_config().our_team_source("away")))
    except AnchorConfigError:
        pass
    for name, cell in sorted(PANEL_CELLS.items()):
        try:
            out.append((name, load_config(cell.path).our_team_source(cell.teamset)))
        except AnchorConfigError:
            pass
    return out


def test_the_pool_is_always_among_the_sources_proved():
    assert _sources()[0][0] == "home-pool"


@pytest.mark.slow       # ~30 s for the 719-team pool (two Node passes), 1-2 s per 72-team panel
@pytest.mark.parametrize("label, spec", _sources(), ids=[s[0] for s in _sources()])
def test_the_validators_repack_plays_the_same_sets_as_our_side_pack(label, spec):
    """Every pack our side may draw: the validator's re-pack (what the front end now stores) plays the set our raw pack
    played. Nothing is skipped: an empty source, a refused pack or a single field difference fails."""
    packs = list(build_team_source(spec, 1, []).packed_teams)
    assert packs, f"{label}: no packs built"
    verdicts = validate_teams_locally("gen3ou", packs)
    assert all(v["valid"] and v.get("packed") for v in verdicts), f"{label}: the validator refused one of our own packs"
    # showdown_pack.js: the validator's pack, and how the set the RAW string plays differs from it
    rows = showdown_packs(packs, packs, "gen3ou", _BRIDGE_DIR)
    assert [r["packed"] for r in rows] == [v["packed"] for v in verdicts], "one validator, two packs"
    changed = [(i, r["legacy_diffs"]) for i, r in enumerate(rows) if r["legacy_diffs"]]
    assert not changed, f"{label}: {len(changed)} of {len(packs)} packs PLAY DIFFERENTLY once re-packed, e.g. {changed[:3]}"


class _WS:
    def __init__(self):
        self.sent = []

    async def send(self, text):
        self.sent.append(text)


async def test_the_front_end_stores_the_repack_of_a_real_validator():
    """Real Node, real Showdown: a named Hidden Power with no IVs is rewritten by the validator (IVs and `hpType`), and the
    connection's team is THAT, as on a real server. Reverting `_cmd_utm` to store the raw pack fails here."""
    raw = "Salamence||leftovers|intimidate|hiddenpowerwater,earthquake,rockslide,doubleedge|Adamant|252,252,0,0,4,0||||100|"
    front = ShowdownFrontEnd(impl="node", validate_teams=True)
    conn = _Conn(ws=_WS(), guest_n=1)
    await front._cmd_trn(conn, "Alpha,0,assertion")
    await front._cmd_utm(conn, raw)
    (verdict,) = validate_teams_locally("gen3ou", [raw])
    assert verdict["valid"] and conn.team == verdict["packed"] != raw
    assert ",Water" in conn.team, conn.team          # the validator wrote Hidden Power's type
