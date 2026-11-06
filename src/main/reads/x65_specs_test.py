"""X65 ExIt-0's committed read specs (``designs/research_state/measurements/x65_exit0_2026-11-05/read_spec_*.json``, PREREG
§4 / §6, build B5): every registered number, each student on its read panel only, G-BANK's games, look 2's seeds and its
imports of the earlier looks' states, S′-LIVE's import pinned to its committed spec, and the values PREREG Amendment 3
filled (the students' exported zips and sha256s, the play pin, each earlier look's committed spec sha256)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from utils.paths import repo_path

U = ["heldout72_hpfix", "forum72", "p72"]
STU = {"heldout72_hpfix": "s_h", "forum72": "s_f", "p72": "s_p"}
ENGINE = "port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129"
BANK_SHA256 = "d13fd55f10ba649ad8e728ce1585e4203c4200148d772919eb175d3273992b7a"

X65 = "designs/research_state/measurements/x65_exit0_2026-11-05"
SLIVE_SPEC = "designs/research_state/measurements/sprime_live_play_2026-11-01/read_spec.json"
#: the specs name the reads archive by absolute path, as S′-LIVE's does (the box's archive, ``~/gen3ai_archive``)
ARCHIVE = str(Path.home() / "gen3ai_archive")
BASE = f"{ARCHIVE}/x65_exit0_2026-11-05"


#: PREREG Amendment 3: the play pin and each student's exported zip and sha256 (``export.json``, ``bit_identical``)
PIN = "320a535d4"
STUDENTS = {
    "s_h": (f"{BASE}/students/s_h/model/x65_S_H.zip", "7af676dd0be5f536494de45fab4a4a9a7da02f3c2e1b3058bbfb8a46685f3f44"),
    "s_f": (f"{BASE}/students/s_f/model/x65_S_F.zip", "e400efb5819d4a77b7c9fc97710832e8139f2fca27bdb32d3d9794baad407f17"),
    "s_p": (f"{BASE}/students/s_p/model/x65_S_P.zip", "ec767f51c793df797a451a44b4f75df2a017303e62e1ca1067c25fde005e1ac3"),
}
#: each look's state and the committed spec it was planned from (an import names the earlier spec FILE's sha256)
STATE_SPEC = {f"{BASE}/look1": "read_spec_look1.json", f"{BASE}/look1_fresh": "read_spec_look1_fresh.json"}


def _committed(name: str) -> Any:
    from main.reads.spec import load

    return load(repo_path(f"{X65}/{name}"))                             # refuses any PENDING value left


@pytest.mark.parametrize("name,look,control", [("read_spec_look1.json", 1, "banked"),
                                               ("read_spec_look2.json", 2, "banked"),
                                               ("read_spec_look1_fresh.json", 1, "fresh"),
                                               ("read_spec_look2_fresh.json", 2, "fresh")])
def test_the_committed_specs_carry_the_prereg_numbers(name, look, control):
    """PREREG §4 / §6: the students on their read panels, z 2.576 / 2.0 and 1.96, the +2 pp bar, B 4,000 at
    default_rng(20261106), the 0.0005 boundary and 0.5 % drop cap, E4, the gate's B / seed, look 2's 600-674."""
    from utils.hashing import sha256_file

    sp = _committed(name)
    p = sp.params
    assert (sp.rule, p["look"], p["control"]) == ("distill_play", look, control)
    assert p["u_panels"] == U and p["students"] == STU and p["raw"] == "pstar"
    assert p["seeds"] == {"1": "300-344", **({"2": "600-674"} if look == 2 else {})}
    assert (p["z_go"], p["z_null"], p["null_bar_pp"]) == ({"1": 2.576, "2": 2.0}, 1.96, 2.0)
    assert p["bootstrap"] == {"B": 4000, "seed": 20261106} and (p["rule8_eps"], p["max_drop_frac"]) == (0.0005, 0.005)
    assert p["engine"] == ENGINE and p["delivery"] == {"path": f"{BASE}/delivery.json", "B": 4000, "seed": 20261105}
    assert sp.models["pstar"].ref.sha256 == "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
    assert [sp.models[s].seed for s in ("s_h", "s_f", "s_p")] == [6511, 6512, 6513]
    for ts, lab in STU.items():                                        # each student plays its read panel only
        assert {tuple(a.teamsets) for a in sp.anchor.values() if lab in a.models} == {(ts,)}
    seeds2 = set(range(600, 675))
    played2 = {s for a in sp.anchor.values() if a.imports is None for s in a.unit_seeds if s >= 600}
    assert played2 == (seeds2 if look == 2 else set())
    if control == "banked":
        assert p["gbank"] == {"group": p["gbank"]["group"], "origin": 300, "min_identical": 133, "n": 135}
        gb = sp.anchor[p["gbank"]["group"]]
        assert gb.games == {s: ((s - 300) % 20,) for s in range(300, 345)} and set(gb.teamsets) == set(U)
        if look == 1:
            # REVIEW R-m6: the anchor queue runs unit seeds outermost, then groups in spec order
            # (runner._anchor_todo), so G-BANK, first in spec order and on every seed the students play, is spread
            # through the look (each seed's 3 replays, then that seed's 60 student games), never played all first
            order = list(sp.anchor)
            assert order[0] == p["gbank"]["group"] and len(order) == 4
            assert all(tuple(sp.anchor[g].unit_seeds) == tuple(range(300, 345)) for g in order)
            assert all(STU[sp.anchor[g].teamsets[0]] in sp.anchor[g].models for g in order[1:])
    else:
        assert p["gbank"] is None
    for a in sp.anchor.values():                                       # S′-LIVE's import names its committed spec
        if a.imports is not None and a.imports.state.endswith("sprime_live_play_2026-11-01"):
            assert a.imports.spec_sha256 == sha256_file(repo_path(SLIVE_SPEC)) and a.imports.group == "u"
        elif a.imports is not None:                                    # an earlier look: its committed spec's bytes
            assert a.imports.spec_sha256 == sha256_file(repo_path(f"{X65}/{STATE_SPEC[a.imports.state]}"))
    # the registered bank (PREREG Amendment 2): the manifest the students train on and the delivery gate reads
    assert p["bank_manifest_sha256"] == BANK_SHA256
    # PREREG Amendment 3: nothing PENDING, the play pin, each student's exported bytes
    assert sp.pending == () and sp.pin == PIN
    assert {lab: (sp.models[lab].ref.path, sp.models[lab].ref.sha256) for lab in STUDENTS} == STUDENTS


def test_look2_imports_each_earlier_look_from_its_registered_state():
    sp = _committed("read_spec_look2.json")
    src = {g: (a.imports.state, a.imports.group, a.imports.models) for g, a in sp.anchor.items() if a.imports}
    assert src["gbank1"] == (f"{BASE}/look1", "gbank", ("pstar",))
    assert src["ctl1"] == (f"{ARCHIVE}/sprime_live_play_2026-11-01", "u", ("pstar",))
    assert {src[g] for g in ("h1", "f1", "p1")} == {(f"{BASE}/look1", g[0], (STU[ts],))
                                                    for g, ts in (("h1", U[0]), ("f1", U[1]), ("p1", U[2]))}
    fresh = _committed("read_spec_look2_fresh.json")
    assert {a.imports.state for a in fresh.anchor.values() if a.imports} == {f"{BASE}/look1", f"{BASE}/look1_fresh"}
    assert {a.imports.state for a in _committed("read_spec_look1_fresh.json").anchor.values()} == {f"{BASE}/look1"}
