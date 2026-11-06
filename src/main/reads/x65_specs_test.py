"""X65 ExIt-0's committed read specs (``designs/research_state/measurements/x65_exit0_2026-11-05/read_spec_*.json``, PREREG
§4 / §6, build B5): every registered number, each student on its read panel only, G-BANK's games, look 2's seeds and its
imports of the earlier looks' states, S′-LIVE's import pinned to its committed spec, and the PENDING fields that keep
each spec unplayable until the pin amendment fills them."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from utils.paths import repo_path

U = ["heldout72_hpfix", "forum72", "p72"]
STU = {"heldout72_hpfix": "s_h", "forum72": "s_f", "p72": "s_p"}
ENGINE = "port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129"

X65 = "designs/research_state/measurements/x65_exit0_2026-11-05"
SLIVE_SPEC = "designs/research_state/measurements/sprime_live_play_2026-11-01/read_spec.json"
#: the specs name the reads archive by absolute path, as S′-LIVE's does (the box's archive, ``~/gen3ai_archive``)
ARCHIVE = str(Path.home() / "gen3ai_archive")
BASE = f"{ARCHIVE}/x65_exit0_2026-11-05"


def _committed(name: str) -> Any:
    from main.reads.spec import load

    return load(repo_path(f"{X65}/{name}"), allow_pending=True)


@pytest.mark.parametrize("name,look,control", [("read_spec_look1.json", 1, "banked"),
                                               ("read_spec_look2.json", 2, "banked"),
                                               ("read_spec_look1_fresh.json", 1, "fresh"),
                                               ("read_spec_look2_fresh.json", 2, "fresh")])
def test_the_committed_specs_carry_the_prereg_numbers(name, look, control):
    """PREREG §4 / §6: the students on their read panels, z 2.576 / 2.0 and 1.96, the +2 pp bar, B 4,000 at
    default_rng(20261106), the 0.0005 boundary and 0.5 % drop cap, E4, the gate's B / seed, look 2's 600-674."""
    from main.reads.spec import PENDING, SpecError, load
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
    else:
        assert p["gbank"] is None
    for a in sp.anchor.values():                                       # S′-LIVE's import names its committed spec
        if a.imports is not None and a.imports.state.endswith("sprime_live_play_2026-11-01"):
            assert a.imports.spec_sha256 == sha256_file(repo_path(SLIVE_SPEC)) and a.imports.group == "u"
    assert sp.pending and all(x.endswith(("ref.path", "ref.sha256", "spec_sha256", "bank_manifest_sha256"))
                              for x in sp.pending)
    with pytest.raises(SpecError, match=PENDING):
        load(repo_path(f"{X65}/{name}"))                                # never planned while anything is PENDING


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
