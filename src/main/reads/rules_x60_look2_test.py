"""X60-T look 2's REGISTERED read spec (PREREG ``x60t_h_teams_2026-11-06`` §23, AM-X60T-2): the committed file is
:func:`look2_spec` exactly; it parses at main and passes rule ``x60``'s look-2 check; every import names the committed
earlier spec by its sha256; the three S regime ids it registers are re-derived here from main's own regime code and the
registry (R72: from the content digest §9 commits, since main never registers R72); the guards artefact is the committed
not-played stub, which reads every guard INCONCLUSIVE. No game, no state."""
from __future__ import annotations

import json
from typing import Any, Dict, List

from utils.hashing import sha256_file
from utils.paths import repo_path

MEAS = "designs/research_state/measurements"
X60T = f"{MEAS}/x60t_h_teams_2026-11-06"
LOOK2_SPEC = repo_path(f"{X60T}/read_spec_look2.json")
GUARDS_STUB = f"{X60T}/read/guards_look2_not_played.json"
PLAY_PIN = "f564c333f9437f6a636e9daf890535c2c300aeb2"
SIDE_PIN_X60T = "7fe15726c77cfa823aee78110444bf25c3c7370e"
REFPOP = f"{MEAS}/x35_breakers_stage_2026-10-17/reference_population_production.json"
REFPOP_CONTENT = "cef0960eac605f1dc026442d1089dc23654e23df91010feaaeea7890901e4519"
#: The earlier states and their specs' sha256 (each import names one; ``imports.check`` re-hashes the state's spec).
X60T_LOOK1 = ("~/gen3ai_archive/x60t_read/look1",
              "0f5d9a5fe18aa0b9545f66cc40759024d407618c0d7b389a13e27c7a94d0c57a", f"{X60T}/read_spec_look1.json")
X61P_LOOK1 = ("~/gen3ai_archive/x61p_n1p_read/look1",
              "1266daadeefe4142a2f2f339bbdedaf4e052c3f2ef3fb8a8b0d5550fb8729f03",
              f"{MEAS}/x61p_n1_production_2026-10-26/read_spec_look1.json")
X61P_LOOK2 = ("~/gen3ai_archive/x61p_n1p_read/look2",
              "f28654ccf31c3c3e6a0a3c3f9d61b904bf504f59d197b67baa3f17d510c2c620",
              f"{MEAS}/x61p_n1_production_2026-10-26/read_spec_look2.json")
#: The S regime ids (``main.h2h.play.plan_regime(250, "off", panel)`` at the play pin; re-derived at main below).
REGIME = {"home72": "d4de0fd7c16a2658", "forum72": "0c972bc06eab468f", "r72": "1bfe0b8a101ba8fb"}
#: R72 (§9): what main's regime code needs to derive its team set without the registry entry.
R72_DIGEST, R72_N = "10364442c1c3fe5216f46ab445455497dc350a9f9601fc7af9af1eb39a335501", 72
ON_SHA = {3501: "d576e096b496bc913736db98c7885b66ca03eddb7cccb013b27ab824770b8bb9",       # look 1's finals
          3502: "135017856da5939c56d4bcdf766d163d80439387851d4c3a5aafce4d18a68c6a",
          3503: "5979dbb18d7a60222a0cbb1ac68fd6984f754c28d04dfa8c546d9d2e3082faad"}
C0_SHA = {3501: "09a6a2df48509a59fd9aefe6d20d9722aad6717f1730c6dfd75274446b283303",
          3502: "7e9a9890e2b397de6d8b87588c597d7f53e39b84efd3d1080305f77dbf22feec",
          3503: "99096be8eccfc1cef4268ab4d6de15063a386cda475e72340212fdbf3a1b7e26",
          3504: "26835480f06840020880b3dd0e4de0558d9677ff1529360b99994dcf1dfa5fae",
          3505: "bb10cd96b4c5f7e79fcc8256884685053c2ac3c142f6a14c996f809e0b2626a0"}
D1_SHA = {3501: "3d5c9d45ce367eb50034d2a66dd8a6b43a8c71d086872f9ae4bf9bb0bbd7940c",
          3502: "dc3d5cd7d60a5f723f7323f73290fe673a893c3a6796d1428dcb38df066f8d39",
          3503: "30e4e79a38d83c6066c37ddc93e1807135f6acfcb70b59ede0e3e589a0296aec",
          3504: "fdaf6daa66a31c1559b034763d711607de652c4f693be91eaaef49e5e9eaf645",
          3505: "2e36697e57bb0facf81cad5dd79ff8bd606ac81d0ea2a8b3398e65d6ab4f2450"}
LOOK2 = tuple(range(3501, 3509))
CONTRAST = tuple(range(3501, 3506))
KAKUNA = "metamon:Kakuna"
NOTE = ("X60-T look 2 (AM-X60T-2, PREREG §23): 8 v 8. Δ_F on forum72 at 60 units (ON s3501-s3503 imported from look 1, "
        "C0 s3501-s3505 from X61P's two states, ON s3504-s3508 and C0 s3506-s3508 played); the determinism guard; "
        "goal (a) on home72; Δ_P (S home72, C0 s3501-s3505 imported), I_F (S forum72) and R72 (S r72, registered only "
        "at the play pin: main checks it through its regime id); REPORTED H / P72 / home72 at 7 units; T - D1 (5 v 5, "
        "D1 re-played at E3); the not-played guards stub; the VOID gate on all 21 finals with a pin per arm. Plan and "
        "play at the play pin; assemble / decide / record from a main checkout")


def _final(run: str) -> Dict[str, Any]:
    return {"kind": "final", "run": run, "min_step": 300000170, "quiet_s": 300}


def _pinned(run: str, sha: str) -> Dict[str, Any]:
    return {"kind": "pinned", "path": f"{run}/final_model.zip", "sha256": sha}


def _imp(src: tuple, group: str, models: List[str]) -> Dict[str, Any]:
    return {"state": src[0], "spec_sha256": src[1], "group": group, "models": models}


def _a(teamsets: List[str], seeds: str, models: List[str], imp: Any = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"opponent": KAKUNA, "teamsets": teamsets, "unit_seeds": seeds, "models": models}
    if imp is not None:
        d["import"] = imp
    return d


def _s(panel: str, family: str, models: List[str], imp: Any = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"refpop": REFPOP, "refpop_content_sha256": REFPOP_CONTENT, "panel": panel, "pairs": 500,
                         "batch_pairs": 100, "family": family, "models": models, "regime_id": REGIME[panel]}
    if imp is not None:
        d["import"] = imp
    return d


def look2_spec() -> Dict[str, Any]:
    """Look 2's spec as registered (PREREG §23.3). Labels: ``x60t_s<s>`` (ON), ``c0_s<s>`` (OFF), ``d1_s<s>`` (the
    contrast's D1). ON s3501-s3503 and every C0 / D1 final that exists are pinned by sha256; the eight finals X60Tb and
    X60T-C0b will train are the settled ``final`` rule (§13.1), as look 1's ON were."""
    on = [f"x60t_s{s}" for s in LOOK2]
    off = [f"c0_s{s}" for s in LOOK2]
    d1 = [f"d1_s{s}" for s in CONTRAST]
    models: Dict[str, Any] = {}
    for s in LOOK2:
        run = f"models/rb_x60t_s{s}"
        models[f"x60t_s{s}"] = {"arm": "x60t", "seed": s, "ref": _pinned(run, ON_SHA[s]) if s in ON_SHA else _final(run)}
    for s in LOOK2:
        run = f"models/rb_x35_c0_s{s}"
        models[f"c0_s{s}"] = {"arm": "c0", "seed": s, "ref": _pinned(run, C0_SHA[s]) if s in C0_SHA else _final(run)}
    for s in CONTRAST:
        models[f"d1_s{s}"] = {"arm": "d1", "seed": s, "ref": _pinned(f"models/rb_x35_d1wide_s{s}", D1_SHA[s])}
    new_c0 = off[5:]
    on_l1, c0a, c0b = on[:3], off[:3], off[3:5]
    hrep = ["heldout72_hpfix", "p72", "home72"]
    anchor = {
        "primary": _a(["forum72"], "0-59", on + new_c0, _imp(X60T_LOOK1, "primary", on_l1)),
        "primary_c0a": _a(["forum72"], "0-59", c0a, _imp(X61P_LOOK1, "primary", c0a)),
        "primary_c0b": _a(["forum72"], "0-59", c0b, _imp(X61P_LOOK2, "primary", c0b)),
        "determinism": _a(["forum72"], "0", ["c0_s3501"]),
        "home72": _a(["home72"], "0-6", on + new_c0),
        "p72": _a(["p72"], "0-6", on + new_c0),
        "h": _a(["heldout72_hpfix"], "0-6", on[5:] + new_c0),
        "h_contrast": _a(["heldout72_hpfix"], "0-29", on[:5]),
        "d1": _a(["heldout72_hpfix", "forum72"], "0-29", d1),
        "reported_c0a": _a(hrep, "0-6", c0a, _imp(X61P_LOOK1, "reported", c0a)),
        "reported_c0b": _a(hrep, "0-6", c0b, _imp(X61P_LOOK2, "reported", c0b)),
    }
    s = {
        "p": _s("home72", "x60t_p_look2_2026-11-07", on + new_c0),
        "p_c0a": _s("home72", "x60t_p_c0a_look2_2026-11-07", c0a, _imp(X61P_LOOK1, "p", c0a)),
        "p_c0b": _s("home72", "x60t_p_c0b_look2_2026-11-07", c0b, _imp(X61P_LOOK2, "p", c0b)),
        "i_f": _s("forum72", "x60t_i_f_look2_2026-11-07", on + off),
        "r72": _s("r72", "x60t_r72_look2_2026-11-07", on + off),
    }
    runs = {**{lab: f"models/rb_{lab}" for lab in on}, **{lab: f"models/rb_x35_{lab}" for lab in off},
            **{f"d1_s{x}": f"models/rb_x35_d1wide_s{x}" for x in CONTRAST}}
    params = {
        "look": 2, "on": on, "off": off,
        "primary": {"groups": ["primary", "primary_c0a", "primary_c0b"], "teamset": "forum72", "units": 60},
        "determinism_guard": {"group": "determinism", "against": "primary_c0a"},
        "goal_a": {"groups": ["home72"], "teamset": "home72", "units": 7},
        "p": {"groups": ["p", "p_c0a", "p_c0b"], "panel": "home72"},
        "i_f": {"groups": ["i_f"], "panel": "forum72"},
        "r72": {"groups": ["r72"], "panel": "r72"},
        "contrast": {"on": on[:5], "off": d1, "groups": ["primary", "h_contrast", "d1"],
                     "panels": ["heldout72_hpfix", "forum72"], "units": 30},
        "reported": ["home72", "p72", "h", "h_contrast", "reported_c0a", "reported_c0b"],
        "guards": {"path": GUARDS_STUB, "sha256": sha256_file(repo_path(GUARDS_STUB))},
        "void": {"k9b_dir": f"{MEAS}/x35_breakers_stage_2026-10-17/k9b", "runs": runs,
                 "pin": {"x60t": SIDE_PIN_X60T, "c0": "f4b25238d", "d1": "440b61c32"},
                 "dose": {"reference": "models/rb_x35_c0_s3501", "rel_tol": 1e-6}},
    }
    return {"schema": "gen3_read_spec_v1", "read": "x60t_look2_2026-11-07", "prereg": f"{X60T}/PREREG.md",
            "note": NOTE, "pin": PLAY_PIN, "models": models, "anchor": anchor, "s": s,
            "decision": {"rule": "x60", "params": params}}


def render(doc: Dict[str, Any]) -> str:
    """The committed file's bytes (look 1's form: indent 1, UTF-8)."""
    return json.dumps(doc, indent=1, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------------------------------------ the tests
def test_the_registered_look2_spec_is_the_builders_exactly():
    assert LOOK2_SPEC.read_text(encoding="utf-8") == render(look2_spec())


def test_the_look2_spec_parses_at_main_and_passes_rule_x60s_look2_check():
    from main.reads import rules_x60 as X
    from main.reads import spec as SP

    got = SP.load(LOOK2_SPEC)
    assert got.rule == "x60" and got.pin == PLAY_PIN and got.params["look"] == 2
    assert sorted(int(got.models[lab].seed) for lab in got.params["on"]) == list(X.LOOK_SEEDS[2])
    assert sorted(int(got.models[lab].seed) for lab in got.params["off"]) == list(X.LOOK_SEEDS[2])
    assert sorted(int(got.models[lab].seed) for lab in got.params["contrast"]["off"]) == list(X.CONTRAST_SEEDS)
    assert set(X.LOOK2_REQUIRED) <= set(got.params)
    # every final the read reads is gated, each arm at its own pin (REVIEW I m4)
    assert set(got.params["void"]["runs"]) == set(got.models) and set(got.params["void"]["pin"]) == {"x60t", "c0", "d1"}
    # the S lane plays in tier order (T1's P first, then T2's I_F and R72): runner._s_pending walks spec.s in order
    assert list(got.s) == ["p", "p_c0a", "p_c0b", "i_f", "r72"]


def test_every_import_names_the_committed_earlier_spec_by_its_sha256():
    for _state, sha, path in (X60T_LOOK1, X61P_LOOK1, X61P_LOOK2):
        assert sha256_file(repo_path(path)) == sha, path


def test_the_registered_s_regime_ids_are_main_s_own_derivation():
    """home72 / forum72 from main's registry; R72 (never on main) from §9's committed digest and n: the id the spec
    registers is the one ``assemble.panel_team_set`` relies on to pin R72's team set at main."""
    from agents.training.eval_ledger import schema as L
    from main import heldout_panels as HP
    from main.h2h import panel as PN
    from main.h2h.play import regime_for

    for panel in ("home72", "forum72"):
        assert regime_for(250, team_set=PN.panel_team_set_id(panel))["regime_id"] == REGIME[panel], panel
    ts = L.team_set_id({"builder": "Gen3Teambuilder", "kind": PN.KIND, "panel": "r72", "content_digest": R72_DIGEST,
                        "n": R72_N, "registry_schema": HP.SCHEMA, "draw": PN.DRAW_RULE})
    assert regime_for(250, team_set=ts)["regime_id"] == REGIME["r72"] and ts == "ts:a81d977d7a44b6f4"
    assert "r72" not in HP.panel_names()
    assert R72_DIGEST in repo_path(f"{X60T}/PREREG.md").read_text(encoding="utf-8")


def test_the_guards_stub_reads_every_guard_inconclusive():
    """§23.4: the guards are produced only on a Δ_F GAIN; until then the registered artefact holds none of them."""
    from main.reads import rules_x60 as X

    p = look2_spec()["decision"]["params"]
    doc = X._guard_doc(p["guards"])
    assert doc.get("error") is None and doc["doc"]["g_f"] is None and doc["doc"]["r2"] is None
    got = X._guards({"guards": doc, "seeds": {f"x60t_s{s}": s for s in LOOK2} | {f"c0_s{s}": s for s in LOOK2},
                     "sides": {"on": p["on"], "off": p["off"]}})
    assert {n: g["status"] for n, g in got.items()} == {"G-F": "INCONCLUSIVE", "G-E": "INCONCLUSIVE",
                                                       "G-C": "INCONCLUSIVE"}


def test_every_final_has_units_0_6_on_h_p72_and_home72_among_the_reported_groups():
    """FINDINGS 25 and 28 (AM-X60T-3, PREREG §24): §12's REPORTED Δ_H / Δ_P72 / Δ_home72 are 8 v 8 at units 0-6.
    The spec plays ON's H 0-6 at s3506-s3508 (group ``h``; the 420 games §17 omitted) and ON s3501-s3505's inside
    ``h_contrast``'s 0-29, and rule ``x60`` pools them per team set in code (``reported.pooled``), so every ON and OFF
    final must hold units 0-6 of each panel in exactly one reported group."""
    spec = look2_spec()
    p = spec["decision"]["params"]
    from main.x35.kakuna import parse_unit_seeds

    for panel in ("heldout72_hpfix", "p72", "home72"):
        for lab in p["on"] + p["off"]:
            holders = [g for g in p["reported"] if panel in spec["anchor"][g]["teamsets"]
                       and lab in spec["anchor"][g]["models"]]
            assert len(holders) == 1, (panel, lab, holders)
            assert set(range(7)) <= set(parse_unit_seeds(spec["anchor"][holders[0]]["unit_seeds"])), (panel, lab)
    assert spec["anchor"]["h"]["models"][:3] == [f"x60t_s{s}" for s in (3506, 3507, 3508)]
