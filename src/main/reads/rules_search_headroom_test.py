"""``main.reads`` rule ``search_headroom`` (S-U, ``design_search_u_headroom.md`` §3, §4.4; build B4): every class of
both primaries reachable, the phase-1 futility-only read of I, the exact-fraction rule-8 boundary, each VOID reason,
the bootstrap's determinism and documented scheme, the paired difference with a tie as ½, a one-sided deal excluded,
the discordance without its sign, and the spec-load parameter check. Synthetic cells, no game."""
from __future__ import annotations

import copy
import json
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pytest

from main.anchors.search_presets import PRESETS
from main.reads import rules as RU
from main.reads import rules_search_headroom as SH
from main.reads.assemble import SEARCH_PLAYED_KEYS, unit_row
from main.reads.spec import SpecError, parse

SHA = "a" * 64
ENGINE = "port=aaaaaaaaaaaaaaaa/10;env=bbbbbbbbbbbbbbbb/20"
HOME, U = "home72", ["heldout72_hpfix", "forum72", "p72"]
PANELS = [HOME, *U]
SEEDS, GAMES = range(4), range(4)
SEARCH = "r0s4@10"


def _params(**over: Any) -> Dict[str, Any]:
    p = {"group": "su", "searched": "pstar_s", "raw": "pstar", "home_panel": HOME, "u_panels": list(U), "phase": 2,
         "bars": {"no_u_premium_pp": 5.5, "dnc_b_pp": 4.3, "goal_b": 0.5}, "bootstrap": {"B": 2000, "seed": 20261027},
         "rule8_eps": 0.0005, "expected_search": SEARCH}
    p.update(over)
    return p


def _unit(label: str, ts: str, seed: int, game: int, result: str) -> Dict[str, Any]:
    searched = label == "pstar_s"
    u = {"unit": f"{label}__{ts}__{seed:02d}__g{game:02d}", "label": label, "panel": ts, "unit_seed": seed,
         "game": game, "counted": True, "wins": int(result == "win"), "ties": int(result == "tie"), "n": 1,
         "our_regime": f"greedy+search={SEARCH}" if searched else "greedy", "search": SEARCH if searched else None,
         "model_sha256": SHA, "engine": ENGINE, "games": [["ours_challenge", game + 1, "t.txt", result, 30]]}
    if searched:
        u["search_played"] = {"search": SEARCH, "our_search": SEARCH, "preset": PRESETS["r0s4"].as_dict(),
                              "searched": 30, "deviations": 3, "excluded": 0, "search_cpu_s": 180.0}
    return u


def _by_k(k: Dict[str, List[int]]) -> Callable[[str, int, int], "tuple[str, str]"]:
    """Unit (ts, seed) of statistic k[ts][seed] / 4: its first |k| games discordant in k's direction, the rest
    concordant (win / win, loss / loss alternating)."""
    def out(ts: str, seed: int, game: int) -> "tuple[str, str]":
        kk = k[ts][seed]
        if game < abs(kk):
            return ("loss", "win") if kk > 0 else ("win", "loss")
        return ("win", "win") if game % 2 else ("loss", "loss")
    return out


def _cells(outcome: Callable[[str, int, int], "tuple[str, str]"],
           mutate: Optional[Callable[[List[Dict[str, Any]]], None]] = None) -> Dict[str, Any]:
    units = []
    for ts in PANELS:
        for s in SEEDS:
            for g in GAMES:
                r, x = outcome(ts, s, g)
                units += [_unit("pstar", ts, s, g, r), _unit("pstar_s", ts, s, g, x)]
    if mutate:
        mutate(units)
    done = {(u["panel"], u["unit_seed"], u["game"], u["label"]) for u in units if u["counted"]}
    checked = sum(1 for (ts, s, g, lab) in done if lab == "pstar" and (ts, s, g, "pstar_s") in done)
    return {"models": {"pstar": {"sha256": SHA}, "pstar_s": {"sha256": SHA}},
            "anchor": {"su": {"units": units, "engine": ENGINE, "regime_id": "r" * 16, "teamsets": PANELS,
                              "unit_seeds": list(SEEDS), "unplayed": 0, "inconclusive": [],
                              "g_prefix": {"schema": "gen3_read_prefix_v1", "pairs": len(PANELS) * 16,
                                           "checked": checked, "pending": 0, "mismatches": []}}}}


SPEC = SimpleNamespace(models={"pstar": SimpleNamespace(ref=SimpleNamespace(sha256=SHA))})


def _read(cells: Dict[str, Any], **over: Any) -> Dict[str, Any]:
    p = _params(**over)
    doc = json.loads(json.dumps(RU.inputs("search_headroom", cells, p, SPEC)))
    return RU.decide("search_headroom", doc, p)


def _k(home: List[int], u: List[int]) -> Dict[str, List[int]]:
    return {HOME: home, **{ts: u for ts in U}}


HEAD = _k([-1, 1, -1, 1], [1, 3, 1, 3])        # G_home 0, G_U +0.5
LESS = _k([-1, 1, -1, 1], [-1, -3, -1, -3])    # G_U −0.5
FLAT = _k([0, 0, 0, 0], [0, 0, 0, 0])          # every bound exactly 0
WIDE = _k([-2, 2, -2, 2], [-2, 2, -2, 2])      # every interval wide around 0


# ------------------------------------------------------------------------------------------------ the classes
def test_u_headroom_at_phase_2_and_u_gain():
    d = _read(_cells(_by_k(HEAD)))["decision"]
    assert d["primary_a"]["classes"] == [SH.U_HEADROOM]
    assert d["primary_b"]["classes"] == [SH.U_GAIN, SH.GOAL_B]      # searched U score 7/8
    assert d["verdict"].startswith("A (I, phase 2): U-HEADROOM;")


def test_phase_1_reads_i_futility_only_a_would_be_headroom_is_undecided_and_proceeds():
    d = _read(_cells(_by_k(HEAD)), phase=1)["decision"]
    a = d["primary_a"]
    assert a["classes"] == [SH.UNDECIDED] and a["not_read_at_phase_1"] == [SH.U_HEADROOM]
    assert a["proceed_to_phase_2"] and not a["stop"] and "-> phase 2" in d["verdict"]
    assert d["primary_b"]["classes"] == [SH.U_GAIN, SH.GOAL_B], "G_U's classes are read (final) at phase 1"


def test_u_less_with_no_u_premium_stops_the_read_at_phase_1():
    d = _read(_cells(_by_k(LESS)), phase=1)["decision"]
    assert d["primary_a"]["classes"] == [SH.U_LESS, SH.NO_U_PREMIUM]
    assert d["primary_a"]["stop"] and not d["primary_a"]["proceed_to_phase_2"] and "-> STOP" in d["verdict"]
    assert d["primary_b"]["classes"] == [SH.DNC_B]
    d2 = _read(_cells(_by_k(LESS)), phase=2)["decision"]
    assert d2["primary_a"]["classes"] == [SH.U_LESS, SH.NO_U_PREMIUM] and "stop" not in d2["primary_a"]


def test_no_u_premium_alone_and_does_not_close_b_on_a_flat_read():
    out = _read(_cells(_by_k(FLAT)))
    d = out["decision"]
    assert out["estimates"]["I"]["ci95"] == [0.0, 0.0]
    assert d["primary_a"]["classes"] == [SH.NO_U_PREMIUM], "UB(I) = 0 sits ON the U-LESS bar: rule 8, not U-LESS"
    assert d["tests"]["UB(I) < 0"] == {"crosses": False, "near": True}
    assert d["primary_b"]["classes"] == [SH.DNC_B]


def test_undecided_on_both_primaries_when_the_intervals_are_wide():
    d = _read(_cells(_by_k(WIDE)))["decision"]
    assert d["primary_a"]["classes"] == [SH.UNDECIDED] and d["primary_b"]["classes"] == [SH.UNDECIDED]


def test_goal_b_reached_on_the_searched_arms_stratified_u_score():
    def outcome(ts: str, seed: int, game: int) -> "tuple[str, str]":
        return ("loss", "win") if game < 2 else ("win", "win") if game == 2 else ("loss", "loss")
    out = _read(_cells(outcome))
    g = out["goal_b_score"]
    assert g["searched"]["counts"]["forum72"] == [12, 16] and g["raw"]["counts"]["forum72"] == [4, 16]
    assert g["searched"]["ci95"][0] > 0.5
    assert SH.GOAL_B in out["decision"]["primary_b"]["classes"]
    assert g["newcombe_strat_searched_minus_raw"][0] == pytest.approx(0.5)


# ------------------------------------------------------------------------------------------------ rule 8
@pytest.mark.parametrize("bar_pp, holds", [(25.0, False), (25.04, False), (25.05, False), (25.06, True),
                                           (24.94, False), (25.10, True)])
def test_a_bound_within_eps_of_a_bar_is_undecided_by_exact_fractions(bar_pp, holds):
    # every U unit at +0.25 and home72 at 0: every bootstrap replicate of I is exactly 0.25
    out = _read(_cells(_by_k(_k([0, 0, 0, 0], [1, 1, 1, 1]))), bars={"no_u_premium_pp": bar_pp, "dnc_b_pp": 4.3,
                                                                     "goal_b": 0.5})
    assert out["estimates"]["I"]["ci95"] == [0.25, 0.25] and out["estimates"]["exact"]["I"] == "1/4"
    assert (SH.NO_U_PREMIUM in out["decision"]["primary_a"]["classes"]) is holds


def test_cmp_is_exact_and_eps_inclusive():
    assert SH.cmp(0.25, "0.2505", "0.0005", above=False) == (False, True)            # exactly eps (0.25 is exact): near
    assert SH.cmp(0.25, "0.25051", "0.0005", above=False) == (True, False)
    assert SH.cmp(0.25, "0.2495", 0.0005, above=True) == (False, True)
    # the float is taken EXACTLY: 0.1 + 0.2 lies above the decimal 0.3, so with eps 0 it crosses "above"
    assert SH.cmp(0.1 + 0.2, "0.3", 0, above=True) == (True, False)
    ci = {"I": [-0.1, 0.2], "G_U": [2.0 ** -11, 0.0429]}                          # 0.000488 and 0.0429: both near
    c = SH.classify(ci, None, {"no_u_premium_pp": 5.5, "dnc_b_pp": 4.3, "goal_b": 0.5}, 0.0005, 2)
    assert c["primary_b"]["classes"] == [SH.UNDECIDED], "both G_U bounds are within eps of their bars"


# ------------------------------------------------------------------------------------------------ estimates
def test_the_paired_difference_scores_a_tie_as_half():
    def outcome(ts: str, seed: int, game: int) -> "tuple[str, str]":
        if ts == HOME:
            return ("tie", "win") if game % 2 else ("loss", "tie")
        return ("win", "tie") if seed % 2 else ("tie", "tie")
    out = _read(_cells(outcome))
    ex = out["estimates"]["exact"]
    assert ex["G"][HOME] == "1/2" and ex["G"]["forum72"] == "-1/4" and ex["I"] == "-3/4"


def test_the_bootstrap_is_deterministic_and_is_the_documented_scheme():
    a, b = _read(_cells(_by_k(WIDE))), _read(_cells(_by_k(WIDE)))
    assert a["estimates"] == b["estimates"]
    c = _read(_cells(_by_k(WIDE)), bootstrap={"B": 2000, "seed": 1})
    assert c["estimates"]["I"]["ci95"] != a["estimates"]["I"]["ci95"]
    rng = np.random.default_rng(20261027)
    reps = {}
    for ts in [*U, HOME]:
        x = np.array([-0.5, 0.5, -0.5, 0.5])
        reps[ts] = x[rng.integers(0, 4, size=(2000, 4))].mean(axis=1)
    i = np.stack([reps[ts] for ts in U]).mean(axis=0) - reps[HOME]
    assert a["estimates"]["I"]["ci95"] == [float(v) for v in np.percentile(i, [2.5, 97.5])]


def test_g_u_interval_does_not_depend_on_the_home72_units():
    st = SH.unit_stats(SH.pair_deals(json.loads(json.dumps(RU.inputs("search_headroom", _cells(_by_k(WIDE)),
                                                                        _params(), SPEC))))[0])
    more = {**st, HOME: {**st[HOME], **{s: Fraction(1, 4) for s in range(4, 12)}}}
    a = SH.bootstrap(st, HOME, U, 2000, 20261027)
    b = SH.bootstrap(more, HOME, U, 2000, 20261027)
    assert a["G_U"] == b["G_U"] and a["I"] != b["I"]


def test_the_discordance_is_b_plus_c_and_never_its_sign():
    up = _read(_cells(_by_k(HEAD)))
    down = _read(_cells(_by_k(_k([1, -1, 1, -1], [-1, -3, -1, -3]))))
    assert set(up["discordance"]) == {"pairs", "discordant", "share"}
    assert up["discordance"] == down["discordance"] == {"pairs": 64, "discordant": 28, "share": 28 / 64}


def test_a_one_sided_deal_is_excluded_and_counted_never_imputed():
    def drop(units: List[Dict[str, Any]]) -> None:
        for u in units:
            if u["label"] == "pstar_s" and u["panel"] == "forum72" and u["unit_seed"] == 0 and u["game"] == 0:
                u.update(counted=False, our_regime="greedy+search=bogus", engine=None)   # uncounted: never read
    out = _read(_cells(_by_k(HEAD), mutate=drop))
    c = out["counts"]
    assert c["excluded_one_side"]["forum72"] == {"searched_only": 0, "raw_only": 1}
    assert c["pairs"]["forum72"] == 15 and c["uncounted_units"] == {"pstar_s": 1, "pstar": 0}
    assert out["decision"]["verdict"] != SH.VOID
    # forum72 seed 0 keeps games 1-3 of k = 1, all concordant -> 0 (not 1/4); seeds 1-3: 3/4, 1/4, 3/4
    assert out["estimates"]["exact"]["G"]["forum72"] == str(Fraction(7, 16))


def test_the_search_report_by_set():
    s = _read(_cells(_by_k(HEAD)))["search"]
    assert s["U"]["games"] == 48 and s["U"]["deviation_rate"] == 0.1 and s["home"]["cpu_s_per_searched_decision"] == 6.0


# ------------------------------------------------------------------------------------------------ VOID
def _set(pred: Callable[[Dict[str, Any]], bool], **kv: Any) -> Callable[[List[Dict[str, Any]]], None]:
    def m(units: List[Dict[str, Any]]) -> None:
        for u in units:
            if pred(u):
                u.update(copy.deepcopy(kv))
    return m


def _first(label: str) -> Callable[[Dict[str, Any]], bool]:
    return lambda u: u["label"] == label and u["panel"] == "p72" and u["unit_seed"] == 1 and u["game"] == 2


BAD_PRESET = dict(PRESETS["r0s4"].as_dict(), pikl_lambda=0.02)


@pytest.mark.parametrize("mutate, over, needle", [
    (_set(_first("pstar"), engine="port=cccccccccccccccc/10;env=bbbbbbbbbbbbbbbb/20"), {}, "span 2 engines"),
    (_set(_first("pstar_s"), engine=None), {}, "unknown engine"),
    (None, {"engine": "port=dddddddddddddddd/1;env=eeeeeeeeeeeeeeee/2"}, "not the registered"),
    (_set(_first("pstar"), model_sha256="b" * 64), {}, "played other bytes"),
    (_set(_first("pstar_s"), our_regime="greedy+search=r0s4@12"), {}, "search config"),
    (_set(_first("pstar_s"), search_played={"search": SEARCH, "our_search": SEARCH, "preset": BAD_PRESET}), {},
     "search config"),
    (_set(_first("pstar_s"), search_played=None), {}, "search config"),
    (_set(_first("pstar"), our_regime=f"greedy+search={SEARCH}"), {}, "raw arm"),
    (_set(_first("pstar"), search_played={"search": SEARCH}), {}, "raw arm"),
    (_set(_first("pstar"), games=[["ours_challenge", 3, "t.txt", "tie", 30]]), {}, "games.jsonl disagrees"),
])
def test_each_void_reason(mutate, over, needle):
    d = _read(_cells(_by_k(HEAD), mutate=mutate), **over)["decision"]
    assert d["verdict"] == SH.VOID and any(needle in r for r in d["reasons"]), d["reasons"]


def test_void_on_g_prefix_a_mismatch_not_run_or_an_unchecked_pair():
    cells = _cells(_by_k(HEAD))
    cells["anchor"]["su"]["g_prefix"]["mismatches"] = [{"deal": "forum72:0:3", "why": "p1's protocol differs"}]
    assert any(r.startswith("G-PREFIX: deal forum72:0:3") for r in _read(cells)["decision"]["reasons"])
    cells = _cells(_by_k(HEAD))
    cells["anchor"]["su"]["g_prefix"] = None
    assert "G-PREFIX: not run on the group" in _read(cells)["decision"]["reasons"]
    cells = _cells(_by_k(HEAD))
    cells["anchor"]["su"]["g_prefix"]["checked"] -= 1
    assert any("every pair must be checked" in r for r in _read(cells)["decision"]["reasons"])


def test_void_on_a_resolved_model_on_other_bytes_and_a_panel_with_one_unit():
    cells = _cells(_by_k(HEAD))
    cells["models"]["pstar_s"]["sha256"] = "c" * 64
    assert any("'pstar_s' resolved to" in r for r in _read(cells)["decision"]["reasons"])
    cells = _cells(_by_k(HEAD), mutate=_set(lambda u: u["panel"] == HOME and u["unit_seed"] > 0, counted=False))
    assert any(r.startswith("home72: 1 unit(s) hold a pair") for r in _read(cells)["decision"]["reasons"])


def test_the_unit_row_carries_the_searched_halfs_search_record_and_a_raw_row_none():
    class _U:
        name, label, teamset, unit_seed, game, games, model_sha256 = "x", "pstar_s", "forum72", 0, 3, 1, SHA
        search, belief_prior_mix = SEARCH, None
    rec = {"search": SEARCH, "preset": PRESETS["r0s4"].as_dict(), "searched": 10, "deviations": 1, "excluded": 0,
           "search_cpu_s": 60.0, "decisions_log": "/x"}
    s = {"status": "OK", "n": 1, "wins": 1, "cell": {"our_regime": f"greedy+search={SEARCH}"},
         "provenance": {"peer_report": {"our_search": SEARCH, "search": rec}}}
    row = unit_row(_U(), s)
    assert row["search_played"] == {**{k: rec[k] for k in SEARCH_PLAYED_KEYS}, "our_search": SEARCH}
    assert "search_played" not in unit_row(_U(), {"status": "OK", "n": 1, "cell": {"our_regime": "greedy"},
                                                  "provenance": {"peer_report": {}}})


# ------------------------------------------------------------------------------------------------ the spec check
def _spec(**over: Any) -> Dict[str, Any]:
    ref = {"kind": "pinned", "path": "/x/checkpoint_280000170_steps.zip", "sha256": SHA}
    g = {"opponent": "metamon:Kakuna", "teamsets": PANELS, "unit_seeds": "0-19", "per_game": True,
         "models": ["pstar", "pstar_s"]}
    g.update(over.pop("group_over", {}))
    return {"schema": "gen3_read_spec_v1", "read": "su_demo_2026-10-27",
            "models": {"pstar": {"arm": "pstar", "seed": 2601, "ref": ref},
                       "pstar_s": {"arm": "pstar_s", "seed": 2601, "ref": ref, "load": {"search": "r0s4"}}},
            "anchor": {"su": g}, "decision": {"rule": "search_headroom", "params": _params(**over)}}


def test_the_spec_check_accepts_the_registered_params():
    spec = parse(_spec())
    assert spec.rule == "search_headroom" and spec.params["expected_search"] == SEARCH


@pytest.mark.parametrize("over, needle", [
    ({"bogus": 1}, "unknown 'bogus'"),
    ({"group": "nope"}, "is not an anchor group"),
    ({"searched": "pstar"}, "exactly the searched label"),
    ({"expected_search": "r0s4"}, "spell it 'r0s4@10'"),
    ({"expected_search": "r0s4@12"}, "not expected_search"),
    ({"u_panels": ["heldout72_hpfix", "forum72"]}, "must be exactly group"),
    ({"u_panels": ["heldout72_hpfix", "forum72", "p72", "rc72b"]}, "must be exactly group"),
    ({"home_panel": "forum72"}, "home_panel not among them"),
    ({"phase": 3}, "phase"),
    ({"bars": {"no_u_premium_pp": 5.5, "dnc_b_pp": 4.3}}, "bars"),
    ({"bootstrap": {"B": 100, "seed": 1}}, "bootstrap"),
    ({"rule8_eps": 0.5}, "rule8_eps"),
])
def test_the_spec_check_refuses(over, needle):
    with pytest.raises(SpecError, match="search_headroom") as e:
        parse(_spec(**over))
    assert needle in str(e.value)


def test_the_rule_check_refuses_a_group_that_is_not_per_game():
    spec = parse(_spec())
    g = SimpleNamespace(**{**spec.anchor["su"].__dict__, "per_game": False})
    probs = SH.check(spec.params, models=spec.models, anchor={"su": g}, s={})
    assert any("is not per_game" in x for x in probs)


def test_the_spec_check_refuses_a_missing_param_and_a_searched_label_without_its_twin():
    doc = _spec()
    del doc["decision"]["params"]["rule8_eps"]
    with pytest.raises(SpecError, match="missing 'rule8_eps'"):
        parse(doc)
    doc = _spec()
    doc["models"]["pstar"]["ref"] = dict(doc["models"]["pstar"]["ref"], sha256="d" * 64)
    with pytest.raises(SpecError, match="raw twin"):
        parse(doc)
