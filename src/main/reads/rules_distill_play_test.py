"""``main.reads`` rule ``distill_play`` (X65 ExIt-0, ``x65_exit0_2026-11-05`` PREREG §1, §4, §6, §10): a constructed case
for every class (NOT-DELIVERED; GO at look 1; CONTINUE at look 1 that is GO at look 2, the z by look; NULL at looks 1
and 2; UNDECIDED), GO checked before NULL, the boundary (a bound within 0.0005 of 0 and of +2 pp), the worst-case
imputation adverse to each class, the drop cap, every VOID reason, G-BANK's 133-of-135 threshold and its whole-game
comparison, the delivery gate file's checks, the rule's spec check, and ``by_hand.py`` agreeing with the rule on every
constructed case. Synthetic inputs, no game."""
from __future__ import annotations

import copy
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pytest

from main.reads import rules as RU
from main.reads import rules_distill_play as DP
from utils.paths import repo_path

U = ["heldout72_hpfix", "forum72", "p72"]
STU = {"heldout72_hpfix": "s_h", "forum72": "s_f", "p72": "s_p"}
RAW = "pstar"
ENGINE = "port=e00a212a102f5ce2/94;env=3d0d5c9c0f5d3c23/129"
SHA = {"pstar": "a" * 64, "s_h": "b" * 64, "s_f": "c" * 64, "s_p": "d" * 64}
MANIFEST = "e" * 64
SEEDS = list(range(300, 306))
P = {"look": 1, "u_panels": U, "students": STU, "raw": RAW, "groups": {ts: [ts[:1]] for ts in U},
     "seeds": {"1": "300-305"}, "control": "banked",
     "gbank": {"group": "gbank", "origin": 300, "min_identical": 16, "n": 18}, "engine": ENGINE,
     "z_go": {"1": 2.576, "2": 2.0}, "z_null": 1.96, "null_bar_pp": 2.0,
     "bootstrap": {"B": 1000, "seed": 20261106}, "rule8_eps": 0.0005, "max_drop_frac": 0.025,
     "delivery": {"path": "/x/delivery.json", "B": 4000, "seed": 20261105}, "bank_manifest_sha256": MANIFEST}
Outcome = Callable[[str, int, int], Tuple[str, str]]


def _by_hand():
    path = repo_path("designs", "research_state", "measurements", "x65_exit0_2026-11-05", "by_hand.py")
    spec = importlib.util.spec_from_file_location("x65_by_hand", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


BH = _by_hand()


def _unit(label: str, ts: str, seed: int, game: int, result: str, *, role: str = "deal") -> Dict[str, Any]:
    return {"unit": DP.unit_name(label, ts, seed, game), "label": label, "panel": ts, "unit_seed": seed, "game": game,
            "counted": True, "wins": int(result == "win"), "ties": int(result == "tie"), "n": 1,
            "our_regime": "greedy", "search": None, "search_played": None, "belief_prior_mix": None,
            "model_sha256": SHA[label], "engine": ENGINE, "games": [["ours_challenge", 1, "t.txt", result, 40]],
            "regime_id": f"rid_{ts}", "peer_code": "their=9cc9caacde85f8ea", "policy_mode": "eval",
            "peer_pack": "legacy", "imported_from": None, "cell_dir": None, "group": ts[:1], "role": role,
            "status": DP.COUNTED}


def _wl(table: Sequence[Tuple[int, int]]) -> Outcome:
    """Seed i (+ the panel's index, rotating) has its first w games student-win / raw-loss, the next l the reverse,
    and the rest both win."""
    def out(ts: str, seed: int, game: int) -> Tuple[str, str]:
        w, l = table[(SEEDS.index(seed if seed < 600 else seed - 300) + U.index(ts)) % len(table)]
        return ("win", "loss") if game < w else ("loss", "win") if game < w + l else ("win", "win")
    return out


def _doc(k: Outcome, *, look: int = 1, seeds2: Sequence[int] = (), delivered: bool = True,
         gbank_identical: Optional[int] = 18, mutate: Optional[Callable[[Dict[str, Any]], None]] = None
         ) -> Dict[str, Any]:
    seeds = {s: 1 for s in SEEDS} | {s: 2 for s in seeds2}
    units, deals = [], []
    for ts in U:
        for s in sorted(seeds):
            for g in range(20):
                ys, yr = k(ts, s, g)
                units += [_unit(STU[ts], ts, s, g, ys), _unit(RAW, ts, s, g, yr)]
                deals.append({"deal": f"{ts}:{s}:{g}", "panel": ts, "unit_seed": s, "game": g, "look": seeds[s],
                              "student": {"unit": DP.unit_name(STU[ts], ts, s, g), "status": DP.COUNTED},
                              "raw": {"unit": DP.unit_name(RAW, ts, s, g), "status": DP.COUNTED}})
    gb = None
    if gbank_identical is not None:
        comps = [{"deal": f"{ts}:{s}:{s - 300}", "replay": "r", "banked": "b", "identical": True, "why": None}
                 for ts in U for s in SEEDS]
        for c in comps[gbank_identical:]:
            c.update(identical=False, why="differs")
        gb = {"group": "gbank", "origin": 300, "n": 18, "min_identical": 16, "compared": comps}
    doc = {"schema": DP.SCHEMA, "look": look, "u_panels": list(U), "students": dict(STU), "raw": RAW,
           "control": "banked", "seeds": {"1": SEEDS, **({"2": list(seeds2)} if seeds2 else {})},
           "engine_registered": ENGINE, "want_sha256": dict(SHA), "resolved_sha256": dict(SHA), "groups": {},
           "units": units, "deals": deals, "gbank": gb,
           "delivery": {"path": "/x", "sha256": "f" * 64, "problems": [], "delivered": delivered,
                        "boundary": False, "pooled": 0.03 if delivered else -0.01, "lb": 0.01 if delivered else -0.02,
                        "per_panel": {}}}
    if mutate:
        mutate(doc)
    return doc


def _p(**over: Any) -> Dict[str, Any]:
    p = copy.deepcopy(P)
    p.update(over)
    return p


def _decide(doc: Dict[str, Any], **over: Any) -> Dict[str, Any]:
    p = _p(**{"look": doc["look"], "seeds": {"1": "300-305", **({"2": "600-605"} if doc["look"] == 2 else {})},
              **over})
    out = DP.decide(doc, p)
    hand = BH.recompute(json.loads(json.dumps(doc)), p)                 # by hand, on the same inputs (JSON-round-tripped)
    assert hand["class"] == out["decision"]["class"], (hand, out["decision"])
    if "estimates" in out:
        for e in ("go_imputed", "null_imputed", "paired"):
            assert hand["estimates"][e]["G"] == out["estimates"][e]["G"]
            assert hand["estimates"][e]["G_all"] == out["estimates"][e]["G_all"]
            assert hand["estimates"][e]["se"] == out["estimates"][e]["se"]
        assert (hand["lb_exact"], hand["ub_exact"]) == (out["bounds"]["lb_exact"], out["bounds"]["ub_exact"])
        assert hand["boundary"] == out["decision"]["boundary"]
    return out


STRONG = _wl([(6, 1), (5, 0), (7, 2), (4, 0), (6, 0), (5, 1)])          # Ĝ ≈ +24 pp, small SE
LOSING = _wl([(0, 4), (1, 3), (0, 2), (1, 5), (0, 3), (2, 4)])          # Ĝ ≈ −14 pp
MIDDLE = _wl([(3, 0), (0, 2), (2, 0), (0, 1), (1, 0), (0, 0)])          # Ĝ ≈ +2.5 pp, wide


def _between(table_i: int) -> Outcome:
    """Ĝ / SE between 2.0 and 2.576 (measured below): GO at look 2's z, not at look 1's."""
    def out(ts: str, seed: int, game: int) -> Tuple[str, str]:
        i = SEEDS.index(seed if seed < 600 else seed - 300) + U.index(ts)
        w = (i * table_i) % 7
        return ("win", "loss") if game < w else ("loss", "win") if game < w + 2 else ("win", "win")
    return out


# ------------------------------------------------------------------------------------------------ the classes
def test_not_delivered_reads_no_game_and_needs_no_finished_deal():
    def unplayed(doc):
        for d in doc["deals"]:
            d["student"]["status"] = DP.MISSING
        doc["units"] = [u for u in doc["units"] if u["label"] == RAW]
    out = _decide(_doc(STRONG, delivered=False, mutate=unplayed))
    assert out["decision"]["class"] == DP.NOT_DELIVERED and out["students_counted_units"] == 0
    # delivered, the same unplayed students are an unfinished look: VOID, never NOT-DELIVERED
    out = _decide(_doc(STRONG, mutate=unplayed))
    assert out["decision"]["class"] == DP.VOID and "unfinished" in out["decision"]["reasons"][0]


def test_go_at_look_1():
    out = _decide(_doc(STRONG))
    assert out["decision"]["class"] == DP.GO and not out["decision"]["boundary"]
    assert out["bounds"]["z_go"] == 2.576 and out["decision"]["lb"] > 0


def test_a_ratio_between_the_two_zs_continues_at_look_1_and_is_go_at_look_2():
    k = _between(3)
    one = _decide(_doc(k))
    e = one["estimates"]["go_imputed"]
    assert 2.05 < e["G_all_float"] / e["se"] < 2.5                        # the constructed case's precondition
    assert one["decision"]["class"] == DP.CONTINUE and one["bounds"]["z_go"] == 2.576
    # the same deals read as look 2's cumulative pool: the same estimate and SE, and z = 2.0 decides
    two = _decide(_doc(k, look=2))
    assert two["decision"]["class"] == DP.GO and two["bounds"]["z_go"] == 2.0


def test_null_at_look_1_and_at_look_2():
    assert _decide(_doc(LOSING))["decision"]["class"] == DP.NULL
    out = _decide(_doc(LOSING, look=2, seeds2=range(600, 606)))
    assert out["decision"]["class"] == DP.NULL and out["look1"]["class"]["class"] == DP.NULL
    assert out["counts"]["p72"]["deals"] == 240                         # look 2 reads look 1's deals too


def test_undecided_at_look_2_is_continue_at_look_1():
    one = _decide(_doc(MIDDLE))
    assert one["decision"]["class"] == DP.CONTINUE
    lb, ub = one["decision"]["lb"], one["decision"]["ub"]
    assert lb < 0 and ub > 0.02
    two = _decide(_doc(MIDDLE, look=2, seeds2=range(600, 606)))
    assert two["decision"]["class"] == DP.UNDECIDED


def test_go_is_checked_before_null_and_the_boundary_never_crosses():
    p = _p()
    F = Fraction
    assert DP.classify(F("0.01"), F("0.015"), 1, p)["class"] == DP.GO          # both bars cleared: GO first
    for lb, cls, near in ((F("0.0005"), DP.CONTINUE, True), (F("-0.0005"), DP.CONTINUE, True),
                          (F("0.0005001"), DP.GO, False), (F("-0.0005001"), DP.CONTINUE, False)):
        got = DP.classify(lb, F("0.05"), 1, p)
        assert (got["class"], got["boundary"]) == (cls, near), lb
    for ub, cls, near in ((F("0.0195"), DP.UNDECIDED, True), (F("0.0205"), DP.UNDECIDED, True),
                          (F("0.0194999"), DP.NULL, False), (F("0.0205001"), DP.UNDECIDED, False)):
        got = DP.classify(F("-0.01"), ub, 2, p)
        assert (got["class"], got["boundary"]) == (cls, near), ub
    assert DP.classify(F("0.0004"), F("0.019"), 1, p)["class"] == DP.NULL       # GO near, NULL clear: NULL, flagged


# ------------------------------------------------------------------------------------------------ imputation, drops
def _drop(ts: str, seed: int, game: int, side: str) -> Callable[[Dict[str, Any]], None]:
    def m(doc):
        lab = STU[ts] if side == "student" else RAW
        for d in doc["deals"]:
            if (d["panel"], d["unit_seed"], d["game"]) == (ts, seed, game):
                d[side]["status"] = DP.EXHAUSTED
        for u in doc["units"]:
            if u["unit"] == DP.unit_name(lab, ts, seed, game):
                u.update(counted=False, status=DP.EXHAUSTED)
    return m


@pytest.mark.parametrize("side,go,null", [("student", Fraction(-2), Fraction(0)), ("raw", Fraction(0), Fraction(2))])
def test_a_drop_is_imputed_adverse_to_each_class(side, go, null):
    """A deal both arms won (2D = 0): the student missing is 0 − 2 for GO and 2 − 2 for NULL; the raw missing is
    2 − 2 for GO and 2 − 0 for NULL (in units of ½). The paired estimate leaves it out."""
    base = _decide(_doc(STRONG))
    out = _decide(_doc(STRONG, mutate=_drop("p72", 305, 19, side)))
    g0 = Fraction(base["estimates"]["go_imputed"]["G"]["p72"])
    n = 120
    assert Fraction(out["estimates"]["go_imputed"]["G"]["p72"]) == g0 + go / 2 / n
    assert Fraction(out["estimates"]["null_imputed"]["G"]["p72"]) == g0 + null / 2 / n
    assert Fraction(out["estimates"]["paired"]["G"]["p72"]) == g0 * n / (n - 1)
    assert out["counts"]["p72"][f"{side}_missing"] == 1 and out["estimates"]["paired"]["deals"]["p72"] == 119


def test_both_sides_missing_is_minus_one_for_go_and_plus_one_for_null():
    m1, m2 = _drop("forum72", 300, 19, "student"), _drop("forum72", 300, 19, "raw")
    out = _decide(_doc(STRONG, mutate=lambda d: (m1(d), m2(d))))
    base = _decide(_doc(STRONG))
    g0 = Fraction(base["estimates"]["go_imputed"]["G"]["forum72"])
    assert Fraction(out["estimates"]["go_imputed"]["G"]["forum72"]) == g0 - Fraction(1, 120)
    assert Fraction(out["estimates"]["null_imputed"]["G"]["forum72"]) == g0 + Fraction(1, 120)
    assert out["counts"]["forum72"]["both_missing"] == 1


def test_drops_above_the_cap_void_the_look_and_at_the_cap_do_not():
    three = [_drop("heldout72_hpfix", 301, g, "raw") for g in (17, 18, 19)]           # 3 / 120 = 0.025: at the cap
    out = _decide(_doc(STRONG, mutate=lambda d: [m(d) for m in three]))
    assert out["decision"]["class"] == DP.GO
    four = three + [_drop("heldout72_hpfix", 302, 19, "student")]
    out = _decide(_doc(STRONG, mutate=lambda d: [m(d) for m in four]))
    assert out["decision"]["class"] == DP.VOID and "above max_drop_frac" in " ".join(out["decision"]["reasons"])


# ------------------------------------------------------------------------------------------------ VOID
def _set(**kv: Any) -> Callable[[Dict[str, Any]], None]:
    def m(doc):
        u = next(x for x in doc["units"] if x["label"] == "s_f")
        u.update(kv)
    return m


@pytest.mark.parametrize("mutate,needle", [
    (_set(engine="port=x/1;env=y/2"), "engine"),
    (_set(engine=None), "engine"),
    (_set(model_sha256="9" * 64), "other bytes"),
    (lambda d: d["resolved_sha256"].update(s_p="9" * 64), "resolved to"),
    (_set(our_regime="greedy+search=r0s4_rolls@10"), "not greedy"),
    (_set(search="r0s4_rolls@10"), "not greedy"),
    (_set(belief_prior_mix=0.25), "not greedy"),
    (_set(games=[["ours_challenge", 1, "t.txt", "tie", 40]]), "games.jsonl"),
    (_set(regime_id="other"), "regime_id"),
    (_set(peer_code="their=0000000000000000"), "peer_code"),
    (_set(policy_mode="train"), "policy_mode"),
    (lambda d: d["deals"][0]["raw"].update(status=DP.PENDING), "unfinished"),
    (lambda d: d["deals"][5]["student"].update(status=DP.MISSING), "unfinished"),
])
def test_every_void_reason(mutate, needle):
    out = _decide(_doc(STRONG, mutate=mutate))
    assert out["decision"]["class"] == DP.VOID and needle in " ".join(out["decision"]["reasons"]), out["decision"]
    assert _decide(_doc(STRONG))["decision"]["class"] == DP.GO                         # and passes without it


def test_a_panel_with_one_seed_is_void():
    def one_seed(doc):
        doc["deals"] = [d for d in doc["deals"] if d["panel"] != "p72" or d["unit_seed"] == 300]
    out = _decide(_doc(STRONG, mutate=one_seed))
    assert out["decision"]["class"] == DP.VOID and "fewer than 2 unit seeds" in " ".join(out["decision"]["reasons"])


def test_gbank_passes_at_its_threshold_and_voids_one_below():
    """The registered 133 of 135, on a full-size G-BANK (45 seeds × 3 panels)."""
    def gb(k):
        comps = [{"deal": f"{ts}:{s}:{(s - 300) % 20}", "replay": "r", "banked": "b", "identical": i < k,
                  "why": None if i < k else "differs"} for i, (ts, s) in enumerate((t, s) for t in U
                                                                                    for s in range(300, 345))]
        return {"group": "gbank", "origin": 300, "n": 135, "min_identical": 133, "compared": comps}
    assert DP.gbank_summary(gb(133))["passed"] and not DP.gbank_summary(gb(132))["passed"]
    ok = _decide(_doc(STRONG, mutate=lambda d: d.update(gbank=gb(133))))
    assert ok["decision"]["class"] == DP.GO and ok["gbank"]["identical"] == 133
    bad = _decide(_doc(STRONG, mutate=lambda d: d.update(gbank=gb(132))))
    assert bad["decision"]["class"] == DP.VOID and "fresh raw arm" in " ".join(bad["decision"]["reasons"])


def test_an_unfinished_gbank_replay_voids_and_a_fresh_control_has_no_gbank():
    def unfinished(doc):
        doc["gbank"]["compared"][0].update(identical=None, why="the replay is unfinished")
    out = _decide(_doc(STRONG, mutate=unfinished))
    assert out["decision"]["class"] == DP.VOID and "unfinished" in " ".join(out["decision"]["reasons"])
    out = _decide(_doc(STRONG, gbank_identical=None))
    assert out["decision"]["class"] == DP.GO and out["gbank"] is None


def test_a_delivery_file_problem_is_void_before_anything_else():
    out = _decide(_doc(STRONG, mutate=lambda d: d["delivery"].update(problems=["schema 'x'"])))
    assert out["decision"]["class"] == DP.VOID and out["decision"]["reasons"] == ["delivery gate: schema 'x'"]


# ------------------------------------------------------------------------------------------------ the delivery file
def _delivery(tmp: Path, **over: Any) -> Dict[str, Any]:
    doc = {"schema": DP.DELIVERY_SCHEMA, "bank_manifest_sha256": MANIFEST,
           "students": {ts: {"zip_sha256": SHA[STU[ts]]} for ts in U},
           "per_panel": {ts: {"n_rows": 30000, "n_deals": 900, "agree_student": 0.7, "agree_pstar": 0.63,
                              "delta": 0.07} for ts in U},
           "pooled": 0.07, "lb": 0.05, "B": 4000, "seed": 20261105, "delivered": True, "boundary": False,
           "reported": {}}
    doc.update(over)
    path = tmp / "delivery.json"
    path.write_text(json.dumps(doc))
    return DP.read_delivery({"path": str(path), "B": 4000, "seed": 20261105},
                            students={ts: SHA[STU[ts]] for ts in U}, u_panels=U, bank_manifest=MANIFEST, eps=0.0005)


def test_a_good_delivery_file_has_no_problem(tmp_path):
    got = _delivery(tmp_path)
    assert got["problems"] == [] and got["delivered"] is True and len(got["sha256"]) == 64
    assert _delivery(tmp_path, lb=-0.01, delivered=False)["problems"] == []
    assert _delivery(tmp_path, lb=0.0004, delivered=False, boundary=True)["problems"] == []


@pytest.mark.parametrize("over,needle", [
    ({"schema": "gen3_x65_delivery_v0"}, "schema"),
    ({"extra": 1}, "keys"),
    ({"B": 2000}, "B 2000"),
    ({"seed": 1}, "seed 1"),
    ({"bank_manifest_sha256": "1" * 64}, "bank_manifest_sha256"),
    ({"students": {ts: {"zip_sha256": "1" * 64} for ts in U}}, "zip_sha256"),
    ({"students": {"forum72": {"zip_sha256": SHA["s_f"]}}}, "students"),
    ({"per_panel": {}}, "per_panel"),
    ({"delivered": "yes"}, "delivered"),
    ({"lb": -0.01}, "are not its LB"),                                     # delivered True with LB below 0
    ({"lb": 0.0004}, "are not its LB"),                                    # within rule8_eps of 0: not delivered
])
def test_a_delivery_file_that_does_not_match_is_named(tmp_path, over, needle):
    got = _delivery(tmp_path, **over)
    assert any(needle in x for x in got["problems"]), got["problems"]


def test_a_missing_delivery_file_is_a_problem(tmp_path):
    got = DP.read_delivery({"path": str(tmp_path / "none.json"), "B": 4000, "seed": 20261105},
                           students={}, u_panels=U, bank_manifest=MANIFEST, eps=0.0005)
    assert "missing" in got["problems"][0]


# ------------------------------------------------------------------------------------------------ inputs from cells
START = {"formatid": "gen3ou", "p1": {"name": "A", "team": "x"}, "p2": {"name": "B", "team": "y"}, "seed": [1, 2]}


def _capture(cell: Path, lines: List[str], start: Optional[Dict[str, Any]] = None) -> None:
    (cell / "capture").mkdir(parents=True, exist_ok=True)
    (cell / "capture" / "battle-gen3ou-1.json").write_text(json.dumps({
        "commands": ["START " + json.dumps(start or START)], "chunks": [["p1", "\n".join(lines)],
                                                                         ["p2", "\n".join(lines)]]}))


LINES = ["|player|p1|A", "|t:|1700000000", "|turn|1", "|move|p1a: X|Surf|p2a: Y", "|turn|2", "|win|A"]


def test_the_whole_game_comparison_ignores_t_lines_only(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    _capture(a, LINES)
    _capture(b, [x if not x.startswith("|t:|") else "|t:|1800000000" for x in LINES])
    row = lambda c: {"status": DP.COUNTED, "cell_dir": str(c)}               # noqa: E731
    assert DP.gbank_compare(row(b), row(a))["identical"] is True
    c = tmp_path / "c"
    _capture(c, LINES[:-1] + ["|win|B"])                                       # the LAST line differs: past any turn
    got = DP.gbank_compare(row(c), row(a))
    assert got["identical"] is False and "line 4" in got["why"]
    d = tmp_path / "d"
    _capture(d, LINES, dict(START, seed=[9, 9]))
    assert "START" in DP.gbank_compare(row(d), row(a))["why"]
    assert DP.gbank_compare({"status": DP.EXHAUSTED, "cell_dir": str(a)}, row(a))["identical"] is False
    assert DP.gbank_compare({"status": DP.PENDING, "cell_dir": str(a)}, row(a))["identical"] is None
    assert DP.gbank_compare(row(a), {"status": DP.EXHAUSTED, "cell_dir": str(a)})["identical"] is False


def _cells(tmp: Path) -> Dict[str, Any]:
    """Look-1 cells: per panel group (the student played here, the raw arm imported) and the G-BANK group, over two
    seeds; each unit's capture under ``tmp``."""
    def row(lab, ts, s, g, *, counted=True, imported=False):
        u = _unit(lab, ts, s, g, "win" if lab != RAW else "loss")
        cell = tmp / ("bank" if imported else "here") / lab / u["unit"]
        _capture(cell, LINES)
        u.update(counted=counted, cell_dir=str(cell), imported_from="/bank" if imported else None)
        for k in ("group", "role", "status"):
            u.pop(k)
        return u
    anchor: Dict[str, Any] = {}
    for ts in U:
        units = [row(STU[ts], ts, s, g) for s in (300, 301) for g in range(20)]
        units += [row(RAW, ts, s, g, imported=True) for s in (300, 301) for g in range(20)]
        anchor[ts[:1]] = {"units": units, "exhausted_units": [], "engine": ENGINE, "regime_id": "r",
                          "regime_ids": {}, "unplayed": 0, "inconclusive": []}
    anchor["gbank"] = {"units": [dict(row(RAW, ts, s, s - 300), cell_dir=str(tmp / "rep" / f"{ts}{s}"))
                                 for ts in U for s in (300, 301)], "exhausted_units": [], "engine": ENGINE,
                       "regime_id": "r", "regime_ids": {}, "unplayed": 0, "inconclusive": []}
    for ts in U:
        for s in (300, 301):
            _capture(tmp / "rep" / f"{ts}{s}", LINES)
    return {"models": {lab: {"sha256": sha} for lab, sha in SHA.items()}, "anchor": anchor}


def _spec() -> Any:
    return SimpleNamespace(models={lab: SimpleNamespace(ref=SimpleNamespace(sha256=sha)) for lab, sha in SHA.items()})


def test_inputs_status_pairs_and_gbank_from_cells(tmp_path):
    cells = _cells(tmp_path)
    (tmp_path / "delivery.json").write_text("{}")
    p = _p(seeds={"1": "300-301"}, gbank={"group": "gbank", "origin": 300, "min_identical": 6, "n": 6},
           delivery={"path": str(tmp_path / "delivery.json"), "B": 4000, "seed": 20261105})
    h = cells["anchor"]["h"]["units"]
    imp = next(u for u in h if u["label"] == RAW and u["unit_seed"] == 301 and u["game"] == 3)
    imp["counted"] = False                                         # an imported raw drop (S′-LIVE OP-7's kind)
    own = next(u for u in h if u["label"] == "s_h" and u["unit_seed"] == 301 and u["game"] == 4)
    own["counted"] = False                                         # retries left: pending
    spent = next(u for u in h if u["label"] == "s_h" and u["unit_seed"] == 301 and u["game"] == 5)
    spent["counted"] = False
    cells["anchor"]["h"]["exhausted_units"] = [spent["unit"]]
    h.remove(next(u for u in h if u["label"] == "s_h" and u["unit_seed"] == 300 and u["game"] == 7))
    doc = DP.inputs(cells, p, _spec())
    st = {(d["panel"], d["unit_seed"], d["game"]): (d["student"]["status"], d["raw"]["status"]) for d in doc["deals"]}
    assert len(doc["deals"]) == 3 * 2 * 20
    assert st[("heldout72_hpfix", 301, 3)] == (DP.COUNTED, DP.EXHAUSTED)
    assert st[("heldout72_hpfix", 301, 4)] == (DP.PENDING, DP.COUNTED)
    assert st[("heldout72_hpfix", 301, 5)] == (DP.EXHAUSTED, DP.COUNTED)
    assert st[("heldout72_hpfix", 300, 7)] == (DP.MISSING, DP.COUNTED)
    assert doc["gbank"] and all(c["identical"] for c in doc["gbank"]["compared"]) and len(doc["gbank"]["compared"]) == 6
    assert doc["delivery"]["problems"]                             # "{}" is not the schema
    t = DP.tally(doc)
    assert t["heldout72_hpfix"]["unfinished"] == 2 and t["heldout72_hpfix"]["raw_missing"] == 1
    assert t["heldout72_hpfix"]["student_missing"] == 1
    # every counted pair's two STARTs were compared (REVIEW R-m5); a one-sided deal is not compared
    sm = {(d["panel"], d["unit_seed"], d["game"]): d["start"] for d in doc["deals"]}
    assert sm[("heldout72_hpfix", 301, 3)] is None and sm[("heldout72_hpfix", 300, 7)] is None
    assert sum(1 for v in sm.values() if v is not None) == 3 * 2 * 20 - 4
    assert all(v["match"] for v in sm.values() if v is not None)


def test_inputs_refuse_a_deal_read_from_two_groups(tmp_path):
    cells = _cells(tmp_path)
    cells["anchor"]["h2"] = copy.deepcopy(cells["anchor"]["h"])
    p = _p(seeds={"1": "300-301"}, groups={"heldout72_hpfix": ["h", "h2"], "forum72": ["f"], "p72": ["p"]})
    with pytest.raises(RU.RuleInputError, match="two groups"):
        DP.inputs(cells, p, _spec())


# ------------------------------------------------------------------------------------------------ the spec check
def _spec_doc(**over: Any) -> Dict[str, Any]:
    ref = lambda lab: {"kind": "pinned", "path": f"/m/{lab}.zip", "sha256": SHA[lab]}     # noqa: E731
    g = lambda ts, models, **kw: dict({"opponent": "metamon:Kakuna", "teamsets": [ts] if isinstance(ts, str)  # noqa: E731
                                       else ts, "unit_seeds": "300-302", "per_game": True, "models": models}, **kw)
    imp = {"state": "/bank", "spec_sha256": "1" * 64, "group": "u", "models": [RAW]}
    doc = {"schema": "gen3_read_spec_v1", "read": "x65_test_2026-11-05",
           "models": {lab: {"arm": lab, "seed": 1, "ref": ref(lab)} for lab in SHA},
           "anchor": {"gbank": g(U, [RAW], games={str(s): [s - 300] for s in (300, 301, 302)}),
                      **{ts[:1]: g(ts, [STU[ts], RAW], **{"import": imp}) for ts in U}},
           "decision": {"rule": "distill_play", "params": _p(seeds={"1": "300-302"}, gbank={
               "group": "gbank", "origin": 300, "min_identical": 8, "n": 9})}}
    doc["decision"]["params"].update(over)
    return doc


def test_the_rule_check_takes_the_look1_shape():
    from main.reads.spec import parse

    sp = parse(_spec_doc())
    assert sp.rule == "distill_play" and sp.anchor["gbank"].games == {300: (0,), 301: (1,), 302: (2,)}


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["anchor"]["gbank"].update(games={"300": [1], "301": [1], "302": [2]}), "mod 20"),
    (lambda d: d["anchor"]["h"].update(teamsets=U), "plays its read panel"),
    (lambda d: d["anchor"]["h"].pop("import"), "imports the raw arm"),
    (lambda d: d["decision"]["params"].update(control="fresh"), "gbank: null"),
    (lambda d: d["decision"]["params"]["gbank"].update(n=135), "gbank.n"),
    (lambda d: d["decision"]["params"].update(seeds={"1": "300-303"}), "cover unit seeds"),
    (lambda d: d["decision"]["params"].update(look=2), "seeds"),
    (lambda d: d["decision"]["params"].update(bank_manifest_sha256="x"), "bank_manifest_sha256"),
    (lambda d: d["decision"]["params"]["groups"].update(p72=["p", "f"]), "not only"),
    (lambda d: d["anchor"].update(extra=dict(d["anchor"]["p"])), "read by nothing"),
    (lambda d: d["models"]["s_h"].update(load={"belief_prior_mix": 0.5}), "greedy with no load"),
])
def test_the_rule_check_refuses(mutate, needle):
    from main.reads.spec import SpecError, parse

    doc = _spec_doc()
    mutate(doc)
    with pytest.raises(SpecError, match=needle):
        parse(doc)



# ------------------------------------------------------------------------------------------------ the pairing (R-m5)
def test_a_pair_is_one_deal_only_if_its_two_starts_agree(tmp_path):
    """REVIEW R-m5: a student unit and its raw unit are paired by (panel, seed, game); their captures' START must agree
    on the format, the seed, both packed teams and our core side (the players' names may differ)."""
    row = lambda c: {"status": DP.COUNTED, "cell_dir": str(c)}               # noqa: E731
    a, b, c, d, e = (tmp_path / x for x in "abcde")
    _capture(a, LINES)
    _capture(b, LINES, {**START, "p1": {"name": "Other", "team": "x"}})
    _capture(c, LINES, {**START, "p2": {"name": "B", "team": "z"}})
    _capture(d, LINES, {**START, "seed": [1, 3]})
    assert DP.start_match(row(a), row(b)) == {"match": True, "why": None}
    assert DP.start_match(row(a), row(c))["match"] is False and "p2.team" in DP.start_match(row(a), row(c))["why"]
    assert DP.start_match(row(a), row(d))["match"] is False
    assert DP.start_match(row(a), row(e))["match"] is False                     # no capture: unreadable, no match
    assert DP.start_match(row(a), {"status": DP.EXHAUSTED, "cell_dir": str(a)}) is None
    assert DP.start_match(row(a), None) is None


def test_a_pair_whose_starts_differ_voids_the_look_by_rule_and_by_hand():
    def moved(doc):
        for x in doc["deals"]:
            x["start"] = {"match": True, "why": None}
        doc["deals"][7]["start"] = {"match": False, "why": "the START documents differ in ['seed']"}
    out = _decide(_doc(STRONG, mutate=moved))                    # _decide also holds by_hand.py to the same class
    assert out["decision"]["class"] == DP.VOID and "pairing" in " ".join(out["decision"]["reasons"])
    ok = _decide(_doc(STRONG, mutate=lambda doc: [x.update(start={"match": True, "why": None}) for x in doc["deals"]]))
    assert ok["decision"]["class"] == DP.GO
