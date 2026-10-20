"""PLANTED-EFFECT tests of stage 1's registered rules and read (``rules_s1``, ``read_s1``) and of the samplers' pure
parts (``samplers_s1.corpus_seen_moves``, ``peek``; ``corpus_s1.normalize_entry``). No archive, no engine: a synthetic
table. Run by hand before the read (as the maturity battery runs ``read_mat_test.py``):

    python -m pytest read_s1_test.py -q -p no:cacheprovider        (PYTHONPATH = a checkout's src)
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import read_s1 as RD  # noqa: E402
import rules_s1 as R  # noqa: E402
import s1_common as E  # noqa: E402

ARMS_ALL = ["O", "F", "FN", *E.ARMS, "F_rolls_w"]


def _table(n_battles=120, per=2, lift=None, seed=0, curve_slope=0.10):
    """A synthetic table: IMP_O ~ 0.06, every honest arm ~ 0, plus ``lift[arm]`` added to that arm (every scorer)."""
    rng = np.random.default_rng(seed)
    lift = lift or {}
    recs = []
    for b in range(n_battles):
        for k in range(per):
            base = rng.normal(0.0, 0.08)
            src = {"O": {"ok": True, "imp": {z: 0.06 + rng.normal(0, 0.05) for z in RD.SCORERS}}}
            for a in ARMS_ALL[1:]:
                imp = {z: base + rng.normal(0, 0.03) + lift.get(a, 0.0) for z in RD.SCORERS}
                imp["CRITIC"] = base + rng.normal(0, 0.03)
                src[a] = {"ok": True, "imp": imp}
            p = np.round(np.linspace(0, 1, 101), 2)
            dk = {"F_legal": {"M8|LEAF-K8": src["F_legal"]["imp"]["LEAF-K8"] + curve_slope * p,
                              "M8|TRUNC-4-K8": src["F_legal"]["imp"]["TRUNC-4-K8"] + curve_slope * p[::5]}}
            worlds = {a: {"qSM": 0.001, "U_exact": 0.1, "M_exact": 0.01, "hit": 0.2, "near4": 0.0,
                          "cost_build_cpu_s": 1.0, "cost_play_cpu_s": 10.0} for a in E.ARMS}
            recs.append({"id": f"s{b}_{k}", "battle": f"b{b}", "tie": False, "src": src, "dk": dk, "worlds": worlds,
                         "stratum": "bot" if b % 2 else "snapshot"})
    return {"recs": recs, "counts": {"complete": len(recs), "is_prefix": True,
                                     "arms": {a: {"ok": len(recs)} for a in E.ARMS}}, "ids_sha256": "x"}


def test_a_null_series_reads_null_and_triggers_nothing():
    res = RD.read(_table())
    for x in E.PRIMARY:
        assert res["PRIMARY"][x]["verdict"] == "NULL", (x, res["PRIMARY"][x]["delta"])
        assert not res["PRIMARY"][x]["MATERIAL"] and not res["PRIMARY"][x]["SEARCH-ENABLED"]
    assert res["TRIGGER_1b"]["trigger"] is False
    assert res["CALIBRATED"] == "CALIBRATED"


def test_a_planted_lift_pays_is_material_and_triggers_1b():
    res = RD.read(_table(lift={"C_ladder": 0.04}))
    c = res["PRIMARY"]["C_ladder"]
    assert c["verdict"] == "PAYS" and c["MATERIAL"]
    assert c["r"]["mean"] == pytest.approx(0.04 / 0.06, abs=0.15)
    assert res["TRIGGER_1b"]["trigger"] and any("C_ladder" in w for w in res["TRIGGER_1b"]["why"])
    assert res["PRIMARY"]["B"]["verdict"] == "NULL"
    # the lift sits on the true-world dose curve at p ≈ 0.4 (slope 0.10): TODAY moves off 0, q_SM 0.001 is EXCLUDED
    assert c["TODAY"]["p_eq"] == pytest.approx(0.4, abs=0.1)
    assert c["today_contains_qSM"] == "EXCLUDES" and res["CALIBRATED"] == "NOT CALIBRATED"


def test_a_planted_harm_reads_hurts():
    res = RD.read(_table(lift={"NS": -0.04}))
    assert res["PRIMARY"]["NS"]["verdict"] == "HURTS"


def test_search_enabled_needs_both_halves():
    # every honest IMP lifted (the arm beats the policy) but its TRUNC-4 vs its own critic is not: not enabled
    t = _table(lift={"C_pool": 0.05})
    for r in t["recs"]:
        r["src"]["C_pool"]["imp"]["CRITIC"] = r["src"]["C_pool"]["imp"]["TRUNC-4-K8"]
    assert RD.read(t)["PRIMARY"]["C_pool"]["SEARCH-ENABLED"] is False
    assert RD.read(_table(lift={"C_pool": 0.05}))["PRIMARY"]["C_pool"]["SEARCH-ENABLED"] is True


def test_secondary_contrasts_read_the_planted_one():
    res = RD.read(_table(lift={"F_rolls_w": 0.04}))
    assert res["SECONDARY"]["F_rolls_w − F_legal"]["verdict"] == "PAYS"
    assert res["SECONDARY"]["F_rolls_w − F_rolls"]["verdict"] == "PAYS"
    assert res["SECONDARY"]["F_legal − F_off"]["verdict"] == "NULL"


def test_an_exclusion_drops_the_state_from_its_contrasts_only():
    t = _table()
    for r in t["recs"][:30]:
        r["src"]["C_pool"] = {"ok": False, "excluded": True}
    res = RD.read(t)
    assert res["PRIMARY"]["C_pool"]["n"] == len(t["recs"]) - 30
    assert res["PRIMARY"]["B"]["n"] == len(t["recs"])


def test_rule8_margins():
    assert R.delta_verdict([0.0006, 0.02], [0.0006, 0.02]) == "PAYS"
    assert R.delta_verdict([0.0005, 0.02], [0.001, 0.02]) != "PAYS"                    # on the edge: not clear
    assert R.delta_verdict([-0.0094, 0.0094], [-0.05, 0.05]) == "NULL"
    assert R.delta_verdict([-0.0096, 0.0094], [-0.05, 0.05]) == "NOT DETECTED"          # within the margin of ±0.01
    assert R.material("PAYS", [0.2506, 0.9]) and not R.material("PAYS", [0.2504, 0.9])
    assert R.contains([0.0, 0.05], 0.0001) == "CONTAINS"                                # the clamp floor is open
    assert R.contains([0.01, 0.05], 0.0001) == "EXCLUDES"
    assert R.contains([0.0003, 0.05], 0.0001) == "BOUNDARY"
    assert R.calibrated({"a": "CONTAINS", "b": "BOUNDARY"}) == "UNDECIDED"


def test_peek_leaves_the_world_rng_untouched():
    import samplers_s1 as SS

    a, b = np.random.default_rng(7), np.random.default_rng(7)
    p1 = SS.peek(a, "x").random()
    assert a.random() == b.random()
    assert SS.peek(np.random.default_rng(7), "x").random() == p1
    assert SS.peek(np.random.default_rng(7), "y").random() != p1


def _orig_fill(rng, prior, keep, n_slots, lever, truth, hp_type, hp_rule="any"):
    """A stand-in for F's draw: fills the free slots from ``prior`` in order."""
    have = set(keep.values())
    rest = [m for m in prior if m not in have]
    slots = [keep.get(i) for i in range(n_slots)]
    for i in range(n_slots):
        if slots[i] is None:
            slots[i] = rest.pop(0)
    return slots, {"how": "drawn"}


def test_corpus_seen_moves_hidden_power_is_the_screens():
    import samplers_s1 as SS

    rng = np.random.default_rng(0)
    truth = ["earthquake", "rockslide", "hiddenpowerbug", "dragondance"]
    keep = {0: "earthquake"}
    c = Counter()
    # the screen's IVs carry HP Grass, the source set has none: HP added, one source move cut at random
    slots, f = SS.corpus_seen_moves(rng, ["earthquake", "rockslide", "dragondance", "taunt"], ["roar"], keep, 4, truth,
                                    "grass", "force", _orig_fill, c, "t")
    assert slots[0] == "earthquake" and "hiddenpowergrass" in slots and len(set(slots)) == 4
    assert c["hp_added"] == 1 and c["moves_cut"] == 1 and f["how"] == "corpus"
    # the screen carries NO Hidden Power, the source holds one: dropped, the slot filled by F's draw
    c = Counter()
    slots, f = SS.corpus_seen_moves(rng, ["earthquake", "rockslide", "hiddenpowerfire", "dragondance"], ["roar"], keep, 4,
                                    truth, None, "none", _orig_fill, c, "t")
    assert sorted(slots) == ["dragondance", "earthquake", "roar", "rockslide"] and c["hp_dropped"] == 1
    # the source IS the truth modulo the HP type, and the screen carries Bug: an exact match
    c = Counter()
    slots, f = SS.corpus_seen_moves(rng, ["earthquake", "rockslide", "hiddenpowerfire", "dragondance"], ["roar"], keep, 4,
                                    truth, "bug", "force", _orig_fill, c, "t")
    assert f["match"] is True and f["true_hits"] == 3 and sorted(slots) == sorted(truth)


def test_normalize_entry_fills_the_hidden_power_ivs():
    import corpus_s1 as CS

    e = "Raikou||Leftovers|Pressure|Substitute,Thunderbolt,HiddenPower,CalmMind|Modest|,,,252,4,252|N||||,Grass,,,,"
    out = CS.normalize_entry(e, lambda s: "Raikou")
    from agents.info_levers.packed import hp_type_of, parse_set, set_ivs, typed_moves

    f = parse_set(out)
    assert hp_type_of(set_ivs(f)) == "grass" and "hiddenpowergrass" in typed_moves(f)
    assert ",Grass" not in out
    # explicit non-default IVs decide: the stale field goes, the IVs stay
    e2 = "Raikou||Leftovers|Pressure|HiddenPower,Thunderbolt|Modest||N|30,30,30,30,30,||,Grass,,,,"
    f2 = parse_set(CS.normalize_entry(e2, lambda s: "Raikou"))
    assert set_ivs(f2) == [30, 30, 30, 30, 30, 31] and typed_moves(f2)[0] == "hiddenpower" + hp_type_of(set_ivs(f2))
