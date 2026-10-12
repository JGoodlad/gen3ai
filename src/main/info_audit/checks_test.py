"""The audit checks on SYNTHETIC worlds: each check passes on honest input and FAILS on its planted defect."""
from types import SimpleNamespace

from agents.info_levers import spec as S
from main.info_audit import checks as C


def _worlds(facts, samplers=()):
    return SimpleNamespace(facts=facts, samplers=list(samplers), teams=[], logs=[])


def _seen(slot, **kw):
    f = {"slot": slot, "species": "x", "seen": True,
         "moves": {"how": "drawn", "source": "", "match": False, "true_hits": 0, "expected_true_hits": 0.5,
                   "slots": 2},
         "item": {"how": "drawn", "source": "", "match": False},
         "ability": {"how": "revealed", "source": "", "match": True},
         "spread": {"how": "drawn", "source": "", "match": False}}
    for k, v in kw.items():
        f[k] = dict(f[k], **v)
    return f


def _unseen(slot, how="drawn", match=False):
    return {"slot": slot, "species": "y", "seen": False, "true_mon": False,
            "species_fact": {"how": how, "source": "truth" if how == "told" else "", "match": match},
            "moves": {"how": "drawn", "source": "", "match": False}, "item": {"how": "drawn", "source": "", "match": False},
            "ability": {"how": "drawn", "source": "", "match": False}, "spread": {"how": "drawn", "source": "", "match": False}}


def test_provenance_passes_honest_facts_and_fails_each_planted_violation():
    off = S.parse("off")
    assert C.provenance(off, _worlds([[_seen(0), _unseen(1)]]))["pass"]
    # a fact TOLD under a spec with no lever for it (a builder that keeps the item true and says so)
    v = C.provenance(off, _worlds([[_seen(0, item={"how": "told", "source": "truth", "match": True})]]))
    assert not v["pass"] and v["examples"][0]["why"] == "told without a lever"
    # a mass-1 truth lever left undone
    v = C.provenance(S.parse("item:seen"), _worlds([[_seen(0)]]))
    assert not v["pass"] and v["examples"][0]["why"] == "a mass-1 lever not applied"
    # a mass-1 truth lever whose value is not the truth
    v = C.provenance(S.parse("item:seen"), _worlds([[_seen(0, item={"how": "told", "source": "truth", "match": False})]]))
    assert not v["pass"]
    # a revealed fact that is not the truth
    v = C.provenance(off, _worlds([[_seen(0, ability={"how": "revealed", "match": False})]]))
    assert not v["pass"] and v["examples"][0]["why"] == "a revealed fact not the truth"
    # a species told without a species lever
    assert not C.provenance(off, _worlds([[_unseen(1, how="told", match=True)]]))["pass"]
    assert C.provenance(S.parse("species"), _worlds([[_unseen(1, how="told", match=True)]]))["pass"]


def test_the_leak_read_passes_the_honest_expectation_and_fails_a_planted_peek():
    honest = C.Leakage()
    peek = C.Leakage()
    for i in range(400):
        # honest: 1 true move in 2 slots drawn per world, expectation 0.5 each
        honest.add(_worlds([[_seen(0, moves={"true_hits": i % 2})]],
                           [{"steps": 0, "failed": False, "prop_stats": {"item": [1, int(i % 4 == 0), 0.25]}}]))
        # planted: the world's moves and the proposals always hold the truth
        peek.add(_worlds([[_seen(0, moves={"true_hits": 2})]],
                         [{"steps": 0, "failed": False, "prop_stats": {"item": [1, 1, 0.25]}}]))
    hv = {v["check"]: v for v in honest.verdicts()}
    pv = {v["check"]: v for v in peek.verdicts()}
    assert hv["leak:moves:seen:drawn"]["pass"] and hv["leak:item:seen:proposals"]["pass"]
    assert abs(hv["leak:moves:seen:drawn"]["z"]) < 1.0 and abs(hv["leak:item:seen:proposals"]["z"]) < 1.0
    assert not pv["leak:moves:seen:drawn"]["pass"] and pv["leak:moves:seen:drawn"]["z"] > 20
    assert not pv["leak:item:seen:proposals"]["pass"] and pv["leak:item:seen:proposals"]["z"] > 20
    assert honest.table()["seen:ability:revealed"] == {"n": 400, "truth_match": 1.0}


def test_the_dose_read_holds_a_told_move_lever_to_its_mixture():
    def told(hits, expect):
        return _seen(0, moves={"how": "told", "source": "truth", "true_hits": hits, "expected_true_hits": expect})

    good, diluted = C.Leakage(), C.Leakage()
    for i in range(400):
        good.add(_worlds([[told(1 + (i % 2), 1.5)]]))           # D50-like: 1.5 expected true hits of 2 slots
        diluted.add(_worlds([[told(1 if i % 4 else 2, 1.5)]]))   # PLANT: the mass applied twice (≈ 1.25 hits)
    gv = {v["check"]: v for v in good.verdicts()}["dose:moves:seen:told"]
    dv = {v["check"]: v for v in diluted.verdicts()}["dose:moves:seen:told"]
    assert gv["pass"] and abs(gv["z"]) < 1.0
    assert not dv["pass"] and dv["z"] < -6                      # -8.16, exactly (no randomness here)


def test_template_identity_catches_a_slot_rendered_one_off():
    team = "]".join(["Skarmory||leftovers|keeneye|spikes,roar,drillpeck,toxic|Impish|252,,4,,252,||||||",
                     "Zapdos||leftovers|pressure|thunderbolt,hiddenpowerice,toxic,rest|Modest|||||||"])
    cmds = [["p1", "move surf"], ["p2", "move roar"], ["p2", "switch Zapdos"]]
    good = SimpleNamespace(tmpl=[["p1", "move surf"], ["p2", ("MOVE", 0, 1)], ["p2", ("SLOT", 1)]], team=team, cmds=cmds)
    assert C.template_identity(good)["pass"]
    bad = SimpleNamespace(tmpl=[["p1", "move surf"], ["p2", ("MOVE", 0, 2)], ["p2", ("SLOT", 1)]], team=team, cmds=cmds)
    v = C.template_identity(bad)
    assert not v["pass"] and v["mismatches"] == 1 and v["examples"][0]["i"] == 1


def test_merge_sums_counts_and_keeps_the_worst_verdict():
    a = {"check": "x", "pass": True, "n": 2, "examples": [1]}
    b = {"check": "x", "pass": False, "n": 3, "examples": [2]}
    m = C.merge([a, b])
    assert m == {"check": "x", "pass": False, "n": 5, "examples": [1, 2]}
