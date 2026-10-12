"""The information-lever audit suite on a FIXED small corpus of committed-bank decisions (early turns: few seen mons,
so the honest screen is cheap): every check passes on the real builder, and each planted defect is caught.

The larger corpus is the slow tier (``info_audit_slow_test.py``)."""
import pytest

from main.info_audit import __main__ as A
from main.info_audit import checks as C

pytestmark = [pytest.mark.sim, pytest.mark.integration]

SPECS = ("oracle", "pilot", "off", "F+M")


@pytest.fixture(scope="module")
def corpus():
    ds = A.select(2, min_turn=3, max_turn=6)
    assert len(ds) == 2
    return ds


@pytest.fixture(scope="module")
def report(corpus):
    return A.run(corpus, SPECS, 3, log=lambda m: None)


def test_every_check_passes_on_the_real_builder(report):
    bad = {k: v for k, v in report["checks"].items() if not v["pass"]}
    assert not bad, bad
    for s in ("oracle", "pilot"):
        assert report["specs"][s]["built"] == 2 and not report["specs"][s]["excluded"]
    assert "oracle_identity" in report["checks"] and report["checks"]["oracle_identity"]["worlds"] == 6
    assert report["checks"]["log_consistency"]["moves_agree"] > 0


def test_the_pilot_keeps_seen_facts_true_and_the_honest_builder_draws_them(report):
    pilot, off = report["leakage"]["pilot"]["table"], report["leakage"]["off"]["table"]
    told = [k for k in pilot if k.startswith("seen:") and k.endswith(":told") and k.split(":")[1] != "moves"]
    assert told and all(pilot[k]["truth_match"] == 1.0 for k in told)
    assert not any(k.startswith("seen:") and k.endswith(":told") for k in off), "honest: nothing told"


def _moves_peek(monkeypatch):
    """PLANT: a world that COPIES a seen mon's true unrevealed moves but reports an honest draw (the leak the honest
    builder exists to remove, disguised as a draw)."""
    from main.value_search import world_draws as WD

    orig = WD.draw_moves

    def peek(rng, prior, keep, n_slots, lever, truth, hp_type, hp_rule="any"):
        got = orig(rng, prior, keep, n_slots, lever, truth, hp_type, hp_rule)
        if got is None or lever is not None:
            return got
        slots, fact = got
        free = [i for i in range(n_slots) if i not in keep]
        out = list(slots)
        for i in free:
            if i < len(truth):
                out[i] = truth[i]
        return out, dict(fact, true_hits=len(free), match=True)

    monkeypatch.setattr(WD, "draw_moves", peek)


def test_teeth_a_world_that_copies_the_true_moves_is_caught(corpus, monkeypatch):
    honest = A.run(corpus, ("off",), 4, log=lambda m: None)
    h = honest["checks"]["leak:moves:seen:drawn[off]"]
    assert h["pass"] and h["z"] < 2.5, h
    _moves_peek(monkeypatch)
    planted = A.run(corpus, ("off",), 4, log=lambda m: None)
    v = planted["checks"]["leak:moves:seen:drawn[off]"]
    assert not v["pass"] and v["z"] > 6.0, v


def test_teeth_regate_legality_and_genders_catch_a_tampered_world(corpus):
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import worlds as W

    item = corpus[1]
    d, bat = item["decision"], item["battle"]
    log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
    root = W.prepare_root(log, at, d["side"], item["id"])
    w = W.WorldSampler(reveal="pilot").sample(log, at, d["side"], 2, item["id"])
    assert C.regate(root, w)["pass"] and C.genders(root, w)["pass"] and C.legality(root, w)["pass"]
    seen = [i for i, s in enumerate(root.sids) if s in root.kept and root.rv.mons[s].item is not None]
    assert seen, "the fixture decision has a seen mon whose item play revealed"
    ents = w.teams[0].split("]")
    f = W.parse_set(ents[seen[0]])
    f[2] = "choiceband" if W.toid(f[2]) != "choiceband" else "leftovers"   # PLANT: a revealed item swapped
    ents[seen[0]] = "|".join(f)
    bad = type(w)(logs=w.logs, teams=["]".join(ents)] + w.teams[1:], tries=w.tries, leak=w.leak, revealed=w.revealed)
    assert not C.regate(root, bad)["pass"]
    never = [i for i, s in enumerate(root.sids) if s not in root.kept]
    ents = w.teams[0].split("]")
    ratio = [i for i in never if W.species_draws_gender(W.toid(W.parse_set(ents[i])[1] or W.parse_set(ents[i])[0]))]
    assert ratio, "the fixture world has a gender-ratio never-seen mon"
    g = W.parse_set(ents[ratio[0]])
    g[7] = "M" if not g[7] else ""                  # PLANT: flip whether it draws a gender at turn 0
    ents[ratio[0]] = "|".join(g)
    bad2 = type(w)(logs=w.logs, teams=["]".join(ents)], tries=w.tries, leak=w.leak, revealed=w.revealed)
    assert not C.genders(root, bad2)["pass"]


def test_teeth_log_consistency_catches_an_over_reveal(corpus, monkeypatch):
    from agents.info_levers import revealed as RV
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import worlds as W

    item = corpus[1]
    d, bat = item["decision"], item["battle"]
    root = W.prepare_root(to_log(bat), cmd_index(bat, d["side"], d["n"]), d["side"], item["id"])
    assert C.log_consistency(root.true, root.viewer)["pass"]
    orig = RV._Reader._own_move

    def over(self, key, name):                      # PLANT: every used move also "reveals" one it never used
        orig(self, key, name)
        m = self._opp(key)
        if m is not None:
            m.moves.add("splash")

    monkeypatch.setattr(RV._Reader, "_own_move", over)
    v = C.log_consistency(root.true, root.viewer)
    assert not v["pass"] and v["moves_python_only"] > 0
