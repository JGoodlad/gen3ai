"""The information-lever audit suite on a FIXED small corpus of committed-bank decisions (early turns: few seen mons,
so the honest screen is cheap): every check passes on the real builder, and each planted defect is caught.

The larger corpus is the slow tier (``info_audit_slow_test.py``)."""
import pytest

from main.info_audit import __main__ as A
from main.info_audit import checks as C

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]  # ship gate (2026-10-19): 53 s of routine-gate time beside a live run (designs/ops/testing.md 'The SHIP GATE')

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


def test_teeth_consistency_catches_a_fact_record_at_another_mons_slot(corpus):
    """The 2026-10-15 GIGO, as a suite verdict: the honest builder's worlds are consistent, and the same worlds with
    two never-seen records swapped (the pre-fix pairing) fail ``consistency`` naming the slot."""
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import worlds as W

    item = corpus[0]
    d, bat = item["decision"], item["battle"]
    log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
    root = W.prepare_root(log, at, d["side"], item["id"])
    w = W.WorldSampler(reveal="off").sample(log, at, d["side"], 2, item["id"])
    assert C.consistency(root, w)["pass"]
    never = [i for i, s in enumerate(root.sids) if s not in root.kept]
    assert len(never) >= 2, "the fixture decision has two never-seen slots"
    fl = [dict(f) for f in w.facts[0]]
    a, b = never[0], never[1]
    fl[a], fl[b] = dict(fl[b], slot=a), dict(fl[a], slot=b)
    bad = type(w)(logs=w.logs, teams=w.teams, tries=w.tries, leak=w.leak, revealed=w.revealed,
                  facts=[fl] + w.facts[1:])
    v = C.consistency(root, bad)
    assert not v["pass"] and v["inconsistent"] == 1 and "fact record" in " ".join(v["examples"][0]["problems"])


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
    ents = w.teams[0].split("]")
    k = never[0]
    h = W.parse_set(ents[k])
    # PLANT: an egg-move pair no father passes on (F2), on a species the team does not hold (no species-clause error)
    sp, ab, mv = next(c for c in (("Tyranitar", "sandstream", "dragondance,pursuit,rockslide,earthquake"),
                                  ("Skarmory", "keeneye", "whirlwind,drillpeck,spikes,rest"))
                      if W.toid(c[0]) not in {W.species_id(W.parse_set(e)) for e in ents})
    h[0], h[1], h[2], h[3], h[4] = sp, "", "leftovers", ab, mv
    ents[k] = "|".join(h)
    bad3 = type(w)(logs=w.logs, teams=["]".join(ents)], tries=w.tries, leak=w.leak, revealed=w.revealed)
    v = C.legality(root, bad3)
    assert not v["pass"] and v["showdown_invalid"] == 1 and "egg move" in " ".join(v["examples"][0]["showdown"])


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


#: a decision whose never-seen gender draws CANNOT be matched without the engine's pad (the pilot's builder rejected
#: worlds on their gender count there: the smoke of 2026-10-12)
GENDER_DECISION = "ai_v14_01_base/step_46000032/aggressive_v2/win_s0_001#p1#18"


def _decision(did):
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = B.load_bank(BANK_V1)
    d = next(x for x in bank.decisions if x["id"] == did)
    bat = bank.battles[bank.battle_index[d["battle"]]]
    return to_log(bat), cmd_index(bat, d["side"], d["n"]), d["side"]


def test_teeth_seed_reuse_through_the_gender_dice_is_closed_by_the_pad_and_caught_without_it():
    from main.value_search import worlds as W

    log, at, side = _decision(GENDER_DECISION)
    s = W.WorldSampler(reveal="pilot")
    assert s.pad is not None, "the engine's gender draw pad is in this build"
    w = s.sample(log, at, side, 4, GENDER_DECISION)
    assert C.gender_independence(w, s.pad)["pass"]
    # PLANT: the builder without the pad must reject worlds on their gender count — conditioning the never-seen species
    s.pad = None
    w2 = s.sample(log, at, side, 4, GENDER_DECISION)
    v = C.gender_independence(w2, s.pad)
    assert not v["pass"] and v["gender_count_rejections"] > 0, v


def test_teeth_a_choice_log_core_events_accepts_but_the_playout_tool_refuses_is_rejected(corpus):
    """The blind replica's finding 3: the playout tool resolves a replayed choice BY NAME, core_events also takes a
    numeric ``move k``. A world rendered so is gated green by core_events and must be refused by the builder."""
    from agents.battle.core_replay import run_core
    from main.policy_spectrum.truth import cmd_index, to_log
    from main.value_search import worlds as W

    item = corpus[1]
    d, bat = item["decision"], item["battle"]
    log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
    root = W.prepare_root(log, at, d["side"], item["id"])
    s = W.WorldSampler(reveal="oracle")
    cmds = W.render(root.tmpl, root.team)
    opp = "p2" if d["side"] == "p1" else "p1"
    opp_idx = [i for i, (c, _t) in enumerate(cmds) if c == opp]
    actives = W.opp_actives(root.true, root.oi, len(opp_idx))
    k, slot = next((i, root.moves_of[root.sids.index(a)].index(W.toid(cmds[i][1][5:])) + 1)
                   for i, a in zip(opp_idx, actives)
                   if cmds[i][1].startswith("move ") and a in root.sids
                   and W.toid(cmds[i][1][5:]) in root.moves_of[root.sids.index(a)])
    planted = [list(c) for c in cmds]
    planted[k][1] = f"move {slot}"                               # PLANT: the SAME move as a numeric choice token
    res = run_core([root.rec(root.team, "planted", planted)], trackers=True, obs=True)[0]
    assert res["ok"] and W.protocol_lines(res, root.viewer, root.n) == root.ref_proto, \
        "precondition: core_events gates the numeric choice green"
    lg = dict(log, cmds=[f"FORCELOSE {t}" if c == "forcelose" else f"CHOOSE {c} {t}" for c, t in planted]
              + list(log["cmds"][at:]))
    assert s.playout_refusal(lg, at, d["side"]) is not None, "the playout tool refuses it: the builder rejects it"
    good = dict(log, cmds=[f"FORCELOSE {t}" if c == "forcelose" else f"CHOOSE {c} {t}" for c, t in cmds]
                + list(log["cmds"][at:]))
    assert s.playout_refusal(good, at, d["side"]) is None


#: a decision whose gate rejects both a dice item and a ±1 Speed point of a seen mon (the 2026-10-12 measurement,
#: ``designs/research_state/measurements/info_levers_2026-10-12/seed_reuse_200.json``)
SEED_REUSE_DECISION = "ai_v14_01_base/step_36000000/sentinel_3/loss_s0_004#p1#5"


def test_the_seed_reuse_residual_is_measured_by_counterfactual_flips_against_a_passing_control():
    from main.value_search import worlds as W

    log, at, side = _decision(SEED_REUSE_DECISION)
    root = W.prepare_root(log, at, side, SEED_REUSE_DECISION)
    v = C.seed_reuse(root)
    assert v["pass"], "the unflipped true team passes the gate (the instrument's control)"
    assert v["item_dice_rejected"] >= 1 and v["speed_pinned"] >= 1, v
    assert v["item_honest"] == 0 and v["item_truth_dice"] == 0, "the neutral item swap is accepted there"
    # TEETH of the control: a gate that rejects everything is caught
    bad = W.Root(**{**root.__dict__, "ref_proto": root.ref_proto + ["|planted|"]})
    assert not C.seed_reuse(bad)["pass"]
