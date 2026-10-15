"""THE 2026-10-15 GIGO on a REAL banked decision (the minimal case): the honest builder's never-seen fact records name
the mon the engine built at their slot after the gender placement permuted the drawn never-seen mons, and the throwing
guard raises on a world whose records are paired with the wrong slot."""
import pytest

from main.value_search import world_guard as WG
from main.value_search import worlds as W

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: the decision the GIGO was reproduced on: 4 never-seen slots, every one of its 3 honest worlds mis-paired at HEAD
#: 6a7ee72c (e.g. world 0: team metagross / salamence / starmie / tyranitar at slots 2-5, records tyranitar / starmie /
#: metagross / salamence)
DECISION = "ai_v14_01_base/step_36000000/sentinel_3/loss_s0_004#p1#5"


@pytest.fixture(scope="module")
def built():
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = B.load_bank(BANK_V1)
    d = next(x for x in bank.decisions if x["id"] == DECISION)
    bat = bank.battles[bank.battle_index[d["battle"]]]
    log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
    root = W.prepare_root(log, at, d["side"], DECISION)
    return root, W.WorldSampler(reveal="off").sample(log, at, d["side"], 3, DECISION)


def test_every_fact_record_names_the_mon_at_its_slot(built):
    root, w = built
    never = [i for i, s in enumerate(root.sids) if s not in root.kept]
    assert len(never) >= 2, "the fixture needs never-seen slots to permute"
    permuted = 0
    for team, fl in zip(w.teams, w.facts):
        sp = [WG.species_of(e) for e in team.split("]")]
        assert [f["slot"] for f in fl] == list(range(len(sp)))
        assert [f["species"] for f in fl] == sp                 # FAILS at 6a7ee72c: the records stayed where drawn
        true_never = {root.sids[i] for i in never}
        for i in never:
            sf = fl[i]["species_fact"]
            assert sf["match"] == (sp[i] in true_never) and sf["slot_match"] == (sp[i] == root.sids[i])
            permuted += fl[i]["drawn_slot"] != i
    assert permuted > 0, "the fixture must exercise a real permutation (else it cannot fail on revert)"
    assert WG.problems(root, w.teams, w.facts) == {}


def test_the_guard_raises_on_a_record_paired_with_the_wrong_slot(built):
    root, w = built
    never = [i for i, s in enumerate(root.sids) if s not in root.kept]
    fl = [dict(f) for f in w.facts[0]]
    a, b = never[0], never[1]
    fl[a], fl[b] = dict(fl[b], slot=a), dict(fl[a], slot=b)     # the old pairing: a record at another mon's slot
    with pytest.raises(WG.WorldConsistencyError, match="fact record"):
        WG.check_worlds(root, w.teams[:1], [fl], "planted")
    assert not issubclass(WG.WorldConsistencyError, W.WorldError), "a defect is never an exclusion"


def test_check_teams_guards_a_builder_with_no_records_and_each_planted_defect_raises(built):
    """POOL builds worlds that ARE teams (no provenance records): ``check_teams`` derives the records from the entries and
    still catches what the engine can contradict — validated on 120 archived POOL states (960 worlds: 0 false positives)
    and on the 7 defective POOLSETS worlds (7 / 7 flagged), 2026-10-15."""
    root, w = built
    WG.check_teams(root, w.teams, "real")                       # a real honest world is consistent
    seen = [i for i, s in enumerate(root.sids) if s in root.kept]
    assert len(seen) >= 2
    ents = w.teams[0].split("]")
    swapped = list(ents)
    swapped[seen[0]], swapped[seen[1]] = ents[seen[1]], ents[seen[0]]
    with pytest.raises(WG.WorldConsistencyError, match="moved from its true slot"):
        WG.check_teams(root, ["]".join(swapped)], "planted")
    f = W.parse_set(ents[seen[0]])
    mv = f[4].split(",")
    f[4] = ",".join([mv[0], mv[0]] + mv[2:])                    # one move twice
    twice = list(ents)
    twice[seen[0]] = "|".join(f)
    with pytest.raises(WG.WorldConsistencyError, match="a move twice"):
        WG.check_teams(root, ["]".join(twice)], "planted")
