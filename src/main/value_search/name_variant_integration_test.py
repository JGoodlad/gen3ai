"""A packed NAME VARIANT in the TRUE team (GIGO 2026-10-15) on a REAL banked decision: one of the 719 pool teams packs
Will-O-Wisp as ``wisp``. Rewriting the banked opponent's ``willowisp`` to ``wisp`` replays the SAME battle (the engine
resolves the variant itself), so the builder must read the variant team exactly as it reads the canonical one: the
revealed Will-O-Wisp is recognised on the true slot (it was not: ``wisp`` != ``willowisp``), the choice attribution
finds it, and no world holds the move twice."""
import pytest

from agents.info_levers.packed import parse_set, typed_moves
from main.value_search import world_guard as WG
from main.value_search import worlds as W

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: the opponent's Gengar used Will-O-Wisp before this decision (turn 13 of the banked battle)
DECISION = "ai_v14_01_base/step_36000000/staller/loss_s1_003#p1#13"


@pytest.fixture(scope="module")
def decision():
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = B.load_bank(BANK_V1)
    d = next(x for x in bank.decisions if x["id"] == DECISION)
    bat = bank.battles[bank.battle_index[d["battle"]]]
    log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
    oi = 1 if d["side"] == "p1" else 0
    assert log["teams"][oi].count("willowisp") == 1, "the fixture needs the opponent's one Will-O-Wisp"
    variant = dict(log, teams=[t.replace("willowisp", "wisp") if i == oi else t for i, t in enumerate(log["teams"])])
    assert "wisp" in variant["teams"][oi].replace("willowisp", "")
    return log, variant, at, d["side"]


def test_the_variant_team_is_read_exactly_as_the_canonical_one(decision):
    log, variant, at, side = decision
    a, b = W.prepare_root(log, at, side, DECISION), W.prepare_root(variant, at, side, DECISION)
    assert "willowisp" in a.kept["gengar"], "the fixture's Gengar must have revealed Will-O-Wisp"
    assert b.kept == a.kept
    assert b.moves_of == a.moves_of            # FAILS at 8702a56f: ``wisp`` where the canonical team reads ``willowisp``
    assert b.leak == a.leak and b.leak["attribution_miss"] == 0
    assert b.tmpl == a.tmpl


def test_no_world_of_the_variant_team_holds_a_move_twice_and_the_revealed_one_sits_in_its_true_slot(decision):
    log, variant, at, side = decision
    root = W.prepare_root(variant, at, side, DECISION)
    got = W.WorldSampler(reveal="off").sample(variant, at, side, 3, DECISION)
    canon = W.WorldSampler(reveal="off").sample(log, at, side, 3, DECISION)
    g = root.sids.index("gengar")
    assert WG.problems(root, got.teams, got.facts) == {}
    for team, ref in zip(got.teams, canon.teams):
        mons = [typed_moves(parse_set(e)) for e in team.split("]")]
        assert all(len(set(m)) == len(m) for m in mons), mons
        assert mons[g].count("willowisp") == 1
        assert mons == [typed_moves(parse_set(e)) for e in ref.split("]")]    # the same world, read the same
    assert got.facts == canon.facts
