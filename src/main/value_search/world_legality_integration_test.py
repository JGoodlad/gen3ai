"""F2 against the REAL validator and the REAL builder: Showdown's ``TeamValidator('gen3ou')`` (the pinned submodule,
through ``utils/bridge/validate_stream.js``) rejects the combinations the stage-0 screen measured in the honest worlds
and accepts their legal neighbours; a constructed never-seen Tyranitar's naive draw makes the impossible egg-move pair
(Dragon Dance + Pursuit) and the legal draw never does; and on a real banked decision the builder applies the rule to
the seen mons' unrevealed moves and the never-seen sets, and RAISES when the rule rejects the truth."""
from __future__ import annotations

import numpy as np
import pytest

from agents.info_levers.consumers import SEARCH
from main.value_search import world_draws as WD
from main.value_search import world_legality as WL
from main.value_search import worlds as W

pytestmark = pytest.mark.integration

#: (illegal set, the validator's reason, a legal neighbour) — the stage-0 screen's measured causes (F2)
CASES = [
    ("Tyranitar||leftovers|sandstream|dragondance,pursuit,rockslide,earthquake|Adamant|252,252,,,4,|||||",
     "egg move combination", "Tyranitar||leftovers|sandstream|dragondance,crunch,rockslide,earthquake|Adamant|"
     "252,252,,,4,|||||"),
    ("Skarmory||leftovers|keeneye|whirlwind,drillpeck,spikes,rest|Impish|252,,252,,4,|||||",
     "egg move combination", "Skarmory||leftovers|keeneye|whirlwind,toxic,spikes,rest|Impish|252,,252,,4,|||||"),
    ("Blissey||leftovers|naturalcure|wish,aromatherapy,softboiled,seismictoss|Bold|252,,252,,4,|||||",
     "incompatible", "Blissey||leftovers|naturalcure|wish,toxic,softboiled,seismictoss|Bold|252,,252,,4,|||||"),
    ("Salamence||choiceband|intimidate|dragonclaw,earthquake,rockslide,hiddenpowergrass|Adamant|,248,,8,,252||"
     "30,31,31,30,31,31|||,Flying,", "Hidden Power",
     "Salamence||choiceband|intimidate|dragonclaw,earthquake,rockslide,hiddenpowergrass|Adamant|,248,,8,,252||"
     "30,31,31,30,31,31|||"),
]


def test_showdown_rejects_the_measured_combinations_and_accepts_their_legal_neighbours():
    v = WL.Validator()
    try:
        got = v.problems([c[0] for c in CASES] + [c[2] for c in CASES])
        for (bad, why, _), p in zip(CASES, got[: len(CASES)]):
            assert p and any(why in e for e in p), (bad, p)
        assert got[len(CASES):] == [()] * len(CASES)
        n = v.requests
        assert v.problems([CASES[0][0]]) == [got[0]] and v.requests == n        # memoised: no second round trip
    finally:
        v.close()


def test_the_naive_never_seen_draw_makes_the_impossible_egg_pair_and_the_legal_draw_never_does():
    """The constructed state: a never-seen Tyranitar, drawn from the REAL Smogon prior (Dragon Dance and Pursuit are
    both common; no single father passes both on in gen 3)."""
    P, off, legal = W.Prior(), SEARCH.validate("off"), WL.Legality()
    naive, fixed, tally = [], [], {}
    for seed in range(120):
        naive.append(WD.new_set(np.random.default_rng(seed), P, "tyranitar", None, off)[0])
        got = WD.new_set(np.random.default_rng(seed), P, "tyranitar", None, off, legal=legal.set_ok, tally=tally)
        assert got is not None
        fixed.append(got[0])
    bad_naive = [e for e, p in zip(naive, legal.set_problems(naive)) if p]
    assert any({"dragondance", "pursuit"} <= set(W.parse_set(e)[4].split(",")) for e in bad_naive), \
        "the precondition: the naive draw makes the egg pair"
    assert legal.set_problems(fixed) == [()] * len(fixed)          # FAILS if the draw ignores the rule
    assert tally["illegal_set"] >= len(bad_naive)
    # a draw legal the first time is the naive draw itself (the rule consumes no randomness)
    assert all(a == b for a, b, p in zip(naive, fixed, legal.set_problems(naive)) if not p)


# ---- the builder on a real banked decision (the core) -------------------------------------------------------------
DECISION = "ai_v14_01_base/step_36000000/sentinel_3/loss_s0_004#p1#5"     # world_guard_integration_test's decision


class _Ban:
    """A fake rule: a set of species ``s`` holding move ``m`` is illegal, for each banned (s, m)."""

    def __init__(self, pairs):
        self.pairs = set(pairs)

    def problems(self, teams):
        out = []
        for t in teams:
            bad = []
            for e in t.split("]"):
                f = W.parse_set(e)
                sid = W.species_id(f)
                bad += [f"{sid} {m}" for m in W.typed_moves(f) if (sid, m) in self.pairs]
            out.append(tuple(bad))
        return out


@pytest.fixture(scope="module")
def decision():
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1
    from main.policy_spectrum.truth import cmd_index, to_log

    bank = B.load_bank(BANK_V1)
    d = next(x for x in bank.decisions if x["id"] == DECISION)
    bat = bank.battles[bank.battle_index[d["battle"]]]
    log, at = to_log(bat), cmd_index(bat, d["side"], d["n"])
    return log, at, d["side"], W.prepare_root(log, at, d["side"], DECISION)


@pytest.mark.sim
@pytest.mark.slow  # ship gate (2026-10-20): 130 s beside a live run (x9.35); its module fixture `decision` serves only it
def test_the_builder_applies_the_rule_to_seen_and_never_seen_draws_and_raises_on_one_rejecting_the_truth(decision):
    """World 0 built with no rule, then with ONE (species, move) of it banned — a seen mon's unrevealed move, then a
    never-seen mon's move. Until the banned draw, the ruled build draws exactly what the free one did (the rule spends
    no randomness; one shared screen), so it MUST meet the banned draw, reject it and return a world without it."""
    log, at, side, root = decision
    cache: dict = {}

    def build(pairs):
        lg = WL.Legality(validator=_Ban(pairs), pad=W._gender_pad())
        return W.WorldSampler(reveal="off", legality=lg).sample(log, at, side, 1, DECISION, screen_cache=cache)

    free = build(())
    true_moves = {s: set(m) for s, m in zip(root.sids, root.moves_of)}
    seen_pairs, never_pairs = [], []
    for e in free.teams[0].split("]"):
        f = W.parse_set(e)
        sid = W.species_id(f)
        for m in W.typed_moves(f):
            if m not in true_moves.get(sid, set()) and not W.is_hp(m):
                (seen_pairs if sid in root.kept else never_pairs).append((sid, m))
    assert seen_pairs and never_pairs, "the precondition: the free draw puts non-true moves on seen and never-seen mons"
    for banned in (sorted(seen_pairs)[0], sorted(never_pairs)[0]):
        ruled = build({banned})
        held = {(W.species_id(W.parse_set(e)), m) for e in ruled.teams[0].split("]")
                for m in W.typed_moves(W.parse_set(e))}
        assert banned not in held                                    # FAILS if that draw ignores the rule
        assert ruled.fails.get("illegal_set", 0) > 0, (banned, ruled.fails)   # met at the draw, not by a fallback
    # THE TRUE-TEAM GUARD: a rule that rejects a TRUE set raises before any world is drawn (never an exclusion)
    s0, m0 = root.sids[0], root.moves_of[0][0]
    with pytest.raises(WL.LegalityRuleError, match="TRUE team fails"):
        build({(s0, m0)})
