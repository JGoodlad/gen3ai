"""The ORACLE REVEAL through the real env core (``--oracle-reveal``; gen3_oracle_reveal_v1, v137).

The Rust side is gated in ``src/rust_env/tests/oracle_reveal_test.rs`` (the OFF digest pinned to the build
before the reveal, the `off` vs `species` differential over real battles). This is the PYTHON half over the
real self-check ``cdylib``:

* the spec's ``oracle_reveal`` key reaches the core, and the levels Python lists are the levels Rust parses
  (an unknown one is refused by Python AND by the core);
* at the FIRST decision the `off` row states one opponent species (the lead) and the `species` row states all
  six — in the observation's opponent block, which is what the shared trunk reads;
* the label columns the learner trains on survive it: under `species` every opponent species is stated, so
  ``belief_species`` / ``belief_moves`` are all PAD (no hidden target), while the `off` labels carry the
  hidden mons as targets;
* the information-lever spec (``designs/endstate/design_information_levers.md`` §6): every canonical text Python
  writes is a spec the core parses (and the core's refusals name the lever), a preset reaches the core as its label,
  and a mass-0.5 ``species:unseen`` lever fires on a PINNED count of a fixed set of episodes — with the belief labels
  aligned to exactly the stated slots.
"""
import numpy as np
import pytest

from agents.observation.constants import (
    ITEM_ID_DIM, MOVE_SLOT_DIM, OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_HP_REVEALED_OFFSET, POKEMON_ITEMS_OFFSET,
    POKEMON_MOVES_OFFSET, POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_SPREAD_DIM, POKEMON_SPREAD_OFFSET)
from utils.rust_env import ffi
from utils.rust_env import protocol as P

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N = 4


@pytest.fixture(scope="module")
def lib():
    from utils.rust_env.build import ensure_built

    ensure_built("selfcheck")
    return ffi.load(ffi.default_path("selfcheck"), nan_poison=True)


def _teams():
    from utils.ladder_corpus import teams

    return [t for t in teams("commit")[:8]]


def _first_decision(lib, level, n=N):
    spec = P.spec_json(n=n, threads=1, teams=_teams(), names=("oa", "ob"), turn_limit=250, refusal_budget=4,
                       bank_dir=None, labels=("belief",), oracle_reveal=level)
    core = ffi.FfiCore(spec, lib=lib)
    try:
        for i in range(n):
            core.cols["ep_team"][i, :] = [2 * i % 8, (2 * i + 1) % 8]
            core.cols["ep_seed"][i, :] = [i + 1, 2, 3, 4]
        core.dispatch("RESET")
        obs = np.array(core.cols["obs"], copy=True)
        labels = {k: np.array(core.cols[k], copy=True) for k in ("belief_species", "belief_moves", "known_moves")}
        need = np.array(core.cols["need"], copy=True)
        return obs, labels, need
    finally:
        core.close()


def _known(obs_row):
    return [float(obs_row[OFFSET_OPP_TEAM + j * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET]) for j in range(6)]


def test_the_levels_python_lists_are_the_levels_rust_parses(lib):
    assert P.ORACLE_REVEAL_LEVELS == ("off", "species", "full")
    with pytest.raises(ValueError, match="oracle_reveal"):
        P.spec_json(n=1, threads=1, teams=["x"], names=("a", "b"), turn_limit=None, refusal_budget=0, bank_dir=None,
                    oracle_reveal="everything")
    # the core refuses a level it does not know, by name (a hand-built spec past Python's own check)
    import json

    spec = json.loads(P.spec_json(n=1, threads=1, teams=_teams()[:2], names=("a", "b"), turn_limit=None,
                                  refusal_budget=0, bank_dir=None))
    spec["oracle_reveal"] = "everything"
    with pytest.raises(Exception, match="oracle_reveal"):
        ffi.FfiCore(json.dumps(spec), lib=lib)


def test_the_first_decision_states_the_whole_opponent_team_under_species_and_the_lead_alone_under_off(lib):
    obs_off, _, need = _first_decision(lib, "off")
    obs_sp, _, need_sp = _first_decision(lib, "species")
    assert (need == need_sp).all() and need.any()
    for i in range(N):
        for s in range(2):
            if not need[i, s]:
                continue
            assert _known(obs_off[i, s]) == [1, 0, 0, 0, 0, 0], "off: the seen lead alone"
            assert _known(obs_sp[i, s]) == [1, 1, 1, 1, 1, 1], "species: every opponent species is stated at turn 1"
            # the lead's slot, and everything outside the opponent block's tail, is `off`'s
            a, b = obs_off[i, s], obs_sp[i, s]
            tail = slice(OFFSET_OPP_TEAM + POKEMON_FULL_DIM, OFFSET_OPP_TEAM + 6 * POKEMON_FULL_DIM)
            assert (a[:tail.start] == b[:tail.start]).all() and (a[tail.stop:] == b[tail.stop:]).all()
            assert (a[tail] == 0).all() and (b[tail] != 0).any()


def _slot(row, j):
    return row[OFFSET_OPP_TEAM + j * POKEMON_FULL_DIM:OFFSET_OPP_TEAM + (j + 1) * POKEMON_FULL_DIM]


def test_the_first_decision_under_full_states_every_opponent_set_and_differs_from_species_only_in_the_facts(lib):
    obs_sp, _, need = _first_decision(lib, "species")
    obs_fl, labels, _ = _first_decision(lib, "full")
    for i in range(N):
        for s in range(2):
            if not need[i, s]:
                continue
            a, b = obs_sp[i, s], obs_fl[i, s]
            assert _known(b) == [1, 1, 1, 1, 1, 1]
            # outside the opponent block: identical to `species` (and so to `off`)
            end = OFFSET_OPP_TEAM + 6 * POKEMON_FULL_DIM
            assert (a[:OFFSET_OPP_TEAM] == b[:OFFSET_OPP_TEAM]).all() and (a[end:] == b[end:]).all()
            for j in range(6):
                sa, sb = _slot(a, j), _slot(b, j)
                # the set's facts: the spread block is KNOWN (flag cell, `spread_known` 1), the Hidden-Power block is
                # revealed, every item is known, and each of the (up to) four moves carries its `known` flag
                assert sb[POKEMON_SPREAD_OFFSET + 12] == 1.0, "spread_known"
                assert sb[POKEMON_HP_REVEALED_OFFSET] == 1.0, "hp_revealed"
                moves = [sb[POKEMON_MOVES_OFFSET + m * MOVE_SLOT_DIM:POKEMON_MOVES_OFFSET + (m + 1) * MOVE_SLOT_DIM] for m in range(4)]
                assert sum(1 for mv in moves if mv[0] != 0) >= 1 and all(mv[6] == 1.0 for mv in moves if mv[0] != 0)
                if j >= 1:
                    # an UNSEEN slot differs from the species-level row ONLY in the facts: item, ability, moves, spread, hp block
                    facts = set(range(POKEMON_ITEMS_OFFSET, POKEMON_ITEMS_OFFSET + 3)) | set(range(12, 16)) \
                        | set(range(POKEMON_MOVES_OFFSET, POKEMON_MOVES_OFFSET + 4 * MOVE_SLOT_DIM)) \
                        | set(range(POKEMON_SPREAD_OFFSET, 106))
                    diff = {c for c in range(POKEMON_FULL_DIM) if sa[c] != sb[c]}
                    assert diff and diff <= facts, f"slot {j} differs from species outside the set's facts: {sorted(diff - facts)}"
                    assert POKEMON_SPREAD_DIM == 18
    # the labels are the species level's: every opponent species stated, nothing believed
    assert (labels["belief_species"][:, 0] == -1).all() and (labels["belief_moves"][:, 0] == -1).all()
    assert ITEM_ID_DIM == 1


def test_the_real_oracle_labels_carry_no_hidden_target_and_off_still_does(lib):
    for level, hidden in (("species", False), ("off", True)):
        _, labels, need = _first_decision(lib, level)
        sp = labels["belief_species"][:, 0]                                # p1's labels, (N, 6)
        mv = labels["belief_moves"][:, 0]                                  # (N, 6, 4)
        if hidden:
            assert (sp >= 0).any(axis=1).all(), "off: every row's hidden mons are the targets"
        else:
            assert (sp == -1).all() and (mv == -1).all(), "species: nothing is hidden, nothing is a target"
            km = labels["known_moves"][:, 0]
            assert (km[:, :, 0] >= 1).all(), "every stated species (seen or not) has its true moves as the known-slot label"


# ------------------------------------------------------------------ the information-lever spec

#: Canonical texts `agents.info_levers.spec` writes; the Rust parser's own list (`encoder::oracle`'s
#: `CANONICAL_SAMPLES`) is the same, so a text one language writes is one the other parses, byte for byte.
CANONICAL_SAMPLES = (
    "moves:seen@0.95", "item:seen@0.95", "spread:seen@0.95", "moves:seen@0.5", "moves:seen@0.25", "moves:seen@0.0001",
    "species:unseen@0.5", "species:unseen,moves:unseen", "species:unseen,moves:both@0.9",
    "species:unseen,moves:seen,moves:unseen@0.5", "moves:seen,item:seen,ability:seen,spread:seen",
    "species:unseen@0.25,item:seen@0.75,item:unseen@0.125,spread:both",
)


def test_every_canonical_text_python_writes_is_one_the_core_parses_and_the_core_names_what_it_refuses(lib):
    import json
    import re

    from agents.info_levers.spec import canonical, parse

    from utils.paths import repo_path

    src = open(repo_path("src", "rust_sim", "src", "encoder", "oracle.rs")).read()
    for t in CANONICAL_SAMPLES:
        assert canonical(t) == t and parse(t).label == t, t
        assert f'"{t}"' in src, f"{t} is not in the Rust CANONICAL_SAMPLES list"
        spec = P.spec_json(n=1, threads=1, teams=_teams()[:2], names=("a", "b"), turn_limit=None, refusal_budget=0,
                           bank_dir=None, oracle_reveal=t)
        assert json.loads(spec)["oracle_reveal"] == t
        ffi.FfiCore(spec, lib=lib).close()
    # the core refuses, by name, what the observation cannot honour — even past Python's own check
    base = json.loads(P.spec_json(n=1, threads=1, teams=_teams()[:2], names=("a", "b"), turn_limit=None,
                                  refusal_budget=0, bank_dir=None))
    for bad, why in (("hp:seen", "hp:seen"), ("moves:seen@0.95~placebo", "~placebo"), ("moves:unseen", "species:unseen"),
                     ("moves:seen@0.950", "not canonical"), ("ki_moves", "names no class")):
        base["oracle_reveal"] = bad
        with pytest.raises(Exception, match=re.escape(why)):
            ffi.FfiCore(json.dumps(base), lib=lib)


def test_a_preset_reaches_the_core_as_its_label_and_the_levels_keep_their_json_text():
    import json

    def text(r):
        return json.loads(P.spec_json(n=1, threads=1, teams=["x"], names=("a", "b"), turn_limit=None,
                                      refusal_budget=0, bank_dir=None, oracle_reveal=r))["oracle_reveal"]

    assert text("off") == "off" and text("species") == "species" and text("full") == "full"
    assert text("species:unseen") == "species"
    assert text("species:unseen,moves:both,item:both,ability:both,spread:both") == "full"
    assert text("ki_moves") == "moves:seen@0.95" and text("F+M") == "moves:seen"
    assert text(("ki_moves", "off")) == ["moves:seen@0.95", "off"]
    assert text(("full", "species:unseen,moves:both,item:both,ability:both,spread:both")) == "full"
    for bad, why in (("ki_wrongmoves", "placebo"), ("hp:seen", "hp:seen"), ("oracle", "hp:seen"),
                     ("moves:unseen", "species:unseen"), ("everything", "everything")):
        with pytest.raises(ValueError, match=why):
            text(bad)


#: The told tail mons at the first decision of the fixed corpus below (16 envs x 2 sides x 5 unseen mons = 160 draws
#: of a mass-0.5 `species:unseen` lever), recorded when the per-episode draw was built (`encoder::oracle::uniform_bp`).
#: A change to the draw moves which episodes a recorded run was told: never silently.
PINNED_HALF_SPECIES_TOLD = 77


def test_a_half_mass_species_lever_fires_on_a_pinned_count_and_the_labels_follow_the_stated_slots(lib):
    n = 16
    obs, labels, need = _first_decision(lib, "species:unseen@0.5", n=n)
    obs_off, labels_off, need_off = _first_decision(lib, "off", n=n)
    assert (need == need_off).all() and need.all(), "every env's both sides decide at turn 1"
    told = 0
    for i in range(n):
        for s in range(2):
            known = _known(obs[i, s])
            k = int(sum(known))
            assert known == [1.0] * k + [0.0] * (6 - k), "the stated slots lead"
            told += k - 1
            # the labels follow the stated slots: the unstated mons (6 - k) are exactly the belief targets
            sp = labels["belief_species"][i, s]
            assert int((sp >= 0).sum()) == 6 - k, f"env {i} p{s + 1}: {k} stated, belief targets {sp}"
            assert (labels["known_moves"][i, s][:k, 0] >= 1).all() and (labels["known_moves"][i, s][k:, 0] == -1).all()
            # the lead's slot and everything outside the tail is `off`'s
            a, b = obs_off[i, s], obs[i, s]
            lo = OFFSET_OPP_TEAM + POKEMON_FULL_DIM
            assert (a[:lo] == b[:lo]).all() and (a[OFFSET_OPP_TEAM + 6 * POKEMON_FULL_DIM:] == b[OFFSET_OPP_TEAM + 6 * POKEMON_FULL_DIM:]).all()
    print(f"species:unseen@0.5 told {told} / {n * 2 * 5}")
    assert 0 < told < n * 2 * 5, "non-vacuity: some told, some not"
    assert told == PINNED_HALF_SPECIES_TOLD, "the mass-0.5 firing count over the fixed corpus moved"

