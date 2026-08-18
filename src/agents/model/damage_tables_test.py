"""Unit tests for `damage_tables.build_move_prior_logits` — the UNCONDITIONAL learnset legality gate
(gen3_unconditional_move_legality_v1).

Legality is a CORRECTNESS property, not a feature: a move a species physically cannot learn must
always carry ~zero belief mass. There is no `learnset_gate` parameter and no way to switch it off;
`floor` is only the LEGAL-BUT-UNOBSERVED base."""
import math

import pytest
import torch

from agents import gen3_data
from agents.model.damage_tables import (build_move_prior_logits, build_self_boost_tables,
                                        HIDDEN_POWER_NUM, _PRIOR_FLOOR,
                                        _ILLEGAL_PROB, _MIN_PRIOR_FLOOR)

_N_SPECIES = 600
_N_MOVES = 600
_EPS = _ILLEGAL_PROB
_LOGIT_EPS = math.log(_EPS / (1.0 - _EPS))                     # -13.8155
_LOGIT_FLOOR = math.log(_PRIOR_FLOOR / (1.0 - _PRIOR_FLOOR))   #  -3.8918


def _num(species_id, move_id):
    return gen3_data.species.get(species_id).num, gen3_data.moves.get(move_id).num


# The buffer is expensive to build (a full species × move sweep) and every test below reads the same
# one — build it once.
@pytest.fixture(scope="module")
def prior():
    return build_move_prior_logits(_N_SPECIES, _N_MOVES)


# --- GATE 1: an unlearnable move carries ~zero mass, ALWAYS -------------------------------- #
#
# The pairs are asserted ILLEGAL against `gen3_data.learnset.get_legal_moves` in the test itself,
# so the test cannot quietly become vacuous if the learnset data changes under it.

_ILLEGAL_PAIRS = [
    ("blissey", "explosion"),      # Blissey has no Explosion in gen 3
    ("blissey", "swordsdance"),    # …and no Swords Dance
    ("magikarp", "hydropump"),     # Magikarp's gen-3 pool is Splash / Tackle / Flail
    ("starmie", "explosion"),
]


@pytest.mark.parametrize("species_id,move_id", _ILLEGAL_PAIRS)
def test_unlearnable_move_has_near_zero_prior_mass(prior, species_id, move_id):
    legal = gen3_data.learnset.get_legal_moves(species_id)
    assert legal is not None, f"{species_id} must have a learnset for this test to mean anything"
    assert move_id not in legal, f"FIXTURE STALE: {species_id} CAN learn {move_id}; pick another pair"

    snum, mnum = _num(species_id, move_id)
    assert prior[snum, mnum].item() == pytest.approx(_LOGIT_EPS, abs=1e-3)
    # …and in probability space that is ~0, which is the property that actually matters.
    assert torch.sigmoid(prior[snum, mnum]).item() < 1e-5


def test_illegal_is_materially_below_legal_but_unobserved(prior):
    """The strict separation the gate exists to create. A floor that collapsed onto the illegal value
    (the old `--move-candidate-floor 0.0`) would make this a tie — and the gate a silent no-op."""
    snum, explosion = _num("blissey", "explosion")           # ILLEGAL (asserted above)
    legal = gen3_data.learnset.get_legal_moves("blissey")

    # A LEGAL Blissey move with no recorded Smogon usage → the floor base.
    unobserved = None
    usage_nums = {gen3_data.moves.get(m).num for m in gen3_data.priors.moves("blissey")
                  if gen3_data.moves.get(m) is not None}
    for mid in legal:
        md = gen3_data.moves.get(mid)
        if md is not None and md.num not in usage_nums and 0 <= md.num < _N_MOVES:
            unobserved = md.num
            break
    assert unobserved is not None, "expected at least one legal-but-unused Blissey move"

    illegal_v = prior[snum, explosion].item()
    unobs_v = prior[snum, unobserved].item()
    assert illegal_v < unobs_v                                    # strictly below
    assert unobs_v - illegal_v > 5.0                              # …and MATERIALLY so (~9.92 nats)
    assert unobs_v == pytest.approx(_LOGIT_FLOOR, abs=1e-3)


# --- GATE 2: no cell is -inf or NaN, for ANY species ---------------------------------------- #

def test_no_entry_is_inf_or_nan(prior):
    """The -inf trap: `torch.logit(0.0)` is -inf and would poison every belief logit downstream."""
    assert torch.isfinite(prior).all(), "non-finite entry in the move prior"
    assert not torch.isnan(prior).any()
    assert prior.min().item() == pytest.approx(_LOGIT_EPS, abs=1e-4)   # the floor of the buffer IS eps
    assert prior.min().item() > -20.0                                  # finite with room to spare


def test_floor_of_zero_is_rejected_not_silently_clamped():
    """A 0.0 floor used to mean 'legality OFF'. It is now a hard error: it would make
    legal-unobserved indistinguishable from impossible, and logit(0) = -inf."""
    with pytest.raises(ValueError, match="out of range"):
        build_move_prior_logits(_N_SPECIES, _N_MOVES, floor=0.0)
    with pytest.raises(ValueError, match="out of range"):
        build_move_prior_logits(_N_SPECIES, _N_MOVES, floor=_ILLEGAL_PROB)   # == impossible → collapse
    with pytest.raises(ValueError, match="out of range"):
        build_move_prior_logits(_N_SPECIES, _N_MOVES, floor=1.0)
    assert _MIN_PRIOR_FLOOR > _ILLEGAL_PROB                                  # the bound is meaningful


# --- GATE 3: a species with NO learnset entry still gets the flat floor ---------------------- #

def test_species_with_no_learnset_gets_flat_floor(prior):
    """Unchanged fallback: no movepool known ⇒ nothing to prune ⇒ the flat liftable floor, NOT the
    'everything impossible' default. Covers national-dex num 0 — the UNKNOWN-SPECIES sentinel an
    unrevealed opponent slot carries, which `MoveBelief.move_logits` indexes directly."""
    row0 = prior[0]
    assert torch.isfinite(row0).all()
    assert row0.min().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)
    assert row0.max().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)   # FLAT

    # Any species the data covers but whose learnset is missing gets the same flat row.
    for sid in gen3_data.species.base_form_ids():
        if gen3_data.learnset.get_legal_moves(sid) is None:
            snum = gen3_data.species.get(sid).num
            if 0 <= snum < _N_SPECIES and not gen3_data.priors.moves(sid):
                assert prior[snum].min().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)
                assert prior[snum].max().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)
                break


def test_not_known_illegal_is_never_treated_as_known_illegal(prior):
    """The mirror of `test_illegal_is_materially_below_legal_but_unobserved`, and the regression this
    build's `covered` mask exists for.

    Only a species with a KNOWN movepool may mark a move impossible. Absence of knowledge is not
    knowledge of absence: an unknown species must keep every move POSSIBLE, because the alternative is
    a belief asserting near-certainty that an unidentified opponent cannot use any move at all —
    strictly worse than the phantom-threat bug the legality gate was built to fix.

    The trap is that the builder seeds the whole tensor with the IMPOSSIBLE value and only overwrites
    rows it visits while iterating `base_form_ids()`; every unvisited row (num 0, dex gaps) would
    otherwise keep that default."""
    rows_with_a_known_movepool = {
        gen3_data.species.get(sid).num for sid in gen3_data.species.base_form_ids()
        if gen3_data.learnset.get_legal_moves(sid) is not None
    }
    for snum in range(_N_SPECIES):
        if snum in rows_with_a_known_movepool:
            continue
        row = prior[snum]
        assert row.min().item() > _LOGIT_EPS + 5.0, (
            f"species num {snum} has no known movepool, yet carries 'impossible' cells — "
            "absence of a learnset was read as absence of moves"
        )
        assert row.min().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)


def test_a_dex_gap_row_is_the_flat_floor_not_impossible(prior):
    """A num beyond the real dex (no species at all) is 'nothing known', not 'nothing possible'."""
    used = {gen3_data.species.get(s).num for s in gen3_data.species.base_form_ids()}
    gap = next(n for n in range(1, _N_SPECIES) if n not in used)
    assert prior[gap].min().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)
    assert prior[gap].max().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)


# --- GATE 4: a LEGAL move WITH recorded usage keeps its TRUE usage --------------------------- #

def test_legal_move_with_usage_keeps_its_true_smogon_usage(prior):
    """Neither floored up nor pruned: the recorded rate survives verbatim."""
    for species_id, move_id in (("skarmory", "spikes"), ("tyranitar", "rockslide"),
                                ("celebi", "psychic")):
        legal = gen3_data.learnset.get_legal_moves(species_id)
        assert legal is not None and move_id in legal, f"FIXTURE STALE: {species_id}/{move_id}"
        snum, mnum = _num(species_id, move_id)
        expected = sum(p for mid, p in gen3_data.priors.moves(species_id).items()
                       if gen3_data.moves.get(mid) is not None
                       and gen3_data.moves.get(mid).num == mnum)
        assert expected > _PRIOR_FLOOR, f"{species_id}/{move_id} should be a staple in the priors"
        want = math.log(expected / (1.0 - expected))
        assert prior[snum, mnum].item() == pytest.approx(want, abs=1e-3)   # TRUE usage, verbatim
        assert prior[snum, mnum].item() > _LOGIT_FLOOR                     # …above the floor


def test_legal_rare_move_stays_liftable_never_pruned(prior):
    """A move a species CAN learn but runs below the floor is NOT pruned — it keeps a liftable prior
    (>= floor, far above the impossible eps), so the head can still anticipate a surprise tech."""
    victim = None
    for sid in gen3_data.species.base_form_ids():
        legal = gen3_data.learnset.get_legal_moves(sid)
        if legal is None:
            continue
        for mid, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(mid)
            if md is None or md.num == HIDDEN_POWER_NUM:
                continue
            legal_id = "hiddenpower" if mid.startswith("hiddenpower") else mid
            if legal_id in legal and 0.0 < p < _PRIOR_FLOOR:
                victim = (gen3_data.species.get(sid).num, md.num)
                break
        if victim:
            break
    assert victim is not None, "expected at least one legal sub-2% move in the priors"
    snum, mnum = victim
    # kept at the liftable floor (its sub-floor usage rounds up to the floor base) — NOT pruned to eps.
    assert prior[snum, mnum].item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)
    assert prior[snum, mnum].item() > _LOGIT_EPS + 5.0          # decisively NOT impossible


def test_hidden_power_present_for_hp_user(prior):
    # HP presence = the SUMMED legal HP-type usages — an HP user keeps a real HP prior.
    snum = gen3_data.species.get("zapdos").num
    assert prior[snum, HIDDEN_POWER_NUM].item() > _LOGIT_EPS + 1.0   # not pruned


def test_the_config_default_matches_the_builder_default():
    """`model_version` is deliberately stdlib-only, so it duplicates the 0.02 literal. Pin them."""
    import dataclasses
    from agents.model.model_version import ModelVersion
    field = next(f for f in dataclasses.fields(ModelVersion) if f.name == "move_candidate_floor")
    assert field.default == _PRIOR_FLOOR


# --- gen3_species_formes_v1: the num-indexed tables must stay BASE-FORM ---------------------- #
#
# Every buffer here is `table[species.num] = …`, and an alternate FORME shares its base's
# national-dex num (Deoxys-Speed / Unown-B / Castform-Sunny all landed in the species data
# once the port needed to construct gen3 randbats teams). Iterating `species.raw()` would be
# last-write-wins, so a forme would silently redefine the base's stats/types at that num —
# a plausible-but-false number fed to the model, invisible to any shape check. Every builder
# therefore iterates `species.base_form_ids()`; these pin that.

def test_base_stats_table_holds_the_BASE_forme():
    from agents.model.damage_tables import build_species_base_stats, SPREAD_STAT_COLS
    base = build_species_base_stats(_N_SPECIES)
    deoxys = gen3_data.species.get("deoxys")
    row = [float(deoxys.base_stats[s]) for s in SPREAD_STAT_COLS]
    assert base[deoxys.num].tolist() == row            # NOT Deoxys-Speed's 95/90/95/90/180
    assert base[deoxys.num][SPREAD_STAT_COLS.index("spe")].item() == 150.0


def test_species_type_table_holds_the_BASE_forme():
    from agents.model.damage_tables import build_species_types, _T2I
    types = build_species_types(_N_SPECIES)
    castform = gen3_data.species.get("castform")
    # NORMAL (the base), not Castform-Sunny's FIRE / -Rainy's WATER / -Snowy's ICE.
    assert types[castform.num].tolist() == [_T2I["NORMAL"], 0]


def test_move_prior_does_not_double_count_formes():
    # The prior accumulator is `prob[snum, bnum] += usage`; iterating formes would multiply a
    # species' usage mass by its forme count. Castform has 3 formes AND real Smogon usage, so
    # its cell is the direct probe: it must equal the SINGLE base usage, not 4x it.
    logits = build_move_prior_logits(_N_SPECIES, _N_MOVES)
    snum, mnum = _num("castform", "raindance")
    want = gen3_data.priors.moves("castform")["raindance"]
    assert torch.sigmoid(logits[snum, mnum]).item() == pytest.approx(want, abs=1e-4)


def test_move_self_boosts_ignores_non_battle_stats_instead_of_raising() -> None:
    """A gen-3 self-boost stat OUTSIDE the five battle stats must be SKIPPED, not fatal.

    REGRESSION (2026-08-18, `gen3_double_team_v1`). `MOVE_SELF_BOOSTS` is `[n_moves, 5]` over
    (atk, def, spa, spd, spe) and indexed with a bare `_SELF_BOOST_STAT_ORDER.index(stat)`. The
    moment the extractor admitted Double Team's `selfBoosts: {evasion: 1}`, that raised
    `ValueError: tuple.index(x): x not in tuple` at MODEL-INIT — **321 test failures from a
    one-row data change**.

    The durable lesson is about SCOPE, not about evasion: the extractor's `_self_boosts` guard was
    load-bearing for TWO consumers, and only one of them was the reason written down. The rust
    engine's stated reason ("the accuracy roll ignores the evasion table") had gone stale and was
    safe to relax; THIS table's reason is live. Neither `extractor_parity_test` nor the obs golden
    covers it, because neither builds the damage tables — so relaxing a data guard needs the FULL
    suite, not the two gates that look topical.

    Evasion legitimately has no priced consequence in the DamageOperator, so skipping it leaves the
    tensor bit-identical to before the data changed — which is what this asserts.
    """
    n_moves = 400
    t = build_self_boost_tables(n_moves)["MOVE_SELF_BOOSTS"]

    # Double Team carries `selfBoosts: {evasion: 1}` in the data; its row must be ALL ZERO here
    # (no column exists for evasion) rather than raising or landing in a battle-stat column.
    dt = gen3_data.moves.get("doubleteam")
    assert dt is not None and dt.self_boosts, (
        "doubleteam must carry selfBoosts in the data — if this fails the extractor guard "
        "regressed, not the table"
    )
    assert float(t[dt.num].abs().sum()) == 0.0, (
        f"Double Team must contribute NOTHING to MOVE_SELF_BOOSTS, got {t[dt.num].tolist()}"
    )

    # The canonical battle-stat carrier is unaffected — the skip must not swallow real rows.
    sd = gen3_data.moves.get("swordsdance")
    assert sd is not None and float(t[sd.num, 0]) == 2.0, "Swords Dance must still price +2 atk"
