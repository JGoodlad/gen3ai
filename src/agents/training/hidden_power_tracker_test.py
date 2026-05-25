"""Unit tests for HiddenPowerTracker."""
import numpy as np
import pytest
from dataclasses import dataclass
from poke_env.battle.pokemon_type import PokemonType

from agents.training.hidden_power_tracker import HiddenPowerTracker, HIDDEN_POWER_TYPE_ORDER

# Convenience index lookup by type name
HP_IDX = {t.name.lower(): i for i, t in enumerate(HIDDEN_POWER_TYPE_ORDER)}


@dataclass
class MockMon:
    species: str
    type_1: PokemonType
    type_2: PokemonType | None
    ability: str | None
    status: object = None   # poke_env.battle.status.Status | None at runtime


def make_tracker(priors: dict | None = None) -> HiddenPowerTracker:
    """Build a tracker with an explicit priors dict (bypasses file loading)."""
    return HiddenPowerTracker(_priors=priors if priors is not None else {})


# ---------------------------------------------------------------------------
# Core filtering
# ---------------------------------------------------------------------------

def test_blissey_2x_leaves_fighting_only():
    """Normal/None mon at 2× → only Fighting can be super-effective vs Normal."""
    tracker = make_tracker()
    blissey = MockMon("blissey", PokemonType.NORMAL, None, "naturalcure")
    tracker.observe("blissey", 2.0, blissey)
    probs = tracker.get_probs("blissey")
    surviving = [i for i, p in enumerate(probs) if p > 0]
    assert surviving == [HP_IDX["fighting"]], f"Expected [fighting] only, got {surviving}"


def test_volt_absorb_lanturn_0x_leaves_electric_only():
    """Water/Electric mon with Volt Absorb at 0× → only Electric is immune via ability."""
    tracker = make_tracker()
    lanturn = MockMon("lanturn", PokemonType.WATER, PokemonType.ELECTRIC, "voltabsorb")
    tracker.observe("lanturn", 0.0, lanturn)
    probs = tracker.get_probs("lanturn")
    surviving = [i for i, p in enumerate(probs) if p > 0]
    assert surviving == [HP_IDX["electric"]], f"Expected [electric] only, got {surviving}"


def test_levitate_weezing_0x_leaves_ground_only():
    """Pure Poison mon with Levitate at 0× → only Ground is nullified by Levitate."""
    tracker = make_tracker()
    weezing = MockMon("weezing", PokemonType.POISON, None, "levitate")
    tracker.observe("weezing", 0.0, weezing)
    probs = tracker.get_probs("weezing")
    surviving = [i for i, p in enumerate(probs) if p > 0]
    assert surviving == [HP_IDX["ground"]], f"Expected [ground] only, got {surviving}"


def test_thick_fat_snorlax_05x_leaves_ice_and_fire():
    """Snorlax (Normal, Thick Fat) at 0.5× → only Ice and Fire are halved by Thick Fat.

    Without Thick Fat handling, no HP type gives 0.5× vs pure Normal (all are 1× or
    2× via Fighting, 0× via Ghost), so this observation would wipe every candidate.
    """
    tracker = make_tracker()
    snorlax = MockMon("snorlax", PokemonType.NORMAL, None, "thickfat")
    tracker.observe("snorlax", 0.5, snorlax)
    probs = tracker.get_probs("snorlax")
    surviving = {i for i, p in enumerate(probs) if p > 0}
    assert surviving == {HP_IDX["ice"], HP_IDX["fire"]}, f"got {surviving}"


def test_flash_fire_frozen_arcanine_05x_leaves_fire():
    """Flash Fire is SUPPRESSED while the holder is frozen in Gen 3 — Fire moves
    fall through and get the normal Fire-vs-Fire 0.5× resistance, then thaw the
    target. Without this quirk, Fire would be wrongly eliminated.
    """
    from poke_env.battle.status import Status
    tracker = make_tracker()
    arcanine_frozen = MockMon("arcanine", PokemonType.FIRE, None, "flashfire",
                              status=Status.FRZ)
    tracker.observe("arcanine", 0.5, arcanine_frozen)
    probs = tracker.get_probs("arcanine")
    assert probs[HP_IDX["fire"]] > 0.0, "Fire HP must survive 0.5× on frozen Flash Fire mon"


def test_flash_fire_healthy_arcanine_0x_leaves_fire_only():
    """When the holder is NOT frozen, Flash Fire fully blocks Fire (0×). Other
    types vs Fire/None are 0.5× or 1× — only Fire itself produces 0×.
    """
    tracker = make_tracker()
    arcanine = MockMon("arcanine", PokemonType.FIRE, None, "flashfire", status=None)
    tracker.observe("arcanine", 0.0, arcanine)
    probs = tracker.get_probs("arcanine")
    surviving = {i for i, p in enumerate(probs) if p > 0}
    assert surviving == {HP_IDX["fire"]}, f"got {surviving}"


def test_quad_se_hit_matches_bucketed_2x():
    """Quad-effective hits (4×) report as 2.0 via Showdown's effectiveness bucket.

    HP Grass on Swampert (Water/Ground): 2× × 2× = 4×, but Showdown emits
    |-supereffective| → poke-env stores 2.0. The tracker must match the
    bucketed value, not the raw 4× multiplier, or Grass gets wrongly eliminated
    as a candidate and the whole vector zeros out.
    """
    tracker = make_tracker()
    swampert = MockMon("swampert", PokemonType.WATER, PokemonType.GROUND, "torrent")
    # Observed effectiveness from poke-env is 2.0, not 4.0
    tracker.observe("zapdos", 2.0, swampert)
    probs = tracker.get_probs("zapdos")
    # Grass is the only type that gives any SE multiplier on Water/Ground (4×),
    # so it must be the lone survivor — and the prior weight must be preserved.
    assert probs[HP_IDX["grass"]] > 0.0, "Grass (4×, bucketed to SE) must survive"
    survivors = {i for i, p in enumerate(probs) if p > 0}
    assert survivors == {HP_IDX["grass"]}, f"got {survivors}"


def test_double_resisted_quarter_hit_matches_bucketed_0_5x():
    """0.25× double-resisted hits report as 0.5 via the bucket.

    HP Grass on Charizard (Fire/Flying): 0.5× × 0.5× = 0.25×, bucketed to 0.5.
    Tracker must accept Grass even though raw mult is 0.25 not 0.5.
    """
    tracker = make_tracker()
    charizard = MockMon("charizard", PokemonType.FIRE, PokemonType.FLYING, "blaze")
    tracker.observe("starmie", 0.5, charizard)
    probs = tracker.get_probs("starmie")
    # Grass survives (0.25× bucketed to resisted).
    assert probs[HP_IDX["grass"]] > 0.0, "Grass (0.25×, bucketed to resisted) must survive"


# ---------------------------------------------------------------------------
# Prior initialisation
# ---------------------------------------------------------------------------

def test_prior_weights_preserved_for_surviving_types():
    """After first observe, non-eliminated types retain their prior weights."""
    priors = {"jolteon": {"ice": 0.70, "grass": 0.25, "fire": 0.05}}
    tracker = HiddenPowerTracker(_priors=priors)
    jolteon = MockMon("jolteon", PokemonType.ELECTRIC, None, "voltabsorb")
    # Neutral hit (1.0×) on Electric/Volt Absorb mon eliminates:
    # - Electric (0× via Volt Absorb), not 1.0
    # - Fire/Water/Grass/Ice/Ground: check actual multipliers
    # Ice vs Electric = 1×, Grass vs Electric = 1×, Fire vs Electric = 1×
    # So ice, grass, fire should all survive 1.0× observation
    tracker.observe("jolteon", 1.0, jolteon)
    probs = tracker.get_probs("jolteon")
    assert abs(probs[HP_IDX["ice"]] - 0.70) < 1e-6
    assert abs(probs[HP_IDX["grass"]] - 0.25) < 1e-6
    assert abs(probs[HP_IDX["fire"]] - 0.05) < 1e-6
    # Electric must be zeroed (Volt Absorb gives 0× ≠ 1.0)
    assert probs[HP_IDX["electric"]] == 0.0


def test_species_not_in_priors_uses_flat_prior():
    """Unknown species (not in priors dict) starts with flat 1/16 per type."""
    tracker = make_tracker()  # empty priors
    # Dragon resists Fire, Water, Grass, Electric (0.5×); Ice and Dragon give 2×;
    # all others give 1×. Observing 0.5× leaves exactly the four resisted types.
    dratini = MockMon("dratini", PokemonType.DRAGON, None, "shedskin")
    tracker.observe("dratini", 0.5, dratini)
    probs = tracker.get_probs("dratini")
    resisted = {PokemonType.FIRE, PokemonType.WATER, PokemonType.GRASS, PokemonType.ELECTRIC}
    for i, t in enumerate(HIDDEN_POWER_TYPE_ORDER):
        if t in resisted:
            assert probs[i] > 0.0, f"{t.name} should survive 0.5× vs Dragon"
        else:
            assert probs[i] == 0.0, f"{t.name} should be eliminated at 0.5× vs Dragon"


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_all_zero_with_prior_entry_raises():
    """Filtering out all prior-listed types raises ValueError (tracker bug)."""
    # Jolteon prior only has ice/grass, but we observe 0× on Electric/Volt Absorb
    # 0× requires Electric type → but ice and grass are 1× vs Electric, not 0×
    # So all prior entries get eliminated → ValueError
    priors = {"jolteon": {"ice": 0.70, "grass": 0.30}}
    tracker = HiddenPowerTracker(_priors=priors)
    jolteon = MockMon("jolteon", PokemonType.ELECTRIC, None, "voltabsorb")
    with pytest.raises(ValueError, match="tracker bug"):
        tracker.observe("jolteon", 0.0, jolteon)


def test_all_zero_without_prior_entry_raises_data_gap():
    """Filtering flat 1/16 to all zeros raises ValueError (data gap message)."""
    # A move impossible against this specific type combo would do it.
    # Use a double-immune case: Ghost/Normal doesn't exist, but we can construct
    # a mon that's immune to all 16 HP types via its type chart.
    # Simpler: use a Pokémon with an ability immunity and arrange it so all
    # flat-prior types are eliminated.
    # Weezing (Poison, Levitate) at 2×: Ground 0× (Levitate), Fighting 0.5×...
    # Actually just pick a case where no HP type gives a 4× hit on a known type pair
    # but we observe 4.0 — but 4.0 isn't a valid effectiveness for a single-type attacker.
    # Easiest: observe 4.0 (impossible effectiveness for HP) against anything.
    tracker = make_tracker()
    blissey = MockMon("blissey", PokemonType.NORMAL, None, "naturalcure")
    # 4.0× is not achievable by any HP type against any single-type Normal mon
    # (max is 2× from Fighting) → all candidates eliminated → ValueError (data gap)
    with pytest.raises(ValueError, match="data gap"):
        tracker.observe("blissey", 4.0, blissey)


# ---------------------------------------------------------------------------
# Behavioural contracts
# ---------------------------------------------------------------------------

def test_idempotent():
    """Calling observe twice with identical arguments produces the same result."""
    tracker = make_tracker()
    blissey = MockMon("blissey", PokemonType.NORMAL, None, "naturalcure")
    tracker.observe("blissey", 2.0, blissey)
    probs_first = tracker.get_probs("blissey").copy()
    tracker.observe("blissey", 2.0, blissey)
    probs_second = tracker.get_probs("blissey")
    np.testing.assert_array_equal(probs_first, probs_second)


def test_get_probs_before_observation_returns_zeros():
    """get_probs on an unseen species returns all-zero before any observe call."""
    tracker = make_tracker()
    probs = tracker.get_probs("jolteon")
    assert probs.shape == (16,)
    assert probs.dtype == np.float32
    assert not np.any(probs)


def test_reset_clears_state():
    """After reset(), get_probs returns zeros for a previously observed species."""
    tracker = make_tracker()
    blissey = MockMon("blissey", PokemonType.NORMAL, None, "naturalcure")
    tracker.observe("blissey", 2.0, blissey)
    assert np.any(tracker.get_probs("blissey"))
    tracker.reset()
    assert not np.any(tracker.get_probs("blissey"))


def test_multiple_observations_narrow_further():
    """Two consecutive 1× observations on different targets narrow the candidate set."""
    tracker = make_tracker()
    # Step 1: 1× on Blissey (Normal).
    # Ghost gives 0×, Fighting gives 2× — both eliminated. 14 survivors.
    blissey = MockMon("blissey", PokemonType.NORMAL, None, "naturalcure")
    tracker.observe("jolteon", 1.0, blissey)
    probs_first = tracker.get_probs("jolteon")
    surviving_first = sum(p > 0 for p in probs_first)
    assert probs_first[HP_IDX["fighting"]] == 0.0
    assert probs_first[HP_IDX["ghost"]] == 0.0
    assert surviving_first == 14

    # Step 2: 1× on Dratini (Dragon).
    # Dragon resists Fire/Water/Grass/Electric (0.5×) and is 2× to Ice/Dragon — all 6 eliminated.
    # 14 - 6 = 8 survivors: Bug, Dark, Flying, Ground, Poison, Psychic, Rock, Steel.
    dratini = MockMon("dratini", PokemonType.DRAGON, None, "shedskin")
    tracker.observe("jolteon", 1.0, dratini)
    probs_second = tracker.get_probs("jolteon")
    surviving_second = sum(p > 0 for p in probs_second)
    assert surviving_second < surviving_first
    assert surviving_second == 8
