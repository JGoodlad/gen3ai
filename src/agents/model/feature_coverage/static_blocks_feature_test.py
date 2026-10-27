"""
Network-REACHABILITY coverage for the STATIC (non-history) observation blocks.

The turn-history (TurnDelta) edge cases get full capture+network tests elsewhere
in this folder. THIS file proves the major *static* obs regions — per-Pokémon
fields, the global env/weather/hazards/screens block, the reactive scalars
(including the gen3_trapping_signals_v1 trapped/maybe_trapped bits and one
matchup cell), and the active-context volatiles — actually reach the real
`Gen3FeaturesExtractor` output (no dead obs regions a stale layout index or a
dropped concat would silently bury).

Because these blocks are produced by `Gen3ObservationEncoder.encode(battle)`,
which needs a LIVE battle, we do NOT capture them here. Their *capture
correctness* (the right bit/scalar landing at the right offset off a real
protocol stream) is covered by the encoders' own unit + fuzz tests:
  - per-Pokémon: `pokemon.py` / `moves.py` tests + the poke_env_gaps fuzz suite
  - global env:  `global_env_test.py`
  - reactive:    `reactive_test.py` (+ `alignment_test`)
  - volatiles:   `gen3_effects_test.py` + `active_context` tests
This file closes the orthogonal hop those don't exercise: that a value written
into the block's layout offset moves the network's policy/value heads.

APPROACH (mirrors `features_extractor_test.py`): write ONE dim at its absolute
layout offset on an otherwise-zero-but-VALID obs and assert the output moves.
For per-Pokémon slot dims the obs must first make the slot PRESENT (HP +
species_known on our active slot), exactly as the exemplar does — a slot that
looks empty/fainted is removed by the transformer's key-padding mask, so a lone
dim there would be (correctly) invisible. All absolute offsets are derived from
the live layout dicts + the imported OFFSET_/_DIM constants — never hardcoded.

Each test asserts BOTH heads move via `assert_reaches_network`, except where a
region only feeds one head (called out in a comment).
"""

from agents.model.feature_coverage._support import (
    feature_model,
    obs_zero,
    set_region,
    assert_reaches_network,
    policy_value_changed,
)
from agents.observation.constants import (
    OFFSET_OUR_TEAM,
    POKEMON_FULL_DIM,
    POKEMON_VECTOR_DIM,
    BOOSTS_DIM,
)
from agents.observation.gen3_effects import VOLATILE_SLOTS


# ---------------------------------------------------------------------------
# Helpers for placing dims at absolute offsets derived from the live layout
# ---------------------------------------------------------------------------

def _pmon_field_offset(layout, field, sub=None):
    """Absolute offset of a per-Pokémon sub-field within OUR ACTIVE slot (slot 0).

    `field` is a key of layout["pokemon"]; `sub` (optional) is a key of that
    field's nested "layout" (or "slot_layout" for moves). All from the layout
    dicts — no magic numbers.
    """
    pkmn = layout["pokemon"][field]
    off_within_slot = pkmn["offset"]
    if sub is not None:
        sub_layout = pkmn.get("layout", {})
        # moves nests its per-slot layout under "slot_layout"
        sub_layout = sub_layout.get("slot_layout", sub_layout)
        off_within_slot += sub_layout[sub]["offset"]
    # We always use slot 0 (i == 0), so the slot base is OFFSET_OUR_TEAM.
    return OFFSET_OUR_TEAM + 0 * POKEMON_FULL_DIM + off_within_slot


def _present_active_slot_base(layout):
    """A valid zero obs with OUR slot-0 made PRESENT and ACTIVE.

    Sets HP fraction > 0 + species_known (so the transformer key-padding mask
    treats the slot as a live team token, not padding) and the active flag (so
    the slot is also surfaced as `our_active_refined`). Mirrors the populate()
    logic in features_extractor_test.py. Returns the obs to use as the baseline;
    touching any further dim in this slot is then isolated against it.
    """
    obs = obs_zero(layout)
    hp_off = _pmon_field_offset(layout, "hp")
    sk_off = _pmon_field_offset(layout, "species_known")
    # active flag is the LAST dim of the slot (POKEMON_VECTOR_DIM within slot).
    active_off = OFFSET_OUR_TEAM + 0 * POKEMON_FULL_DIM + POKEMON_VECTOR_DIM
    obs[0, hp_off] = 1.0
    obs[0, sk_off] = 1.0
    obs[0, active_off] = 1.0
    return obs


def _global_offset(layout, field, extra=0):
    """Absolute offset into the global env block (layout["global_layout"])."""
    start = layout["parts"]["global"]["start"]
    return start + layout["global_layout"][field]["offset"] + extra


def _reactive_offset(layout, field, extra=0):
    """Absolute offset into the reactive block (layout["reactive_layout"])."""
    start = layout["parts"]["reactive"]["start"]
    return start + layout["reactive_layout"][field]["offset"] + extra


def _context_offset(layout, which, extra=0):
    """Absolute offset into an active-context block.

    which=0 → our active context, which=1 → opp active context. Each context is
    `active_context_dim` wide (boosts + volatiles).
    """
    start = layout["parts"]["context"]["start"]
    return start + which * layout["active_context_dim"] + extra


# ===========================================================================
# A. Per-Pokémon slot (on OUR active slot 0, made present + active)
# ===========================================================================

def test_item_consumed_bit_reaches_network():
    # Item block is [id, known, consumed]; the consumed flag must reach the net.
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    off = _pmon_field_offset(layout, "items", "consumed")
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "per-mon item consumed bit")


def test_status_onehot_reaches_network():
    # 7-dim condition one-hot; touch the BRN slot (index 1).
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    off = _pmon_field_offset(layout, "condition")
    set_region(variant, off + 1, [1.0])  # BRN
    assert_reaches_network(model, base, variant, "per-mon status one-hot (BRN)")


def test_sleep_counter_reaches_network():
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    off = _pmon_field_offset(layout, "status_counters")  # [sleep, toxic]
    set_region(variant, off, [0.75])
    assert_reaches_network(model, base, variant, "per-mon sleep_counter_norm")


def test_toxic_counter_reaches_network():
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    off = _pmon_field_offset(layout, "status_counters")
    set_region(variant, off + 1, [0.5])  # toxic is the 2nd status-counter dim
    assert_reaches_network(model, base, variant, "per-mon toxic_counter_norm")


def test_spread_ev_reaches_network():
    # Spread block: IVs(6) EVs(6) spread_known(1) nature(5). Touch an EV dim.
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    spread = layout["pokemon"]["spread"]["layout"]
    off = _pmon_field_offset(layout, "spread") + spread["evs"]["offset"] + 1  # Atk EV
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "per-mon spread EV dim")


def test_spread_nature_reaches_network():
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    spread = layout["pokemon"]["spread"]["layout"]
    off = _pmon_field_offset(layout, "spread") + spread["nature"]["offset"]  # Atk nature mod
    set_region(variant, off, [1.1])
    assert_reaches_network(model, base, variant, "per-mon spread nature dim")


def test_move_never_miss_bit_reaches_network():
    # Move slot 0's never-miss accuracy bit.
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    off = _pmon_field_offset(layout, "moves", "never_miss")
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "per-mon move never_miss bit")


def test_move_current_pp_reaches_network():
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    off = _pmon_field_offset(layout, "moves", "current_pp")
    set_region(variant, off, [0.5])
    assert_reaches_network(model, base, variant, "per-mon move current_pp dim")


def test_hp_candidate_block_reaches_network():
    # HP-candidate block = hp_revealed(1) + 16 type probs. Touch a type-prob dim.
    model, layout, _ = feature_model()
    base = _present_active_slot_base(layout)
    variant = base.clone()
    hp_block = layout["pokemon"]["hp_block"]["layout"]
    off = _pmon_field_offset(layout, "hp_block") + hp_block["hp_type_probs"]["offset"]
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "per-mon HP-candidate type-prob dim")


def test_species_known_flag_reaches_network():
    # species_known toggles slot presence (key-padding). Compare a present slot
    # WITH species_known against the same slot WITHOUT it: the flag itself, plus
    # the masking it controls, must move the net. Base here is a slot that is
    # present via HP only; variant adds species_known.
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    hp_off = _pmon_field_offset(layout, "hp")
    active_off = OFFSET_OUR_TEAM + POKEMON_VECTOR_DIM
    base[0, hp_off] = 1.0
    base[0, active_off] = 1.0
    variant = base.clone()
    off = _pmon_field_offset(layout, "species_known")
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "per-mon species_known flag")


# ===========================================================================
# B. Global env block (weather / hazards / screens)
# ===========================================================================

def test_weather_onehot_reaches_network():
    # weather block = 5-dim one-hot + permanence + turns. Touch the SUN bit (idx 1).
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    off = _global_offset(layout, "weather")
    set_region(variant, off + 1, [1.0])  # SUN one-hot
    assert_reaches_network(model, base, variant, "global weather one-hot (SUN)")


def test_weather_permanence_reaches_network():
    # permanence is the 6th dim of the 7-dim weather block (after the 5 one-hots).
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    from agents.observation.constants import WEATHER_ONEHOT_DIM
    off = _global_offset(layout, "weather") + WEATHER_ONEHOT_DIM
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "global weather permanence dim")


def test_weather_turns_remaining_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    from agents.observation.constants import WEATHER_ONEHOT_DIM
    off = _global_offset(layout, "weather") + WEATHER_ONEHOT_DIM + 1
    set_region(variant, off, [0.6])
    assert_reaches_network(model, base, variant, "global weather turns-remaining dim")


def test_spikes_layer_reaches_network():
    # hazards = [our_spikes, opp_spikes]; touch our-side spikes.
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    off = _global_offset(layout, "hazards")
    set_region(variant, off, [0.66])
    assert_reaches_network(model, base, variant, "global our-side Spikes dim")


def test_screen_reflect_both_sides_distinct_and_reach_network():
    # screens = 8 dims = [reflect ours, reflect opp, light_screen ours, opp,
    # safeguard ours, opp, mist ours, opp]. Confirm our-side != their-side dim
    # (different offsets) and BOTH reach the net.
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    sc = _global_offset(layout, "screens")
    our = base.clone(); set_region(our, sc + 0, [1.0])      # Reflect ours
    their = base.clone(); set_region(their, sc + 1, [1.0])  # Reflect opp
    assert_reaches_network(model, base, our, "global Reflect ours")
    assert_reaches_network(model, base, their, "global Reflect opp")
    # Distinct dims → distinct network outputs.
    pi, vf = policy_value_changed(model, our, their)
    assert pi and vf, "our-side vs their-side Reflect must be different obs dims"


def test_screen_light_screen_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    sc = _global_offset(layout, "screens")
    our = base.clone(); set_region(our, sc + 2, [1.0])
    their = base.clone(); set_region(their, sc + 3, [1.0])
    assert_reaches_network(model, base, our, "global Light Screen ours")
    assert_reaches_network(model, base, their, "global Light Screen opp")
    pi, vf = policy_value_changed(model, our, their)
    assert pi and vf, "our vs their Light Screen must be different dims"


def test_screen_safeguard_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    sc = _global_offset(layout, "screens")
    our = base.clone(); set_region(our, sc + 4, [1.0])
    their = base.clone(); set_region(their, sc + 5, [1.0])
    assert_reaches_network(model, base, our, "global Safeguard ours")
    assert_reaches_network(model, base, their, "global Safeguard opp")
    pi, vf = policy_value_changed(model, our, their)
    assert pi and vf, "our vs their Safeguard must be different dims"


def test_screen_mist_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    sc = _global_offset(layout, "screens")
    our = base.clone(); set_region(our, sc + 6, [1.0])
    their = base.clone(); set_region(their, sc + 7, [1.0])
    assert_reaches_network(model, base, our, "global Mist ours")
    assert_reaches_network(model, base, their, "global Mist opp")
    pi, vf = policy_value_changed(model, our, their)
    assert pi and vf, "our vs their Mist must be different dims"


# ===========================================================================
# C. Reactive block (scalars + trapping signals + one matchup cell)
# ===========================================================================

# gen3_entity_rehome_v1: forced_struggle is DELETED from the obs (derivable — the action mask
# carries the authoritative bit and the req-move legal bits are all-zero exactly then), and the
# matchup matrices are deleted with it (D/V edges own pair physics). trapped / maybe_trapped /
# protect_odds moved to the PER-MON slots — covered below at their entity homes.


def test_fainted_count_reaches_network():
    # fainted = [our_count, opp_count]; touch our-count.
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    off = _reactive_offset(layout, "fainted")
    set_region(variant, off, [0.5])
    assert_reaches_network(model, base, variant, "reactive our fainted-count")

def _our_mon_offset(layout, slot: int, field_offset: int) -> int:
    """Absolute offset of a per-mon field on OUR team slot `slot`."""
    from agents.observation.constants import POKEMON_FULL_DIM
    return layout["parts"]["our_team"]["start"] + slot * POKEMON_FULL_DIM + field_offset


def test_trapped_bit_reaches_network():
    # gen3_entity_rehome_v1: `trapped` rides OUR ACTIVE's mon slot now. Must reach the net.
    from agents.observation.constants import POKEMON_TRAPPED_OFFSET
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    set_region(variant, _our_mon_offset(layout, 0, POKEMON_TRAPPED_OFFSET), [1.0])
    assert_reaches_network(model, base, variant, "per-mon trapped bit")


def test_maybe_trapped_bit_reaches_network():
    # gen3_entity_rehome_v1: `maybe_trapped` — the highest-value bit (switches stay legal,
    # so it's the only way the model sees trap risk) — on the mon slot. Must reach.
    from agents.observation.constants import POKEMON_MAYBE_TRAPPED_OFFSET
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    set_region(variant, _our_mon_offset(layout, 0, POKEMON_MAYBE_TRAPPED_OFFSET), [1.0])
    assert_reaches_network(model, base, variant, "per-mon maybe_trapped bit")


def test_protect_odds_reaches_network():
    # gen3_entity_rehome_v1: per-mon protect-success odds (every mon owns its stall state).
    # On a BENCH slot on purpose — the re-home's point is that the fact rides every entity,
    # not just the active. The bench mon must be alive (hp>0) or its token is key-masked.
    from agents.observation.constants import POKEMON_PROTECT_OFFSET
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    hp_off = layout["pokemon"]["hp"]["offset"]
    set_region(base, _our_mon_offset(layout, 1, hp_off), [0.8])
    variant = base.clone()
    set_region(variant, _our_mon_offset(layout, 1, POKEMON_PROTECT_OFFSET), [1.0])
    assert_reaches_network(model, base, variant, "per-mon protect odds (live bench mon)")


def test_turns_since_progress_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    off = _reactive_offset(layout, "turns_since_progress")
    set_region(variant, off, [0.7])
    assert_reaches_network(model, base, variant, "board turns_since_progress")


def test_wish_floating_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    off = _reactive_offset(layout, "wish_floating_our")
    set_region(variant, off, [0.5])
    assert_reaches_network(model, base, variant, "board wish_floating_our")


# ===========================================================================
# D. Active-context / volatiles block (2 active contexts: boosts + volatiles)
# ===========================================================================

def test_active_boost_dim_reaches_network():
    # Boosts occupy the first BOOSTS_DIM dims of OUR active context.
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    off = _context_offset(layout, which=0, extra=0)  # atk-positive boost dim
    set_region(variant, off, [0.5])
    assert_reaches_network(model, base, variant, "active-context our boost dim")


def test_volatile_substitute_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    vol_idx = VOLATILE_SLOTS.index("substitute")
    off = _context_offset(layout, which=0, extra=BOOSTS_DIM + vol_idx)
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "active-context Substitute volatile")


def test_volatile_leech_seed_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    vol_idx = VOLATILE_SLOTS.index("leechseed")
    off = _context_offset(layout, which=0, extra=BOOSTS_DIM + vol_idx)
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "active-context Leech Seed volatile")


def test_volatile_confusion_reaches_network():
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    vol_idx = VOLATILE_SLOTS.index("confusion")
    off = _context_offset(layout, which=0, extra=BOOSTS_DIM + vol_idx)
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "active-context Confusion volatile")


def test_opp_active_volatile_reaches_network():
    # The OPP active context (which=1) is a different block; confirm it too is live.
    model, layout, _ = feature_model()
    base = obs_zero(layout)
    variant = base.clone()
    vol_idx = VOLATILE_SLOTS.index("substitute")
    off = _context_offset(layout, which=1, extra=BOOSTS_DIM + vol_idx)
    set_region(variant, off, [1.0])
    assert_reaches_network(model, base, variant, "opp active-context Substitute volatile")
