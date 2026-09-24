"""Pins for the fork's READING fixes found by the TRUTH AUDIT (`designs/rust_sim/one_sided_view.md`
§4b, R1–R3) and by the Rust core's `present()` (M2's findings PE-V10 / PE-R1b / PE-V16, fixed as
`gen3_pe_reading_fixes_v1`, `designs/rust_sim/present.md` §3). Each is a sim fact poke-env read
wrong, established against the engine and the pinned Showdown source; each test FAILS on the
upstream behaviour.

The end-to-end gate for all three is the Rust Core parity harness's slice V
(`agents/battle/rust_core_parity_views.py`), which compares the reading against the SIM at every
decision; these are the constructed, one-rule-per-test pins.
"""
from poke_env.battle.battle import Battle


def _battle() -> Battle:
    b = Battle("tag", "A", None, gen=3)  # type: ignore[arg-type]
    b.player_role = "p1"
    return b


def _feed(battle: Battle, *lines: str) -> None:
    for line in lines:
        battle.parse_message(line.split("|"))


def test_r1_a_new_status_starts_its_own_count():
    """A Rest taken while badly poisoned: the sim's sleep is brand new, so its count is 0 — the
    upstream setter carried the toxic count (2) into the sleep."""
    b = _battle()
    _feed(b, "|switch|p2a: Suicune|Suicune|100/100", "|-status|p2a: Suicune|tox",
          "|-damage|p2a: Suicune|94/100 tox|[from] psn", "|turn|2",
          "|-damage|p2a: Suicune|82/100 tox|[from] psn", "|turn|3")
    mon = b.opponent_active_pokemon
    assert mon.status_counter == 2, "fixture: two toxic chips counted"
    _feed(b, "|-status|p2a: Suicune|slp|[from] move: Rest")
    assert mon.status_counter == 0
    _feed(b, "|cant|p2a: Suicune|slp")
    assert mon.status_counter == 1


def test_r1_the_same_status_restated_keeps_its_count():
    b = _battle()
    _feed(b, "|switch|p2a: Snorlax|Snorlax, M|100/100", "|-status|p2a: Snorlax|slp",
          "|cant|p2a: Snorlax|slp")
    mon = b.opponent_active_pokemon
    mon.status = "slp"
    assert mon.status_counter == 1


def test_r2_psych_up_copies_the_TARGETS_stages_onto_the_USER():
    """`|-copyboost|USER|TARGET|[from] move: Psych Up` — the sim: `source.boosts = target.boosts`.
    Upstream wrote the user's (empty) stages over the target's."""
    b = _battle()
    _feed(b, "|switch|p1a: Suicune|Suicune|100/100", "|switch|p2a: Regice|Regice|100/100",
          "|-boost|p1a: Suicune|spa|2", "|-boost|p1a: Suicune|spd|2",
          "|-copyboost|p2a: Regice|p1a: Suicune|[from] move: Psych Up")
    assert b.active_pokemon.boosts["spa"] == 2, "the target keeps its own stages"
    assert b.opponent_active_pokemon.boosts["spa"] == 2, "the user takes the target's stages"
    assert b.opponent_active_pokemon.boosts["spd"] == 2


def test_r3_our_active_pp_is_the_requests():
    """The request's `pp` is the sim's word. A Pressure the client cannot infer costs 2 in the sim
    and 1 in the sighting count; upstream never re-read it."""
    b = _battle()
    side = {"name": "A", "id": "p1", "pokemon": [{
        "ident": "p1: Snorlax", "details": "Snorlax, M", "condition": "400/400", "active": True,
        "stats": {"atk": 250, "def": 200, "spa": 150, "spd": 250, "spe": 100},
        "moves": ["selfdestruct", "bodyslam", "rest", "curse"], "baseAbility": "immunity",
        "item": "leftovers", "pokeball": "pokeball"}]}
    moves = [{"move": "Self-Destruct", "id": "selfdestruct", "pp": 8, "maxpp": 8,
              "target": "allAdjacent", "disabled": False},
             {"move": "Body Slam", "id": "bodyslam", "pp": 24, "maxpp": 24, "target": "normal",
              "disabled": False}]
    b.parse_request({"active": [{"moves": moves}], "side": side})
    b.parse_message("|move|p1a: Snorlax|Self-Destruct|p2a: Aerodactyl".split("|"))
    assert b.active_pokemon.moves["selfdestruct"].current_pp == 7, "fixture: counted 1"
    moves[0]["pp"] = 6                     # the sim charged Pressure: 2
    b.parse_request({"active": [{"moves": moves}], "side": side})
    assert b.active_pokemon.moves["selfdestruct"].current_pp == 6


# ---------------------------------------------------------------------------------------------
# gen3_pe_reading_fixes_v1 — the Rust core M2 findings, fixed in the fork. The truth for each is
# the pinned Showdown source (`deps/pokemon-showdown`): `sim/pokemon.ts` clearVolatile (a faint
# zeroes `boosts`); `data/conditions.ts` tox (`onSwitchIn` stage = 0, `onResidual` stage++ capped
# at 15 then chips; gen 3 inherits it through gen 4's mod); `data/abilities.ts` flashfire (the
# volatile ends only with the holder leaving the field).
# ---------------------------------------------------------------------------------------------

_TOX = ["|switch|p2a: Zapdos|Zapdos|100/100", "|-status|p2a: Zapdos|tox"]


def test_pe_v10_a_faint_clears_the_stat_stages():
    """Upstream kept the stages the mon died with until `switch_out` — the replacement decision
    encoded them in `active_context`."""
    b = _battle()
    _feed(b, "|switch|p2a: Zapdos|Zapdos|100/100", "|-boost|p2a: Zapdos|spa|1",
          "|-unboost|p2a: Zapdos|spe|2", "|faint|p2a: Zapdos")
    mon = b.opponent_active_pokemon
    assert mon.fainted and mon.active, "fixture: the corpse is still in its slot"
    assert all(v == 0 for v in mon.boosts.values()), mon.boosts


def test_pe_v10_a_zero_hp_token_faints_and_clears_too():
    """`set_hp_status("0 fnt")` is the other road into `faint()` (a `-damage` KO line)."""
    b = _battle()
    _feed(b, "|switch|p2a: Zapdos|Zapdos|100/100", "|-boost|p2a: Zapdos|def|2",
          "|-damage|p2a: Zapdos|0 fnt")
    assert all(v == 0 for v in b.opponent_active_pokemon.boosts.values())


def test_pe_r1b_the_count_moves_at_the_residual_chip_not_at_turn():
    b = _battle()
    _feed(b, *_TOX)
    mon = b.opponent_active_pokemon
    _feed(b, "|turn|2")
    assert mon.status_counter == 0, "no residual yet: stage 0 (upstream ticked to 1 at |turn|)"
    _feed(b, "|-damage|p2a: Zapdos|94/100 tox|[from] psn")
    assert mon.status_counter == 1, "the chip is stage 1"


def test_pe_r1b_between_the_residual_and_the_next_turn_the_count_is_current():
    """An end-of-turn forced replacement's decision sits after the residual and before `|turn|`:
    the sim is already at the new stage (upstream read one BEHIND)."""
    b = _battle()
    _feed(b, *_TOX, "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
          "|-damage|p2a: Zapdos|82/100 tox|[from] psn")
    assert b.opponent_active_pokemon.status_counter == 2


def test_pe_r1b_a_post_residual_entrant_starts_at_zero():
    """A mon that re-enters after the residual (a post-faint replacement) has had no chip since
    `tox.onSwitchIn` — upstream read one AHEAD at the next `|turn|`."""
    b = _battle()
    _feed(b, *_TOX, "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
          "|switch|p2a: Snorlax|Snorlax, M|100/100")
    zapdos = b.get_pokemon("p2a: Zapdos")
    assert zapdos.status_counter == 0, "benched: the effective stage is 0"
    _feed(b, "|faint|p2a: Snorlax", "|switch|p2a: Zapdos|Zapdos|94/100 tox", "|turn|3")
    assert zapdos.status_counter == 0, "no residual since re-entry"
    _feed(b, "|-damage|p2a: Zapdos|88/100 tox|[from] psn")
    assert zapdos.status_counter == 1


def test_pe_r1b_the_switch_in_resets_a_stale_count():
    """`tox.onSwitchIn` — even for a count the reading did not zero at the switch-out."""
    b = _battle()
    _feed(b, *_TOX, "|switch|p2a: Snorlax|Snorlax, M|100/100")
    zapdos = b.get_pokemon("p2a: Zapdos")
    zapdos._status_counter = 3
    _feed(b, "|switch|p2a: Zapdos|Zapdos|100/100 tox")
    assert zapdos.status_counter == 0


def test_pe_r1b_a_chip_that_kos_does_not_count_and_plain_poison_never_counts():
    b = _battle()
    _feed(b, *_TOX, "|-damage|p2a: Zapdos|6/100 tox|[from] psn", "|-damage|p2a: Zapdos|0 fnt|[from] psn")
    assert b.opponent_active_pokemon.status_counter == 1, "the KO chip: status already fnt"
    c = _battle()
    _feed(c, "|switch|p2a: Zapdos|Zapdos|100/100", "|-status|p2a: Zapdos|psn",
          "|-damage|p2a: Zapdos|88/100 psn|[from] psn")
    assert c.opponent_active_pokemon.status_counter == 0, "regular poison has no stage"


def test_pe_r1b_the_stage_caps_at_15():
    b = _battle()
    _feed(b, *_TOX)
    for hp in range(99, 79, -1):
        _feed(b, f"|-damage|p2a: Zapdos|{hp}/100 tox|[from] psn")
    assert b.opponent_active_pokemon.status_counter == 15


def test_pe_v16_flash_fire_survives_its_holders_own_fire_move():
    """Upstream `Pokemon.moved` ended FLASH_FIRE on the holder's damaging Fire move; the sim keeps
    the volatile (and its ×1.5) until the holder leaves the field."""
    from poke_env.battle.effect import Effect

    b = _battle()
    _feed(b, "|switch|p1a: Metagross|Metagross|100/100",
          "|switch|p2a: Houndoom|Houndoom, M|100/100",
          "|move|p1a: Metagross|Hidden Power|p2a: Houndoom",
          "|-immune|p2a: Houndoom",
          "|-start|p2a: Houndoom|ability: Flash Fire",
          "|move|p2a: Houndoom|Flamethrower|p1a: Metagross",
          "|move|p2a: Houndoom|Fire Blast|p1a: Metagross")
    houndoom = b.opponent_active_pokemon
    assert Effect.FLASH_FIRE in houndoom.effects
    _feed(b, "|switch|p2a: Zapdos|Zapdos|100/100")
    assert Effect.FLASH_FIRE not in houndoom.effects, "ends with the holder leaving the field"
