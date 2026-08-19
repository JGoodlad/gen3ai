"""Gen3Battle scripted-parse tests (design §8.1): event emission, conservation,
and state-equivalence vs the classic Battle.

Pure unit tests — Gen3Battle replays hand-built protocol lines offline (move metadata
comes from poke-env's bundled gen3 data, not the Node server).
"""

import logging

import pytest

from poke_env.battle.battle import Battle

from agents.battle.battle_event import (
    EVENT_VALUE_KEYS,
    MESSAGE_POLICY,
    EventKind,
    Policy,
    UnknownMessageType,
    UnsupportedMessageType,
)
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.turn_view import TurnView

LOG = logging.getLogger("gen3battle-test")


# A canonical multi-turn singles script exercising switches, damage, status,
# boost, effectiveness, crit, faint and a forced replacement.
CANONICAL = [
    ["", "player", "p1", "p1user", "", ""],
    ["", "player", "p2", "p2user", "", ""],
    ["", "teamsize", "p1", "6"],
    ["", "teamsize", "p2", "6"],
    ["", "gametype", "singles"],
    ["", "gen", "3"],
    ["", "tier", "[Gen 3] OU"],
    ["", "start"],
    ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
    ["", "switch", "p2a: Tyra", "Tyranitar, L100, M", "100/100"],
    ["", "turn", "1"],
    # turn 1: Zapdos thunderbolts Tyranitar; Tyranitar rock slides (super eff), crit
    ["", "move", "p1a: Zappy", "Thunderbolt", "p2a: Tyra"],
    ["", "-damage", "p2a: Tyra", "52/100"],
    ["", "move", "p2a: Tyra", "Rock Slide", "p1a: Zappy"],
    ["", "-supereffective", "p1a: Zappy"],
    ["", "-crit", "p1a: Zappy"],
    ["", "-damage", "p1a: Zappy", "10/100"],
    ["", "turn", "2"],
    # turn 2: we switch to Skarmory; Tyra earthquakes (immune)
    ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
    ["", "move", "p2a: Tyra", "Earthquake", "p1a: Skarm"],
    ["", "-immune", "p1a: Skarm"],
    ["", "turn", "3"],
    # turn 3: Skarmory toxics Tyranitar; Tyra crunches; status + damage
    ["", "move", "p1a: Skarm", "Toxic", "p2a: Tyra"],
    ["", "-status", "p2a: Tyra", "tox"],
    ["", "move", "p2a: Tyra", "Crunch", "p1a: Skarm"],
    ["", "-damage", "p1a: Skarm", "70/100"],
    ["", "-damage", "p2a: Tyra", "46/100", "[from] psn"],
    ["", "turn", "4"],
    # turn 4: Skarmory whirlwinds (phaze) -> Blissey dragged in; Tyra cant (none)
    ["", "move", "p1a: Skarm", "Whirlwind", "p2a: Tyra"],
    ["", "drag", "p2a: Bliss", "Blissey, L100, F", "100/100"],
    ["", "turn", "5"],
]


def feed(battle, lines):
    for line in lines:
        battle.parse_message(line)


def make(cls):
    return cls("battle-gen3ou-canon", "p1user", LOG, gen=3)


# --------------------------------------------------------------------------- #
# State-equivalence: Gen3Battle must not change current-state behaviour.        #
# --------------------------------------------------------------------------- #
def test_state_equivalence_with_classic_battle():
    """Gen3Battle delegates ALL state to super(), so current-state must match the
    classic Battle line-for-line. (active_pokemon is populated from the |request|
    JSON, not parsed protocol, so we compare the team dicts that switch lines fill.)
    """
    classic = make(Battle)
    g3 = make(Gen3Battle)
    feed(classic, CANONICAL)
    feed(g3, CANONICAL)

    assert g3.turn == classic.turn
    assert g3._player_role == classic._player_role
    # opponent_active_pokemon IS derived from switch lines -> must match.
    assert (
        g3.opponent_active_pokemon.species
        == classic.opponent_active_pokemon.species
    )

    def state_of(b):
        out = {}
        for owner in (b.team, b.opponent_team):
            for key, mon in owner.items():
                out[key] = (
                    mon.species,
                    mon.current_hp_fraction,
                    mon.status,
                    mon.fainted,
                    tuple(sorted(mon.boosts.items())),
                )
        return out

    assert state_of(g3) == state_of(classic)


# --------------------------------------------------------------------------- #
# Conservation: every line lands in exactly one policy bucket (design §4.3).    #
# --------------------------------------------------------------------------- #
def test_conservation_balances_mid_battle():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    report = g3.assert_conservation()
    assert report["events_recorded"] > 0
    assert report["unsupported"] == 0


def test_conservation_balances_with_terminal_win():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    g3.won_by("p1user")  # appends a ["", "win", name] sentinel to _replay_data
    report = g3.assert_conservation()
    assert report["terminal_sentinels"] == 1


def test_conservation_detects_a_dropped_line():
    """The invariant must actually FAIL when a line bypasses classification —
    otherwise it proves nothing. Simulate a line appended to _replay_data that
    never went through parse_message (the exact failure it guards against)."""
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    g3.assert_conservation()  # balanced to start
    g3._replay_data.append(["", "move", "p1a: Zappy", "Thunderbolt"])  # not parsed
    with pytest.raises(AssertionError):
        g3.assert_conservation()


def test_sethp_is_evented_with_signed_delta():
    """|-sethp| (Pain Split) is an EVENT so the log is HP-complete on its own: it
    carries the new HP plus a signed delta the reward manager can read directly."""
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL[:11])  # through turn 1 marker, both leads in (full HP)
    events_before = len(g3.events)
    g3.parse_message(["", "-sethp", "p1a: Zappy", "50/100", "[from] move: Pain Split"])
    assert len(g3.events) == events_before + 1
    ev = g3.events[-1]
    assert ev.kind is EventKind.SETHP
    assert ev.side == "ours" and ev.actor_species == "zapdos"
    assert ev.value["hp"] == pytest.approx(0.5)
    assert ev.value["amount"] == pytest.approx(-0.5)   # 1.0 -> 0.5
    g3.assert_conservation()


# --------------------------------------------------------------------------- #
# Information-preservation guarantees (the "can't lose info" contract)          #
# --------------------------------------------------------------------------- #
def test_event_payload_schema_holds():
    """Every emitted event carries its kind's REQUIRED payload keys. This is the
    structural guard against the Focus-Punch class of loss: a builder that forgot to
    populate (say) the cant reason or the item id fails here, loudly."""
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    for e in g3.events:
        required = EVENT_VALUE_KEYS.get(e.kind, frozenset())
        missing = required - set(e.value)
        assert not missing, f"{e.kind.name} event missing payload keys {missing}: {e}"


def test_no_event_carries_an_UNDECLARED_payload_key():
    """The other half of the schema (gen3_event_value_schema_v1).

    `EVENT_VALUE_KEYS` is a LOWER bound — it catches a FORGOTTEN key and is blind to an
    invented or renamed one, which is the direction that fails silently: a consumer reading a
    key the producer stopped writing (or never wrote) gets `None` and reads it as "absent". The
    positional-binding sweep's fourth live site was exactly that shape, in an ORACLE. Declaring
    the full vocabulary makes required ∪ optional EXACT, so the drift fails at the producer."""
    from agents.battle.battle_event import declared_value_keys, undeclared_value_keys
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    assert g3.events, "the canonical feed must produce events or this proves nothing"
    for e in g3.events:
        extra = undeclared_value_keys(e)
        assert not extra, (
            f"{e.kind.name} event carries UNDECLARED payload key(s) {sorted(extra)} — declare "
            f"them in EVENT_VALUE_KEYS (a consumer may rely on it) or EVENT_OPTIONAL_KEYS "
            f"(it may be absent). Declared: {sorted(declared_value_keys(e.kind))}. Event: {e}")


def test_the_two_schema_halves_cover_the_same_kinds_and_never_overlap():
    """A kind declared in one half and not the other is a half-written schema; a key in BOTH
    halves is a contradiction (a consumer cannot both rely on it and not)."""
    from agents.battle.battle_event import EVENT_OPTIONAL_KEYS
    assert set(EVENT_VALUE_KEYS) == set(EVENT_OPTIONAL_KEYS), (
        "every kind must appear in BOTH schema halves — missing from optional: "
        f"{sorted(k.name for k in set(EVENT_VALUE_KEYS) - set(EVENT_OPTIONAL_KEYS))}; "
        f"missing from required: "
        f"{sorted(k.name for k in set(EVENT_OPTIONAL_KEYS) - set(EVENT_VALUE_KEYS))}")
    for kind, req in EVENT_VALUE_KEYS.items():
        clash = req & EVENT_OPTIONAL_KEYS[kind]
        assert not clash, f"{kind.name} declares {sorted(clash)} as BOTH required and optional"


def test_the_undeclared_key_guard_FAILS_on_a_planted_key():
    """The guard above passes trivially if `declared_value_keys` is over-permissive — plant an
    undeclared key and prove it is caught. (The control the sweep's rules ask for: a check that
    has never been seen to fail is a check nobody has verified.)"""
    from agents.battle.battle_event import BattleEvent, undeclared_value_keys
    clean = BattleEvent(seq=1, turn=1, kind=EventKind.MOVE, side="ours",
                        actor_species="tyranitar", value={"move_id": "rockslide"})
    assert undeclared_value_keys(clean) == frozenset()
    planted = BattleEvent(seq=2, turn=1, kind=EventKind.MOVE, side="ours",
                          actor_species="tyranitar",
                          value={"move_id": "rockslide", "from": "sleeptalk"})
    assert undeclared_value_keys(planted) == frozenset({"from"}), (
        "MOVE does not declare a `from` key (delegation rides `from_move`) — the guard must "
        "catch it, or the whole schema is decorative")


def test_event_log_fabricates_nothing_and_preserves_order():
    """The event log invents nothing and reorders nothing: every event's ``raw`` is a
    real line from the protocol archive, and the events appear in archive order.

    This is the true lossless property. Note it is NOT a 1:1 line↔event mapping — a
    single ``|move|…|[miss]`` line legitimately yields two events (the MOVE plus a
    synthetic MISS) so the legacy suffix format is captured. So we assert the event
    raws are an order-preserving SUBSEQUENCE of the archive, never a fabrication.
    Combined with conservation (every line is bucketed) this proves nothing
    battle-relevant is dropped."""
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    archive = [tuple(line) for line in g3._replay_data]
    # walk the archive once, matching each event's raw in order
    it = iter(archive)
    for e in g3.events:
        found = False
        for line in it:
            if line == e.raw:
                found = True
                break
        assert found, f"event {e.kind.name} raw {e.raw} not found in archive order"


def test_synthetic_events_are_only_move_suffix_outcomes():
    """The ONLY events whose raw keyword doesn't match their kind are the synthetic
    move-suffix outcomes (legacy ``[miss]``/``[notarget]`` on a ``|move|`` line). Any
    other kind/raw mismatch is a real bug (an event built from the wrong line)."""
    from agents.battle.battle_event import EVENT_KIND
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    for e in g3.events:
        if EVENT_KIND.get(e.raw[1]) is not e.kind:
            assert e.raw[1] == "move" and e.kind in (EventKind.MISS, EventKind.FAIL), (
                f"unexpected kind/raw mismatch: {e.kind.name} from {e.raw!r}"
            )


@pytest.mark.parametrize("trailing", [
    [],                                                    # [miss] last — the common case
    ["[anim] Hydro Pump p1a: Skarm"],                      # [miss] then [anim] — battle-3736 turn 7
    ["[spread]"],                                          # [miss] then [spread]
    ["[spread]", "[anim] Hydro Pump p1a: Skarm"],          # [miss] then [spread] then [anim]
])
def test_move_miss_suffix_stranded_after_trailing_tags_parses(trailing):
    """REGRESSION (bridge [miss] parse crash, battle-3736 turn 7). A |move| line whose [miss] suffix is
    FOLLOWED by a later cosmetic tag ([anim]/[spread]) must still parse. The early single-position [miss]
    strip inspects event[-1] only ONCE — BEFORE [anim]/[spread] are removed — so the [miss] was left
    stranded and the len(event) dispatch raised 'Unhandled move message format'. The robust final pass
    strips any trailing failure/cosmetic suffix in ANY order; assert it parses AND still records the miss."""
    for cls in (Battle, Gen3Battle):
        b = make(cls)
        feed(b, [
            ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
            ["", "switch", "p2a: Cune", "Suicune, L100", "100/100"],
            ["", "turn", "1"],
        ])
        # Must NOT raise regardless of where [miss] sits among the trailing tags.
        b.parse_message(["", "move", "p2a: Cune", "Hydro Pump", "p1a: Skarm", "[miss]", *trailing])
        assert b.opponent_active_pokemon.species == "suicune"      # state still advanced
    # Gen3Battle records the miss as a synthetic MISS event regardless of the trailing-tag order.
    g3 = make(Gen3Battle)
    feed(g3, [
        ["", "switch", "p1a: Skarm", "Skarmory, L100, F", "100/100"],
        ["", "switch", "p2a: Cune", "Suicune, L100", "100/100"],
        ["", "turn", "1"],
    ])
    g3.parse_message(["", "move", "p2a: Cune", "Hydro Pump", "p1a: Skarm", "[miss]", *trailing])
    assert any(e.kind is EventKind.MISS for e in g3.events), "the miss outcome was lost"


def test_every_event_kind_has_a_schema_entry():
    """No EventKind may ship without declaring its required payload keys (even if the
    set is empty) — forces a deliberate decision for each new kind."""
    for kw, (policy, _r) in MESSAGE_POLICY.items():
        if policy is Policy.EVENT:
            from agents.battle.battle_event import EVENT_KIND
            assert EVENT_KIND[kw] in EVENT_VALUE_KEYS, (
                f"{kw!r} -> {EVENT_KIND[kw].name} has no EVENT_VALUE_KEYS entry"
            )


# --------------------------------------------------------------------------- #
# Event content                                                                #
# --------------------------------------------------------------------------- #
def test_events_emitted_for_each_action():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    kinds = [e.kind for e in g3.events]
    assert EventKind.MOVE in kinds
    assert EventKind.SWITCH in kinds
    assert EventKind.DRAG in kinds
    assert EventKind.DAMAGE in kinds
    assert EventKind.STATUS in kinds
    assert EventKind.SUPEREFFECTIVE in kinds
    assert EventKind.IMMUNE in kinds
    assert EventKind.CRIT in kinds


def test_crit_attributed_to_resolving_mover():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    crit = next(e for e in g3.events if e.kind is EventKind.CRIT)
    # Tyranitar's Rock Slide landed the crit on our Zapdos.
    assert crit.side == "opp"
    assert crit.actor_species == "tyranitar"
    assert crit.target_species == "zapdos"


def test_immune_effectiveness_attributed_to_attacker():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    imm = next(e for e in g3.events if e.kind is EventKind.IMMUNE)
    assert imm.side == "opp"  # Tyranitar's Earthquake
    assert imm.value["multiplier"] == 0.0


def test_an_ability_blocked_fail_carries_its_from_cause_and_never_marks_the_move():
    """`|-fail|p2a: Tyra|unboost|[from] ability: Clear Body` — Intimidate blocked on a
    switch-in — is not the mover's move failing, and the consumer can only know that if the
    producer carries the `[from]` cause. A bare `-fail` (a genuine move fail) stays
    clause-free and still marks the move."""
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    # turn 5: Tyranitar Crunches (it lands), then Intimidate fizzles on Clear Body —
    # pre-guard, that fizzle marked Crunch as `failed` while it had dealt full damage.
    g3.parse_message(["", "move", "p2a: Tyra", "Crunch", "p1a: Skarm"])
    g3.parse_message(["", "-damage", "p1a: Skarm", "40/100"])
    g3.parse_message(["", "-fail", "p2a: Tyra", "unboost",
                      "[from] ability: Clear Body", "[of] p2a: Tyra"])
    g3.parse_message(["", "turn", "6"])
    # turn 6: a genuine bare fail on the mover's own move must still mark it.
    g3.parse_message(["", "move", "p2a: Tyra", "Crunch", "p1a: Skarm"])
    g3.parse_message(["", "-fail", "p2a: Tyra"])
    g3.parse_message(["", "turn", "7"])
    fails = [e for e in g3.events if e.kind is EventKind.FAIL]
    assert fails[-2].from_clause == "ability: Clear Body"
    assert fails[-1].from_clause is None
    assert TurnView.for_turn(g3, 5).opp.failed is False   # Clear Body's fizzle ≠ Crunch failing
    assert TurnView.for_turn(g3, 6).opp.failed is True    # a bare fail still counts


def test_turn_view_over_real_parse():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    v1 = TurnView.for_turn(g3, 1)
    assert v1.we_moved_first is True
    assert v1.ours.move_id == "thunderbolt"
    assert v1.opp.move_id == "rockslide"
    assert v1.opp.crit is True
    assert v1.opp.effectiveness == 2.0

    v2 = TurnView.for_turn(g3, 2)
    assert v2.ours.switched is True
    assert v2.opp.effectiveness == 0.0  # Earthquake immune into Skarmory

    v4 = TurnView.for_turn(g3, 4)
    assert v4.opp.drag is True
    assert v4.opp.switched_to == "blissey"


def test_events_for_turn_slices_correctly():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL)
    # Turn 0 holds the lead switch-ins; turn 1 the first action pair.
    assert all(e.kind in (EventKind.SWITCH,) for e in g3.events_for_turn(0))
    assert {e.turn for e in g3.events_for_turn(1)} == {1}
    assert {e.turn for e in g3.events_for_turn(3)} == {3}


# --------------------------------------------------------------------------- #
# Tripwires: non-gen3 / unknown keywords raise (design §4.2).                   #
# --------------------------------------------------------------------------- #
def test_unsupported_keyword_raises():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL[:11])  # up to and including turn 1 marker
    with pytest.raises(UnsupportedMessageType):
        g3.parse_message(["", "-terastallize", "p1a: Zappy", "Electric"])


def test_unknown_keyword_raises():
    g3 = make(Gen3Battle)
    feed(g3, CANONICAL[:11])
    with pytest.raises(UnknownMessageType):
        g3.parse_message(["", "totallyboguskeyword", "p1a: Zappy"])


# --------------------------------------------------------------------------- #
# Ability reveal on -activate: an ability activating discloses it persistently. #
# --------------------------------------------------------------------------- #
def _setup_with_opp(cls):
    """Battle with p1=us, p2=opp Snorlax active."""
    b = make(cls)
    feed(b, [
        ["", "player", "p1", "p1user", "", ""],
        ["", "player", "p2", "p2user", "", ""],
        ["", "teamsize", "p1", "6"],
        ["", "teamsize", "p2", "6"],
        ["", "gametype", "singles"],
        ["", "gen", "3"],
        ["", "start"],
        ["", "switch", "p1a: Zappy", "Zapdos, L100", "100/100"],
        ["", "switch", "p2a: Snorlax", "Snorlax, L100, M", "100/100"],
        ["", "turn", "1"],
    ])
    return b


def test_ability_activation_reveals_opponent_ability():
    """|-activate|opp|ability: Immunity reveals the opponent's ability (Immunity has
    two possible abilities for Snorlax — Immunity / Thick Fat — so the activation
    resolves the ambiguity). Previously this was dropped: only a transient volatile
    fired, the persistent ability stayed unknown."""
    g3 = _setup_with_opp(Gen3Battle)
    opp = g3.opponent_active_pokemon
    assert opp.ability is None, "ability should start unknown"
    g3.parse_message(["", "-activate", "p2a: Snorlax", "ability: Immunity"])
    assert opp.ability == "immunity", "ability not revealed by -activate"


def test_ability_reveal_matches_classic_battle():
    """The reveal lives in the base AbstractBattle handler, so classic Battle and
    Gen3Battle agree (state-equivalence preserved)."""
    classic = _setup_with_opp(Battle)
    g3 = _setup_with_opp(Gen3Battle)
    line = ["", "-activate", "p2a: Snorlax", "ability: Immunity"]
    classic.parse_message(line)
    g3.parse_message(line)
    assert (
        g3.opponent_active_pokemon.ability
        == classic.opponent_active_pokemon.ability
        == "immunity"
    )


def test_known_ability_not_clobbered_by_activation():
    """When the ability is already known, a later activation must NOT overwrite it
    into temporary_ability (which would mask the real ability)."""
    g3 = _setup_with_opp(Gen3Battle)
    opp = g3.opponent_active_pokemon
    g3.parse_message(["", "-activate", "p2a: Snorlax", "ability: Immunity"])
    assert opp.ability == "immunity"
    # A second activation is a no-op (ability already known, no temp override)
    g3.parse_message(["", "-activate", "p2a: Snorlax", "ability: Immunity"])
    assert opp.ability == "immunity"
    assert opp.temporary_ability is None
