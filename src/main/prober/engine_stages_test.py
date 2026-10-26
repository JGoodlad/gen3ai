"""The RESULT timeline says what the LOG says about stat stages and statuses (2026-10-26).

The `turns` / `analyze` timeline never showed a stat stage on a current run (0 of 92 `-boost` /
`-unboost` lines over 14 battles) because its only source was the recorder's TurnDelta boost block,
which every current run has lost; and it dropped a damaging move's secondary status (71 of 162 over 40
battles of two runs) because `timeline_entry_text` chose the damage clause OR the status clause. Both
fixes live in the engine (`engine/protocol.py`, `engine/timeline.py`), so `turns` and `analyze` change
together; the surface cases at the end pin that.

Every fixture is a protocol slice copied VERBATIM from a CURRENT run's trace (`models/` is read-only and
gitignored, so the lines are pinned here), with the decision's recorded outcome and the trace's recorded
trainee seat (`meta.trace_source.trainee_side`). Each assertion is on text the previous code could not
produce, so each fails with its fix reverted.
"""

import io
import json
import os
import re
from collections import namedtuple
from contextlib import redirect_stdout

import numpy as np
import pytest

from main.prober.engine import (
    BOOST_TAGS_GEN3, BOOST_TAGS_LATER_GEN, TimelineContradiction, _timeline_for, analyze_invocation,
    build_result_timeline, protocol_action_fate, protocol_boost_changes, protocol_move_effects,
    protocol_move_result, timeline_entry_text, trainee_side_of, verify_timeline_against_protocol,
    move_order_from_protocol,
)
from utils.paths import repo_path

_Fx = namedtuple("_Fx", "seat our opp phase outcome lines")


def _timeline(fx, *, seat="recorded"):
    """The decision's timeline exactly as `battle_turns` / `analyze` build it, as plain-text lines.
    ``seat="recorded"`` passes the trace's recorded seat; ``None`` is an older trace that records none."""
    inv = {"our": {"species": fx.our}, "opp": {"species": fx.opp}, "phase": fx.phase}
    entries = _timeline_for(inv, None, fx.outcome, protocol=fx.lines,
                            our_side=fx.seat if seat == "recorded" else seat)
    return [timeline_entry_text(e) for e in entries], entries


def _reader(fx):
    return protocol_boost_changes(fx.lines, fx.our, fx.opp, our_side=fx.seat)


# rb_x35_n1p_s3501 step_300000095/staller/loss_003 turn 16 (decision 19, trainee seat p1): Salamence mirror: Dragon Dance (+1 atk +1 spe) after the opponent's Dragon Claw; the opponent moved FIRST.
_DRAGON_DANCE_MIRROR = _Fx(
    seat='p1', our='salamence', opp='salamence', phase='move_selection',
    outcome={"our": {"action": "dragondance", "hp_delta": "-81%"}, "opp": {"action": "dragonclaw", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|16',
        '|',
        '|move|p2a: Salamence|Dragon Claw|p1a: Salamence',
        '|-supereffective|p1a: Salamence',
        '|-damage|p1a: Salamence|64/331',
        '|move|p1a: Salamence|Dragon Dance|p1a: Salamence',
        '|-boost|p1a: Salamence|atk|1',
        '|-boost|p1a: Salamence|spe|1',
        '|',
        '|-weather|Sandstorm|[upkeep]',
        '|-damage|p1a: Salamence|44/331|[from] Sandstorm',
        '|-damage|p2a: Salamence|94/100|[from] Sandstorm',
        '|-heal|p1a: Salamence|64/331|[from] item: Leftovers',
        '|-heal|p2a: Salamence|100/100|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_296000104/random/win_1001 turn 1 (decision 0, trainee seat p1): the opponent switches in an Intimidate Gyarados; OUR Swampert (not the mover's target) loses a stage, before our move.
_INTIMIDATE_ON_ARRIVAL = _Fx(
    seat='p1', our='swampert', opp='jirachi', phase='move_selection',
    outcome={"our": {"action": "earthquake", "hp_delta": "+0%"}, "opp": {"action": "switched_to:gyarados", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|1',
        '|',
        '|switch|p2a: Gyarados|Gyarados, F|100/100',
        '|-ability|p2a: Gyarados|Intimidate|boost',
        '|-unboost|p1a: Swampert|atk|1',
        '|move|p1a: Swampert|Earthquake|p2a: Gyarados',
        '|-immune|p2a: Gyarados',
        '|',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_296000104/heuristic2/win_1001 turn 15 (decision 15, trainee seat p1): our Psychic's secondary effect drops THEIR Sp. Def; the drop sits inside our move's window.
_PSYCHIC_SPD_DROP = _Fx(
    seat='p1', our='celebi', opp='skarmory', phase='move_selection',
    outcome={"our": {"action": "psychic", "hp_delta": "-27%"}, "opp": {"action": "drillpeck", "hp_delta": "-25%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|15',
        '|',
        '|move|p1a: Celebi|Psychic|p2a: Skarmory',
        '|-resisted|p2a: Skarmory',
        '|-damage|p2a: Skarmory|70/100',
        '|-unboost|p2a: Skarmory|spd|1',
        '|move|p2a: Skarmory|Drill Peck|p1a: Celebi',
        '|-supereffective|p1a: Celebi',
        '|-damage|p1a: Celebi|237/404',
        '|',
        '|-weather|Sandstorm|[upkeep]',
        '|-damage|p1a: Celebi|212/404|[from] Sandstorm',
        '|-heal|p1a: Celebi|237/404|[from] item: Leftovers',
        '|-heal|p2a: Skarmory|76/100|[from] item: Leftovers',
        '|-damage|p2a: Skarmory|63/100|[from] Leech Seed|[of] p1a: Celebi',
        '|-heal|p1a: Celebi|278/404|[silent]',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_296000104/heuristic/win_1001 turn 2 (decision 1, trainee seat p1): a Salac Berry fires in the residual phase (after the blank), `[from] item:`.
_SALAC_BERRY = _Fx(
    seat='p1', our='swampert', opp='tyranitar', phase='move_selection',
    outcome={"our": {"action": "hydropump", "hp_delta": "-46%"}, "opp": {"action": "crunch", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|2',
        '|',
        '|move|p1a: Swampert|Hydro Pump|p2a: Tyranitar|[miss]',
        '|-miss|p1a: Swampert|p2a: Tyranitar',
        '|move|p2a: Tyranitar|Crunch|p1a: Swampert',
        '|-damage|p1a: Swampert|65/341',
        '|',
        '|-weather|Sandstorm|[upkeep]',
        '|-enditem|p1a: Swampert|Salac Berry|[eat]',
        '|-boost|p1a: Swampert|spe|1|[from] item: Salac Berry',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_296000104/sentinel_0/loss_1004 turn 19 (decision 21, trainee seat p1): Belly Drum: `-setboost`, the stage it SET.
_BELLY_DRUM = _Fx(
    seat='p1', our='charizard', opp='blissey', phase='move_selection',
    outcome={"our": {"action": "bellydrum", "hp_delta": "-50%"}, "opp": {"action": "switched_to:suicune", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|19',
        '|',
        '|-curestatus|p2a: Blissey|brn|[from] ability: Natural Cure|[silent]',
        '|switch|p2a: Suicune|Suicune|100/100',
        '|move|p1a: Charizard|Belly Drum|p1a: Charizard',
        '|-damage|p1a: Charizard|159/318',
        '|-setboost|p1a: Charizard|atk|6|[from] move: Belly Drum',
        '|',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_296000104/heuristic/win_002 turn 13 (decision 12, trainee seat p1): Curse: -1 spe, +1 atk, +1 def in one window, spe first.
_CURSE = _Fx(
    seat='p1', our='camerupt', opp='swampert', phase='move_selection',
    outcome={"our": {"action": "switched_to:suicune", "hp_delta": "+0%"}, "opp": {"action": "curse", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|13',
        '|',
        '|switch|p1a: Suicune|Suicune|403/403',
        '|-ability|p1a: Suicune|Pressure|[silent]',
        '|move|p2a: Swampert|Curse|p2a: Swampert',
        '|-unboost|p2a: Swampert|spe|1',
        '|-boost|p2a: Swampert|atk|1',
        '|-boost|p2a: Swampert|def|1',
        '|',
        '|-weather|Sandstorm|[upkeep]',
        '|-damage|p1a: Suicune|378/403|[from] Sandstorm',
        '|-heal|p1a: Suicune|403/403|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_298000112/sentinel_1/loss_006 turn 25 (decision 29, trainee seat p1): our Charm into a Jirachi already at -6 atk: `-unboost|…|atk|0`; their Calm Mind.
_CAPPED_CHARM = _Fx(
    seat='p1', our='umbreon', opp='jirachi', phase='move_selection',
    outcome={"our": {"action": "charm", "hp_delta": "+0%"}, "opp": {"action": "calmmind", "hp_delta": "+6%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|25',
        '|',
        '|move|p2a: Jirachi|Calm Mind|p2a: Jirachi',
        '|-boost|p2a: Jirachi|spa|1',
        '|-boost|p2a: Jirachi|spd|1',
        '|move|p1a: Umbreon|Charm|p2a: Jirachi',
        '|-unboost|p2a: Jirachi|atk|0',
        '|',
        '|-heal|p2a: Jirachi|64/100|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_300000095/sentinel_0/win_001 turn 11 (decision 10, trainee seat p1): Suicune mirror: their Calm Mind, then our Roar drags a Blissey in.
_SUICUNE_MIRROR = _Fx(
    seat='p1', our='suicune', opp='suicune', phase='move_selection',
    outcome={"our": {"action": "roar", "hp_delta": "+6%"}, "opp": {"action": "calmmind → phazed_to:blissey", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": []},
    lines=(
        '|turn|11',
        '|',
        '|move|p2a: Suicune|Calm Mind|p2a: Suicune',
        '|-boost|p2a: Suicune|spa|1',
        '|-boost|p2a: Suicune|spd|1',
        '|move|p1a: Suicune|Roar|p2a: Suicune',
        '|drag|p2a: Blissey|Blissey, F|100/100',
        '|',
        '|-heal|p1a: Suicune|328/342|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_296000104/heuristic/loss_2001 turn 9 (decision 8, trainee seat p1): Thunderbolt paralyzes a Synchronize Umbreon; the ability relays PAR back.
_SYNCHRONIZE = _Fx(
    seat='p1', our='umbreon', opp='gengar', phase='move_selection',
    outcome={"our": {"action": "pursuit", "hp_delta": "-10%"}, "opp": {"action": "thunderbolt", "hp_delta": "-19%"}, "reward": {"total": 0.0}, "events": ["our:umbreon:PAR", "opp:gengar:PAR"]},
    lines=(
        '|turn|9',
        '|',
        '|move|p2a: Gengar|Thunderbolt|p1a: Umbreon',
        '|-damage|p1a: Umbreon|31/394',
        '|-status|p1a: Umbreon|par',
        '|-status|p2a: Gengar|par|[from] ability: Synchronize|[of] p1a: Umbreon',
        '|move|p1a: Umbreon|Pursuit|p2a: Gengar',
        '|-supereffective|p2a: Gengar',
        '|-damage|p2a: Gengar|15/100 par',
        '|',
        '|-heal|p2a: Gengar|21/100 par|[from] item: Leftovers',
        '|-heal|p1a: Umbreon|55/394 par|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_n1p_s3501 step_300000095/random/win_3001 turn 7 (decision 7, trainee seat p1): Body Slam's secondary paralysis, beside its damage.
_BODY_SLAM_PAR = _Fx(
    seat='p1', our='snorlax', opp='skarmory', phase='move_selection',
    outcome={"our": {"action": "bodyslam", "hp_delta": "+0%"}, "opp": {"action": "spikes", "hp_delta": "-10%"}, "reward": {"total": 0.0}, "events": ["opp:skarmory:PAR"]},
    lines=(
        '|turn|7',
        '|',
        '|move|p2a: Skarmory|Spikes|p1a: Snorlax',
        '|-sidestart|p1: rhone|Spikes',
        '|move|p1a: Snorlax|Body Slam|p2a: Skarmory',
        '|-resisted|p2a: Skarmory',
        '|-damage|p2a: Skarmory|24/100',
        '|-status|p2a: Skarmory|par',
        '|',
        '|-heal|p2a: Skarmory|30/100 par|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_d3mut_s3502 step_300000084/staller_v2/win_1001 turn 2 (decision 2, trainee seat p1): the opponent's nickname is mojibake (`MagnÃ©ton` for Magneton): no nickname match exists.
_MOJIBAKE_NICKNAME = _Fx(
    seat='p1', our='skarmory', opp='magneton', phase='move_selection',
    outcome={"our": {"action": "spikes", "hp_delta": "+6%"}, "opp": {"action": "thunderwave", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": ["our:skarmory:PAR"]},
    lines=(
        '|turn|2',
        '|',
        "|move|p2a: Magn\xc3\xa9ton|Thunder Wave|p1a: Skarmory",
        '|-status|p1a: Skarmory|par',
        "|move|p1a: Skarmory|Spikes|p2a: Magn\xc3\xa9ton",
        '|-sidestart|p2: rhtwo|Spikes',
        '|',
        '|-heal|p1a: Skarmory|165/331 par|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_d3mut_s3502 step_300000084/staller/loss_003 turn 70 (decision 75, trainee seat p1): our Rest (sleep) after their Thunderbolt: the recorded `our:zapdos:SLP` event is OUR Rest's.
_REST_AFTER_THEIR_THUNDERBOLT = _Fx(
    seat='p1', our='zapdos', opp='jolteon', phase='move_selection',
    outcome={"our": {"action": "rest", "hp_delta": "+54%"}, "opp": {"action": "thunderbolt", "hp_delta": "+0%"}, "reward": {"total": 0.0}, "events": ["our:zapdos:SLP"]},
    lines=(
        '|turn|70',
        '|',
        '|move|p2a: Jolteon|Thunderbolt|p1a: Zapdos',
        '|-damage|p1a: Zapdos|52/384 frz',
        '|-curestatus|p1a: Zapdos|frz|[msg]',
        '|move|p1a: Zapdos|Rest|p1a: Zapdos',
        '|-status|p1a: Zapdos|slp|[from] move: Rest',
        '|-heal|p1a: Zapdos|384/384 slp|[silent]',
        '|',
        '|-weather|Sandstorm|[upkeep]',
        '|-damage|p2a: Jolteon|82/100|[from] Sandstorm',
        '|-damage|p1a: Zapdos|360/384 slp|[from] Sandstorm',
        '|-heal|p2a: Jolteon|88/100|[from] item: Leftovers',
        '|-heal|p1a: Zapdos|384/384 slp|[from] item: Leftovers',
        '|upkeep',
    ))

# rb_x35_n1p_s3502 step_292000064/staller/win_3001 turn 8 (decision 8, trainee seat p1): their Rest (sleep) after our Body Slam: the recorded `opp:suicune:SLP` event is THEIR Rest's.
_REST_AFTER_OUR_BODY_SLAM = _Fx(
    seat='p1', our='snorlax', opp='suicune', phase='move_selection',
    outcome={"our": {"action": "bodyslam", "hp_delta": "+6%"}, "opp": {"action": "rest", "hp_delta": "+64%"}, "reward": {"total": 0.0}, "events": ["opp:suicune:SLP"]},
    lines=(
        '|turn|8',
        '|',
        '|move|p1a: Snorlax|Body Slam|p2a: Suicune',
        '|-damage|p2a: Suicune|12/100 par',
        '|move|p2a: Suicune|Rest|p2a: Suicune',
        '|-status|p2a: Suicune|slp|[from] move: Rest',
        '|-heal|p2a: Suicune|100/100 slp|[silent]',
        '|',
        '|-heal|p1a: Snorlax|304/477|[from] item: Leftovers',
        '|upkeep',
    ))

# ── the vocabulary, pinned against Showdown's protocol and the Rust port's emitter ───────────────────

def test_the_boost_vocabulary_is_every_stage_message_showdowns_protocol_lists():
    """`deps/pokemon-showdown/sim/SIM-PROTOCOL.md` lists the messages that move a stat stage. A reader
    that folds fewer than the protocol defines lets a stage change vanish from the timeline, so the two
    sets are EQUAL: a message added upstream fails here instead of silently reading as nothing."""
    doc = repo_path("deps/pokemon-showdown/sim/SIM-PROTOCOL.md").read_text()
    listed = set(re.findall(r"^`\|(-[a-z]*boost)\b", doc, flags=re.M))
    assert {"-boost", "-unboost", "-setboost", "-swapboost", "-invertboost", "-clearboost",
            "-clearallboost", "-clearpositiveboost", "-clearnegativeboost", "-copyboost"} == listed
    assert BOOST_TAGS_GEN3 | BOOST_TAGS_LATER_GEN == listed
    assert not BOOST_TAGS_GEN3 & BOOST_TAGS_LATER_GEN


def test_the_gen3_boost_messages_are_exactly_what_the_rust_port_emits():
    """The port writes the protocol every current trace carries (`protocol.rs`, one doc line per
    emitter): the messages it emits are `BOOST_TAGS_GEN3`, no more (a later-gen message the port never
    writes is a note, not a promise) and no fewer."""
    src = repo_path("src/rust_sim/src/protocol.rs").read_text()
    emitted = set(re.findall(r"^\s*///.*?`\|(-[a-z]*boost)\b", src, flags=re.M))
    emitted |= set(re.findall(r"^\s*///.*?or `\|(-[a-z]*boost)\b", src, flags=re.M))
    assert emitted == set(BOOST_TAGS_GEN3), emitted


def test_the_trainee_seat_is_read_only_where_a_trace_records_it():
    assert trainee_side_of({"meta": {"trace_source": {"trainee_side": "p2"}}}) == "p2"
    for summary in ({}, None, {"meta": {}}, {"meta": {"trace_source": {}}},
                    {"meta": {"trace_source": {"trainee_side": "p3"}}}):
        assert trainee_side_of(summary) is None


# ── stat stages ───────────────────────────────────────────────────────────────────────────────────────

def test_a_self_boost_reads_off_the_log_and_the_seat_orders_a_mirror():
    """THE REPORTED GAP: `we dragondance` said nothing of the +1 Atk +1 Spe. A Salamence MIRROR is the
    case a nickname cannot place at all, and the opponent moved first (the seat reads that too)."""
    lines, _ = _timeline(_DRAGON_DANCE_MIRROR)
    assert _reader(_DRAGON_DANCE_MIRROR) == {"side": {"we": "atk+1 spe+1"}, "caused": {}}
    assert lines[0].startswith("opp dragonclaw") and lines[1] == "we dragondance  ·  atk+1 spe+1"


def test_an_intimidate_on_arrival_is_the_victims_stage_not_the_movers():
    """The unboost lands on OUR Swampert right after THEIR `-ability … Intimidate`, before our move:
    it is ours (side level), and nothing is `caused` by our Earthquake."""
    lines, _ = _timeline(_INTIMIDATE_ON_ARRIVAL)
    assert _reader(_INTIMIDATE_ON_ARRIVAL) == {"side": {"we": "atk-1"}, "caused": {}}
    assert lines == ["opp switch jirachi → gyarados", "we earthquake — no effect (immune)  ·  atk-1"]


def test_a_secondary_drop_is_said_on_the_move_that_caused_it():
    """Psychic's Sp. Def drop is inside OUR move's window, on THEIR mon. Said as a side annotation it
    read `opp thunderwave — no effect · spd-1` on a real turn (their line, not the cause)."""
    lines, entries = _timeline(_PSYCHIC_SPD_DROP)
    assert _reader(_PSYCHIC_SPD_DROP) == {"side": {}, "caused": {"we": "spd-1"}}
    assert lines[0].startswith("we psychic did") and lines[0].endswith("+  skarmory spd-1")
    assert not any("spd-1" in ln for ln in lines[1:]) and not any(e.get("boost") for e in entries)


def test_a_berry_in_the_residual_phase_is_its_holders_own_boost():
    """After the blank line, `[from] item:` — nobody's move caused it, and it must not be billed to
    the Tyranitar whose Crunch dropped the holder under 25%."""
    lines, _ = _timeline(_SALAC_BERRY)
    assert _reader(_SALAC_BERRY) == {"side": {"we": "spe+1"}, "caused": {}}
    assert lines[0] == "we hydropump — missed  ·  spe+1" and "spe+1" not in lines[1]


def test_belly_drum_is_the_stage_it_set_never_a_delta():
    """A slice never says where a stage started, so `-setboost|…|atk|6` is `atk=+6`, not `atk+6`."""
    lines, _ = _timeline(_BELLY_DRUM)
    assert _reader(_BELLY_DRUM) == {"side": {"we": "atk=+6"}, "caused": {}}
    assert lines[-1] == "we bellydrum  ·  atk=+6"


def test_curse_nets_its_three_stages_in_the_recorders_order():
    """`-unboost spe` arrives FIRST in the log; the text is the recorder's atk, def, … spe order."""
    lines, _ = _timeline(_CURSE)
    assert _reader(_CURSE) == {"side": {"opp": "atk+1 def+1 spe-1"}, "caused": {}}
    assert "opp curse  ·  atk+1 def+1 spe-1" in lines


def test_a_zero_amount_line_is_a_limit_not_a_change():
    """`-unboost|…|atk|0` is the sim's report that Charm hit a stage already at -6. It is said, on the
    move that tried it, and it is not a `+0`."""
    lines, _ = _timeline(_CAPPED_CHARM)
    assert _reader(_CAPPED_CHARM) == {"side": {"opp": "spa+1 spd+1"},
                                      "caused": {"we": "atk at its limit"}}
    assert "we charm  +  jirachi atk at its limit" in lines and "opp calmmind  ·  spa+1 spd+1" in lines


def test_a_mirror_reads_by_the_recorded_seat_and_without_one_says_nothing():
    """Suicune against Suicune: both nicknames are the species. With the seat the Calm Mind is
    THEIRS; with no seat (an older trace) the sides cannot be told apart, so NOTHING is claimed — the
    reader returns None and the lines carry no stage, never a guessed side."""
    lines, _ = _timeline(_SUICUNE_MIRROR)
    assert lines[0].startswith("opp calmmind") and lines[0].endswith("·  spa+1 spd+1")
    assert protocol_boost_changes(_SUICUNE_MIRROR.lines, "suicune", "suicune") is None
    blind, _ = _timeline(_SUICUNE_MIRROR, seat=None)
    assert not any("spa+1" in ln for ln in blind)


def test_the_stage_reader_on_shapes_no_current_run_has_met():
    """SYNTHETIC (no current trace carries them): shaped from the Rust emitters' own doc lines
    (`protocol.rs`) and Showdown's `battle.boost()`. The reader is the unit under test, so the lines
    are hand-written; each case is one message of the vocabulary."""
    def read(*body):
        out = protocol_boost_changes(("|turn|3", "|move|p1a: Zapdos|Thunderbolt|p2a: Gengar", *body),
                                     "zapdos", "gengar")
        return out

    # +2 then -1 of the same stat NET (and a stat that nets to 0 vanishes)
    assert read("|-boost|p1a: Zapdos|atk|2", "|-unboost|p1a: Zapdos|atk|1")["side"] == {"we": "atk+1"}
    assert read("|-boost|p1a: Zapdos|atk|1", "|-unboost|p1a: Zapdos|atk|1")["side"] == {}
    # Haze: nobody is named, both sides are told; a stage raised AFTER it is absolute
    assert read("|-clearallboost")["side"] == {"we": "boosts cleared", "opp": "boosts cleared"}
    assert read("|-clearallboost", "|move|p2a: Gengar|Calm Mind|p2a: Gengar",
                "|-boost|p2a: Gengar|spa|2")["side"] == {
        "we": "boosts cleared", "opp": "boosts cleared; gengar spa=+2"}
    # Psych Up, White Herb (it restores what it can see dropped THIS turn)
    assert read("|-copyboost|p1a: Zapdos|p2a: Gengar|[from] move: Psych Up")["side"] == {
        "we": "copied gengar's boosts"}
    assert read("|-unboost|p1a: Zapdos|atk|1", "|-enditem|p1a: Zapdos|White Herb",
                "|-clearnegativeboost|p1a: Zapdos|[silent]")["side"] == {"we": "drops restored"}
    # a stage message of a LATER generation is a note, never silence
    assert read("|-swapboost|p1a: Zapdos|p2a: Gengar|atk, spa|[from] move: Power Swap")["side"] == {
        "we": "swapped boosts with gengar"}
    assert read("|-invertboost|p2a: Gengar|[from] move: Topsy-Turvy")["side"] == {
        "opp": "boosts inverted"}
    # an unreadable amount is said, not dropped
    assert read("|-boost|p1a: Zapdos|atk|many")["side"] == {"we": "stat change"}
    # a switch mid-turn puts two mons of one side in the text: they are labelled by SPECIES
    two = protocol_boost_changes(
        ("|turn|4", "|move|p1a: Zapdos|Baton Pass|p1a: Zapdos", "|-boost|p1a: Zapdos|spe|1", "|",
         "|switch|p1a: Leuphorie|Blissey, F|100/100", "|-boost|p1a: Leuphorie|def|1"),
        "zapdos", "gengar")
    assert two["side"] == {"we": "zapdos spe+1; blissey def+1"}
    # an Intimidate that fires INSIDE a move's window (a Roar dragging it in) is the dragged-in mon's,
    # not the roarer's: it stays on the side, never `caused`
    drag = protocol_boost_changes(
        ("|turn|5", "|move|p1a: Zapdos|Roar|p2a: Gengar", "|drag|p2a: Gyarados|Gyarados, F|100/100",
         "|-ability|p2a: Gyarados|Intimidate|boost", "|-unboost|p1a: Zapdos|atk|1"), "zapdos", "gengar")
    assert drag == {"side": {"we": "atk-1"}, "caused": {}}
    # no stage message ⇒ empty, whatever the nicknames; some but no anchor ⇒ None
    assert protocol_boost_changes(("|turn|1", "|move|p1a: Zapdos|Thunderbolt|p2a: Gengar"),
                                  "zapdos", "gengar") == {"side": {}, "caused": {}}
    assert protocol_boost_changes(("|turn|1", "|-boost|p1a: Zapdos|atk|1"), "zapdos", "gengar") is None
    assert protocol_boost_changes(None, "zapdos", "gengar") == {"side": {}, "caused": {}}


def test_a_stage_is_never_claimed_on_a_forced_replacement_decision():
    """A forced switch shares its turn's slice with the move decision before it, which already says
    the whole turn's stages; saying them again on the replacement would attach THIS turn's Dragon Dance
    to a `send_in` line."""
    inv = {"our": {"species": "salamence"}, "opp": {"species": "salamence"}, "phase": "forced_switch"}
    entries = _timeline_for(inv, None, {"our": {"action": "switched_to:blissey"}, "opp": {"action": "none"},
                                        "events": []},
                            protocol=_DRAGON_DANCE_MIRROR.lines, our_side="p1")
    assert [timeline_entry_text(e) for e in entries] == ["we send in blissey"]


def test_a_side_with_no_move_line_of_its_own_still_carries_its_stage():
    """The side that SWITCHED has no move entry to hang a stage on: its switch line carries it. A side
    with no entry at all gets a line of its own. A stage is never dropped for want of a place."""
    sw = build_result_timeline(
        {"our": {"action": "switched_to:snorlax", "hp_delta": "+0%"}, "opp": {"action": "none"},
         "events": []}, "zapdos", "gengar", "move_selection", our_boost_hint="atk-1")
    assert timeline_entry_text(next(e for e in sw if e["kind"] == "switch")).endswith("snorlax  ·  atk-1")
    only_boost = build_result_timeline(
        {"our": {"action": "none"}, "opp": {"action": "curse", "hp_delta": "+0%"}, "events": []},
        "zapdos", "gengar", "move_selection", our_boost_hint="atk-1")
    assert "we stat stages  ·  atk-1" in [timeline_entry_text(e) for e in only_boost]
    # and a stage a move PUT on the other mon, whose mover never executed (it fainted first), goes
    # to the other mon's own side rather than being lost
    handed = build_result_timeline(
        {"our": {"action": "none"}, "opp": {"action": "curse", "hp_delta": "+0%"}, "events": []},
        "zapdos", "gengar", "move_selection", our_boost_on_hint="spd-1")
    assert "opp curse  ·  spd-1" in [timeline_entry_text(e) for e in handed]


def test_a_recorded_boost_still_wins_over_the_log():
    """The recorder's TurnDelta boost (older runs) is kept: the protocol only fills what it did not
    decode, exactly as for a recorded fate."""
    entries = build_result_timeline(
        {"our": {"action": "swordsdance", "hp_delta": "+0%"}, "opp": {"action": "none"}, "events": [],
         "our_boost": "atk+2"}, "zapdos", "gengar", "move_selection", our_boost_hint="atk+9")
    assert timeline_entry_text(next(e for e in entries if e["kind"] == "move")) == "we swordsdance  ·  atk+2"


# ── statuses ──────────────────────────────────────────────────────────────────────────────────────────

def test_a_secondary_status_is_said_beside_the_damage_not_instead_of_it():
    """THE REPORTED GAP: `we bodyslam did 10% (skarmory …)` — the PAR the move applied was a structured
    field the text never printed."""
    lines, entries = _timeline(_BODY_SLAM_PAR)
    we = next(e for e in entries if e["side"] == "we" and e["kind"] == "move")
    assert we["status"] == "PAR" and we["status_target"] == "skarmory" and we["damage"]
    assert re.fullmatch(r"we bodyslam did 10%  \(skarmory .*\)  \+  skarmory PAR", lines[1])


def test_a_synchronize_relay_is_named_beside_the_status_it_relayed():
    lines, entries = _timeline(_SYNCHRONIZE)
    assert entries[0]["also"] == [{"target": "gengar", "status": "PAR", "via": "Synchronize"}]
    assert lines[0].endswith("+  umbreon PAR  +  gengar PAR (Synchronize)")
    eff = protocol_move_effects(_SYNCHRONIZE.lines, "gengar", "umbreon")
    assert [(n, c, self_t, via) for n, c, self_t, via in eff["statuses"]] == [
        ("umbreon", "PAR", False, ""), ("gengar", "PAR", True, "Synchronize")]
    assert eff["status"] == ("umbreon", "PAR", False)          # the first, as before


def test_a_rest_is_not_billed_to_the_other_sides_attack():
    """MEASURED: 3 of 18 status turns read `opp thunderbolt → zapdos SLP` over a Rest the ZAPDOS used
    (and `we bodyslam → suicune SLP` over THEIR Rest). The recorded event names a side and a species,
    never a cause, and Thunderbolt / Body Slam can paralyze, so the old gate let it through. A located
    window decides."""
    for fx, mover, rester in ((_REST_AFTER_THEIR_THUNDERBOLT, "opp thunderbolt", "we rest → zapdos SLP"),
                              (_REST_AFTER_OUR_BODY_SLAM, "we bodyslam", "opp rest → suicune SLP")):
        lines, entries = _timeline(fx)
        assert rester in lines, lines
        theirs = next(ln for ln in lines if ln.startswith(mover))
        assert "SLP" not in theirs, theirs
        assert fx.outcome["events"] and "SLP" in fx.outcome["events"][0]      # the event WAS recorded


def test_with_no_window_the_recorded_status_is_all_there_is():
    """An older trace (or a nickname that places no move line): no protocol, so the recorded event
    credits the move that could have caused it — the pre-existing behaviour, unchanged."""
    entries = build_result_timeline(
        {"our": {"action": "thunderwave", "hp_delta": "+0%"}, "opp": {"action": "none"},
         "events": ["opp:gengar:PAR"]}, "zapdos", "gengar", "move_selection")
    assert timeline_entry_text(next(e for e in entries if e["kind"] == "move")) == "we thunderwave → gengar PAR"


def test_the_recorders_status_counter_survives_when_the_log_confirms_the_status():
    """The log says `tox`; the recorder said `TOX(2)`. The counter is information the log lacks."""
    proto = ("|turn|6", "|move|p1a: Zapdos|Toxic|p2a: Gengar", "|-status|p2a: Gengar|tox", "|upkeep")
    entries = build_result_timeline(
        {"our": {"action": "toxic", "hp_delta": "+0%"}, "opp": {"action": "none"},
         "events": ["opp:gengar:TOX(2)"]}, "zapdos", "gengar", "move_selection",
        our_effects_hint=protocol_move_effects(proto, "zapdos", "gengar"))
    assert timeline_entry_text(next(e for e in entries if e["kind"] == "move")) == "we toxic → gengar TOX(2)"


# ── nicknames: the recorded seat finds a side the nickname cannot ───────────────────────────────────

def test_a_mojibake_nickname_is_found_by_its_seat():
    """`MagnÃ©ton` is not `magneton`, so the nickname reader called the opponent's Thunder Wave
    `absent` (it never moved) and the line was BLANKED before any status or damage was attributed —
    MEASURED over 12.5k side-turns of four current runs: 389 sides said `absent` that had moved, 886
    said nothing (mirrors). By seat the move, its order and its status are all there."""
    fx = _MOJIBAKE_NICKNAME
    assert protocol_action_fate(fx.lines, fx.opp, fx.our) == "absent"                       # the old reading
    assert protocol_action_fate(fx.lines, fx.opp, fx.our, actor_side="p2") == "moved"
    assert protocol_action_fate(fx.lines, fx.our, fx.opp, actor_side="p1") == "moved"
    assert protocol_move_result(fx.lines, fx.opp, fx.our, actor_side="p2") is None
    assert move_order_from_protocol(fx.lines, fx.our, fx.opp, our_side="p1") == "opp_first"
    assert protocol_move_effects(fx.lines, fx.opp, fx.our) is None                          # no nickname match
    assert protocol_move_effects(fx.lines, fx.opp, fx.our, actor_side="p2")["status"] == ("skarmory", "PAR", False)
    lines, _ = _timeline(fx)
    assert lines == ["opp thunderwave → skarmory PAR", "we spikes"]


def test_the_seat_readers_keep_the_nickname_path_when_no_seat_is_given():
    """An older trace records no seat; every reader behaves exactly as before for it."""
    proto = ("|turn|1", "|move|p1a: Zapdos|Thunderbolt|p2a: Gengar", "|-immune|p2a: Gengar", "|upkeep")
    assert protocol_move_result(proto, "zapdos", "gengar") == "immune"
    assert protocol_move_result(proto, "zapdos", "gengar", actor_side="p1") == "immune"
    assert protocol_move_result(proto, "gengar", "gengar") is None                          # a mirror
    assert protocol_action_fate(proto, "gengar", "zapdos") == "absent"
    assert protocol_action_fate(proto, "gengar", "zapdos", actor_side="p2") == "absent"
    assert move_order_from_protocol(proto, "zapdos", "gengar") == "we_first"
    assert move_order_from_protocol(proto, "zapdos", "gengar", our_side="p2") == "opp_first"


# ── the throwing guard ───────────────────────────────────────────────────────────────────────────────

def test_the_guard_calls_the_old_rendering_a_contradiction():
    """FAILS ON REVERT: a timeline built WITHOUT the log's stage and status evidence is exactly what
    the owner read, and the assertion form of the rule must refuse it."""
    fx = _SYNCHRONIZE
    inv = {"our": {"species": fx.our}, "opp": {"species": fx.opp}, "phase": fx.phase}
    good = _timeline_for(inv, None, fx.outcome, protocol=fx.lines, our_side=fx.seat)
    verify_timeline_against_protocol(good, fx.lines, fx.our, fx.opp, our_side=fx.seat)
    relay_dropped = [dict(e, also=[]) for e in good]
    with pytest.raises(TimelineContradiction, match="applied"):
        verify_timeline_against_protocol(relay_dropped, fx.lines, fx.our, fx.opp, our_side=fx.seat)

    for fx, why in ((_DRAGON_DANCE_MIRROR, "stat stages"), (_PSYCHIC_SPD_DROP, "stat-stage change")):
        inv = {"our": {"species": fx.our}, "opp": {"species": fx.opp}, "phase": fx.phase}
        good = _timeline_for(inv, None, fx.outcome, protocol=fx.lines, our_side=fx.seat)
        verify_timeline_against_protocol(good, fx.lines, fx.our, fx.opp, our_side=fx.seat)
        blind = build_result_timeline(fx.outcome, fx.our, fx.opp, fx.phase)      # no hints at all
        with pytest.raises(TimelineContradiction, match=why):
            verify_timeline_against_protocol(blind, fx.lines, fx.our, fx.opp, our_side=fx.seat)


# ── the two surfaces ─────────────────────────────────────────────────────────────────────────────────

_SURFACE_FIXTURES = (_INTIMIDATE_ON_ARRIVAL, _SALAC_BERRY, _BODY_SLAM_PAR, _PSYCHIC_SPD_DROP,
                     _DRAGON_DANCE_MIRROR)       # their own turns: 1, 2, 7, 15, 16, in order


def _surface_run(tmp_path, *, with_seat=True):
    """One battle whose replay.html is the five fixtures' slices back to back and whose decisions are
    theirs — the shape `battle_turns` reads on a current run (a `*_replay.html` sibling, or the core
    trace's stand-in for one)."""
    run = tmp_path / "run"
    bd = run / "eval_traces" / "step_1000" / "staller"
    os.makedirs(bd)
    lines = ["|player|p1|A|", "|player|p2|B|", "|start|"]
    invs = []
    for i, fx in enumerate(_SURFACE_FIXTURES):
        lines += list(fx.lines)
        turn = int(fx.lines[0].split("|")[2])
        invs.append({"i": i, "turn": turn, "phase": fx.phase, "chosen": fx.outcome["our"]["action"],
                     "our": {"species": fx.our, "hp": "100%"}, "opp": {"species": fx.opp, "hp": "100%"},
                     "actions": {fx.outcome["our"]["action"]: {"prob": "90.0%", "valid": True}},
                     "outcome": fx.outcome})
    with open(bd / "win_001_replay.html", "w", encoding="utf-8") as f:
        f.write('<!DOCTYPE html>\n<script type="text/plain" class="battle-log-data">\n'
                + "\n".join(lines) + "\n</script>\n")
    meta = {"step": 1000, "result": "WIN", "turns": 16, "invocations": len(invs)}
    if with_seat:
        meta["trace_source"] = {"trainee_side": "p1"}        # no `schema`: not a core trace, the html is read
    with open(bd / "win_001_summary.json", "w") as f:
        json.dump({"meta": meta, "invocations": invs}, f)
    np.savez(bd / "win_001_states.npz", obs=np.zeros((len(invs), 8), dtype=np.float32),
             has_state=np.ones(len(invs), dtype=np.int8),
             values=np.zeros(len(invs) + 1, dtype=np.float32))
    with open(run / "metadata.json", "w") as f:
        json.dump({"gamma": 0.95}, f)
    return str(run), str(bd / "win_001_summary.json")


def _turn_texts(turns_json):
    return {d["inv"]: [e["text"] for e in d["timeline"]]
            for t in turns_json["turns"] for d in t["decisions"]}


def test_turns_says_every_stage_and_status_the_log_records(tmp_path):
    from main.prober.session import ProbeSession

    run, summ = _surface_run(tmp_path)
    got = _turn_texts(ProbeSession(run).battle_turns(summ))
    assert "we earthquake — no effect (immune)  ·  atk-1" in got[0]                      # Intimidate
    assert any(ln.startswith("we hydropump") and ln.endswith("·  spe+1") for ln in got[1])  # the berry
    assert any(ln.endswith("+  skarmory PAR") for ln in got[2])                           # secondary status
    assert any(ln.endswith("+  skarmory spd-1") for ln in got[3])                         # secondary drop
    assert "we dragondance  ·  atk+1 spe+1" in got[4]                                     # a mirror, by seat


def test_the_cli_turns_command_prints_them(tmp_path):
    import main.prober.query as q

    run, summ = _surface_run(tmp_path)
    buf = io.StringIO()
    with redirect_stdout(buf):
        q.main(["turns", summ])
    got = _turn_texts(json.loads(buf.getvalue()))            # the JSON the agent reads, `\u00b7` and all
    assert "we dragondance  ·  atk+1 spe+1" in got[4] and any(ln.endswith("+  skarmory PAR") for ln in got[2])


def test_a_trace_with_no_recorded_seat_is_still_read_by_nickname(tmp_path):
    """The older-trace path: no `trainee_side`, so the nickname readers run. The Intimidate and the
    Body Slam are non-mirrors and read; the Salamence mirror (no anchor) says no stage rather than
    guess a side."""
    from main.prober.session import ProbeSession

    run, summ = _surface_run(tmp_path, with_seat=False)
    got = _turn_texts(ProbeSession(run).battle_turns(summ))
    assert "we earthquake — no effect (immune)  ·  atk-1" in got[0]
    assert any(ln.endswith("+  skarmory PAR") for ln in got[2])
    assert not any("atk+1" in ln for ln in got[4])


def test_analyze_says_them_too():
    """`analyze` builds the same timeline from the same engine function (`analyze_invocation(…,
    protocol=…)`); the summary's recorded seat reaches it with no session in between."""
    from main.prober.engine_current_run_test import _NO_BLOCKS, _FlatModel
    from main.prober.engine_test import _npz, _summary

    fx = _SYNCHRONIZE
    summary = _summary("thunderbolt")
    inv = summary["invocations"][0]
    inv.update(turn=9, phase="move_selection", our={"species": fx.our}, opp={"species": fx.opp})
    inv["outcome"] = fx.outcome
    summary["meta"]["trace_source"] = {"trainee_side": "p1"}
    a = analyze_invocation(_FlatModel(_NO_BLOCKS), summary, _npz(), 0, protocol=fx.lines)
    texts = [timeline_entry_text(e) for e in a.outcome["timeline"]]
    assert texts[0].endswith("+  umbreon PAR  +  gengar PAR (Synchronize)")

    fx = _DRAGON_DANCE_MIRROR
    summary = _summary("thunderbolt")        # the policy panel needs a label in the action table
    inv = summary["invocations"][0]
    inv.update(turn=16, phase="move_selection", our={"species": fx.our}, opp={"species": fx.opp})
    inv["outcome"] = fx.outcome
    summary["meta"]["trace_source"] = {"trainee_side": "p1"}
    a = analyze_invocation(_FlatModel(_NO_BLOCKS), summary, _npz(), 0, protocol=fx.lines)
    assert "we dragondance  ·  atk+1 spe+1" in [timeline_entry_text(e) for e in a.outcome["timeline"]]
