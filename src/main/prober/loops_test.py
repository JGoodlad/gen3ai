"""loops.py — the bait-loop detector, pinned on hand-written protocol lines.

Every scenario is a literal Showdown line list, because the whole point of this module is that it
reads the PROTOCOL rather than the prober's rendered timeline: a test built on a summary dict would
be testing the same abstraction the detector exists to bypass. The cases are the distinctions that
cost the gen-15 sweep a re-derivation each — a drag is not a pivot, a post-faint replacement is not
a pivot, a miss is not a whiff, a self-targeting move is not a bait, and a re-click is ordered
while a loop step is not.
"""
import pytest

from main.prober import loops as L

_HEAD = ["|player|p1|RLEval|", "|player|p2|Sentinel|", "|start|"]


def _lead(turn0_extra=()):
    return _HEAD + ["|switch|p1a: Claydol|Claydol|100/100",
                    "|switch|p2a: Snorlax|Snorlax|100/100", *turn0_extra]


def _battle(*turns):
    """`_battle(["|switch|…", …], …)` → the lead block plus one `|turn|N` block per argument."""
    out = _lead()
    for i, blk in enumerate(turns, start=1):
        out.append(f"|turn|{i}")
        out.extend(blk)
        out.append("|upkeep")
    return out


# --------------------------------------------------------------------- parsing


def test_hp_frac_reads_both_absolute_and_percentage_forms():
    assert L.hp_frac("224/404 par") == pytest.approx(224 / 404)
    assert L.hp_frac("90/100 frz") == pytest.approx(0.90)
    assert L.hp_frac("0 fnt") == 0.0
    assert L.hp_frac("") is None and L.hp_frac("weird") is None


def test_species_of_strips_the_details_field():
    assert L.species_of("Salamence, F") == "salamence"
    assert L.species_of("Mr. Mime") == "mrmime"


def test_split_turns_puts_the_leads_in_turn_zero():
    blocks = dict(L.split_turns(_battle(["|move|p1a: Claydol|Psychic|p2a: Snorlax"])))
    assert any("switch|p1a" in ln for ln in blocks[0]), "the leads belong to turn 0"
    assert any("|move|" in ln for ln in blocks[1])


def test_active_by_turn_is_the_board_the_decision_was_made_on():
    ev = L.parse_events(_battle(
        ["|switch|p2a: Salamence|Salamence, F|100/100"],
        ["|move|p1a: Claydol|Psychic|p2a: Salamence", "|-damage|p2a: Salamence|60/100"]))
    board = L.active_by_turn(ev)
    # turn 1's board is the LEADS — the switch happens during turn 1, after the decision.
    assert board[1] == {"p1": "claydol", "p2": "snorlax"}
    assert board[2]["p2"] == "salamence"


# ------------------------------------------------------------- side identification


def test_our_side_is_decided_by_the_recorded_board_not_by_p1():
    ev = L.parse_events(_battle(["|move|p1a: Claydol|Psychic|p2a: Snorlax"]))
    side, _ = L.identify_our_side(ev, {1: "claydol"})
    assert side == "p1"
    # the SAME protocol, with the trace saying we had Snorlax out: we are p2.
    side, _ = L.identify_our_side(ev, {1: "snorlax"})
    assert side == "p2"


def test_an_undecidable_side_returns_none_rather_than_guessing():
    ev = L.parse_events(_battle(["|move|p1a: Claydol|Psychic|p2a: Snorlax"]))
    assert L.identify_our_side(ev, {})[0] is None
    assert L.identify_our_side(ev, {1: "blissey"})[0] is None, "matches neither side"


def test_analyze_battle_skips_rather_than_assuming_a_side():
    lines = _battle(["|move|p1a: Claydol|Psychic|p2a: Snorlax"])
    r = L.analyze_battle(lines, [{"turn": 1, "phase": "move_selection", "our": {"species": "ho-oh"}}])
    assert r.skipped and r.skipped.startswith("side_undetermined")
    assert r.whiffs == 0 and r.moved_into_pivots == 0


# ------------------------------------------------------------------ pivot kinds


def _decisions(*turns):
    return [{"turn": t, "phase": "move_selection", "our": {"species": "claydol"},
             "chosen": "earthquake", "actions": {"earthquake": {"prob": "99.0%"}}}
            for t in turns]


def test_a_voluntary_pivot_we_fire_into_immune_is_a_whiff():
    lines = _battle(["|switch|p2a: Salamence|Salamence, F|100/100",
                     "|move|p1a: Claydol|Earthquake|p2a: Salamence",
                     "|-immune|p2a: Salamence"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.our_side == "p1"
    assert r.opp_voluntary_pivots == 1 and r.moved_into_pivots == 1
    assert r.whiffs == 1 and r.whiff_kinds == {"immune": 1}
    assert r.baits[0].kind == "immune" and r.baits[0].chosen_prob == pytest.approx(0.99)


def test_a_drag_is_not_a_voluntary_pivot():
    """Whirlwind/Roar put the mon there — charging the opponent with a bait it did not choose
    would count OUR phazing as THEIR trap."""
    lines = _battle(["|move|p1a: Claydol|Roar|p2a: Snorlax",
                     "|drag|p2a: Salamence|Salamence, F|100/100"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.opp_voluntary_pivots == 0 and r.moved_into_pivots == 0


def test_a_post_faint_replacement_is_not_a_voluntary_pivot():
    lines = _battle(["|move|p1a: Claydol|Earthquake|p2a: Snorlax",
                     "|-damage|p2a: Snorlax|0 fnt", "|faint|p2a: Snorlax",
                     "|switch|p2a: Salamence|Salamence, F|100/100"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.opp_voluntary_pivots == 0, "the replacement was forced by the faint"


def test_a_miss_is_counted_separately_and_is_never_a_whiff():
    lines = _battle(["|switch|p2a: Salamence|Salamence, F|100/100",
                     "|move|p1a: Claydol|Earthquake|p2a: Salamence",
                     "|-miss|p1a: Claydol|p2a: Salamence"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.moved_into_pivots == 1 and r.whiffs == 0 and r.misses == 1


def test_a_self_targeting_move_is_not_a_bait():
    """We recovered into their pivot: nothing was fired at the arrival, so this is not an
    opportunity to whiff and must not enter the denominator."""
    lines = _battle(["|switch|p2a: Salamence|Salamence, F|100/100",
                     "|move|p1a: Claydol|Recover|p1a: Claydol",
                     "|-heal|p1a: Claydol|100/100"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.opp_voluntary_pivots == 1 and r.moved_into_pivots == 0


def test_a_fail_with_an_external_from_cause_is_not_our_move_failing():
    lines = _battle(["|switch|p2a: Salamence|Salamence, F|100/100",
                     "|move|p1a: Claydol|Earthquake|p2a: Salamence",
                     "|-fail|p2a: Salamence|[from] ability: Wonder Guard",
                     "|-damage|p2a: Salamence|55/100"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.baits[0].kind == "hit", "the [from] cause is not the move failing on its own"


def test_a_bare_fail_is_a_whiff():
    lines = _battle(["|switch|p2a: Salamence|Salamence, F|100/100",
                     "|move|p1a: Claydol|Toxic|p2a: Salamence", "|-fail|p2a: Salamence"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.whiff_kinds == {"fail": 1}


def test_a_one_percent_hit_is_near_zero_and_a_real_hit_is_not():
    """The exact calibration-battle case (91% → 90% Rapid Spin). Float arithmetic makes this
    0.010000000000000009, so a bare `<=` reads it as a real hit."""
    def _one(after):
        return _battle(["|switch|p2a: Metagross|Metagross|91/100",
                        "|move|p1a: Claydol|Rapid Spin|p2a: Metagross",
                        "|-resisted|p2a: Metagross", f"|-damage|p2a: Metagross|{after}/100"])
    assert L.analyze_battle(_one(90), _decisions(1)).baits[0].kind == "near_zero"
    assert L.analyze_battle(_one(70), _decisions(1)).baits[0].kind == "hit"


def test_entry_hazard_damage_is_not_charged_to_our_attack():
    """Spikes take 12% off the arrival before we move; our 0-damage attack must still read
    near_zero, not a 12% hit."""
    lines = _battle(["|switch|p2a: Salamence|Salamence, F|100/100",
                     "|-damage|p2a: Salamence|88/100|[from] Spikes",
                     "|move|p1a: Claydol|Earthquake|p2a: Salamence",
                     "|-immune|p2a: Salamence"])
    r = L.analyze_battle(lines, _decisions(1))
    assert r.baits[0].kind == "immune"


# ----------------------------------------------------------- loops and re-clicks


def _loop_battle(n: int):
    turns = []
    for _ in range(n):
        turns.append(["|switch|p2a: Salamence|Salamence, F|100/100",
                      "|move|p1a: Claydol|Earthquake|p2a: Salamence",
                      "|-immune|p2a: Salamence"])
        turns.append(["|switch|p2a: Snorlax|Snorlax|100/100"])   # pivot back out
    return _battle(*turns)


def test_two_of_the_same_pair_is_a_loop_and_the_second_is_the_reclick():
    r = L.analyze_battle(_loop_battle(2), _decisions(1, 3))
    assert r.loop_battle and r.worst_loop == 2
    assert [g["turns"] for g in r.loops] == [[1, 3]]
    assert [b.loop_step for b in r.baits] == [True, True], "a loop is symmetric over the battle"
    assert [b.reclick for b in r.baits] == [False, True], "a re-click is ordered"
    assert r.reclicks == 1


def test_one_whiff_is_neither_a_loop_nor_a_reclick():
    r = L.analyze_battle(_loop_battle(1), _decisions(1))
    assert r.whiffs == 1 and not r.loop_battle and r.reclicks == 0 and r.worst_loop == 0


def test_different_pairs_do_not_form_a_loop():
    lines = _battle(
        ["|switch|p2a: Salamence|Salamence, F|100/100",
         "|move|p1a: Claydol|Earthquake|p2a: Salamence", "|-immune|p2a: Salamence"],
        ["|switch|p2a: Gengar|Gengar|100/100",
         "|move|p1a: Claydol|Earthquake|p2a: Gengar", "|-immune|p2a: Gengar"])
    r = L.analyze_battle(lines, _decisions(1, 2))
    assert r.whiffs == 2 and not r.loop_battle and r.reclicks == 0


def test_the_mirror_is_the_same_detector_with_the_sides_swapped():
    """WE pivot, THEY fire into it. The control arm — it measures the opponent's policy."""
    lines = _battle(["|switch|p1a: Salamence|Salamence, F|100/100",
                     "|move|p2a: Snorlax|Earthquake|p1a: Salamence",
                     "|-immune|p1a: Salamence"])
    r = L.analyze_battle(lines, [{"turn": 1, "phase": "move_selection",
                                  "our": {"species": "claydol"}}])
    assert r.our_side == "p1"
    assert r.whiffs == 0, "this is not OUR whiff"
    assert r.our_voluntary_pivots == 1 and r.mirror_moved_into == 1 and r.mirror_whiffs == 1


# ----------------------------------------------------------------- the α/β read


def _intent(alpha_switch_p, beta_slot):
    return {"alpha": [{"name": "SWITCH", "p": alpha_switch_p},
                      {"name": "Body Slam", "p": 1.0 - alpha_switch_p}],
            "beta": [{"slot": beta_slot, "p": 0.7, "species": "salamence"}]}


def test_beta_slot_is_graded_against_the_REVEAL_ORDER_not_the_species_name():
    """Obs slot k is the k-th REVEALED opponent mon. Snorlax led (slot 0), Salamence was revealed
    on turn 1, so on its SECOND pivot-in the true slot is 1 — and β's own species label is a belief
    decode, which cannot be the ground truth it is graded against."""
    lines = _battle(
        ["|switch|p2a: Salamence|Salamence, F|100/100",
         "|move|p1a: Claydol|Earthquake|p2a: Salamence", "|-immune|p2a: Salamence"],
        ["|switch|p2a: Snorlax|Snorlax|100/100"],
        ["|switch|p2a: Salamence|Salamence, F|100/100",
         "|move|p1a: Claydol|Earthquake|p2a: Salamence", "|-immune|p2a: Salamence"])
    invs = _decisions(1, 2, 3)
    invs[0]["opp_intent"] = _intent(0.8, 1)
    invs[2]["opp_intent"] = _intent(0.9, 1)
    r = L.analyze_battle(lines, invs)
    first, _back, repeat = r.reads
    assert first.first_time and not first.arrival_revealed
    assert first.slot_true is None and first.slot_correct is None, "undecidable, not wrong"
    assert repeat.arrival_revealed and repeat.slot_true == 1 and repeat.slot_correct is True
    assert repeat.loop_step and repeat.alpha_top_is_switch and repeat.alpha_switch_p == 0.9


def test_a_pivot_with_no_recorded_intent_reads_none_everywhere():
    r = L.analyze_battle(_loop_battle(1), _decisions(1))
    assert r.reads[0].slot_correct is None and r.reads[0].alpha_top_is_switch is None


# ------------------------------------------------------------- the critic joins


def test_turn_deltas_bucket_every_turn_and_the_worst_decision_on_a_turn_wins():
    lines = _battle(
        ["|switch|p2a: Salamence|Salamence, F|100/100",
         "|move|p1a: Claydol|Earthquake|p2a: Salamence", "|-immune|p2a: Salamence"],
        ["|move|p1a: Claydol|Psychic|p2a: Salamence", "|-damage|p2a: Salamence|40/100"])
    invs = _decisions(1, 2)
    invs.append({"turn": 2, "phase": "forced_switch", "our": {"species": "claydol"}})
    r = L.analyze_battle(lines, invs, values=[10.0, 8.0, 7.0, 1.0], win_probs=[0.9, 0.8, 0.7, 0.5])
    assert r.baits[0].delta_v == pytest.approx(-2.0)
    buckets = {d["turn"]: d for d in r.turn_deltas}
    assert buckets[1]["bucket"] == "other_bait", "a lone whiff is a bait, not a loop step"
    # turn 2 holds two decisions (−1.0 then −6.0): the WORST is what the turn is charged with.
    assert buckets[2]["delta_v"] == pytest.approx(-6.0) and buckets[2]["bucket"] == "other"


def test_chosen_prob_parses_the_recorded_percentage():
    assert L.chosen_prob({"chosen": "eq", "actions": {"eq": {"prob": "96.3%"}}}) == pytest.approx(0.963)
    assert L.chosen_prob({"chosen": "eq", "actions": {}}) is None
    assert L.chosen_prob({"chosen": "eq", "actions": {"eq": {"prob": "n/a"}}}) is None


def test_median_is_none_on_empty_and_averages_an_even_count():
    assert L.median([]) is None
    assert L.median([1.0, 3.0]) == pytest.approx(2.0)
    assert L.median([5.0, 1.0, 3.0]) == pytest.approx(3.0)


# --------------------------------------------------------------- coverage/degradation


def test_an_empty_protocol_is_skipped_with_a_reason_never_silently_zero():
    r = L.analyze_battle([], _decisions(1))
    assert r.skipped == "empty_protocol"


def test_baselines_carry_their_provenance():
    """A number without its source becomes a target the moment someone reads it twice."""
    assert {"generation", "measured", "n_battles", "source"} <= set(L.LOOP_BASELINES)
    assert "ledger.md" in L.LOOP_BASELINES["source"]


# --------------------------------------------------------------- the calibration gate


def test_the_calibration_battle_shape_is_detected_exactly():
    """The gen-15 linchpin, `step_22000032/sentinel_0/win_s0_001`, rebuilt as protocol.

    Nine Earthquake-into-a-switched-in-Salamence immunities on turns 3, 7, …, 35 — ONE loop of
    count 9 — plus two whiffs that are NOT part of it: a 1%-damage Rapid Spin on turn 40 and a
    different immunity on turn 44. That real trace lives under `models/`, which is gitignored and
    exists only in the main checkout, so the gate is pinned HERE as a shape and RUN against the
    trace by hand (see `designs/research_state/bait_loop_hunt.md`). A detector that merges the two
    stragglers into the loop, or drops them, fails this.
    """
    lines = list(_HEAD) + ["|switch|p1a: Claydol|Claydol|100/100",
                           "|switch|p2a: Snorlax|Snorlax|100/100"]
    for turn in range(1, 45):
        lines.append(f"|turn|{turn}")
        if turn in (3, 7, 11, 15, 19, 23, 27, 31, 35):
            lines += ["|switch|p2a: Salamence|Salamence, F|100/100",
                      "|move|p1a: Claydol|Earthquake|p2a: Salamence", "|-immune|p2a: Salamence"]
        elif turn == 40:
            lines += ["|switch|p2a: Metagross|Metagross|91/100",
                      "|move|p1a: Claydol|Rapid Spin|p2a: Metagross",
                      "|-resisted|p2a: Metagross", "|-damage|p2a: Metagross|90/100"]
        elif turn == 44:
            lines += ["|switch|p2a: Claydol|Claydol|100/100",
                      "|move|p1a: Claydol|Earthquake|p2a: Claydol", "|-immune|p2a: Claydol"]
        else:                      # a plain turn: our attack lands on the incumbent
            lines += ["|switch|p2a: Snorlax|Snorlax|100/100"]
        lines.append("|upkeep")

    r = L.analyze_battle(lines, _decisions(*range(1, 45)))
    assert r.skipped is None and r.our_side == "p1"
    assert len(r.loops) == 1
    assert r.loops[0] == {"move": "earthquake", "arrival": "salamence", "count": 9,
                          "turns": [3, 7, 11, 15, 19, 23, 27, 31, 35]}
    assert r.whiffs == 11 and r.whiff_kinds == {"immune": 10, "near_zero": 1}
    assert [(b.turn, b.kind) for b in r.baits if b.whiff and not b.loop_step] == [
        (40, "near_zero"), (44, "immune")]
    assert r.reclicks == 8, "9 loop clicks = 1 first + 8 re-clicks"
