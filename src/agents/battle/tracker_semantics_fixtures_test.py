"""The Rust core M3 loss catalogue's GIGO cases on REAL constructed battles, through SLICE T.

Each battle is one of M3's constructed fixtures (`src/rust_sim/tests/window_record_test.rs`, exported
to `designs/research_state/measurements/rust_core_m3_2026-09-24/catalogue/e12_fixtures.jsonl`) or one
of its GIGO reproductions (`…/catalogue/gigo_repros.jsonl`): fixed levels and sets, a fixed seed,
scripted choices. It is replayed through ``core_events --trackers`` with slice T on, so:

* slice T must hold the core's trackers EQUAL to the ``EpisodeTracker`` training drives (0
  divergences) — a revert of EITHER side's fix is a divergence;
* the corrected VALUE is asserted on the (equal) tracker state — a revert of BOTH sides together is
  a wrong value.

The cases: `gen3_event_window_semantics_fixes_v1` (W1 stat drop, W2 Destiny Bond / Perish Song
faint, W3 Trick / Thief, W4 Protect, W5 Rapid Spin), `gen3_intent_label_semantics_fixes_v1` (L1 / L2
drag, L3 straddling replacement, L4 caller, L5 Encore) and `gen3_progress_clock_attribution_fix_v1`
(T1 status moves in sand / beside the opponent's recoil, T2 a blocked attack freezes the clock). The
record: `designs/research_state/measurements/training_input_gigo_fixes_2026-09-24/`. Also
`gen3_hp_prior_support_v1` (an off-prior Hidden Power type falls back to the flat prior).

Tier: unmarked, like slice T's COMMIT tier — 14 short battles through the in-process core, ~seconds.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import pytest

from agents.battle import rust_core_parity as P
from agents.battle import rust_core_parity_trackers as T

_T_MOVE, _T_FAINT, _T_BOOST, _T_ITEM, _T_HAZARD = 1, 3, 6, 7, 8
_SWAPPED = 4
_KIND_MOVE, _KIND_UNKNOWN = 0, 2

# name → (p1 team, p2 team, seed, [(side, choice), …]) — M3's inputs verbatim.
FIXTURES: Dict[str, Tuple[str, str, str, List[Tuple[int, str]]]] = {
    'destiny_bond': ('gengar|||levitate|destinybond|Hardy|||||50|]snorlax|||immunity|rest|Hardy|||||100|', 'tyranitar|||sandstream|crunch|Hardy|||||60|]snorlax|||immunity|rest|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'switch 2'), (1, 'switch 2'), (0, 'move 1'), (1, 'move 1')]),
    'thief': ('sneasel|||innerfocus|thief|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'blissey||leftovers|naturalcure|softboiled|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'move 1'), (1, 'move 1')]),
    'trick': ('alakazam||choiceband|synchronize|trick,psychic|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'blissey||leftovers|naturalcure|softboiled|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1')]),
    'sleep_talk': ('snorlax|||immunity|rest,sleeptalk,bodyslam|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|', 'skarmory|||keeneye|drillpeck|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', '1,2,3,4', [(1, 'move 1'), (0, 'move 1'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1')]),
    'rapid_spin': ('starmie|||naturalcure|rapidspin|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'skarmory|||keeneye|spikes,drillpeck|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', '1,2,3,4', [(1, 'move 1'), (0, 'move 1'), (1, 'move 2'), (0, 'move 1')]),
    'protect': ('skarmory|||keeneye|protect|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'snorlax|||immunity|bodyslam|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1')]),
    'perish': ('lapras|||waterabsorb|perishsong,rest|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'blissey|||naturalcure|softboiled|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1'), (0, 'switch 2'), (1, 'switch 2'), (0, 'move 1'), (1, 'move 1')]),
    'encore_lands': ('gengar|||levitate|encore,splash|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'snorlax|||immunity|curse,bodyslam|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|', '1,2,3,4', [(0, 'move 2'), (1, 'move 1'), (0, 'move 1'), (1, 'move 2')]),
    'taunt_in_sand': ('tyranitar||leftovers|sandstream|taunt|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'snorlax|||immunity|bodyslam|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'move 1'), (1, 'move 1'), (0, 'move 1'), (1, 'move 1')]),
    'opp_recoil_beside_our_status_move': ('blissey|||naturalcure|toxic,thunderwave|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'snorlax|||immunity|doubleedge|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|', '1,2,3,4', [(0, 'move 2'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1')]),
    'roar_overrides_their_chosen_switch': ('skarmory|||keeneye|roar|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'snorlax|||immunity|rest|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|]starmie|||naturalcure|recover|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'switch 2'), (0, 'move 1'), (1, 'move 1')]),
    # gen3_move_target_class_v1 (R4): a Refresh with no status to cure fails (`|move|…|Refresh||[still]`).
    # (A Snatch-stolen use is pinned on hand-built lines only: a constructed Snatch battle is REFUSED
    # by the core's own parse-vs-step check today — a pre-existing `-fail` OWNER disagreement.)
    'failed_refresh': ('umbreon|||synchronize|splash,toxic|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'swampert|||torrent|refresh|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1')]),
    # gen3_hp_prior_support_v1: an IV-less Lunatone's Hidden Power is DARK (every IV 31), a type its
    # Smogon usage row gives 0.0; Dark is 2x on Gengar (the cutover stress's `ladderA_3459` refusal).
    'hp_dark_off_prior': ('gengar|||levitate|splash|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'lunatone|||levitate|hiddenpower|Hardy|||||50|]snorlax|||immunity|rest|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'move 1'), (1, 'move 1')]),
    'asleep_then_dragged': ('skarmory|||keeneye|drillpeck,roar|Hardy|||||100|]snorlax|||immunity|rest|Hardy|||||100|', 'snorlax|||immunity|rest|Hardy|||||100|]blissey|||naturalcure|softboiled|Hardy|||||100|]starmie|||naturalcure|recover|Hardy|||||100|', '1,2,3,4', [(0, 'move 1'), (1, 'move 1'), (0, 'move 2'), (1, 'move 1'), (0, 'move 1'), (1, 'move 1')]),
}


@pytest.fixture(scope="module")
def replays() -> Dict[str, List[List[dict]]]:
    """name → per viewer → the core's tracker state at each decision, slice T clean."""
    logging.getLogger("poke-env").setLevel(logging.ERROR)
    real_scope = T.SCOPE_FORMATS
    # the fixtures are `gen3customgame` (fixed levels and sets); slice T's production scope is gen3ou
    T.SCOPE_FORMATS = frozenset(real_scope | {"gen3customgame"})
    out: Dict[str, List[List[dict]]] = {}
    try:
        for name, (p1, p2, seed, script) in FIXTURES.items():
            b = P.RecordedBattle(label=f"gigo.{name}", format_id="gen3customgame", seed=seed,
                                 p1={"name": "P1", "team": p1}, p2={"name": "P2", "team": p2},
                                 commands=[[f"p{s + 1}", tok, "if_open"] for s, tok in script])
            t, got = T.TrackerCensus(), {}
            P.check_battles([b], P.Census(), trackers=t, on_result=lambda _b, res: got.update(res))
            assert t.decisions and not t.divergences and not t.refused, f"{name}: slice T\n{t.render()}"
            out[name] = [[c["trackers"] for c in viewer if "trackers" in c] for viewer in got["trackers"]]
    finally:
        T.SCOPE_FORMATS = real_scope
    return out


def _rows(r, name, viewer):
    return r[name][viewer][-1]["window"]["rows"]


def test_w1_curse_speed_drop_is_a_negative_boost_row(replays):
    for v in (0, 1):
        mags = sorted(x["hp_delta"] for x in _rows(replays, "encore_lands", v) if x["t"] == _T_BOOST)[:1]
        assert mags == [-1.0], f"viewer {v}"


def test_w2_destiny_bond_and_perish_song_faints_are_not_attacks(replays):
    for v in (0, 1):
        faints = {(x["actor"], x["faint_cause"]) for x in _rows(replays, "destiny_bond", v) if x["t"] == _T_FAINT}
        assert faints == {("gengar", "attack"), ("tyranitar", "other")}, f"viewer {v}"
        causes = {x["faint_cause"] for x in _rows(replays, "perish", v) if x["t"] == _T_FAINT}
        assert causes == {"other"}, f"viewer {v}"


def test_w3_trick_and_thief_item_lines_are_swapped(replays):
    for v in (0, 1):
        trick = [x["item_tr"] for x in _rows(replays, "trick", v) if x["t"] == _T_ITEM][:2]
        assert trick == [_SWAPPED, _SWAPPED], f"viewer {v}"
        taker = [x["item_tr"] for x in _rows(replays, "thief", v) if x["t"] == _T_ITEM and x["actor"] == "sneasel"]
        assert taker[:1] == [_SWAPPED], f"viewer {v}"


def test_w4_t2_a_protect_block_fails_the_move_and_freezes_the_clock(replays):
    for v in (0, 1):
        slam = [x for x in _rows(replays, "protect", v) if x["t"] == _T_MOVE and x["move_id"] == "bodyslam"]
        assert slam and all(x["failed"] for x in slam), f"viewer {v}"
    p2 = replays["protect"][1]        # the Body Slam user
    assert p2[-1]["delta"]["our_move_outcome"] == "fail"
    assert p2[-1]["clock"]["n"] == p2[-2]["clock"]["n"], "a blocked attack is an exogenous FREEZE"


def test_w5_rapid_spin_clear_is_a_negative_hazard_row(replays):
    for v in (0, 1):
        assert [x["hp_delta"] for x in _rows(replays, "rapid_spin", v) if x["t"] == _T_HAZARD] == [1.0, -1.0]


@pytest.mark.parametrize("name", ["roar_overrides_their_chosen_switch", "asleep_then_dragged"])
def test_l1_l2_a_dragged_mon_is_masked(replays, name):
    dragged = [s for s in replays[name][0] if s["delta"] and s["delta"]["opp_dragged"]]
    assert dragged, "the fixture drags the opponent"
    assert all(s["label"]["kind"] == _KIND_UNKNOWN for s in dragged)


def test_l3_the_replacement_in_the_window_after_its_faint_is_masked(replays):
    for v in (0, 1):
        rep = [s for s in replays["destiny_bond"][v] if s["delta"] and s["delta"]["opp_switch_is_replacement"]]
        assert rep and all(s["label"]["kind"] == _KIND_UNKNOWN for s in rep), f"viewer {v}"


def test_l4_sleep_talk_is_the_label_not_the_called_move(replays):
    called = [s for s in replays["sleep_talk"][1] if s["delta"] and s["delta"]["opp_called_via"] == "sleeptalk"]
    assert called
    assert all((s["label"]["kind"], s["label"]["move_id"]) == (_KIND_MOVE, "sleeptalk") for s in called)


def test_l5_the_encore_override_is_masked(replays):
    over = [s for s in replays["encore_lands"][0] if s["delta"] and s["delta"]["opp_choice_overridden"]]
    assert len(over) == 1 and over[0]["label"]["kind"] == _KIND_UNKNOWN


def test_t1_chip_our_move_did_not_deal_is_not_progress(replays):
    """M3's repro: the target's net HP fell to Sandstorm every turn while our Taunt dealt nothing.
    Clause (i) held the clock at 0 through three Taunts; it must climb. (M3's second T1 repro,
    `opp_double_edge_recoil_credits_our_status_move`, never reaches its case — Double-Edge KOs the
    Blissey on turn 1, so that window closes on a forced switch and the clock sits out; it is kept
    below only as slice-T coverage.)"""
    ours = replays["taunt_in_sand"][0]
    spurious = [s for s in ours[1:] if s["delta"]["our_damaging"] and s["delta"]["our_move_hit_delta"] == 0.0]
    assert spurious, "the fixture holds the old clause's spurious inputs"
    n = [s["clock"]["n"] for s in ours]
    assert n[-1] >= 2 and all(b >= a for a, b in zip(n[1:], n[2:])), n


def test_r4_a_self_move_row_targets_its_user_on_both_paths(replays):
    """gen3_move_target_class_v1 (R4): `|move|p2a: Swampert|Refresh||[still]` (no status to cure) is
    Swampert's row, and Splash (`self`) is Umbreon's — on the core AND (slice T clean, asserted in the
    fixture) the Python path; Toxic keeps the foe. Reverting either side's rule fails here."""
    for v in (0, 1):
        rows = [x for x in _rows(replays, "failed_refresh", v) if x["t"] == _T_MOVE]
        got = [(x["actor"], x["move_id"], x["target"]) for x in rows]
        assert got == [("umbreon", "splash", "umbreon"), ("swampert", "refresh", "swampert"),
                       ("umbreon", "toxic", "swampert"), ("swampert", "refresh", "swampert")], f"viewer {v}: {got}"


def test_hp_an_off_prior_hidden_power_type_falls_back_to_the_flat_prior_on_both_paths(replays):
    """gen3_hp_prior_support_v1: Lunatone's usage prior gives HP Dark 0.0, but an IV-less set IS HP
    Dark (`sim/pokemon.ts:387-394` + `sim/dex.ts` `getHiddenPower`), and its 2x on Gengar eliminated
    every prior-supported type — both paths REFUSED ("all candidates eliminated"). Now the refuted row
    is replaced by the flat prior: Dark / Ghost / Psychic (every type 2x on Ghost / Poison) survive at
    1/16 on the core AND the Python path (slice T clean, asserted in the fixture). Reverting either
    side refuses the battle; reverting both refuses it on both."""
    flat = 1.0 / 16.0
    names = ["bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost", "grass", "ground",
             "ice", "poison", "psychic", "rock", "steel", "water"]
    want = [flat if n in ("dark", "ghost", "psychic") else 0.0 for n in names]
    hp = replays["hp_dark_off_prior"][0][-1]["hp"]           # p1 = Gengar's side, after the hit
    assert hp["state"] == [["lunatone", want]], hp
    assert hp["prior_discarded"] == ["lunatone"], hp
    assert replays["hp_dark_off_prior"][1][-1]["hp"]["state"] == []   # p1 carries no Hidden Power
