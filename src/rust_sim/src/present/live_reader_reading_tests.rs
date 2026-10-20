//! The live reader's reading fixes (`gen3_live_reader_reading_v1`): each pin reads a REAL player
//! stream — the pinned Showdown's, captured by `harness/probe_live_reader_reading.js` into
//! `tests/vectors/live_reader_reading/` — through the core's `BoardReading` (the chain training's
//! rows and the live reader share) and asserts the reading at a decision equals the sim's truth,
//! which the same stream states (a `|request|`'s PP, the next `-end` line) or the sim's source fixes
//! (cited per test). Every pin failed before its fix.

use super::view::MonView;
use super::*;
use crate::core_events::Line;

const SPITE: [&str; 2] = [
    include_str!("../../tests/vectors/live_reader_reading/spite_p1.txt"),
    include_str!("../../tests/vectors/live_reader_reading/spite_p2.txt"),
];
const SPITE_TEAMS: [&str; 2] = [
    "Gengar|||levitate|spite,lockon,mindreader,shadowball|Hardy|85,85,85,85,85,|M||||]Electrode|||static|lockon,zapcannon,mindreader,thunderbolt|Hardy|85,85,85,85,85,85|M||||",
    "Snorlax|||immunity|bodyslam,curse,rest,sleeptalk|Hardy|85,85,85,85,85,|M||||]Blissey|||naturalcure|softboiled,seismictoss,toxic,aromatherapy|Hardy|85,85,85,85,85,85|M||||",
];
const MIMIC: [&str; 2] = [
    include_str!("../../tests/vectors/live_reader_reading/mimic_p1.txt"),
    include_str!("../../tests/vectors/live_reader_reading/mimic_p2.txt"),
];
const MIMIC_TEAMS: [&str; 2] = [
    "Snorlax|||immunity|mimic,curse,rest,bodyslam|Hardy|85,85,85,85,85,|M||||",
    "Jolteon|||voltabsorb|hiddenpower,substitute,batonpass,agility|Hardy|85,85,85,85,85,85|M|30,,,30,,30|||",
];
const YAWN: [&str; 2] = [
    include_str!("../../tests/vectors/live_reader_reading/yawn_p1.txt"),
    include_str!("../../tests/vectors/live_reader_reading/yawn_p2.txt"),
];
const YAWN_TEAMS: [&str; 2] = [
    "Snorlax||chestoberry|immunity|rest,curse,bodyslam,sleeptalk|Hardy|85,85,85,85,85,|M||||",
    "Slowbro|||owntempo|yawn,surf,calmmind,thunderwave|Hardy|85,85,85,85,85,85|M||||",
];
const SLEEPTALK: [&str; 2] = [
    include_str!("../../tests/vectors/live_reader_reading/sleeptalk_p1.txt"),
    include_str!("../../tests/vectors/live_reader_reading/sleeptalk_p2.txt"),
];
const SLEEPTALK_TEAMS: [&str; 2] = [
    "Snorlax|||immunity|rest,sleeptalk,bodyslam,return|Hardy|85,85,85,85,85,|M||||]Skarmory|||keeneye|spikes,whirlwind,drillpeck,toxic|Hardy|85,85,85,85,85,85|M||||",
    "Zapdos|||pressure|thunderbolt,drillpeck,agility,substitute|Hardy|85,85,85,85,85,85|M||||",
];
const FOCUSBAND: [&str; 2] = [
    include_str!("../../tests/vectors/live_reader_reading/focusband_p1.txt"),
    include_str!("../../tests/vectors/live_reader_reading/focusband_p2.txt"),
];
const FOCUSBAND_TEAMS: [&str; 2] = [
    "Machamp||choiceband|guts|crosschop,bulkup,rockslide,earthquake|Hardy|85,85,85,85,85,85|M||||",
    "Abra||focusband|synchronize|teleport,calmmind,psychic,substitute|Hardy|85,85,85,85,85,|M|||5|]Blissey|||naturalcure|softboiled,seismictoss,toxic,aromatherapy|Hardy|85,85,85,85,85,85|M||||",
];

/// The reading of `side`'s stream at its decision of turn `turn`: every line up to and including
/// the first `|request|` after `|turn|<turn>` (the write that opens that decision).
fn at_turn(stream: &[&str; 2], teams: &[&str; 2], side: usize, turn: u32) -> BoardReading {
    let name = ["me", "foe"][side];
    let mut t = BoardReading::new(side, name, Some(teams[side])).unwrap();
    let mut in_turn = false;
    for l in stream[side].lines() {
        t.feed(&Line::parse(l).unwrap()).unwrap_or_else(|e| panic!("{l}: {e}"));
        if l == format!("|turn|{turn}") {
            in_turn = true;
        } else if in_turn && l.starts_with("|request|") {
            return t;
        }
    }
    panic!("the stream has no decision at turn {turn}");
}

fn mon<'a>(v: &'a OneSidedView, own: bool, sp: &str) -> &'a MonView {
    let s = if own { &v.ours } else { &v.opp };
    s.mons.iter().find(|m| m.species == sp).unwrap_or_else(|| panic!("{sp} on {} side", if own { "our" } else { "their" }))
}

fn pp(m: &MonView, id: &str) -> u32 {
    m.moves.iter().find(|x| x.id == id).unwrap_or_else(|| panic!("{} holds no {id}: {:?}", m.species, m.moves)).current_pp
}

fn has_vol(m: &MonView, id: &str) -> bool {
    m.volatiles.iter().any(|(k, _)| k == id)
}

/// Mimic copies Jolteon's Hidden Power (`-activate|p1a: Snorlax|move: Mimic|Hidden Power`, no
/// type); the sim types it by its USER — our request lists `hiddenpowerdark` (Snorlax's IVs) — so
/// our copy is `hiddenpowerdark`: the move-type channel and `own_hp_typed_id` read Dark, not the
/// bare untyped id. The FOE's reading of the same copy stays bare (it cannot know our IVs).
#[test]
fn a_mimicked_hidden_power_takes_our_request_type() {
    let line = MIMIC[0].lines().filter(|l| l.starts_with("|request|")).nth(2).expect("turn-3 request");
    assert!(line.contains("\"hiddenpowerdark\""), "precondition: the sim's request types the copy");
    let t = at_turn(&MIMIC, &MIMIC_TEAMS, 0, 3);
    let v = present(&t).unwrap();
    let hp = mon(&v, true, "snorlax").moves.iter().find(|m| m.id == "hiddenpower").expect("the Mimic copy");
    assert_eq!(hp.move_id, "hiddenpowerdark");
    assert_eq!(legal_actions(&t).and_then(|l| l.own_hp_typed_id).as_deref(), Some("hiddenpowerdark"));
    // the next request keeps the one key: no second, base `hiddenpower` appears
    let v4 = present(&at_turn(&MIMIC, &MIMIC_TEAMS, 0, 4)).unwrap();
    let keys: Vec<&str> = mon(&v4, true, "snorlax").moves.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(keys, ["bodyslam", "curse", "hiddenpower", "rest"]);
    let theirs = present(&at_turn(&MIMIC, &MIMIC_TEAMS, 1, 3)).unwrap();
    let copy = mon(&theirs, false, "snorlax").moves.iter().find(|m| m.id == "hiddenpower").expect("their view of the copy");
    assert_eq!(copy.move_id, "hiddenpower", "the foe cannot type our copy");
}

/// Slowbro Yawns Snorlax, which Rests the same turn (`-heal|p1a: Snorlax|482/482 slp|[silent]`) and
/// wakes on its Chesto Berry: the sim's `yawn` volatile holds until its own residual `-end` (turn
/// 3's end), so at turn 3's decision BOTH readings hold it; at turn 4's it has ended and the mon
/// sleeps. (Upstream poke-env dropped it on the `slp` HP token.)
#[test]
fn a_rest_does_not_end_a_pending_yawn() {
    assert!(YAWN[0].contains("|-heal|p1a: Snorlax|482/482 slp|[silent]\n|-enditem|p1a: Snorlax|Chesto Berry|[eat]"), "precondition");
    assert!(YAWN[0].contains("|-end|p1a: Snorlax|move: Yawn|[silent]\n|-status|p1a: Snorlax|slp"), "precondition: the Yawn lands");
    for (side, own) in [(0, true), (1, false)] {
        let v3 = present(&at_turn(&YAWN, &YAWN_TEAMS, side, 3)).unwrap();
        let s3 = mon(&v3, own, "snorlax");
        assert!(has_vol(s3, "yawn"), "side {side} turn 3: {:?}", s3.volatiles);
        assert_eq!(s3.status, None, "side {side}: the Chesto Berry woke it");
        let v4 = present(&at_turn(&YAWN, &YAWN_TEAMS, side, 4)).unwrap();
        let s4 = mon(&v4, own, "snorlax");
        assert!(!has_vol(s4, "yawn"), "side {side} turn 4: {:?}", s4.volatiles);
        assert_eq!(s4.status, Some("slp"));
    }
}

/// Snorlax's Sleep Talk calls Body Slam into a Pressure Zapdos (`|move|p1a: Snorlax|Body
/// Slam|p2a: Zapdos|[from] Sleep Talk`): gen 3 charges a called move no Pressure PP, so Sleep
/// Talk costs ONE (our turn-4 request: 15 of 16) — the foe's reading of it too. The Body Slam it
/// called is revealed but not charged (22: turn 1's own use, two under Pressure).
#[test]
fn a_move_called_by_sleep_talk_charges_no_pressure() {
    assert!(SLEEPTALK[1].contains("|move|p1a: Snorlax|Body Slam|p2a: Zapdos|[from] Sleep Talk"), "precondition");
    let req4 = SLEEPTALK[0].lines().filter(|l| l.starts_with("|request|")).nth(3).expect("turn-4 request");
    assert!(req4.contains("\"id\":\"sleeptalk\",\"pp\":15"), "precondition: the sim charged one: {req4}");
    for turn in [4, 5] {
        let v = present(&at_turn(&SLEEPTALK, &SLEEPTALK_TEAMS, 1, turn)).unwrap();
        let s = mon(&v, false, "snorlax");
        assert_eq!(pp(s, "sleeptalk"), 15, "turn {turn}");
        assert_eq!(pp(s, "bodyslam"), 22, "turn {turn}");
    }
}

/// `-activate|p2a: Abra|item: Focus Band` discloses the band (it is not consumed): the opponent's
/// item reads `focusband`, publicly; the holder's own reading marks it public too.
#[test]
fn a_focus_band_survive_discloses_the_item() {
    assert!(FOCUSBAND[0].contains("|-activate|p2a: Abra|item: Focus Band"), "precondition");
    let v2 = present(&at_turn(&FOCUSBAND, &FOCUSBAND_TEAMS, 0, 2)).unwrap();
    assert_eq!(mon(&v2, false, "abra").item, None, "unknown before the survive");
    let v3 = present(&at_turn(&FOCUSBAND, &FOCUSBAND_TEAMS, 0, 3)).unwrap();
    let a = mon(&v3, false, "abra");
    assert_eq!((a.item.as_deref(), a.item_public), (Some("focusband"), true));
    let own = present(&at_turn(&FOCUSBAND, &FOCUSBAND_TEAMS, 1, 3)).unwrap();
    let a = mon(&own, true, "abra");
    assert_eq!((a.item.as_deref(), a.item_public), (Some("focusband"), true));
}

/// `-activate|p2a: Snorlax|move: Spite|bodyslam|4` cuts four PP from the foe's Body Slam: our
/// estimate reads the sim's 18 (24 − one use − 4 − one use; the foe's own turn-3 request says 18).
#[test]
fn spite_cuts_the_named_moves_pp() {
    assert!(SPITE[0].contains("|-activate|p2a: Snorlax|move: Spite|bodyslam|4"), "precondition");
    let req3 = SPITE[1].lines().filter(|l| l.starts_with("|request|")).nth(2).expect("turn-3 request");
    assert!(req3.contains("\"id\":\"bodyslam\",\"pp\":18"), "precondition: the sim's PP: {req3}");
    let v = present(&at_turn(&SPITE, &SPITE_TEAMS, 0, 3)).unwrap();
    assert_eq!(pp(mon(&v, false, "snorlax"), "bodyslam"), 18);
}

/// The sim's `lockon` volatile (added by Lock-On AND Mind Reader, `duration: 2`, ended silently at
/// the residual of the turn after) holds at the next turn's decision and is gone at the one after —
/// on both readings. Electrode: Lock-On on turn 4, Mind Reader on turns 6 and 9 (the turn-9 one
/// succeeds only because the turn-6 lock had ENDED: `onTryHit` fails while `lockon` is held).
#[test]
fn lock_on_and_mind_reader_last_through_the_next_turn() {
    assert!(SPITE[0].contains("|-activate|p1a: Electrode|move: Lock-On|[of] p2a: Snorlax"), "precondition");
    assert_eq!(SPITE[0].matches("|-activate|p1a: Electrode|move: Mind Reader|[of] p2a: Snorlax").count(), 2, "precondition");
    let held = |turn: u32, side: usize, own: bool| -> Vec<String> {
        let v = present(&at_turn(&SPITE, &SPITE_TEAMS, side, turn)).unwrap();
        let m = mon(&v, own, "electrode");
        m.volatiles.iter().map(|(k, _)| k.clone()).filter(|k| k == "lockon" || k == "mindreader").collect()
    };
    for (side, own) in [(0, true), (1, false)] {
        assert_eq!(held(4, side, own), Vec::<String>::new(), "side {side} turn 4");
        assert_eq!(held(5, side, own), ["lockon"], "side {side} turn 5: the turn after Lock-On");
        assert_eq!(held(6, side, own), Vec::<String>::new(), "side {side} turn 6: ended at turn 5's residual");
        assert_eq!(held(7, side, own), ["mindreader"], "side {side} turn 7: the turn after Mind Reader");
        assert_eq!(held(8, side, own), Vec::<String>::new(), "side {side} turn 8");
        assert_eq!(held(10, side, own), ["mindreader"], "side {side} turn 10");
    }
}
