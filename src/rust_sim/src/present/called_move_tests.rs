//! The CALLED-MOVE reading class (`gen3_called_move_reading_v1`) in the core's `BoardReading`:
//! a move another move called — `|move|<user>|<called>|<target>|[from] Metronome` (gen3's BARE
//! form; also Assist and Nature Power) — is neither revealed nor charged, and a multi-flag tail
//! after the `[from]` (`…|[from] Metronome|[miss]|[miss]`, a real gen3 shape) parses. poke-env
//! reads the same (`src/poke_env/battle/called_move_reading_test.py`); both are the SIM's
//! reading: gen3 `useMoveInner` charges no Pressure PP for a sourced move. Every shape is one
//! `harness/probe_called_move_shapes.js` saw the real sim emit.

use super::view::MonView;
use super::*;
use crate::core_events::Line;

const OWN_TEAM: &str = "Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]\
Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||";

fn request(active: &str) -> String {
    let (am, as_) = if active == "Metagross" { ("true", "false") } else { ("false", "true") };
    format!(
        "|request|{{\"active\":[{{\"moves\":[{{\"move\":\"Surf\",\"id\":\"surf\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false}}]}}],\
\"side\":{{\"name\":\"me\",\"id\":\"p1\",\"pokemon\":[\
{{\"ident\":\"p1: Metagross\",\"details\":\"Metagross\",\"condition\":\"301/301\",\"active\":{am},\"stats\":{{\"atk\":405,\"def\":296,\"spa\":203,\"spd\":216,\"spe\":214}},\"moves\":[\"meteormash\",\"earthquake\",\"explosion\",\"agility\"],\"baseAbility\":\"clearbody\",\"item\":\"leftovers\",\"pokeball\":\"pokeball\"}},\
{{\"ident\":\"p1: Suicune\",\"details\":\"Suicune\",\"condition\":\"341/341\",\"active\":{as_},\"stats\":{{\"atk\":139,\"def\":266,\"spa\":260,\"spd\":308,\"spe\":213}},\"moves\":[\"calmmind\",\"surf\",\"rest\",\"icebeam\"],\"baseAbility\":\"pressure\",\"item\":\"leftovers\",\"pokeball\":\"pokeball\"}}]}}}}"
    )
}

const PREFIX: &[&str] = &["|player|p1|me||", "|player|p2|foe||", "|teamsize|p1|2", "|teamsize|p2|2", "|gen|3", "|start"];

/// Our Suicune (Pressure, from the request) faces the opponent's Clefable; then `extra`.
fn clefable(extra: &[&str]) -> MonView {
    let mut t = BoardReading::new(0, "me", Some(OWN_TEAM)).unwrap();
    let mut lines: Vec<String> = PREFIX.iter().map(|s| s.to_string()).collect();
    lines.push(request("Suicune"));
    lines.push("|switch|p1a: Suicune|Suicune|341/341".into());
    lines.push("|switch|p2a: Clefable|Clefable, F|100/100".into());
    lines.push("|turn|1".into());
    lines.extend(extra.iter().map(|s| s.to_string()));
    for l in &lines {
        t.feed(&Line::parse(l).unwrap()).unwrap_or_else(|e| panic!("{l}: {e}"));
    }
    let v = present(&t).unwrap();
    v.opp.mons.iter().find(|m| m.species == "clefable").expect("opp clefable").clone()
}

fn moves(m: &MonView) -> Vec<(String, u32, u32)> {
    m.moves.iter().map(|x| (x.id.clone(), x.current_pp, x.max_pp)).collect()
}

/// Every random-caller shape: only the CALLER is revealed, charged one PP on its own line.
#[test]
fn a_called_move_is_neither_revealed_nor_charged() {
    let cases: &[(&str, &str, u32)] = &[
        ("Metronome", "|move|p2a: Clefable|Thunderbolt|p1a: Suicune|[from] Metronome", 16),
        ("Metronome", "|move|p2a: Clefable|Swords Dance|p2a: Clefable|[from] Metronome", 16),
        ("Metronome", "|move|p2a: Clefable|Slack Off||[from] Metronome|[still]", 16),
        ("Metronome", "|move|p2a: Clefable|Rock Slide|p1a: Suicune|[from] Metronome|[miss]", 16),
        ("Metronome", "|move|p2a: Clefable|Super Fang|p1a: Suicune|[from] Metronome|[miss]|[miss]", 16),
        ("Assist", "|move|p2a: Clefable|Explosion|p1a: Suicune|[from] Assist", 32),
        ("Assist", "|move|p2a: Clefable|Roar||[from] Assist|[still]", 32),
        ("Nature Power", "|move|p2a: Clefable|Swift|p1a: Suicune|[from] Nature Power", 32),
        ("Nature Power", "|move|p2a: Clefable|Swift|p1a: Suicune|[from]Nature Power|[miss]", 32),
    ];
    for (caller, called, max_pp) in cases {
        let caller_line = format!("|move|p2a: Clefable|{caller}|p2a: Clefable");
        let m = clefable(&[&caller_line, called]);
        let id = caller.to_lowercase().replace(' ', "");
        assert_eq!(moves(&m), vec![(id, max_pp - 1, *max_pp)], "{called}");
    }
}

/// Metronome -> Nature Power -> Swift: only the outer caller is the actor's own move.
#[test]
fn a_nested_call_reveals_only_the_outer_caller() {
    let m = clefable(&[
        "|move|p2a: Clefable|Metronome|p2a: Clefable",
        "|move|p2a: Clefable|Nature Power|p2a: Clefable|[from] Metronome",
        "|move|p2a: Clefable|Swift|p1a: Suicune|[from] Nature Power",
    ]);
    assert_eq!(moves(&m), vec![("metronome".to_string(), 15, 16)]);
}

#[test]
fn the_bare_caller_predicate_and_the_canonical_tail() {
    for t in ["[from] Metronome", "[from]Metronome", "[from] Assist", "[from] Nature Power", "[from]Nature Power"] {
        assert!(board_reading::is_gen3_bare_move_caller(t), "{t}");
    }
    for t in ["[from] Sleep Talk", "[from] move: Metronome", "[from] lockedmove", "[from] Mirror Move", "Metronome"] {
        assert!(!board_reading::is_gen3_bare_move_caller(t), "{t}");
    }
    let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    let mut ev = v(&["", "move", "p2a: X", "Tackle", "p1a: Y", "[from] Metronome", "[miss]", "[miss]"]);
    board_reading::canonical_from_tail(&mut ev);
    assert_eq!(ev, v(&["", "move", "p2a: X", "Tackle", "p1a: Y", "[from] Metronome", "[miss]"]));
    let mut ev = v(&["", "move", "p2a: X", "Tackle", "", "[from] Assist", "[miss]", "[still]", "[notarget]"]);
    board_reading::canonical_from_tail(&mut ev);
    assert_eq!(ev, v(&["", "move", "p2a: X", "Tackle", "", "[from] Assist", "[notarget]", "[still]", "[miss]"]));
    for keep in [
        v(&["", "move", "p2a: X", "Tackle", "p1a: Y", "[from] Metronome", "[miss]"]),
        v(&["", "move", "p2a: X", "Tackle", "p1a: Y", "[miss]", "[still]"]),
    ] {
        let mut ev = keep.clone();
        board_reading::canonical_from_tail(&mut ev);
        assert_eq!(ev, keep, "a single flag, or no [from], is left alone");
    }
}
