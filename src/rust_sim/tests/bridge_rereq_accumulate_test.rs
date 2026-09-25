//! Successive refusals in ONE decision (`gen3_rereq_accumulate_v1`) and a choice of the wrong KIND
//! for the request (`gen3_choice_kind_mismatch_v1`). Every expected value is the REAL sim's,
//! measured by `harness/probe_rereq_accumulate.js` (fail-loud; the case ids A*/K* match its rows).
//! Each test drives the SAME teams, seed and choices through the port's `BridgeSession` (the
//! `sim_bridge` engine path). The all-imprisoned F rows live in `bridge_imprison_struggle_test.rs`.
//!
//! - A: successive refusals ACCUMULATE on the one `activeRequest` the sim keeps. WRONG (pre-fix):
//!   the port rendered each re-request from the latest refusal alone.
//! - K: a `move` sent to a FORCED-SWITCH request is refused FIRST by `Side.chooseMove`'s
//!   `requestState !== 'move'` gate — `[Invalid choice] Can't move: You need a switch response`,
//!   nothing follows. WRONG (pre-fix): the engine accepted it, the flat driver dropped the
//!   decision and the boundary RE-OPENED, re-issuing the forceSwitch AND the other side's `wait`.

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{BridgeSession, Cmd, WireChoice};
use pokesim::dex::Dex;

fn opts(format: &str, p1: &str, p2: &str) -> BattleOptions {
    BattleOptions {
        format_id: format.to_string(),
        seed: Some("1,2,3,4".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    }
}
fn mv(side: usize, slot: usize) -> Cmd {
    Cmd { side, choice: WireChoice::Move(slot) }
}
fn sw(side: usize, slot: usize) -> Cmd {
    Cmd { side, choice: WireChoice::Switch(slot) }
}
fn drive(format: &str, p1: &str, p2: &str, turns: &[(Cmd, Cmd)], dex: &Dex) -> BridgeSession {
    let mut s = BridgeSession::new_construct_turn0(&opts(format, p1, p2), dex).expect("session");
    for (a, b) in turns {
        s.feed_cmd(a.clone(), dex);
        s.feed_cmd(b.clone(), dex);
        assert!(!s.is_ended(), "fixture faulted: the battle ended inside the prefix");
    }
    s
}
/// Lines emitted to `side` since chunk index `from`.
fn since(s: &BridgeSession, side: usize, from: usize) -> Vec<String> {
    s.chunks().chunks.iter().skip(from).filter(|c| c.side == side).flat_map(|c| c.lines.iter().cloned()).collect()
}
/// The `active[0]` block's keys AFTER its `moves` array.
fn active_flags(req: &str) -> String {
    let start = req.find("\"active\":[{\"moves\":[").expect("a move request") + "\"active\":[".len();
    let end = req[start..].find("],\"side\"").expect("active closes") + start;
    let active = &req[start..end];
    let moves_end = active.find("}],").or_else(|| active.find("}]}")).expect("moves end") + 2;
    active[moves_end..active.len() - 1].to_string()
}
/// Per offered move, `D` (disabled:true) / `-`, plus `s` when it carries `disabledSource` — the
/// probe's `reqSummary` shape, e.g. `icebeam/psychic/splash[Ds--]`.
fn moves_summary(req: &str) -> String {
    let start = req.find("\"active\":[{\"moves\":[").expect("a move request");
    let body = &req[start..];
    let body = &body[..body.find("}]").expect("moves end")];
    let (mut ids, mut marks) = (Vec::new(), String::new());
    for entry in body.split("{\"move\":").skip(1) {
        let id = entry.split("\"id\":\"").nth(1).and_then(|t| t.split('"').next()).unwrap_or("?");
        ids.push(id.to_string());
        marks.push(if entry.contains("\"disabled\":true") { 'D' } else { '-' });
        if entry.contains("\"disabledSource\"") {
            marks.push('s');
        }
    }
    format!("{}[{}]", ids.join("/"), marks)
}
fn req(s: &BridgeSession, side: usize) -> String {
    s.active_request_json(side).expect("an open request").to_string()
}
/// Feed `cmd` and return its side's lines, `|request|` collapsed to `|request|…`, and the
/// `Can't …:` tail collapsed as the probe does.
fn feed(s: &mut BridgeSession, cmd: Cmd, dex: &Dex) -> Vec<String> {
    let side = cmd.side;
    let mark = s.chunks().chunks.len();
    s.feed_cmd(cmd, dex);
    since(s, side, mark)
        .into_iter()
        .map(|l| {
            if l.starts_with("|request|") {
                "|request|…".to_string()
            } else if let Some(i) = l.find("Can't move: ") {
                format!("{}Can't move…", &l[..i])
            } else if let Some(i) = l.find("Can't switch: ") {
                format!("{}Can't switch…", &l[..i])
            } else {
                l
            }
        })
        .collect()
}

const DUGTRIO: &str = "Dugtrio||Leftovers|ArenaTrap|Imprison,IceBeam,Splash|Jolly|,,,,,|M||||]\
                       Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
const JYNX: &str = "Jynx||Leftovers|Oblivious|IceBeam,Psychic,Splash|Hardy|,,,,,|F||||]\
                    Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
const U_MOVE: &str = "|error|[Unavailable choice] Can't move…";
const U_SWITCH: &str = "|error|[Unavailable choice] Can't switch…";
const RR: &str = "|request|…";

fn acc(picks: &[Cmd], dex: &Dex) -> (Vec<Vec<String>>, String) {
    let mut s = drive("gen3customgame", DUGTRIO, JYNX, &[(mv(0, 0), mv(1, 2))], dex);
    let rows = picks.iter().map(|c| feed(&mut s, c.clone(), dex)).collect();
    let r = req(&s, 1);
    (rows, format!("{} {}", moves_summary(&r), active_flags(&r)))
}
fn v(xs: &[&str]) -> Vec<String> {
    xs.iter().map(|x| x.to_string()).collect()
}

/// A1 — refused MOVE then refused SWITCH: re-request #2 keeps slot 1 `disabled:true` and the
/// dropped `maybeLocked`, and gains `trapped`. WRONG (pre-fix): re-rendered from the switch alone.
#[test]
fn a1_move_then_switch_refusals_accumulate() {
    let dex = Dex::for_gen(3);
    let (rows, last) = acc(&[mv(1, 0), sw(1, 1)], &dex);
    assert_eq!(rows, vec![v(&[U_MOVE, RR]), v(&[U_SWITCH, RR])]);
    assert_eq!(last, "icebeam/psychic/splash[Ds--] ,\"maybeDisabled\":true,\"trapped\":true", "sim A1");
}

/// A2 — refused SWITCH then refused MOVE: re-request #2 keeps `trapped:true`.
#[test]
fn a2_switch_then_move_refusals_accumulate() {
    let dex = Dex::for_gen(3);
    let (rows, last) = acc(&[sw(1, 1), mv(1, 0)], &dex);
    assert_eq!(rows, vec![v(&[U_SWITCH, RR]), v(&[U_MOVE, RR])]);
    assert_eq!(last, "icebeam/psychic/splash[Ds--] ,\"maybeDisabled\":true,\"trapped\":true", "sim A2");
}

/// A3 — two refused MOVES: both slots flipped on re-request #2.
#[test]
fn a3_two_move_refusals_flip_both_slots() {
    let dex = Dex::for_gen(3);
    let (rows, last) = acc(&[mv(1, 0), mv(1, 2)], &dex);
    assert_eq!(rows, vec![v(&[U_MOVE, RR]), v(&[U_MOVE, RR])]);
    assert_eq!(last, "icebeam/psychic/splash[Ds-Ds] ,\"maybeDisabled\":true,\"maybeTrapped\":true", "sim A3");
}

/// A4 — the SAME imprisoned move twice is STILL `[Unavailable choice]` + a re-request:
/// `chooseMove` re-derives `maybeLocked` from Imprison's never-cleared `maybeDisabled`, so
/// `updateDisabledRequest` reports a change every time. A CONTROL against an over-wide
/// "no change ⇒ Invalid" fix.
#[test]
fn a4_same_imprisoned_move_twice_still_re_requests() {
    let dex = Dex::for_gen(3);
    let (rows, last) = acc(&[mv(1, 0), mv(1, 0)], &dex);
    assert_eq!(rows, vec![v(&[U_MOVE, RR]), v(&[U_MOVE, RR])]);
    assert_eq!(last, "icebeam/psychic/splash[Ds--] ,\"maybeDisabled\":true,\"maybeTrapped\":true", "sim A4");
}

/// A5 — the SAME hidden-trap switch twice: the 2nd finds `trapped` already set → `[Invalid
/// choice]`, NO re-request. WRONG (pre-fix): `[Unavailable choice]` + a re-request both times.
#[test]
fn a5_same_hidden_trap_switch_twice_is_invalid() {
    let dex = Dex::for_gen(3);
    let (rows, last) = acc(&[sw(1, 1), sw(1, 1)], &dex);
    assert_eq!(rows, vec![v(&[U_SWITCH, RR]), v(&["|error|[Invalid choice] Can't switch…"])]);
    assert_eq!(
        last,
        "icebeam/psychic/splash[---] ,\"maybeDisabled\":true,\"maybeLocked\":true,\"trapped\":true",
        "sim A5"
    );
}

/// A6 — a VISIBLE disable (the Choice Band lock) refused twice, no Imprison: the 2nd changes
/// nothing → `[Invalid choice]`, NO re-request. WRONG (pre-fix): Unavailable + re-request twice.
#[test]
fn a6_same_visible_disable_twice_is_invalid() {
    let dex = Dex::for_gen(3);
    let p1 = "Snorlax||ChoiceBand|Immunity|Splash,Rest|Adamant|,,,,,|M||||]\
              Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
    let p2 = "Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||]\
              Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
    let mut s = drive("gen3customgame", p1, p2, &[(mv(0, 0), mv(1, 0))], &dex);
    let a = feed(&mut s, mv(0, 1), &dex);
    let b = feed(&mut s, mv(0, 1), &dex);
    assert_eq!(a, v(&[U_MOVE, RR]), "sim A6 #1");
    assert_eq!(b, v(&["|error|[Invalid choice] Can't move…"]), "sim A6 #2");
    let r = req(&s, 0);
    assert_eq!(format!("{} {}", moves_summary(&r), active_flags(&r)), "splash/rest[-Ds] ", "sim A6");
}

/// K1 — a `move` at a FORCED-SWITCH boundary (`gen3_choice_kind_mismatch_v1`): Jynx's Psychic KOs
/// Dugtrio, p1 answers the forceSwitch with `move 3`. Node: ONE `|error|[Invalid choice] Can't
/// move: You need a switch response` to p1, NOTHING to p2 (`sim/side.ts:540-542`). Reverting the
/// `classify_reject` kind gate fails this (the port re-issued both requests, no `|error|`).
#[test]
fn k1_a_move_at_a_forced_switch_is_invalid_and_nothing_follows() {
    let dex = Dex::for_gen(3);
    let p2 = "Jynx||Leftovers|Oblivious|IceBeam,Psychic,Splash|Hardy|,,,,,|F||||]\
              Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
    let mut s = drive("gen3customgame", DUGTRIO, p2, &[(mv(0, 0), mv(1, 2)), (mv(0, 2), mv(1, 1))], &dex);
    // p1 may still need a turn if Psychic did not KO yet — play Splash vs Psychic until it does.
    for _ in 0..4 {
        if req(&s, 0).contains("\"forceSwitch\"") {
            break;
        }
        s.feed_cmd(mv(0, 2), &dex);
        s.feed_cmd(mv(1, 1), &dex);
    }
    assert!(req(&s, 0).contains("\"forceSwitch\""), "fixture: Dugtrio never fainted");
    let mark = s.chunks().chunks.len();
    s.feed_cmd(mv(0, 2), &dex);
    assert_eq!(
        since(&s, 0, mark),
        vec!["|error|[Invalid choice] Can't move: You need a switch response".to_string()],
        "sim K1: p1 gets the error alone"
    );
    assert!(since(&s, 1, mark).is_empty(), "sim K1: p2 receives NOTHING");
    // …and the boundary is still open: the switch it asked for is accepted.
    let mark = s.chunks().chunks.len();
    s.feed_cmd(sw(0, 1), &dex);
    assert!(since(&s, 0, mark).iter().all(|l| !l.starts_with("|error|")), "the replacement is accepted");
}
