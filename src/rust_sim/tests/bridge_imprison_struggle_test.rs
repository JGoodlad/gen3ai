//! An M6-cutover choice class (`gen3_imprison_all_struggle_v1`; the sibling A rows,
//! `gen3_rereq_accumulate_v1`, are not landed yet). Every expected value is the REAL sim's, measured by
//! `harness/probe_rereq_accumulate.js` (fail-loud; the case ids F*/A* match its rows). Each test
//! drives the SAME teams, seed and choices through the port's `BridgeSession` (the `sim_bridge`
//! engine path).
//!
//! - F: EVERY move imprisoned. The request still OFFERS the full list (the hidden disable renders
//!   `disabled:false`), and any in-range pick is SUBSTITUTED by Struggle — no `|error|`, an
//!   owner-only `|-activate|…|move: Struggle`, and the turn runs `|move|…|Struggle`. WRONG
//!   (pre-fix): `MonState::must_struggle` cannot see the foe's Imprison, so the flat driver's
//!   `choice_is_legal` dropped the decision and the boundary re-opened for BOTH sides.
//! - A: successive refusals in ONE decision ACCUMULATE on the one `activeRequest` the sim keeps.
//!   WRONG (pre-fix): the port rendered each re-request from the latest refusal alone.

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

const DUSCLOPS: &str = "Dusclops||Leftovers|Pressure|Imprison,Rest,Splash|Careful|,,,,,|M||||]\
                        Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
const SUICUNE: &str = "Suicune||Leftovers|Pressure|Rest,Splash|Calm|,,,,,|N||||]\
                       Celebi||Leftovers|NaturalCure|Recover,LeechSeed|Bold|,,,,,|N||||";

/// F1-F3 — every move imprisoned: the full list is offered, the pick is ACCEPTED (no `|error|`),
/// the owner alone gets `|-activate|p2a: Suicune|move: Struggle`, and the turn runs Struggle.
/// Reverting `BattleState::forced_struggle` to `MonState::must_struggle` at the driver's
/// `choice_is_legal` / queue build fails this (the decision is dropped, no Struggle runs).
#[test]
fn f1_all_imprisoned_pick_is_struggle_substituted() {
    let dex = Dex::for_gen(3);
    let mut s = drive("gen3ou", DUSCLOPS, SUICUNE, &[(mv(0, 0), mv(1, 1))], &dex);
    let r = req(&s, 1);
    assert_eq!(moves_summary(&r), "rest/splash[--]", "sim F1: the FULL list, hidden\n{r}");
    assert_eq!(active_flags(&r), ",\"maybeDisabled\":true,\"maybeLocked\":true", "sim F1\n{r}");
    let p2 = feed(&mut s, mv(1, 1), &dex);
    assert!(p2.iter().all(|l| !l.starts_with("|error|")), "sim F2: no |error|: {p2:#?}");
    let mark = s.chunks().chunks.len();
    let p1 = feed(&mut s, mv(0, 2), &dex);
    assert!(p1.iter().all(|l| !l.contains("move: Struggle")), "sim F2: the announce is OWNER-only: {p1:#?}");
    let p2_turn = since(&s, 1, mark);
    let act = p2_turn.iter().position(|l| l == "|-activate|p2a: Suicune|move: Struggle");
    let mvl = p2_turn.iter().position(|l| l.starts_with("|move|p2a: Suicune|Struggle|p1a: Dusclops"));
    assert!(act.is_some() && mvl.is_some() && act < mvl, "sim F2/F3: announce, then Struggle runs: {p2_turn:#?}");
}

/// F4 — the offered-count bound still applies: `move 3` on a 2-move all-imprisoned mon is
/// `[Invalid choice]` (the request offered 2, not the Struggle-only 1).
#[test]
fn f4_all_imprisoned_out_of_range_pick_is_still_refused() {
    let dex = Dex::for_gen(3);
    let mut s = drive("gen3ou", DUSCLOPS, SUICUNE, &[(mv(0, 0), mv(1, 1))], &dex);
    let mark = s.chunks().chunks.len();
    s.feed_cmd(mv(1, 2), &dex);
    assert_eq!(
        since(&s, 1, mark),
        vec!["|error|[Invalid choice] Can't move: Your Suicune doesn't have a move 3".to_string()]
    );
}
