//! The `|request|` "maybe" display flags (`gen3_known_type_maybe_trap_v1` +
//! `gen3_imprison_maybe_flags_v1`) — pins for the two M6 cutover-stress request divergences.
//!
//! Every expected value below is the REAL sim's, measured by `harness/probe_maybe_flags.js`
//! (fail-loud; the case ids K*/S*/I*/R* match its rows). Each test drives the SAME teams, seed and
//! choices through the port's `BridgeSession` (the `sim_bridge` engine path) and compares the
//! `active[0]` block's trailing flag keys — the bytes after the `moves` array.
//!
//! - `maybeTrapped` is WIDER than the real trap: endTurn's `MaybeTrapPokemon` fold treats a mon
//!   TRANSFORMED into its foe as type-UNKNOWN (`knownType` false), so Magnet Pull drops its Steel
//!   gate and Arena Trap its Flying gate; and in an `obtainableabilities` format a foe whose
//!   SPECIES could hold Arena Trap / Shadow Tag raises it even when it does not (a Sand Veil
//!   Dugtrio). poke-env READS this key (`LegalActions.maybe_trapped` → the observation's
//!   maybe-trapped bit), so these pins guard a training input. The switch stays LEGAL.
//! - Imprison raises `maybeDisabled` + `maybeLocked` on EVERY live foe of the holder, shared move
//!   or not; a refused imprisoned MOVE re-requests WITHOUT `maybeLocked`, a refused trapped SWITCH
//!   re-requests WITH it.

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

/// Drive `turns` (p1 then p2 each) and return the live session. Fails loud if the battle ends
/// early — a fixture whose boundary under test never opens tests nothing.
fn drive(format: &str, p1: &str, p2: &str, turns: &[(Cmd, Cmd)], dex: &Dex) -> BridgeSession {
    let mut s = BridgeSession::new_construct_turn0(&opts(format, p1, p2), dex).expect("session");
    for (a, b) in turns {
        s.feed_cmd(a.clone(), dex);
        s.feed_cmd(b.clone(), dex);
        assert!(!s.is_ended(), "fixture faulted: the battle ended inside the prefix");
    }
    s
}

/// The `active[0]` block's keys AFTER its `moves` array, e.g. `,"maybeTrapped":true`.
fn active_flags(req: &str) -> String {
    let start = req.find("\"active\":[{\"moves\":[").expect("a move request") + "\"active\":[".len();
    let end = req[start..].find("],\"side\"").expect("the active block closes before side") + start;
    let active = &req[start..end]; // `{"moves":[...]<flags>}`
    let moves_end = active.find("}],").or_else(|| active.find("}]}")).expect("moves array end") + 2;
    active[moves_end..active.len() - 1].to_string()
}

fn p_request(s: &BridgeSession, side: usize) -> String {
    s.active_request_json(side).expect("an open request").to_string()
}

/// Lines emitted to `side` since chunk index `from`.
fn since(s: &BridgeSession, side: usize, from: usize) -> Vec<String> {
    s.chunks()
        .chunks
        .iter()
        .skip(from)
        .filter(|c| c.side == side)
        .flat_map(|c| c.lines.iter().cloned())
        .collect()
}

/// Feed `cmd` for `side` and report whether it drew an `|error|` (a REFUSED choice).
fn refused(s: &mut BridgeSession, cmd: Cmd, dex: &Dex) -> bool {
    let side = cmd.side;
    let mark = s.chunks().chunks.len();
    s.feed_cmd(cmd, dex);
    since(s, side, mark).iter().any(|l| l.starts_with("|error|"))
}

const SMEARGLE: &str = "Smeargle||Leftovers|OwnTempo|Transform,Splash|Jolly|,,,,,|M||||]\
                        Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";

/// K1 — THE cutover-stress repro's mechanic (`rmugytne6_bab_2_10`): a Smeargle TRANSFORMED into a
/// Water/Flying Gyarados faces a Magnet Pull Magneton. Not Steel ⇒ NOT trapped (the switch is
/// accepted), but `knownType` is false ⇒ gen-3 Magnet Pull's `onAnyMaybeTrapPokemon` sets
/// `maybeTrapped`. WRONG (pre-fix): the port tied `maybeTrapped` to the real trap and emitted no
/// flag. Reverting `is_maybe_trapped` or the Transform's `type_known = false` fails this.
#[test]
fn k1_transformed_into_a_flying_foe_is_maybe_trapped_by_magnet_pull() {
    let dex = Dex::for_gen(3);
    let p1 = "Gyarados||Leftovers|Intimidate|Splash|Jolly|,,,,,|M||||]\
              Magneton||Leftovers|MagnetPull|Splash|Modest|,,,,,|N||||";
    let mut s = drive("gen3ou", p1, SMEARGLE, &[(mv(0, 0), mv(1, 0)), (sw(0, 1), mv(1, 0))], &dex);
    let req = p_request(&s, 1);
    assert!(req.contains("\"id\":\"splash\""), "fixture: the Transform did not land:\n{req}");
    assert_eq!(active_flags(&req), ",\"maybeTrapped\":true", "sim K1: maybeTrapped\n{req}");
    assert!(!refused(&mut s, sw(1, 1), &dex), "sim K1: the switch is ACCEPTED (not really trapped)");
}

/// K2 — the Arena Trap half: `isGrounded(!knownType)` ignores the transformed mon's Flying type.
#[test]
fn k2_transformed_into_a_flying_foe_is_maybe_trapped_by_arena_trap() {
    let dex = Dex::for_gen(3);
    let p1 = "Gyarados||Leftovers|Intimidate|Splash|Jolly|,,,,,|M||||]\
              Dugtrio||Leftovers|ArenaTrap|Splash|Jolly|,,,,,|M||||";
    let mut s = drive("gen3ou", p1, SMEARGLE, &[(mv(0, 0), mv(1, 0)), (sw(0, 1), mv(1, 0))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), ",\"maybeTrapped\":true", "sim K2");
    assert!(!refused(&mut s, sw(1, 1), &dex), "sim K2: the switch is ACCEPTED");
}

/// K3 — the control: Levitate still escapes an unknown-type Arena Trap check (the negation only
/// drops the Flying clause), so NO flag. Guards an over-wide fix.
#[test]
fn k3_transformed_into_a_levitate_foe_is_not_maybe_trapped() {
    let dex = Dex::for_gen(3);
    let p1 = "Gengar||Leftovers|Levitate|Splash|Timid|,,,,,|M||||]\
              Dugtrio||Leftovers|ArenaTrap|Splash|Jolly|,,,,,|M||||";
    let s = drive("gen3ou", p1, SMEARGLE, &[(mv(0, 0), mv(1, 0)), (sw(0, 1), mv(1, 0))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), "", "sim K3: no flag");
}

/// K4 — the control for K1: an UNTRANSFORMED Gyarados (knownType true) vs Magnet Pull — no flag.
/// Without it K1 would pass on an engine that flagged every Magnet Pull foe.
#[test]
fn k4_untransformed_non_steel_vs_magnet_pull_is_not_maybe_trapped() {
    let dex = Dex::for_gen(3);
    let p1 = "Magneton||Leftovers|MagnetPull|Splash|Modest|,,,,,|N||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let p2 = "Gyarados||Leftovers|Intimidate|Splash|Jolly|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let s = drive("gen3ou", p1, p2, &[(mv(0, 0), mv(1, 0))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), "", "sim K4: no flag");
}

/// S1 — the "canceling switches would leak information" loop: a SAND VEIL Dugtrio still raises
/// `maybeTrapped` on a grounded foe in gen3ou (its species could hold Arena Trap); the switch is
/// accepted.
#[test]
fn s1_sand_veil_dugtrio_raises_maybe_trapped_in_gen3ou() {
    let dex = Dex::for_gen(3);
    let p1 = "Dugtrio||Leftovers|SandVeil|Splash|Jolly|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||]\
              Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
    let mut s = drive("gen3ou", p1, p2, &[(mv(0, 0), mv(1, 0))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), ",\"maybeTrapped\":true", "sim S1");
    assert!(!refused(&mut s, sw(1, 1), &dex), "sim S1: the switch is ACCEPTED");
}

/// S2 — the same board in gen3customgame: no `obtainableabilities` rule ⇒ the leak loop is
/// skipped ⇒ no flag. Pins the format gate.
#[test]
fn s2_sand_veil_dugtrio_raises_nothing_in_customgame() {
    let dex = Dex::for_gen(3);
    let p1 = "Dugtrio||Leftovers|SandVeil|Splash|Jolly|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let p2 = "Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||]\
              Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
    let s = drive("gen3customgame", p1, p2, &[(mv(0, 0), mv(1, 0))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), "", "sim S2: no flag");
}

/// S3 — the leak loop's Arena Trap still respects a KNOWN Flying type.
#[test]
fn s3_sand_veil_dugtrio_vs_a_flying_foe_raises_nothing() {
    let dex = Dex::for_gen(3);
    let p1 = "Dugtrio||Leftovers|SandVeil|Splash|Jolly|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let p2 = "Skarmory||Leftovers|KeenEye|Splash|Impish|,,,,,|M||||]\
              Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
    let s = drive("gen3ou", p1, p2, &[(mv(0, 0), mv(1, 0))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), "", "sim S3: no flag");
}

/// S4 — a STURDY Magneton vs a Steel foe raises nothing: the gen-3 mod sets Magnet Pull's
/// `onFoeMaybeTrapPokemon: undefined`, so Magnet Pull never enters the leak table.
#[test]
fn s4_sturdy_magneton_vs_steel_raises_nothing() {
    let dex = Dex::for_gen(3);
    let p1 = "Magneton||Leftovers|Sturdy|Splash|Modest|,,,,,|N||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let p2 = "Skarmory||Leftovers|KeenEye|Splash|Impish|,,,,,|M||||]\
              Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";
    let s = drive("gen3ou", p1, p2, &[(mv(0, 0), mv(1, 0))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), "", "sim S4: no flag");
}

/// I1 — THE cutover-stress repro's mechanic (`sbd_mugr6edd_b24`): the holder imprisoned a Suicune
/// (shared Rest), then a Celebi that shares NOTHING comes in. `onFoeDisableMove` sets
/// `maybeDisabled` unconditionally ⇒ the entrant's request carries both keys. WRONG (pre-fix): the
/// port required a shared move and emitted neither.
#[test]
fn i1_imprison_flags_a_foe_that_shares_no_move() {
    let dex = Dex::for_gen(3);
    let p1 = "Dusclops||Leftovers|Pressure|Imprison,Rest,Splash|Careful|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let p2 = "Suicune||Leftovers|Pressure|Rest,Splash|Calm|,,,,,|N||||]\
              Celebi||Leftovers|NaturalCure|Recover,LeechSeed|Bold|,,,,,|N||||";
    let s = drive("gen3ou", p1, p2, &[(mv(0, 0), mv(1, 1)), (mv(0, 2), sw(1, 1))], &dex);
    let req = p_request(&s, 1);
    assert!(req.contains("\"id\":\"recover\""), "fixture: Celebi is not active:\n{req}");
    assert_eq!(
        active_flags(&req),
        ",\"maybeDisabled\":true,\"maybeLocked\":true",
        "sim I1: both keys on a no-shared-move foe\n{req}"
    );
}

const IMPRISON_FOE: &str = "Jynx||Leftovers|Oblivious|IceBeam,Psychic,Splash|Hardy|,,,,,|F||||]\
                            Blissey||Leftovers|NaturalCure|Splash|Bold|,,,,,|F||||";

/// R1 — a refused IMPRISONED move re-requests WITHOUT `maybeLocked` (`updateDisabledRequest`
/// deletes it; `maybeDisabled` stays in singles). WRONG (pre-fix): the port keyed the drop on
/// `trapped_firm`, so this re-request kept `maybeLocked`.
#[test]
fn r1_a_refused_imprisoned_move_re_requests_without_maybe_locked() {
    let dex = Dex::for_gen(3);
    let p1 = "Gengar||Leftovers|Levitate|Imprison,IceBeam,Splash|Hardy|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let mut s = drive("gen3customgame", p1, IMPRISON_FOE, &[(mv(0, 0), mv(1, 2))], &dex);
    assert_eq!(active_flags(&p_request(&s, 1)), ",\"maybeDisabled\":true,\"maybeLocked\":true");
    s.feed_cmd(mv(0, 2), &dex); // p1 commits first, so p2's refused pick is the one under test
    let mark = s.chunks().chunks.len();
    s.feed_cmd(mv(1, 0), &dex); // Ice Beam — imprisoned
    let p2 = since(&s, 1, mark);
    assert_eq!(
        p2.first().map(String::as_str),
        Some("|error|[Unavailable choice] Can't move: Jynx's Ice Beam is disabled"),
        "p2 lines: {p2:#?}"
    );
    let rereq = p2.get(1).expect("an Unavailable reject re-requests");
    assert_eq!(active_flags(rereq), ",\"maybeDisabled\":true", "sim R1: maybeLocked dropped\n{rereq}");
}

/// R2 — a refused TRAPPED switch re-requests WITH `maybeLocked` (the trap closure touches only
/// `maybeTrapped`/`trapped`). WRONG (pre-fix): the port dropped it here.
#[test]
fn r2_a_refused_trapped_switch_re_requests_with_maybe_locked() {
    let dex = Dex::for_gen(3);
    let p1 = "Dugtrio||Leftovers|ArenaTrap|Imprison,IceBeam,Splash|Jolly|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let mut s = drive("gen3customgame", p1, IMPRISON_FOE, &[(mv(0, 0), mv(1, 2))], &dex);
    assert_eq!(
        active_flags(&p_request(&s, 1)),
        ",\"maybeDisabled\":true,\"maybeLocked\":true,\"maybeTrapped\":true",
        "sim R2 fresh (the trap flag is LAST)"
    );
    s.feed_cmd(mv(0, 2), &dex);
    let mark = s.chunks().chunks.len();
    s.feed_cmd(sw(1, 1), &dex);
    let p2 = since(&s, 1, mark);
    assert_eq!(
        p2.first().map(String::as_str),
        Some("|error|[Unavailable choice] Can't switch: The active Pokémon is trapped"),
        "p2 lines: {p2:#?}"
    );
    let rereq = p2.get(1).expect("a hidden-trap reject re-requests");
    assert_eq!(
        active_flags(rereq),
        ",\"maybeDisabled\":true,\"maybeLocked\":true,\"trapped\":true",
        "sim R2: maybeLocked kept, `trapped` appended LAST\n{rereq}"
    );
}

/// O1 — KEY ORDER on a FRESH request: a Mean Look (FIRM trap) + Imprison Dusclops — a real gen3ou
/// moveset — gives `maybeDisabled`, `maybeLocked`, THEN `trapped`. WRONG (pre-fix): the port
/// wrote the trap flag first. poke-env parses JSON, so this is a byte-parity pin, not a
/// training-input one.
#[test]
fn o1_mean_look_plus_imprison_orders_the_trap_flag_last() {
    let dex = Dex::for_gen(3);
    let p1 = "Dusclops||Leftovers|Pressure|MeanLook,Imprison,Rest|Careful|,,,,,|M||||]\
              Snorlax||Leftovers|Immunity|Splash|Adamant|,,,,,|M||||";
    let p2 = "Suicune||Leftovers|Pressure|Rest,Splash|Calm|,,,,,|N||||]\
              Celebi||Leftovers|NaturalCure|Recover,LeechSeed|Bold|,,,,,|N||||";
    let mut s = drive("gen3ou", p1, p2, &[(mv(0, 0), mv(1, 1)), (mv(0, 1), mv(1, 1))], &dex);
    assert_eq!(
        active_flags(&p_request(&s, 1)),
        ",\"maybeDisabled\":true,\"maybeLocked\":true,\"trapped\":true",
        "sim O1"
    );
    assert!(refused(&mut s, sw(1, 1), &dex), "sim O1: a Mean Look trap refuses the switch");
}
