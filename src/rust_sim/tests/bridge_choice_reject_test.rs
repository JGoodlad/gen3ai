//! `gen3_choice_reject_framing_v1` — the CHOICE-REJECT byte forms.
//!
//! The bridge used to model exactly ONE reject class (the trapped switch): every other illegal
//! choice emitted no `|error|` at all and re-opened the boundary to BOTH sides. These pin the
//! forms MEASURED against the real sim by `harness/probe_choice_reject_framing.js`:
//!
//! | choice | error | re-request? | other side |
//! |---|---|---|---|
//! | disabled move | `[Unavailable choice] Can't move: X's Y is disabled` | yes, that side only | 0 lines |
//! | switch -> ACTIVE | `[Invalid choice] Can't switch: You can't switch to an active Pokémon` | no | 0 lines |
//! | out-of-range move | `[Invalid choice] Can't move: Your X doesn't have a move N` | no | 0 lines |
//!
//! THE RULE these encode: `Side.emitChoiceError` emits `[Unavailable choice]` + re-issues the
//! request IFF its update callback actually CHANGED the request; with nothing to change it is
//! `[Invalid choice]` and nothing follows. So the assertions below check the CONDITION (did a
//! re-request follow?) rather than just the message text — a fix that got the message right and
//! the re-request wrong would still be a desync.
//!
//! EVERY test also asserts the NON-offending side received ZERO new chunks. That half is not
//! decoration: the port's old behaviour re-opened the whole boundary, which both duplicated a
//! request the sim never sends and made the other side record a phantom extra pick (the
//! `choices_used` divergence the replay parity harness used to allowlist).

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{BridgeSession, Cmd, WireChoice};
use pokesim::dex::Dex;

/// A Choice-Band Aerodactyl: once it moves, the Choice lock marks its other slots `disabled`,
/// which is the search harness's real case. Snorlax opposite so nothing faints early — a lead
/// that dies inside the prefix turns the boundary under test into a forced replacement and the
/// test silently stops testing the thing it names.
const P1: &str = "Aerodactyl||choiceband|rockhead|doubleedge,earthquake,rockslide,substitute|||||100|]\
                  Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|";
const P2: &str = "Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|]\
                  Blissey||leftovers|naturalcure|seismictoss,softboiled,toxic,icebeam|||||100|";

fn opts() -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("7,11,13,17".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(P1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(P2.to_string()) },
    }
}

/// Chunks emitted to `side` since `from`, flattened to lines.
fn since(sess: &BridgeSession, side: usize, from: usize) -> Vec<String> {
    sess.chunks()
        .chunks
        .iter()
        .skip(from)
        .filter(|c| c.side == side)
        .flat_map(|c| c.lines.iter().cloned())
        .collect()
}

fn mv(side: usize, slot: usize) -> Cmd {
    Cmd { side, choice: WireChoice::Move(slot) }
}
fn sw(side: usize, slot: usize) -> Cmd {
    Cmd { side, choice: WireChoice::Switch(slot) }
}

/// Drive to a fresh move boundary with p1's Choice lock ARMED (it has moved once).
fn locked_session(dex: &Dex) -> BridgeSession {
    let mut s = BridgeSession::new_construct_turn0(&opts(), dex).expect("session");
    s.feed_cmd(mv(0, 0), dex); // Double-Edge — arms the Choice lock
    s.feed_cmd(mv(1, 0), dex);
    assert!(!s.is_ended(), "fixture faulted: the battle ended before the reject boundary");
    s
}

#[test]
fn a_disabled_move_is_unavailable_and_re_requests_that_side_only() {
    let dex = Dex::for_gen(3);
    let mut s = locked_session(&dex);

    // NON-VACUITY: the slot we are about to feed must really be disabled, or this test passes
    // while exercising nothing.
    let req = s.active_request_json(0).expect("p1 request").to_string();
    assert!(
        req.contains("\"id\":\"earthquake\",\"pp\":16,\"maxpp\":16,\"target\":\"allAdjacent\",\"disabled\":true"),
        "fixture: Earthquake is not disabled in p1's request — the Choice lock did not arm.\n{req}"
    );

    let mark = s.chunks().chunks.len();
    s.feed_cmd(mv(0, 1), &dex); // Earthquake — refused by the Choice lock

    let p1 = since(&s, 0, mark);
    let p2 = since(&s, 1, mark);

    assert_eq!(
        p1.first().map(String::as_str),
        Some("|error|[Unavailable choice] Can't move: Aerodactyl's Earthquake is disabled"),
        "p1 lines: {p1:#?}"
    );
    // The class that DOES re-issue — and the re-request carries BOTH deltas the sim applies.
    let rereq = p1.get(1).expect("a re-request must follow an Unavailable reject");
    assert!(rereq.starts_with("|request|"), "expected a re-request, got {rereq}");
    assert!(rereq.contains("\"update\":true"), "re-request lacks the update flag: {rereq}");
    assert!(
        rereq.contains("\"id\":\"earthquake\"") && rereq.contains("\"disabledSource\":\"\""),
        "re-request lacks disabledSource on the refused slot: {rereq}"
    );
    assert!(p2.is_empty(), "the NON-offending side must receive nothing, got {p2:#?}");
}

#[test]
fn a_switch_into_the_active_is_invalid_with_no_re_request() {
    let dex = Dex::for_gen(3);
    let mut s = locked_session(&dex);
    let mark = s.chunks().chunks.len();
    s.feed_cmd(sw(0, 0), &dex); // slot 0 IS the active

    let p1 = since(&s, 0, mark);
    let p2 = since(&s, 1, mark);
    assert_eq!(
        p1,
        vec!["|error|[Invalid choice] Can't switch: You can't switch to an active Pokémon"],
        "an Invalid reject must emit the error and NOTHING else"
    );
    assert!(p2.is_empty(), "the NON-offending side must receive nothing, got {p2:#?}");
}

#[test]
fn an_out_of_range_move_slot_is_invalid_with_no_re_request() {
    let dex = Dex::for_gen(3);
    let mut s = locked_session(&dex);
    let mark = s.chunks().chunks.len();
    s.feed_cmd(mv(0, 8), &dex); // slot 9 on the wire

    let p1 = since(&s, 0, mark);
    let p2 = since(&s, 1, mark);
    assert_eq!(
        p1,
        vec!["|error|[Invalid choice] Can't move: Your Aerodactyl doesn't have a move 9"],
        "the reported slot must be the one the CLIENT sent, not the internal fallback"
    );
    assert!(p2.is_empty(), "the NON-offending side must receive nothing, got {p2:#?}");
}

/// A FORCED STRUGGLE is a SUBSTITUTION, not a refusal — the sim's request offers only Struggle
/// and `side.choose` swaps the pick, emitting nothing. Classifying it as a disabled-move reject
/// made the incremental bridge emit an `|error|` the genesis reference never sends; this is the
/// regression guard for that (it was caught by `bridge_test`'s genesis-parity check on the
/// `taunt_struggle` scenario, which exists because that wedge shipped once before).
#[test]
fn a_forced_struggle_substitutes_rather_than_rejecting() {
    let dex = Dex::for_gen(3);
    // A single-move mon drained to 0 PP is the cheapest forced-Struggle board.
    let p1 = "Shuckle||leftovers|sturdy|bide|||||100|";
    let p2 = "Blissey||leftovers|naturalcure|softboiled,softboiled,softboiled,softboiled|||||100|";
    let o = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("7,11,13,17".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    };
    let mut s = match BridgeSession::new_construct_turn0(&o, &dex) {
        Ok(s) => s,
        // Bide is a fail-loud unmodeled move on some trees; skip rather than assert a build.
        Err(_) => return,
    };
    // Burn the single slot's PP down. Once `must_struggle` holds, feeding that slot must NOT
    // produce an `|error|` — the engine substitutes Struggle.
    for _ in 0..40 {
        if s.is_ended() || s.fatal().is_some() {
            break;
        }
        let mark = s.chunks().chunks.len();
        s.feed_cmd(mv(0, 0), &dex);
        s.feed_cmd(mv(1, 0), &dex);
        let emitted = since(&s, 0, mark);
        assert!(
            !emitted.iter().any(|l| l.starts_with("|error|")),
            "a forced Struggle must never emit a reject error, got {emitted:#?}"
        );
    }
}

// ---------------------------------------------------------------------------------------------
// `gen3_locked_choice_never_rejected_v1` — the request may not offer what the classifier refuses.
//
// This one is NOT a framing question, which is why it survived `gen3_choice_reject_framing_v1`
// above and then killed two production launches at ~8 minutes. It is rust contradicting ITSELF:
//
//   * `build_request` branches on `move_locked()` and emits a SINGLE entry with `trapped:true`
//     and no `pp` / `disabled` key — the sim's hardLocked shape. The request offers ONE action.
//   * `classify_reject`'s `Choice::Move` arm fell through to `move_disabled` -> `!move_usable`,
//     which models the Choice lock, Disable, Encore, Taunt and PP and knows NOTHING about
//     `two_turn` / `must_recharge`. So it could refuse that one offered action.
//
// poke-env then re-picks from the same single-entry request, sends the same token, and
// `REJECT_STREAK_CAP` fires `__ERR__` — which is not an in-band error: it retires the reader,
// trips `_signal_transport_dead()` and raises in every in-flight `step()`. Observed as
// "9 consecutive rejects of MoveName(\"solarbeam\")".
//
// WHY THE EXISTING FUZZ CANNOT CATCH THIS CLASS: `bridge_session_fuzz_test` drives only
// masked-LEGAL tokens — and here the token IS masked-legal, because the mask is built FROM the
// request. A fuzz that trusts the mask can never see a request/classifier contradiction. 22k
// episodes passed clean while this was live. So the assertion below is the INVARIANT itself
// (offered => accepted), not a scenario that hopes to stumble into it.

/// Two-turn charge user opposite a Disable carrier, so Disable can land on the charging slot.
const LOCK_P1: &str = "Venusaur||leftovers|overgrow|solarbeam,sludgebomb,leechseed,sleeppowder|||||100|]\
                       Snorlax||leftovers|immunity|bodyslam,earthquake,rest,curse|||||100|";
const LOCK_P2: &str = "Umbreon||leftovers|synchronize|disable,bodyslam,toxic,rest|||||100|]\
                       Blissey||leftovers|naturalcure|seismictoss,softboiled,toxic,icebeam|||||100|";

fn lock_opts() -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("7,11,13,17".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(LOCK_P1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(LOCK_P2.to_string()) },
    }
}

#[test]
fn a_move_locked_mon_is_never_rejected_for_its_only_offered_move() {
    let dex = Dex::for_gen(3);
    let mut s = BridgeSession::new_construct_turn0(&lock_opts(), &dex).expect("session");
    // Turn 1: Venusaur starts charging Solar Beam; Umbreon Disables.
    s.feed_cmd(mv(0, 0), &dex);
    s.feed_cmd(mv(1, 0), &dex);
    assert!(!s.is_ended(), "fixture faulted: battle ended before the locked boundary");

    // Turn 2: the mon is LOCKED, so its request offers exactly one move. Send it.
    let before = s.chunks().chunks.len();
    s.feed_cmd(mv(0, 0), &dex);
    s.feed_cmd(mv(1, 1), &dex);
    let p1 = since(&s, 0, before).join("\n");

    assert!(
        !p1.contains("[Unavailable choice]") && !p1.contains("[Invalid choice]"),
        "a move-LOCKED mon was rejected for the ONLY move its own request offered — the \
         request/classifier contradiction that wedges a whole run:\n{p1}"
    );
    assert!(s.fatal().is_none(), "the locked boundary raised __ERR__: {:?}", s.fatal());
}
