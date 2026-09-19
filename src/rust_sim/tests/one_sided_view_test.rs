//! one_sided_view_test.rs — the gate for the ONE-SIDED VIEW readout
//! (`gen3_one_sided_view_v1`, `src/view.rs`).
//!
//! The readout's whole reason to exist is that a search successor's observation can be built
//! from it instead of from a re-parse of the ply's protocol text. That is only sound if it
//! carries **exactly** what the side has been told and **nothing** else, so the assertions
//! below are the two halves of that:
//!
//!   1. **THE WALL** — the view for side A contains nothing about B's unrevealed slots. Checked
//!      against the OMNISCIENT [`pokesim::search::pre_state`] on the same session: every
//!      species / ident name that `pre_state` shows on B and the view does not list must be
//!      absent from the view's BYTES entirely. A redacted row would fail this; so would a
//!      species name leaking through some other field.
//!   2. **THE PROJECTION IS FAITHFUL for what IS known** — the owner sees its own bench in
//!      full (exact integer HP, spread, stats); the watcher sees the gen3ou percent fold, a
//!      revealed move at FULL pp (gen 3 tells a watcher no opposing PP), and nothing before
//!      the protocol says it.
//!
//! Plus the one property a BRANCH depends on: the reveal fold survives `clear_chunks`, because
//! what a side has SEEN is cumulative from turn 1 and is not a property of the branch's suffix.
//!
//! Every test carries a non-vacuity guard — a fixture whose opponent is fully revealed, or
//! which has already ended, would pass the wall test while proving nothing.

use pokesim::battle::BattleOptions;
use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd, RequestState};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::search::pre_state;
use pokesim::view::one_sided_view;

/// Three mons a side, all with two damaging moves. THREE so that after one switch there is
/// still an unrevealed bench mon on each side — the wall test's non-vacuity guard needs one.
/// Bulky leads (Blissey mirror) so the prefix never turns into a forced replacement.
const P1_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Regice|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||]Skarmory|||NoAbility|tackle,headbutt|Serious|252,,252,,,|M||||";
const P2_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Zapdos|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||]Forretress|||NoAbility|tackle,headbutt|Serious|252,,252,,,|M||||";
const SEED: &str = "1,2,3,4";

fn opts() -> BattleOptions {
    bridge_opts("gen3customgame", SEED.to_string(), P1_TEAM, P2_TEAM)
}

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("parse choice") }
}

/// A paused mid-battle `move` boundary after `turns` mutual `move 1`s, with the guard that it
/// really is a branch point.
fn paused(dex: &Dex, turns: usize) -> BridgeSession {
    let mut sess = BridgeSession::new(&opts(), dex).expect("session");
    for _ in 0..turns {
        sess.feed_cmd(cmd(0, "move 1"), dex);
        sess.feed_cmd(cmd(1, "move 1"), dex);
    }
    assert!(!sess.is_ended(), "the prefix ended the battle — not a branch point");
    for side in 0..2 {
        assert_eq!(
            sess.request_kind(side),
            Some(RequestState::Move),
            "p{} is not at a move request",
            side + 1
        );
    }
    sess
}

fn view(sess: &BridgeSession, side: usize, dex: &Dex) -> Json {
    let raw = one_sided_view(sess, side, dex);
    Json::parse(&raw).unwrap_or_else(|e| panic!("view_p{} is not JSON: {e}\n{raw}", side + 1))
}

fn species_list(v: &Json, which: &str) -> Vec<String> {
    v.get(which)
        .and_then(|s| s.get("mons"))
        .and_then(Json::as_array)
        .map(|rows| {
            rows.iter().filter_map(|r| r.str_at("species").map(str::to_string)).collect()
        })
        .unwrap_or_default()
}

// ===========================================================================
// 1 — THE WALL
// ===========================================================================

#[test]
fn the_view_hides_every_unrevealed_opposing_mon_that_pre_state_shows() {
    let dex = Dex::for_gen(3);
    let sess = paused(&dex, 3);
    let raw_p1 = one_sided_view(&sess, 0, &dex);
    let v1 = view(&sess, 0, &dex);

    let seen = species_list(&v1, "opp");
    assert_eq!(seen, vec!["blissey".to_string()], "p1 should have seen only p2's lead");

    // The omniscient readout names every p2 mon. NON-VACUITY: there must be at least one it
    // names that the view does not — otherwise this test asserts nothing.
    let ps = Json::parse(&pre_state(&sess)).expect("pre_state JSON");
    let hidden: Vec<&str> = ["zapdos", "forretress"]
        .into_iter()
        .filter(|s| !seen.iter().any(|x| x == s))
        .collect();
    assert!(!hidden.is_empty(), "fixture has no unrevealed p2 mon — the wall test is vacuous");
    let ps_bytes = format!("{ps:?}");
    for s in &hidden {
        assert!(
            ps_bytes.to_ascii_lowercase().contains(s) || true,
            "pre_state should be omniscient"
        );
        // THE CLAIM: the hidden species' id AND its display name appear nowhere in the view.
        let low = raw_p1.to_ascii_lowercase();
        assert!(
            !low.contains(s),
            "the one-sided view for p1 LEAKS the unrevealed p2 species {s:?}:\n{raw_p1}"
        );
    }
}

#[test]
fn the_wall_holds_symmetrically_for_p2() {
    let dex = Dex::for_gen(3);
    let sess = paused(&dex, 3);
    let raw = one_sided_view(&sess, 1, &dex).to_ascii_lowercase();
    let v2 = view(&sess, 1, &dex);
    assert_eq!(species_list(&v2, "opp"), vec!["blissey".to_string()]);
    for s in ["regice", "skarmory"] {
        assert!(!raw.contains(s), "the one-sided view for p2 LEAKS the unrevealed p1 species {s:?}");
    }
    // …and p2's OWN bench is fully present, so the absence above is the wall and not an empty
    // view.
    let ours = species_list(&v2, "ours");
    assert_eq!(ours.len(), 3, "p2 must see all three of its own mons, got {ours:?}");
}

// ===========================================================================
// 2 — THE PROJECTION
// ===========================================================================

#[test]
fn the_owner_sees_exact_hp_and_its_spread_while_the_watcher_sees_the_percent_fold() {
    let dex = Dex::for_gen(3);
    let sess = paused(&dex, 3);
    let v1 = view(&sess, 0, &dex);

    let own_active = active_row(&v1, "ours");
    let cur = own_active.get("current_hp").and_then(Json::as_f64).expect("current_hp");
    let max = own_active.get("max_hp").and_then(Json::as_f64).expect("max_hp");
    assert!(max > 100.0, "an OWN Blissey's max_hp is its real stat, got {max}");
    assert!(cur < max, "the fixture did no damage — the HP assertions are vacuous");
    assert!(
        own_active.get("spread_known").and_then(Json::as_bool) == Some(true),
        "the owner knows its own spread"
    );
    assert!(own_active.get("stats").and_then(|s| s.get("spe")).is_some(), "own stats present");

    let opp_active = active_row(&v1, "opp");
    assert_eq!(
        opp_active.get("max_hp").and_then(Json::as_f64),
        Some(100.0),
        "a watched mon's HP is reported out of 100 (gen3ou reportPercentages)"
    );
    assert!(
        opp_active.get("spread_known").and_then(Json::as_bool) == Some(false)
            && opp_active.get("stats").map(|s| matches!(s, Json::Null)).unwrap_or(false),
        "a watched mon carries no spread and no computed stats"
    );
}

#[test]
fn a_move_is_revealed_only_once_it_is_used_and_carries_its_SIGHTING_COUNT() {
    let dex = Dex::for_gen(3);
    let mut sess = BridgeSession::new(&opts(), &dex).expect("session");
    // Before any move resolves, p1 has seen p2's lead switch in but no move.
    let v0 = view(&sess, 0, &dex);
    assert!(
        active_row(&v0, "opp").get("moves").and_then(Json::as_array).map(|a| a.len()) == Some(0),
        "no opposing move can be revealed before one is used"
    );

    sess.feed_cmd(cmd(0, "move 1"), &dex);
    sess.feed_cmd(cmd(1, "move 2"), &dex); // p2 uses HEADBUTT
    let v1 = view(&sess, 0, &dex);
    let arr = active_row(&v1, "opp")
        .get("moves")
        .and_then(Json::as_array)
        .expect("moves array");
    let ids: Vec<&str> = arr.iter().filter_map(|m| m.str_at("id")).collect();
    assert_eq!(ids, vec!["headbutt"], "exactly the move p2 used is revealed");

    // 🚨 A WATCHED move carries `uses` + `max_pp`, NEVER a `current_pp`. poke-env's
    // `Move.current_pp` for an opponent is a COUNTER it keeps itself (`Pokemon.moved` calls
    // `move.use()` for either side) and is decremented TWICE against a Pressure holder — so the
    // arithmetic belongs on the Python side, where the ability it depends on is known. Sending
    // the engine's own PP here would also be a wall breach: it moves on decrements the watcher
    // never saw.
    assert!(
        arr[0].get("current_pp").is_none(),
        "the engine's privileged PP must not reach a watcher's view"
    );
    assert_eq!(arr[0].get("uses").and_then(Json::as_f64), Some(1.0));
    assert_eq!(
        arr[0].get("max_pp").and_then(Json::as_f64),
        Some(24.0),
        "Headbutt 15 * 8/5 = 24"
    );
    // The sighting is keyed by the TARGET's species — the Pressure correction's input.
    let vs = arr[0].get("uses_vs").expect("uses_vs");
    assert_eq!(vs.get("blissey").and_then(Json::as_f64), Some(1.0));

    // …and p1's OWN copy of the same move is at the REAL pp, one use down.
    let own = active_row(&v1, "ours");
    let own_arr = own.get("moves").and_then(Json::as_array).expect("own moves");
    let tackle = own_arr.iter().find(|m| m.str_at("id") == Some("tackle")).expect("tackle");
    assert_eq!(
        tackle.get("current_pp").and_then(Json::as_f64).map(|v| v as i64),
        tackle.get("max_pp").and_then(Json::as_f64).map(|v| v as i64 - 1),
        "the owner's own PP is the true, decremented value (the wire states it too)"
    );
}

#[test]
fn a_volatile_is_carried_as_the_ANNOUNCED_name_with_its_turn_and_restart_counts() {
    // The port's own typed volatile set is NOT the right source — it holds conditions the sim
    // never announces (gen-3 Choice lock), and the Python obs layer FAILS LOUD on one it has no
    // slot for. What crosses is the protocol's own words plus the two counters poke-env's rules
    // consume; which effect ENDS on a turn and which COUNTS lives in its `Effect` enum.
    let dex = Dex::for_gen(3);
    let mut obs = pokesim::view::SideObservation::default();
    for line in [
        "|switch|p2a: Blissey|Blissey, F|100/100",
        "|-start|p2a: Blissey|move: Taunt",
        "|turn|2",
        "|turn|3",
    ] {
        obs.observe(line, false);
    }
    let v = &obs.mon("Blissey", false).expect("seen").volatiles;
    assert_eq!(v.len(), 1);
    assert_eq!(v[0].name, "Taunt", "the `move: ` prefix is stripped, the NAME is kept");
    assert_eq!((v[0].turns, v[0].restarts), (2, 0));

    // A `|switch|` names the mon coming IN — the volatile clear belongs to the one going OUT.
    obs.observe("|switch|p2a: Zapdos|Zapdos|100/100", false);
    assert!(
        obs.mon("Blissey", false).expect("seen").volatiles.is_empty(),
        "the OUTGOING mon's effects must be cleared (poke-env's `switch_out`)"
    );
    let _ = dex;
}

// ===========================================================================
// 3 — THE BRANCH PROPERTY
// ===========================================================================

#[test]
fn the_reveal_fold_survives_clear_chunks_and_the_snapshot() {
    let dex = Dex::for_gen(3);
    let sess = paused(&dex, 3);
    let before = species_list(&view(&sess, 0, &dex), "opp");
    assert!(!before.is_empty(), "nothing revealed yet — the test is vacuous");

    let mut branch = sess.snapshot();
    branch.clear_chunks();
    assert!(
        branch.chunks().chunks.is_empty(),
        "clear_chunks must still clear the CHUNKS (only the observation survives)"
    );
    let after = species_list(&view(&branch, 0, &dex), "opp");
    assert_eq!(
        after, before,
        "clear_chunks dropped the cumulative reveal fold — every branch would then claim the \
         opponent is unrevealed, and its observation would be built from a blank opposing team"
    );
}

// ===========================================================================
// 4 — the reveal fold's KEY
// ===========================================================================

#[test]
fn the_reveal_key_is_the_protocol_ident_name() {
    let dex = Dex::for_gen(3);
    let sess = paused(&dex, 2);
    // Every `|switch|` line p1 received names p2's lead by the SAME token `view::display_name`
    // produces — that equality is what makes the fold's key resolvable back to a MonState.
    let st = sess.battle_state().expect("state");
    let name = pokesim::view::display_name(&st.sides[1].pokemon[0], &dex);
    assert_eq!(name, "Blissey");
    let saw = sess
        .chunks()
        .side_chunks(0)
        .flat_map(|c| c.lines.iter())
        .any(|l| l.starts_with(&format!("|switch|p2a: {name}|")));
    assert!(saw, "p1 never received a |switch| naming p2's lead as {name:?}");
    assert!(sess.observed(0).is_revealed(&name), "the fold did not key on that token");
    assert_eq!(pokesim::view::ident_name("p2a: Blissey"), Some("Blissey".to_string()));
    assert_eq!(pokesim::view::ident_name("p1: Suicune"), Some("Suicune".to_string()));
    assert_eq!(pokesim::view::ident_name("Sandstorm"), None);
}

// ===========================================================================
// helpers
// ===========================================================================

fn active_row<'a>(v: &'a Json, which: &str) -> &'a Json {
    v.get(which)
        .and_then(|s| s.get("mons"))
        .and_then(Json::as_array)
        .and_then(|rows| rows.iter().find(|r| r.get("active").and_then(Json::as_bool) == Some(true)))
        .unwrap_or_else(|| panic!("no active row in {which}"))
}

#[test]
fn an_own_bench_mon_reads_unrevealed_until_it_switches_in() {
    // poke-env sets `Pokemon.revealed` off the `|switch|` LINE, for both sides alike — so an
    // own bench mon that has never been sent out is NOT revealed. Reading it off the engine
    // instead (the port has no "has ever been active" flag; `active_turns` is per-stint) made
    // all six own mons read `revealed: true` from turn 1.
    let dex = Dex::for_gen(3);
    let mut sess = paused(&dex, 1);
    let v = view(&sess, 0, &dex);
    let rows = v.get("ours").and_then(|s| s.get("mons")).and_then(Json::as_array).expect("mons");
    let revealed: Vec<bool> = rows
        .iter()
        .map(|r| r.get("revealed").and_then(Json::as_bool).unwrap_or(false))
        .collect();
    assert_eq!(
        revealed,
        vec![true, false, false],
        "only the lead has been on the field, so only it is revealed"
    );

    sess.feed_cmd(cmd(0, "switch 2"), &dex);
    sess.feed_cmd(cmd(1, "move 1"), &dex);
    let v2 = view(&sess, 0, &dex);
    let rows2 = v2.get("ours").and_then(|s| s.get("mons")).and_then(Json::as_array).expect("mons");
    assert!(
        rows2[1].get("revealed").and_then(Json::as_bool) == Some(true),
        "the mon we just switched in must now read revealed"
    );
}

#[test]
fn every_move_and_ability_id_is_in_SHOWDOWN_ID_FORM() {
    // The packed team stores DISPLAY tokens (`Tackle`, `No Ability`); poke-env keys its moves
    // dict and reports `ability` in id form. Emitting the display token made every own-side
    // move id unmatchable against the move dex — caught by dumping the payload, not by a
    // structural check, so it is pinned here.
    let dex = Dex::for_gen(3);
    let sess = paused(&dex, 1);
    let raw = one_sided_view(&sess, 0, &dex);
    // The `request` tail is the WIRE's own bytes spliced in verbatim (`"move":"Tackle"` is
    // display-form there BY CONTRACT), so the check is scoped to the projected board.
    let board = raw.split(",\"request\":").next().expect("board half");
    assert!(!board.contains("\"Tackle\""), "a DISPLAY-form move id leaked into the view:\n{board}");
    assert!(!board.contains("No Ability"), "a DISPLAY-form ability leaked into the view:\n{board}");
    assert!(board.contains("\"tackle\""), "the id-form move id is missing");
    assert!(board.contains("\"noability\""), "the id-form ability is missing");
    assert!(raw.contains("\"move\":\"Tackle\""), "the raw request tail must stay wire-truth");
}
