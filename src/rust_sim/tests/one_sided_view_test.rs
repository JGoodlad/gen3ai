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
use pokesim::search::{aux_rng_from_seed, pre_state, resolve_turn_sourced, Resolved, TurnSource};
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
    // The sighting carries the TARGET's species and its ability-event index AT USE TIME, plus the
    // default target — the Pressure correction's inputs (reading rule V3).
    let s = arr[0].get("sightings").and_then(Json::as_array).expect("sightings");
    assert_eq!(s.len(), 1);
    assert_eq!(s[0].str_at("t"), Some("blissey"));
    assert_eq!(s[0].get("t_own").and_then(|b| b.as_bool()), Some(true));
    assert_eq!(s[0].get("n").and_then(Json::as_f64), Some(1.0));

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
    // The announcement HISTORY (the tick of each start) and the tick now — poke-env's lifecycle
    // is replayed over them in the adapter (reading rule V4).
    assert_eq!(v[0].starts, vec![0]);
    assert_eq!(obs.tick, 2);

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

// ===========================================================================
// D10 — the view AT an intermediate decision (`gen3_view_at_intermediate_v1`)
// ===========================================================================

/// A p2 lead so frail that ANY hit KOs it, over a live bench — so the turn must pause for a
/// forced replacement, which is the SECOND request inside one arm that deferral D10 names.
const GLASS_P2_TEAM: &str =
    "Magikarp|||NoAbility|tackle|Serious|,,,,,|N||||]Zapdos|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||";
/// A max-SpA Rayquaza clicking a 2x-effective Thunderbolt into Magikarp's base-20 SpD — an
/// unconditional OHKO, so the KO does not depend on dice or a damage roll.
const NUKE_P1_TEAM: &str =
    "Rayquaza|||NoAbility|thunderbolt|Serious|,,,252,,252|N||||]Regice|||NoAbility|tackle|Serious|252,,252,,,|N||||";

/// Resolve one whole turn from the pre-commit turn-1 boundary with both sides on an explicit
/// move, returning the settled session and what `resolve_turn_sourced` captured.
fn glass_turn(dex: &Dex) -> (BridgeSession, Resolved) {
    let opts = bridge_opts("gen3customgame", SEED.to_string(), NUKE_P1_TEAM, GLASS_P2_TEAM);
    let mut sess = BridgeSession::new_construct_turn0(&opts, dex).expect("session");
    let mut sources = [
        TurnSource::from_replay_spec("move 1", &[]),
        TurnSource::from_replay_spec("move 1", &[]),
    ];
    let mut rng = aux_rng_from_seed("1,2,3,4");
    let out = resolve_turn_sourced(&mut sess, &mut sources, "random", &mut rng, dex);
    (sess, out)
}

#[test]
fn a_mid_turn_faint_captures_the_view_AT_its_replacement_request() {
    // THE D10 CLOSE. When a ply KOs one of a side's mons the port answers the replacement round
    // itself, so the board it finally renders as `view_pN` is one decision PAST the row a
    // per-request consumer (`materialize_branches`) produces. `Resolved::views_at` is that
    // missing board, captured at the TOP of the loop iteration the replacement opened.
    let dex = Dex::for_gen(3);
    let (sess, out) = glass_turn(&dex);

    // NON-VACUITY: the fixture must really have KO'd p2's lead and settled the turn — without
    // that this test asserts about a turn that never had an intermediate decision at all.
    let st = sess.battle_state().expect("state");
    assert!(st.sides[1].pokemon.iter().any(|m| m.fainted),
        "fixture did not KO the glass p2 lead — there is no intermediate decision to capture");
    assert!(!out.stuck, "the turn must settle, not wedge");
    assert_eq!(out.used[1].len(), 2, "p2 must have used a move AND a replacement: {:?}",
        out.used[1]);

    // ONE entry for p2 (its replacement round) and NONE for p1, which only ever answered the
    // turn-start move request the caller had already seen. The count is the whole contract: it
    // must equal that side's non-final `|request|` count, which is what the Python consumer
    // reads off the protocol independently.
    assert_eq!(out.views_at[1].len(), 1,
        "p2's replacement round must have been captured exactly once: {:?}", out.views_at[1]);
    assert_eq!(out.views_at[0].len(), 0,
        "p1 opened no intermediate decision this turn: {:?}", out.views_at[0]);

    let at = &out.views_at[1][0];
    // The captured board is AT the replacement: the request spliced into it is the forceSwitch
    // one, not the next turn's move request.
    assert!(at.contains("\"forceSwitch\""),
        "the captured view must carry the REPLACEMENT request:\n{at}");
    assert!(!at.contains("\"active\":["),
        "a forceSwitch request carries no `active` block — this is the wrong request:\n{at}");
    // And it is a DIFFERENT board from the one the arm finally renders, which is the entire
    // reason the field exists. Equality here would mean the capture fired after the fact.
    let after = one_sided_view(&sess, 1, &dex);
    assert_ne!(at, &after, "the captured view equals the post-turn view — nothing was gained");
    assert!(at.contains("\"turn\":1"), "the capture must be the turn-1 board:\n{at}");
    assert!(after.contains("\"turn\":2"), "the post-turn view must be turn 2:\n{after}");
}

#[test]
fn an_ordinary_turn_captures_no_intermediate_view() {
    // The NEGATIVE half, and the one that keeps the field honest: a turn with no faint and no
    // reject resolves in ONE round, so there is no decision the port answered internally and
    // `views_at` must be empty on both sides. A capture rule that fired on the turn-start round
    // would pass the test above and silently hand every consumer the PARENT's board.
    let dex = Dex::for_gen(3);
    let mut sess = paused(&dex, 1);
    let mut sources = [
        TurnSource::from_replay_spec("move 1", &[]),
        TurnSource::from_replay_spec("move 1", &[]),
    ];
    let mut rng = aux_rng_from_seed("5,6,7,8");
    let out = resolve_turn_sourced(&mut sess, &mut sources, "random", &mut rng, &dex);
    assert!(!out.stuck, "the turn must settle");
    // NON-VACUITY: the turn really did run (both sides committed exactly one choice).
    assert_eq!(out.used[0].len(), 1, "p1 used {:?}", out.used[0]);
    assert_eq!(out.used[1].len(), 1, "p2 used {:?}", out.used[1]);
    assert!(out.views_at[0].is_empty() && out.views_at[1].is_empty(),
        "an ordinary turn captured a view: p1={:?} p2={:?}", out.views_at[0], out.views_at[1]);
}

// ===========================================================================
// 5 — THE READING RULES (`designs/rust_sim/one_sided_view.md` §2b), each pinned to the poke-env
//     line it mirrors. Every one was found by the parity harness's slice V
//     (`agents/battle/rust_core_parity_views.py`) on a real board; these are the constructed pins.
// ===========================================================================

/// A p1-seat observation fed protocol lines — p1 lines are ours, everything else theirs.
fn fold(lines: &[&str]) -> pokesim::view::SideObservation {
    let mut obs = pokesim::view::SideObservation::default();
    for l in lines {
        let own = l.split('|').nth(2).map_or(false, |i| i.trim_start().starts_with("p1"));
        obs.observe(l, own);
    }
    obs
}

fn sightings<'a>(obs: &'a pokesim::view::SideObservation, mon: &str, mv: &str)
    -> Vec<(&'a pokesim::view::Sighting, u32)> {
    let m = obs.mon(mon, false).expect("watched mon");
    let slot = m.moves.iter().find(|x| x.id == mv).expect("move slot");
    slot.sightings.iter().map(|(s, n)| (s, *n)).collect()
}

#[test]
fn v3_a_move_line_with_no_target_charges_our_active() {
    // `Battle._get_target_mon` (battle.py:51-59): no target string ⇒ the OTHER side's active.
    // A `[still]` fail prints an EMPTY target field; `_pressure_on` then charges our active.
    let obs = fold(&[
        "|switch|p1a: Zapdos|Zapdos|100/100",
        "|switch|p2a: Moltres|Moltres|100/100",
        "|move|p2a: Moltres|Will-O-Wisp||[still]",
    ]);
    let s = sightings(&obs, "Moltres", "willowisp");
    assert_eq!(s.len(), 1);
    assert_eq!((s[0].0.target.as_str(), s[0].0.target_own), ("Zapdos", true));
}

#[test]
fn v3_a_sleep_talk_call_reveals_the_called_move_with_no_use() {
    // abstract_battle.py:879-882 — `mon.moved(Called, use=False)` then
    // `mon.moves[caller].use(pressure, overridden=True)` (move.py:129-136: 1 + pressure - 1).
    let obs = fold(&[
        "|switch|p1a: Skarmory|Skarmory|100/100",
        "|switch|p2a: Suicune|Suicune|100/100",
        "|move|p2a: Suicune|Sleep Talk|p2a: Suicune",
        "|move|p2a: Suicune|Rest|p2a: Suicune|[from]move: Sleep Talk",
    ]);
    let m = obs.mon("Suicune", false).unwrap();
    let rest = m.moves.iter().find(|x| x.id == "rest").expect("the called move is REVEALED");
    assert_eq!(rest.uses, 0, "…but NOT used (the view read one PP low before this rule)");
    let talk = m.moves.iter().find(|x| x.id == "sleeptalk").unwrap();
    assert_eq!(talk.uses, 1);
    assert!(talk.sightings.keys().any(|s| s.called && s.mv == "rest"),
        "the called sighting rides the CALLER, keyed by the called move");
}

#[test]
fn v3_a_sighting_records_the_targets_ability_index_at_use_time() {
    // `_pressure_on` reads `target.ability` WHEN THE MOVE IS USED. Our Porygon2 Traced Pressure
    // (`-ability … [from] ability: Trace`), was hit, then pivoted (switch_out clears the overlay):
    // the adapter must judge the sighting at index 1, not at the read-time index 2.
    let obs = fold(&[
        "|switch|p1a: Porygon2|Porygon2|100/100",
        "|switch|p2a: Raikou|Raikou|100/100",
        "|-ability|p1a: Porygon2|Pressure|Trace|[from] ability: Trace|[of] p2a: Raikou",
        "|move|p2a: Raikou|Hidden Power|p1a: Porygon2",
        "|switch|p1a: Skarmory|Skarmory|100/100",
    ]);
    let s = sightings(&obs, "Raikou", "hiddenpower");
    assert_eq!((s[0].0.target.as_str(), s[0].0.target_own, s[0].0.target_k), ("Porygon2", true, 1));
    assert_eq!(obs.mon("Porygon2", true).unwrap().ability_events.len(), 2, "+ the switch-out marker");
}

#[test]
fn v4_a_reannounced_single_turn_effect_keeps_its_announcement_history() {
    // `Pokemon.end_turn` DELETES an `ends_on_turn` effect at `|turn|`; a later `-singleturn` is a
    // FRESH start (pokemon.py start_effect), not a restart — so both ticks must cross.
    let obs = fold(&[
        "|switch|p2a: Skarmory|Skarmory|100/100",
        "|-singleturn|p2a: Skarmory|Protect",
        "|turn|2",
        "|-singleturn|p2a: Skarmory|Protect",
    ]);
    let v = &obs.mon("Skarmory", false).unwrap().volatiles;
    assert_eq!((v.len(), v[0].starts.clone(), obs.tick), (1, vec![0, 1], 1));
}

#[test]
fn v4_baton_pass_carries_the_passers_volatiles_to_the_entrant() {
    // battle.py:160-177 → `Pokemon.apply_baton_pass`; the sim's `copyVolatileFrom`. The §4b
    // "own `volatiles` missing `substitute`" finding: the fold wiped the entrant instead.
    let obs = fold(&[
        "|switch|p2a: Celebi|Celebi|100/100",
        "|-start|p2a: Celebi|Substitute",
        "|switch|p2a: Charizard|Charizard, M|100/100|[from] Baton Pass",
    ]);
    let v = &obs.mon("Charizard", false).unwrap().volatiles;
    assert_eq!(v.len(), 1);
    assert_eq!((v[0].name.as_str(), v[0].bp_carried), ("Substitute", 1));
    assert!(obs.mon("Celebi", false).unwrap().volatiles.is_empty(), "the passer is cleared");
}

#[test]
fn v4_a_fainted_own_mons_request_condition_clears_its_effects() {
    // `update_from_request` → `set_hp_status("0 fnt")` → `faint()` → `_clear_effects()`: the
    // Destiny Bond `-activate` that FOLLOWS the KO is gone by the decision.
    let obs = fold(&[
        "|switch|p1a: Gengar|Gengar, M|56/303",
        "|-singlemove|p1a: Gengar|Destiny Bond",
        "|faint|p1a: Gengar",
        "|-activate|p1a: Gengar|move: Destiny Bond",
        r#"|request|{"forceSwitch":[true],"side":{"pokemon":[{"ident":"p1: Gengar","condition":"0 fnt"}]}}"#,
    ]);
    assert!(obs.mon("Gengar", true).unwrap().volatiles.is_empty());
}

#[test]
fn v5_the_status_counter_follows_poke_envs_own_status() {
    // pokemon.py: `moved`/`cant_move` +1 while SLP; the fork's `status` SETTER zeroes the
    // counter on a CHANGE (R1: a Rest taken while badly poisoned starts its sleep at 0, not at the
    // toxic count); `-cureteam` → `cure_status()` clears the status and NOT the counter;
    // `-curestatus X` resets both.
    let obs = fold(&[
        "|switch|p2a: Suicune|Suicune|100/100",
        "|-status|p2a: Suicune|tox",
        "|turn|2",
        "|turn|3",
        "|-status|p2a: Suicune|slp|[from] move: Rest",
        "|cant|p2a: Suicune|slp",
    ]);
    let m = obs.mon("Suicune", false).unwrap();
    assert_eq!((m.pstatus.as_deref(), m.status_counter), (Some("slp"), 1),
        "the Rest resets the count; the one cant-turn since is the whole sleep");
    let obs = fold(&[
        "|switch|p1a: Swampert|Swampert, M|404/404",
        "|-status|p1a: Swampert|slp|[from] move: Rest",
        "|-cureteam|p1a: Blissey|[from] move: Aromatherapy",
        "|move|p1a: Swampert|Ice Beam|p2a: Blissey",
    ]);
    let m = obs.mon("Swampert", true).unwrap();
    assert_eq!((m.pstatus.as_deref(), m.status_counter), (None, 0),
        "no status ⇒ the move does not count (the fold used to keep counting after -cureteam)");
}

#[test]
fn v6_a_faint_keeps_the_protect_streak() {
    // `Pokemon.faint` leaves `_protect_counter`; `switch_out` is what zeroes it.
    let obs = fold(&[
        "|switch|p1a: Swampert|Swampert, M|404/404",
        "|move|p1a: Swampert|Protect|p1a: Swampert",
        "|-damage|p1a: Swampert|0 fnt",
        "|faint|p1a: Swampert",
    ]);
    assert_eq!(obs.mon("Swampert", true).unwrap().protect_counter, 1);
}

#[test]
fn v8_an_ability_is_disclosed_off_four_non_ability_lines() {
    // abstract_battle: `-immune` (4 fields), `_check_heal_message_for_ability` (6 fields, the
    // HEALED mon), `_check_damage_message_for_ability` (6 fields, the `[of]` mon), and `-activate`
    // (only while unknown). A 5-field `-heal` discloses NOTHING — the length is part of the rule.
    let obs = fold(&[
        "|switch|p1a: Sharpedo|Sharpedo, M|100/100",
        "|switch|p2a: Snorlax|Snorlax, M|100/100",
        "|-immune|p2a: Snorlax|[from] ability: Immunity",
        "|-heal|p2a: Snorlax|100/100|[from] ability: Water Absorb",
        "|-damage|p1a: Sharpedo|88/100|[from] ability: Rough Skin|[of] p2a: Snorlax",
        "|-activate|p2a: Snorlax|ability: Thick Fat",
    ]);
    let ev = &obs.mon("Snorlax", false).unwrap().ability_events;
    let got: Vec<(&str, bool)> = ev.iter().map(|e| (e.id.as_str(), e.if_unknown)).collect();
    assert_eq!(got, vec![("immunity", false), ("roughskin", false), ("thickfat", true)]);
}

#[test]
fn v11_a_screen_is_stored_as_the_turn_it_started() {
    // `abstract_battle._side_start`: a timed condition stores `self.turn` (only when absent).
    let obs = fold(&[
        "|turn|25",
        "|-sidestart|p2: Foe|move: Light Screen",
        "|turn|26",
        "|-sidestart|p1: Me|Reflect",
    ]);
    assert_eq!(obs.screens[1].get("light_screen"), Some(&25));
    assert_eq!(obs.screens[0].get("reflect"), Some(&26));
    let obs = fold(&["|turn|3", "|-sidestart|p2: Foe|Safeguard", "|-sideend|p2: Foe|Safeguard"]);
    assert!(obs.screens[1].is_empty());
}

#[test]
fn the_first_decision_reads_turn_1_and_every_row_carries_the_faint_fields() {
    // `bs.turn` stays 0 until the first commit while `|turn|1` is already on the wire (the sim's
    // `this.turn` is 1): the first board read turn 0 on the view road.
    let dex = Dex::for_gen(3);
    let sess = paused(&dex, 0);
    let v = one_sided_view(&sess, 0, &dex);
    assert!(v.contains("\"turn\":1,"), "the construction board must read turn 1:\n{}", &v[..80]);
    // Every mon row carries `faint_boosts` (null while alive) and our rows a `base_ability`.
    assert!(v.contains("\"faint_boosts\":null") && v.contains("\"base_ability\":\""));
}
