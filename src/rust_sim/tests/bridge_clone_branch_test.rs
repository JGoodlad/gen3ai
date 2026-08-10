//! bridge_clone_branch_test.rs — the gate for the CLONE-AND-BRANCH search primitive
//! (`gen3_bridge_clone_branch_v1`): `BridgeSession::snapshot` + the read-only search API
//! (`clear_chunks` / `request_kind` / `is_choice_done` / `active_request_json` /
//! `battle_state` / `winner`).
//!
//! A search driver builds a TREE out of one paused battle: snapshot a boundary, advance
//! each branch with a different choice, score the leaves, throw the branches away. That is
//! only sound if a snapshot is a DEEP, independent copy — so the assertions below are the
//! two halves of that property, plus the bookkeeping a per-ply driver needs:
//!
//!   1. **DETERMINISM** — two snapshots advanced with the SAME choices emit byte-identical
//!      chunks, and so does the ORIGINAL advanced the same way. (A snapshot that shared
//!      the PRNG would still pass this by accident, hence 2.)
//!   2. **INDEPENDENCE** — two snapshots advanced with DIFFERENT choices leave the parent
//!      bit-for-bit untouched (turn, chunk stream, outstanding request), and a `reseed` on
//!      a clone does not perturb the parent's dice.
//!   3. **SUFFIX** — `clear_chunks` on a branch makes its chunk stream exactly the tail the
//!      un-cleared branch would have appended (the per-ply payload a search returns).
//!   4. **BOUNDARY BOOKKEEPING** — `request_kind` / `is_choice_done` / `active_request_json`
//!      report the sim's `requestState` / `isChoiceDone()` / `activeRequest`, INCLUDING
//!      across a rejected choice (which re-issues the request and holds the boundary open).
//!
//! Every non-vacuity guard matters here: a test that snapshots a session which is already
//! ENDED, or branches on two choices that happen to produce the same bytes, would pass
//! while proving nothing. Each test asserts the mechanic is genuinely exercised first.

use pokesim::battle::BattleOptions;
use pokesim::bridge::{bridge_opts, parse_choice, BridgeChunks, BridgeSession, Cmd, RequestState};
use pokesim::dex::Dex;

// ===========================================================================
// Fixtures — real seeded gen3 battles (the `bridge_test` conventions: explicit
// genders so no construction-time gender `sample` is needed, MODELED moves only,
// gen3customgame, a fixed seed → byte-reproducible).
// ===========================================================================

/// Two mons a side, TWO damaging moves each, so a branch can genuinely diverge (`move 1`
/// Tackle vs `move 2` Headbutt — different damage AND a 30% flinch roll) while every
/// advanced turn draws accuracy + crit + damage; a snapshot that shared its parent's PRNG
/// would show up immediately.
///
/// The leads are a BULKY MIRROR on purpose. A snapshot fixture has to sit at a plain `move`
/// boundary for many turns in a row: if a lead faints inside the prefix the pause becomes a
/// forced-replacement (`Wait` for the other side) and the branch assertions stop testing
/// what they claim. Max-HP/Def Blisseys hitting each other with a 35-BP Tackle need dozens
/// of turns to resolve, so every test below has ample headroom. The mirror also ties on
/// Speed every turn, which adds the action-order tie-shuffle draw to the stream — more dice
/// for the isolation assertions to catch.
const P1_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Regice|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||";
const P2_TEAM: &str = "Blissey|||NoAbility|tackle,headbutt|Serious|252,,252,,,|F||||]Zapdos|||NoAbility|tackle,headbutt|Serious|252,,252,,,|N||||";
const SEED: &str = "1,2,3,4";

/// The ARENA TRAP matchup from `bridge_trapping_golden.txt` — p2's Snorlax is trapped by
/// p1's Dugtrio, so a `switch 2` at a move boundary is REJECTED (`|error|[Unavailable
/// choice]` + a `trapped:true` re-request). The one board that exercises the reject path.
const TRAP_P1_TEAM: &str = "Dugtrio|||ArenaTrap|earthquake,splash|Serious||N||||";
const TRAP_P2_TEAM: &str =
    "Snorlax|||NoAbility|bodyslam,splash|Serious||M||||]Regice|||NoAbility|icebeam,splash|Serious||N||||";
const TRAP_SEED: &str = "44317,42357,9927,48760";

fn opts() -> BattleOptions {
    bridge_opts("gen3customgame", SEED.to_string(), P1_TEAM, P2_TEAM)
}

fn trap_opts() -> BattleOptions {
    bridge_opts("gen3customgame", TRAP_SEED.to_string(), TRAP_P1_TEAM, TRAP_P2_TEAM)
}

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("parse choice") }
}

/// A mutual `move <k1>` / `move <k2>` boundary, as the two CMDs the bridge consumes.
fn both_move(k1: usize, k2: usize) -> Vec<Cmd> {
    vec![cmd(0, &format!("move {k1}")), cmd(1, &format!("move {k2}"))]
}

/// The chunk stream as comparable strings — `p<N>#<chunkIdx>` + the chunk's lines. Compared
/// rather than `SideChunk` directly so a failure prints the diverging LINE, and so chunk
/// BOUNDARIES (not just the flattened line sequence) are part of the assertion.
fn dump(chunks: &BridgeChunks) -> Vec<String> {
    chunks
        .chunks
        .iter()
        .enumerate()
        .map(|(i, c)| format!("p{}#{i}\n{}", c.side + 1, c.lines.join("\n")))
        .collect()
}

fn assert_dumps_equal(what: &str, got: &[String], want: &[String]) {
    let n = got.len().min(want.len());
    for i in 0..n {
        assert!(
            got[i] == want[i],
            "[{what}] chunk {i} DIVERGES:\n  want: {}\n  got:  {}",
            want[i],
            got[i]
        );
    }
    assert!(
        got.len() == want.len(),
        "[{what}] chunk COUNT differs: want {} got {}",
        want.len(),
        got.len()
    );
}

/// Build a session and feed it `turns` mutual `move 1` boundaries, leaving it PAUSED at a
/// mid-battle `move` request. Asserts the pause is a real branch point (the battle is still
/// running, both sides are being asked for a move, and both have an outstanding request) —
/// without that guard every snapshot assertion below could pass vacuously on a finished game.
fn paused_mid_battle(dex: &Dex, turns: usize) -> BridgeSession {
    let mut sess = BridgeSession::new(&opts(), dex).expect("session");
    for _ in 0..turns {
        for c in both_move(1, 1) {
            sess.feed_cmd(c, dex);
        }
    }
    assert!(!sess.is_ended(), "prefix ended the battle — the fixture is not a branch point");
    assert!(sess.turn() > 1, "prefix did not advance past turn 1 (turn = {})", sess.turn());
    for side in 0..2 {
        assert!(
            sess.request_kind(side) == Some(RequestState::Move),
            "p{} is not at a move request: {:?}",
            side + 1,
            sess.request_kind(side)
        );
        assert!(!sess.is_choice_done(side), "p{} should still owe a choice", side + 1);
        assert!(sess.active_request_json(side).is_some(), "p{} has no outstanding request", side + 1);
    }
    sess
}

/// The 1-based wire `switch N` for the first LIVE, non-active mon in a request's
/// `side.pokemon` array (the `bridge_test` helper, read off `active_request_json` here).
fn first_live_bench_slot(request_line: &str) -> Option<usize> {
    let json = request_line.strip_prefix("|request|")?;
    let v = pokesim::json::Json::parse(json).ok()?;
    let pokemon = v.get("side")?.get("pokemon")?.as_array()?;
    for (i, p) in pokemon.iter().enumerate() {
        let active = p.get("active").and_then(|a| a.as_bool()).unwrap_or(false);
        let cond = p.str_at("condition").unwrap_or("");
        if !active && !cond.contains("fnt") {
            return Some(i + 1);
        }
    }
    None
}

fn advance(sess: &mut BridgeSession, cmds: &[Cmd], dex: &Dex) {
    for c in cmds {
        sess.feed_cmd(c.clone(), dex);
    }
}

// ===========================================================================
// 1. DETERMINISM — same snapshot, same choices, same bytes (and the parent agrees).
// ===========================================================================

#[test]
fn two_snapshots_advanced_identically_match_each_other_and_the_original() {
    let dex = Dex::for_gen(3);
    let mut parent = paused_mid_battle(&dex, 3);
    let before = dump(parent.chunks());

    let mut a = parent.snapshot();
    let mut b = parent.snapshot();

    // Three further boundaries, so the comparison spans several dice-consuming turns
    // rather than a single roll that could coincide by chance.
    let line: Vec<Cmd> = (0..3).flat_map(|_| both_move(1, 1)).collect();
    advance(&mut a, &line, &dex);
    advance(&mut b, &line, &dex);

    let da = dump(a.chunks());
    let db = dump(b.chunks());
    assert!(da.len() > before.len(), "clone A emitted nothing new — the advance did not happen");
    assert_dumps_equal("clone A vs clone B", &db, &da);

    // ...and the ORIGINAL, advanced the same way, produces the same stream. Done LAST
    // because it consumes the parent.
    advance(&mut parent, &line, &dex);
    assert_dumps_equal("clone A vs original", &da, &dump(parent.chunks()));
    assert!(a.turn() == parent.turn(), "clone/original turn differs");
    assert!(
        a.active_request_json(0) == parent.active_request_json(0),
        "clone/original outstanding p1 request differs"
    );
}

// ===========================================================================
// 2. INDEPENDENCE — diverging branches leave the parent untouched.
// ===========================================================================

#[test]
fn diverging_clones_leave_the_parent_bit_for_bit_unchanged() {
    let dex = Dex::for_gen(3);
    let parent = paused_mid_battle(&dex, 3);

    // The parent's exact pre-branch observables.
    let turn_before = parent.turn();
    let chunks_before = dump(parent.chunks());
    let req_before: Vec<Option<String>> =
        (0..2).map(|s| parent.active_request_json(s).map(str::to_string)).collect();
    let seed_before = parent.battle_state().expect("state").prng_seed();

    let mut a = parent.snapshot();
    let mut b = parent.snapshot();
    advance(&mut a, &both_move(1, 1), &dex); // X: Body Slam
    advance(&mut b, &both_move(2, 1), &dex); // Y: Earthquake

    // Non-vacuity: the two choices must actually produce different battles, else
    // "the parent is unchanged" would be trivially true of a no-op branch.
    assert!(
        dump(a.chunks()) != dump(b.chunks()),
        "branch X and branch Y produced identical chunks — the fixture does not diverge"
    );

    assert!(parent.turn() == turn_before, "parent turn moved: {turn_before} -> {}", parent.turn());
    assert_dumps_equal("parent chunk stream", &dump(parent.chunks()), &chunks_before);
    for s in 0..2 {
        assert!(
            parent.active_request_json(s).map(str::to_string) == req_before[s],
            "parent's outstanding p{} request changed",
            s + 1
        );
        assert!(!parent.is_choice_done(s), "parent's boundary was consumed by a branch");
    }
    assert!(
        parent.battle_state().expect("state").prng_seed() == seed_before,
        "parent's PRNG advanced — the clone shared its dice"
    );
}

// ===========================================================================
// 3. RESEED — a clone's dice swap cannot reach the parent's stream.
// ===========================================================================

#[test]
fn reseeding_a_clone_leaves_the_parents_dice_stream_untouched() {
    let dex = Dex::for_gen(3);
    let mut parent = paused_mid_battle(&dex, 3);
    // An independently-built session at the same point: the CONTROL the parent must still
    // match after a clone has been reseeded and advanced.
    let mut control = paused_mid_battle(&dex, 3);

    let mut branch = parent.snapshot();
    branch.reseed("sodium,deadbeef");

    let line: Vec<Cmd> = (0..3).flat_map(|_| both_move(1, 1)).collect();
    advance(&mut branch, &line, &dex);
    advance(&mut parent, &line, &dex);
    advance(&mut control, &line, &dex);

    assert_dumps_equal("parent vs control after a clone reseed", &dump(parent.chunks()), &dump(control.chunks()));
    // Non-vacuity: the reseed must actually have changed the branch's dice, else the
    // parent-vs-control equality proves nothing about isolation.
    assert!(
        dump(branch.chunks()) != dump(parent.chunks()),
        "the reseeded branch matched the parent — the reseed did not take"
    );
    assert!(
        parent.battle_state().expect("state").prng_seed()
            == control.battle_state().expect("state").prng_seed(),
        "parent's PRNG state diverged from the control"
    );
}

// ===========================================================================
// 4. SUFFIX — `clear_chunks` yields exactly the per-ply tail.
// ===========================================================================

#[test]
fn clear_chunks_makes_a_branch_emit_exactly_the_suffix() {
    let dex = Dex::for_gen(3);
    let parent = paused_mid_battle(&dex, 3);
    let prefix_len = parent.chunks().chunks.len();
    assert!(prefix_len > 0, "the prefix emitted no chunks — nothing to clear");

    let mut full = parent.snapshot();
    let mut cleared = parent.snapshot();
    cleared.clear_chunks();
    assert!(cleared.chunks().chunks.is_empty(), "clear_chunks left chunks behind");

    let line: Vec<Cmd> = (0..2).flat_map(|_| both_move(1, 1)).collect();
    advance(&mut full, &line, &dex);
    advance(&mut cleared, &line, &dex);

    let tail = dump(full.chunks())[prefix_len..].to_vec();
    assert!(!tail.is_empty(), "the advance emitted no new chunks — the suffix is vacuous");
    // The dump embeds the chunk INDEX, which differs between the two streams by construction
    // (the cleared one restarts at 0), so compare on the chunk CONTENT only.
    let strip = |v: &[String]| -> Vec<String> {
        v.iter().map(|s| s.splitn(2, '\n').nth(1).unwrap_or("").to_string()).collect()
    };
    assert_dumps_equal("cleared branch vs the full branch's tail", &strip(&dump(cleared.chunks())), &strip(&tail));

    // And clearing must NOT have disturbed the battle itself (the `prev_log_len` trap: a
    // reset log cursor would re-emit the whole battle's log into the next chunk).
    assert!(cleared.turn() == full.turn(), "clear_chunks changed the battle's turn");
    assert!(
        cleared.battle_state().expect("state").prng_seed()
            == full.battle_state().expect("state").prng_seed(),
        "clear_chunks perturbed the PRNG"
    );
}

// ===========================================================================
// 5. BOUNDARY BOOKKEEPING — requestState / isChoiceDone / activeRequest.
// ===========================================================================

#[test]
fn boundary_accessors_track_a_partially_answered_move_request() {
    let dex = Dex::for_gen(3);
    let mut sess = paused_mid_battle(&dex, 2);
    let p1_req = sess.active_request_json(0).expect("p1 request").to_string();

    // Answer p1 only: the boundary stays open, waiting on p2.
    sess.feed_cmd(cmd(0, "move 1"), &dex);
    assert!(sess.is_choice_done(0), "p1's accepted choice was not recorded");
    assert!(!sess.is_choice_done(1), "p2 should still owe a choice");
    assert!(sess.request_kind(1) == Some(RequestState::Move));
    assert!(
        sess.active_request_json(0).map(str::to_string) == Some(p1_req),
        "p1's outstanding request changed while the boundary was still open"
    );

    // Answer p2: the boundary COMMITS, so the next one opens and both owe a choice again.
    sess.feed_cmd(cmd(1, "move 1"), &dex);
    assert!(!sess.is_ended());
    for s in 0..2 {
        assert!(!sess.is_choice_done(s), "p{} should owe a choice at the new boundary", s + 1);
    }
    // The freshly-issued request is the last `|request|` line in that side's chunk stream —
    // i.e. `active_request_json` reports the bytes the WIRE carried, not a rebuild.
    for s in 0..2 {
        let last_on_wire = sess
            .chunks()
            .chunks
            .iter()
            .filter(|c| c.side == s)
            .flat_map(|c| c.lines.iter())
            .filter(|l| l.starts_with("|request|"))
            .next_back()
            .cloned()
            .expect("a request line");
        assert!(
            sess.active_request_json(s) == Some(last_on_wire.as_str()),
            "p{} active_request_json != the last |request| chunk line",
            s + 1
        );
    }
}

#[test]
fn a_rejected_switch_holds_the_boundary_open_and_reissues_the_request() {
    let dex = Dex::for_gen(3);
    let mut sess = BridgeSession::new(&trap_opts(), &dex).expect("session");
    let first = sess.active_request_json(1).expect("p2 request").to_string();
    // Non-vacuity: this must be the HIDDEN-trap shape (Arena Trap → `maybeTrapped`), which
    // is the one that re-issues on a reject. A FIRM trap would emit no re-request.
    assert!(first.contains("\"maybeTrapped\":true"), "p2's first request is not maybeTrapped: {first}");

    // p2 tries to switch out of the trap — rejected.
    sess.feed_cmd(cmd(1, "switch 2"), &dex);

    assert!(!sess.is_choice_done(1), "a REJECTED choice must not count as done");
    assert!(sess.request_kind(1) == Some(RequestState::Move), "the boundary must stay a move request");
    let reissued = sess.active_request_json(1).expect("p2 re-request").to_string();
    assert!(reissued != first, "the re-issued request is identical to the original");
    assert!(reissued.contains("\"trapped\":true"), "the re-request does not firm the trap: {reissued}");
    assert!(reissued.contains("\"update\":true"), "the re-request lacks the update flag: {reissued}");
    // It is the bytes that went out on the wire.
    let last_on_wire = sess
        .chunks()
        .chunks
        .iter()
        .filter(|c| c.side == 1)
        .flat_map(|c| c.lines.iter())
        .filter(|l| l.starts_with("|request|"))
        .next_back()
        .cloned()
        .expect("a request line");
    assert!(reissued == last_on_wire, "active_request_json != the re-request chunk line");

    // A snapshot taken across the reject resumes in the same rejected state — a search
    // kernel branching here must see "p2 still owes a choice, from THIS request".
    let clone = sess.snapshot();
    assert!(!clone.is_choice_done(1));
    assert!(clone.active_request_json(1) == Some(reissued.as_str()));

    // The legal move then commits the boundary on the ORIGINAL, leaving the clone paused.
    sess.feed_cmd(cmd(1, "move 1"), &dex);
    sess.feed_cmd(cmd(0, "move 1"), &dex);
    assert!(sess.turn() > clone.turn(), "the original did not advance past the clone");
    assert!(!clone.is_choice_done(1), "advancing the original disturbed the clone's boundary");
}

// ===========================================================================
// 6. The terminal reads — `winner` / `request_kind` at game-end.
// ===========================================================================

#[test]
fn winner_and_request_kind_report_the_terminal_state() {
    let dex = Dex::for_gen(3);
    let mut sess = BridgeSession::new(&opts(), &dex).expect("session");

    // Play mutual `move 1`, answering forced replacements with the first live bench mon
    // READ OFF THE REQUEST (a hard-coded `switch 2` breaks once a swap reorders the array),
    // until the battle ends.
    for _ in 0..2000 {
        if sess.is_ended() {
            break;
        }
        for side in 0..2 {
            match sess.request_kind(side) {
                Some(RequestState::Move) => sess.feed_cmd(cmd(side, "move 1"), &dex),
                Some(RequestState::Switch) => {
                    let req = sess.active_request_json(side).expect("a forceSwitch request");
                    let n = first_live_bench_slot(req).expect("a live bench mon to replace with");
                    sess.feed_cmd(cmd(side, &format!("switch {n}")), &dex);
                }
                _ => {}
            }
        }
    }
    assert!(sess.is_ended(), "the battle never ended (turn {})", sess.turn());
    assert!(sess.winner().is_some(), "a decisive battle reported no winner");
    for side in 0..2 {
        assert!(
            sess.request_kind(side).is_none(),
            "p{} still has an open request after game-end",
            side + 1
        );
        assert!(sess.active_request_json(side).is_none(), "a stale request survived game-end");
        assert!(sess.is_choice_done(side), "no boundary is open, so no side owes a choice");
    }

    // A snapshot of a FINISHED battle is still a faithful copy (a search driver will hit
    // terminal leaves constantly).
    let leaf = sess.snapshot();
    assert!(leaf.is_ended() && leaf.winner() == sess.winner());
    assert_dumps_equal("terminal snapshot", &dump(leaf.chunks()), &dump(sess.chunks()));
}
