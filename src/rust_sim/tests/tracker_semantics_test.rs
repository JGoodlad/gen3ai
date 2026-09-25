//! tracker_semantics_test.rs — the TRAINING-INPUT semantics the Rust core's trackers fold, pinned on
//! CONSTRUCTED battles (fixed teams, fixed seed, scripted choices): the fixes of the Rust core M3 loss
//! catalogue's GIGO findings (`designs/research_state/measurements/rust_core_m3_2026-09-24/` §5), in
//! the core's own trackers. The Python path (`EpisodeTracker`, `build_opp_intent_label`,
//! `ProgressClock`) is held equal to these by slice T (`rust_core_parity_test.py`); these pins make a
//! Rust-side REVERT fail in `cargo test` as well.
//!
//! * `gen3_event_window_semantics_fixes_v1` — the 22-column event window: a stat DROP is a negative
//!   BOOST row (W1), a faint no damage line caused is not an `attack` (W2), Trick / Thief item lines
//!   are SWAPPED on both mons (W3), a Protect block FAILS the blocked move (W4), a side condition's end
//!   is a −1 HAZARD row and its start a +1 (W5).
//! * `gen3_intent_label_semantics_fixes_v1` — the α/β label: a DRAGGED entrant (L1 / L2) and a
//!   replacement in the window after its faint (L3) are masked, a called move is labelled as its
//!   CALLER (L4), an Encore override is masked (L5).
//! * `gen3_progress_clock_attribution_fix_v1` — clause (i) needs our move's OWN hit (T1), and a
//!   blocked attack is an exogenous freeze, not a charged no-op (T2).
//! * `gen3_hp_prior_support_v1` — the Hidden-Power belief: a usage prior's zero is not an
//!   impossibility; a species whose own observations refute its prior row restarts from the flat
//!   prior and replays them (the 2026-09-25 cutover-stress refusal `ladderA_3459`).

use std::sync::Arc;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::trackers::clock::ClockConfig;
use pokesim::trackers::history::{item_tr, t, EventRecord};
use pokesim::trackers::{SideTrackers, KIND_MOVE, KIND_UNKNOWN};
use pokesim::version::BattleVersion;

fn set(species: &str, item: &str, ability: &str, moves: &str, level: u32) -> String {
    format!("{species}||{item}|{ability}|{moves}|Hardy|||||{level}|")
}

fn team(sets: &[String]) -> String {
    sets.join("]")
}

/// Per viewer, the tracker state at each of its decisions, in order.
type Decisions = [Vec<Arc<SideTrackers>>; 2];

fn play(p1: &str, p2: &str, seed: &str, script: &[(usize, &str)]) -> Decisions {
    let dex = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(seed.to_string()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut sess = BridgeSession::new_core(&opts, false, &dex).expect("session");
    let mut v = BattleVersion::observe_root_with(&sess, ["P1", "P2"], [Some(p1), Some(p2)], [true, true],
                                                 Some(ClockConfig::default()))
        .expect("root");
    let mut out: Decisions = [Vec::new(), Vec::new()];
    let collect = |v: &BattleVersion, out: &mut Decisions| {
        for s in 0..2 {
            if v.decision(s).is_some() {
                let trk = v.stream(s).and_then(|st| st.trk.as_ref()).expect("trackers on");
                out[s].push(Arc::clone(&trk.trackers));
            }
        }
    };
    collect(&v, &mut out);
    for (side, tok) in script {
        if sess.is_ended() || sess.fatal().is_some() {
            break;
        }
        if sess.is_choice_done(*side) {
            continue;
        }
        v.note_choice(*side, tok);
        sess.feed_cmd(Cmd { side: *side, choice: parse_choice(tok).expect("choice") }, &dex);
        v = v.observe(&sess).expect("observe");
        collect(&v, &mut out);
    }
    assert!(sess.fatal().is_none(), "fixture faulted: {:?}", sess.fatal());
    out
}

/// The event-window rows of a viewer's LAST decision (every fixture here is shorter than 32 rows).
fn rows(d: &Decisions, viewer: usize) -> Vec<EventRecord> {
    d[viewer].last().expect("a decision").window.events.iter().cloned().collect()
}

// ------------------------------------------------------------------ the event window

#[test]
fn a_stat_drop_is_a_negative_boost_row() {
    // Curse: spe −1, atk +1, def +1. The window stored all three as +1 (W1).
    let p1 = team(&[set("gengar", "", "levitate", "encore,splash", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "curse,bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 2"), (1, "move 1"), (0, "move 1"), (1, "move 2")]);
    for viewer in 0..2 {
        let mut mags: Vec<f64> = rows(&d, viewer).iter().filter(|r| r.t == t::BOOST).map(|r| r.hp_delta).collect();
        mags.truncate(3);
        mags.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(mags, vec![-1.0, 1.0, 1.0], "viewer {viewer}: Curse's first three stage rows");
    }
}

#[test]
fn a_protect_block_fails_the_blocked_move_and_freezes_the_clock() {
    let p1 = team(&[set("skarmory", "", "keeneye", "protect", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1")]);
    for viewer in 0..2 {
        let rs = rows(&d, viewer);
        let slam = rs.iter().find(|r| r.t == t::MOVE && r.move_id.as_deref() == Some("bodyslam")).expect("the Body Slam row");
        assert!(slam.failed, "viewer {viewer}: a blocked Body Slam must read OUT_FAIL, not OUT_HIT: {slam:?}");
        let prot = rs.iter().find(|r| r.t == t::MOVE && r.move_id.as_deref() == Some("protect")).expect("the Protect row");
        assert!(!prot.failed, "viewer {viewer}: the successful Protect itself is not failed");
    }
    // p2 (the Body Slam user): the TurnDelta outcome is "fail", so the clock's "our attack blocked"
    // branch FREEZES the window instead of charging it (T2) — n stays at the previous decision's.
    let last = d[1].last().unwrap();
    assert_eq!(last.delta.as_ref().unwrap().our_move_outcome, Some("fail"));
    let before = d[1][d[1].len() - 2].clock.n;
    assert_eq!(last.clock.n, before, "a blocked attack is an exogenous freeze");
}

/// `gen3_move_target_class_v1` (R4): a `[still]` line's target is what the sim blanked — a failed
/// Refresh (the 2026-09-25 cutover-stress evidence shape: `|move|p2a: Swampert|Refresh||[still]`)
/// is its USER's row, not the foe active's; Splash (`self`) is its user's; Toxic keeps the foe.
/// (A Snatch-stolen use is pinned on hand-built lines in `core_events::reading`'s tests: a
/// constructed Snatch battle is refused by the version's parse-vs-step check today.)
#[test]
fn a_self_move_row_targets_its_user_even_when_still() {
    let p1 = team(&[set("umbreon", "", "synchronize", "splash,toxic", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("swampert", "", "torrent", "refresh", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1")]);
    for viewer in 0..2 {
        let rs = rows(&d, viewer);
        let got: Vec<(Option<&str>, Option<&str>, Option<&str>)> = rs.iter()
            .filter(|r| r.t == t::MOVE)
            .map(|r| (r.actor.as_deref(), r.move_id.as_deref(), r.target.as_deref())).collect();
        assert_eq!(got, vec![(Some("umbreon"), Some("splash"), Some("umbreon")),
                             (Some("swampert"), Some("refresh"), Some("swampert")),
                             (Some("umbreon"), Some("toxic"), Some("swampert")),
                             (Some("swampert"), Some("refresh"), Some("swampert"))],
                   "viewer {viewer}: the MOVE rows' targets");
    }
}

#[test]
fn a_rapid_spin_clear_is_a_negative_hazard_row() {
    let p1 = team(&[set("starmie", "", "naturalcure", "rapidspin", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("skarmory", "", "keeneye", "spikes,drillpeck", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(1, "move 1"), (0, "move 1"), (1, "move 2"), (0, "move 1")]);
    for viewer in 0..2 {
        let hz: Vec<f64> = rows(&d, viewer).iter().filter(|r| r.t == t::HAZARD).map(|r| r.hp_delta).collect();
        assert_eq!(hz, vec![1.0, -1.0], "viewer {viewer}: Spikes laid, then spun away");
    }
}

#[test]
fn trick_and_thief_item_lines_are_swapped_on_both_mons() {
    let p1 = team(&[set("alakazam", "choiceband", "synchronize", "trick,psychic", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "leftovers", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1")]);
    for viewer in 0..2 {
        let it: Vec<(Option<String>, Option<u8>)> =
            rows(&d, viewer).iter().filter(|r| r.t == t::ITEM_REVEAL).map(|r| (r.actor.clone(), r.item_tr)).take(2).collect();
        assert_eq!(it.len(), 2, "viewer {viewer}: two item lines");
        assert!(it.iter().all(|(_, tr)| *tr == Some(item_tr::SWAPPED)), "viewer {viewer}: Trick = SWAPPED ×2, got {it:?}");
    }
    let p1 = team(&[set("sneasel", "", "innerfocus", "thief", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "leftovers", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1")]);
    for viewer in 0..2 {
        let taker = rows(&d, viewer)
            .into_iter()
            .find(|r| r.t == t::ITEM_REVEAL && r.actor.as_deref() == Some("sneasel"))
            .expect("the taker's item line");
        assert_eq!(taker.item_tr, Some(item_tr::SWAPPED), "viewer {viewer}: the Thief taker's |-item| changed hands");
    }
}

#[test]
fn a_faint_no_damage_line_caused_is_not_an_attack() {
    // Destiny Bond: Gengar is KO'd by Crunch (an attack), Tyranitar faints by the bond (no damage line).
    let p1 = team(&[set("gengar", "", "levitate", "destinybond", 50), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("tyranitar", "", "sandstream", "crunch", 60), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4",
                 &[(0, "move 1"), (1, "move 1"), (0, "switch 2"), (1, "switch 2"), (0, "move 1"), (1, "move 1")]);
    for viewer in 0..2 {
        let causes: Vec<(Option<String>, Option<&str>)> =
            rows(&d, viewer).iter().filter(|r| r.t == t::FAINT).map(|r| (r.actor.clone(), r.faint_cause)).collect();
        assert!(causes.contains(&(Some("gengar".into()), Some("attack"))), "viewer {viewer}: {causes:?}");
        assert!(causes.contains(&(Some("tyranitar".into()), Some("other"))), "viewer {viewer}: {causes:?}");
    }
    // Perish Song: both mons' count hits 0; neither faint has a damage line.
    let p1 = team(&[set("lapras", "", "waterabsorb", "perishsong,rest", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1"), (0, "move 2"), (1, "move 1"),
                                       (0, "move 2"), (1, "move 1"), (0, "switch 2"), (1, "switch 2"), (0, "move 1"), (1, "move 1")]);
    for viewer in 0..2 {
        let causes: Vec<Option<&str>> = rows(&d, viewer).iter().filter(|r| r.t == t::FAINT).map(|r| r.faint_cause).collect();
        assert!(!causes.is_empty() && causes.iter().all(|c| *c == Some("other")), "viewer {viewer}: {causes:?}");
    }
}

// ------------------------------------------------------------------ the α/β label

#[test]
fn a_dragged_mon_is_never_labelled_a_chosen_switch() {
    // L1: the opponent chose Blissey, our Roar dragged in Starmie. L2: asleep (Rest), then dragged.
    let p1 = team(&[set("skarmory", "", "keeneye", "roar", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "rest", 100), set("blissey", "", "naturalcure", "softboiled", 100),
                    set("starmie", "", "naturalcure", "recover", 100)]);
    let l1 = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "switch 2"), (0, "move 1"), (1, "move 1")]);
    let p1 = team(&[set("skarmory", "", "keeneye", "drillpeck,roar", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let l2 = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1"), (0, "move 1"), (1, "move 1")]);
    for (name, d) in [("chose-then-dragged", &l1), ("asleep-then-dragged", &l2)] {
        let dragged: Vec<&Arc<SideTrackers>> =
            d[0].iter().filter(|s| s.delta.as_ref().is_some_and(|x| x.opp_dragged)).collect();
        assert!(!dragged.is_empty(), "{name}: the fixture must drag the opponent");
        for s in dragged {
            assert_eq!(s.label.as_ref().unwrap().kind, KIND_UNKNOWN, "{name}: a drag is not their choice");
        }
    }
}

#[test]
fn a_replacement_in_the_window_after_its_faint_is_masked() {
    // Destiny Bond trade: both sides replace at the forced switch; p1's NEXT window holds the
    // opponent's replacement Snorlax while the faint lay in the window before (L3).
    let p1 = team(&[set("gengar", "", "levitate", "destinybond", 50), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("tyranitar", "", "sandstream", "crunch", 60), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4",
                 &[(0, "move 1"), (1, "move 1"), (0, "switch 2"), (1, "switch 2"), (0, "move 1"), (1, "move 1")]);
    for viewer in 0..2 {
        let rep: Vec<&Arc<SideTrackers>> =
            d[viewer].iter().filter(|s| s.delta.as_ref().is_some_and(|x| x.opp_switch_is_replacement)).collect();
        assert!(!rep.is_empty(), "viewer {viewer}: the straddling replacement must occur");
        for s in rep {
            assert_eq!(s.label.as_ref().unwrap().kind, KIND_UNKNOWN, "viewer {viewer}: a free replacement is not a chosen switch");
        }
    }
}

#[test]
fn a_called_move_is_labelled_as_its_caller() {
    let p1 = team(&[set("snorlax", "", "immunity", "rest,sleeptalk,bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let p2 = team(&[set("skarmory", "", "keeneye", "drillpeck", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4",
                 &[(1, "move 1"), (0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1"), (0, "move 2"), (1, "move 1")]);
    // viewer p2: the opponent is the Snorlax using Sleep Talk
    let called: Vec<&Arc<SideTrackers>> =
        d[1].iter().filter(|s| s.delta.as_ref().is_some_and(|x| x.opp_called_via.as_deref() == Some("sleeptalk"))).collect();
    assert!(!called.is_empty(), "the fixture must call a move through Sleep Talk");
    for s in called {
        let l = s.label.as_ref().unwrap();
        assert_eq!((l.kind, l.move_id.as_deref()), (KIND_MOVE, Some("sleeptalk")), "the CALLER is the choice");
    }
}

#[test]
fn an_encore_override_is_masked() {
    let p1 = team(&[set("gengar", "", "levitate", "encore,splash", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "curse,bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 2"), (1, "move 1"), (0, "move 1"), (1, "move 2")]);
    let over: Vec<&Arc<SideTrackers>> =
        d[0].iter().filter(|s| s.delta.as_ref().is_some_and(|x| x.opp_choice_overridden)).collect();
    assert_eq!(over.len(), 1, "exactly the Encore turn");
    assert_eq!(over[0].label.as_ref().unwrap().kind, KIND_UNKNOWN, "the executed Curse was not their pick");
    // the Encore user's own view: its choice (Encore) was not overridden, and is labelled on p2's side
    assert!(d[1].iter().all(|s| s.delta.as_ref().is_none_or(|x| !x.opp_choice_overridden)));
}

// ------------------------------------------------------------------ the progress clock

#[test]
fn a_status_move_in_sand_is_not_progress() {
    // Taunt ×3 in Sandstorm: the sand chips the target every turn, the Taunt deals nothing. Clause (i)
    // read the target's net fall as our damage and held the clock at 0 through all three (T1); it
    // must climb as the no-op it is.
    let p2 = team(&[set("snorlax", "", "immunity", "bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let script = [(0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1")];
    let taunt = team(&[set("tyranitar", "leftovers", "sandstream", "taunt", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&taunt, &p2, "1,2,3,4", &script);
    let n: Vec<i64> = d[0].iter().map(|s| s.clock.n).collect();
    assert!(n.windows(2).all(|w| w[1] >= w[0]), "the clock never resets on a Taunt: {n:?}");
    assert!(n.last().copied().unwrap_or(0) >= 3, "the clock climbs: {n:?}");
    // the attribution itself: the target's net HP fell every turn, our Taunt's own hits are 0
    for s in d[0].iter().skip(1) {
        let x = s.delta.as_ref().unwrap();
        assert!(x.opp_target_hp_delta.is_some_and(|t| t < 0.0) || x.our_damaging_event.is_none(), "{x:?}");
        assert_eq!(x.our_move_hit_delta, 0.0);
    }
}

// ------------------------------------------------------------------ the Hidden-Power belief

/// `gen3_hp_prior_support_v1` — the cutover stress's `ladderA_3459` / `ladderB_3459` refusal,
/// constructed. A set with NO IVs has every IV 31 (`sim/pokemon.ts:387-394`), so its Hidden Power is
/// DARK (`sim/dex.ts` `getHiddenPower`: 63·15/63 → `hpTypes[15]`). Lunatone's Smogon usage row gives
/// Dark (and Ghost, Psychic) mass 0.0, and Dark is 2x on Gengar (Ghost / Poison), so the belief
/// eliminated every prior-supported type and the fold REFUSED ("all candidates eliminated"). Now the
/// refuted row is replaced by the flat prior: the three types a 2x on Ghost / Poison allows survive
/// at 1/16 — the true type among them — and the species is recorded as `prior_discarded`.
#[test]
fn a_hidden_power_type_its_usage_prior_excludes_falls_back_to_the_flat_prior() {
    let p1 = team(&[set("gengar", "", "levitate", "splash", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("lunatone", "", "levitate", "hiddenpower", 50), set("snorlax", "", "immunity", "rest", 100)]);
    let d = play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1")]);
    let hp = &d[0].last().expect("a p1 decision after the hit").hp;
    let flat = 1.0f32 / 16.0;
    let mut want = [0.0f32; 16];
    for n in ["dark", "ghost", "psychic"] {
        want[pokesim::trackers::hp_belief::HP_TYPES.iter().position(|x| *x == n).unwrap()] = flat;
    }
    assert_eq!(hp.state.get("lunatone"), Some(&want), "the flat-prior posterior of a 2x on Gengar");
    assert!(hp.prior_discarded.contains("lunatone"), "the refuted prior row is counted: {hp:?}");
    // p2 (the Hidden Power user's side) observes nothing: p1 carries no Hidden Power.
    assert!(d[1].last().unwrap().hp.state.is_empty());
}
