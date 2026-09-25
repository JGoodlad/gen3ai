//! window_record_test.rs — the NATIVE per-decision record (`gen3_core_window_record_v1`, the Rust
//! Core Program's M3; `designs/rust_sim/trackers.md` §3) on CONSTRUCTED battles: fixed teams, a fixed
//! seed (or a deterministic seed search for a chance event), scripted choices. Each fixture asserts
//! the record carries its mechanic WITH its attribution and fails if it is flattened — the E12
//! coverage requirement (`designs/endstate/obs_enrichment_backlog.md` §1a) and ACTION DENIAL.
//!
//! The record is built from ONE side's stream, so every fixture reads it from the viewer named, and
//! the denial fixtures also read the OTHER viewer to prove the information boundary: the denied
//! side's own record keeps its choice, the opponent's record holds `Choice::Opp` and nothing else.

use std::sync::Arc;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::trackers::clock::ClockConfig;
use pokesim::trackers::record::{Action, ActionKind, Cause, Choice, DenialWhy, Entry, What, Window};
use pokesim::version::BattleVersion;

/// A packed set: `species|…|item|ability|moves|nature|evs|gender|ivs|shiny|level|` (0 EVs, 31 IVs).
fn set(species: &str, item: &str, ability: &str, moves: &str, level: u32) -> String {
    format!("{species}||{item}|{ability}|{moves}|Hardy|||||{level}|")
}

fn team(sets: &[String]) -> String {
    sets.join("]")
}

/// One constructed battle, replayed as a version chain OBSERVING one session, trackers on.
struct Run {
    /// Per viewer: every closed decision window, then the still-open last one.
    windows: [Vec<Arc<Window>>; 2],
    sess: BridgeSession,
    /// The battle's INPUTS (p1 team, p2 team, seed, script) — what `export_fixtures` writes.
    input: (String, String, String, Vec<(usize, String)>),
}

fn play(p1: &str, p2: &str, seed: &str, script: &[(usize, &str)]) -> Run {
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
    let mut windows: [Vec<Arc<Window>>; 2] = [Vec::new(), Vec::new()];
    let mut collect = |v: &BattleVersion, w: &mut [Vec<Arc<Window>>; 2]| {
        for s in 0..2 {
            if let Some(d) = v.decision(s) {
                w[s].push(Arc::clone(d.window.as_ref().expect("a recording stream")));
            }
        }
    };
    collect(&v, &mut windows);
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
        collect(&v, &mut windows);
    }
    assert!(sess.fatal().is_none(), "fixture faulted: {:?}", sess.fatal());
    for s in 0..2 {
        let open = v.stream(s).and_then(|st| st.trk.as_ref()).map(|t| t.record.as_ref().expect("a recording stream").current().clone()).expect("trk");
        windows[s].push(Arc::new(open));
    }
    let input = (p1.to_string(), p2.to_string(), seed.to_string(),
                 script.iter().map(|(s, t)| (*s, t.to_string())).collect());
    Run { windows, sess, input }
}

fn json(w: &Window) -> String {
    let mut o = String::new();
    w.json_into(&mut o);
    o
}

/// Every action of viewer `side`'s record, in order, across its windows.
fn actions(r: &Run, side: usize) -> Vec<Action> {
    r.windows[side].iter().flat_map(|w| w.actions.iter().cloned()).collect()
}

#[test]
#[ignore = "a debugging aid: prints every fixture's record"]
fn dump() {
    for (name, r) in fixtures() {
        for s in 0..2 {
            for (i, w) in r.windows[s].iter().enumerate() {
                println!("{name} p{} w{i}: {}", s + 1, json(w));
            }
        }
        for c in &r.sess.chunks().chunks {
            if c.side == 0 {
                println!("{name} TEXT {}", c.lines.join(" ¦ "));
            }
        }
    }
}

fn fixtures() -> Vec<(&'static str, Run)> {
    vec![
        ("faster_ko", faster_ko()),
        ("explosion_first", explosion_first()),
        ("double_edge", double_edge()),
        ("destiny_bond", destiny_bond()),
        ("fake_out", fake_out()),
        ("baton_pass", baton_pass()),
        ("roar_spikes", roar_spikes()),
        ("shedinja_spikes", shedinja_spikes()),
        ("pursuit", pursuit()),
        ("thief", thief()),
        ("trick", trick()),
        ("knock_off", knock_off()),
        ("sleep_talk", sleep_talk()),
        ("rapid_spin", rapid_spin()),
        ("charge_recharge", charge_recharge()),
        ("focus_punch", focus_punch()),
        ("wish_sub_protect", wish_sub_protect()),
        ("taunt_encore_disable", taunt_encore_disable()),
        ("protect", protect()),
        ("perish", perish()),
        ("explosion_cuts_turn", explosion_cuts_turn()),
        ("recoil_cuts_softboiled", recoil_cuts_softboiled()),
        ("encore_lands", encore_lands()),
        ("disable_refuses", disable_refuses().expect("a seed in 1..60 lands the Disable")),
    ]
}

// ------------------------------------------------------------------ the constructed battles

fn faster_ko() -> Run {
    let p1 = team(&[set("mewtwo", "", "pressure", "psychic", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("rattata", "", "guts", "tackle", 5), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (1, "switch 2"), (0, "move 1"), (1, "move 1")])
}

fn explosion_first() -> Run {
    let p1 = team(&[set("electrode", "", "static", "explosion", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("rattata", "", "guts", "tackle", 5), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "switch 2"), (1, "switch 2"), (0, "move 1"), (1, "move 1")])
}

fn double_edge() -> Run {
    // A FASTER 1-HP Double-Edge user (Shedinja; customgame does not police learnsets) KOs the slower
    // foe — denying it — and the recoil KOs the user: a trade, two faints, two causes, in order.
    let p1 = team(&[set("shedinja", "", "wonderguard", "doubleedge", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("rattata", "", "guts", "tackle", 5), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "switch 2"), (1, "switch 2"), (0, "move 1"), (1, "move 1")])
}

fn destiny_bond() -> Run {
    let p1 = team(&[set("gengar", "", "levitate", "destinybond", 50), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("tyranitar", "", "sandstream", "crunch", 60), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "switch 2"), (1, "switch 2"), (0, "move 1"), (1, "move 1")])
}

fn fake_out() -> Run {
    let p1 = team(&[set("persian", "", "limber", "fakeout,tackle", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1")])
}

fn baton_pass() -> Run {
    let p1 = team(&[set("ninjask", "", "speedboost", "swordsdance,batonpass", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("skarmory", "", "keeneye", "spikes", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(1, "move 1"), (0, "move 1"), (1, "move 1"), (0, "move 2"), (0, "switch 2"), (1, "move 1"), (0, "move 1")])
}

fn roar_spikes() -> Run {
    let p1 = team(&[set("snorlax", "", "immunity", "rest", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let p2 = team(&[set("skarmory", "", "keeneye", "spikes,roar", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(1, "move 1"), (0, "move 1"), (1, "move 2"), (0, "move 1"), (1, "move 2"), (0, "move 1")])
}

fn shedinja_spikes() -> Run {
    let p1 = team(&[set("snorlax", "", "immunity", "rest", 100), set("shedinja", "", "wonderguard", "shadowball", 100),
                    set("blissey", "", "naturalcure", "softboiled", 100)]);
    let p2 = team(&[set("skarmory", "", "keeneye", "spikes", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(1, "move 1"), (0, "move 1"), (0, "switch 2"), (1, "move 1"), (0, "switch 3"), (0, "move 1"), (1, "move 1")])
}

fn pursuit() -> Run {
    let p1 = team(&[set("tyranitar", "", "sandstream", "pursuit", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("alakazam", "", "synchronize", "recover", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "switch 2"), (0, "move 1"), (1, "move 1")])
}

fn thief() -> Run {
    let p1 = team(&[set("sneasel", "", "innerfocus", "thief", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "leftovers", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1")])
}

fn trick() -> Run {
    let p1 = team(&[set("alakazam", "choiceband", "synchronize", "trick,psychic", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "leftovers", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1")])
}

fn knock_off() -> Run {
    let p1 = team(&[set("snorlax", "", "immunity", "knockoff", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let p2 = team(&[set("blissey", "leftovers", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1")])
}

fn sleep_talk() -> Run {
    let p1 = team(&[set("snorlax", "", "immunity", "rest,sleeptalk,bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let p2 = team(&[set("skarmory", "", "keeneye", "drillpeck", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(1, "move 1"), (0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1"), (0, "move 2"), (1, "move 1")])
}

fn rapid_spin() -> Run {
    let p1 = team(&[set("starmie", "", "naturalcure", "rapidspin", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("skarmory", "", "keeneye", "spikes,drillpeck", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(1, "move 1"), (0, "move 1"), (1, "move 2"), (0, "move 1")])
}

fn charge_recharge() -> Run {
    let p1 = team(&[set("venusaur", "", "overgrow", "solarbeam,hyperbeam", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1"),
                               (0, "move 2"), (1, "move 1"), (0, "move 1"), (1, "move 1")])
}

fn focus_punch() -> Run {
    let p1 = team(&[set("breloom", "", "effectspore", "focuspunch", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("jolteon", "", "voltabsorb", "thunderbolt", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 1"), (1, "move 1")])
}

fn wish_sub_protect() -> Run {
    let p1 = team(&[set("jirachi", "", "serenegrace", "wish,substitute,protect", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1"), (0, "move 3"), (1, "move 1")])
}

fn taunt_encore_disable() -> Run {
    let p1 = team(&[set("gengar", "", "levitate", "taunt,encore,disable", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "", "naturalcure", "toxic,softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 2"), (0, "move 3"), (1, "move 1"),
                               (1, "move 2"), (0, "move 1"), (1, "move 1")])
}

fn encore_lands() -> Run {
    // A faster Gengar Encores the Snorlax's Curse; the Snorlax's NEXT chosen Body Slam (turn 2) is
    // overridden into Curse — the Encore holds the choice.
    let p1 = team(&[set("gengar", "", "levitate", "encore,splash", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "curse,bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 2"), (1, "move 1"), (0, "move 1"), (1, "move 2")])
}

/// The Snorlax Body Slams (the Ghost is immune), a faster Gengar Disables Body Slam (a 55 % hit
/// in gen 3 — a deterministic seed search), and the Snorlax's next chosen Body Slam is refused
/// (`|cant|…|Disable|Body Slam`).
fn disable_refuses() -> Option<Run> {
    let p1 = team(&[set("gengar", "", "levitate", "disable,splash", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "bodyslam,curse", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    (1..60).map(|k| play(&p1, &p2, &format!("{k},2,3,4"),
                          &[(0, "move 2"), (1, "move 1"), (0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1")]))
        .find(|r| actions(r, 0).iter().any(|a| matches!(&a.kind, ActionKind::Cant { reason, .. } if reason.contains("Disable"))))
}

fn protect() -> Run {
    let p1 = team(&[set("skarmory", "", "keeneye", "protect", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "bodyslam", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1")])
}

fn perish() -> Run {
    let p1 = team(&[set("lapras", "", "waterabsorb", "perishsong,rest", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("blissey", "", "naturalcure", "softboiled", 100), set("snorlax", "", "immunity", "rest", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "move 2"), (1, "move 1"), (0, "move 2"), (1, "move 1"),
                               (0, "move 2"), (1, "move 1"), (0, "switch 2"), (1, "switch 2"), (0, "move 1"), (1, "move 1")])
}

fn explosion_cuts_turn() -> Run {
    // A FASTER Electrode explodes into a Snorlax that SURVIVES: the Electrode's own faint cuts the
    // turn (gen 3), so the Snorlax's chosen Curse never happens.
    let p1 = team(&[set("electrode", "", "static", "explosion", 30), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "curse", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "switch 2"), (0, "move 1"), (1, "move 1")])
}

fn recoil_cuts_softboiled() -> Run {
    // A FASTER 1-HP Double-Edge user (speed 116) hits a slower Snorlax (96) that SURVIVES; the
    // user's recoil faint cuts the turn, so the Snorlax's chosen Softboiled never happens.
    let p1 = team(&[set("shedinja", "", "wonderguard", "doubleedge", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "softboiled", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    play(&p1, &p2, "1,2,3,4", &[(0, "move 1"), (1, "move 1"), (0, "switch 2"), (0, "move 1"), (1, "move 1")])
}

// ------------------------------------------------------------------ finders

fn mon(side: &str, sp: &str) -> pokesim::trackers::record::Mon {
    use pokesim::core_events::Rel;
    pokesim::trackers::record::Mon { side: if side == "ours" { Rel::Ours } else { Rel::Opp }, species: sp.into() }
}

/// The faints of viewer `side`'s record, in order: (mon, cause).
fn faints(r: &Run, side: usize) -> Vec<(String, Cause)> {
    actions(r, side)
        .iter()
        .flat_map(|a| a.effects.iter())
        .filter(|e| e.what == What::Faint)
        .map(|e| (e.on.as_ref().expect("a faint names its mon").species.clone(), e.cause.clone()))
        .collect()
}

fn position(acts: &[Action], f: impl Fn(&Action) -> bool) -> usize {
    acts.iter().position(f).unwrap_or_else(|| panic!("no such action in {acts:#?}"))
}

fn is_move(a: &Action, id: &str) -> bool {
    matches!(&a.kind, ActionKind::Move { id: m, .. } if m == id)
}

/// The denied action of `actor` (this viewer's record), with its cause and denier.
fn denied<'a>(acts: &'a [Action], actor: &pokesim::trackers::record::Mon) -> (&'a DenialWhy, &'a Choice) {
    acts.iter()
        .find_map(|a| match &a.kind {
            ActionKind::Denied { actor: x, why, choice } if x == actor => Some((why, choice)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no denial of {actor:?} in {acts:#?}"))
}

// ------------------------------------------------------------------ ACTION DENIAL

#[test]
fn a_faster_ko_denies_the_opponents_move_right_after_the_ko_and_hides_its_choice() {
    let r = faster_ko();
    // p1 (the KOer) sees THAT the opponent was denied, never what it chose
    let a1 = actions(&r, 0);
    let (why, choice) = denied(&a1, &mon("opp", "rattata"));
    assert_eq!(choice, &Choice::Opp);
    let DenialWhy::FaintedFirst { by, .. } = why else { panic!("{why:?}") };
    assert_eq!(by.as_ref().map(|(m, mv)| (m.species.as_str(), mv.as_str())), Some(("mewtwo", "psychic")));
    let ko = position(&a1, |a| is_move(a, "psychic"));
    assert!(matches!(&a1[ko + 1].kind, ActionKind::Denied { .. }), "the denial sits right after the KO: {a1:#?}");
    // p2 (the denied side) keeps its own choice
    let a2 = actions(&r, 1);
    assert_eq!(denied(&a2, &mon("ours", "rattata")).1, &Choice::Own(Some("move 1".into())));
    // and its replacement is the FREE switch that follows the faint
    assert!(a2.iter().any(|a| matches!(&a.kind, ActionKind::Switch { entry: Entry::Replacement { fainted }, .. } if fainted == "rattata")));
}

#[test]
fn an_explosion_first_denies_the_target_and_records_both_faints_with_their_causes() {
    let r = explosion_first();
    let a1 = actions(&r, 0);
    let DenialWhy::FaintedFirst { by, .. } = denied(&a1, &mon("opp", "rattata")).0 else { panic!("not fainted-first") };
    assert_eq!(by.as_ref().map(|(_, mv)| mv.as_str()), Some("explosion"));
    let f = faints(&r, 0);
    assert_eq!(f, vec![("electrode".into(), Cause::SelfKo), ("rattata".into(), Cause::Direct)], "both faints, in order");
}

#[test]
fn a_double_edge_trade_denies_the_target_and_the_recoil_ko_is_attributed() {
    let r = double_edge();
    let a1 = actions(&r, 0);
    let DenialWhy::FaintedFirst { by, .. } = denied(&a1, &mon("opp", "rattata")).0 else { panic!("not fainted-first") };
    assert_eq!(by.as_ref().map(|(_, mv)| mv.as_str()), Some("doubleedge"));
    assert_eq!(faints(&r, 0), vec![("rattata".into(), Cause::Direct), ("shedinja".into(), Cause::Recoil)]);
}

#[test]
fn a_destiny_bond_trade_is_two_faints_and_no_denial() {
    let r = destiny_bond();
    assert_eq!(faints(&r, 0), vec![("gengar".into(), Cause::Direct), ("tyranitar".into(), Cause::DestinyBond)]);
    // both moved: Destiny Bond, then the Crunch that KOed its user
    assert!(!actions(&r, 0).iter().any(|a| matches!(a.kind, ActionKind::Denied { .. })), "nobody was denied");
}

#[test]
fn a_flinch_is_a_refused_action_with_the_opponents_choice_hidden() {
    let r = fake_out();
    let refused = |side: usize| -> (String, bool, Choice) {
        actions(&r, side)
            .iter()
            .find_map(|a| match &a.kind {
                ActionKind::Cant { reason, then_moved, choice, .. } => Some((reason.clone(), *then_moved, choice.clone())),
                _ => None,
            })
            .expect("a cant")
    };
    assert_eq!(refused(0), ("flinch".into(), false, Choice::Opp));
    assert_eq!(refused(1), ("flinch".into(), false, Choice::Own(Some("move 1".into()))));
}

/// A deterministic seed search: the first seed whose battle shows the paralysed Snorlax fully
/// paralysed (a 25 % roll) — fixed teams, scripted choices, reproducible.
fn full_paralysis() -> Option<(Run, Action)> {
    let p1 = team(&[set("jolteon", "", "voltabsorb", "thunderwave,tackle", 100), set("snorlax", "", "immunity", "rest", 100)]);
    let p2 = team(&[set("snorlax", "", "immunity", "bodyslam,rest", 100), set("blissey", "", "naturalcure", "softboiled", 100)]);
    let script: Vec<(usize, &str)> = std::iter::once([(0, "move 1"), (1, "move 2")])
        .chain(std::iter::repeat([(0, "move 2"), (1, "move 1")]).take(6))
        .flatten()
        .collect();
    for k in 1..60 {
        let r = play(&p1, &p2, &format!("{k},2,3,4"), &script);
        let par = actions(&r, 0).into_iter().find(|a| matches!(&a.kind,
            ActionKind::Cant { reason, mon: m, .. } if reason == "par" && *m == mon("opp", "snorlax")));
        if let Some(a) = par {
            return Some((r, a));
        }
    }
    None
}

#[test]
fn full_paralysis_is_a_refused_action() {
    let (_, a) = full_paralysis().expect("no seed in 1..60 fully paralysed the Snorlax — the fixture must find its event");
    let ActionKind::Cant { mon: m, choice, then_moved, .. } = a.kind else { unreachable!() };
    assert_eq!((m, choice, then_moved), (mon("opp", "snorlax"), Choice::Opp, false));
}

/// Writes every constructed battle's INPUTS as JSON lines to `$WINDOW_FIXTURES_OUT` — the M3
/// loss catalogue replays them through `core_events` and the Python trackers to read what the
/// frozen `TurnDelta`, the α/β label and the 22-column event window keep of each mechanic.
#[test]
#[ignore = "an export for the M3 catalogue (designs/research_state/measurements/rust_core_m3_2026-09-24/catalogue/e12_fixtures.py)"]
fn export_fixtures() {
    use std::io::Write;
    let path = std::env::var("WINDOW_FIXTURES_OUT").expect("set WINDOW_FIXTURES_OUT");
    let mut out = std::fs::File::create(&path).expect("create");
    let mut runs = fixtures();
    runs.push(("full_paralysis", full_paralysis().expect("paralysis seed").0));
    for (name, r) in runs {
        let (p1, p2, seed, script) = &r.input;
        let mut j = String::new();
        j.push_str("{\"name\":");
        pokesim::core_events::json_out::str_into(&mut j, name);
        for (k, v) in [("p1", p1), ("p2", p2), ("seed", seed)] {
            j.push_str(&format!(",\"{k}\":"));
            pokesim::core_events::json_out::str_into(&mut j, v);
        }
        j.push_str(",\"script\":[");
        for (i, (side, tok)) in script.iter().enumerate() {
            if i > 0 {
                j.push(',');
            }
            j.push_str(&format!("[{side},"));
            pokesim::core_events::json_out::str_into(&mut j, tok);
            j.push(']');
        }
        j.push_str("]}");
        writeln!(out, "{j}").expect("write");
    }
}

#[test]
fn the_opponents_denied_choice_never_reaches_a_viewer() {
    // THE INFORMATION BOUNDARY, over every fixture: a record's opponent denial is `Choice::Opp`
    // (no data), and every one of our own carries our token.
    let mut own = 0;
    let mut opp = 0;
    for (name, r) in fixtures() {
        for side in 0..2 {
            for a in actions(&r, side) {
                let (who, choice) = match &a.kind {
                    ActionKind::Cant { mon, choice, .. } => (mon.clone(), choice.clone()),
                    ActionKind::Denied { actor, choice, .. } => (actor.clone(), choice.clone()),
                    _ => continue,
                };
                match (who.side, choice) {
                    (pokesim::core_events::Rel::Opp, Choice::Opp) => opp += 1,
                    (pokesim::core_events::Rel::Ours, Choice::Own(Some(_))) => own += 1,
                    (s, c) => panic!("{name} p{}: a {s:?} denial carries {c:?}", side + 1),
                }
            }
            for w in &r.windows[side] {
                let j = json(w);
                assert!(!j.contains("\"choice\":{\"own\"") || j.matches("\"choice\":{\"own\"").count()
                        == w.actions.iter().filter(|a| matches!(&a.kind, ActionKind::Cant { mon, .. } | ActionKind::Denied { actor: mon, .. } if mon.side == pokesim::core_events::Rel::Ours)).count(),
                        "{name}: an own-choice field outside our own denials");
            }
        }
    }
    assert!(own >= 5 && opp >= 5, "non-vacuity: {own} own / {opp} opponent denials");
}

/// A turn-cut denial of `actor` in viewer `side`'s record: (the faint that cut it, its cause, choice).
fn turn_cut(r: &Run, side: usize, actor: &pokesim::trackers::record::Mon) -> (pokesim::trackers::record::Mon, Cause, Choice) {
    actions(r, side)
        .into_iter()
        .find_map(|a| match a.kind {
            ActionKind::Denied { actor: x, why: DenialWhy::TurnCut { by_faint, cause }, choice } if x == *actor => Some((by_faint, cause, choice)),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no turn-cut denial of {actor:?}: {:#?}", actions(r, side)))
}

#[test]
fn a_self_ko_explosion_cuts_the_turn_and_denies_the_survivors_move() {
    let r = explosion_cuts_turn();
    // the Snorlax survived — it is NOT a fainted-first denial, it is the gen-3 turn cut
    assert_eq!(faints(&r, 0), vec![("electrode".into(), Cause::SelfKo)], "only the exploder fainted");
    assert_eq!(turn_cut(&r, 0, &mon("opp", "snorlax")), (mon("ours", "electrode"), Cause::SelfKo, Choice::Opp));
    assert_eq!(turn_cut(&r, 1, &mon("ours", "snorlax")).2, Choice::Own(Some("move 1".into())), "the denied side keeps its choice");
    // and the Curse really never ran (the sim's truth, not the record's assumption)
    assert!(!actions(&r, 1).iter().any(|a| a.turn == 1 && is_move(a, "curse")), "Curse must not have been used on turn 1");
}

#[test]
fn a_recoil_self_ko_cuts_the_turn_and_denies_softboiled() {
    let r = recoil_cuts_softboiled();
    assert_eq!(faints(&r, 0), vec![("shedinja".into(), Cause::Recoil)]);
    assert_eq!(turn_cut(&r, 0, &mon("opp", "snorlax")), (mon("ours", "shedinja"), Cause::Recoil, Choice::Opp));
    assert_eq!(turn_cut(&r, 1, &mon("ours", "snorlax")).2, Choice::Own(Some("move 1".into())));
    assert!(!actions(&r, 1).iter().any(|a| a.turn == 1 && is_move(a, "softboiled")));
}

// ------------------------------------------------------------------ E12 mechanics

#[test]
fn baton_pass_records_exactly_what_was_passed_and_the_spikes_the_receiver_met() {
    let r = baton_pass();
    let a = actions(&r, 0);
    let sw = a
        .iter()
        .find(|a| matches!(&a.kind, ActionKind::Switch { entry: Entry::BatonPass { .. }, .. }))
        .unwrap_or_else(|| panic!("no Baton Pass entry: {a:#?}"));
    let ActionKind::Switch { to, entry: Entry::BatonPass { passer, boosts, .. }, .. } = &sw.kind else { unreachable!() };
    assert_eq!((to.species.as_str(), passer.as_str()), ("snorlax", "ninjask"));
    assert!(boosts.contains(&("atk", 2)) && boosts.contains(&("spe", 1)), "{boosts:?}");
    assert!(sw.effects.iter().any(|e| matches!(e.what, What::Damage { .. }) && e.cause == Cause::Spikes { layers: 1 }));
}

#[test]
fn roar_is_a_forced_switch_with_its_phazer_and_the_spikes_chip() {
    let r = roar_spikes();
    let a = actions(&r, 0);
    let d = a.iter().find(|a| matches!(a.kind, ActionKind::Drag { .. })).expect("a drag");
    let ActionKind::Drag { to, by, by_move, .. } = &d.kind else { unreachable!() };
    assert_eq!(to, &mon("ours", "blissey"));
    assert_eq!((by.clone(), by_move.as_deref()), (Some(mon("opp", "skarmory")), Some("roar")));
    assert!(d.effects.iter().any(|e| e.cause == Cause::Spikes { layers: 1 }));
    // the viewer that PHAZED sees its own move and the revealed mon
    assert!(actions(&r, 1).iter().any(|a| matches!(&a.kind, ActionKind::Drag { to, .. } if *to == mon("opp", "blissey"))));
}

#[test]
fn a_hazard_ko_on_entry_is_attributed_to_spikes_and_the_free_switch_follows() {
    let r = shedinja_spikes();
    assert!(faints(&r, 0).contains(&("shedinja".into(), Cause::Spikes { layers: 1 })));
    assert!(actions(&r, 0).iter().any(|a| matches!(&a.kind, ActionKind::Switch { entry: Entry::Replacement { fainted }, .. } if fainted == "shedinja")));
}

#[test]
fn pursuit_on_a_switching_target_lands_before_the_switch() {
    let r = pursuit();
    let a = actions(&r, 0);
    let hit = position(&a, |a| matches!(&a.kind, ActionKind::Move { id, pursuit_on_switch: true, .. } if id == "pursuit"));
    let sw = position(&a, |a| matches!(&a.kind, ActionKind::Switch { out: Some(o), .. } if o == "alakazam"));
    assert!(hit < sw, "Pursuit hits BEFORE the switch it punishes");
}

#[test]
fn item_transfers_carry_their_direction() {
    // Thief: the victim's item is gone TO the thief, and the thief now holds it
    let r = thief();
    let eff: Vec<_> = actions(&r, 0).into_iter().flat_map(|a| a.effects).collect();
    assert!(eff.iter().any(|e| e.on == Some(mon("opp", "blissey"))
        && matches!(&e.what, What::Item { item, gone: true, how, to: Some(t) } if item == "leftovers" && how.contains("Thief") && *t == mon("ours", "sneasel"))));
    assert!(eff.iter().any(|e| e.on == Some(mon("ours", "sneasel")) && matches!(&e.what, What::Item { item, gone: false, .. } if item == "leftovers")));
    // Trick: both sides' new items revealed
    let r = trick();
    let eff: Vec<_> = actions(&r, 0).into_iter().flat_map(|a| a.effects).collect();
    for (who, it) in [(mon("opp", "blissey"), "choiceband"), (mon("ours", "alakazam"), "leftovers")] {
        assert!(eff.iter().any(|e| e.on.as_ref() == Some(&who) && matches!(&e.what, What::Item { item, how, .. } if item == it && how.contains("Trick"))), "{who:?} {it}");
    }
    // Knock Off: removed
    let r = knock_off();
    assert!(actions(&r, 0).into_iter().flat_map(|a| a.effects)
        .any(|e| matches!(&e.what, What::Item { item, gone: true, how, .. } if item == "leftovers" && how.contains("Knock Off"))));
}

#[test]
fn a_called_move_keeps_its_caller_and_sleep_is_not_a_denial_when_the_mon_moves() {
    let r = sleep_talk();
    let a = actions(&r, 0);
    assert!(a.iter().any(|x| matches!(&x.kind, ActionKind::Move { id, called_by: Some(c), .. } if id == "bodyslam" && c == "sleeptalk")));
    assert!(a.iter().any(|x| matches!(&x.kind, ActionKind::Cant { reason, then_moved: true, .. } if reason == "slp")),
            "the sleep `cant` before Sleep Talk is recorded AND marked as not a denial");
}

#[test]
fn rapid_spin_clears_the_spikes_it_names() {
    let r = rapid_spin();
    assert!(actions(&r, 0).into_iter().flat_map(|a| a.effects)
        .any(|e| matches!(&e.what, What::SideEnd(c) if c == "Spikes") && e.cause == Cause::Move("Rapid Spin".into())));
}

#[test]
fn charge_and_recharge_turns_are_visible() {
    let r = charge_recharge();
    let a = actions(&r, 0);
    assert!(a.iter().flat_map(|x| x.effects.iter()).any(|e| e.what == What::Prepare("solarbeam".into())));
    assert!(a.iter().any(|x| matches!(&x.kind, ActionKind::Move { id, locked: true, .. } if id == "solarbeam")), "the release");
    assert!(a.iter().flat_map(|x| x.effects.iter()).any(|e| e.what == What::MustRecharge));
    assert!(a.iter().any(|x| matches!(&x.kind, ActionKind::Cant { reason, then_moved: false, .. } if reason == "recharge")));
}

#[test]
fn a_lost_focus_punch_is_a_denial() {
    let r = focus_punch();
    assert!(actions(&r, 0).iter().any(|x| matches!(&x.kind,
        ActionKind::Cant { reason, move_id: Some(m), then_moved: false, .. } if reason == "Focus Punch" && m == "focuspunch")));
}

#[test]
fn wish_substitute_and_protect_carry_their_effects() {
    let r = wish_sub_protect();
    let eff: Vec<_> = actions(&r, 0).into_iter().flat_map(|a| a.effects).collect();
    assert!(eff.iter().any(|e| matches!(e.what, What::Heal { .. }) && matches!(&e.cause, Cause::Wish { wisher: Some(w) } if w == "jirachi")));
    assert!(eff.iter().any(|e| e.what == What::SubstituteHit { broke: false }));
    let r = protect();
    assert!(actions(&r, 0).into_iter().flat_map(|a| a.effects).any(|e| matches!(&e.what, What::Blocked(_))),
            "a move into Protect records its target LOST");
}

#[test]
fn taunt_refusals_name_the_refused_move_only_because_the_line_is_public() {
    let r = taunt_encore_disable();
    assert!(actions(&r, 0).iter().any(|x| matches!(&x.kind,
        ActionKind::Cant { mon: m, reason, move_id: Some(mv), choice: Choice::Opp, .. }
            if *m == mon("opp", "blissey") && reason == "move: Taunt" && mv == "toxic")),
            "the refused move is on the PUBLIC `|cant|` line; the choice field stays Opp");
}

#[test]
fn encore_is_recorded_on_its_target_and_the_overridden_move_is_the_one_executed() {
    let r = encore_lands();
    let acts = actions(&r, 0);
    let enc = position(&acts, |a| is_move(a, "encore"));
    assert!(acts[enc].effects.iter().any(|e| matches!(&e.what, What::VolatileStart(v) if v == "Encore")
                                              && e.on == Some(mon("opp", "snorlax"))),
            "the Encore volatile on its target: {:?}", acts[enc]);
    // Curse's three stage changes, SIGNED: the Speed DROP reads -1 (an unboost is not a rise)
    let curse = &acts[position(&acts, |a| is_move(a, "curse"))];
    let stages: Vec<(String, i64)> = curse.effects.iter().filter_map(|e| match &e.what {
        What::Boost { stat, n } => Some((stat.clone(), *n)),
        _ => None,
    }).collect();
    assert_eq!(stages, vec![("spe".into(), -1), ("atk".into(), 1), ("def".into(), 1)]);
    // turn 2: the Snorlax chose Body Slam; the Encore made it Curse
    let t2: Vec<&Action> = acts.iter().filter(|a| a.turn == 2 && matches!(&a.kind, ActionKind::Move { .. })).collect();
    assert!(t2.iter().any(|a| is_move(a, "curse")) && !t2.iter().any(|a| is_move(a, "bodyslam")), "{t2:?}");
}

#[test]
fn a_disable_refusal_names_the_disabled_move_and_hides_the_opponents_choice() {
    let r = disable_refuses().expect("no seed in 1..60 landed the Disable — the fixture must find its event");
    let cant = |side: usize| actions(&r, side).into_iter().find_map(|a| match a.kind {
        ActionKind::Cant { mon: m, reason, move_id, choice, .. } if reason.contains("Disable") => Some((m, move_id, choice)),
        _ => None,
    }).expect("a Disable refusal");
    assert_eq!(cant(0), (mon("opp", "snorlax"), Some("bodyslam".into()), Choice::Opp));
    assert_eq!(cant(1), (mon("ours", "snorlax"), Some("bodyslam".into()), Choice::Own(Some("move 1".into()))));
}

#[test]
fn a_perish_song_trade_is_two_perish_faints() {
    let r = perish();
    let f = faints(&r, 0);
    assert!(f.contains(&("lapras".into(), Cause::PerishSong)) && f.contains(&("blissey".into(), Cause::PerishSong)), "{f:?}");
}

/// A parse-path record: ONE side's hand-written protocol text (the engine does not model these
/// callers — `scan_move_probe` panics on Metronome / Mirror Move / Assist / Nature Power).
fn parse_record(lines: &[&str]) -> Window {
    let mut s = pokesim::version::SideStream::new(0, "P1", None).expect("stream").with_trackers(ClockConfig::default());
    for l in lines {
        s.fold_text(l).unwrap_or_else(|e| panic!("{l}: {e}"));
    }
    s.trk.expect("trk").record.as_ref().expect("a recording stream").current().clone()
}

#[test]
fn every_gen3_caller_is_recorded_as_caller_then_called_or_refused_as_poke_env_refuses() {
    // The gen-3 form is `[from] <Name>` (`data/mods/gen3/scripts.ts:165`). poke-env RAISED
    // `ValueError` on Metronome, Assist and Nature Power until `gen3_called_move_reading_v1` fixed
    // the fork (and the core's `BoardReading`) by class; this pin has FLIPPED: all five callers
    // are now read, each as the caller then the called move.
    let head = ["|player|p1|P1||", "|player|p2|P2||", "|teamsize|p1|1", "|teamsize|p2|1", "|gen|3", "|start",
                "|switch|p1a: Clefable|Clefable|100/100", "|switch|p2a: Snorlax|Snorlax|100/100", "|turn|1"];
    for (caller, called, read) in [
        ("Sleep Talk", "Body Slam", true),
        ("Mirror Move", "Body Slam", true),
        ("Metronome", "Thunderbolt", true),
        ("Assist", "Ice Beam", true),
        ("Nature Power", "Swift", true),
    ] {
        let caller_line = format!("|move|p1a: Clefable|{caller}|p1a: Clefable");
        let called_line = format!("|move|p1a: Clefable|{called}|p2a: Snorlax|[from] {caller}");
        let mut s = pokesim::version::SideStream::new(0, "P1", None).expect("stream").with_trackers(ClockConfig::default());
        let mut refused = None;
        for l in head.iter().copied().chain([caller_line.as_str(), called_line.as_str()]) {
            if let Err(e) = s.fold_text(l) {
                refused = Some(e);
                break;
            }
        }
        if read {
            assert!(refused.is_none(), "{caller}: {refused:?}");
            let w = s.trk.expect("trk").record.as_ref().expect("a recording stream").current().clone();
            let (want_called, want_caller) = (pokesim::core_events::to_id(called), pokesim::core_events::to_id(caller));
            assert!(w.actions.iter().any(|a| matches!(&a.kind,
                ActionKind::Move { id, called_by: Some(c), .. } if *id == want_called && *c == want_caller)), "{caller}: {}", json(&w));
        } else {
            let e = refused.unwrap_or_else(|| panic!("{caller}: poke-env refuses this line and so must the core"));
            assert_eq!(e.class(), Some(pokesim::core_error::PyExc::ValueError), "{caller}: {e}");
        }
    }
}
