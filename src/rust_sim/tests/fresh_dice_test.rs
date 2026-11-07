//! The leak-honest builder's FRESH-DICE MATCHER (`pokesim::fresh_dice`, `designs/endstate/design_sprime_honest.md`
//! D3; contract: `designs/rust_sim/search_and_replay_drivers.md` "Fresh dice").
//!
//! 1. A battle's own teams and commands, matched against its viewer's lines on FRESH dice (the fixed-commands
//!    mode), are KEPT, and the report's schedule (START seed, reseeds, commands, damage-roll table) replayed in a
//!    bare session reproduces every viewer line, with the damage-roll control off and on (`rolls::Mode::Seg`).
//! 2. A world that cannot ship those lines (another level: its switch line differs) is NOT kept, at segment 0.
//! 3. The verdict is deterministic; `try_seed` is a valid, distinct seed per (segment, try); a reseed schedule is
//!    refused by name when it is malformed.
//! 4. A CARRIED trajectory (`prefix`, the incremental builder): a world kept at an earlier decision of the same battle
//!    extends to a later one by replaying its trail exactly and matching only the new segments (its schedule then
//!    rebuilds every line, for fewer feeds than a cold match); a trail that stops reproducing ends the replay there
//!    and the rest is re-matched; `upto` bounds the replay; a malformed or misused prefix is refused by name.

use std::collections::BTreeMap;
use std::sync::Arc;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd, RequestState};
use pokesim::dex::Dex;
use pokesim::fresh_dice::{match_world, offered, schedule_from_json, try_seed, InfoSet, MatchCfg, Prefix, Report};
use pokesim::json::Json;
use pokesim::prng::Prng;
use pokesim::rolls::{Mode, RollCtl};

const SEED: &str = "5,6,7,8";

/// Two corpus teams with every gender NAMED (a fresh START would otherwise redraw an unnamed one).
fn teams() -> [String; 2] {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/bridge_corpus");
    let mut files: Vec<_> = std::fs::read_dir(dir).expect("corpus").flatten().map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    files.sort();
    let mut out: Vec<String> = Vec::new();
    for f in files {
        for l in std::fs::read_to_string(&f).unwrap().lines().filter(|l| l.starts_with("TEAM\t")) {
            let t = l.split('\t').nth(3).unwrap_or("").to_string();
            if !t.is_empty() && !out.contains(&t) {
                out.push(t);
            }
        }
    }
    let named = |t: &str| -> String {
        t.split(']')
            .map(|e| {
                let mut f: Vec<String> = e.split('|').map(str::to_string).collect();
                while f.len() < 8 {
                    f.push(String::new());
                }
                if f[7].is_empty() {
                    f[7] = "M".into();
                }
                f.join("|")
            })
            .collect::<Vec<_>>()
            .join("]")
    };
    [named(&out[2]), named(&out[3])]
}

fn opts(t: &[String; 2], seed: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3ou".into(),
        seed: Some(seed.into()),
        p1: PlayerOptions { name: "a".into(), team: PackedTeam(t[0].clone()) },
        p2: PlayerOptions { name: "b".into(), team: PackedTeam(t[1].clone()) },
    }
}

fn needs(s: &BridgeSession, side: usize) -> bool {
    matches!(s.request_kind(side), Some(RequestState::Move | RequestState::Switch)) && !s.is_choice_done(side)
}

/// A battle on its own seed, each side's first offered move (a switch when forced), `updates` updates: the viewer
/// p1's information set (its lines split after each request, its commands) and the opponent's commands.
fn battle(t: &[String; 2], updates: usize, dex: &Dex) -> (InfoSet, Vec<String>) {
    let mut s = BridgeSession::new_construct_turn0(&opts(t, SEED), dex).expect("session");
    let (mut ours, mut theirs) = (Vec::new(), Vec::new());
    for _ in 0..updates {
        if s.is_ended() {
            break;
        }
        for side in [1usize, 0] {
            if needs(&s, side) {
                let tok = offered(&s, side).expect("offered").into_iter().next().expect("a choice");
                s.feed_cmd(Cmd { side, choice: parse_choice(&tok).unwrap() }, dex);
                if side == 0 { ours.push(tok) } else { theirs.push(tok) }
            }
        }
    }
    assert!(needs(&s, 0), "the battle must stop at a decision of the viewer");
    let mut segments: Vec<Vec<String>> = vec![Vec::new()];
    for l in s.side_lines(0) {
        let req = l.starts_with("|request|");
        segments.last_mut().unwrap().push(l);
        if req {
            segments.push(Vec::new());
        }
    }
    if segments.last().is_some_and(|x| x.is_empty()) {
        segments.pop();
    }
    let n = segments.len();
    assert!(n > 8, "non-vacuity: {n} segments");
    let info = InfoSet {
        viewer: 0,
        segments: segments.into_iter().map(Arc::from).collect(),
        ours,
        intents: vec![None; n],
        backtrack: vec![Vec::new(); n],
        hp_last: vec![Vec::new(); n],
    };
    (info, theirs)
}

fn cfg(theirs: &[String], rolls: bool) -> MatchCfg {
    MatchCfg { dice: "fresh_dice_test".into(), tries: 512, rerolls: 16, rolls, prev_backtrack: false, max_feeds: 500_000,
               theirs: Some(theirs.to_vec()), from_segment: 0, seed0: None, prefix: None }
}

/// The report's schedule replayed in a bare session: the viewer's lines.
fn replay(t: &[String; 2], r: &Report, dex: &Dex) -> Vec<String> {
    let mut s = BridgeSession::new_construct_turn0(&opts(t, r.seed.as_deref().unwrap()), dex).expect("session");
    if !r.rolls.is_empty() {
        let table: BTreeMap<u32, u8> = r.rolls.iter().copied().collect();
        s.set_roll_ctl(Some(RollCtl::new(0, Mode::Force(table)))).unwrap();
    }
    let sched: BTreeMap<usize, &String> = r.reseed.iter().map(|(k, x)| (*k, x)).collect();
    for (k, c) in r.cmds.iter().enumerate() {
        if let Some(seed) = sched.get(&k) {
            s.reseed(seed);
        }
        let rest = c.strip_prefix("CHOOSE p").unwrap();
        let (tag, tok) = rest.split_once(' ').unwrap();
        s.feed_cmd(Cmd { side: if tag == "1" { 0 } else { 1 }, choice: parse_choice(tok).unwrap() }, dex);
    }
    s.side_lines(0)
}

#[test]
fn a_true_battle_matches_on_fresh_dice_and_its_schedule_replays() {
    let dex = Dex::for_gen(3);
    let t = teams();
    let (info, theirs) = battle(&t, 14, &dex);
    let want: Vec<String> = info.segments.iter().flat_map(|s| s.iter().cloned()).collect();
    for rolls in [false, true] {
        let r = match_world(&opts(&t, "unused"), &info, &cfg(&theirs, rolls), &dex);
        assert!(r.ok, "rolls {rolls}: the truth is kept on fresh dice ({:?} at {} of {})", r.error, r.matched, r.segments);
        assert_ne!(r.seed.as_deref(), Some(SEED), "the START seed is a fresh one");
        assert!(!r.reseed.is_empty(), "non-vacuity: the schedule reseeds");
        assert_eq!(replay(&t, &r, &dex), want, "rolls {rolls}: the schedule rebuilds the viewer's every line");
        let again = match_world(&opts(&t, "unused"), &info, &cfg(&theirs, rolls), &dex);
        assert_eq!((again.seed, again.reseed, again.cmds, again.rolls), (r.seed, r.reseed, r.cmds, r.rolls),
                   "deterministic");
    }
}

#[test]
fn a_world_that_cannot_ship_the_lines_is_not_kept() {
    let dex = Dex::for_gen(3);
    let t = teams();
    let (info, theirs) = battle(&t, 8, &dex);
    // the opponent's lead at level 50: its switch line (and stats) differ on every die
    let mut w = t.clone();
    let mut ents: Vec<String> = w[1].split(']').map(str::to_string).collect();
    let mut f: Vec<String> = ents[0].split('|').map(str::to_string).collect();
    while f.len() < 11 {
        f.push(String::new());
    }
    f[10] = "50".into();
    ents[0] = f.join("|");
    w[1] = ents.join("]");
    let r = match_world(&opts(&w, "unused"), &info, &MatchCfg { tries: 16, ..cfg(&theirs, true) }, &dex);
    assert!(!r.ok);
    assert_eq!(r.matched, 0, "it fails at the START write ({:?})", r.error);
}

#[test]
fn try_seeds_are_valid_and_distinct() {
    let mut seen = std::collections::BTreeSet::new();
    for seg in 0..20 {
        for t in 0..20 {
            let s = try_seed("base", seg, t);
            Prng::validate_seed(&s).expect("a valid seed");
            assert!(seen.insert(s), "segment {seg} try {t} repeats a seed");
        }
    }
    assert_ne!(try_seed("base", 1, 0), try_seed("other", 1, 0));
}

#[test]
fn a_malformed_schedule_is_refused() {
    let ok = Json::parse("[[0,\"sodium,00\"],[3,\"1,2,3,4\"]]").unwrap();
    assert_eq!(schedule_from_json(Some(&ok)).unwrap().len(), 2);
    assert!(schedule_from_json(None).unwrap().is_empty());
    for bad in ["[[3,\"1,2,3,4\"],[1,\"1,2,3,4\"]]", "[[0,\"nope\"]]", "[[0]]", "{}"] {
        assert!(schedule_from_json(Some(&Json::parse(bad).unwrap())).is_err(), "{bad}");
    }
}

/// The BUILDER's settings (no fixed commands: every opponent choice a stand-in, as `battle`'s intents say).
fn builder(rolls: bool) -> MatchCfg {
    MatchCfg { theirs: None, ..cfg(&[], rolls) }
}

fn carried(r: &Report, upto: usize) -> Prefix {
    Prefix { seed: r.seed.clone().expect("a kept seed"), trail: r.trail.clone(), cmds: r.cmds.clone(), upto }
}

#[test]
fn a_carried_trajectory_extends_to_a_later_decision() {
    let dex = Dex::for_gen(3);
    let t = teams();
    let (early, _) = battle(&t, 8, &dex);
    let (late, _) = battle(&t, 14, &dex);
    let n0 = early.segments.len();
    assert!(late.segments.len() > n0 && late.segments[..n0] == early.segments[..], "the later I extends the earlier");
    let want: Vec<String> = late.segments.iter().flat_map(|s| s.iter().cloned()).collect();
    for rolls in [false, true] {
        let r0 = match_world(&opts(&t, "unused"), &early, &builder(rolls), &dex);
        assert!(r0.ok, "rolls {rolls}: the early decision matches ({:?})", r0.error);
        assert_eq!(r0.trail.len(), n0, "one trail entry a segment");
        assert_eq!(r0.trail.iter().map(|x| x.2).sum::<usize>(), r0.cmds.len(), "the trail partitions the commands");
        let ext = MatchCfg { dice: "fresh_dice_test:ext".into(), prefix: Some(carried(&r0, n0)), ..builder(rolls) };
        let r = match_world(&opts(&t, "unused"), &late, &ext, &dex);
        assert!(r.ok, "rolls {rolls}: the carried world extends ({:?} at {} of {})", r.error, r.matched, r.segments);
        assert_eq!(r.fixed, n0, "every carried segment replayed");
        assert_eq!(r.seed, r0.seed, "the carried START seed");
        assert_eq!(r.trail[..n0], r0.trail[..], "the carried trail kept");
        assert_eq!(r.cmds[..r0.cmds.len()], r0.cmds[..], "the carried commands kept");
        assert_eq!(replay(&t, &r, &dex), want, "rolls {rolls}: the extended schedule rebuilds every line");
        let cold = match_world(&opts(&t, "unused"), &late, &builder(rolls), &dex);
        assert!(cold.ok);
        // a carried segment costs one feed a command (its one attempt), the floor a cold match reaches only when every
        // segment matches on its first try (here with the damage-roll control on); without it, damage segments retry
        assert!(r.feeds <= cold.feeds, "rolls {rolls}: the extension costs {} feeds, a cold match {}", r.feeds, cold.feeds);
        if !rolls {
            assert!(r.feeds < cold.feeds, "non-vacuity: the cold match retried ({} feeds, the extension {})", cold.feeds, r.feeds);
        }
        let again = match_world(&opts(&t, "unused"), &late, &ext, &dex);
        assert_eq!((again.seed, again.reseed, again.cmds, again.rolls, again.trail), (r.seed, r.reseed, r.cmds, r.rolls, r.trail),
                   "deterministic");
    }
}

#[test]
fn a_trail_that_stops_reproducing_is_re_matched_from_there() {
    let dex = Dex::for_gen(3);
    let t = teams();
    let (early, _) = battle(&t, 8, &dex);
    let (late, _) = battle(&t, 14, &dex);
    let n0 = early.segments.len();
    let r0 = match_world(&opts(&t, "unused"), &early, &builder(true), &dex);
    assert!(r0.ok);
    // `upto` bounds the replay
    let up = MatchCfg { prefix: Some(carried(&r0, 3)), ..builder(true) };
    let r = match_world(&opts(&t, "unused"), &late, &up, &dex);
    assert!(r.ok && r.fixed == 3, "upto 3: {} replayed ({:?})", r.fixed, r.error);
    assert_eq!(r.trail[..3], r0.trail[..3]);
    // a trail whose seeds no longer reproduce a segment (another seed at every update from segment 2 on): the replay
    // stops at the first segment that differs and the match takes over there
    let mut bad = carried(&r0, n0);
    for x in bad.trail.iter_mut().skip(2) {
        x.0 = Some(try_seed("another", 7, 7));
    }
    let r = match_world(&opts(&t, "unused"), &late, &MatchCfg { prefix: Some(bad.clone()), ..builder(true) }, &dex);
    assert!(r.ok, "{:?}", r.error);
    // a segment the other seed still reproduces (no die shows in it) is KEPT on it: only the lines decide
    assert!((2..n0).contains(&r.fixed), "the replay stopped inside the trail ({} of {n0})", r.fixed);
    assert_eq!(r.trail[..r.fixed], bad.trail[..r.fixed], "the segments before it are the carried ones");
    assert_ne!(r.trail[r.fixed], bad.trail[r.fixed], "the first that did not reproduce is re-matched");
    let want: Vec<String> = late.segments.iter().flat_map(|s| s.iter().cloned()).collect();
    assert_eq!(replay(&t, &r, &dex), want);
}

#[test]
fn a_malformed_or_misused_prefix_is_refused() {
    let base = r#""dice":"d","tries":4"#;
    let ok = format!(r#"{{{base},"prefix":{{"seed":"1,2,3,4","trail":[[null,0,0],["5,6,7,8",0,1]],"cmds":["CHOOSE p1 move 1"],"upto":2}}}}"#);
    assert!(MatchCfg::from_json(&Json::parse(&ok).unwrap()).unwrap().prefix.is_some());
    for bad in [
        r#""prefix":{"seed":"nope","trail":[[null,0,0]],"cmds":[],"upto":1}"#,
        r#""prefix":{"seed":"1,2,3,4","trail":[["5,6,7,8",0,0]],"cmds":[],"upto":1}"#,
        r#""prefix":{"seed":"1,2,3,4","trail":[[null,0,0],[null,0,2]],"cmds":["CHOOSE p1 move 1"],"upto":2}"#,
        r#""prefix":{"seed":"1,2,3,4","trail":[[null,0,0]],"cmds":["move 1"],"upto":1}"#,
        r#""prefix":{"seed":"1,2,3,4","trail":[[null,0,0]],"cmds":[]}"#,
        r#""theirs":["move 1"],"prefix":{"seed":"1,2,3,4","trail":[[null,0,0]],"cmds":[],"upto":1}"#,
    ] {
        let line = format!("{{{base},{bad}}}");
        assert!(MatchCfg::from_json(&Json::parse(&line).unwrap()).is_err(), "{bad}");
    }
}
