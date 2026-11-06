//! The leak-honest builder's FRESH-DICE MATCHER (`pokesim::fresh_dice`, `designs/endstate/design_sprime_honest.md`
//! D3; contract: `designs/rust_sim/search_and_replay_drivers.md` "Fresh dice").
//!
//! 1. A battle's own teams and commands, matched against its viewer's lines on FRESH dice (the fixed-commands
//!    mode), are KEPT, and the report's schedule (START seed, reseeds, commands, damage-roll table) replayed in a
//!    bare session reproduces every viewer line, with the damage-roll control off and on (`rolls::Mode::Seg`).
//! 2. A world that cannot ship those lines (another level: its switch line differs) is NOT kept, at segment 0.
//! 3. The verdict is deterministic; `try_seed` is a valid, distinct seed per (segment, try); a reseed schedule is
//!    refused by name when it is malformed.

use std::collections::BTreeMap;
use std::sync::Arc;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd, RequestState};
use pokesim::dex::Dex;
use pokesim::fresh_dice::{match_world, offered, schedule_from_json, try_seed, InfoSet, MatchCfg, Report};
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
               theirs: Some(theirs.to_vec()), from_segment: 0, seed0: None }
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
