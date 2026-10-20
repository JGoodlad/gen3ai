//! rolls_test.rs — the DAMAGE-ROLL CONTROL (`pokesim::rolls`, `gen3_roll_control_v1`; F1 tier A of
//! `designs/endstate/design_world_sampler.md` §3.3.1).
//!
//! 1. THE IDENTITY: an armed Record pass, a Force pass naming every site's own drawn roll, and a
//!    Match pass against the battle's OWN targets are byte-identical to the unarmed battle (both
//!    sides' chunks), and the Match pass gives the truth a finite weight (never infeasible).
//! 2. THE ENGINE IS THE ORACLE for the consistent set: at a site, each of the 16 rolls is FORCED and
//!    what the viewer saw is read off its own stream; a Match against any of those sights must pick
//!    a roll that prints the SAME line, with k = the number of rolls that print it. Covered: the
//!    opponent's mon (a percent), the viewer's own mon (exact HP), a Substitute (held / broke).
//! 3. WIDENING: a world whose attacker differs (its Attack EVs) prints the truth's HP lines only on
//!    other rolls; unarmed it diverges, matched it reproduces the viewer's stream byte for byte with
//!    the dice stream aligned, and Force with its reported table replays that trajectory exactly.
//! 4. PINS and ALIGNMENT: a lethal hit into a Focus Band holder is pinned (the stream's roll, k = 1);
//!    a target whose key differs is misaligned (the stream's roll); bad control JSON is refused.

use std::collections::BTreeMap;

use pokesim::battle::BattleOptions;
use pokesim::bridge::{bridge_opts, parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::rolls::{Mode, RollCtl, Target};

const SEED: &str = "1,2,3,4";

fn cmd(side: usize, tok: &str) -> Cmd {
    Cmd { side, choice: parse_choice(tok).expect("parse choice") }
}

/// Play `turns` (p1 token, p2 token) with `ctl` armed (or none); the session at the end.
fn play(opts: &BattleOptions, turns: &[(&str, &str)], ctl: Option<RollCtl>) -> BridgeSession {
    let dex = Dex::for_gen(3);
    let mut s = BridgeSession::new(opts, &dex).expect("session");
    s.set_roll_ctl(ctl).expect("arm");
    for (a, b) in turns {
        if s.is_ended() {
            break;
        }
        s.feed_cmds(&[cmd(0, a), cmd(1, b)], &dex);
    }
    s
}

fn chunks(s: &BridgeSession) -> Vec<(usize, Vec<String>)> {
    s.chunks().chunks.iter().map(|c| (c.side, c.lines.clone())).collect()
}

fn ctl(s: &BridgeSession) -> RollCtl {
    s.roll_ctl().expect("armed").clone()
}

fn record(opts: &BattleOptions, turns: &[(&str, &str)], viewer: u8) -> (BridgeSession, Vec<Target>) {
    let s = play(opts, turns, Some(RollCtl::new(viewer, Mode::Record)));
    let t = ctl(&s).targets;
    (s, t)
}

/// The viewer-visible HP field of a `-damage` / sub line, in the control's encoding: `0` for a faint,
/// the exact `hp/max` for the viewer's own mon, the percent numerator for the other side's (its
/// exact `hp/max` in a debug format, `percent` false); `sub:held` / `sub:broke` for a Substitute.
fn sight(line: &str, viewer: u8, percent: bool) -> Option<String> {
    let p: Vec<&str> = line.split('|').collect();
    match p.get(1).copied() {
        Some("-damage") if p.len() == 4 => {
            let hp = p[3].split(' ').next().unwrap();
            if p[3].ends_with("fnt") {
                return Some("0".into());
            }
            let owner = if p[2].starts_with("p1") { 0 } else { 1 };
            Some(if owner == viewer || !percent { hp.to_string() } else { hp.split('/').next().unwrap().to_string() })
        }
        Some("-activate") if line.ends_with("|Substitute|[damage]") => Some("sub:held".into()),
        Some("-end") if line.ends_with("|Substitute") => Some("sub:broke".into()),
        _ => None,
    }
}

/// The viewer's sight of the LAST hit line of a one-hit-turn stream.
fn last_sight(s: &BridgeSession, viewer: u8) -> String {
    let percent = s.report_percent();
    s.side_lines(viewer as usize).iter().rev().find_map(|l| sight(l, viewer, percent)).expect("a hit line")
}

// ---------------------------------------------------------------------------------------------
// fixtures (gen3customgame, explicit genders, a fixed seed)
// ---------------------------------------------------------------------------------------------

/// A bulky p1 Snorlax trading Body Slams / Earthquakes with a p2 Tyranitar for several turns.
fn trade(p2_atk_ev: u32) -> BattleOptions {
    let p1 = "Snorlax|||NoAbility|bodyslam,earthquake|Careful|252,,4,,252,|M||||";
    let p2 = format!("Tyranitar|||NoAbility|rockslide,earthquake|Adamant|252,{p2_atk_ev},4,,,|M||||");
    bridge_opts("gen3customgame", SEED.to_string(), p1, &p2)
}

const TRADE: [(&str, &str); 4] = [("move 1", "move 1"), ("move 2", "move 2"), ("move 1", "move 2"), ("move 2", "move 1")];

/// p1 hits a passive p2 (Splash) once: the one site of the battle, n = 0. `gen3ou` shows the
/// non-owner a PERCENT (`customgame`, a debug format, shows exact HP to both sides).
fn one_hit_in(format: &str) -> BattleOptions {
    let p1 = "Metagross|||NoAbility|meteormash,splash|Adamant|,252,,,,|N||||";
    let p2 = "Zapdos|||NoAbility|splash,substitute|Bold|252,,252,,4,|N||||";
    bridge_opts(format, SEED.to_string(), p1, p2)
}

fn one_hit() -> BattleOptions {
    one_hit_in("gen3customgame")
}

/// p2 subs on turn 1 (p1 splashes), p1 hits the sub on turn 2: the one site, n = 0, into a sub.
fn into_sub_with(mv: &str, atk_ev: u32) -> BattleOptions {
    let p1 = format!("Snorlax|||NoAbility|{mv},splash|Adamant|,{atk_ev},,,,|M||||");
    let p2 = "Zapdos|||NoAbility|splash,substitute|Bold|252,,252,,4,|N||||";
    bridge_opts("gen3customgame", SEED.to_string(), &p1, p2)
}

const SUB_TURNS: [(&str, &str); 2] = [("move 2", "move 2"), ("move 1", "move 1")];

// ---------------------------------------------------------------------------------------------
// 1. the identity
// ---------------------------------------------------------------------------------------------

#[test]
fn record_force_truth_and_self_match_are_byte_identical_to_unarmed() {
    let o = trade(252);
    let plain = play(&o, &TRADE, None);
    for viewer in [0u8, 1] {
        let (rec, targets) = record(&o, &TRADE, viewer);
        assert!(targets.len() >= 4, "non-vacuity: the fixture has damage sites ({})", targets.len());
        assert_eq!(chunks(&rec), chunks(&plain), "Record changed the battle");
        let r0: BTreeMap<u32, u8> = ctl(&rec).sites.iter().map(|s| (s.n, s.r0)).collect();
        let forced = play(&o, &TRADE, Some(RollCtl::new(viewer, Mode::Force(r0))));
        assert_eq!(chunks(&forced), chunks(&plain), "Force(the truth's own rolls) changed the battle");
        assert!(ctl(&forced).sites.iter().all(|s| s.flag == 'f' && s.r == s.r0));
        let m = play(&o, &TRADE, Some(RollCtl::new(viewer, Mode::Match(targets.clone()))));
        assert_eq!(chunks(&m), chunks(&plain), "Match against its own targets changed the battle");
        let c = ctl(&m);
        assert!(c.sites.iter().all(|s| s.flag == 'm' && s.r == s.r0 && s.k >= 1), "{:?}", c.sites);
        let w = c.log_w.expect("the truth is never infeasible");
        assert!(w.is_finite() && w <= 0.0);
        let want: f64 = c.sites.iter().map(|s| (s.k as f64 / 16.0).ln()).sum();
        assert!((w - want).abs() < 1e-12);
        // the report names no forced site: the truth's trajectory is the stream's
        assert!(c.json().contains("\"force\":[]"), "{}", c.json());
    }
}

// ---------------------------------------------------------------------------------------------
// 2. the engine is the oracle for the consistent set
// ---------------------------------------------------------------------------------------------

/// For every roll r*: Match against the sight forcing r* printed must choose a roll that prints the
/// same sight, with k = the count of rolls printing it. Returns the distinct sights (non-vacuity).
fn consistent_set_is_the_engines(o: &BattleOptions, turns: &[(&str, &str)], viewer: u8, n0: u32) -> usize {
    let (_, base) = record(o, turns, viewer);
    assert!(base.len() as u32 > n0 && !base[n0 as usize].pinned, "site {n0} exists and is not pinned");
    let seen: Vec<String> = (0..16u8)
        .map(|r| {
            let s = play(o, turns, Some(RollCtl::new(viewer, Mode::Force(BTreeMap::from([(n0, r)])))));
            last_sight(&s, viewer)
        })
        .collect();
    for want in &seen {
        let mut t = base.clone();
        t[n0 as usize].out = want.clone();
        let m = play(o, turns, Some(RollCtl::new(viewer, Mode::Match(t))));
        let site = ctl(&m).sites[n0 as usize].clone();
        let k = seen.iter().filter(|x| *x == want).count();
        assert_eq!(site.flag, 'm', "site {n0} matched");
        assert_eq!(site.k as usize, k, "k for sight {want:?} (engine sights {seen:?})");
        assert_eq!(&seen[site.r as usize], want, "the chosen roll prints the sight");
        assert_eq!(&last_sight(&m, viewer), want);
    }
    let mut d = seen.clone();
    d.sort();
    d.dedup();
    d.len()
}

#[test]
fn consistent_set_matches_the_engine_on_the_opponents_percent() {
    let n = consistent_set_is_the_engines(&one_hit_in("gen3ou"), &[("move 1", "move 1")], 0, 0);
    assert!(n >= 2 && n < 16, "non-vacuity: the 16 rolls print several percents, some shared ({n})");
    // the debug format shows the non-owner exact HP: the same hit, almost every roll its own sight
    let n = consistent_set_is_the_engines(&one_hit(), &[("move 1", "move 1")], 0, 0);
    assert!(n >= 8, "customgame: exact HP for both sides ({n})");
}

#[test]
fn consistent_set_matches_the_engine_on_the_viewers_exact_hp() {
    let n = consistent_set_is_the_engines(&one_hit(), &[("move 1", "move 1")], 1, 0);
    assert!(n >= 8, "non-vacuity: the owner sees exact HP, most rolls differ ({n})");
}

#[test]
fn consistent_set_matches_the_engine_on_a_substitute() {
    // a 0-Attack-EV Body Slam: the max roll breaks Zapdos's sub, the min roll does not
    let n = consistent_set_is_the_engines(&into_sub_with("bodyslam", 0), &SUB_TURNS, 0, 0);
    assert_eq!(n, 2, "non-vacuity: some rolls break the sub and some do not");
}

// ---------------------------------------------------------------------------------------------
// 3. widening
// ---------------------------------------------------------------------------------------------

#[test]
fn match_reproduces_the_viewers_stream_for_a_world_today_rejects() {
    let truth = trade(252);
    let (t_sess, targets) = record(&truth, &TRADE, 0);
    let want = t_sess.side_lines(0);
    // a world whose Tyranitar has other Attack EVs: on the truth's dice it prints other HP for p1
    let mut found = None;
    for ev in (180..252).rev().step_by(4) {
        let world = trade(ev);
        let unarmed = play(&world, &TRADE, None);
        if unarmed.side_lines(0) == want {
            continue;
        }
        let m = play(&world, &TRADE, Some(RollCtl::new(0, Mode::Match(targets.clone()))));
        if m.side_lines(0) == want {
            found = Some((ev, world, m));
            break;
        }
    }
    let (ev, world, m) = found.expect("some Attack EV fits the truth's sights only on other rolls");
    let c = ctl(&m);
    let moved: Vec<_> = c.sites.iter().filter(|s| s.r != s.r0).collect();
    assert!(!moved.is_empty(), "EV {ev}: the widening used a roll other than the stream's");
    assert!(c.log_w.is_some() && c.sites.iter().all(|s| s.flag == 'm'));
    // the dice stream stayed aligned: the same number of draws as the truth's
    assert_eq!(m.battle_state().unwrap().prng_seed(), t_sess.battle_state().unwrap().prng_seed());
    // Force with the reported table replays the matched trajectory byte for byte (both sides)
    let table: BTreeMap<u32, u8> = moved.iter().map(|s| (s.n, s.r)).collect();
    let f = play(&world, &TRADE, Some(RollCtl::new(0, Mode::Force(table))));
    assert_eq!(chunks(&f), chunks(&m));
}

// ---------------------------------------------------------------------------------------------
// 4. pins, alignment, refusals
// ---------------------------------------------------------------------------------------------

#[test]
fn a_lethal_hit_into_a_focus_band_holder_is_pinned() {
    let p1 = "Metagross|||NoAbility|meteormash,splash|Adamant|,252,,,,|N||||";
    let p2 = "Abra||focusband|NoAbility|splash,substitute|Timid|,,,,,252|M||||";
    let o = bridge_opts("gen3customgame", SEED.to_string(), p1, p2);
    let (_, t) = record(&o, &[("move 1", "move 1")], 0);
    assert!(!t.is_empty() && t[0].pinned, "{t:?}");
    let m = play(&o, &[("move 1", "move 1")], Some(RollCtl::new(0, Mode::Match(t))));
    let s = &ctl(&m).sites[0];
    assert_eq!((s.flag, s.k, s.r), ('p', 1, s.r0));
}

#[test]
fn a_misaligned_site_takes_the_streams_roll() {
    let o = one_hit();
    let (_, mut t) = record(&o, &[("move 1", "move 1")], 0);
    t[0].mv = "tackle".into();
    t[0].out = "1".into();
    let m = play(&o, &[("move 1", "move 1")], Some(RollCtl::new(0, Mode::Match(t))));
    let s = &ctl(&m).sites[0];
    assert_eq!((s.flag, s.r), ('x', s.r0));
}

#[test]
fn control_json_is_strict() {
    let p = |s: &str| RollCtl::from_json(&Json::parse(s).unwrap());
    assert!(p(r#"{"viewer":0,"mode":"record"}"#).is_ok());
    assert!(p(r#"{"viewer":1,"mode":"force","force":[[0,15],[3,0]]}"#).is_ok());
    assert!(p(r#"{"viewer":0,"mode":"match","targets":[["hit",0,1,"tackle","50",false]]}"#).is_ok());
    for (bad, why) in [
        (r#"{"viewer":2,"mode":"record"}"#, "viewer"),
        (r#"{"viewer":0,"mode":"replay"}"#, "unknown mode"),
        (r#"{"viewer":0,"mode":"record","x":1}"#, "unknown key"),
        (r#"{"viewer":0,"mode":"force","force":[[0,16]]}"#, "not in 0..16"),
        (r#"{"viewer":0,"mode":"force","force":[[0,1],[0,2]]}"#, "twice"),
        (r#"{"viewer":0,"mode":"match","targets":[["kick",0,1,"tackle","50",false]]}"#, "kind"),
    ] {
        let e = p(bad).expect_err(bad);
        assert!(e.contains(why), "{bad}: {e}");
    }
}

