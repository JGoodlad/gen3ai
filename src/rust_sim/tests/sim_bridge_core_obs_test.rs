//! sim_bridge_core_obs_test.rs — `gen3_bridge_core_obs_v1`: `sim_bridge`'s OPT-IN core observation
//! mode (the Rust half of the Rust Core Program's M6 cutover, program §6c: the training observation
//! comes THROUGH THE PARSER) and `core_events`' parse-encode gate (`gen3_core_parse_obs_gate_v1`).
//!
//! These drive the REAL binaries over their REAL stdin/stdout protocols (`CARGO_BIN_EXE_*`) on
//! battles between the bridge corpus's real teams, scripted by a seeded random policy that also
//! sends REJECTED choices (a switch to the active mon → `[Invalid choice]`, no re-request; a
//! disabled move → `[Unavailable choice]` + a re-request) and forfeits one battle mid-way:
//!
//! * the bridge's `__OBS__` rows == `core_events --obs`'s rows for the same battle, BYTE for byte,
//!   with the mask and the choice tokens equal — one frame per decision, NaN-free in a test build,
//!   written BEFORE the chunk that carries its request, `n` counting per side, `rqid` null;
//! * one persistent child (recycled across every battle) == a fresh child per battle, and a
//!   one-side request (`["p1"]` / `["p2"]`) ships exactly that side's frames;
//! * the mode ABSENT (or `null`) ships BYTE-identical stdout, and ON it ships exactly those bytes
//!   plus the `__OBS__` lines;
//! * the two clock booleans reach the rows (they equal an in-process parse chain built with that
//!   `ClockConfig`, and differ from the default somewhere);
//! * a malformed `core_obs` key is a LOUD `__ERR__`;
//! * `core_events` REFUSES a battle whose parse-chain row differs from its step-chain row (the
//!   `POKESIM_CORE_EVENTS_TEETH=parse_clock` hook of a test build: the parse chain alone runs the
//!   other `decision_tense`).
//!
//! `bench_core_obs_cost` (ignored) is the cost measurement: `cargo test --release --test
//! sim_bridge_core_obs_test -- --ignored --nocapture bench_core_obs_cost`; `bench_core_obs_stages`
//! (ignored) is its in-process per-stage breakdown (`designs/rust_sim/encoder.md` §5a).

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::core_events::jsonval::Val;
use pokesim::dex::Dex;
use pokesim::encoder::{wire, OBS_DIM};
use pokesim::present;
use pokesim::trackers::clock::ClockConfig;
use pokesim::version::BattleVersion;

const NAMES: [&str; 2] = ["Alice", "Bob"];
const CORPUS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/bridge_corpus");

// ------------------------------------------------------------------------------------ battles

#[derive(Clone)]
struct Battle {
    label: String,
    format: String,
    seed: String,
    teams: [String; 2],
    /// The command lines after START, verbatim (`CHOOSE pN <tok>` / `FORCELOSE pN`).
    cmds: Vec<String>,
}

fn q(s: &str) -> String {
    let mut o = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

impl Battle {
    fn start(&self, extra: &str) -> String {
        format!(
            "START {{{extra}\"label\":{},\"formatid\":{},\"seed\":{},\"p1\":{{\"name\":{},\"team\":{}}},\"p2\":{{\"name\":{},\"team\":{}}}}}\n",
            q(&self.label), q(&self.format), q(&self.seed), q(NAMES[0]), q(&self.teams[0]), q(NAMES[1]), q(&self.teams[1])
        )
    }
    fn opts(&self) -> BattleOptions {
        BattleOptions {
            format_id: self.format.clone(),
            seed: Some(self.seed.clone()),
            p1: PlayerOptions { name: NAMES[0].into(), team: PackedTeam(self.teams[0].clone()) },
            p2: PlayerOptions { name: NAMES[1].into(), team: PackedTeam(self.teams[1].clone()) },
        }
    }
}

/// `core_obs` START key for `sides` with the given clock flags.
fn core_obs(sides: &str, cfg: ClockConfig) -> String {
    format!(
        "\"core_obs\":{{\"sides\":{sides},\"decision_tense\":{},\"switch_freeze\":{}}},",
        cfg.decision_tense, cfg.switch_freeze
    )
}
const BOTH: &str = "[\"p1\",\"p2\"]";

/// A persistent `sim_bridge` script over `battles` (every battle ends: naturally or by FORCELOSE).
fn bridge_script(battles: &[Battle], extra: &str) -> String {
    let mut s = String::new();
    for b in battles {
        s.push_str(&b.start(&format!("\"persistent\":true,{extra}")));
        for c in &b.cmds {
            s.push_str(c);
            s.push('\n');
        }
    }
    s.push_str("END\n");
    s
}

fn core_events_script(battles: &[Battle]) -> String {
    let mut s = String::new();
    for b in battles {
        s.push_str(&b.start(""));
        for c in &b.cmds {
            s.push_str(c);
            s.push('\n');
        }
        s.push_str("END\n");
    }
    s
}

fn run(bin: &str, args: &[&str], env: &[(&str, &str)], input: String) -> String {
    let mut cmd = Command::new(bin);
    cmd.args(args).stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped());
    for (k, v) in env {
        cmd.env(k, v);
    }
    let mut child = cmd.spawn().expect("spawn");
    let mut stdin = child.stdin.take().expect("stdin");
    let w = std::thread::spawn(move || stdin.write_all(input.as_bytes()));
    let out = child.wait_with_output().expect("output");
    w.join().expect("writer").expect("write stdin");
    assert!(out.status.success(), "{bin} exited {:?}: {}", out.status, String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).expect("utf8")
}

fn sim_bridge(input: String) -> String {
    run(env!("CARGO_BIN_EXE_sim_bridge"), &[], &[], input)
}

// ------------------------------------------------------------ the in-process driver + policy

/// One side's view of a write: the lines it was newly shipped, and its decision (if it took one).
struct SideOut {
    lines: Vec<String>,
    decision: Option<Dec>,
}

struct Dec {
    tokens: Vec<(usize, String)>,
    disabled: Vec<usize>,
    can_move: bool,
    row: Option<Vec<u8>>,
}

/// The library road the bridge's mode is built on: a live session plus one PARSE chain per side
/// (trackers on), advanced by each side's newly shipped lines, `note_choice` before each CHOOSE.
struct Driver {
    dex: Dex,
    sess: BridgeSession,
    chains: [Option<BattleVersion>; 2],
    encode: bool,
}

impl Driver {
    fn new(b: &Battle, cfg: ClockConfig, encode: bool) -> (Driver, [SideOut; 2]) {
        let dex = Dex::for_gen(3);
        let sess = BridgeSession::new_construct_turn0(&b.opts(), &dex).expect("session");
        let chains = [0, 1].map(|s| Some(BattleVersion::parse_root_with(s, NAMES[s], Some(&b.teams[s]), Some(cfg)).expect("root")));
        let mut d = Driver { dex, sess, chains, encode };
        let out = d.advance();
        (d, out)
    }

    fn advance(&mut self) -> [SideOut; 2] {
        [0, 1].map(|side| {
            let c = self.chains[side].take().expect("chain");
            let from = c.stream(side).unwrap().lines;
            let lines: Vec<String> = self.sess.side_lines(side)[from..].to_vec();
            let next = c.parse_advance(&lines).expect("parse");
            let decision = next.decision(side).map(|_| {
                let legal = next.legal(side).expect("legal");
                let reading = &next.stream(side).unwrap().board_reading;
                let mask = present::mask(&legal);
                let row = self.encode.then(|| {
                    let mut r = [0.0f32; OBS_DIM];
                    next.encode(side, &mut r).expect("encode");
                    wire::row_bytes(&r)
                });
                Dec {
                    tokens: present::choice_tokens(reading, &legal).expect("tokens"),
                    disabled: (0..legal.move_slots.len()).filter(|&i| legal.move_slots[i].disabled).map(|i| i + 1).collect(),
                    can_move: mask[6..10].iter().any(|m| *m == 1) && !legal.force_switch,
                    row,
                }
            });
            self.chains[side] = Some(next);
            SideOut { lines, decision }
        })
    }

    /// Feed one recorded command line; `None` once the battle is over (the bridge drops it).
    fn feed(&mut self, cmd: &str) -> Option<[SideOut; 2]> {
        if self.sess.is_ended() {
            return None;
        }
        let (verb, rest) = cmd.split_once(' ').unwrap();
        let (side_tok, tok) = rest.split_once(' ').unwrap_or((rest, ""));
        let side = if side_tok == "p1" { 0 } else { 1 };
        match verb {
            "CHOOSE" => {
                self.chains[side].as_mut().unwrap().note_choice(side, tok);
                let choice = parse_choice(tok).expect("choice");
                self.sess.feed_cmd(Cmd { side, choice }, &self.dex);
                assert!(self.sess.fatal().is_none(), "bridge fatal: {:?}", self.sess.fatal());
            }
            "FORCELOSE" => self.sess.forfeit(side),
            v => panic!("verb {v}"),
        }
        Some(self.advance())
    }
}

struct Rng(u64);
impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        ((z ^ (z >> 31)) % n as u64) as usize
    }
}

/// Play one battle with a seeded random policy over the legal choice tokens, sending a REJECTED
/// choice at ~1 decision in 8 (once per decision), and forfeiting at command `forfeit_at`.
fn generate(label: &str, format: &str, seed: &str, teams: [String; 2], rng_seed: u64, forfeit_at: Option<usize>) -> Battle {
    let mut b = Battle { label: label.into(), format: format.into(), seed: seed.into(), teams, cmds: Vec::new() };
    let (mut drv, first) = Driver::new(&b, ClockConfig::default(), false);
    let mut rng = Rng(rng_seed);
    let mut pending: [Option<Dec>; 2] = [None, None];
    let mut injected = [false, false];
    let take = |outs: [SideOut; 2], pending: &mut [Option<Dec>; 2], sent: Option<usize>, injected: &mut [bool; 2]| {
        for (side, o) in outs.into_iter().enumerate() {
            let rejected = o.lines.iter().any(|l| l.starts_with("|error|"));
            match o.decision {
                Some(d) => {
                    pending[side] = Some(d);
                    injected[side] = false;
                }
                None if sent == Some(side) && !rejected => pending[side] = None,
                None => {}
            }
        }
    };
    take(first, &mut pending, None, &mut injected);
    loop {
        if drv.sess.is_ended() {
            break;
        }
        let side = (0..2).find(|&s| pending[s].is_some()).unwrap_or_else(|| panic!("{label}: stuck with no decision open"));
        if forfeit_at == Some(b.cmds.len()) || b.cmds.len() >= 1500 {
            let c = format!("FORCELOSE p{}", side + 1);
            drv.feed(&c);
            b.cmds.push(c);
            break;
        }
        let d = pending[side].as_ref().unwrap();
        let tok = if d.can_move && !injected[side] && rng.below(8) == 0 {
            injected[side] = true;
            match d.disabled.first() {
                Some(k) => format!("move {k}"),
                None => "switch 1".to_string(),
            }
        } else {
            d.tokens[rng.below(d.tokens.len())].1.clone()
        };
        let c = format!("CHOOSE p{} {tok}", side + 1);
        let outs = drv.feed(&c).expect("battle live");
        b.cmds.push(c);
        take(outs, &mut pending, Some(side), &mut injected);
    }
    b
}

/// Every (p1, p2) team pair of the bridge corpus, in file order.
fn corpus_teams() -> Vec<[String; 2]> {
    let mut files: Vec<_> = std::fs::read_dir(CORPUS).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "txt")).collect();
    files.sort();
    let mut out = Vec::new();
    for f in files {
        let text = std::fs::read_to_string(&f).unwrap();
        let mut t: [Option<String>; 2] = [None, None];
        for l in text.lines().filter(|l| l.starts_with("TEAM\t")) {
            let f: Vec<&str> = l.split('\t').collect();
            let s = if f[2] == "p1" { 0 } else { 1 };
            t[s].get_or_insert_with(|| f[3].to_string());
        }
        if let [Some(a), Some(b)] = t {
            out.push([a, b]);
        }
    }
    out
}

fn battles(n: usize) -> Vec<Battle> {
    let teams = corpus_teams();
    assert!(teams.len() >= n, "the corpus holds {} team pairs", teams.len());
    // Training's format is `gen3ou`; ONE battle runs `gen3customgame` (exact HP to both sides) —
    // the first odd slot whose teams carry no Beat Up: `gen3customgame` lacks gen3 OU's "Beat Up Nicknames
    // Mod", so its `-activate|…|move: Beat Up` lands as poke-env's `Effect.UNKNOWN` and BOTH encoders
    // refuse (`UnknownVolatileError`, by design — `gen3_effect_sources.RULE_GATED_LINES`).
    let beat_up = |t: &[String; 2]| t.iter().any(|x| x.to_lowercase().contains("beatup"));
    let cg = (0..n).find(|&i| i % 2 == 1 && !beat_up(&teams[i]));
    let bs: Vec<Battle> = (0..n)
        .map(|i| {
            let format = if Some(i) == cg { "gen3customgame" } else { "gen3ou" };
            let seed = format!("{},{},{},{}", 101 + i, 7 * i + 3, 13, 17 + i);
            let forfeit = (i == 2).then_some(21);
            generate(&format!("b{i}"), format, &seed, teams[i].clone(), 0xC0FFEE + i as u64, forfeit)
        })
        .collect();
    assert!(n < 4 || bs.iter().any(|b| b.format == "gen3customgame"), "no gen3customgame battle");
    bs
}

// ------------------------------------------------------------------------ output parsing

const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

fn b64_decode(s: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(s.len() / 4 * 3);
    let val = |c: u8| B64.iter().position(|&x| x == c).expect("base64 char") as u32;
    for q in s.as_bytes().chunks(4) {
        let n = (val(q[0]) << 18) | (val(q[1]) << 12) | if q[2] == b'=' { 0 } else { val(q[2]) << 6 } | if q[3] == b'=' { 0 } else { val(q[3]) };
        v.push((n >> 16) as u8);
        if q[2] != b'=' {
            v.push((n >> 8) as u8);
        }
        if q[3] != b'=' {
            v.push(n as u8);
        }
    }
    v
}

enum Frame {
    Chunk(usize, Vec<String>),
    Obs(usize, Val),
    Recon,
}

/// Split a persistent child's stdout into battles (at `__END__`); an `__ERR__` fails the test.
fn parse_bridge(out: &str) -> Vec<Vec<Frame>> {
    let mut all = vec![Vec::new()];
    for l in out.lines() {
        let (tag, rest) = l.split_once(' ').unwrap_or((l, ""));
        let cur = all.last_mut().unwrap();
        match tag {
            "p1" | "p2" => {
                let text = String::from_utf8(b64_decode(rest)).unwrap();
                cur.push(Frame::Chunk((tag == "p2") as usize, text.split('\n').map(str::to_string).collect()));
            }
            "__OBS__" => {
                let (side, json) = rest.split_once(' ').unwrap();
                cur.push(Frame::Obs((side == "p2") as usize, Val::parse(json).expect("__OBS__ json")));
            }
            "__RECON__" => cur.push(Frame::Recon),
            "__END__" => all.push(Vec::new()),
            "__ERR__" => panic!("__ERR__ {}", String::from_utf8_lossy(&b64_decode(rest))),
            other => panic!("unknown frame {other}"),
        }
    }
    assert!(all.last().unwrap().is_empty(), "stdout ends mid-battle");
    all.pop();
    all
}

fn obs_of(frames: &[Frame], side: usize) -> Vec<&Val> {
    frames.iter().filter_map(|f| match f {
        Frame::Obs(s, v) if *s == side => Some(v),
        _ => None,
    }).collect()
}

fn int(v: &Val) -> i64 {
    match v {
        Val::Int(i) => *i,
        o => panic!("not an int: {o:?}"),
    }
}

fn b64_of(frame: &Val) -> &str {
    assert_eq!(frame.str_at("dtype"), Some("<f4"));
    frame.str_at("b64").expect("b64")
}

/// `core_events --obs` for `battles`: per battle, its parsed JSON (refused battles included).
fn core_events(battles: &[Battle], env: &[(&str, &str)]) -> Vec<Val> {
    let out = run(env!("CARGO_BIN_EXE_core_events"), &["--obs"], env, core_events_script(battles));
    out.lines().map(|l| Val::parse(l).expect("core_events json")).collect()
}

struct Fixture {
    battles: Vec<Battle>,
    /// ONE persistent child over every battle, both sides, the default clock.
    both: String,
    ce: Vec<Val>,
}

fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let battles = battles(8);
        let both = sim_bridge(bridge_script(&battles, &core_obs(BOTH, ClockConfig::default())));
        let ce = core_events(&battles, &[]);
        Fixture { battles, both, ce }
    })
}

// ------------------------------------------------------------------------------------ tests

/// THE PIN: every `__OBS__` row the bridge ships == `core_events --obs`'s row for the same decision,
/// BYTE for byte, mask and tokens equal, one frame per decision, NaN-free, BEFORE its request chunk.
#[test]
fn core_obs_rows_equal_core_events_obs_byte_for_byte() {
    let f = fixture();
    let per = parse_bridge(&f.both);
    assert_eq!(per.len(), f.battles.len());
    let (mut frames_total, mut one_sided_writes, mut rerequests) = (0usize, 0usize, 0usize);
    let mut ended_by_forfeit = 0;
    for (k, (frames, ce)) in per.iter().zip(&f.ce).enumerate() {
        let label = &f.battles[k].label;
        assert!(matches!(ce.get("ok"), Some(Val::Bool(true))), "{label}: core_events refused: {:?}", ce.get("error"));
        // the SAME battle: the chunk sequences agree (side + lines)
        let bchunks: Vec<(usize, Vec<String>)> = frames.iter().filter_map(|f| match f {
            Frame::Chunk(s, l) => Some((*s, l.clone())),
            _ => None,
        }).collect();
        let Some(Val::Arr(cchunks)) = ce.get("chunks") else { panic!("chunks") };
        let cchunks: Vec<(usize, Vec<String>)> = cchunks.iter().map(|c| match c {
            Val::Arr(p) => (int(&p[0]) as usize, match &p[1] {
                Val::Arr(ls) => ls.iter().map(|l| match l { Val::Str(s) => s.to_string(), o => panic!("{o:?}") }).collect(),
                o => panic!("{o:?}"),
            }),
            o => panic!("{o:?}"),
        }).collect();
        assert_eq!(bchunks, cchunks, "{label}: the bridge and core_events played different battles");
        assert!(matches!(frames.last(), Some(Frame::Recon)), "{label}: the record is the last frame (no __OBS__ for the terminal board)");
        let flat: [Vec<&String>; 2] = [0, 1].map(|s| bchunks.iter().filter(|c| c.0 == s).flat_map(|c| c.1.iter()).collect());
        if bchunks.iter().any(|c| c.1.iter().any(|l| l.contains("|win|"))) && f.battles[k].cmds.last().is_some_and(|c| c.starts_with("FORCELOSE")) {
            ended_by_forfeit += 1;
        }
        let Some(Val::Arr(trk)) = ce.get("trackers") else { panic!("{label}: no trackers") };
        for side in 0..2 {
            let bobs = obs_of(frames, side);
            let Val::Arr(recs) = &trk[side] else { panic!() };
            let cobs: Vec<&Val> = recs.iter().filter(|r| r.get("obs").is_some()).collect();
            assert_eq!(bobs.len(), cobs.len(), "{label} p{}: {} frames vs {} core_events decisions", side + 1, bobs.len(), cobs.len());
            for (i, (b, c)) in bobs.iter().zip(&cobs).enumerate() {
                let what = format!("{label} p{} decision {i}", side + 1);
                let (bb, cb) = (b64_of(b.get("frame").unwrap()), b64_of(c.get("obs").unwrap()));
                assert!(bb == cb, "{what}: the bridge row differs from core_events --obs");
                assert_eq!(b.get("mask"), c.get("mask"), "{what}: mask");
                assert_eq!(b.get("tokens"), c.get("tokens"), "{what}: tokens");
                assert_eq!(int(b.get("n").unwrap()), i as i64, "{what}: n");
                assert_eq!(b.get("rqid"), Some(&Val::Null), "{what}: rqid");
                let bytes = b64_decode(bb);
                assert_eq!(bytes.len(), OBS_DIM * 4, "{what}: frame length");
                if cfg!(debug_assertions) {
                    let nan = bytes.chunks(4).filter(|x| f32::from_le_bytes([x[0], x[1], x[2], x[3]]).is_nan()).count();
                    assert_eq!(nan, 0, "{what}: {nan} cells never written (NaN poison)");
                }
                // `line` is the side's request line, shipped in a chunk AFTER this frame
                let line = int(b.get("line").unwrap()) as usize;
                assert!(flat[side][line].starts_with("|request|"), "{what}: line {line} is {:?}", flat[side][line]);
                assert!(int(b.get("turn").unwrap()) >= 1, "{what}: turn");
            }
            frames_total += bobs.len();
        }
        // ordering: a write's frames come BEFORE its chunks, so at each frame the side has been
        // shipped FEWER lines than its request's index; after the frame group, the request ships
        // before the next group.
        let mut shipped = [0usize; 2];
        let mut open: Vec<(usize, usize)> = Vec::new();
        let mut prev_obs = false;
        for fr in frames {
            match fr {
                Frame::Obs(side, v) => {
                    if !prev_obs {
                        for (s, l) in open.drain(..) {
                            assert!(l < shipped[s], "{label}: a request line {l} of p{} not shipped before the next write's frames", s + 1);
                        }
                    }
                    let l = int(v.get("line").unwrap()) as usize;
                    assert!(l >= shipped[*side], "{label}: the frame for line {l} came AFTER its request was shipped");
                    open.push((*side, l));
                    prev_obs = true;
                }
                Frame::Chunk(s, ls) => {
                    shipped[*s] += ls.len();
                    prev_obs = false;
                }
                Frame::Recon => {}
            }
        }
        for (s, l) in open {
            assert!(l < shipped[s], "{label}: the request line {l} of p{} was never shipped", s + 1);
        }
        // writes in which exactly one side decided (a forced replacement), and re-requests
        let groups: Vec<Vec<usize>> = frames.split(|f| !matches!(f, Frame::Obs(..))).filter(|g| !g.is_empty())
            .map(|g| g.iter().map(|f| match f { Frame::Obs(s, _) => *s, _ => 9 }).collect()).collect();
        one_sided_writes += groups.iter().filter(|g| g.len() == 1).count();
        rerequests += flat.iter().flatten().filter(|l| l.starts_with("|error|[Unavailable choice]")).count();
    }
    // non-vacuity: the corpus exercised what this pin claims
    assert!(frames_total >= 400, "only {frames_total} frames");
    assert!(one_sided_writes >= 5, "only {one_sided_writes} one-sided decision writes");
    assert!(ended_by_forfeit >= 1, "no battle ended by FORCELOSE");
    let invalid = per.iter().flatten().any(|fr| matches!(fr, Frame::Chunk(_, ls) if ls.iter().any(|l| l.starts_with("|error|[Invalid choice]"))));
    assert!(invalid, "no [Invalid choice] rejection was exercised");
    eprintln!("core_obs: {frames_total} frames over {} battles; {one_sided_writes} one-sided writes; {rerequests} [Unavailable choice] re-requests", f.battles.len());
}

/// A persistent child recycled across every battle == a fresh child per battle, and a one-side
/// request ships exactly that side's frames (the other side's chain is never built).
#[test]
fn recycling_leaks_nothing_and_a_one_side_request_ships_that_side_only() {
    let f = fixture();
    let per = parse_bridge(&f.both);
    for (k, b) in f.battles.iter().enumerate() {
        for (side, sides) in [(0usize, "[\"p1\"]"), (1usize, "[\"p2\"]")] {
            let alone = sim_bridge(bridge_script(std::slice::from_ref(b), &core_obs(sides, ClockConfig::default())));
            let alone = parse_bridge(&alone);
            assert_eq!(alone.len(), 1);
            assert!(obs_of(&alone[0], 1 - side).is_empty(), "{} {sides}: frames for the other side", b.label);
            let a: Vec<String> = obs_of(&alone[0], side).iter().map(|v| format!("{v:?}")).collect();
            let r: Vec<String> = obs_of(&per[k], side).iter().map(|v| format!("{v:?}")).collect();
            assert!(!a.is_empty());
            assert!(a == r, "{} p{}: a fresh child's frames differ from the recycled child's", b.label, side + 1);
        }
    }
}

/// OFF is the default and changes NO byte: the key absent == `null`, has no `__OBS__`, and the
/// ON stream minus its `__OBS__` lines is the same bytes.
#[test]
fn without_core_obs_the_stdout_is_byte_identical() {
    let f = fixture();
    let absent = sim_bridge(bridge_script(&f.battles, ""));
    let null = sim_bridge(bridge_script(&f.battles, "\"core_obs\":null,"));
    assert!(!absent.contains("__OBS__"), "the mode must be OFF by default");
    assert!(absent == null, "`core_obs: null` must be the mode's absence");
    let stripped: String = f.both.lines().filter(|l| !l.starts_with("__OBS__ ")).map(|l| format!("{l}\n")).collect();
    assert!(stripped == absent, "the mode changed a byte of the transport's own frames");
    assert!(absent.matches("__END__").count() == f.battles.len());

    // ONE persistent child alternating ON / OFF battles: an OFF battle after an ON one ships no
    // frame (the reset DROPS the chains — a START without the key does not overwrite them), and
    // each battle's bytes are its all-ON / all-OFF bytes.
    let per_battle = |out: &str| -> Vec<String> {
        out.split_inclusive("__END__\n").map(str::to_string).collect()
    };
    let (on_b, off_b) = (per_battle(&f.both), per_battle(&absent));
    let mut mixed = String::new();
    for (k, b) in f.battles.iter().enumerate() {
        let extra = if k % 2 == 0 { core_obs(BOTH, ClockConfig::default()) } else { String::new() };
        mixed.push_str(&bridge_script(std::slice::from_ref(b), &extra).replace("END\n", "").trim_end_matches('\n'));
        mixed.push('\n');
    }
    mixed.push_str("END\n");
    let mixed = per_battle(&sim_bridge(mixed));
    assert_eq!(mixed.len(), f.battles.len());
    for (k, got) in mixed.iter().enumerate() {
        let want = if k % 2 == 0 { &on_b[k] } else { &off_b[k] };
        assert!(got == want, "{} ({}) in a mixed ON/OFF child differs from its uniform run", f.battles[k].label, if k % 2 == 0 { "ON" } else { "OFF" });
    }
}

/// The two booleans ARE the progress clock's `ClockConfig`: the rows equal an in-process parse chain
/// built with that config (the library road), and each flag moves some row off the default.
#[test]
fn the_clock_flags_reach_the_rows() {
    let f = fixture();
    let default_rows = rows_of(&parse_bridge(&f.both));
    for cfg in [ClockConfig { decision_tense: true, switch_freeze: false }, ClockConfig { decision_tense: false, switch_freeze: true }] {
        let got = rows_of(&parse_bridge(&sim_bridge(bridge_script(&f.battles, &core_obs(BOTH, cfg)))));
        let mut differs = 0;
        for (k, b) in f.battles.iter().enumerate() {
            let want = reference_rows(b, cfg);
            assert!(got[k] == want, "{} {cfg:?}: the bridge rows differ from the in-process parse chain", b.label);
            differs += (0..2).map(|s| got[k][s].iter().zip(&default_rows[k][s]).filter(|(x, y)| x != y).count()).sum::<usize>();
        }
        assert!(differs > 0, "{cfg:?} moved no row — the flag does not reach the clock (or the corpus never exercises it)");
    }
    // …and the default config's rows are the library road's too
    for (k, b) in f.battles.iter().enumerate() {
        assert!(default_rows[k] == reference_rows(b, ClockConfig::default()), "{}: default rows vs the in-process chain", b.label);
    }
}

fn rows_of(per: &[Vec<Frame>]) -> Vec<[Vec<Vec<u8>>; 2]> {
    per.iter().map(|frames| [0, 1].map(|s| obs_of(frames, s).iter().map(|v| b64_decode(b64_of(v.get("frame").unwrap()))).collect())).collect()
}

/// Replay `b` in-process with `cfg`: the parse chains' rows at every decision.
fn reference_rows(b: &Battle, cfg: ClockConfig) -> [Vec<Vec<u8>>; 2] {
    let mut rows: [Vec<Vec<u8>>; 2] = [Vec::new(), Vec::new()];
    let (mut drv, first) = Driver::new(b, cfg, true);
    let mut push = |outs: [SideOut; 2]| {
        for (s, o) in outs.into_iter().enumerate() {
            if let Some(d) = o.decision {
                rows[s].push(d.row.unwrap());
            }
        }
    };
    push(first);
    for c in &b.cmds {
        match drv.feed(c) {
            Some(outs) => push(outs),
            None => break,
        }
    }
    rows
}

#[test]
fn a_malformed_core_obs_key_is_refused_loudly() {
    let b = &fixture().battles[0];
    for (bad, why) in [
        ("{\"sides\":[],\"decision_tense\":false,\"switch_freeze\":false}", "is empty"),
        ("{\"sides\":[\"p3\"],\"decision_tense\":false,\"switch_freeze\":false}", "not \"p1\""),
        ("{\"sides\":[\"p1\",\"p1\"],\"decision_tense\":false,\"switch_freeze\":false}", "twice"),
        ("{\"sides\":[\"p1\"],\"decision_tense\":false}", "switch_freeze"),
        ("{\"sides\":[\"p1\"],\"decision_tense\":1,\"switch_freeze\":false}", "decision_tense"),
        ("{\"sides\":[\"p1\"],\"decision_tense\":false,\"switch_freeze\":false,\"side\":1}", "unknown key"),
        ("[\"p1\"]", "must be an object"),
    ] {
        let out = sim_bridge(format!("{}END\n", b.start(&format!("\"core_obs\":{bad},"))));
        let mut lines = out.lines();
        let first = lines.next().unwrap_or("");
        assert!(first.starts_with("__ERR__ "), "{bad}: expected __ERR__, got {first:?}");
        let msg = String::from_utf8(b64_decode(&first[8..])).unwrap();
        assert!(msg.contains(why), "{bad}: {msg}");
        assert!(lines.next().is_none(), "{bad}: a refused START shipped frames");
    }
}

/// `core_events --obs` REFUSES a battle whose PARSE-chain row differs from its step-chain row. The
/// test build's `POKESIM_CORE_EVENTS_TEETH=parse_clock` flips `decision_tense` on the parse chain
/// alone: the reading, the events and the view still agree (the M2 gate cannot see it) — only the
/// row does.
#[test]
fn core_events_refuses_a_parse_chain_row_that_differs() {
    let f = fixture();
    assert!(f.ce.iter().all(|v| matches!(v.get("ok"), Some(Val::Bool(true)))), "unperturbed: every battle passes");
    let teeth = core_events(&f.battles, &[("POKESIM_CORE_EVENTS_TEETH", "parse_clock")]);
    let refused: Vec<&str> = teeth.iter().filter(|v| matches!(v.get("ok"), Some(Val::Bool(false)))).map(|v| v.str_at("error").unwrap()).collect();
    assert!(!refused.is_empty(), "the perturbed parse chain was never refused");
    for e in &refused {
        assert!(e.contains("[PARSE-OBS]") && e.contains("the parse-built row differs from the step-built row") && e.contains("first cell"), "{e}");
    }
    eprintln!("teeth: {} of {} battles refused, e.g. {}", refused.len(), teeth.len(), refused[0]);
}

/// The COST of the mode (release): the same battles through one persistent child, OFF vs ON, per
/// decision. `cargo test --release --test sim_bridge_core_obs_test -- --ignored --nocapture
/// bench_core_obs_cost`. Also writes the OFF script to `$CARGO_TARGET_TMPDIR/core_obs_off.txt`
/// (the byte comparison against a pre-change binary is run by hand on it).
#[test]
#[ignore]
fn bench_core_obs_cost() {
    let n: usize = std::env::var("CORE_OBS_BENCH_BATTLES").ok().and_then(|s| s.parse().ok()).unwrap_or(17);
    let reps: usize = std::env::var("CORE_OBS_BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(7);
    let teams = corpus_teams();
    let mut bs = Vec::new();
    for rep in 0..(n.div_ceil(teams.len())) {
        for (i, t) in teams.iter().enumerate() {
            if bs.len() < n {
                let seed = format!("{},{},{},{}", 11 + i, 5 * rep + 1, 29 + i, 31 + rep);
                bs.push(generate(&format!("bench{rep}_{i}"), "gen3ou", &seed, t.clone(), (rep * 1000 + i) as u64, None));
            }
        }
    }
    let off = bridge_script(&bs, "");
    std::fs::write(concat!(env!("CARGO_TARGET_TMPDIR"), "/core_obs_off.txt"), &off).unwrap();
    std::fs::write(concat!(env!("CARGO_TARGET_TMPDIR"), "/core_obs_p1.txt"), bridge_script(&bs, &core_obs("[\"p1\"]", ClockConfig::default()))).unwrap();
    let modes = [("off", off.clone()), ("p1", bridge_script(&bs, &core_obs("[\"p1\"]", ClockConfig::default()))), ("both", bridge_script(&bs, &core_obs(BOTH, ClockConfig::default())))];
    let mut times: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
    let mut frames = [0usize; 3];
    for _ in 0..reps {
        for (m, (_, script)) in modes.iter().enumerate() {
            let t = std::time::Instant::now();
            let out = sim_bridge(script.clone());
            times[m].push(t.elapsed().as_secs_f64());
            frames[m] = out.lines().filter(|l| l.starts_with("__OBS__")).count();
        }
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let base = med(&mut times[0].clone());
    let cmds: usize = bs.iter().map(|b| b.cmds.len()).sum();
    println!("bench: {} battles, {cmds} commands, reps {reps}, debug_assertions {}", bs.len(), cfg!(debug_assertions));
    for (m, (name, _)) in modes.iter().enumerate() {
        let t = med(&mut times[m]);
        let per = if frames[m] > 0 { format!(", +{:.1} µs per frame", (t - base) * 1e6 / frames[m] as f64) } else { String::new() };
        println!("  {name:>4}: median {:.1} ms ({:.1} µs per command), {} frames{per}", t * 1e3, t * 1e6 / cmds as f64, frames[m]);
    }
}


/// The battles both benches replay: `n` corpus team pairs under the seeded random policy, `gen3ou`,
/// no forfeit.
fn bench_battles(n: usize) -> Vec<Battle> {
    let teams = corpus_teams();
    let mut bs = Vec::new();
    for rep in 0..(n.div_ceil(teams.len())) {
        for (i, t) in teams.iter().enumerate() {
            if bs.len() < n {
                let seed = format!("{},{},{},{}", 11 + i, 5 * rep + 1, 29 + i, 31 + rep);
                bs.push(generate(&format!("bench{rep}_{i}"), "gen3ou", &seed, t.clone(), (rep * 1000 + i) as u64, None));
            }
        }
    }
    bs
}

/// Named wall-clock accumulators, reported in insertion order.
#[derive(Default)]
struct Stages(Vec<(&'static str, bool, std::time::Duration)>);
impl Stages {
    /// `prod`: the stage is part of the shipped pipeline (summed into the total); otherwise it is an
    /// ATTRIBUTION re-run of a piece of a production stage on the same input.
    fn add(&mut self, name: &'static str, prod: bool, d: std::time::Duration) {
        match self.0.iter_mut().find(|(n, _, _)| *n == name) {
            Some(s) => s.2 += d,
            None => self.0.push((name, prod, d)),
        }
    }
}

/// The PER-STAGE cost of the core observation mode, in-process (release): the bridge's
/// `CoreObs::step` pipeline replayed over `bench_core_obs_cost`'s battles, each stage timed on its
/// own. `cargo test --release --test sim_bridge_core_obs_test -- --ignored --nocapture
/// bench_core_obs_stages` (`CORE_OBS_BENCH_BATTLES`, `CORE_OBS_BENCH_REPS`, `CORE_OBS_BENCH_SIDES`
/// = `p1` | `both`).
///
/// PRODUCTION stages (`PROD`; their sum is the mode's in-process cost per frame): `collect` (the
/// write's new lines), `fold` (`parse_advance_lean` of the bridge's chain — `parse_root_unrecorded`,
/// the trackers on and no native record — the parse, the reading
/// folds, the trackers and, at a decision, `present()` + legality + the tracker decide),
/// `obs_json` (`wire::obs_json_into`, the bridge's own frame builder: the encode with the view
/// memoized, legality, `choice_tokens`, the mask and the JSON), `write` (the line into a buffer).
/// ATTRIBUTION stages (`attr`; pieces of a production stage re-run standalone on the same input,
/// NOT summed): `parse` (`Line::parse`), `req_json` (one `Val::parse` of each request payload),
/// `reader` (M1's reading fold), `board_req` / `board_other` (the board reading on request / other
/// lines), `plain` (a TRACKERLESS parse chain: parse + both reading folds), `present`, `legal`,
/// `decide` (the tracker decide replayed from the previous decision's state), `encode`, `tokens`
/// (legality + tokens + mask), `frame_json` (the frame + the tokens JSON). Every replayed result is
/// ASSERTED equal to the chain's (the standalone board, the view, the tracker state, the frame), so
/// each number is about the code that ships.
#[test]
#[ignore]
fn bench_core_obs_stages() {
    use pokesim::core_events::reading::Reader;
    use pokesim::core_events::{Kw, Line};
    use pokesim::present::board_reading::BoardReading;
    use pokesim::trackers::turnview::DamagingMove;
    use std::hint::black_box;
    use std::time::Instant;

    let n: usize = std::env::var("CORE_OBS_BENCH_BATTLES").ok().and_then(|s| s.parse().ok()).unwrap_or(17);
    let reps: usize = std::env::var("CORE_OBS_BENCH_REPS").ok().and_then(|s| s.parse().ok()).unwrap_or(5);
    let both = std::env::var("CORE_OBS_BENCH_SIDES").is_ok_and(|s| s == "both");
    let sides: Vec<usize> = if both { vec![0, 1] } else { vec![0] };
    let bs = bench_battles(n);
    let dex = Dex::for_gen(3);
    let cfg = ClockConfig::default();

    let mut per_rep: Vec<Stages> = Vec::new();
    let (mut frames, mut writes, mut lines_n) = (0usize, 0usize, 0usize);
    for _ in 0..reps {
        let mut st = Stages::default();
        (frames, writes, lines_n) = (0, 0, 0);
        let mut sink: Vec<u8> = Vec::with_capacity(1 << 16);
        for b in &bs {
            let mut sess = BridgeSession::new_construct_turn0(&b.opts(), &dex).expect("session");
            let mut trk: [Option<BattleVersion>; 2] = [0, 1].map(|s| {
                sides.contains(&s).then(|| BattleVersion::parse_root_unrecorded(s, NAMES[s], Some(&b.teams[s]), cfg).expect("root"))
            });
            let mut plain: [Option<BattleVersion>; 2] = [0, 1].map(|s| {
                sides.contains(&s).then(|| BattleVersion::parse_root(s, NAMES[s], Some(&b.teams[s])).expect("root"))
            });
            let mut readers = [Reader::new(0), Reader::new(1)];
            let mut boards = [0, 1].map(|s| BoardReading::new(s, NAMES[s], Some(&b.teams[s])).expect("board"));
            // the tracker state at each side's previous decision + the readings since (the decide replay)
            let mut pre: [Option<pokesim::trackers::SideTrackers>; 2] = [0, 1].map(|s| trk[s].as_ref().and_then(|v| v.trackers(s).cloned()));
            let mut pending: [Vec<pokesim::core_events::Reading>; 2] = [Vec::new(), Vec::new()];
            let mut decided = [0u32; 2];
            let mut emitted = 0usize;
            let mut cmds = b.cmds.iter();
            loop {
                let from = emitted;
                for &side in &sides {
                    // ---- PROD: collect
                    let t0 = Instant::now();
                    let new: Vec<&str> = sess.chunks().chunks[from..]
                        .iter()
                        .filter(|c| c.side == side)
                        .flat_map(|c| c.lines.iter().map(String::as_str))
                        .collect();
                    st.add("collect", true, t0.elapsed());
                    lines_n += new.len();
                    // ---- attr: parse / req_json / reader / board, standalone
                    let mut parsed = Vec::with_capacity(new.len());
                    let t0 = Instant::now();
                    for l in &new {
                        parsed.push(black_box(Line::parse(l).expect("parse")));
                    }
                    st.add("parse", false, t0.elapsed());
                    for l in parsed.iter().filter(|l| l.kw == Kw::Request) {
                        let sm = l.split_message();
                        if sm.len() > 2 && !sm[2].is_empty() {
                            let j = sm[2..].join("|");
                            let t0 = Instant::now();
                            black_box(Val::parse(&j).expect("request json"));
                            st.add("req_json", false, t0.elapsed());
                        }
                    }
                    let t0 = Instant::now();
                    for l in &parsed {
                        black_box(readers[side].feed(l).expect("reader"));
                    }
                    st.add("reader", false, t0.elapsed());
                    for l in &parsed {
                        let t0 = Instant::now();
                        boards[side].feed(l).expect("board");
                        st.add(if l.kw == Kw::Request { "board_req" } else { "board_other" }, false, t0.elapsed());
                    }
                    // ---- attr: the trackerless chain
                    let p = plain[side].take().unwrap();
                    let t0 = Instant::now();
                    let p = p.parse_advance(&new).expect("plain parse");
                    st.add("plain", false, t0.elapsed());
                    plain[side] = Some(p);
                    // ---- PROD: the tracker chain
                    let c = trk[side].take().unwrap();
                    let t0 = Instant::now();
                    let next = c.parse_advance_lean(&new).expect("parse");
                    st.add("fold", true, t0.elapsed());
                    let s = next.stream(side).unwrap();
                    assert!(s.board_reading == boards[side], "{}: the standalone board drifted from the chain's", b.label);
                    // the lean chain keeps no events; the trackerless chain's readings are the same fold's
                    pending[side].extend(plain[side].as_ref().unwrap().events(side).iter().flat_map(|e| e.readings.iter().cloned()));
                    let decisions = next.trackers(side).map_or(0, |t| t.decisions);
                    if let Some(d) = next.decision(side) {
                        assert_eq!(decisions, decided[side] + 1);
                        assert_eq!(d.line + 1, s.lines);
                        let br = &s.board_reading;
                        // ---- attr: present / legal / decide, replayed
                        let t0 = Instant::now();
                        let view = present::present(br).expect("present");
                        st.add("present", false, t0.elapsed());
                        assert!(&view == next.view(side).unwrap(), "present replay");
                        let t0 = Instant::now();
                        let legal = present::legal_actions(br).expect("legal");
                        black_box(present::mask(&legal));
                        st.add("legal", false, t0.elapsed());
                        let dm = br.last_damaging_move(1).map(|d| DamagingMove {
                            user_species: Some(d.user_species.clone()),
                            target_species: Some(d.target_species.clone()),
                            target_status: d.target_status.map(|s| s.live().to_uppercase()),
                            move_id: Some(d.move_id.clone()),
                            effectiveness: Some(d.effectiveness),
                        });
                        let mut tr = pre[side].take().unwrap();
                        let pend = std::mem::take(&mut pending[side]);
                        let t0 = Instant::now();
                        tr.decide(&view, Some(&legal), &pend, dm, pokesim::trackers::dex()).expect("decide");
                        st.add("decide", false, t0.elapsed());
                        assert!(&tr == next.trackers(side).unwrap(), "{}: the decide replay differs from the chain's", b.label);
                        pre[side] = Some(tr);
                        // ---- PROD: the frame (`wire::obs_json_into`, what the bridge ships) + its write
                        let t0 = Instant::now();
                        let mut obs = Vec::with_capacity(wire::FRAME_LEN + 512);
                        obs.extend_from_slice(if side == 0 { b"__OBS__ p1 " } else { b"__OBS__ p2 " });
                        wire::obs_json_into(&next, side, d.line, decided[side], &mut obs).expect("obs json");
                        obs.push(b'\n');
                        st.add("obs_json", true, t0.elapsed());
                        let t0 = Instant::now();
                        sink.extend_from_slice(&obs);
                        st.add("write", true, t0.elapsed());
                        // ---- attr: the frame's pieces — the encode, legality + tokens + mask, the JSON
                        let t0 = Instant::now();
                        let mut row = [0.0f32; OBS_DIM];
                        next.encode(side, &mut row).expect("encode");
                        st.add("encode", false, t0.elapsed());
                        let t0 = Instant::now();
                        let legal = next.legal(side).expect("legal");
                        let tokens = present::choice_tokens(br, &legal).expect("tokens");
                        black_box(present::mask(&legal));
                        st.add("tokens", false, t0.elapsed());
                        let t0 = Instant::now();
                        let mut f = String::with_capacity(wire::FRAME_LEN);
                        wire::frame_into(&row, &mut f);
                        black_box(present::tokens_json(&tokens));
                        st.add("frame_json", false, t0.elapsed());
                        let at = b"__OBS__ p1 {\"frame\":".len();
                        assert!(&obs[at..at + f.len()] == f.as_bytes(), "the frame piece is not the shipped frame");
                        sink.clear();
                        frames += 1;
                    } else {
                        assert_eq!(decisions, decided[side]);
                    }
                    decided[side] = decisions;
                    trk[side] = Some(next);
                }
                writes += 1;
                emitted = sess.chunks().chunks.len();
                if sess.is_ended() {
                    break;
                }
                let Some(cmd) = cmds.next() else { break };
                let (verb, rest) = cmd.split_once(' ').unwrap();
                let (side_tok, tok) = rest.split_once(' ').unwrap_or((rest, ""));
                let side = if side_tok == "p1" { 0 } else { 1 };
                match verb {
                    "CHOOSE" => {
                        if let Some(c) = trk[side].as_mut() {
                            c.note_choice(side, tok);
                        }
                        sess.feed_cmd(Cmd { side, choice: parse_choice(tok).expect("choice") }, &dex);
                    }
                    "FORCELOSE" => sess.forfeit(side),
                    v => panic!("verb {v}"),
                }
            }
        }
        per_rep.push(st);
    }
    println!(
        "stages: {} battles, sides {:?}, reps {reps}, {frames} frames, {writes} writes, {lines_n} lines ({:.1} per frame), debug_assertions {}",
        bs.len(), sides, lines_n as f64 / frames as f64, cfg!(debug_assertions)
    );
    println!("  µs per FRAME, median of {reps} reps");
    let mut total = 0.0;
    for (k, (name, prod, _)) in per_rep[0].0.iter().enumerate() {
        let mut v: Vec<f64> = per_rep.iter().map(|r| r.0[k].2.as_secs_f64() * 1e6 / frames as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let m = v[v.len() / 2];
        if *prod {
            total += m;
        }
        println!("  {} {name:>11}: {m:8.2}", if *prod { "PROD" } else { "attr" });
    }
    println!("  PROD       total: {total:8.2}");
}
