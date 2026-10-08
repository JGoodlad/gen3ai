//! A BOT OVER A SIDE STREAM IS THE ENV CORE'S BOT (poke-env retirement P6; `bot_side.rs`,
//! `bin/bot_reader.rs`).
//!
//! Two bots play whole battles in a `BridgeSession`, each deciding ONLY from its own side's shipped
//! lines folded through [`BotSide`] (the `SideReader` chain a websocket client's reader uses). Beside
//! them a [`Game`] — the env core's own row path (`tests/search_game_test.rs` holds it byte-equal to
//! the pool) — replays the same commands, and at every decision an identically seeded ORACLE [`Bot`]
//! decides on the Game's reading over the Game's tokens. Every decision must agree: the token, the
//! action index and both RNG stream offsets. Each `BotSide` plays TWO battles back to back without
//! being rebuilt, as a websocket client's one bot per half-series does — the streams must carry over
//! (a reader that re-seeded its bot per battle diverges from the oracle in battle two).

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::trackers::clock::ClockConfig;
use pokesim_env::bot_side::BotSide;
use pokesim_env::bots::{Bot, Kind};
use pokesim_env::opponents::stream_seed;
use pokesim_env::search::game::{Game, Log};

mod common;

const NAMES: [&str; 2] = ["bsone", "bstwo"];
const MAX_CMDS: usize = 2_000;

struct Tally {
    decisions: usize,
    drew: usize,
}

/// One battle: `sides` (the readers under test) vs `oracle` (the env core's bots on a `Game`).
fn battle(dex: &Dex, teams: [&str; 2], seed: &str, sides: &mut [BotSide; 2], oracle: &mut [Bot; 2], t: &mut Tally) {
    let log = Log {
        format_id: "gen3ou".into(),
        seed: seed.into(),
        names: [NAMES[0].into(), NAMES[1].into()],
        teams: [teams[0].into(), teams[1].into()],
        cmds: Vec::new(),
    };
    let opts = BattleOptions {
        format_id: log.format_id.clone(),
        seed: Some(log.seed.clone()),
        p1: PlayerOptions { name: NAMES[0].into(), team: PackedTeam(teams[0].into()) },
        p2: PlayerOptions { name: NAMES[1].into(), team: PackedTeam(teams[1].into()) },
    };
    let mut sess = BridgeSession::new_construct_turn0(&opts, dex).expect("battle starts");
    let mut game = Game::start(&log, ClockConfig::default()).expect("game starts");
    for (s, side) in sides.iter_mut().enumerate() {
        side.open(s, NAMES[s], Some(teams[s])).expect("reader opens");
    }
    let mut emitted = 0usize;
    let mut cmds = 0usize;
    while !sess.is_ended() {
        let chunks = &sess.chunks().chunks[emitted..];
        let mut chosen: [Option<(usize, String)>; 2] = [None, None];
        for s in 0..2 {
            let lines: Vec<&str> = chunks.iter().filter(|c| c.side == s).flat_map(|c| c.lines.iter().map(String::as_str)).collect();
            let adv = sides[s].advance(&lines).unwrap_or_else(|e| panic!("p{} reader/bot refused: {e}", s + 1));
            let Some(c) = adv.choice else {
                assert!(game.open(s).is_none(), "p{}: the env core opened a decision the side reader did not", s + 1);
                continue;
            };
            assert!(adv.frame.is_some(), "a decision without its frame");
            let open = game.open(s).unwrap_or_else(|| panic!("p{}: the side reader decided where the env core did not", s + 1));
            let want = oracle[s].decide(game.reading(s), &open.tokens).expect("the oracle bot decides");
            assert_eq!(
                (c.decision.token.as_str(), c.decision.index, c.choice_words, c.protect_words),
                (want.token.as_str(), want.index, oracle[s].choice.words, oracle[s].protect.words),
                "p{} ({:?}) decision {} of seed {seed}",
                s + 1,
                oracle[s].kind,
                t.decisions
            );
            t.decisions += 1;
            if c.choice_words + c.protect_words > 0 {
                t.drew += 1;
            }
            chosen[s] = Some((c.decision.index.expect("an index"), c.decision.token));
        }
        emitted = sess.chunks().chunks.len();
        if chosen.iter().all(Option::is_none) {
            panic!("seed {seed}: the battle is live but neither side has a decision (cmds {cmds})");
        }
        for (s, ch) in chosen.into_iter().enumerate() {
            let Some((idx, tok)) = ch else { continue };
            sides[s].note_choice(&tok).expect("note");
            game.feed(s, idx as i32).expect("the env core takes the bot's index");
            sess.feed_cmd(Cmd { side: s, choice: parse_choice(&tok).expect("a parseable token") }, dex);
            cmds += 1;
        }
        assert!(cmds < MAX_CMDS, "seed {seed}: {cmds} commands without an end");
    }
    assert!(game.is_ended(), "the session ended but the env core's game did not");
    assert_eq!(game.winner(), sess.winner(), "the two roads disagree on the winner");
}

#[test]
fn a_bot_over_its_side_stream_decides_as_the_env_cores_bot() {
    let dex = Dex::for_gen(3);
    let teams = common::corpus_teams();
    let pairs = [
        ("random", "heuristic"),
        ("heuristic2", "staller"),
        ("staller_v2", "aggressive"),
        ("aggressive_v2", "setup_sweep"),
        ("setup_sweep_v2", "staller_v2"),
    ];
    let mut t = Tally { decisions: 0, drew: 0 };
    for (k, (a, b)) in pairs.iter().enumerate() {
        let seed = 1_000 + k as u64;
        let mut sides = [
            BotSide::new(a, stream_seed(seed, 0, 0), stream_seed(seed, 0, 1)).unwrap(),
            BotSide::new(b, stream_seed(seed, 1, 0), stream_seed(seed, 1, 1)).unwrap(),
        ];
        let mut oracle = [
            Bot::new(Kind::from_name(a).unwrap(), stream_seed(seed, 0, 0), stream_seed(seed, 0, 1)),
            Bot::new(Kind::from_name(b).unwrap(), stream_seed(seed, 1, 0), stream_seed(seed, 1, 1)),
        ];
        for g in 0..2usize {
            let i = (k * 4 + g * 2) % teams.len();
            let j = (i + 1 + g) % teams.len();
            let bseed = format!("{},{},{},{}", 11 + k, 22 + g, 33, 44);
            battle(&dex, [&teams[i], &teams[j]], &bseed, &mut sides, &mut oracle, &mut t);
        }
    }
    assert!(t.decisions >= 200, "only {} bot decisions were compared", t.decisions);
    assert!(t.drew > 0, "no compared decision drew from a bot stream — the RNG carry-over is untested");
    println!("BOT_SIDE decisions={} drew={}", t.decisions, t.drew);
}

#[test]
fn an_unknown_bot_and_a_feed_before_open_are_refused() {
    assert!(BotSide::new("baitbot", 1, 2).err().expect("refused").contains("not a bot"));
    let mut b = BotSide::new("random", 1, 2).unwrap();
    assert!(b.advance(&["|turn|1"]).unwrap_err().contains("FEED before OPEN"));
    assert!(b.note_choice("move 1").unwrap_err().contains("CHOOSE before OPEN"));
}
