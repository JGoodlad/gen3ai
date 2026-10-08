//! `bot_reader` — a SCRIPTED BOT over one side's protocol stream, as a session a websocket client
//! feeds (poke-env retirement P6: `main.anchors`' `bot:<name>` our-side, `main.live.bot_reader`).
//!
//! `live_reader`'s protocol (`src/rust_sim/src/bin/live_reader.rs`, `designs/rust_sim/live_reader.md`)
//! with the bot added: the side's lines go through the SAME reader chain
//! (`pokesim::side_reader::SideReader`, via [`pokesim_env::bot_side::BotSide`]), and at every decision
//! the Lane-F bot ([`pokesim_env::bots`]) decides on that side's reading over the decision's own
//! choice tokens — the env core's bot, on the env core's reading.
//!
//! # Protocol (stdin / stdout, newline-delimited; one reply per command, in order)
//!
//! - `BOT <json>` `{"bot":<name>,"seed":<int ≤ 2^53-1>,"env":<int>}` → `__OK__` — install the bot
//!   whose streams are seeded exactly as the env core's `{"kind":"bot","seed":S}` route seeds env
//!   `env`'s bot (`opponents::stream_seed(S, env, k)`, choice k = 0, protect k = 1). It OUTLIVES every
//!   battle (one bot per half-series, its streams running across its battles); required before OPEN.
//! - `OPEN <json>` `{"side":"p1"|"p2","name":<str>,"team":<packed>|null}` → `__OK__` (a new battle)
//! - `FEED <json array of str>` → when the write ended at a decision: `__BOT__ <json>`
//!   (`{"index","token","choice_words","protect_words","branch"}`) then `__OBS__ <p1|p2> <frame json>`;
//!   then `__FED__ {"folded","decided","turn"}`
//! - `CHOOSE <token>` → `__OK__` (the choice sent, noted BEFORE the next FEED)
//! - `CLOSE` → `__OK__` · `END` → exit
//!
//! Any failure (a malformed command, a reader refusal, a BOT refusal, a panic) is ONE
//! `__ERR__ {"kind","message"}` line in place of the reply; the reader is then failed for the rest of
//! the battle (`SideReader`'s sticky rule).
use std::io::{self, BufRead, Write};

use pokesim::json::Json;
use pokesim_env::bot_side::BotSide;
use pokesim_env::opponents::stream_seed;

const MAX_SEED: f64 = 9_007_199_254_740_991.0; // 2^53 - 1, the route table's own bound

fn main() {
    let _ = pokesim::trackers::dex();
    let mut bot: Option<BotSide> = None;
    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let line = line.trim_end_matches(['\r', '\n']);
        if line.is_empty() {
            continue;
        }
        if line == "END" {
            return;
        }
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(&mut bot, line, &mut out)));
        match res {
            Ok(Ok(())) => {}
            Ok(Err((kind, msg))) => emit_err(&mut out, kind, &msg),
            Err(panic) => {
                let msg = panic
                    .downcast_ref::<&str>()
                    .map(|s| s.to_string())
                    .or_else(|| panic.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic".to_string());
                emit_err(&mut out, "panic", &msg);
            }
        }
    }
}

type Res = Result<(), (&'static str, String)>;

fn uint(v: &Json, key: &str) -> Result<u64, (&'static str, String)> {
    let x = v.get(key).and_then(Json::as_f64).ok_or(("command", format!("BOT: `{key}` must be an integer")))?;
    if x < 0.0 || x.fract() != 0.0 || x > MAX_SEED {
        return Err(("command", format!("BOT: `{key}` must be a non-negative integer <= 2^53-1, got {x}")));
    }
    Ok(x as u64)
}

fn handle(bot: &mut Option<BotSide>, line: &str, out: &mut impl Write) -> Res {
    let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
    match cmd {
        "BOT" => {
            let v = Json::parse(rest).map_err(|e| ("command", format!("BOT: bad json: {e}")))?;
            let name = v.str_at("bot").ok_or(("command", "BOT: `bot` is required".to_string()))?;
            let seed = uint(&v, "seed")?;
            let env = uint(&v, "env")? as usize;
            *bot = None;
            *bot = Some(BotSide::new(name, stream_seed(seed, env, 0), stream_seed(seed, env, 1)).map_err(|e| ("command", e))?);
            reply(out, b"__OK__\n");
            Ok(())
        }
        "OPEN" => {
            let b = bot.as_mut().ok_or(("command", "OPEN before BOT".to_string()))?;
            let v = Json::parse(rest).map_err(|e| ("command", format!("OPEN: bad json: {e}")))?;
            let side = match v.str_at("side") {
                Some("p1") => 0,
                Some("p2") => 1,
                other => return Err(("command", format!("OPEN: `side` must be \"p1\" / \"p2\", got {other:?}"))),
            };
            let name = v.str_at("name").ok_or(("command", "OPEN: `name` is required".to_string()))?;
            let team = match v.get("team") {
                None => None,
                Some(t) if t.is_null() => None,
                Some(t) => Some(t.as_str().ok_or(("command", "OPEN: `team` must be a string or null".to_string()))?),
            };
            b.open(side, name, team).map_err(|e| ("reader", e))?;
            reply(out, b"__OK__\n");
            Ok(())
        }
        "FEED" => {
            let b = bot.as_mut().ok_or(("command", "FEED before BOT".to_string()))?;
            let v = Json::parse(rest).map_err(|e| ("command", format!("FEED: bad json: {e}")))?;
            let arr = v.as_array().ok_or(("command", "FEED: expected a json array of lines".to_string()))?;
            let mut lines: Vec<&str> = Vec::with_capacity(arr.len());
            for x in arr {
                lines.push(x.as_str().ok_or(("command", "FEED: every line must be a string".to_string()))?);
            }
            // The reader chain's refusals are all `core_obs: …`; everything else is the bot's.
            let adv = b.advance(&lines).map_err(|e| (if e.starts_with("core_obs") { "reader" } else { "bot" }, e))?;
            let r = b.reader().ok_or(("command", "FEED before OPEN".to_string()))?;
            let mut buf = Vec::with_capacity(adv.frame.as_ref().map_or(0, Vec::len) + 256);
            if let (Some(frame), Some(choice)) = (adv.frame.as_ref(), adv.choice.as_ref()) {
                buf.extend_from_slice(b"__BOT__ ");
                buf.extend_from_slice(choice.json().as_bytes());
                buf.push(b'\n');
                buf.extend_from_slice(if r.side() == 0 { b"__OBS__ p1 " } else { b"__OBS__ p2 " });
                buf.extend_from_slice(frame);
                buf.push(b'\n');
            }
            let turn = r.version().and_then(|v| v.stream(r.side())).map_or(0, |s| s.board_reading.turn);
            buf.extend_from_slice(
                format!("__FED__ {{\"folded\":{},\"decided\":{},\"turn\":{}}}\n", r.folded(), r.decided(), turn).as_bytes(),
            );
            reply(out, &buf);
            Ok(())
        }
        "CHOOSE" => {
            let b = bot.as_mut().ok_or(("command", "CHOOSE before BOT".to_string()))?;
            if rest.is_empty() {
                return Err(("command", "CHOOSE: empty token".to_string()));
            }
            b.note_choice(rest).map_err(|e| ("command", e))?;
            reply(out, b"__OK__\n");
            Ok(())
        }
        "CLOSE" => {
            reply(out, b"__OK__\n");
            Ok(())
        }
        other => Err(("command", format!("unknown command {other:?}"))),
    }
}

fn reply(out: &mut impl Write, bytes: &[u8]) {
    out.write_all(bytes).ok();
    out.flush().ok();
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn emit_err(out: &mut impl Write, kind: &str, msg: &str) {
    let line = format!("__ERR__ {{\"kind\":{},\"message\":{}}}\n", json_str(kind), json_str(msg));
    reply(out, line.as_bytes());
}
