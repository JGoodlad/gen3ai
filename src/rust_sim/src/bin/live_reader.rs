//! `live_reader` — the READER SESSION a live websocket client feeds (poke-env retirement P4).
//!
//! One side of one battle, read from a FOREIGN stream (a Showdown server's, our `--server rust`
//! front end's, a public replay's) through the SAME chain `sim_bridge`'s core observation mode ships
//! to training ([`pokesim::side_reader::SideReader`]): protocol lines in, the training core's
//! `__OBS__` frame (row, mask, choice tokens, turn, line, rqid, n) out at every decision. The Python
//! side is `main.live.reader`; the websocket client that drives it is `main.live.client`.
//!
//! # Protocol (stdin / stdout, newline-delimited; one reply per command, in order)
//!
//! - `OPEN <json>` `{"side":"p1"|"p2","name":<str>,"team":<packed>|null}` → `__OK__`
//!   (a new battle: any previous reader is dropped)
//! - `FEED <json array of str>` — the lines this side was newly shipped in ONE write (the battle
//!   protocol only: the caller strips the room framing and every declared non-battle line) →
//!   `__OBS__ <p1|p2> <json>` when the write ended at a decision, then `__FED__ <json>`
//!   `{"folded":<int>,"decided":<int>,"turn":<int>}`
//! - `CHOOSE <token>` — the choice this side sends for its open decision, noted BEFORE the next FEED
//!   (`note_choice`) → `__OK__`
//! - `CLOSE` → `__OK__` (drops the reader)
//! - `END` → exit
//!
//! Any failure — a malformed command, a parse / fold / encode / alignment refusal, a panic — is ONE
//! `__ERR__ <json>` `{"kind":"…","message":"…"}` line IN PLACE of the reply, and the reader is then
//! FAILED for the rest of the battle (every later FEED is refused): a skipped frame must never pass for
//! a quiet one. The Python side turns every `__ERR__` into a T28 halt (`main.live.halt`).
use std::io::{self, BufRead, Write};

use pokesim::json::Json;
use pokesim::side_reader::SideReader;
use pokesim::trackers::clock::ClockConfig;

fn main() {
    // Warm the process-wide dex the trackers read, so the first decision pays no load mid-battle.
    let _ = pokesim::trackers::dex();
    let mut reader: Option<SideReader> = None;
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
        let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handle(&mut reader, line, &mut out)));
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

fn handle(reader: &mut Option<SideReader>, line: &str, out: &mut impl Write) -> Res {
    let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
    match cmd {
        "OPEN" => {
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
            *reader = None;
            *reader = Some(SideReader::new(side, name, team, ClockConfig::default()).map_err(|e| ("reader", e))?);
            reply(out, b"__OK__\n");
            Ok(())
        }
        "FEED" => {
            let r = reader.as_mut().ok_or(("command", "FEED before OPEN".to_string()))?;
            let v = Json::parse(rest).map_err(|e| ("command", format!("FEED: bad json: {e}")))?;
            let arr = v.as_array().ok_or(("command", "FEED: expected a json array of lines".to_string()))?;
            let mut lines: Vec<&str> = Vec::with_capacity(arr.len());
            for x in arr {
                lines.push(x.as_str().ok_or(("command", "FEED: every line must be a string".to_string()))?);
            }
            let mut frame = Vec::new();
            let adv = r.advance(&lines, &mut frame).map_err(|e| ("reader", e))?;
            let mut buf = Vec::with_capacity(frame.len() + 128);
            if adv.decided {
                buf.extend_from_slice(if r.side() == 0 { b"__OBS__ p1 " } else { b"__OBS__ p2 " });
                buf.extend_from_slice(&frame);
                buf.push(b'\n');
            }
            let turn = r
                .version()
                .and_then(|v| v.stream(r.side()))
                .map_or(0, |s| s.board_reading.turn);
            buf.extend_from_slice(
                format!("__FED__ {{\"folded\":{},\"decided\":{},\"turn\":{}}}\n", r.folded(), r.decided(), turn).as_bytes(),
            );
            reply(out, &buf);
            Ok(())
        }
        "CHOOSE" => {
            let r = reader.as_mut().ok_or(("command", "CHOOSE before OPEN".to_string()))?;
            if rest.is_empty() {
                return Err(("command", "CHOOSE: empty token".to_string()));
            }
            r.note_choice(rest);
            reply(out, b"__OK__\n");
            Ok(())
        }
        "PROBE" => {
            // The DRIFT SCAN's encoder check (never on the live path): encode the side's CURRENT reading
            // whether or not a decision is open — a spectator chain over a public replay takes none — so
            // an effect / volatile / status the encoder cannot classify surfaces as an `__ERR__` here.
            // A probe failure is reported, but it does NOT fail the reader (the read itself succeeded).
            let r = reader.as_ref().ok_or(("command", "PROBE before OPEN".to_string()))?;
            let v = r.version().ok_or(("command", "PROBE on a failed reader".to_string()))?;
            let mut row = vec![0.0f32; pokesim::encoder::OBS_DIM];
            let arr: &mut [f32; pokesim::encoder::OBS_DIM] =
                row.as_mut_slice().try_into().map_err(|_| ("probe", "row length".to_string()))?;
            v.encode(r.side(), arr).map_err(|e| ("probe", e.message().to_string()))?;
            if let Some(i) = row.iter().position(|x| !x.is_finite()) {
                return Err(("probe", format!("a non-finite cell at {i}")));
            }
            reply(out, b"__OK__\n");
            Ok(())
        }
        "CLOSE" => {
            *reader = None;
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
