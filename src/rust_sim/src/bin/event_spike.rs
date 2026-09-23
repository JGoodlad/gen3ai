//! `event_spike` — SPIKE binary (`rust_core_phase0_2026-09-23`), built only with
//! `--features event_spike`. NOT a production transport and not referenced by any Python path.
//!
//! Replays ONE battle from a `sim_bridge`-shaped command script on stdin —
//!
//! ```text
//! START <json>                 (the same START payload sim_bridge takes; `seed` required)
//! CHOOSE <p1|p2> <choice>      (every command the live child processed, in order)
//! FORCELOSE <p1|p2>
//! ```
//!
//! — through the SAME [`pokesim::bridge::BridgeSession`] the production `sim_bridge` drives,
//! and prints ONE JSON object:
//!
//! ```text
//! {"chunks":[{"side":0,"lines":[…]},…],      // the per-side stream, for a byte check vs live
//!  "recs":[<event_spike::Rec JSON>,…],        // one per OMNISCIENT line, typed at the source
//!  "names":{"p1":{nick:species,…},"p2":{…}}}  // the team sheet, for the reader's species key
//! ```
//!
//! Each rec's `line` is re-read from the final log, so a line the engine retro-edited
//! (`[miss]` / `[still]`) carries its final bytes.

use std::io::{self, BufRead, Write};

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::dex::Dex;
use pokesim::json::Json;
use pokesim::prng::normalize_seed;

fn q(s: &str) -> String {
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

fn player(v: &Json, key: &str) -> Result<PlayerOptions, String> {
    let p = v.get(key).ok_or_else(|| format!("START: missing {key}"))?;
    Ok(PlayerOptions {
        name: p.str_at("name").ok_or("START: name")?.to_string(),
        team: PackedTeam(p.str_at("team").ok_or("START: team")?.to_string()),
    })
}

fn seed_of(v: &Json) -> Result<String, String> {
    let j = v.get("seed").filter(|s| !s.is_null()).ok_or("START: this spike needs an explicit seed")?;
    if let Some(a) = j.as_array() {
        let parts: Vec<String> = a
            .iter()
            .map(|x| x.as_f64().map(|n| format!("{}", n as u64)).ok_or("seed element"))
            .collect::<Result<_, _>>()?;
        Ok(parts.join(","))
    } else {
        Ok(normalize_seed(j.as_str().ok_or("seed must be a string or array")?))
    }
}

/// `nick -> species display name` for one packed team (the SOURCE's team sheet). A packed set
/// is `nick|species|…` with an empty species meaning "same as nick".
fn names(packed: &str) -> Vec<(String, String)> {
    packed
        .split(']')
        .filter(|s| !s.is_empty())
        .map(|set| {
            let mut f = set.split('|');
            let nick = f.next().unwrap_or("").to_string();
            let sp = f.next().unwrap_or("").to_string();
            let sp = if sp.is_empty() { nick.clone() } else { sp };
            (nick, sp)
        })
        .collect()
}

fn run() -> Result<String, String> {
    let dex = Dex::for_gen(3);
    let stdin = io::stdin();
    let mut sess: Option<BridgeSession> = None;
    let mut teams: [String; 2] = [String::new(), String::new()];
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| e.to_string())?;
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        let (cmd, rest) = line.split_once(' ').unwrap_or((line, ""));
        match cmd {
            "START" => {
                let v = Json::parse(rest).map_err(|e| format!("START JSON: {e}"))?;
                let p1 = player(&v, "p1")?;
                let p2 = player(&v, "p2")?;
                teams = [p1.team.0.clone(), p2.team.0.clone()];
                let opts = BattleOptions {
                    format_id: v.str_at("formatid").ok_or("formatid")?.to_string(),
                    seed: Some(seed_of(&v)?),
                    p1,
                    p2,
                };
                let mut s = BridgeSession::new_construct_turn0(&opts, &dex)?;
                // The spike hooks record only while emission is enabled, which the session
                // turned on for its framing; nothing else to arm.
                let _ = &mut s;
                sess = Some(s);
            }
            "CHOOSE" => {
                let s = sess.as_mut().ok_or("CHOOSE before START")?;
                if s.is_ended() {
                    continue;
                }
                let (side, choice) = rest.split_once(' ').unwrap_or((rest, ""));
                let side = match side {
                    "p1" => 0,
                    "p2" => 1,
                    o => return Err(format!("bad side {o}")),
                };
                let choice = parse_choice(choice).ok_or_else(|| format!("bad choice {choice:?}"))?;
                s.feed_cmd(Cmd { side, choice }, &dex);
            }
            "FORCELOSE" => {
                let s = sess.as_mut().ok_or("FORCELOSE before START")?;
                let side = if rest.trim() == "p1" { 0 } else { 1 };
                s.forfeit(side);
            }
            "END" => break,
            o => return Err(format!("unknown command {o}")),
        }
    }
    let s = sess.ok_or("no START")?;
    let bs = s.battle_state().ok_or("no battle state")?;
    let final_lines = bs.log.lines();
    let mut recs = bs.log.spike.recs.clone();
    if recs.len() != final_lines.len() {
        return Err(format!(
            "CONSERVATION: {} source records vs {} log lines — a line bypassed push_raw",
            recs.len(),
            final_lines.len()
        ));
    }
    for (r, l) in recs.iter_mut().zip(final_lines.iter()) {
        r.line = l.0.clone();
    }
    let sink = pokesim::event_spike::Sink::from_recs(recs);
    let mut out = String::from("{\"chunks\":[");
    for (i, c) in s.chunks().chunks.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let lines: Vec<String> = c.lines.iter().map(|l| q(l)).collect();
        out.push_str(&format!("{{\"side\":{},\"lines\":[{}]}}", c.side, lines.join(",")));
    }
    out.push_str("],\"recs\":[");
    let jl = sink.to_jsonl();
    out.push_str(&jl.lines().collect::<Vec<_>>().join(","));
    out.push_str("],\"names\":{");
    for (side, team) in teams.iter().enumerate() {
        if side > 0 {
            out.push(',');
        }
        let pairs: Vec<String> =
            names(team).iter().map(|(n, sp)| format!("{}:{}", q(n), q(sp))).collect();
        out.push_str(&format!("\"p{}\":{{{}}}", side + 1, pairs.join(",")));
    }
    out.push_str("}}");
    Ok(out)
}

fn main() {
    match run() {
        Ok(s) => {
            let mut o = io::stdout().lock();
            let _ = o.write_all(s.as_bytes());
            let _ = o.write_all(b"\n");
        }
        Err(e) => {
            eprintln!("event_spike: {e}");
            std::process::exit(1);
        }
    }
}
