//! THE PERSISTED TRACE of a finished episode (M5 Lane H — eval on the core).
//!
//! An eval game the forensic quota keeps is persisted from its INPUT LOG (`episode::Finished`'s
//! script — `InputLog::script`), never from a live session: [`trace_json`] replays the script on a
//! fresh engine session exactly as the core played it (`BridgeSession::new_construct_turn0`, the raw
//! staged seed, every `CHOOSE` token through `parse_choice`, a `FORCELOSE` as the real forfeit) and
//! returns, as one JSON object:
//!
//! * `records` — each side's `gen3_core_event_v1` record (`pokesim::core_events::record`, the TEXT
//!   plus the typed stream; `record::write`, re-read and re-parsed here before it is returned, so a
//!   record that does not round-trip is refused rather than persisted);
//! * `recon` — the `gen3_bridge_recon_record_v1` reconstruction record `sim_bridge` emits as
//!   `__RECON__` for the same battle (`>start` with the resolved seed, both `>player` lines, the
//!   committed choices from the engine's own script; `commands` = every `CHOOSE` as `[side, token]`
//!   plus `["forcelose", side]`), the file the prober's replay views drive;
//! * `winner` (1 / 2 / 0 = tie) and `turn` — the replay's own end, which the host checks against the
//!   core's `Finished` (a replay that ends elsewhere is refused).
//!
//! Pure: no core, no env, no thread — the FFI's `rust_env_trace_json` and any Rust caller use it.
//! Only a persistence point pays for it (program §2 M1: "the end of an eval game").

use pokesim::battle::{BattleOptions, PackedTeam, PlayerOptions};
use pokesim::bridge::{parse_choice, BridgeSession, Cmd};
use pokesim::core_events::record::{self, Header, Path, Record};
use pokesim::dex::Dex;
use pokesim::json::Json;

use crate::core::refusal::json_str;

fn player(v: &Json, key: &str) -> Result<(PlayerOptions, String), String> {
    let p = v.get(key).ok_or_else(|| format!("trace: START has no {key}"))?;
    let name = p.str_at("name").ok_or("trace: START player name")?.to_string();
    let team = p.str_at("team").ok_or("trace: START player team")?.to_string();
    let payload = format!("{{\"name\":{},\"team\":{}}}", json_str(&name), json_str(&team));
    Ok((PlayerOptions { name, team: PackedTeam(team) }, payload))
}

fn side_of(tok: &str) -> Result<usize, String> {
    match tok {
        "p1" => Ok(0),
        "p2" => Ok(1),
        o => Err(format!("trace: bad side {o:?}")),
    }
}

/// See the module docs. `commit` is stamped into each record's header (`core_commit`); `label` is
/// the records' `battle` field.
pub fn trace_json(script: &str, commit: &str, label: &str) -> Result<String, String> {
    let dex = Dex::for_gen(3);
    let mut lines = script.lines().map(str::trim).filter(|l| !l.is_empty());
    let start = lines.next().ok_or("trace: empty script")?;
    let rest = start.strip_prefix("START ").ok_or("trace: the script does not begin with START")?;
    let v = Json::parse(rest).map_err(|e| format!("trace: START JSON: {e}"))?;
    let format_id = v.str_at("formatid").ok_or("trace: START formatid")?.to_string();
    let seed = v.str_at("seed").ok_or("trace: START seed (the core stages a string seed)")?.to_string();
    let (p1, p1_json) = player(&v, "p1")?;
    let (p2, p2_json) = player(&v, "p2")?;
    let opts = BattleOptions { format_id: format_id.clone(), seed: Some(seed.clone()), p1, p2 };
    let mut sess = BridgeSession::new_construct_turn0_core(&opts, &dex)?;
    let mut cmds: Vec<(String, String)> = Vec::new();
    for line in lines {
        let (op, arg) = line.split_once(' ').unwrap_or((line, ""));
        match op {
            "CHOOSE" => {
                let (side, tok) = arg.split_once(' ').ok_or_else(|| format!("trace: malformed {line:?}"))?;
                let s = side_of(side)?;
                let choice = parse_choice(tok).ok_or_else(|| format!("trace: unparseable token {tok:?}"))?;
                if sess.is_ended() {
                    return Err(format!("trace: {line:?} after the battle ended"));
                }
                cmds.push((side.to_string(), tok.to_string()));
                sess.feed_cmd(Cmd { side: s, choice }, &dex);
            }
            "FORCELOSE" => {
                let s = side_of(arg.trim())?;
                cmds.push(("forcelose".to_string(), arg.trim().to_string()));
                sess.forfeit(s);
            }
            "END" => break,
            other => return Err(format!("trace: unknown script command {other:?}")),
        }
        if let Some(f) = sess.fatal() {
            return Err(format!("trace: engine fatal while replaying: {f}"));
        }
    }
    if !sess.is_ended() {
        return Err("trace: the replayed script does not end the battle".into());
    }
    let mut recs: Vec<String> = Vec::with_capacity(2);
    for side in 0..2 {
        let step = sess.core_events(side)?;
        let r = Record::new(
            Header {
                event_schema: record::EVENT_SCHEMA.into(),
                core_commit: commit.to_string(),
                path: Path::Step,
                viewer: side as u8,
                format: format_id.clone(),
                showdown_version: None,
                battle: label.to_string(),
            },
            &step,
        );
        let bytes = record::write(&r);
        let back = record::read(&bytes).map_err(|e| format!("trace: p{} record does not read back: {e:?}", side + 1))?;
        if record::write(&back) != bytes {
            return Err(format!("trace: p{} record does not round-trip byte-identically", side + 1));
        }
        record::check_reparse(&back).map_err(|e| format!("trace: p{} record does not re-parse: {e:?}", side + 1))?;
        recs.push(bytes);
    }
    // The reconstruction record, `sim_bridge::emit_recon`'s shape.
    let mut input_log = vec![
        format!(">start {{\"formatid\":{},\"seed\":{}}}", json_str(&format_id), json_str(&seed)),
        format!(">player p1 {p1_json}"),
        format!(">player p2 {p2_json}"),
    ];
    for dec in sess.script() {
        for (side, ch) in [(1usize, dec.p1), (2usize, dec.p2)] {
            if let Some(c) = ch {
                let tok = match c {
                    pokesim::turn::Choice::Move(k) => format!("move {}", k + 1),
                    pokesim::turn::Choice::Switch(n) => format!("switch {}", n + 1),
                };
                input_log.push(format!(">p{side} {tok}"));
            }
        }
    }
    let recon = format!(
        "{{\"v\":1,\"format_id\":{},\"prng_seed\":{},\"input_log\":[{}],\"commands\":[{}]}}",
        json_str(&format_id),
        json_str(&seed),
        input_log.iter().map(|l| json_str(l)).collect::<Vec<_>>().join(","),
        cmds.iter().map(|(s, c)| format!("[{},{}]", json_str(s), json_str(c))).collect::<Vec<_>>().join(",")
    );
    let winner = match sess.winner() {
        Some(0) => 1,
        Some(1) => 2,
        _ => 0,
    };
    Ok(format!(
        "{{\"records\":{{\"p1\":{},\"p2\":{}}},\"recon\":{recon},\"winner\":{winner},\"turn\":{}}}",
        json_str(&recs[0]),
        json_str(&recs[1]),
        sess.turn()
    ))
}
