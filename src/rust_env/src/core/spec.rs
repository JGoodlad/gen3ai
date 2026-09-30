//! THE STARTUP DECLARATION — everything a pool will ever use, stated before anything runs (the M5
//! DECLARED LIFECYCLE: STARTUP declares and acquires, STEADY STATE acquires nothing).
//!
//! A [`Spec`] is parsed from ONE JSON object ([`Spec::from_json`]) so both front ends build it the
//! same way (the process front end receives the same text in a file). EVERY key is REQUIRED — the
//! clock flags are read, never defaulted (`sim_bridge`'s `core_obs` rule) — and an unknown key is
//! refused, so a typo can never select a default silently. `turn_limit` and `bank_dir` are present
//! but may be `null`. `turn_limit` is the STALL FORFEIT threshold and `terminal` the terminal-reward
//! declaration (M5 Lane D, `crate::episode`).
//!
//! ```text
//! {"n": 48, "threads": 8, "format_id": "gen3ou", "names": ["p1name", "p2name"],
//!  "teams": ["<packed>", …], "decision_tense": false, "switch_freeze": false,
//!  "turn_limit": null | <int>,
//!  "terminal": {"victory_value": <num>, "terminal_indicator": <bool>, "draw_penalty": <num>,
//!               "timeout_turn_cap": <int>},
//!  "refusal_budget": <int>, "bank_dir": null | "<abs path>",
//!  "labels": ["<family>", …],
//!  "opponents": [{"kind": "external"} | {"kind": "policy", "slot": <int>} | {"kind": "bot", "bot": "<name>"}, …]}
//! ```

use std::path::PathBuf;

use pokesim::json::Json;
use pokesim::trackers::clock::ClockConfig;

/// See the module docs.
#[derive(Clone, Debug, PartialEq)]
pub struct Spec {
    /// Envs in the pool (the columns' `N`). Fixed for the pool's life.
    pub n: usize,
    /// Worker threads (clamped to `n`; `<= 1` steps on the caller's thread). Fixed for the pool's life.
    pub threads: usize,
    /// The battle format (`gen3ou`).
    pub format_id: String,
    /// The two player names the battles are started with (they reach the protocol, and each parse
    /// chain finds its side by name).
    pub names: [String; 2],
    /// The TEAM TABLE — packed teams, indexed by the `ep_team` column. Validated (unpacked) at startup.
    pub teams: Vec<String>,
    /// The progress clock's two flags (training's `--progress-decision-tense` /
    /// `--progress-switch-freeze`).
    pub clock: ClockConfig,
    /// The STALL FORFEIT threshold (`StallConfig().threshold` in production): at a p1 decision whose
    /// turn is `>=` it, p1 forfeits instead of acting (`FORCELOSE p1` in its input log; nothing else
    /// is fed that op). `None`: no forfeit (harnesses only). `crate::episode`.
    pub turn_limit: Option<u32>,
    /// The terminal reward (`crate::episode::Terminal`).
    pub terminal: crate::episode::Terminal,
    /// Quarantines allowed over the pool's life; the next one is a `BUDGET` failure (a refusal STORM
    /// is systemic). The refusal bank reserves exactly this capacity at startup.
    pub refusal_budget: usize,
    /// Where each quarantined battle's input log is written the moment it is banked (`None`: kept
    /// in memory only).
    pub bank_dir: Option<PathBuf>,
    /// The LABEL FAMILIES the core writes (Lane C; `crate::labels::declare` — a family the core
    /// does not build is refused here, at startup), in table order.
    pub labels: Vec<&'static str>,
    /// The OPPONENT ROUTE TABLE (M5 Lane E; `crate::opponents`): the `ep_opp` column indexes it.
    pub opponents: crate::opponents::Routes,
}

/// GENERATED from `protocol.SPEC_KEYS` (one table for both languages).
use super::columns::SPEC_KEYS as KEYS;

fn uint(v: &Json, key: &str) -> Result<u64, String> {
    let x = v.get(key).and_then(Json::as_f64).ok_or_else(|| format!("spec: `{key}` must be a non-negative integer"))?;
    if x < 0.0 || x.fract() != 0.0 || x > 4.0e9 {
        return Err(format!("spec: `{key}` must be a non-negative integer, got {x}"));
    }
    Ok(x as u64)
}

impl Spec {
    /// Parse and validate (see the module docs). The error names the key.
    pub fn from_json(text: &str) -> Result<Spec, String> {
        let v = Json::parse(text).map_err(|e| format!("spec: not JSON: {e}"))?;
        let obj = v.as_object().ok_or("spec: must be a JSON object")?;
        for k in obj.keys() {
            if !KEYS.contains(&k.as_str()) {
                return Err(format!("spec: unknown key {k:?}"));
            }
        }
        for k in KEYS {
            if !obj.contains_key(k) {
                return Err(format!("spec: missing key {k:?} (every key is required; null where allowed)"));
            }
        }
        let names = v.get("names").and_then(Json::as_array).ok_or("spec: `names` must be an array of two strings")?;
        if names.len() != 2 {
            return Err("spec: `names` must hold exactly two strings".into());
        }
        let name = |i: usize| names[i].as_str().map(str::to_string).ok_or("spec: `names` must hold strings");
        let teams = v
            .get("teams")
            .and_then(Json::as_array)
            .ok_or("spec: `teams` must be an array of packed teams")?
            .iter()
            .map(|t| t.as_str().map(str::to_string).ok_or("spec: every team must be a packed-team string".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let flag = |k: &str| v.get(k).and_then(Json::as_bool).ok_or_else(|| format!("spec: `{k}` must be a boolean"));
        let turn_limit = match v.get("turn_limit") {
            Some(j) if j.is_null() => None,
            _ => Some(uint(&v, "turn_limit")? as u32),
        };
        let bank_dir = match v.get("bank_dir") {
            Some(j) if j.is_null() => None,
            Some(j) => Some(PathBuf::from(j.as_str().ok_or("spec: `bank_dir` must be a path string or null")?)),
            None => None,
        };
        let labels = v
            .get("labels")
            .and_then(Json::as_array)
            .ok_or("spec: `labels` must be an array of label-family names")?
            .iter()
            .map(|t| t.as_str().map(str::to_string).ok_or("spec: every label family must be a string".to_string()))
            .collect::<Result<Vec<_>, _>>()?;
        let labels = crate::labels::declare(&labels)?;
        let opponents = crate::opponents::Routes::from_json(v.get("opponents"))?;
        let spec = Spec {
            n: uint(&v, "n")? as usize,
            threads: uint(&v, "threads")? as usize,
            format_id: v.str_at("format_id").ok_or("spec: `format_id` must be a string")?.to_string(),
            names: [name(0)?, name(1)?],
            teams,
            clock: ClockConfig { decision_tense: flag("decision_tense")?, switch_freeze: flag("switch_freeze")? },
            turn_limit,
            terminal: crate::episode::Terminal::from_json(v.get("terminal"))?,
            refusal_budget: uint(&v, "refusal_budget")? as usize,
            bank_dir,
            labels,
            opponents,
        };
        spec.validate()?;
        Ok(spec)
    }

    /// The structural checks (the team table's CONTENT is validated by the pool, which holds the dex).
    pub fn validate(&self) -> Result<(), String> {
        if self.n == 0 {
            return Err("spec: `n` must be >= 1".into());
        }
        if self.threads == 0 {
            return Err("spec: `threads` must be >= 1".into());
        }
        if self.format_id != "gen3ou" {
            return Err(format!("spec: `format_id` {:?} is not supported (gen3ou only)", self.format_id));
        }
        if self.names[0].is_empty() || self.names[1].is_empty() || self.names[0] == self.names[1] {
            return Err("spec: `names` must be two distinct non-empty strings".into());
        }
        if self.teams.is_empty() {
            return Err("spec: `teams` must hold at least one team".into());
        }
        if self.turn_limit == Some(0) {
            return Err("spec: `turn_limit` must be >= 1 or null".into());
        }
        self.terminal.validate()?;
        self.opponents.validate()?;
        if let Some(d) = &self.bank_dir {
            if !d.is_absolute() {
                return Err(format!("spec: `bank_dir` must be absolute, got {}", d.display()));
            }
        }
        Ok(())
    }

    /// The worker count the pool actually runs (`threads` clamped to `n`).
    pub fn effective_threads(&self) -> usize {
        self.threads.min(self.n)
    }

    /// The canonical JSON (round-trips through [`Spec::from_json`]).
    pub fn to_json(&self) -> String {
        let q = crate::core::refusal::json_str;
        let teams: Vec<String> = self.teams.iter().map(|t| q(t)).collect();
        format!(
            "{{\"n\":{},\"threads\":{},\"format_id\":{},\"names\":[{},{}],\"teams\":[{}],\"decision_tense\":{},\
             \"switch_freeze\":{},\"turn_limit\":{},\"terminal\":{},\"refusal_budget\":{},\"bank_dir\":{},\"labels\":[{}],\"opponents\":{}}}",
            self.n,
            self.threads,
            q(&self.format_id),
            q(&self.names[0]),
            q(&self.names[1]),
            teams.join(","),
            self.clock.decision_tense,
            self.clock.switch_freeze,
            self.turn_limit.map_or("null".to_string(), |t| t.to_string()),
            self.terminal.to_json(),
            self.refusal_budget,
            self.bank_dir.as_ref().map_or("null".to_string(), |d| q(&d.to_string_lossy())),
            self.labels.iter().map(|f| q(f)).collect::<Vec<_>>().join(","),
            self.opponents.to_json(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Spec {
        Spec {
            n: 3,
            threads: 2,
            format_id: "gen3ou".into(),
            names: ["a\"b".into(), "c".into()],
            teams: vec!["X|||".into()],
            clock: ClockConfig { decision_tense: true, switch_freeze: false },
            turn_limit: Some(300),
            terminal: crate::episode::Terminal { victory_value: 30.0, indicator: false, draw_penalty: -35.0, timeout_turn_cap: 250 },
            refusal_budget: 4,
            bank_dir: Some(PathBuf::from("/tmp/x")),
            labels: Vec::new(),
            opponents: crate::opponents::Routes(vec![
                crate::opponents::Route::External,
                crate::opponents::Route::Policy { slot: 2 },
            ]),
        }
    }

    #[test]
    fn the_json_round_trips() {
        let s = sample();
        assert_eq!(Spec::from_json(&s.to_json()).unwrap(), s);
        let mut t = s.clone();
        t.turn_limit = None;
        t.bank_dir = None;
        assert_eq!(Spec::from_json(&t.to_json()).unwrap(), t);
    }

    fn pairs() -> Vec<(&'static str, &'static str)> {
        vec![
            ("n", "3"), ("threads", "2"), ("format_id", "\"gen3ou\""), ("names", "[\"a\",\"b\"]"),
            ("teams", "[\"X|||\"]"), ("decision_tense", "false"), ("switch_freeze", "true"), ("turn_limit", "null"),
            ("terminal", "{\"victory_value\":1,\"terminal_indicator\":true,\"draw_penalty\":0,\"timeout_turn_cap\":250}"),
            ("refusal_budget", "0"), ("bank_dir", "null"), ("labels", "[]"),
            ("opponents", "[{\"kind\":\"external\"}]"),
        ]
    }

    fn text(p: &[(&str, &str)]) -> String {
        let body: Vec<String> = p.iter().map(|(k, v)| format!("\"{k}\":{v}")).collect();
        format!("{{{}}}", body.join(","))
    }

    #[test]
    fn every_key_is_required_and_no_other_key_is_accepted() {
        let full = pairs();
        assert!(Spec::from_json(&text(&full)).is_ok());
        for k in KEYS {
            let dropped: Vec<_> = full.iter().copied().filter(|(kk, _)| *kk != k).collect();
            let e = Spec::from_json(&text(&dropped)).unwrap_err();
            assert!(e.contains(k), "{k}: {e}");
        }
        let mut extra = full.clone();
        extra.push(("nn", "1"));
        assert!(Spec::from_json(&text(&extra)).unwrap_err().contains("unknown key"));
        let bad: Vec<_> = full.iter().map(|&(k, v)| if k == "decision_tense" { (k, "1") } else { (k, v) }).collect();
        assert!(Spec::from_json(&text(&bad)).unwrap_err().contains("decision_tense"));
    }
}
