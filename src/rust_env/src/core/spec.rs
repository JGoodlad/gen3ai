//! THE STARTUP DECLARATION — everything a pool will ever use, stated before anything runs (the M5
//! DECLARED LIFECYCLE: STARTUP declares and acquires, STEADY STATE acquires nothing).
//!
//! A [`Spec`] is parsed from ONE JSON object ([`Spec::from_json`]) so both front ends build it the
//! same way (the process front end receives the same text in a file). EVERY key is REQUIRED — a mode
//! is read, never defaulted (`sim_bridge`'s `core_obs` rule) — and an unknown key is refused (the two
//! clock keys, `decision_tense` / `switch_freeze`, were deleted with their trainer flags and are refused
//! as unknown), so a typo can never select a default silently. `turn_limit` and `bank_dir` are present
//! but may be `null`. `turn_limit` is the STALL FORFEIT threshold and `terminal` the terminal-reward
//! declaration (M5 Lane D, `crate::episode`).
//!
//! ```text
//! {"n": 48, "threads": 8, "format_id": "gen3ou", "names": ["p1name", "p2name"],
//!  "teams": ["<packed>", …],
//!  "turn_limit": null | <int>,
//!  "terminal": {"victory_value": <num>, "terminal_indicator": <bool>, "draw_penalty": <num>,
//!               "timeout_turn_cap": <int>},
//!  "refusal_budget": <int>, "bank_dir": null | "<abs path>",
//!  "labels": ["<family>", …],
//!  "opponents": [{"kind": "external"} | {"kind": "policy", "slot": <int>} | {"kind": "bot", "bot": "<name>"}, …],
//!  "oracle_reveal": "<spec label>" | [<p1 label>, <p2 label>]}
//! ```
//!
//! `oracle_reveal` is the ORACLE REVEAL (`pokesim::encoder::oracle`, a DIAGNOSTIC observation mode,
//! `--oracle-reveal <spec>`; the training observation's consumer of the information-lever spec,
//! `designs/endstate/design_information_levers.md` §6): a level name (`off` / `species` / `full`) or the CANONICAL
//! clause text of an `agents.info_levers.spec.InfoReveal` (`moves:seen@0.95`, …; `OracleSpec::parse`, strict). `off`
//! is the production row; any other spec tells a side's observation graded facts of the other side's true team. A
//! STRING is one spec for BOTH sides — the run's recorded mode, which the training pool and its eval core take. A
//! two-element ARRAY is a PER-SIDE spec (p1's chain, p2's chain: `[Reveal]`), which only the head-to-head engine
//! declares (`main.h2h`, X5 A/B §7.7(a): the one-sided clairvoyance cells). The canonical JSON writes each spec as its
//! LABEL (the level name when it is one, else the canonical text) and a string whenever the two sides agree, so a
//! symmetric `off` / `species` / `full` spec's text is exactly what it was before the spec existed.

use std::path::PathBuf;

use pokesim::encoder::oracle::OracleSpec;
use pokesim::json::Json;

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
    /// The ORACLE REVEAL spec PER SIDE: what of the other side's team each side's observation row is told
    /// (`pokesim::encoder::oracle`). `off` leaves that side's rows byte-identical to the build without it.
    pub oracle_reveal: Reveal,
}

/// The ORACLE REVEAL spec of each side's chain (index = side: p1, p2). `Reveal::both(s)` is the symmetric
/// mode a training run records; a split spec is the head-to-head engine's per-side reveal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reveal(pub [OracleSpec; 2]);

impl Reveal {
    /// No reveal on either side (the production observation).
    pub const OFF: Reveal = Reveal([OracleSpec::OFF; 2]);

    /// One spec for both sides (the run's recorded mode).
    pub const fn both(spec: OracleSpec) -> Reveal {
        Reveal([spec; 2])
    }

    /// Side `side`'s spec (0 = p1, 1 = p2).
    pub fn of(self, side: usize) -> OracleSpec {
        self.0[side]
    }

    /// The canonical JSON: each spec's LABEL, a string when both sides agree (the symmetric spec's text,
    /// unchanged), else `[p1, p2]`.
    pub fn to_json(self) -> String {
        let q = crate::core::refusal::json_str;
        if self.0[0] == self.0[1] {
            q(&self.0[0].label())
        } else {
            format!("[{},{}]", q(&self.0[0].label()), q(&self.0[1].label()))
        }
    }

    /// Parse a string (both sides) or a two-element array of strings (p1, p2); each a level name or a canonical
    /// spec text (`OracleSpec::parse`, which refuses everything else by name).
    pub fn from_json(v: Option<&Json>) -> Result<Reveal, String> {
        let bad = "spec: `oracle_reveal` must be a spec string or an array of two spec strings [p1, p2]";
        let v = v.ok_or(bad)?;
        if let Some(s) = v.as_str() {
            return Ok(Reveal::both(OracleSpec::parse(s).map_err(|e| format!("spec: {e}"))?));
        }
        let a = v.as_array().ok_or(bad)?;
        if a.len() != 2 {
            return Err(bad.into());
        }
        let sp = |j: &Json| -> Result<OracleSpec, String> {
            OracleSpec::parse(j.as_str().ok_or(bad)?).map_err(|e| format!("spec: {e}"))
        };
        Ok(Reveal([sp(&a[0])?, sp(&a[1])?]))
    }
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
            turn_limit,
            terminal: crate::episode::Terminal::from_json(v.get("terminal"))?,
            refusal_budget: uint(&v, "refusal_budget")? as usize,
            bank_dir,
            labels,
            opponents,
            oracle_reveal: Reveal::from_json(v.get("oracle_reveal"))?,
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
            "{{\"n\":{},\"threads\":{},\"format_id\":{},\"names\":[{},{}],\"teams\":[{}],\
             \"turn_limit\":{},\"terminal\":{},\"refusal_budget\":{},\"bank_dir\":{},\"labels\":[{}],\"opponents\":{},\"oracle_reveal\":{}}}",
            self.n,
            self.threads,
            q(&self.format_id),
            q(&self.names[0]),
            q(&self.names[1]),
            teams.join(","),
            self.turn_limit.map_or("null".to_string(), |t| t.to_string()),
            self.terminal.to_json(),
            self.refusal_budget,
            self.bank_dir.as_ref().map_or("null".to_string(), |d| q(&d.to_string_lossy())),
            self.labels.iter().map(|f| q(f)).collect::<Vec<_>>().join(","),
            self.opponents.to_json(),
            self.oracle_reveal.to_json(),
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
            turn_limit: Some(300),
            terminal: crate::episode::Terminal { victory_value: 30.0, indicator: false, draw_penalty: -35.0, timeout_turn_cap: 250 },
            refusal_budget: 4,
            bank_dir: Some(PathBuf::from("/tmp/x")),
            labels: Vec::new(),
            opponents: crate::opponents::Routes(vec![
                crate::opponents::Route::External,
                crate::opponents::Route::Policy { slot: 2 },
            ]),
            oracle_reveal: Reveal::both(OracleSpec::SPECIES),
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

    #[test]
    fn the_per_side_reveal_round_trips_and_a_symmetric_one_is_a_string() {
        let mut s = sample();
        // symmetric: the text is the string form, exactly as before the per-side form existed
        s.oracle_reveal = Reveal::OFF;
        assert!(s.to_json().ends_with(",\"oracle_reveal\":\"off\"}"), "{}", s.to_json());
        let ki = OracleSpec::parse("moves:seen@0.95").unwrap();
        for (a, b) in [(OracleSpec::SPECIES, OracleSpec::OFF), (OracleSpec::OFF, OracleSpec::FULL), (OracleSpec::FULL, OracleSpec::SPECIES), (ki, OracleSpec::OFF)] {
            s.oracle_reveal = Reveal([a, b]);
            let j = s.to_json();
            assert!(j.contains(&format!("\"oracle_reveal\":[\"{}\",\"{}\"]", a.label(), b.label())), "{j}");
            assert_eq!(Spec::from_json(&j).unwrap(), s);
        }
        // a symmetric spec is its LABEL: the level name for a level (whichever text it was given as), else the canonical text
        for (given, label) in [("\"full\"", "full"), ("\"species:unseen,moves:both,item:both,ability:both,spread:both\"", "full"),
                               ("\"species:unseen\"", "species"), ("\"moves:seen@0.95\"", "moves:seen@0.95")] {
            let mut p: Vec<_> = pairs().into_iter().filter(|(k, _)| *k != "oracle_reveal").collect();
            p.push(("oracle_reveal", given));
            let sp = Spec::from_json(&text(&p)).unwrap();
            assert!(sp.to_json().ends_with(&format!(",\"oracle_reveal\":\"{label}\"}}")), "{given}: {}", sp.to_json());
        }
        let base: Vec<_> = pairs().into_iter().filter(|(k, _)| *k != "oracle_reveal").collect();
        for (v, ok) in [("[\"species\",\"off\"]", true), ("[\"off\"]", false), ("[\"off\",\"off\",\"off\"]", false),
                        ("[\"off\",\"x\"]", false), ("[\"off\",1]", false), ("1", false), ("\"hp:seen\"", false),
                        ("\"moves:seen@0.95~placebo\"", false), ("\"moves:unseen\"", false), ("\"moves:seen@0.950\"", false)] {
            let mut p = base.clone();
            p.push(("oracle_reveal", v));
            assert_eq!(Spec::from_json(&text(&p)).is_ok(), ok, "{v}");
        }
    }

    fn pairs() -> Vec<(&'static str, &'static str)> {
        vec![
            ("n", "3"), ("threads", "2"), ("format_id", "\"gen3ou\""), ("names", "[\"a\",\"b\"]"),
            ("teams", "[\"X|||\"]"), ("turn_limit", "null"),
            ("terminal", "{\"victory_value\":1,\"terminal_indicator\":true,\"draw_penalty\":0,\"timeout_turn_cap\":250}"),
            ("refusal_budget", "0"), ("bank_dir", "null"), ("labels", "[]"),
            ("opponents", "[{\"kind\":\"external\"}]"), ("oracle_reveal", "\"off\""),
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
        // the deleted clock keys are refused by name, as every unknown key is
        for stale in ["decision_tense", "switch_freeze"] {
            let mut old = full.clone();
            old.push((stale, "false"));
            let e = Spec::from_json(&text(&old)).unwrap_err();
            assert!(e.contains("unknown key") && e.contains(stale), "{stale}: {e}");
        }
        let bad: Vec<_> = full.iter().map(|&(k, v)| if k == "turn_limit" { (k, "\"x\"") } else { (k, v) }).collect();
        assert!(Spec::from_json(&text(&bad)).unwrap_err().contains("turn_limit"));
    }
}
