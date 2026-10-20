//! DAMAGE-ROLL CONTROL (`gen3_roll_control_v1`): an OFFLINE replay mode that lets a world's prefix
//! replay take, at each damage roll, a roll VALUE other than the battle seed's, while the PRNG stream
//! stays aligned with the truth's (F1 tier A of `designs/endstate/design_world_sampler.md` §3.3.1;
//! contract: `designs/rust_sim/search_and_replay_drivers.md` "Damage-roll control").
//!
//! WHY. The honest world builder replays each candidate world on the TRUE battle seed, so a seen mon's
//! hit reproduces the printed HP only with the truth's own damage roll. An honest player knows only
//! that the roll was one of 16. Here, at every damage-roll SITE (each `random(16)` the gen-3
//! randomizer draws), the draw is CONSUMED exactly as today (so every later die is still the truth's),
//! and only the VALUE the site uses may be replaced.
//!
//! THE THREE MODES (all opt-in; a battle without a [`RollCtl`] runs the code it always ran):
//! - [`Mode::Record`] (the TRUTH's pass): each site records, for the `viewer`, what that side SAW of
//!   the hit (the [`Target`]: our mon's exact HP, the opponent's percent, a Substitute held / broke).
//!   The values used are the stream's own.
//! - [`Mode::Match`] (a WORLD's pass): at the n-th site the consistent set C is the rolls whose
//!   predicted outcome equals target n's. The site uses the stream's roll when it is in C (today's
//!   trajectory), else the member of C nearest it (ties: the lower index), and multiplies the world's
//!   likelihood by |C| / 16; an empty C makes the world infeasible (the stream's roll is used, and the
//!   byte gate rejects it). A site whose outcome a post-roll modifier can change (a Focus Band holder,
//!   an Endure volatile or False Swipe when some roll is lethal; Beat Up, a Future Sight cast, a Jump
//!   Kick crash) is PINNED: the stream's roll, factor 1/16 (the known lower bound), never widened.
//! - [`Mode::Force`] (the REPLAY of a matched world, and the identity test): the n-th site uses the
//!   table's roll when it names n, else the stream's.
//!
//! Draw-free and emission-free by construction: the site draws exactly the one `random(16)` it always
//! drew, and the prediction reads state only.

use std::collections::BTreeMap;

use crate::core_events::side::hp_percent;
use crate::dex::Dex;
use crate::json::Json;
use crate::state::BattleState;

/// The kind of a damage-roll site (one per `random(16)` call in `turn/moves.rs`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SiteKind {
    /// A single-hit damaging move (`run_move`'s main path).
    Hit,
    /// One strike of a multi-hit move.
    Multi,
    /// A confusion self-hit (the target is the user).
    Confusion,
    /// One Beat Up strike (pinned).
    BeatUp,
    /// A Future Sight / Doom Desire cast: the damage lands two turns later (pinned).
    FutureCast,
    /// A Jump Kick / Hi Jump Kick crash (pinned).
    Crash,
}

impl SiteKind {
    pub fn tag(self) -> &'static str {
        match self {
            SiteKind::Hit => "hit",
            SiteKind::Multi => "multi",
            SiteKind::Confusion => "confusion",
            SiteKind::BeatUp => "beatup",
            SiteKind::FutureCast => "future",
            SiteKind::Crash => "crash",
        }
    }

    fn of_tag(t: &str) -> Option<SiteKind> {
        Some(match t {
            "hit" => SiteKind::Hit,
            "multi" => SiteKind::Multi,
            "confusion" => SiteKind::Confusion,
            "beatup" => SiteKind::BeatUp,
            "future" => SiteKind::FutureCast,
            "crash" => SiteKind::Crash,
            _ => return None,
        })
    }

    /// The kinds whose printed outcome is not the roll's direct HP result at the site.
    fn always_pinned(self) -> bool {
        matches!(self, SiteKind::BeatUp | SiteKind::FutureCast | SiteKind::Crash)
    }
}

/// What the viewer saw of one site (recorded on the truth's pass) and the key a world's site must
/// carry to be aligned with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Target {
    pub kind: SiteKind,
    /// The attacking side (0 / 1) and the hit side.
    pub att: u8,
    pub tgt: u8,
    /// The move's id, Hidden Power's type folded (`hiddenpower`): a world draws that type.
    pub mv: String,
    /// The viewer-visible outcome at the stream's roll (`""` at a pinned site).
    pub out: String,
    /// Whether the TRUTH's site was pinned (then every world pins it too).
    pub pinned: bool,
}

#[derive(Clone, Debug)]
pub enum Mode {
    Record,
    Match(Vec<Target>),
    Force(BTreeMap<u32, u8>),
}

/// One site as the report states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SiteRec {
    pub n: u32,
    pub kind: SiteKind,
    /// The consistent-roll count (Match), 1 at a pinned site, 0 when infeasible; 16 elsewhere.
    pub k: u8,
    /// The roll the site used and the stream's own.
    pub r: u8,
    pub r0: u8,
    /// `r` record, `m` matched, `p` pinned, `x` misaligned (the key differs from the target's), `u`
    /// untargeted (past the targets), `0` infeasible (C empty), `f` forced, `s` the stream's (Force,
    /// not named).
    pub flag: char,
}

/// The armed control: its mode, the viewer whose sight defines a site's outcome, and the report.
#[derive(Clone, Debug)]
pub struct RollCtl {
    pub viewer: u8,
    pub mode: Mode,
    /// Sites seen so far (the ordinal of the next).
    pub n: u32,
    pub sites: Vec<SiteRec>,
    /// Record: the targets; Match: the per-site key / outcome this world produced (diagnostic).
    pub targets: Vec<Target>,
    /// Match: Σ log(k / 16) over the matched and pinned sites; `None` once a site was infeasible.
    pub log_w: Option<f64>,
}

impl RollCtl {
    pub fn new(viewer: u8, mode: Mode) -> RollCtl {
        RollCtl { viewer, mode, n: 0, sites: Vec::new(), targets: Vec::new(), log_w: Some(0.0) }
    }

    /// The control a START `"rolls"` object asks for: `{"viewer":0|1,"mode":"record"}`,
    /// `{"viewer":…,"mode":"match","targets":[[kind,att,tgt,mv,out,pinned],…]}`, or
    /// `{"viewer":…,"mode":"force","force":[[n,r],…]}`. Unknown keys are refused.
    pub fn from_json(v: &Json) -> Result<RollCtl, String> {
        let obj = v.as_object().ok_or("rolls: must be an object")?;
        for k in obj.keys() {
            if !["viewer", "mode", "targets", "force"].contains(&k.as_str()) {
                return Err(format!("rolls: unknown key {k:?}"));
            }
        }
        let viewer = match v.get("viewer").and_then(Json::as_f64) {
            Some(x) if x == 0.0 || x == 1.0 => x as u8,
            _ => return Err("rolls: `viewer` must be 0 or 1".into()),
        };
        let mode = match v.str_at("mode") {
            Some("record") => Mode::Record,
            Some("match") => {
                let a = v.get("targets").and_then(Json::as_array).ok_or("rolls: match needs `targets` (an array)")?;
                Mode::Match(a.iter().map(target_of).collect::<Result<_, _>>()?)
            }
            Some("force") => {
                let a = v.get("force").and_then(Json::as_array).ok_or("rolls: force needs `force` (an array of [n, r])")?;
                let mut t = BTreeMap::new();
                for e in a {
                    let p = e.as_array().filter(|p| p.len() == 2).ok_or("rolls: a force entry is [n, r]")?;
                    let n = uint(&p[0], "force n")? as u32;
                    let r = uint(&p[1], "force r")?;
                    if r >= 16 {
                        return Err(format!("rolls: force roll {r} is not in 0..16"));
                    }
                    if t.insert(n, r as u8).is_some() {
                        return Err(format!("rolls: force names site {n} twice"));
                    }
                }
                Mode::Force(t)
            }
            other => return Err(format!("rolls: unknown mode {other:?} (record / match / force)")),
        };
        Ok(RollCtl::new(viewer, mode))
    }

    /// The report: `{"mode","sites":n,"log_w":x|null,"feasible":b,"force":[[n,r],…],
    /// "counts":{flag:count},"site_log":[[n,kind,k,r,r0,flag],…]}` (+ `"targets"` on a Record pass).
    /// `force` lists the sites whose roll differs from the stream's: Force with it replays this
    /// trajectory exactly.
    pub fn json(&self) -> String {
        let mode = match self.mode {
            Mode::Record => "record",
            Mode::Match(_) => "match",
            Mode::Force(_) => "force",
        };
        let mut o = format!("{{\"mode\":\"{mode}\",\"sites\":{}", self.n);
        match self.log_w {
            Some(w) => o.push_str(&format!(",\"log_w\":{w:?},\"feasible\":true")),
            None => o.push_str(",\"log_w\":null,\"feasible\":false"),
        }
        o.push_str(",\"force\":[");
        let mut first = true;
        for s in self.sites.iter().filter(|s| s.r != s.r0) {
            if !first {
                o.push(',');
            }
            first = false;
            o.push_str(&format!("[{},{}]", s.n, s.r));
        }
        o.push_str("],\"counts\":{");
        let mut counts: BTreeMap<char, u32> = BTreeMap::new();
        for s in &self.sites {
            *counts.entry(s.flag).or_default() += 1;
        }
        for (i, (f, c)) in counts.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!("\"{f}\":{c}"));
        }
        o.push_str("},\"site_log\":[");
        for (i, s) in self.sites.iter().enumerate() {
            if i > 0 {
                o.push(',');
            }
            o.push_str(&format!("[{},\"{}\",{},{},{},\"{}\"]", s.n, s.kind.tag(), s.k, s.r, s.r0, s.flag));
        }
        o.push(']');
        if matches!(self.mode, Mode::Record) {
            o.push_str(",\"targets\":[");
            for (i, t) in self.targets.iter().enumerate() {
                if i > 0 {
                    o.push(',');
                }
                o.push_str(&format!("[\"{}\",{},{},", t.kind.tag(), t.att, t.tgt));
                crate::core_events::json_out::str_into(&mut o, &t.mv);
                o.push(',');
                crate::core_events::json_out::str_into(&mut o, &t.out);
                o.push_str(&format!(",{}]", t.pinned));
            }
            o.push(']');
        }
        o.push('}');
        o
    }
}

fn uint(v: &Json, what: &str) -> Result<u64, String> {
    v.as_f64()
        .filter(|x| *x >= 0.0 && x.fract() == 0.0)
        .map(|x| x as u64)
        .ok_or_else(|| format!("rolls: {what} must be a non-negative integer"))
}

fn target_of(v: &Json) -> Result<Target, String> {
    let a = v.as_array().filter(|a| a.len() == 6).ok_or("rolls: a target is [kind, att, tgt, mv, out, pinned]")?;
    let kind = a[0].as_str().and_then(SiteKind::of_tag).ok_or("rolls: a target's kind")?;
    let side = |j: &Json| -> Result<u8, String> {
        match j.as_f64() {
            Some(x) if x == 0.0 || x == 1.0 => Ok(x as u8),
            _ => Err("rolls: a target's side must be 0 or 1".into()),
        }
    };
    Ok(Target {
        kind,
        att: side(&a[1])?,
        tgt: side(&a[2])?,
        mv: a[3].as_str().ok_or("rolls: a target's move")?.to_string(),
        out: a[4].as_str().ok_or("rolls: a target's outcome")?.to_string(),
        pinned: match a[5] {
            Json::Bool(b) => b,
            _ => return Err("rolls: a target's pinned flag must be a bool".into()),
        },
    })
}

/// The move key a site is aligned on: the id, every typed Hidden Power folded to `hiddenpower`.
pub fn move_key(move_id: &str) -> String {
    if move_id.starts_with("hiddenpower") {
        "hiddenpower".to_string()
    } else {
        move_id.to_string()
    }
}

/// The site's place in the battle, as the hook passes it.
pub struct Site<'a> {
    pub kind: SiteKind,
    pub att_side: usize,
    pub tgt_side: usize,
    pub tgt_slot: usize,
    pub move_id: &'a str,
    pub rolls: &'a [u16; 16],
    /// Whether the damage meets the target's Substitute first (the move paths: the sub-intercept).
    pub via_sub: bool,
    /// The move-specific survive-at-1 modifiers this path applies (False Swipe on the single hit,
    /// Endure on the move paths).
    pub false_swipe: bool,
    pub endure_applies: bool,
}

impl BattleState {
    /// The damage roll at a site: ONE `random(16)` draw (always, exactly as before), and the roll the
    /// site uses — the drawn one unless a [`RollCtl`] is armed (module docs).
    pub(crate) fn damage_roll(&mut self, site: Site<'_>, dex: &Dex) -> usize {
        let r0 = self.prng.random_below(16) as usize;
        if self.rolls.is_none() {
            return r0;
        }
        let pred = self.roll_outcomes(&site, dex);
        let kind = site.kind;
        let key = Target {
            kind,
            att: site.att_side as u8,
            tgt: site.tgt_side as u8,
            mv: move_key(site.move_id),
            out: String::new(),
            pinned: pred.is_none(),
        };
        let ctl = self.rolls.as_mut().expect("checked");
        let n = ctl.n;
        ctl.n += 1;
        let (r, k, flag) = match &ctl.mode {
            Mode::Record => {
                let out = pred.as_ref().map_or(String::new(), |p| p[r0].clone());
                ctl.targets.push(Target { out, ..key });
                (r0, 16, 'r')
            }
            Mode::Force(t) => match t.get(&n) {
                Some(&r) => (r as usize, 16, 'f'),
                None => (r0, 16, 's'),
            },
            Mode::Match(targets) => match targets.get(n as usize) {
                None => (r0, 16, 'u'),
                Some(t) if t.kind != key.kind || t.att != key.att || t.tgt != key.tgt || t.mv != key.mv => (r0, 16, 'x'),
                Some(t) => match (&pred, t.pinned) {
                    (None, _) | (_, true) => (r0, 1, 'p'),
                    (Some(p), false) => {
                        let c: Vec<usize> = (0..16).filter(|&r| p[r] == t.out).collect();
                        if c.is_empty() {
                            (r0, 0, '0')
                        } else if c.contains(&r0) {
                            (r0, c.len() as u8, 'm')
                        } else {
                            // the nearest consistent roll (ties: the lower index)
                            let r = *c.iter().min_by_key(|&&r| (r.abs_diff(r0), r)).expect("non-empty");
                            (r, c.len() as u8, 'm')
                        }
                    }
                },
            },
        };
        if matches!(ctl.mode, Mode::Match(_)) {
            match flag {
                'm' | 'p' => {
                    if let Some(w) = ctl.log_w.as_mut() {
                        *w += (k as f64 / 16.0).ln();
                    }
                }
                '0' => ctl.log_w = None,
                _ => {}
            }
        }
        ctl.sites.push(SiteRec { n, kind, k, r: r as u8, r0: r0 as u8, flag });
        r
    }

    /// Per roll, what the controlled viewer would SEE of this site's hit, or `None` when the site is
    /// PINNED (module docs). Reads state only. Mirrors the apply path after the roll: the
    /// sub-intercept (`absorb_into_sub`: the sub breaks iff the roll reaches its HP), then the
    /// survive-at-1 modifiers (pinned whenever one could fire), then `apply_damage` (HP saturates at 0).
    pub(crate) fn roll_outcomes(&self, site: &Site<'_>, dex: &Dex) -> Option<Vec<String>> {
        if site.kind.always_pinned() {
            return None;
        }
        let viewer = self.rolls.as_ref().map_or(0, |c| c.viewer) as usize;
        let mon = &self.sides[site.tgt_side].pokemon[site.tgt_slot];
        if site.via_sub {
            if let Some(sub) = mon.substitute {
                return Some(site.rolls.iter().map(|&d| if d >= sub { "sub:broke" } else { "sub:held" }.to_string()).collect());
            }
        }
        // the non-owner sees a percent except in a debug (`customgame`) format (`bridge::format_is_debug`)
        let percent = !crate::bridge::format_is_debug(&self.format_id);
        let hp = mon.hp;
        let lethal = site.rolls.iter().any(|&d| d >= hp);
        let band = dex.item(&crate::dex::to_id(&mon.item)).and_then(|i| i.survive_lethal).is_some();
        if lethal && (site.false_swipe || (site.endure_applies && mon.endure) || band) {
            return None;
        }
        Some(
            site.rolls
                .iter()
                .map(|&d| {
                    let after = hp - d.min(hp);
                    if after == 0 {
                        "0".to_string()
                    } else if site.tgt_side == viewer || !percent {
                        format!("{after}/{}", mon.maxhp)
                    } else {
                        format!("{}", hp_percent(after as u32, mon.maxhp as u32))
                    }
                })
                .collect(),
        )
    }

    /// Arm (or, with `None`, disarm) the damage-roll control.
    pub fn set_roll_ctl(&mut self, ctl: Option<RollCtl>) {
        self.rolls = ctl.map(Box::new);
    }
}
