//! One side's typed stream on the STEP path — the typed twin of the bridge's per-side split.
//!
//! The production bridge (`bridge.rs`) derives each side's TEXT from the omniscient text: an HP
//! field the side does not own becomes a percent, and two owner-only lines are dropped for the
//! other side (`derive_side`). With core recording on, the bridge also notes, for every per-side
//! line it ships, WHICH source record it came from (`bridge::CoreTrack`). This module rebuilds
//! that side's lines from the TYPED source records ([`side_view`]) and REFUSES unless each one
//! renders to exactly the bytes the bridge shipped — so the typed stream and the wire cannot
//! disagree — and unless the side saw every source record exactly once, in order, except the ones
//! the privacy split withholds from it (conservation per side).

use super::line::{Field, Hp, Line};
use super::reading::Reader;
use super::schema::Kw;
use super::{is_outcome, CoreEvent, SourceRec};

/// The percent a NON-owner sees (`bridge.rs::hp_percent`, `pokemon.js::getHealth`).
pub fn hp_percent(hp: u32, maxhp: u32) -> u32 {
    if maxhp == 0 {
        return 0;
    }
    let mut pct = (100 * hp).div_ceil(maxhp);
    if pct == 100 && hp < maxhp {
        pct = 99;
    }
    pct
}

/// `line` as `for_side` receives it: in a percent format, an HP field (`switch`/`drag`/
/// `-damage`/`-heal`/`-sethp`) the side does not own becomes `pct/100` (status kept, `0 fnt`
/// unchanged). Every other line is unchanged. (Owner-only DROPS are a presence decision, made by
/// the bridge and checked by [`step_events`]'s conservation.)
pub fn side_view(line: &Line, for_side: u8, report_percent: bool) -> Line {
    if !report_percent {
        return line.clone();
    }
    let hp_idx = match line.kw {
        Kw::Switch | Kw::Drag => 2,
        Kw::Damage | Kw::Heal | Kw::Sethp => 1,
        _ => return line.clone(),
    };
    match line.owner_side() {
        Some(o) if o != for_side => {}
        _ => return line.clone(),
    }
    let mut out = line.clone();
    if let Some(Field::Hp(Hp::Alive { hp, max, status })) = out.fields.get(hp_idx).cloned() {
        out.fields[hp_idx] = Field::Hp(Hp::Alive { hp: hp_percent(hp, max), max: 100, status });
    }
    out
}

/// Which side (if any) is the ONLY one to receive source record `idx`: gen-3 Pressure's
/// `|-ability|…|[silent]` (`addSplit`) and the Intimidate-vs-Substitute `|-hint|` (owned by the
/// preceding switch-in's side), exactly the bridge's two owner-only rules.
pub fn owner_only(recs: &[SourceRec], idx: usize) -> Option<u8> {
    let l = &recs[idx].line;
    if l.kw == Kw::Ability && matches!(l.fields.last(), Some(Field::Tag(t)) if t == "[silent]") {
        return l.owner_side();
    }
    if l.kw == Kw::Hint && l.render().contains("Intimidate does not activate") {
        return recs[..idx]
            .iter()
            .rev()
            .find(|r| matches!(r.line.kw, Kw::Switch | Kw::Drag))
            .and_then(|r| r.line.owner_side());
    }
    None
}

/// The format-framing lines the bridge REWRITES per format (`bridge::reframe`: the tier and the
/// rule list), which a side therefore receives as side-only frames rather than as their source.
fn reframed(kw: Kw) -> bool {
    matches!(kw, Kw::Tier | Kw::Rule)
}

/// The per-side lines the BRIDGE builds itself (no omniscient source record): a `|request|`, an
/// `|error|` (a rejected choice), a reframed tier/rule, and the forced-Struggle
/// `|-activate|<mon>|move: Struggle` the bridge announces to the struggling side at the commit
/// (`bridge.rs`, "struggle announce"). Anything else without a source is REFUSED.
fn bridge_frame(line: &Line) -> bool {
    match line.kw {
        Kw::Request | Kw::Error | Kw::Tier | Kw::Rule => true,
        Kw::Activate => line.fields.len() == 2 && matches!(&line.fields[1], Field::Text(t) if t == "move: Struggle"),
        _ => false,
    }
}

/// ONE shipped per-side line, typed on the STEP path: rebuilt from its source record through the
/// privacy fold ([`side_view`]) and REFUSED unless it renders the exact bytes the bridge shipped;
/// a side-only frame (no source) is parsed and must be one of the bridge's own frames.
pub fn step_line(recs: &[SourceRec], text: &str, src: Option<u32>, viewer: u8, report_percent: bool) -> Result<Line, String> {
    match src {
        Some(s) => {
            let rec = recs.get(s as usize).ok_or_else(|| format!("source record {s} out of range"))?;
            let line = side_view(&rec.line, viewer, report_percent);
            let rendered = line.render();
            if rendered != text {
                return Err(format!("p{}: the typed source renders {rendered:?} but the bridge shipped {text:?}", viewer + 1));
            }
            Ok(line)
        }
        None => {
            let line = Line::parse(text).map_err(|e| format!("p{}: {e}", viewer + 1))?;
            if !bridge_frame(&line) {
                return Err(format!("p{}: {text:?} has no source record and is not a side-only frame", viewer + 1));
            }
            Ok(line)
        }
    }
}

/// The STEP path's [`CoreEvent`]s for one side: `shipped` is that side's per-side lines in order
/// with the source-record index each was derived from (`None` = a side-only frame).
pub fn step_events(
    recs: &[SourceRec],
    shipped: &[(String, Option<u32>)],
    viewer: u8,
    report_percent: bool,
) -> Result<Vec<CoreEvent>, String> {
    // Conservation per side: the sourced lines are exactly the records this side receives, once
    // each, in order. (The tier/rule framing records are exempt: a per-format reframe REPLACES
    // them with side-only frames in one format and ships them verbatim in another.)
    let sourced: Vec<u32> =
        shipped.iter().filter_map(|(_, s)| *s).filter(|&s| !reframed(recs[s as usize].line.kw)).collect();
    let expected: Vec<u32> = (0..recs.len())
        .filter(|&i| !reframed(recs[i].line.kw))
        .filter(|&i| owner_only(recs, i).map_or(true, |o| o == viewer))
        .map(|i| i as u32)
        .collect();
    if sourced != expected {
        let first = sourced.iter().zip(expected.iter()).position(|(a, b)| a != b).unwrap_or(sourced.len().min(expected.len()));
        return Err(format!(
            "CONSERVATION (p{}): the side received {} source records, the privacy split says {} — first difference at #{first}",
            viewer + 1,
            sourced.len(),
            expected.len()
        ));
    }
    let mut reader = Reader::new(viewer as usize);
    let mut out = Vec::with_capacity(shipped.len());
    for (i, (text, src)) in shipped.iter().enumerate() {
        let (line, owner) = match src {
            Some(s) => {
                let rec = &recs[*s as usize];
                let line = side_view(&rec.line, viewer, report_percent);
                let rendered = line.render();
                if &rendered != text {
                    return Err(format!(
                        "p{} line {i}: the typed source renders {rendered:?} but the bridge shipped {text:?}",
                        viewer + 1
                    ));
                }
                let owner = if is_outcome(line.kw) { rec.scope.move_side() } else { None };
                (line, owner)
            }
            None => {
                let line = Line::parse(text).map_err(|e| format!("p{} line {i}: {e}", viewer + 1))?;
                if !bridge_frame(&line) {
                    return Err(format!("p{} line {i}: {text:?} has no source record and is not a side-only frame", viewer + 1));
                }
                (line, None)
            }
        };
        let readings = reader.feed(&line).map_err(|e| format!("p{} line {i} {text:?}: {e}", viewer + 1))?;
        out.push(CoreEvent { idx: i as u32, line, src: *src, owner, readings });
    }
    Ok(out)
}
