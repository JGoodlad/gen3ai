//! `parse(lines)` — ONE side's protocol TEXT into [`CoreEvent`]s (the design's entry point 2).
//!
//! What a real Showdown server sends a player is one side's stream, so that is what this reads:
//! every line typed ([`Line::parse`], which REFUSES an unknown keyword), folded through the same
//! [`Reader`] the step path uses, plus the one truth attribution the protocol does not print —
//! the owner of an outcome line — recovered from line ORDER ([`OwnerScan`]). The M1 gate is
//! [`parse_matches_step`]: on the simulator's own emission, `parse(text) == step`, per viewer, per
//! line, on the typed line, the owner and every reading.

use super::line::Line;
use super::reading::Reader;
use super::schema::Kw;
use super::{is_outcome, CoreEvent};

/// The text rule for "whose MOVE owns this line" (Phase-0 parse-back, 12,850 / 12,850 on
/// omniscient lines): `|move|X` opens X's move; any `|switch|` (a Baton Pass entry included),
/// `|turn|`, `|upkeep` and the bare `|` separator close it; `|drag|` does NOT (a Roar's drag is
/// inside the Roar). Finer than poke-env's R1, which closes only at `|turn|`.
#[derive(Debug, Clone, Default)]
pub struct OwnerScan {
    open: Option<u8>,
}

impl OwnerScan {
    /// Advance over `line`; return the owner that applies TO it.
    pub fn step(&mut self, line: &Line) -> Option<u8> {
        match line.kw {
            Kw::Move => self.open = line.ident(0).map(|i| i.side),
            Kw::Switch | Kw::Turn | Kw::Upkeep | Kw::Separator => self.open = None,
            _ => {}
        }
        self.open
    }
}

/// Parse one side's stream. `viewer` is the side (0 = p1). `Err` names the first line that cannot
/// be read — an unknown keyword, a non-gen-3 line, or a line poke-env itself would raise on.
pub fn parse<S: AsRef<str>>(lines: &[S], viewer: usize) -> Result<Vec<CoreEvent>, String> {
    let mut reader = Reader::new(viewer);
    let mut owners = OwnerScan::default();
    let mut out = Vec::with_capacity(lines.len());
    for (i, text) in lines.iter().enumerate() {
        let text = text.as_ref();
        let line = Line::parse(text).map_err(|e| format!("line {i} {text:?}: {e}"))?;
        let owner = owners.step(&line);
        let readings = reader.feed(&line).map_err(|e| format!("line {i} {text:?}: {e}"))?;
        let owner = if is_outcome(line.kw) { owner } else { None };
        out.push(CoreEvent { idx: i as u32, line, src: None, owner, readings });
    }
    Ok(out)
}

/// The M1 gate: `parse(text)` reproduces the STEP path's events on every field but `src` (which
/// only the step path knows). `Err` describes the first difference.
pub fn parse_matches_step(step: &[CoreEvent], parsed: &[CoreEvent]) -> Result<(), String> {
    if step.len() != parsed.len() {
        return Err(format!("line count: step {} vs parse {}", step.len(), parsed.len()));
    }
    for (a, b) in step.iter().zip(parsed) {
        if a.line != b.line {
            return Err(format!("line {}: typed line differs: step {:?} parse {:?}", a.idx, a.line, b.line));
        }
        if a.owner != b.owner {
            return Err(format!(
                "line {} {:?}: outcome OWNER differs: engine scope says {:?}, line order says {:?}",
                a.idx,
                a.line.render(),
                a.owner,
                b.owner
            ));
        }
        if a.readings != b.readings {
            return Err(format!("line {}: readings differ:\n  step  {:?}\n  parse {:?}", a.idx, a.readings, b.readings));
        }
    }
    Ok(())
}
