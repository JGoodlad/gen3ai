//! The WHOLE-BATTLE guard for the two per-side construction-WINDOW keys: B1
//! (`turn0-construction-speed-tie-order-flip`) and the mirror key
//! (`perside-construction-speed-tie-mirror-of-flip`).
//!
//! **Why.** The `--ab` verdict is FIRST-divergence: once a battle's first per-side divergence is a
//! construction-window reorder and a key admits it, nothing later in that battle is ever read. B1
//! and the mirror key judged only the framing window of the first-diverging side, so a REAL
//! divergence later in an allowlisted battle (or in the other side's window) would be hidden behind
//! the harmless reorder. The block-swap key (`switchin_block_swap.rs`) closed that for itself with
//! its clause (3); this module gives B1 and the mirror key the same clause
//! (`gen3_perside_window_keys_whole_battle_v1`, the allowlist-gate half of the Perish Song row:
//! the Rollout perish faint `bab_0_8` was the kind of real bug such a key could have masked).
//!
//! **The predicate** — ALL must hold, else `None` and the gate FAILS:
//!  (W1) on EACH side, both the golden and the engine stream contain a `|turn|` line and the FIRST
//!       one is exactly `|turn|1`, at the SAME index (equal-length construction windows);
//!  (W2) on EACH side, everything from that `|turn|1` to the end of the stream is BYTE-IDENTICAL
//!       (the rest of the battle; a truncated stream fails here too);
//!  (W3) on EACH side, the two windows are identical OR admitted by B1 / the mirror key (tried in
//!       that order, exactly as before) — so a content change in the OTHER side's window fails;
//!  (W4) every side whose windows differ is admitted under the SAME reason (one root, one key);
//!  (W5) at least one side's windows differ.
//!
//! The window classifiers themselves are unchanged; this only narrows WHEN their verdict counts.

use super::{classify_perside_construction_mirror_flip, classify_perside_construction_order_flip};

/// Index of the FIRST `|turn|` line, only when it is exactly `|turn|1`; `None` otherwise.
fn turn_one_index(stream: &[String]) -> Option<usize> {
    let i = stream.iter().position(|l| l.starts_with("|turn|"))?;
    (stream[i] == "|turn|1").then_some(i)
}

/// Classify a `kind=perside` first divergence against B1 then the mirror key, over BOTH full
/// per-side streams (`[p1, p2]`). See the module doc for clauses (W1)-(W5).
pub(crate) fn classify_window_keys_whole_battle(
    golden: [&[String]; 2],
    engine: [&[String]; 2],
    leads_speed_tie: bool,
) -> Option<&'static str> {
    let mut reason: Option<&'static str> = None;
    for side in 0..2 {
        let (g, e) = (golden[side], engine[side]);
        // (W1) a real turn-0 window ending at `|turn|1`, the same length on both.
        let gt = turn_one_index(g)?;
        let et = turn_one_index(e)?;
        if gt != et {
            return None;
        }
        // (W2) the rest of the battle, byte-identical on this side.
        if g[gt..] != e[et..] {
            return None;
        }
        let (gw, ew) = (&g[..gt], &e[..et]);
        if gw == ew {
            continue;
        }
        // (W3) this side's window is admitted by a window key.
        let r = classify_perside_construction_order_flip(gw, ew, leads_speed_tie)
            .or_else(|| classify_perside_construction_mirror_flip(gw, ew, leads_speed_tie))?;
        // (W4) one reason across both sides.
        match reason {
            None => reason = Some(r),
            Some(prev) if prev == r => {}
            Some(_) => return None,
        }
    }
    // (W5) at least one side differed (`reason` is set only by a differing side).
    reason
}

// ── WHOLE-BATTLE guard integrity tests ──
//
// Each negative is a way a genuine per-side bug could sit behind a B1 / mirror construction
// reorder. Reverting any one clause makes at least one of them wrongly return `Some(..)`.
#[cfg(test)]
mod tests {
    use super::*;

    const B1: &str = "turn0-construction-speed-tie-order-flip";
    const MIRROR: &str = "perside-construction-speed-tie-mirror-of-flip";

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    // The bab_3_15 shape (B1): Tyranitar vs Suicune at a 213 tie; the Pressure / Sandstorm framing
    // lines flip.
    const HEAD: &[&str] = &[
        "|start",
        "|switch|p1a: Tyranitar|Tyranitar, M|100/100",
        "|switch|p2a: Suicune|Suicune|100/100",
    ];
    const PRESSURE: &str = "|-ability|p2a: Suicune|Pressure|[silent]";
    const SAND: &str = "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar";
    const TAIL: &[&str] = &[
        "|turn|1",
        "|move|p1a: Tyranitar|Rock Slide|p2a: Suicune",
        "|-damage|p2a: Suicune|80/100",
        "|turn|2",
    ];

    fn stream(framing: &[&str], tail: &[&str]) -> Vec<String> {
        let mut s = v(HEAD);
        s.extend(v(framing));
        s.extend(v(tail));
        s
    }

    /// Both sides see the same public flip.
    fn b1_repro() -> ([Vec<String>; 2], [Vec<String>; 2]) {
        let g = stream(&[PRESSURE, SAND], TAIL);
        let e = stream(&[SAND, PRESSURE], TAIL);
        ([g.clone(), g], [e.clone(), e])
    }

    fn classify(g: &[Vec<String>; 2], e: &[Vec<String>; 2]) -> Option<&'static str> {
        classify_window_keys_whole_battle([&g[0], &g[1]], [&e[0], &e[1]], true)
    }

    #[test]
    fn the_b1_flip_on_both_sides_is_allowlisted() {
        let (g, e) = b1_repro();
        assert_eq!(classify(&g, &e), Some(B1));
    }

    #[test]
    fn a_flip_on_one_side_only_is_allowlisted() {
        let (g, mut e) = b1_repro();
        e[1] = g[1].clone();
        assert_eq!(classify(&g, &e), Some(B1));
    }

    #[test]
    fn the_mirror_of_flip_is_allowlisted() {
        let head = [
            "|start",
            "|switch|p1a: Tyranitar|Tyranitar, M|100/100",
            "|switch|p2a: Tyranitar|Tyranitar, M|100/100",
        ];
        let mk = |of: &str| {
            let mut s = v(&head);
            s.push(format!("|-weather|Sandstorm|[from] ability: Sand Stream|[of] {of}: Tyranitar"));
            s.extend(v(TAIL));
            s
        };
        let (g, e) = (mk("p2a"), mk("p1a"));
        assert_eq!(classify(&[g.clone(), g], &[e.clone(), e]), Some(MIRROR));
    }

    #[test]
    fn a_non_tie_fails() {
        let (g, e) = b1_repro();
        assert_eq!(classify_window_keys_whole_battle([&g[0], &g[1]], [&e[0], &e[1]], false), None);
    }

    #[test]
    fn identical_streams_are_not_a_flip() {
        let (g, _) = b1_repro();
        assert_eq!(classify(&g, &g), None);
    }

    // (W2) — the masking guard.
    #[test]
    fn a_later_divergence_on_the_same_side_fails() {
        let (g, mut e) = b1_repro();
        e[0] = stream(&[SAND, PRESSURE], &["|turn|1", "|move|p1a: Tyranitar|Rock Slide|p2a: Suicune", "|-damage|p2a: Suicune|79/100", "|turn|2"]);
        assert_eq!(classify(&g, &e), None);
    }

    #[test]
    fn a_later_divergence_on_the_other_side_fails() {
        let (g, mut e) = b1_repro();
        e[1] = stream(&[SAND, PRESSURE], &["|turn|1", "|move|p1a: Tyranitar|Rock Slide|p2a: Suicune", "|-damage|p2a: Suicune|79/100", "|turn|2"]);
        assert_eq!(classify(&g, &e), None);
    }

    #[test]
    fn a_later_divergence_behind_an_identical_other_window_fails() {
        // p2's window is identical and only its rest differs: the per-side `continue` must not
        // skip the rest-of-battle check.
        let (g, mut e) = b1_repro();
        e[1] = g[1].clone();
        e[1].insert(e[1].len() - 1, "|-damage|p2a: Suicune|70/100".to_string());
        assert_eq!(classify(&g, &e), None);
    }

    #[test]
    fn a_truncated_engine_stream_fails() {
        let (g, mut e) = b1_repro();
        e[0].pop();
        assert_eq!(classify(&g, &e), None);
    }

    // (W3) — the other side's window.
    #[test]
    fn a_content_change_in_the_other_sides_window_fails() {
        let (g, mut e) = b1_repro();
        e[1] = stream(&[SAND, "|-ability|p2a: Suicune|Insomnia|[silent]"], TAIL);
        assert_eq!(classify(&g, &e), None);
    }

    // (W1)
    #[test]
    fn a_stream_with_no_turn_marker_fails() {
        let g = v(&[HEAD[0], HEAD[1], HEAD[2], PRESSURE, SAND]);
        let e = v(&[HEAD[0], HEAD[1], HEAD[2], SAND, PRESSURE]);
        assert_eq!(classify(&[g.clone(), g], &[e.clone(), e]), None);
    }

    #[test]
    fn a_first_turn_marker_that_is_not_turn_one_fails() {
        let tail = ["|turn|2", "|-damage|p2a: Suicune|80/100"];
        let g = stream(&[PRESSURE, SAND], &tail);
        let e = stream(&[SAND, PRESSURE], &tail);
        assert_eq!(classify(&[g.clone(), g], &[e.clone(), e]), None);
    }

    // (W4) — one root, one key: a B1 flip on one side and a different key's verdict on the
    // other is not this deferral.
    #[test]
    fn two_sides_under_different_keys_fail() {
        let head_m = [
            "|start",
            "|switch|p1a: Tyranitar|Tyranitar, M|100/100",
            "|switch|p2a: Tyranitar|Tyranitar, M|100/100",
        ];
        let mk = |framing: &[String]| {
            let mut s = v(&head_m);
            s.extend(framing.iter().cloned());
            s.extend(v(TAIL));
            s
        };
        let w = |of: &str| format!("|-weather|Sandstorm|[from] ability: Sand Stream|[of] {of}: Tyranitar");
        let a = "|-ability|p2a: Tyranitar|Pressure|[silent]".to_string();
        // p1: a pure permutation of an -ability and a -weather line (B1).
        let g1 = mk(&[a.clone(), w("p2a")]);
        let e1 = mk(&[w("p2a"), a.clone()]);
        // p2: a single-line mirror `[of]` flip (the mirror key).
        let g2 = mk(&[w("p2a")]);
        let e2 = mk(&[w("p1a")]);
        assert_eq!(classify_window_keys_whole_battle([&g1, &g1], [&e1, &e1], true), Some(B1));
        assert_eq!(classify_window_keys_whole_battle([&g2, &g2], [&e2, &e2], true), Some(MIRROR));
        assert_eq!(classify_window_keys_whole_battle([&g1, &g2], [&e1, &e2], true), None);
    }
}
