//! The per-side SWITCH-IN BLOCK SWAP allowlist key (`turn0-construction-speed-tie-switchin-block-swap`).
//!
//! **Root** — the same one as the omniscient E1/A1 keys and the per-side B1 / mirror-flip keys: the
//! unmodelled turn-0 CONSTRUCTION speed-tie shuffle. At a raw-Speed tie between the two leads the
//! sim decides by PRNG which lead's switch-in handlers run first (`BattleQueue.insertChoice` draws
//! `battle.random(firstIndex, lastIndex + 1)` for the tied `runSwitch` actions, and
//! `runSwitch`'s `fieldEvent('SwitchIn')` speed-sorts its handlers with a tie shuffle), while the
//! port's `event::run_start_switchins` falls back to a DETERMINISTIC side order at a tie and
//! draws nothing. The draws themselves are accounted for (the golden's `INIT` seed is the
//! POST-construction seed, and the seed anchor passes); only the resulting ORDER is unmodelled.
//! seed=None-INVISIBLE, so ZERO production impact under `--use-bridge=rust`: at `seed=None` the
//! port is the sole oracle and there is no sim order to disagree with.
//!
//! **The form this key admits, and why B1 / the mirror-flip key miss it.** The cutover-stress
//! repros (`bab_2_16`, `bab_2_18`, `bab_3_4`, a Zapdos-vs-Salamence lead tie at 328 Speed): on
//! the Zapdos owner's per-side stream the sim emits
//! `-ability|<Salamence>|Intimidate|boost`, `-unboost|<Zapdos>|atk|1`, then
//! `-ability|<Zapdos>|Pressure|[silent]`, and the port emits the Pressure line FIRST. The two
//! windows are an identical multiset, but B1's clause (3) admits only `-ability`/`-weather` at a
//! moved position (the `-unboost` moved), and the mirror-flip key needs same-species leads.
//!
//! **The predicate** — ALL must hold, else `None` and the gate FAILS:
//!  (1) `leads_speed_tie` — the two construction-time lead Speeds are EQUAL;
//!  (2) on EACH side, both the golden and the engine stream contain a `|turn|` line and the FIRST
//!      one is exactly `|turn|1` (the window is the turn-0 construction window, never a whole
//!      battle that happens to lack a turn marker), at the SAME index (equal-length windows);
//!  (3) on EACH side, everything from that `|turn|1` to the end of the stream is BYTE-IDENTICAL.
//!      The allowlist is a first-divergence verdict, so without this clause a real divergence
//!      LATER in an allowlisted battle would be hidden behind the construction reorder;
//!  (4) on EACH side, the two windows are an IDENTICAL MULTISET (E1's rule: any content change —
//!      a different stat/value, a dropped or extra line, a mis-targeted `-unboost` — breaks it);
//!  (5) on EACH side whose windows differ, the difference is EXACTLY ONE SWAP of two adjacent
//!      contiguous blocks: over the span from the first to the last differing line, the golden
//!      reads `A ++ B` and the engine `B ++ A`;
//!  (6) `A` and `B` are each a WELL-FORMED turn-0 switch-in block, and their actors are the two
//!      DIFFERENT leads (one `p1a`, one `p2a`). A block is exactly one of:
//!        * a lone `|-weather|<W>|[from] ability: <A>|[of] pNa: <name>` (Sand Stream / Drizzle / Drought),
//!        * a lone `|-ability|pNa: <name>|<Ability>…` whose ability is NOT Intimidate (Pressure, …),
//!        * the Intimidate PAIR `|-ability|pNa: <name>|Intimidate|boost` immediately followed by
//!          `|-unboost|pMa: <name>|atk|<n>` where `pMa` is the FOE's active slot;
//!      so the INTERNAL order of a block is fixed — an `-unboost` ahead of its own Intimidate
//!      announce, or aimed at the Intimidator's own side, is not a block and FAILS even though it
//!      preserves the multiset;
//!  (7) at least one side's windows differ (identical windows are not a swap).
//!
//! What it deliberately does NOT admit (each surfaces as a NEW divergence to be reviewed):
//! Intimidate blocked by Clear Body / Hyper Cutter / White Smoke or a Substitute (a `-fail` /
//! `-immune` / `-activate` tail), Forecast's `-formechange`, a three-block rotation, and a
//! battle that ALSO carries any other per-side or `|request|` residual (clause 3 refuses the
//! composition rather than guessing at it).

use super::parse_ident_field;

/// The allowlist reason this key reports.
pub(crate) const REASON: &str = "turn0-construction-speed-tie-switchin-block-swap";

/// Classify a per-side first divergence against the SWITCH-IN BLOCK SWAP key. `golden` / `engine`
/// are the FULL per-side streams, indexed `[p1, p2]`. See the module doc for clauses (1)-(7).
pub(crate) fn classify_perside_construction_block_swap(
    golden: [&[String]; 2],
    engine: [&[String]; 2],
    leads_speed_tie: bool,
) -> Option<&'static str> {
    // (1) the construction speed tie.
    if !leads_speed_tie {
        return None;
    }
    let mut any_swap = false;
    for side in 0..2 {
        let (g, e) = (golden[side], engine[side]);
        // (2) a real turn-0 window, ending at `|turn|1`, the same length on both.
        let gt = turn_one_index(g)?;
        let et = turn_one_index(e)?;
        if gt != et {
            return None;
        }
        // (3) the rest of the battle, byte-identical on this side.
        if g[gt..] != e[et..] {
            return None;
        }
        let (gw, ew) = (&g[..gt], &e[..et]);
        if gw == ew {
            continue;
        }
        // (4) an identical multiset. IMPLIED by (5) (a swap of two runs preserves the multiset) —
        // kept as E1's rule stated outright; clause (2)'s equal-length check is what keeps (5)'s
        // positional indexing in bounds.
        if !same_multiset(gw, ew) {
            return None;
        }
        // (5) + (6) exactly one adjacent swap of two well-formed blocks of different leads.
        if !is_switchin_block_swap(gw, ew) {
            return None;
        }
        any_swap = true;
    }
    // (7) at least one side actually swapped.
    any_swap.then_some(REASON)
}

/// Index of the FIRST `|turn|` line, only when it is exactly `|turn|1`; `None` otherwise.
fn turn_one_index(stream: &[String]) -> Option<usize> {
    let i = stream.iter().position(|l| l.starts_with("|turn|"))?;
    (stream[i] == "|turn|1").then_some(i)
}

fn same_multiset(a: &[String], b: &[String]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut x = a.to_vec();
    let mut y = b.to_vec();
    x.sort_unstable();
    y.sort_unstable();
    x == y
}

/// Clauses (5) + (6): the two windows differ by exactly one swap of two adjacent contiguous
/// blocks, each a well-formed switch-in block, belonging to the two different leads.
fn is_switchin_block_swap(gw: &[String], ew: &[String]) -> bool {
    if gw.len() != ew.len() {
        return false; // positional comparison below needs equal lengths (never index out of bounds)
    }
    let differs = |i: &usize| gw[*i] != ew[*i];
    let Some(lo) = (0..gw.len()).find(differs) else {
        return false;
    };
    let Some(hi) = (0..gw.len()).rev().find(differs) else {
        return false;
    };
    let (gr, er) = (&gw[lo..=hi], &ew[lo..=hi]);
    let n = gr.len();
    (1..n).any(|k| {
        let (a, b) = (&gr[..k], &gr[k..]);
        if er[..n - k] != *b || er[n - k..] != *a {
            return false;
        }
        match (block_actor_side(a), block_actor_side(b)) {
            (Some(sa), Some(sb)) => sa != sb,
            _ => false,
        }
    })
}

/// The actor's SIDE (`b'1'` / `b'2'`) of a well-formed turn-0 switch-in block, else `None`.
fn block_actor_side(block: &[String]) -> Option<u8> {
    match block {
        [one] => lone_announce_side(one),
        [head, tail] => intimidate_pair_side(head, tail),
        _ => None,
    }
}

/// The side of an active-slot ident field (`pNa: <name>`, optionally `[of] `-prefixed as `want`).
fn ident_side(field: &str, want_prefix: &str) -> Option<u8> {
    let (prefix, slot, _name) = parse_ident_field(field)?;
    (prefix == want_prefix).then(|| slot.as_bytes()[1])
}

/// A single-line block: a switch-in weather announce or a non-Intimidate ability announce.
fn lone_announce_side(line: &str) -> Option<u8> {
    let f: Vec<&str> = line.split('|').collect();
    match f.as_slice() {
        ["", "-weather", weather, from, of] => {
            if weather.is_empty() || !from.starts_with("[from] ability: ") {
                return None;
            }
            ident_side(of, "[of] ")
        }
        ["", "-ability", actor, ability, ..] => {
            if ability.is_empty() || *ability == "Intimidate" {
                return None; // Intimidate is only ever admitted as the full announce+unboost pair.
            }
            ident_side(actor, "")
        }
        _ => None,
    }
}

/// The two-line Intimidate block: the announce, then the FOE's attack drop.
fn intimidate_pair_side(head: &str, tail: &str) -> Option<u8> {
    let h: Vec<&str> = head.split('|').collect();
    let t: Vec<&str> = tail.split('|').collect();
    let (actor, target, amount) = match (h.as_slice(), t.as_slice()) {
        (["", "-ability", actor, "Intimidate", "boost"], ["", "-unboost", target, "atk", amount]) => {
            (*actor, *target, *amount)
        }
        _ => return None,
    };
    if amount.is_empty() || !amount.bytes().all(|c| c.is_ascii_digit()) {
        return None;
    }
    let actor_side = ident_side(actor, "")?;
    let target_side = ident_side(target, "")?;
    // Intimidate lowers the FOE: an `-unboost` on the Intimidator's own side is not this block.
    (actor_side != target_side).then_some(actor_side)
}

// ── Gate-integrity tests: the NEGATIVES are the load-bearing half ──────────────────────────────
//
// Every negative below is a way a genuine per-side bug could sit inside (or behind) a
// construction speed-tie window. Reverting any one clause makes at least one of them wrongly
// return `Some(..)`, so a test fails.
#[cfg(test)]
mod tests {
    use super::*;

    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    const HEAD: &[&str] = &[
        "|t:|<NORMALIZED>",
        "|gametype|singles",
        "|player|p1|P1||",
        "|player|p2|P2||",
        "|gen|3",
        "|tier|[Gen 3] OU",
        "|",
        "|teamsize|p1|6",
        "|teamsize|p2|6",
        "|start",
        "|switch|p1a: Zapdos|Zapdos|321/321",
        "|switch|p2a: Salamence|Salamence, M|100/100",
    ];
    const INTIM: &str = "|-ability|p2a: Salamence|Intimidate|boost";
    const UNBOOST: &str = "|-unboost|p1a: Zapdos|atk|1";
    const PRESSURE: &str = "|-ability|p1a: Zapdos|Pressure|[silent]";
    const TAIL: &[&str] = &[
        "|turn|1",
        "|request|{\"active\":[]}",
        "|",
        "|switch|p1a: Breloom|Breloom, M|261/261",
        "|move|p2a: Salamence|Brick Break|p1a: Breloom",
        "|-damage|p1a: Breloom|172/261",
        "|turn|2",
    ];

    fn stream(framing: &[&str]) -> Vec<String> {
        let mut s = v(HEAD);
        s.extend(v(framing));
        s.extend(v(TAIL));
        s
    }

    /// The bab_2_16 / bab_2_18 shape: p1 (the Zapdos owner) sees the sim's Intimidate-first order,
    /// the port emits Pressure first; p2's stream is byte-identical.
    fn repro() -> ([Vec<String>; 2], [Vec<String>; 2]) {
        let g1 = stream(&[INTIM, UNBOOST, PRESSURE]);
        let e1 = stream(&[PRESSURE, INTIM, UNBOOST]);
        let g2 = stream(&[INTIM, UNBOOST]);
        let e2 = g2.clone();
        ([g1, g2], [e1, e2])
    }

    fn classify(g: &[Vec<String>; 2], e: &[Vec<String>; 2], tie: bool) -> Option<&'static str> {
        classify_perside_construction_block_swap([&g[0], &g[1]], [&e[0], &e[1]], tie)
    }

    // ── POSITIVES ──

    #[test]
    fn the_cutover_stress_repro_shape_is_allowlisted() {
        let (g, e) = repro();
        assert_eq!(classify(&g, &e, true), Some(REASON));
    }

    #[test]
    fn the_mirrored_orientation_is_allowlisted() {
        // The bab_3_4 shape: Salamence is p1, Zapdos p2; the p2 stream carries the swap and the
        // sim ran Zapdos's Pressure FIRST.
        let head2: Vec<String> = v(HEAD)
            .into_iter()
            .map(|l| match l.as_str() {
                "|switch|p1a: Zapdos|Zapdos|321/321" => "|switch|p1a: Salamence|Salamence, M|100/100".to_string(),
                "|switch|p2a: Salamence|Salamence, M|100/100" => "|switch|p2a: Zapdos|Zapdos|321/321".to_string(),
                _ => l,
            })
            .collect();
        let mk = |framing: &[&str]| {
            let mut s = head2.clone();
            s.extend(v(framing));
            s.extend(v(TAIL));
            s
        };
        let intim = "|-ability|p1a: Salamence|Intimidate|boost";
        let unboost = "|-unboost|p2a: Zapdos|atk|1";
        let pressure = "|-ability|p2a: Zapdos|Pressure|[silent]";
        let g = [mk(&[intim, unboost]), mk(&[pressure, intim, unboost])];
        let e = [mk(&[intim, unboost]), mk(&[intim, unboost, pressure])];
        assert_eq!(classify(&g, &e, true), Some(REASON));
    }

    #[test]
    fn a_weather_block_swapped_with_an_intimidate_pair_is_allowlisted() {
        // A Sand-Stream-vs-Intimidate-shaped tie: a lone `-weather` block and the Intimidate pair.
        // (The key is STRUCTURAL: which ability a species carries is the sim's content, and the
        // multiset clause already requires every line to be byte-identical to one the sim wrote.)
        let weather = "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Zapdos";
        let g = [stream(&[INTIM, UNBOOST, weather]), stream(&[INTIM, UNBOOST, weather])];
        let e = [stream(&[weather, INTIM, UNBOOST]), stream(&[weather, INTIM, UNBOOST])];
        assert_eq!(classify(&g, &e, true), Some(REASON));
    }

    // ── NEGATIVES (load-bearing) ──

    #[test]
    fn a_distinct_speed_pair_fails_clause_1() {
        let (g, e) = repro();
        assert_eq!(classify(&g, &e, false), None);
    }

    #[test]
    fn a_changed_unboost_amount_fails() {
        // `atk|1` -> `atk|2` inside the reordered window: the multiset differs.
        let (g, mut e) = repro();
        e[0] = stream(&[PRESSURE, INTIM, "|-unboost|p1a: Zapdos|atk|2"]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_changed_unboost_stat_fails() {
        let (g, mut e) = repro();
        e[0] = stream(&[PRESSURE, INTIM, "|-unboost|p1a: Zapdos|def|1"]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_changed_ability_fails() {
        let (g, mut e) = repro();
        e[0] = stream(&["|-ability|p1a: Zapdos|Insomnia|[silent]", INTIM, UNBOOST]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_dropped_window_line_fails() {
        let (g, mut e) = repro();
        e[0] = stream(&[PRESSURE, INTIM]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn an_extra_window_line_fails() {
        let (g, mut e) = repro();
        e[0] = stream(&[PRESSURE, INTIM, UNBOOST, UNBOOST]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn an_unboost_ahead_of_its_own_intimidate_fails_although_the_multiset_matches() {
        // What plain E1 (multiset-only) would swallow: the Intimidate pair's INTERNAL order broken.
        let (g, mut e) = repro();
        e[0] = stream(&[UNBOOST, INTIM, PRESSURE]);
        assert!(same_multiset(&g[0], &e[0]), "precondition: the multiset is preserved");
        assert_eq!(classify(&g, &e, true), None);
        e[0] = stream(&[PRESSURE, UNBOOST, INTIM]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn an_unboost_on_the_intimidators_own_side_is_not_a_block() {
        // Even when golden and engine AGREE on the (impossible) self-targeted drop, a pair whose
        // `-unboost` lands on the Intimidator's own side is not a switch-in block.
        let self_drop = "|-unboost|p2a: Salamence|atk|1";
        let g = [stream(&[INTIM, self_drop, PRESSURE]), stream(&[INTIM, self_drop])];
        let e = [stream(&[PRESSURE, INTIM, self_drop]), stream(&[INTIM, self_drop])];
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_mistargeted_unboost_fails() {
        let (g, mut e) = repro();
        e[0] = stream(&[PRESSURE, INTIM, "|-unboost|p2a: Salamence|atk|1"]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn two_blocks_of_the_same_lead_fail() {
        // A swap is only a construction speed-tie artifact when the two blocks belong to the two
        // DIFFERENT leads.
        let other = "|-ability|p1a: Zapdos|Trace|[silent]";
        let g = [stream(&[other, PRESSURE]), stream(&[])];
        let e = [stream(&[PRESSURE, other]), stream(&[])];
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_swapped_non_framing_line_fails() {
        let (g, mut e) = repro();
        let mut s = stream(&[INTIM, UNBOOST, PRESSURE]);
        s.swap(10, 11); // the two `|switch|` lines
        e[0] = s;
        assert!(same_multiset(&g[0], &e[0]));
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_later_divergence_on_the_same_side_fails_clause_3() {
        // The masking guard: the construction reorder must not hide a real bug later in the battle.
        let (g, mut e) = repro();
        let last = e[0].len() - 2;
        e[0][last] = "|-damage|p1a: Breloom|171/261".to_string();
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_later_divergence_on_the_other_side_fails_clause_3() {
        let (g, mut e) = repro();
        let last = e[1].len() - 2;
        e[1][last] = "|-damage|p1a: Breloom|171/261".to_string();
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_truncated_engine_stream_fails_clause_3() {
        let (g, mut e) = repro();
        e[0].pop();
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_content_change_in_the_other_sides_window_fails() {
        let (g, mut e) = repro();
        e[1] = stream(&[INTIM, "|-unboost|p1a: Zapdos|atk|2"]);
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_stream_with_no_turn_marker_fails_clause_2() {
        // Without `|turn|1` the "window" would be the whole battle — never admitted.
        let (mut g, mut e) = repro();
        for s in g.iter_mut().chain(e.iter_mut()) {
            s.retain(|l| !l.starts_with("|turn|"));
        }
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_first_turn_marker_that_is_not_turn_one_fails_clause_2() {
        let (mut g, mut e) = repro();
        for s in g.iter_mut().chain(e.iter_mut()) {
            for l in s.iter_mut() {
                if l == "|turn|1" {
                    *l = "|turn|7".to_string();
                }
            }
        }
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_three_block_rotation_fails() {
        let weather = "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Zapdos";
        let g = [stream(&[INTIM, UNBOOST, PRESSURE, weather]), stream(&[])];
        let e = [stream(&[weather, INTIM, UNBOOST, PRESSURE]), stream(&[])];
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_weather_line_without_an_ability_source_is_not_a_block() {
        // Five fields, so only the `[from] ability:` check (not the field count) can refuse it.
        let not_ability = "|-weather|Sandstorm|[from] item: Smooth Rock|[of] p1a: Zapdos";
        let g = [stream(&[INTIM, UNBOOST, not_ability]), stream(&[])];
        let e = [stream(&[not_ability, INTIM, UNBOOST]), stream(&[])];
        assert_eq!(classify(&g, &e, true), None);
        let upkeep = "|-weather|Sandstorm|[upkeep]";
        let g = [stream(&[INTIM, UNBOOST, upkeep]), stream(&[])];
        let e = [stream(&[upkeep, INTIM, UNBOOST]), stream(&[])];
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_three_line_block_is_not_a_block() {
        // A block is one line or the Intimidate pair — never a longer run that merely STARTS
        // with an admissible announce (the swap below is otherwise a clean two-run exchange).
        let trace = "|-ability|p1a: Zapdos|Pressure|[from] ability: Trace|[of] p2a: Salamence";
        let weather = "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Zapdos";
        let g = [stream(&[PRESSURE, trace, weather, INTIM, UNBOOST]), stream(&[])];
        let e = [stream(&[INTIM, UNBOOST, PRESSURE, trace, weather]), stream(&[])];
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_non_numeric_unboost_amount_is_not_a_block() {
        let odd = "|-unboost|p1a: Zapdos|atk|x";
        let g = [stream(&[INTIM, odd, PRESSURE]), stream(&[])];
        let e = [stream(&[PRESSURE, INTIM, odd]), stream(&[])];
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn a_lone_intimidate_announce_is_not_a_block() {
        // Intimidate is admitted ONLY as the announce + foe-drop pair (a blocked Intimidate's
        // `-fail`/`-immune` tail is out of scope and must surface for review).
        let g = [stream(&[INTIM, PRESSURE]), stream(&[])];
        let e = [stream(&[PRESSURE, INTIM]), stream(&[])];
        assert_eq!(classify(&g, &e, true), None);
    }

    #[test]
    fn identical_streams_are_not_a_swap() {
        let (g, _) = repro();
        assert_eq!(classify(&g, &g.clone(), true), None);
    }
}
