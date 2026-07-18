//! `bridge_replay` — replay a bridge-capture golden's (teams, seed, CMD stream)
//! through the crate's per-side bridge emitter and either PRINT the per-side chunk
//! streams or DIFF them against the golden's recorded per-side CHUNK rows.
//!
//! Additive (links only public lib APIs), like `ab_replay`. This is the entry point
//! the future random-team request A/B fuzzer drives: it feeds a (teams, seed, CMD)
//! record and compares the Rust per-side stream to the real sim's.
//!
//! Usage:
//!   bridge_replay <golden.txt>                 # diff EVERY battle, print PASS/FAIL summary
//!   bridge_replay <golden.txt> <battle_id>     # diff ONE battle, print first divergence
//!   bridge_replay <golden.txt> <battle_id> --print   # print the Rust per-side streams
//!   bridge_replay <golden.txt> --ab            # ONE JSON verdict per battle (the A/B fuzzer
//!                                                driver; never panics — a panic is caught +
//!                                                reported as {"verdict":"panic"})
//!   bridge_replay <repro-dir>                  # replay a saved repro dir's battle.txt
//!
//! Exit code 0 = all diffed battles byte-equal; 1 = a divergence (or an error).

use std::process::ExitCode;

use pokesim::bridge::{
    bridge_opts, parse_bridge_golden, run_full_battle_bridge, run_full_battle_bridge_core,
    GoldenBattle,
};
use pokesim::dex::Dex;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: bridge_replay <golden.txt> [battle_id] [--print]");
        return ExitCode::from(2);
    }
    let arg1 = &args[1];
    let battle_id = args.get(2).filter(|a| !a.starts_with("--")).cloned();
    let do_print = args.iter().any(|a| a == "--print");
    let do_ab = args.iter().any(|a| a == "--ab");

    // A repro-dir argument resolves to `<dir>/battle.txt` (the repro→replay path).
    let path: String = {
        let p = std::path::Path::new(arg1);
        if p.is_dir() {
            p.join("battle.txt").to_string_lossy().into_owned()
        } else {
            arg1.clone()
        }
    };

    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    let battles = match parse_bridge_golden(&text) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("parse error: {e}");
            return ExitCode::from(2);
        }
    };
    let dex = Dex::for_gen(3);

    // ── A/B mode: one JSON verdict per battle, panic-caught (never dies). ──
    if do_ab {
        let mut all_ok = true;
        let mut n_ok = 0usize;
        let mut n_div = 0usize;
        for b in &battles {
            let v = ab_verdict(b, &dex);
            if v.contains("\"verdict\":\"ok\"") {
                n_ok += 1;
            } else {
                n_div += 1;
                all_ok = false;
            }
            println!("{v}");
        }
        println!("{{\"chunk_summary\":true,\"battles\":{},\"ok\":{n_ok},\"diverged\":{n_div}}}", battles.len());
        return if all_ok { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }

    let selected: Vec<&GoldenBattle> = match &battle_id {
        Some(id) => battles.iter().filter(|b| &b.id == id).collect(),
        None => battles.iter().collect(),
    };
    if selected.is_empty() {
        eprintln!("no battle matched {battle_id:?}");
        return ExitCode::from(2);
    }

    let mut all_pass = true;
    let mut total_chunks = 0usize;
    let mut total_requests = 0usize;
    let mut total_trapped = 0usize;
    for b in &selected {
        let opts = bridge_opts(&b.format_id, b.seed.clone(), &b.p1_team, &b.p2_team);
        let streams = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            run_full_battle_bridge(&opts, &b.cmds, &dex)
        })) {
            Ok(Ok(s)) => s,
            Ok(Err(e)) => {
                println!("[{}] ERROR: {e}", b.id);
                all_pass = false;
                continue;
            }
            Err(_) => {
                println!("[{}] PANIC during replay", b.id);
                all_pass = false;
                continue;
            }
        };
        if do_print {
            println!("=== {} p1 ===", b.id);
            for l in &streams.p1 {
                println!("{l}");
            }
            println!("=== {} p2 ===", b.id);
            for l in &streams.p2 {
                println!("{l}");
            }
            continue;
        }
        let (ok, msg) = diff_side(&b.id, "p1", &streams.p1, &b.p1_expected);
        let (ok2, msg2) = diff_side(&b.id, "p2", &streams.p2, &b.p2_expected);
        for l in &streams.p1 {
            if l.starts_with("|request|") {
                total_requests += 1;
                if l.contains("\"trapped\":true") {
                    total_trapped += 1;
                }
            }
        }
        for l in &streams.p2 {
            if l.starts_with("|request|") {
                total_requests += 1;
                if l.contains("\"trapped\":true") {
                    total_trapped += 1;
                }
            }
        }
        total_chunks += streams.p1.len() + streams.p2.len();
        if ok && ok2 {
            if battle_id.is_some() {
                println!("[{}] PASS ({} p1 lines, {} p2 lines)", b.id, streams.p1.len(), streams.p2.len());
            }
        } else {
            all_pass = false;
            if !ok {
                println!("{msg}");
            }
            if !ok2 {
                println!("{msg2}");
            }
        }
    }
    if !do_print {
        println!(
            "SUMMARY: {} battles, {} per-side lines, {} |request| frames, {} trapped:true — {}",
            selected.len(),
            total_chunks,
            total_requests,
            total_trapped,
            if all_pass { "ALL BYTE-EQUAL" } else { "DIVERGED" }
        );
    }
    if all_pass {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

/// Compare a Rust per-side stream to the golden's expected lines; return (ok, msg)
/// with a first-divergence description.
fn diff_side(id: &str, side: &str, got: &[String], want: &[String]) -> (bool, String) {
    let n = got.len().min(want.len());
    for i in 0..n {
        if got[i] != want[i] {
            return (
                false,
                format!(
                    "[{id}] {side} DIVERGE at line {i}:\n  want: {}\n  got:  {}",
                    truncate(&want[i]),
                    truncate(&got[i])
                ),
            );
        }
    }
    if got.len() != want.len() {
        let extra_side = if got.len() > want.len() { "GOT extra" } else { "WANT extra" };
        let sample = if got.len() > want.len() {
            got.get(n).map(|s| truncate(s)).unwrap_or_default()
        } else {
            want.get(n).map(|s| truncate(s)).unwrap_or_default()
        };
        return (
            false,
            format!(
                "[{id}] {side} LENGTH mismatch: got {} lines, want {} ({extra_side} at {n}: {sample})",
                got.len(),
                want.len()
            ),
        );
    }
    (true, String::new())
}

fn truncate(s: &str) -> String {
    if s.len() > 200 {
        format!("{}…", &s[..200])
    } else {
        s.to_string()
    }
}

// ===========================================================================
// A/B mode — one JSON verdict per battle for the request/per-side fuzzer.
// ===========================================================================

/// Replay `b` through the bridge emitter (panic-caught) and produce ONE JSON
/// verdict line: `ok` or the FIRST divergence with a taxonomy kind. SEED-ANCHOR
/// PRECEDENCE (mirrors `ab_replay.rs`): a per-decision engine-seed mismatch (an
/// UPSTREAM draw desync) is reported as `kind:"seed"` BEFORE the per-side byte diff,
/// so a per-side/request divergence is partitioned into an upstream engine bug vs a
/// genuine per-side/request-serializer bug. A documented request-DISPLAY deferral
/// (Curse target / return102 / gender-level details) is `allowlisted` NARROWLY.
/// The A2 SEED-ANCHOR alignment (`gen3_perside_seed_anchor_makerequest_align_v1`).
///
/// The port's per-`|request|`-boundary seed list (`got`) can carry MORE boundaries than the
/// omniscient fuzzer's per-decision `rec.seeds` (`exp`): on a phaze-drag / on-entry-faint
/// forced-switch turn the port records an EXTRA **zero-draw** boundary (a forced replacement
/// that draws nothing, so its seed EQUALS the previous boundary's seed) where the fuzzer's
/// per-while-iteration capture collapses it into the surrounding decision. The game is
/// byte-identical to `|win|` (proven by the per-side byte diff without `--ab`), so this is a
/// pure decision-boundary BOOKKEEPING offset, NOT an engine draw bug.
///
/// This aligns the two seed streams as a SUBSEQUENCE, tolerating ONLY the port's extra
/// ZERO-DRAW boundaries — a `got[i]` that equals `got[i-1]` (drew nothing) AND does not match
/// the expected sim seed is skipped as one such artifact. A GENUINE draw-count desync (an
/// extra/missing engine draw) necessarily changes a seed VALUE non-trivially, so it is NEVER a
/// zero-draw duplicate → it surfaces here as `kind:"seed"` (and, being a real RNG divergence,
/// ALSO breaks the per-side byte stream — the load-bearing discriminator: byte-equal ⇒
/// harmless checkpoint offset, byte-diverge ⇒ real desync). Returns `(sim_decision_index,
/// expected, got)` at the first UN-tolerated divergence, else `None`.
fn anchor_seed_divergence(got: &[String], exp: &[String]) -> Option<(usize, String, String)> {
    let mut i = 0usize; // index into the port's request_seeds (`got`)
    let mut j = 0usize; // index into the sim's rec.seeds (`exp`)
    while i < got.len() && j < exp.len() {
        if got[i] == exp[j] {
            i += 1;
            j += 1;
            continue;
        }
        // Tolerate ONLY an extra port ZERO-DRAW boundary: `got[i] == got[i-1]` (no draw since
        // the prior boundary) — the phaze-drag / forced-switch checkpoint artifact. Anything
        // else is a real seed divergence at the sim's decision `j`.
        if i > 0 && got[i] == got[i - 1] {
            i += 1;
            continue;
        }
        return Some((j, exp[j].clone(), got[i].clone()));
    }
    None
}

fn ab_verdict(b: &GoldenBattle, dex: &Dex) -> String {
    let opts = bridge_opts(&b.format_id, b.seed.clone(), &b.p1_team, &b.p2_team);
    // Build BOTH the per-side chunks AND the ScriptDecision list (the seed anchor
    // replays the latter through `run_full_battle`).
    let core = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_full_battle_bridge_core(&opts, &b.cmds, dex)
    }));
    let (chunks, _ended, _script, request_seeds) = match core {
        Ok(Ok(s)) => s,
        Ok(Err(e)) => {
            return format!(
                "{{\"battle\":{},\"verdict\":\"error\",\"kind\":\"error\",\"detail\":{}}}",
                json_str(&b.id),
                json_str(&e)
            );
        }
        Err(panic) => {
            let msg = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "panic".to_string());
            return format!(
                "{{\"battle\":{},\"verdict\":\"panic\",\"kind\":\"panic\",\"detail\":{}}}",
                json_str(&b.id),
                json_str(&msg)
            );
        }
    };
    let streams = chunks.flatten();

    // ── SEED ANCHOR (precedence over the per-side diff). ──
    // Assert each decision's post-decision engine seed — captured by the bridge core at the
    // exact `makeRequest` FLUSH boundary (`request_seeds`, the A2 alignment fix) — equals the
    // omniscient oracle's recorded `seedAfter`. A mismatch means the ENGINE desynced upstream
    // (a genuine draw-count bug — belongs to the omniscient fix-queue, NOT the
    // per-side/request serializer) — reported as `kind:"seed"`. Skipped when the golden
    // carries no SEED rows (backward-compatible). The `request_seeds` checkpoint is
    // makeRequest-aligned (post-endTurn-Quick-Claw, 1:1 with the fuzzer's `rec.seeds.push`),
    // so a harmless phaze-drag/forced-switch checkpoint OFFSET no longer masquerades as a
    // seed desync — while a REAL extra/missing phaze-turn draw (which shifts the whole seed
    // stream) is STILL caught here (proven by the A2 integrity injection).
    if !b.seeds.is_empty() {
        if let Some((j, exp, got)) = anchor_seed_divergence(&request_seeds, &b.seeds) {
            return format!(
                "{{\"battle\":{},\"verdict\":\"diverge\",\"kind\":\"seed\",\"decision\":{j},\"expected\":{},\"got\":{}}}",
                json_str(&b.id),
                json_str(&exp),
                json_str(&got),
            );
        }
    }

    // Diff each side; report the FIRST (earliest-line) divergence across both.
    let p1 = ab_diff_side("p1", &streams.p1, &b.p1_expected);
    let p2 = ab_diff_side("p2", &streams.p2, &b.p2_expected);
    let first = match (p1, p2) {
        (None, None) => {
            return format!("{{\"battle\":{},\"verdict\":\"ok\"}}", json_str(&b.id));
        }
        (Some(a), None) => a,
        (None, Some(b_)) => b_,
        (Some(a), Some(b_)) => {
            if a.line <= b_.line {
                a
            } else {
                b_
            }
        }
    };
    // Narrowly allowlist a documented request-DISPLAY deferral (never a blanket ignore).
    let allowlisted = if first.kind == "request" {
        classify_known_perside_residual(&first.expected, &first.got)
    } else if first.kind == "perside" {
        // B1: the turn0-construction-speed-tie-ORDER-flip — a per-side framing
        // `-ability`/`-weather` line ORDER flip at a construction speed-tie (the SAME
        // turn-0 construction-window root as the omniscient E1/A1 keys, but a DISTINCT
        // per-side form — a NON-mirror pair, a framing ORDER flip of TWO DIFFERENT line
        // types, not A1's single-line `[of]`-attribution flip). Needs `leads_speed_tie`
        // + the FULL framing windows (all per-side lines before the first `|turn|`).
        let leads_speed_tie =
            match (lead_speed(&b.p1_team, dex), lead_speed(&b.p2_team, dex)) {
                (Some(a), Some(c)) => a == c,
                _ => false,
            };
        let (gw, ew) = if first.side == "p1" {
            (framing_window(&b.p1_expected), framing_window(&streams.p1))
        } else {
            (framing_window(&b.p2_expected), framing_window(&streams.p2))
        };
        classify_perside_construction_order_flip(&gw, &ew, leads_speed_tie)
    } else {
        None
    };
    let allow_field = match allowlisted {
        Some(reason) => format!(",\"allowlisted\":{}", json_str(reason)),
        None => String::new(),
    };
    format!(
        "{{\"battle\":{},\"verdict\":\"diverge\",\"kind\":\"{}\",\"side\":\"{}\",\"line\":{},\"expected\":{},\"got\":{}{}}}",
        json_str(&b.id),
        first.kind,
        first.side,
        first.line,
        json_str(&first.expected),
        json_str(&first.got),
        allow_field,
    )
}

/// The KNOWN-RESIDUAL allowlist for the per-side/request byte gate — the parallel of
/// `ab_replay.rs::classify_known_residual`, for the `|request|` JSON surface. Returns a
/// reason string ONLY when the SOLE difference between the expected (sim) and got (port)
/// `|request|` lines matches ONE of the three documented request-DISPLAY deferrals (no
/// legality/draw impact) — else `None`, so a NEW request-form divergence (a genuine bug)
/// keeps `allowlisted: None` and FAILS the gate. Entries are NARROW + STRUCTURAL, keyed on
/// the exact byte form + reason (never a blanket "ignore request").
///
/// Both lines must be `|request|` JSON. The diff is located as the single field whose value
/// differs; the allowlist fires only if that field is one of:
///   - **curse-nonghost-target-self-vs-normal** — a Curse move-slot's `"target":"self"` (sim,
///     Curse's `nonGhostTarget`) vs `"target":"normal"` (port renders the base dex target).
///   - **return102-numeric-alias** — a `side.pokemon[].moves[]` entry `"return102"`↔`"return"`
///     (or `"frustration102"`↔`"frustration"`), the numeric-BP moveid alias.
///   - **gender-level-details-construction-draw** — a `details` field differing SOLELY by a
///     `, L<n>` level suffix or a construction-drawn gender (`, M`/`, F`) — the unspecified-gender
///     / non-L100 construction-window display gap (inactive on the pinned-gender L100 pool run).
fn classify_known_perside_residual(expected: &str, got: &str) -> Option<&'static str> {
    if !expected.starts_with("|request|") || !got.starts_with("|request|") {
        return None;
    }
    // Progressively reconcile `got` toward `expected` by applying EACH documented
    // request-DISPLAY transform (a co-occurring pair — Curse-target + return102 on the same
    // team — is common on real pool teams), tracking which fired. Allowlist ONLY IF the fully
    // reconciled `got` == `expected` (every diff was a documented deferral); any residual
    // difference (a genuine per-side/request bug) leaves them unequal → None → the gate fails.
    let mut cur = got.to_string();
    let mut exp = expected.to_string();
    let mut curse = false;
    let mut alias = false;
    // Curse move-slot target:normal → target:self, single-occurrence anchored at the curse id
    // (so an Earthquake `target:normal` is never touched).
    if let Some(next) = reconcile_curse_target(&cur, &exp) {
        cur = next;
        curse = true;
    }
    // return102 / frustration102 numeric-BP moveid alias. THE SIM RENDERS IT INCONSISTENTLY
    // (SIM-PROBED): the `side.pokemon[].moves[]` ROSTER carries the numeric `return102`, but the
    // `active[].moves[].id` is the BARE `return` and the `.move` DISPLAY is `Return 102` (a SPACE) —
    // so a single-direction `replace("return","return102")` OVER-CORRECTS the active id AND misses
    // the display, and on a co-occurring Curse+Return team leaves a residual → the WHOLE reconcile
    // (incl. the correctly-fixed Curse target) returns None → BOTH escape. Instead NORMALIZE BOTH
    // sides by COLLAPSING every alias form to the bare token, then compare (poke-env resolves both —
    // the numeric BP is a display alias, `gen3_perside_request_byte_fuzz_v1`). Anchored on the
    // numeric/spaced tokens, so a non-alias move is never touched.
    {
        let norm = |s: &str| {
            s.replace("Return 102", "Return")
                .replace("return102", "return")
                .replace("Frustration 102", "Frustration")
                .replace("frustration102", "frustration")
        };
        let cn = norm(&cur);
        let en = norm(&exp);
        if cn != cur || en != exp {
            alias = true;
        }
        cur = cn;
        exp = en;
    }
    // gender/level `details` construction-window display suffix (inactive on pinned-gender L100).
    let details = if cur != exp && details_suffix_only(&exp, &cur) {
        cur = reconcile_details_suffix(&cur, &exp);
        true
    } else {
        false
    };
    if cur != exp {
        return None; // a residual, un-cataloged difference remains → NOT a known deferral.
    }
    // Report the (single or dominant) reason. Order: curse > return102 > details.
    if curse {
        Some("curse-nonghost-target-self-vs-normal")
    } else if alias {
        Some("return102-numeric-alias")
    } else if details {
        Some("gender-level-details-construction-draw")
    } else {
        None
    }
}

/// The construction-time lead Speed for a packed team (`stats[5]` of slot 0) — the
/// `leads_speed_tie` precondition for the B1 `turn0-construction-speed-tie-order-flip`
/// key (the per-side parallel of `ab_replay.rs`'s `leads_speed_tie`, which reads the
/// live battle state; here we compute it from the packed team). `None` on any
/// unpack/stat-compute failure (→ NOT a tie → the key never fires).
fn lead_speed(team: &str, dex: &Dex) -> Option<u16> {
    let sets = pokesim::team::unpack(team, dex).ok()?;
    let lead = sets.first()?;
    pokesim::stats::compute_stats(lead, dex).ok().map(|s| s[5])
}

/// The FULL framing WINDOW of a per-side stream: every line BEFORE the first `|turn|`
/// (the construction / lead-switch-in / first-request framing block). B1's order flip
/// lives ENTIRELY in this window.
fn framing_window(side: &[String]) -> Vec<String> {
    let end = side
        .iter()
        .position(|l| l.starts_with("|turn|"))
        .unwrap_or(side.len());
    side[..end].to_vec()
}

/// B1 — the `turn0-construction-speed-tie-order-flip` per-side allowlist key. The
/// unmodeled turn-0 construction speed-tie Fisher-Yates shuffle (the project-wide
/// seed-convention deferral) can emit two same-side framing `-ability`/`-weather` lines
/// in the OPPOSITE order vs the sim (`bab_3_15`: p2 sees `-ability|Pressure` then
/// `-weather|Sandstorm|[of] Tyranitar` where the port emits them flipped, on a
/// Tyranitar-213-vs-Suicune-213 construction speed-tie). seed=None-INVISIBLE (both the
/// offline replay and the production bridge share `event::run_start_switchins`, which
/// uses a DETERMINISTIC side-order at a raw-Speed tie and draws nothing → correct
/// production obs under `--use-bridge=rust`).
///
/// PREDICATE — all must hold, else `None` → the gate FAILS (a real per-side order bug is
/// never swallowed):
///  (1) `leads_speed_tie` — the two construction-time lead Speeds are EQUAL;
///  (2) the two FULL framing windows are equal-length AND an IDENTICAL MULTISET (a pure
///      PERMUTATION — same lines, different order; a CONTENT change / non-tie / added or
///      dropped line makes the multisets DIFFER);
///  (3) EVERY position on which the two windows differ is, in BOTH windows, a framing
///      `|-ability|`/`|-weather|` line (never a `|request|`, HP-bearing, or content line).
///
/// GATE-INTEGRITY: mangling one framing line's CONTENT (flip the weather `[of]`, change
/// `Sandstorm`→`RainDance`, `Pressure`→`Insomnia`, or add/drop a line) breaks the multiset
/// (2) → `None`; a DISTINCT-speed lead pair breaks (1) → `None`. Locked by the
/// `perside_construction_order_flip_tests` below.
fn classify_perside_construction_order_flip(
    golden_window: &[String],
    engine_window: &[String],
    leads_speed_tie: bool,
) -> Option<&'static str> {
    // (1) construction speed-tie.
    if !leads_speed_tie {
        return None;
    }
    // Equal length is a precondition for both the multiset AND the positional checks.
    if golden_window.len() != engine_window.len() || golden_window.is_empty() {
        return None;
    }
    // (2) identical MULTISET (a pure permutation).
    let mut g = golden_window.to_vec();
    let mut e = engine_window.to_vec();
    g.sort();
    e.sort();
    if g != e {
        return None;
    }
    // (3) every DIFFERING position is a framing `-ability`/`-weather` line in BOTH.
    let is_framing_reorder =
        |l: &str| l.starts_with("|-ability|") || l.starts_with("|-weather|");
    let mut any_diff = false;
    for (a, b) in golden_window.iter().zip(engine_window.iter()) {
        if a != b {
            any_diff = true;
            if !is_framing_reorder(a) || !is_framing_reorder(b) {
                return None; // a NON-framing line moved → a real per-side order bug.
            }
        }
    }
    if !any_diff {
        return None; // identical windows are not a flip (the divergence is elsewhere).
    }
    Some("turn0-construction-speed-tie-order-flip")
}

/// Reconcile the Curse move-slot's `"target":"normal"` (port) toward `"target":"self"` (sim),
/// single-occurrence anchored at the curse id, IFF that leaves the strings closer without
/// disturbing any other slot. Returns the reconciled `got` when the curse slot's target was the
/// diverging field, else None.
fn reconcile_curse_target(got: &str, expected: &str) -> Option<String> {
    if !expected.contains("\"id\":\"curse\"") || !got.contains("\"id\":\"curse\"") {
        return None;
    }
    let anchor = got.find("\"id\":\"curse\"")?;
    let rel = got[anchor..].find("\"target\":\"normal\"").map(|r| anchor + r)?;
    // Only reconcile if the sim's curse slot actually carries `target:self` there.
    if !expected.contains("\"id\":\"curse\",\"pp\"") {
        return None;
    }
    let mut out = String::with_capacity(got.len() + 1);
    out.push_str(&got[..rel]);
    out.push_str("\"target\":\"self\"");
    out.push_str(&got[rel + "\"target\":\"normal\"".len()..]);
    Some(out)
}

/// Reconcile `got` by re-inserting the `, L<n>` / `, <gender>` details suffixes the sim carries.
/// Only called after `details_suffix_only` confirmed that is the SOLE remaining difference; a
/// simple structural transform is impossible in general, so this returns `expected` when the
/// suffix-stripped forms already match (the check guarantees it). Kept for randbats/random modes.
fn reconcile_details_suffix(got: &str, expected: &str) -> String {
    // `details_suffix_only` guarantees stripping the sim's construction suffixes from `expected`
    // yields `got`; so the reconciled form is exactly `expected`.
    let _ = got;
    expected.to_string()
}

/// `true` iff the two `|request|` lines differ SOLELY in `"details":"…"` values, where the
/// sim's value is the port's value plus a trailing `, L<n>` and/or `, <M|F>` construction
/// suffix. (Inactive on the pinned-gender L100 pool run — kept for randbats/random modes.)
fn details_suffix_only(expected: &str, got: &str) -> bool {
    // Strip every `, L<digits>` level suffix from BOTH inside a details value, then check the
    // remainder differs only by a gender-letter suffix the sim adds. Conservative: require the
    // got-string to be obtainable from the expected by DELETING `, L<n>`/`, M`/`, F` details
    // suffixes — i.e. the port omitted a construction-window display token.
    if !expected.contains("\"details\":") || expected == got {
        return false;
    }
    let strip = |s: &str| -> String {
        // Remove ", L<digits>" and a trailing ", M"/", F" only inside details-ish spans; a
        // coarse global strip is safe here because those tokens never appear elsewhere in a
        // request JSON value.
        let mut out = String::with_capacity(s.len());
        let bytes = s.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if s[i..].starts_with(", L") {
                let mut j = i + 3;
                while j < bytes.len() && bytes[j].is_ascii_digit() {
                    j += 1;
                }
                if j > i + 3 {
                    i = j;
                    continue;
                }
            }
            out.push(bytes[i] as char);
            i += 1;
        }
        out
    };
    // After stripping level suffixes, the only allowed residual difference is the sim carrying
    // a `, M`/`, F` gender the port omitted inside a details value.
    let e = strip(expected);
    let g = strip(got);
    e == g || e.replace(", M\"", "\"").replace(", F\"", "\"") == g
}

struct AbDiff {
    side: &'static str,
    line: usize,
    kind: &'static str,
    expected: String,
    got: String,
}

/// Diff one side; return the first divergence (a length mismatch is `chunk_count`,
/// else the kind is classified from the first-diverging expected/got line).
fn ab_diff_side(side: &'static str, got: &[String], want: &[String]) -> Option<AbDiff> {
    let n = got.len().min(want.len());
    for i in 0..n {
        if got[i] != want[i] {
            return Some(AbDiff {
                side,
                line: i,
                kind: classify_line(&want[i], &got[i]),
                expected: want[i].clone(),
                got: got[i].clone(),
            });
        }
    }
    if got.len() != want.len() {
        let (exp, g) = if got.len() > want.len() {
            (String::new(), got.get(n).cloned().unwrap_or_default())
        } else {
            (want.get(n).cloned().unwrap_or_default(), String::new())
        };
        return Some(AbDiff {
            side,
            line: n,
            kind: "chunk_count",
            expected: exp,
            got: g,
        });
    }
    None
}

/// Taxonomy of a first-diverging per-side line: `request` (a `|request|` JSON) /
/// `error` (a trapped `|error|`) / `privacy` (an HP-bearing line — the HP-fold) /
/// `preamble` (framing before any request) / `perside` (anything else).
fn classify_line(expected: &str, got: &str) -> &'static str {
    let is_req = |l: &str| l.starts_with("|request|");
    let is_err = |l: &str| l.starts_with("|error|");
    let is_hp = |l: &str| {
        l.starts_with("|switch|")
            || l.starts_with("|drag|")
            || l.starts_with("|-damage|")
            || l.starts_with("|-heal|")
            || l.starts_with("|-sethp|")
    };
    let is_framing = |l: &str| {
        l.starts_with("|player|")
            || l.starts_with("|gametype|")
            || l.starts_with("|gen|")
            || l.starts_with("|tier|")
            || l.starts_with("|rule|")
            || l.starts_with("|teamsize|")
            || l.starts_with("|start")
            || l.starts_with("|t:|")
    };
    if is_req(expected) || is_req(got) {
        "request"
    } else if is_err(expected) || is_err(got) {
        "error"
    } else if is_hp(expected) || is_hp(got) {
        "privacy"
    } else if is_framing(expected) || is_framing(got) {
        "preamble"
    } else {
        "perside"
    }
}

/// Minimal JSON string escaper for the verdict lines.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ── A2 SEED-ANCHOR alignment integrity tests (gate-integrity is load-bearing) ──
//
// `anchor_seed_divergence` tolerates ONLY the port's extra ZERO-DRAW phaze-drag/forced-switch
// boundaries (a `got[i]` equal to `got[i-1]`). Any GENUINE draw-count desync changes a seed
// VALUE non-trivially, so it is NEVER a zero-draw duplicate → still reported as `kind:"seed"`.
#[cfg(test)]
mod a2_anchor_tests {
    use super::*;
    fn v(xs: &[&str]) -> Vec<String> { xs.iter().map(|s| s.to_string()).collect() }

    #[test]
    fn exact_match_is_clean() {
        let g = v(&["a", "b", "c"]);
        assert_eq!(anchor_seed_divergence(&g, &g), None);
    }

    #[test]
    fn extra_zero_draw_boundary_is_tolerated() {
        // The bab_0_16 shape: the port has an EXTRA boundary duplicating the prior seed (a
        // forced replacement that drew nothing) where the fuzzer collapses it.
        let got = v(&["s28", "s29", "s29", "s30", "s31"]); // extra "s29" (zero-draw)
        let exp = v(&["s28", "s29", "s30", "s31"]);
        assert_eq!(anchor_seed_divergence(&got, &exp), None);
    }

    #[test]
    fn a_genuine_value_shift_is_still_caught() {
        // A REAL extra draw changes the seed to a NEW value (NOT a zero-draw duplicate) → caught
        // at the sim decision index, reported for the omniscient fix-queue.
        let got = v(&["s28", "s29", "sX", "s31"]); // "sX" != "s29" and != prior → real desync
        let exp = v(&["s28", "s29", "s30", "s31"]);
        assert_eq!(
            anchor_seed_divergence(&got, &exp),
            Some((2usize, "s30".to_string(), "sX".to_string()))
        );
    }

    #[test]
    fn a_missing_draw_that_shifts_a_value_is_caught() {
        // The port MISSES a draw → its boundary seed is a value the sim doesn't have there, and
        // (having had OTHER draws) it is NOT equal to the prior boundary → not tolerated.
        let got = v(&["s28", "s29pre", "s30", "s31"]);
        let exp = v(&["s28", "s29", "s30", "s31"]);
        assert_eq!(
            anchor_seed_divergence(&got, &exp),
            Some((1usize, "s29".to_string(), "s29pre".to_string()))
        );
    }

    #[test]
    fn a_legit_shared_zero_draw_decision_aligns_normally() {
        // If BOTH the sim and port legitimately have a zero-draw decision (equal consecutive
        // seeds on BOTH sides), they align 1:1 with no skip.
        let g = v(&["a", "b", "b", "c"]);
        assert_eq!(anchor_seed_divergence(&g, &g), None);
    }
}

#[cfg(test)]
mod perside_construction_order_flip_tests {
    use super::*;
    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    // The bab_3_15 shape: the p2 window differs ONLY in the ORDER of a framing `-ability`
    // and a `-weather` line before `|turn|1`; every other line is byte-identical.
    fn golden() -> Vec<String> {
        v(&[
            "|player|p1|Alice||",
            "|switch|p1a: Tyranitar|Tyranitar, M|100/100",
            "|switch|p2a: Suicune|Suicune|100/100",
            "|-ability|p2a: Suicune|Pressure|[silent]",
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
        ])
    }
    fn engine_flipped() -> Vec<String> {
        v(&[
            "|player|p1|Alice||",
            "|switch|p1a: Tyranitar|Tyranitar, M|100/100",
            "|switch|p2a: Suicune|Suicune|100/100",
            // the TWO framing lines in the OPPOSITE order (a pure permutation):
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar",
            "|-ability|p2a: Suicune|Pressure|[silent]",
        ])
    }

    #[test]
    fn the_construction_order_flip_is_allowlisted_at_a_speed_tie() {
        assert_eq!(
            classify_perside_construction_order_flip(&golden(), &engine_flipped(), true),
            Some("turn0-construction-speed-tie-order-flip")
        );
    }

    #[test]
    fn a_distinct_speed_pair_fails_clause_1() {
        // Even the exact permutation must FAIL when the leads are NOT a construction tie.
        assert_eq!(
            classify_perside_construction_order_flip(&golden(), &engine_flipped(), false),
            None
        );
    }

    #[test]
    fn a_content_flipped_weather_of_fails_the_multiset() {
        // Mangle the weather `[of]` target → the multiset DIFFERS → NOT a pure permutation.
        let mut mangled = engine_flipped();
        mangled[3] = "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Suicune".to_string();
        assert_eq!(
            classify_perside_construction_order_flip(&golden(), &mangled, true),
            None
        );
    }

    #[test]
    fn a_changed_weather_id_fails_the_multiset() {
        let mut mangled = engine_flipped();
        mangled[3] = "|-weather|RainDance|[from] ability: Drizzle|[of] p1a: Tyranitar".to_string();
        assert_eq!(
            classify_perside_construction_order_flip(&golden(), &mangled, true),
            None
        );
    }

    #[test]
    fn a_changed_ability_fails_the_multiset() {
        let mut mangled = engine_flipped();
        mangled[4] = "|-ability|p2a: Suicune|Insomnia|[silent]".to_string();
        assert_eq!(
            classify_perside_construction_order_flip(&golden(), &mangled, true),
            None
        );
    }

    #[test]
    fn a_dropped_framing_line_fails_the_multiset() {
        let mut mangled = engine_flipped();
        mangled.remove(4);
        assert_eq!(
            classify_perside_construction_order_flip(&golden(), &mangled, true),
            None
        );
    }

    #[test]
    fn a_non_framing_line_moved_fails_clause_3() {
        // A real per-side ORDER bug reordering NON-framing (HP-bearing `|switch|`) lines:
        // the multiset stays identical, but the differing positions are NOT -ability/-weather.
        let g = v(&[
            "|switch|p2a: Suicune|Suicune|100/100",
            "|switch|p2a: Raikou|Raikou|100/100",
        ]);
        let e = v(&[
            "|switch|p2a: Raikou|Raikou|100/100",
            "|switch|p2a: Suicune|Suicune|100/100",
        ]);
        assert_eq!(classify_perside_construction_order_flip(&g, &e, true), None);
    }

    #[test]
    fn identical_windows_are_not_a_flip() {
        assert_eq!(
            classify_perside_construction_order_flip(&golden(), &golden(), true),
            None
        );
    }

    #[test]
    fn the_framing_window_stops_at_turn_one() {
        let stream = v(&[
            "|-ability|p2a: Suicune|Pressure|[silent]",
            "|turn|1",
            "|-ability|p2a: Suicune|Pressure|[silent]",
        ]);
        assert_eq!(framing_window(&stream).len(), 1);
    }
}

// ── The per-side REQUEST-residual allowlist reconcile (return102 + curse co-occurrence) ──
//
// `gen3_perside_request_byte_fuzz_v1` — the round-8 wide-sweep find: the sim renders the
// return102/frustration102 numeric-BP alias INCONSISTENTLY (roster `return102`, active id bare
// `return`, active display `Return 102`), so a co-occurring Curse+Return team broke the old
// single-direction reconcile → BOTH escaped. These pin the normalize-both-sides fix.
#[cfg(test)]
mod perside_request_residual_tests {
    use super::*;

    // A `|request|` whose ONLY diffs are the SIM-PROBED Curse `target:self`-vs-`normal` AND the
    // return102 alias (roster numeric / active bare id / active spaced display) must ALLOWLIST.
    #[test]
    fn curse_plus_return_cooccurrence_is_allowlisted() {
        // The sim (`expected`): Curse target:self, active `Return 102`/id `return`, roster `return102`.
        let expected = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Curse\",\"id\":\"curse\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},\
            {\"move\":\"Return 102\",\"id\":\"return\",\"pp\":32,\"maxpp\":32,\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"curse\",\"return102\"]}]}}";
        // The port (`got`): Curse target:normal, active `Return`/id `return`, roster `return`.
        let got = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Curse\",\"id\":\"curse\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false},\
            {\"move\":\"Return\",\"id\":\"return\",\"pp\":32,\"maxpp\":32,\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"curse\",\"return\"]}]}}";
        // Curse is the dominant reported reason (curse > return102).
        assert_eq!(
            classify_known_perside_residual(expected, got),
            Some("curse-nonghost-target-self-vs-normal")
        );
    }

    // Return alone (no curse) reports the return102 reason.
    #[test]
    fn return_alias_alone_is_allowlisted_as_return102() {
        let expected = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Return 102\",\"id\":\"return\",\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"return102\",\"earthquake\"]}]}}";
        let got = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Return\",\"id\":\"return\",\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"return\",\"earthquake\"]}]}}";
        assert_eq!(classify_known_perside_residual(expected, got), Some("return102-numeric-alias"));
    }

    // A GENUINE per-side bug (an extra un-cataloged field diff) must NOT be swallowed → None.
    #[test]
    fn a_genuine_request_diff_is_not_allowlisted() {
        let expected = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Return 102\",\"id\":\"return\",\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"return102\"]}]}}";
        // The port wrongly marks the move `disabled:true` (a real request-legality bug).
        let got = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Return\",\"id\":\"return\",\"target\":\"normal\",\"disabled\":true}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"return\"]}]}}";
        assert_eq!(classify_known_perside_residual(expected, got), None);
    }
}
