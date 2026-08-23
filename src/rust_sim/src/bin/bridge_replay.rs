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
    bridge_opts, parse_bridge_golden, run_full_battle_bridge_core_with_quick_claw,
    run_full_battle_bridge_with_quick_claw,
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
            run_full_battle_bridge_with_quick_claw(&opts, &b.cmds, b.quick_claw_roll, &dex)
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
/// The port's per-`|request|`-boundary seed list (`got`) is a SUPERSET of the omniscient
/// fuzzer's per-decision `rec.seeds` (`exp`): on a phaze-drag / on-entry-faint / SEQUENTIAL
/// double-forced-switch turn the port surfaces an EXTRA makeRequest boundary (checkpoint) that
/// the fuzzer's synchronous write-cascade merges into ONE recorded decision. Concretely, when
/// BOTH actives need a forced replacement the sim presents them one-side-at-a-time (p1 forced /
/// p2 wait, then p1 wait / p2 forced), but the fuzzer's `.write(p1 switch)` synchronously
/// surfaces p2's forced switch, so the SAME `requestState==='switch'` while-iteration writes
/// BOTH sides and pushes ONE `rec.seeds` (the seed AFTER both switch-ins). The port instead
/// checkpoints at each of the two boundaries, inserting one extra intermediate seed. That extra
/// may be a zero-draw duplicate (`got[i]==got[i-1]`, the switch-in that drew nothing) OR a
/// NON-zero intermediate value (draws occurred between the two forced switch-ins) — R20 proved
/// both forms occur (bab_10_4 carries both in one group). In EVERY case the game is
/// byte-identical to `|win|` (proven by the per-side byte diff without `--ab`), so it is a pure
/// decision-boundary BOOKKEEPING offset, NOT an engine draw bug.
///
/// This aligns `exp` as a SUBSEQUENCE of `got`: every sim decision-seed must appear, IN ORDER,
/// among the port's checkpoints (the port may interleave extra checkpoints between them). This
/// is STRUCTURALLY guaranteed for a byte-clean board — each sim decision-seed is captured at a
/// makeRequest pause, which is ALWAYS also a port checkpoint — so it NEVER false-positives on a
/// clean board. (The R20 fix generalizes the round-6 `got[i]==got[i-1]` zero-draw-only tolerance
/// to the full subsequence, which the Pressure/switch-in NON-zero-draw extra requires.)
///
/// SAFETY — it can NEVER mask a genuine draw-count desync: an extra/missing engine draw
/// PERMANENTLY shifts every downstream seed VALUE, so the sim's post-divergence seeds never
/// reappear in the port's stream → `exp` is NOT a subsequence of `got` → reported `kind:"seed"`
/// at the first unreconcilable sim decision. A coincidental subsequence match of a shifted
/// 64-bit seed is ~2^-64; and even in that impossible case, the SAME desync breaks the per-side
/// byte stream, which the downstream per-side byte diff catches as a non-allowlisted divergence
/// (the load-bearing discriminator: byte-equal ⇒ harmless checkpoint offset, byte-diverge ⇒ real
/// desync) — so a real desync FAILS the gate either way. Returns `(sim_decision_index, expected,
/// got_at_divergence)` at the first sim seed that cannot be reconciled, else `None`.
fn anchor_seed_divergence(got: &[String], exp: &[String]) -> Option<(usize, String, String)> {
    let mut i = 0usize; // index into the port's request_seeds (`got`, the SUPERSET)
    for (j, want) in exp.iter().enumerate() {
        // Advance past the port's extra checkpoint boundaries to find this sim decision-seed.
        let start = i;
        while i < got.len() && &got[i] != want {
            i += 1;
        }
        if i >= got.len() {
            // The sim seed was not found in the remaining port stream → a genuine draw-count
            // desync (a real divergence shifts the whole stream so the value never reappears).
            // Report the earliest port seed we could not reconcile to this sim decision.
            let got_at = got.get(start).cloned().unwrap_or_default();
            return Some((j, want.clone(), got_at));
        }
        i += 1; // consume the matched checkpoint
    }
    None
}

fn ab_verdict(b: &GoldenBattle, dex: &Dex) -> String {
    let opts = bridge_opts(&b.format_id, b.seed.clone(), &b.p1_team, &b.p2_team);
    // Build BOTH the per-side chunks AND the ScriptDecision list (the seed anchor
    // replays the latter through `run_full_battle`).
    let core = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        run_full_battle_bridge_core_with_quick_claw(&opts, &b.cmds, b.quick_claw_roll, dex)
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
        // B1 (pure framing PERMUTATION) first, then the A1-analog MIRROR IDENT FLIP (the
        // single-line `[of]` content flip AND the Intimidate `-ability`+`-unboost` permutation —
        // both the same harmless turn-0 construction speed-tie same-species mirror reorder).
        classify_perside_construction_order_flip(&gw, &ew, leads_speed_tie)
            .or_else(|| classify_perside_construction_mirror_flip(&gw, &ew, leads_speed_tie))
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
///     Curse's `nonGhostTarget`) vs `"target":"normal"` (port). **DORMANT/LEGACY as of
///     `gen3_bridge_curse_request_target_v1`:** the port's request serializer now emits the
///     runtime-effective `"self"` for a non-Ghost Curse, so this diff no longer occurs and this
///     reconciliation never fires. Retained defensively (a REVERT would still be classified, not
///     crash), but the byte-CLEAN corpus fixtures `10`/`15` are the real guard — they replay `ok`
///     and FAIL loudly (untagged → `ok`-required) if the serializer regresses.
///   - **return102-numeric-alias** — a `side.pokemon[].moves[]` entry `"return102"`↔`"return"`
///     (or `"frustration1"`↔`"frustration"`), the numeric-BP moveid alias. **DORMANT/LEGACY as
///     of `gen3_happiness_bp_request_alias_v1`, exactly like its Curse sibling above:** the
///     port's request serializer now emits all three of the sim's inconsistent alias forms
///     (roster `return102`, active id BARE `return`, active display `Return 102`), so this
///     reconciliation no longer fires on a correct port. Retained defensively — a REVERT is
///     classified rather than crashing, and the PAIRWISE bail below still refuses a genuine
///     BP-VALUE divergence (`return102` vs `return84`) — but the REAL guard is now the
///     byte-CLEAN, UNTAGGED corpus fixture `11_return102_numeric_alias_cg.txt`, which demands
///     `ok` and so fails loudly if the serializer regresses.
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
    // return<BP> / frustration<BP> numeric-BP moveid alias. THE SIM RENDERS IT INCONSISTENTLY
    // (SIM-PROBED): the `side.pokemon[].moves[]` ROSTER carries the numeric `return102`, but the
    // `active[].moves[].id` is the BARE `return` and the `.move` DISPLAY is `Return 102` (a SPACE) —
    // so a single-direction `replace("return","return102")` OVER-CORRECTS the active id AND misses
    // the display, and on a co-occurring Curse+Return team leaves a residual → the WHOLE reconcile
    // (incl. the correctly-fixed Curse target) returns None → BOTH escape. Instead NORMALIZE BOTH
    // sides by COLLAPSING every alias form to the bare token, then compare (poke-env resolves both —
    // the numeric BP is a display alias, `gen3_perside_request_byte_fuzz_v1`). Anchored on the
    // numeric/spaced tokens, so a non-alias move is never touched.
    {
        // ⚠️ The numeric suffix is the move's COMPUTED BASE POWER, so it VARIES
        // (`gen3_happiness_bp_alias_any_digits_v1`). Return's BP is `max(1, floor(happiness *
        // 2 / 5))` and Frustration's is the mirror `max(1, floor((255 - happiness) * 2 / 5))`,
        // so at the DEFAULT happiness 255 the pair renders `return102` and — because the raw
        // product is 0 and the sim clamps to 1 — **`frustration1`, never `frustration102`**.
        // This used to hardcode `102` on both, which meant the Frustration arm could not match
        // any real battle: every battle carrying a Frustration left a residual, the WHOLE
        // reconcile returned None, and the co-occurring (correctly handled) `return102` escaped
        // with it. Measured on a 25-battle `--mode random` bridge fuzz: 14 diverged / 5
        // allowlisted, where the 14 are exactly the Frustration-bearing boards.
        //
        // ⚠️ PAIRWISE, not a symmetric STRIP. Collapsing the digits on BOTH sides
        // unconditionally would reconcile a genuine VALUE divergence (sim `return102` vs port
        // `return84` — a mis-parsed happiness) into a FALSE PASS: exactly the vacuous-gate
        // failure the round-27 external-consistency work names ("an alias→canonical transform
        // is safe; a symmetric STRIP is not — if a presence-vs-absence residual appears, do it
        // PAIRWISE"). The DEFERRAL is presence-vs-ABSENCE of the numeric suffix, so the digits
        // themselves are compared first: if BOTH sides render a numeric BP for the same move
        // and the values DISAGREE, this is a real bug and the reconcile bails.
        let (cn, c_ret) = strip_bp_alias(&cur, "return", "Return");
        let (cn, c_fru) = strip_bp_alias(&cn, "frustration", "Frustration");
        let (en, e_ret) = strip_bp_alias(&exp, "return", "Return");
        let (en, e_fru) = strip_bp_alias(&en, "frustration", "Frustration");
        for (got_bps, exp_bps) in [(&c_ret, &e_ret), (&c_fru, &e_fru)] {
            let uniq = |v: &Vec<String>| {
                let mut u = v.clone();
                u.sort();
                u.dedup();
                u
            };
            let (g, e) = (uniq(got_bps), uniq(exp_bps));
            // Both sides numeric AND disagreeing = a BASE-POWER divergence, not a display form.
            if !g.is_empty() && !e.is_empty() && g != e {
                return None;
            }
        }
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

/// The A1-ANALOG per-side allowlist key `perside-construction-speed-tie-mirror-of-flip` — the
/// per-side sibling of `ab_replay.rs::classify_construction_mirror_of_flip` (the omniscient A1).
/// The unmodeled turn-0 construction speed-tie Fisher-Yates shuffle (the project-wide
/// seed-convention deferral) decides, on a SAME-SPECIES MIRROR lead, which of the two identical
/// mons' Sand-Stream / Intimidate resolves LAST — so a same-species mirror's framing `-weather`
/// `[of]` clause, or its `-ability` actor + `-unboost` target idents, flip between the two active
/// slots (`p1a` ↔ `p2a`). seed=None-INVISIBLE (`event::run_start_switchins` is deterministic +
/// draws nothing at a raw-Speed tie → correct production obs under `--use-bridge=rust`; the port
/// is the sole oracle at `seed=None`).
///
/// On the PER-SIDE stream this manifests in TWO structural forms — BOTH covered here (the R17
/// find: 2 Tyranitar-mirror + 2 Salamence-mirror repros, all the same harmless root):
///   (a) the SINGLE-LINE `[of]` CONTENT flip (Tyranitar mirror) — ONE `-weather` line whose
///       `[of]` slot flips (multiset DIFFERS → E1/B1's permutation check can't catch it, exactly
///       the A1-analog gap this key closes);
///   (b) the whole-Intimidate-block PERMUTATION (Salamence mirror) — the two `-ability` + two
///       `-unboost` lines reorder together (multiset SAME, but B1's clause-3 rejects the
///       `-unboost` lines, so B1 misses it).
/// Both reduce to the SAME per-field invariant: every framing line on which the two windows
/// differ becomes byte-identical once its flipping `pNa: <name>` idents are resolved to their
/// SPECIES (via the framing `|switch|` details) — i.e. the only thing that changed is WHICH
/// same-species mirror slot an ident points to. This mirrors A1's 6 structural clauses, GENERALIZING
/// A1's "differ in EXACTLY ONE line" to "every differing framing line is a same-species mirror ident
/// flip" (needed because the Intimidate case rides its paired `-unboost` line, so the per-side form
/// is a multi-line permutation rather than a single-line flip).
///
/// PREDICATE — all must hold, else `None` → the gate FAILS (a genuine per-side bug is NEVER swallowed):
///  (1) `leads_speed_tie` — the two construction-time lead Speeds are EQUAL (the caller's precondition);
///  (2) the two framing windows are EQUAL-LENGTH (a missing/extra line → `None`);
///  (3) EVERY position on which they differ is, in BOTH windows, a framing `|-weather|`/`|-ability|`/
///      `|-unboost|` line (a differing `|switch|`/HP/`|request|`/content line → `None`);
///  (4) each such differing line is a PURE MIRROR IDENT FLIP (`line_is_mirror_ident_flip`): after
///      the per-field diff, the ONLY differences are `pNa: <name>` / `[of] pNa: <name>` ident fields
///      whose SLOT flips between the two DIFFERENT active slots that map (via `|switch|` details) to
///      the SAME species; a same-slot name change (a nickname-render bug), a cross-species `[of]`
///      attribution, a changed weather/ability prefix, or a stat/value change breaks it → `None`;
///  (5) at least one line differs (identical windows are not a flip → the divergence is elsewhere);
///  (6) THE STRICTNESS SPLIT (the R18-review gate-integrity fix — clauses (3)/(4) alone would SWALLOW
///      a real mirror MIS-ATTRIBUTION: a same-species Salamence mirror where Intimidate mis-targets —
///      golden `-unboost|p2a: …`, engine `-unboost|p1a: …`, ONE line differs, a valid mirror ident
///      flip, BUT p1a is unboosted twice / p2a never → a REAL boost-state divergence). So a
///      NON-multiset-preserving flip is admitted ONLY as the A1-analog single `[of]`/actor case; the
///      multi-line form MUST be a pure permutation. Allowlist iff EITHER:
///        (form a) EXACTLY ONE differing line, and it is a `|-weather|`/`|-ability|` line (mirroring
///          omniscient A1's "differ in EXACTLY ONE line" — a `-weather` `[of]` / `-ability` actor
///          flip; a lone `-unboost` mis-target lands here and is REJECTED, it is neither); OR
///        (form b) the two framing windows are an IDENTICAL MULTISET (a pure PERMUTATION — the
///          Intimidate `-ability`+`-unboost` block reorder; a mis-target that breaks the multiset
///          returns `None` → the gate FAILS).
fn classify_perside_construction_mirror_flip(
    golden_window: &[String],
    engine_window: &[String],
    leads_speed_tie: bool,
) -> Option<&'static str> {
    // (1) construction speed-tie.
    if !leads_speed_tie {
        return None;
    }
    // (2) equal-length windows.
    if golden_window.len() != engine_window.len() || golden_window.is_empty() {
        return None;
    }
    let species = build_slot_species(&[golden_window, engine_window]);
    let is_framing = |l: &str| {
        l.starts_with("|-weather|") || l.starts_with("|-ability|") || l.starts_with("|-unboost|")
    };
    // Collect the differing positions; require EVERY one to be, in BOTH windows, a framing status
    // line (3) AND a pure same-species mirror ident flip (4).
    let mut diffs: Vec<usize> = Vec::new();
    for (i, (g, e)) in golden_window.iter().zip(engine_window.iter()).enumerate() {
        if g == e {
            continue;
        }
        // (3) both a framing status line.
        if !is_framing(g) || !is_framing(e) {
            return None;
        }
        // (4) a pure same-species mirror ident flip.
        match line_is_mirror_ident_flip(g, e, &species) {
            Some(true) => diffs.push(i),
            // g != e here, so `Some(false)` (identical) is unreachable — treat defensively.
            _ => return None,
        }
    }
    // (5) at least one flip.
    if diffs.is_empty() {
        return None;
    }
    // (6) THE STRICTNESS SPLIT (see the doc comment). Form (b): a pure permutation (identical
    // multiset) — the Intimidate block reorder; a mis-target that breaks the multiset declines here.
    let multiset_preserved = {
        let mut g = golden_window.to_vec();
        let mut e = engine_window.to_vec();
        g.sort();
        e.sort();
        g == e
    };
    if multiset_preserved {
        return Some("perside-construction-speed-tie-mirror-of-flip");
    }
    // Form (a): the A1-analog SINGLE differing `-weather`/`-ability` line (a `[of]`/actor flip). A
    // lone `-unboost` diff (the reviewer's mis-target — it breaks the multiset AND is not
    // weather/ability) is REJECTED here → None → the gate FAILS on a genuine boost mis-attribution.
    if diffs.len() == 1 {
        let g = golden_window[diffs[0]].as_str();
        let e = engine_window[diffs[0]].as_str();
        let is_wa = |l: &str| l.starts_with("|-weather|") || l.starts_with("|-ability|");
        if is_wa(g) && is_wa(e) {
            return Some("perside-construction-speed-tie-mirror-of-flip");
        }
    }
    None
}

/// Build a `slot → (name, species)` map (`p1a`/`p2a` → the lead's on-field ident NAME + species)
/// from the framing `|switch|` / `|drag|` lines across the given windows (the ident's `pNa: <name>`
/// token + the details' species field, the 2nd `|`-token up to the first comma). Used by
/// `line_is_mirror_ident_flip` to prove (a) each flipped ident's NAME is consistent with the roster
/// (a mislabeled `[of]` is a real bug) AND (b) the two flipped slots are the SAME species (the
/// sibling-mirror invariant, read past the nickname).
fn build_slot_species(
    windows: &[&[String]],
) -> std::collections::HashMap<String, (String, String)> {
    let mut map = std::collections::HashMap::new();
    for w in windows {
        for l in w.iter() {
            let rest = match l
                .strip_prefix("|switch|")
                .or_else(|| l.strip_prefix("|drag|"))
            {
                Some(r) => r,
                None => continue,
            };
            let mut it = rest.split('|');
            let ident = it.next().unwrap_or(""); // `pNa: <name>`
            let details = it.next().unwrap_or(""); // `<Species>, M`
            if let Some(colon) = ident.find(": ") {
                let slot = &ident[..colon];
                let name = &ident[colon + 2..];
                let species = details.split(',').next().unwrap_or(details).trim();
                if is_active_slot(slot) && !name.is_empty() && !species.is_empty() {
                    map.entry(slot.to_string())
                        .or_insert_with(|| (name.to_string(), species.to_string()));
                }
            }
        }
    }
    map
}

/// `true` iff `slot` is a singles ACTIVE slot ident prefix (`pNa`: `p`, a digit, then `a`).
fn is_active_slot(slot: &str) -> bool {
    let b = slot.as_bytes();
    b.len() == 3 && b[0] == b'p' && b[1].is_ascii_digit() && b[2] == b'a'
}

/// Parse a per-side line FIELD that is a mon IDENT reference — either `pNa: <name>` or
/// `[of] pNa: <name>`. Returns `(prefix, slot, name)` where `prefix` is `""` or `"[of] "`. `None`
/// if the field is not a well-formed active-slot ident field.
fn parse_ident_field(field: &str) -> Option<(&str, &str, &str)> {
    let (prefix, rest) = match field.strip_prefix("[of] ") {
        Some(r) => ("[of] ", r),
        None => ("", field),
    };
    let colon = rest.find(": ")?;
    let slot = &rest[..colon];
    if !is_active_slot(slot) {
        return None;
    }
    let name = &rest[colon + 2..];
    if name.is_empty() {
        return None;
    }
    Some((prefix, slot, name))
}

/// `Some(true)` iff `gl` and `el` differ ONLY by same-species `p1a`↔`p2a` mirror ident flips
/// (each differing `|`-field is a `pNa: <name>` / `[of] pNa: <name>` ident whose SLOT flips
/// between the two active slots that map to the SAME species); `Some(false)` if byte-identical;
/// `None` on ANY real difference — a non-ident field change (different weather/ability/stat/value),
/// a same-slot name change (a nickname-render bug), a cross-species flip, or a field-count mismatch.
fn line_is_mirror_ident_flip(
    gl: &str,
    el: &str,
    roster: &std::collections::HashMap<String, (String, String)>,
) -> Option<bool> {
    if gl == el {
        return Some(false);
    }
    let gf: Vec<&str> = gl.split('|').collect();
    let ef: Vec<&str> = el.split('|').collect();
    if gf.len() != ef.len() {
        return None;
    }
    let mut any_flip = false;
    for (a, b) in gf.iter().zip(ef.iter()) {
        if a == b {
            continue;
        }
        // The ONLY allowed field difference is a same-species active-slot ident flip.
        let (ap, aslot, aname) = parse_ident_field(a)?;
        let (bp, bslot, bname) = parse_ident_field(b)?;
        if ap != bp {
            return None; // one carries `[of] `, the other doesn't → a real diff.
        }
        if aslot == bslot {
            return None; // SAME slot, DIFFERENT name → a real nickname-render bug.
        }
        // Each ident's NAME must be consistent with its slot's roster mon (a mislabeled
        // `[of]`/actor — e.g. `[of] p2a: Gengar` where p2a is Tyranitar — is a REAL bug).
        let (a_rname, a_rsp) = roster.get(aslot)?;
        let (b_rname, b_rsp) = roster.get(bslot)?;
        if aname != a_rname || bname != b_rname {
            return None;
        }
        // Both slots must be the SAME species (the sibling-mirror invariant).
        if a_rsp != b_rsp {
            return None; // cross-species `[of]`/actor attribution → a real bug.
        }
        any_flip = true;
    }
    Some(any_flip)
}

/// Collapse a happiness-scaled move's numeric-BP alias to its bare form, for ANY base power
/// (`gen3_happiness_bp_alias_any_digits_v1`): `return102` → `return`, `Return 102` → `Return`,
/// `frustration1` → `frustration`, `Frustration 84` → `Frustration`.
///
/// `bare` is the lowercase move id (roster / active `id` form) and `display` its Title-Case
/// name, whose alias carries a SPACE before the number. Both are stripped only when followed by
/// ≥1 ASCII digits AND preceded by a NON-alphanumeric byte — in the request JSON that guard is
/// always a `"`, `,` or space, so a longer move whose name merely ENDS in these letters can
/// never be truncated, and a genuinely different move id is never touched. Digits are the ONLY
/// thing removed, so the gate stays non-vacuous: `return102` vs `tackle` still differs.
///
/// Returns the stripped string AND every removed digit run, in order — the caller compares
/// those PAIRWISE so a real base-power divergence cannot be stripped into a false pass.
fn strip_bp_alias(s: &str, bare: &str, display: &str) -> (String, Vec<String>) {
    let mut out = String::with_capacity(s.len());
    let mut bps: Vec<String> = Vec::new();
    let b = s.as_bytes();
    let mut i = 0usize;
    while i < b.len() {
        // The token must start at a boundary (start-of-string or a non-alphanumeric byte).
        let at_boundary = i == 0 || !b[i - 1].is_ascii_alphanumeric();
        let mut matched = false;
        if at_boundary {
            for (name, sep) in [(bare, ""), (display, " ")] {
                let head = format!("{name}{sep}");
                if s[i..].starts_with(&head) {
                    let after = i + head.len();
                    let digits = b[after..].iter().take_while(|c| c.is_ascii_digit()).count();
                    if digits > 0 {
                        out.push_str(name); // keep the name, DROP the separator+digits
                        bps.push(s[after..after + digits].to_string());
                        i = after + digits;
                        matched = true;
                        break;
                    }
                }
            }
        }
        if !matched {
            // Copy one whole UTF-8 char (the payload is JSON, but never assume ASCII).
            let ch = s[i..].chars().next().expect("in-bounds char");
            out.push(ch);
            i += ch.len_utf8();
        }
    }
    (out, bps)
}

/// Reconcile the Curse move-slot's `"target":"normal"` (port) toward `"target":"self"` (sim),
/// single-occurrence anchored at the curse id, IFF that leaves the strings closer without
/// disturbing any other slot. Returns the reconciled `got` when the curse slot's target was the
/// diverging field, else None.
///
/// ⚠️ SLOT-BOUNDED (`gen3_curse_reconcile_slot_bounded_v1`). The rewrite must stay inside the
/// curse's OWN `{...}` move object. It used to search from the curse id to the END of the line,
/// so once `gen3_bridge_curse_request_target_v1` made the port emit the correct `"target":"self"`
/// for a non-Ghost Curse, the search ran PAST the curse slot and rewrote the NEXT slot's
/// `"target":"normal"` — an unrelated Double Edge — MANUFACTURING a divergence on a line whose
/// curse was already byte-correct. That killed the whole reconcile (a co-occurring, correctly
/// handled `return102`/`frustration1` alias failed the gate with it): 4 of 400 `--mode random`
/// bridge battles. The doc line above ("without disturbing any other slot") was the intent; the
/// bound is what makes it true.
fn reconcile_curse_target(got: &str, expected: &str) -> Option<String> {
    if !expected.contains("\"id\":\"curse\"") || !got.contains("\"id\":\"curse\"") {
        return None;
    }
    let anchor = got.find("\"id\":\"curse\"")?;
    // The request's move slots are `{...},{...}` objects, so the next `}` closes THIS one.
    let slot_end = anchor + got[anchor..].find('}').unwrap_or(got.len() - anchor);
    let rel = got[anchor..slot_end].find("\"target\":\"normal\"").map(|r| anchor + r)?;
    // Only reconcile if the SIM's curse slot actually carries `target:self` — a GHOST holder's
    // Curse genuinely targets `normal` on BOTH sides, and rewriting that would invent a diff.
    let exp_anchor = expected.find("\"id\":\"curse\",\"pp\"")?;
    let exp_end = exp_anchor + expected[exp_anchor..].find('}').unwrap_or(expected.len() - exp_anchor);
    if !expected[exp_anchor..exp_end].contains("\"target\":\"self\"") {
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
// `anchor_seed_divergence` aligns `exp` (the sim's per-decision seeds) as a SUBSEQUENCE of `got`
// (the port's per-`|request|`-boundary seeds, a SUPERSET) — the port may interleave EXTRA
// checkpoint boundaries (zero-draw OR non-zero-draw, both proven by R20's Pressure/switch-in
// repros). Any GENUINE draw-count desync PERMANENTLY shifts every downstream seed VALUE, so the
// sim's post-divergence seeds never reappear in the port's stream → `exp` is NOT a subsequence
// → still reported `kind:"seed"`.
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

    #[test]
    fn extra_nonzero_draw_boundary_is_tolerated() {
        // R20 (bab_9_6 / bab_10_4): a SEQUENTIAL double-forced-switch — the port checkpoints the
        // intermediate p1-switched-in state (a NON-zero-draw value the sim never records, since
        // the fuzzer merges both switches into ONE decision). `got[i]` is NEITHER `exp[j]` NOR a
        // zero-draw duplicate of `got[i-1]`, yet the stream RE-SYNCS immediately after → tolerated
        // by the subsequence alignment (the round-6 zero-draw-only rule wrongly reported this).
        let got = v(&["s28", "s29", "sMID", "s30", "s31"]); // "sMID" = intermediate switch-in seed
        let exp = v(&["s28", "s29", "s30", "s31"]);
        assert_eq!(anchor_seed_divergence(&got, &exp), None);
    }

    #[test]
    fn a_group_with_both_a_zero_draw_and_a_nonzero_extra_is_tolerated() {
        // The bab_10_4 shape: one double-switch group inserts BOTH a zero-draw duplicate (the p1
        // switch-in that drew nothing) AND a non-zero intermediate (the p2 switch-in), i.e. TWO
        // extra port checkpoints between two sim-adjacent seeds. Both are skipped as long as the
        // sim seeds remain an ordered subsequence.
        let got = v(&["s21", "s21", "sMID", "s22", "s23"]); // "s21" dup (zero) + "sMID" (non-zero)
        let exp = v(&["s21", "s22", "s23"]);
        assert_eq!(anchor_seed_divergence(&got, &exp), None);
    }

    #[test]
    fn multiple_isolated_extra_groups_across_a_battle_are_tolerated() {
        // Several double-switch groups over one battle, each inserting an extra checkpoint,
        // separated by clean matching decisions — all tolerated (each is a local, re-syncing
        // superset insertion).
        let got = v(&["a", "aX", "b", "c", "cX", "d", "e"]);
        let exp = v(&["a", "b", "c", "d", "e"]);
        assert_eq!(anchor_seed_divergence(&got, &exp), None);
    }

    #[test]
    fn a_real_desync_that_shifts_the_whole_tail_is_still_caught() {
        // The load-bearing negative: a genuine extra draw at decision 2 shifts EVERY downstream
        // seed to a brand-new value — the sim's `s2..` never reappear in `got`, so the
        // subsequence FAILS at the first shifted decision → reported `kind:"seed"` (it does NOT
        // get swallowed as a checkpoint offset, even though the port merely has "different"
        // values from that point on). This is the class the R20 fix must NOT weaken.
        let got = v(&["s0", "s1", "s2b", "s3b", "s4b"]); // shifted from decision 2 onward
        let exp = v(&["s0", "s1", "s2", "s3", "s4"]);
        assert_eq!(
            anchor_seed_divergence(&got, &exp),
            Some((2usize, "s2".to_string(), "s2b".to_string()))
        );
    }

    #[test]
    fn a_desync_is_not_masked_by_a_coincidental_later_reappearance() {
        // Even if a shifted seed happens to reappear LATER out of order (astronomically unlikely
        // for a real 64-bit seed, but proven here), the subsequence still fails because the sim's
        // NEXT seed after the reappearance is absent — the gate stays red.
        let got = v(&["s0", "s1", "xx", "s2", "yy"]); // s2 reappears but s3/s4 are gone
        let exp = v(&["s0", "s1", "s2", "s3", "s4"]);
        assert!(anchor_seed_divergence(&got, &exp).is_some());
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

// ── A1-ANALOG (perside construction MIRROR IDENT FLIP) integrity tests ──
//
// `classify_perside_construction_mirror_flip` must fire ONLY on the same-species mirror
// construction reorder — the single-line `[of]`/actor content flip (Tyranitar-mirror weather) AND
// the Intimidate `-ability`+`-unboost` block permutation (Salamence-mirror) — and NEVER swallow a
// genuinely-wrong per-side attribution (the load-bearing requirement). These lock each clause;
// reverting a clause makes a NEG case start (wrongly) allowlisting → a test fails.
#[cfg(test)]
mod perside_construction_mirror_flip_tests {
    use super::*;
    fn v(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    // Form (a) — the canonical Tyranitar-mirror WEATHER `[of]` single-line CONTENT flip
    // (bab_7_13 / bab_3_9): the ONLY diff is the `-weather` `[of]` slot (p2a golden vs p1a engine),
    // both Tyranitar. The multiset DIFFERS, so B1 can't catch it — this is exactly the A1-analog gap.
    fn tyranitar_weather_windows() -> (Vec<String>, Vec<String>) {
        let sw1 = "|switch|p1a: Tyranitar|Tyranitar, M|391/391".to_string();
        let sw2 = "|switch|p2a: Tyranitar|Tyranitar, M|100/100".to_string();
        let g = vec![
            "|start".to_string(),
            sw1.clone(),
            sw2.clone(),
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Tyranitar".to_string(),
        ];
        let e = vec![
            "|start".to_string(),
            sw1,
            sw2,
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar".to_string(),
        ];
        (g, e)
    }

    // Form (b) — the Salamence-mirror INTIMIDATE block PERMUTATION (bab_6_14 / bab_7_4): the two
    // `-ability` + two `-unboost` lines reorder together (multiset SAME, but B1's clause-3 rejects
    // the `-unboost` lines). p1a carries the nickname "Drattak", p2a the bare species — so clause
    // (4) must read the species via the `|switch|` details, not the ident name.
    fn salamence_intimidate_windows() -> (Vec<String>, Vec<String>) {
        let sw1 = "|switch|p1a: Drattak|Salamence, M|331/331".to_string();
        let sw2 = "|switch|p2a: Salamence|Salamence, M|100/100".to_string();
        let g = vec![
            sw1.clone(),
            sw2.clone(),
            "|-ability|p2a: Salamence|Intimidate|boost".to_string(),
            "|-unboost|p1a: Drattak|atk|1".to_string(),
            "|-ability|p1a: Drattak|Intimidate|boost".to_string(),
            "|-unboost|p2a: Salamence|atk|1".to_string(),
        ];
        let e = vec![
            sw1,
            sw2,
            "|-ability|p1a: Drattak|Intimidate|boost".to_string(),
            "|-unboost|p2a: Salamence|atk|1".to_string(),
            "|-ability|p2a: Salamence|Intimidate|boost".to_string(),
            "|-unboost|p1a: Drattak|atk|1".to_string(),
        ];
        (g, e)
    }

    #[test]
    fn tyranitar_weather_of_flip_is_allowlisted_at_a_speed_tie() {
        let (g, e) = tyranitar_weather_windows();
        assert_eq!(
            classify_perside_construction_mirror_flip(&g, &e, true),
            Some("perside-construction-speed-tie-mirror-of-flip"),
            "the single-line weather-[of] mirror flip must allowlist at a speed tie"
        );
    }

    #[test]
    fn salamence_intimidate_permutation_is_allowlisted() {
        // The `-ability`+`-unboost` block permutation (with a nickname on one mirror slot).
        let (g, e) = salamence_intimidate_windows();
        assert_eq!(
            classify_perside_construction_mirror_flip(&g, &e, true),
            Some("perside-construction-speed-tie-mirror-of-flip")
        );
    }

    #[test]
    fn a_distinct_speed_pair_fails_clause_1() {
        // Clause (1): NOT a construction speed tie → never allowlisted, both forms.
        let (g, e) = tyranitar_weather_windows();
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, false), None);
        let (g2, e2) = salamence_intimidate_windows();
        assert_eq!(classify_perside_construction_mirror_flip(&g2, &e2, false), None);
    }

    #[test]
    fn a_wrong_of_to_a_real_different_species_mon_fails() {
        // The LOAD-BEARING NEG: a `[of]` attributed to a NON-sibling (DIFFERENT species) mon at a
        // mirror lead MUST NOT be swallowed. Leads are Tyranitar (p1a) vs Gengar (p2a).
        let g = vec![
            "|switch|p1a: Tyranitar|Tyranitar, M|391/391".to_string(),
            "|switch|p2a: Gengar|Gengar, M|281/281".to_string(),
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Gengar".to_string(),
        ];
        let e = vec![
            "|switch|p1a: Tyranitar|Tyranitar, M|391/391".to_string(),
            "|switch|p2a: Gengar|Gengar, M|281/281".to_string(),
            "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar".to_string(),
        ];
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    #[test]
    fn a_changed_weather_prefix_fails() {
        // Clause (4): the non-ident portion (weather id / `[from]`) changed → not a pure ident flip.
        let (g, mut e) = tyranitar_weather_windows();
        e[3] = "|-weather|RainDance|[from] ability: Sand Stream|[of] p1a: Tyranitar".to_string();
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    #[test]
    fn a_changed_ability_prefix_fails() {
        // Clause (4): a changed `-ability` NAME (Intimidate → Insomnia) is a real content diff.
        let (g, mut e) = salamence_intimidate_windows();
        e[2] = "|-ability|p1a: Drattak|Insomnia|boost".to_string();
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    #[test]
    fn a_missing_framing_line_fails() {
        // Clause (2): a dropped framing line → unequal length → None.
        let (g, mut e) = tyranitar_weather_windows();
        e.remove(3);
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    #[test]
    fn a_same_slot_nickname_change_is_a_real_bug_not_allowlisted() {
        // A same-SLOT ident whose NAME differs (a port nickname-render bug) is NOT a mirror slot
        // flip — clause (4) requires the slot to actually flip (p1a ↔ p2a).
        let g = vec![
            "|switch|p1a: Salamence|Salamence, M|331/331".to_string(),
            "|switch|p2a: Salamence|Salamence, M|100/100".to_string(),
            "|-ability|p1a: Salamence|Intimidate|boost".to_string(),
        ];
        let e = vec![
            "|switch|p1a: Salamence|Salamence, M|331/331".to_string(),
            "|switch|p2a: Salamence|Salamence, M|100/100".to_string(),
            "|-ability|p1a: Drattak|Intimidate|boost".to_string(), // wrong NAME, same slot
        ];
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    #[test]
    fn a_non_framing_line_diff_fails_clause_3() {
        // A differing NON-framing line (an HP-bearing `|-damage|` `[of]` flip) is out of scope for
        // this key — clause (3) requires every differing line to be `-weather`/`-ability`/`-unboost`.
        let g = vec![
            "|switch|p1a: Tyranitar|Tyranitar, M|391/391".to_string(),
            "|switch|p2a: Tyranitar|Tyranitar, M|100/100".to_string(),
            "|-damage|p1a: Tyranitar|360/391|[from] Sandstorm|[of] p2a: Tyranitar".to_string(),
        ];
        let mut e = g.clone();
        e[2] = "|-damage|p1a: Tyranitar|360/391|[from] Sandstorm|[of] p1a: Tyranitar".to_string();
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    #[test]
    fn a_mislabeled_of_name_inconsistent_with_the_roster_fails() {
        // The mangled-golden injection (cp-aside): the `[of]` ident NAME points at a mon that is
        // NOT the slot's roster mon (`[of] p2a: Gengar` where the p2a switch line says Tyranitar).
        // Reading species from the switch map alone would swallow it; the name-consistency check
        // catches it → None. (Reverting the name check makes this wrongly allowlist.)
        let (g, mut e) = tyranitar_weather_windows();
        let mut g = g;
        g[3] = "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Gengar".to_string();
        e[3] = "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p1a: Tyranitar".to_string();
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    // THE REVIEWER'S REQUIRED NEG (the R18 gate-integrity hole): a same-species Salamence MIRROR
    // where Intimidate MIS-TARGETS — golden `-unboost|p2a: Salamence|atk|1`, engine
    // `-unboost|p1a: Salamence|atk|1`. Only ONE line differs and it IS a same-species mirror ident
    // flip (so the loop's clause-4 passes), BUT the MULTISET is NOT preserved (p1a is unboosted
    // twice / p2a never → a REAL boost-state divergence, p1a -2 / p2a 0 vs the correct -1/-1). The
    // pre-fix predicate lacked the multiset guard, so it SWALLOWED this. The tightened predicate:
    // form (b) demands an identical multiset (broken here → declines); form (a) admits a lone diff
    // ONLY if it is a `-weather`/`-ability` line (this is `-unboost` → declines) → None → gate FAILS.
    #[test]
    fn a_single_line_unboost_mistarget_breaks_the_multiset_and_is_not_allowlisted() {
        let g = v(&[
            "|switch|p1a: Salamence|Salamence, M|331/331",
            "|switch|p2a: Salamence|Salamence, M|100/100",
            "|-ability|p1a: Salamence|Intimidate|boost",
            "|-unboost|p2a: Salamence|atk|1", // p1a's Intimidate drops the FOE p2a (correct)
            "|-ability|p2a: Salamence|Intimidate|boost",
            "|-unboost|p1a: Salamence|atk|1", // p2a's Intimidate drops the FOE p1a (correct)
        ]);
        let e = v(&[
            "|switch|p1a: Salamence|Salamence, M|331/331",
            "|switch|p2a: Salamence|Salamence, M|100/100",
            "|-ability|p1a: Salamence|Intimidate|boost",
            "|-unboost|p1a: Salamence|atk|1", // BUG: mis-targets p1a's OWN atk (should be p2a)
            "|-ability|p2a: Salamence|Intimidate|boost",
            "|-unboost|p1a: Salamence|atk|1",
        ]);
        // A single-line `-unboost` mirror flip that does NOT preserve the multiset is a REAL
        // per-side content bug — it must NOT be allowlisted (pre-fix this was wrongly Some(...)).
        assert_eq!(classify_perside_construction_mirror_flip(&g, &e, true), None);
    }

    #[test]
    fn identical_windows_are_not_a_flip() {
        let (g, _) = tyranitar_weather_windows();
        assert_eq!(classify_perside_construction_mirror_flip(&g, &g, true), None);
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

    // `gen3_happiness_bp_alias_any_digits_v1` — the numeric suffix is the move's COMPUTED base
    // power, so it is NOT always `102`. At the default happiness 255 Frustration's raw BP is 0
    // and the sim clamps to 1, rendering `frustration1` / `Frustration 1`. The pre-fix arm
    // hardcoded `102` on BOTH moves, so a Frustration board could never reconcile: it left a
    // residual, the WHOLE classify returned None, and a co-occurring (correctly handled)
    // `return102` escaped with it — 14 of 25 `--mode random` bridge battles.
    // REVERT-VERIFIED: restoring the hardcoded `.replace("frustration102", ...)` fails this.
    #[test]
    fn frustration_bp_alias_at_the_clamped_bp_of_one_is_allowlisted() {
        let expected = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Frustration 1\",\"id\":\"frustration\",\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"frustration1\",\"screech\"]}]}}";
        let got = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Frustration\",\"id\":\"frustration\",\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"frustration\",\"screech\"]}]}}";
        assert_eq!(
            classify_known_perside_residual(expected, got),
            Some("return102-numeric-alias")
        );
    }

    // GATE INTEGRITY (the load-bearing negative). The deferral is presence-vs-ABSENCE of the
    // numeric suffix; the digits are the move's BASE POWER. If BOTH sides render a number and
    // the values DISAGREE, that is a real divergence (a mis-parsed happiness) and collapsing
    // both to the bare token would be the "symmetric strip → false pass" failure round 27
    // names. Must NOT be allowlisted.
    #[test]
    fn a_differing_numeric_base_power_on_both_sides_is_not_allowlisted() {
        let expected = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Return 102\",\"id\":\"return\",\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"return102\"]}]}}";
        // The port computed a DIFFERENT base power (happiness parsed wrong) — a real bug.
        let got = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Return 84\",\"id\":\"return\",\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"return84\"]}]}}";
        assert_eq!(classify_known_perside_residual(expected, got), None);
    }

    // `gen3_curse_reconcile_slot_bounded_v1` — the bab_3_24 shape. The port's Curse slot is
    // ALREADY correct (`target:self`, per `gen3_bridge_curse_request_target_v1`), and the ONLY
    // real difference is the frustration alias. The unbounded search used to run past the curse
    // slot and rewrite the NEXT slot's `"target":"normal"` (Double Edge), inventing a residual
    // that failed the whole classify. REVERT-VERIFIED: dropping the `slot_end` bound fails this.
    #[test]
    fn a_correct_curse_slot_does_not_rewrite_a_later_move_target() {
        let expected = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Curse\",\"id\":\"curse\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},\
            {\"move\":\"Double-Edge\",\"id\":\"doubleedge\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"frustration1\",\"curse\",\"doubleedge\"]}]}}";
        let got = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Curse\",\"id\":\"curse\",\"pp\":16,\"maxpp\":16,\"target\":\"self\",\"disabled\":false},\
            {\"move\":\"Double-Edge\",\"id\":\"doubleedge\",\"pp\":24,\"maxpp\":24,\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"frustration\",\"curse\",\"doubleedge\"]}]}}";
        assert_eq!(
            classify_known_perside_residual(expected, got),
            Some("return102-numeric-alias")
        );
    }

    // GATE INTEGRITY: a GHOST holder's Curse genuinely targets `normal` on BOTH sides, so the
    // reconcile must not fire — and a real diff elsewhere must still be reported.
    #[test]
    fn a_ghost_curse_target_normal_is_not_rewritten() {
        let expected = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Curse\",\"id\":\"curse\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"curse\",\"shadowball\"]}]}}";
        // The port drops a bench move — a REAL bug that must not be masked by a curse rewrite.
        let got = "|request|{\"active\":[{\"moves\":[\
            {\"move\":\"Curse\",\"id\":\"curse\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}]}],\
            \"side\":{\"pokemon\":[{\"moves\":[\"curse\"]}]}}";
        assert_eq!(classify_known_perside_residual(expected, got), None);
    }

    // The stripper removes ONLY a digit run that follows the move token at a byte boundary, so
    // a different move is never truncated into a match and the reconcile stays non-vacuous.
    #[test]
    fn the_bp_alias_strip_only_touches_the_alias_token() {
        assert_eq!(
            strip_bp_alias("\"moves\":[\"return102\",\"returnx\",\"doubleedge\"]", "return", "Return"),
            (
                "\"moves\":[\"return\",\"returnx\",\"doubleedge\"]".to_string(),
                vec!["102".to_string()]
            )
        );
        // A move that merely ENDS in the token is not at a boundary → untouched, no digits.
        assert_eq!(
            strip_bp_alias("\"noreturn102\"", "return", "Return"),
            ("\"noreturn102\"".to_string(), Vec::<String>::new())
        );
        // A genuinely different move still differs after the strip (the gate stays live).
        let (a, _) = strip_bp_alias("\"return102\"", "return", "Return");
        let (b, _) = strip_bp_alias("\"tackle\"", "return", "Return");
        assert_ne!(a, b);
    }
}
