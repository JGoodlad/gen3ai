//! Protocol-emission (level-2, Phase 1) — the BYTE-differential gate.
//!
//! Replays the `protocol_capture_golden.txt` battles (seed + teams + recorded
//! choices) through the EMITTING engine (`run_full_battle_logged`) and asserts, per
//! battle, that the engine's emitted lines — FILTERED to the Phase-1 line types —
//! are BYTE-EQUAL, in order, to the golden's lines filtered the SAME way. This is a
//! real subset-equality: the DEFERRED (later-phase) line types are dropped from BOTH
//! sides before diffing, and `|t:|` is position-asserted-but-value-normalized (the
//! golden already stores `|t:|<NORMALIZED>`; the engine emits the same placeholder),
//! so a single wrong Phase-1 token / HP fraction / missing tag / mis-order fails.
//!
//! # Honesty (what this asserts vs defers)
//!
//! - **Phase-1 line types asserted** (see [`PHASE1_TYPES`]): the battle-init framing
//!   (`t:`/`gametype`/`player`/`gen`/`tier`/`rule`/`teamsize`/`start`/the blank `|`
//!   separator), `turn`, `upkeep`, `move` (+ `[miss]`/`[still]`), `switch`, `drag`,
//!   `-damage` (all HP variants + residual `[from]` tags), `-heal` (+ `[from] item:`),
//!   `faint`, `-crit`, `-supereffective`, `-resisted`, `-immune`, `-miss`, `win`/`tie`.
//! - **Deferred types filtered from BOTH sides** (later phases — the engine does NOT
//!   emit them yet, so they must not be in the golden's asserted stream either): the
//!   weather/boost/ability lines (`-weather`/`-boost`/`-unboost`/`-ability`), the
//!   status lines (`-status`/`-curestatus`/`cant`), the volatile/side-condition lines
//!   (`-start`/`-end`/`-activate`/`-singleturn`/`-sidestart`/`-fail`), and `debug`.
//! - **Deferred SCENARIOS** (see [`DEFERRED_SCENARIOS`]): a handful of scenarios whose
//!   Phase-1 STREAM is perturbed by a mechanic whose Phase-1-visible effect the engine
//!   doesn't emit yet (e.g. Substitute's self-move `|move|` + the sub-cost `|-damage|`,
//!   the Rest/Recover self-heal `|-heal|` with no `[from]`, the status-move `|move|`
//!   lines). These are documented + skipped ENTIRELY (not silently loosened) — they
//!   land when their emit hook does, in a later phase. The 4 CORE design scenarios
//!   (both_switch_distinct / post_faint_sweep_win / double_faint_replace /
//!   last_mon_double_ko_tie) + the pure-attacking sand scenario are asserted here.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};
use std::collections::{BTreeMap, BTreeSet};

fn dex() -> Dex {
    Dex::for_gen(3)
}

/// The line types this step (Phase 1 + Phase 2) emits + asserts. A protocol line
/// `|X|…` has "type" `X` (the token after the first pipe); the blank `|` separator has
/// type `""`. Everything NOT in this set is filtered from both sides before the diff.
///
/// Phase 2 ADDS the status / boost / weather / ability / volatile / side-condition line
/// types on top of the Phase-1 core (framing / turn / move / switch / faint / damage /
/// heal / crit / effectiveness / immune / miss / win / tie). The only types still
/// filtered out are `debug` (poke-env-ignored free-form sim text — a deliberate
/// non-emit, §d of the design) and the deferred mechanics' lines that never occur in
/// the ASSERTED scenarios.
fn phase2_types() -> BTreeSet<&'static str> {
    [
        // framing
        "t:", "gametype", "player", "gen", "tier", "rule", "teamsize", "start", "",
        // turn / phase
        "turn", "upkeep",
        // move / switch / faint / cant
        "move", "switch", "drag", "faint", "cant",
        // damage / heal
        "-damage", "-heal",
        // effectiveness / crit / miss / immune / fail
        "-crit", "-supereffective", "-resisted", "-immune", "-miss", "-fail",
        // status
        "-status", "-curestatus",
        // boost / weather / ability
        "-boost", "-unboost", "-weather", "-ability",
        // volatile / side-condition
        "-start", "-end", "-activate", "-singleturn", "-sidestart",
        // Phase 3: items / hints / no-op markers (gen3_protocol_phase3_v1)
        "-enditem", "-hint", "-nothing", "-fieldactivate",
        // end of battle
        "win", "tie",
    ]
    .into_iter()
    .collect()
}

/// Scenarios STILL deferred — NONE. The former blocker (a forced-replacement
/// REQUEST-BOUNDARY resume "phantom") is FIXED. Root cause (probe
/// `harness/probe_forced_replacement_resume.js` + `..._queue.js`): the OMNISCIENT
/// capture, submitting from a stale per-turn plan, recorded a PHANTOM zero-draw `move`
/// decision right after a mid-turn replacement changed the active mon to one with FEWER
/// moves (a 3-move Tyranitar → a 2-move Snorlax, then a scripted `move 3`). The sim's
/// `side.choose` REJECTS that out-of-range slot ("Your Snorlax doesn't have a move 3"),
/// draws NOTHING, and leaves `requestState === 'move'` OPEN — so the real turn runs on the
/// NEXT (valid) submission, and the phantom's `seedAfter` equals the prior forced-switch
/// decision's. The port's `run_full_battle` used to RUN a full turn for that decision
/// (its own move no-op'd but the FOE's Seismic Toss + residual + Quick Claw drew),
/// diverging the seed + the emitted line stream from the phantom onward. The FIX mirrors
/// the sim: `run_full_battle` now VALIDATES a top-of-turn `move` decision
/// (`move_decision_is_legal`) and SKIPS an out-of-range slot (run no turn, emit nothing,
/// draw nothing, record nothing) — re-pulling the next decision for the SAME boundary.
/// Zero-draw / observation-only (the total draw COUNT + outcome were already unaffected;
/// only the capture-vs-engine boundary MAPPING was off). Pinned by
/// `regression_test.rs::forced_replacement_resume_runs_the_post_replacement_move_decision`.
/// Both all-Seismic-Toss scenarios now replay byte-exact (the ST `|move|`/`|-damage|` lines
/// were already proven by `fixeddamage_test.rs` + the FD1-FD4 pins).
const DEFERRED_SCENARIOS: &[(&str, &str)] = &[];

fn is_deferred(scen: &str) -> Option<&'static str> {
    DEFERRED_SCENARIOS.iter().find(|(s, _)| *s == scen).map(|(_, why)| *why)
}

/// A per-BATTLE "unreplayable" guard: some INDIVIDUAL battles (even in an otherwise-
/// asserted scenario) hit an UNMODELED ENGINE MECHANIC mid-battle so the port can't
/// reproduce them bit-for-bit. Detected by scanning the golden's own `|move|` lines for
/// a move the engine provably can't model.
///
/// **NONE remain.** Struggle was the last blocker (a Choice-Band Tyranitar that runs out
/// of Rock-Slide PP in `recover_and_rest` battles 1/3/4 → the sim substitutes Struggle).
/// It is now MODELED bit-for-bit (`gen3_pp_tracking_v1`): per-move PP counters + the
/// forced-Struggle substitution + gen-3 Struggle (typeless '???' 50 BP, accuracy 100 →
/// draws accuracy, crit + damage like a normal move, + the gen-3 `recoil:[1,4]` =
/// `max(floor(dmg/4),1)` recoil, draw-free). So the 3 Struggle battles REPLAY byte-exact
/// and this guard catches nothing (kept as an empty-by-default scanner so a future
/// unmodeled move is caught + documented, not silently mis-compared). Returns the
/// blocking move name, or `None`. (Seismic Toss was the prior blocker, now the
/// fixed-damage layer replays it.)
fn unreplayable_move(_lines: &[String]) -> Option<&'static str> {
    None
}

/// The "type" of a raw protocol line (`|X|…` → `X`; the bare `|` → `""`; a line with
/// no leading pipe → the whole line, which never occurs in the omniscient stream).
fn line_type(raw: &str) -> &str {
    raw.strip_prefix('|').map_or(raw, |rest| rest.split('|').next().unwrap_or(""))
}

/// Normalize a `|t:|<unixtime>` line to the golden's placeholder (the golden already
/// stores `|t:|<NORMALIZED>`; the engine emits that placeholder too, but normalize
/// defensively so the POSITION is asserted, not the timestamp value).
fn normalize(raw: &str) -> String {
    if raw.starts_with("|t:|") {
        "|t:|<NORMALIZED>".to_string()
    } else {
        raw.to_string()
    }
}

/// Keep only the asserted (Phase 1 + Phase 2) line types, normalizing `|t:|`.
fn filter_phase2(lines: &[String], types: &BTreeSet<&'static str>) -> Vec<String> {
    lines
        .iter()
        .filter(|l| types.contains(line_type(l)))
        .map(|l| normalize(l))
        .collect()
}

// ── Golden parsing ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
struct ScenMeta {
    teams: [Option<String>; 2],
}

#[derive(Debug, Clone)]
struct BattleCase {
    scen: String,
    battle_no: u32,
    init_seed: String,
    choices: Vec<ScriptDecision>,
    lines: Vec<String>,
}

fn parse_choice(tok: &str) -> Option<Choice> {
    if tok == "-" {
        return None;
    }
    let (kind, num) = tok.split_at(1);
    let n: usize = num.parse().unwrap_or_else(|e| panic!("bad choice token {tok:?}: {e}"));
    match kind {
        "m" => Some(Choice::Move(n)),
        "s" => Some(Choice::Switch(n)),
        other => panic!("bad choice kind {other:?} in {tok:?}"),
    }
}

fn parse_golden() -> (BTreeMap<String, ScenMeta>, Vec<BattleCase>) {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/protocol_capture_golden.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing protocol capture golden ({path}): {e}\nrun: node src/rust_sim/harness/gen_protocol_capture.js")
    });

    let mut meta: BTreeMap<String, ScenMeta> = BTreeMap::new();
    let mut cases: Vec<BattleCase> = Vec::new();
    // key (scen, battle_no) -> index into `cases`
    let mut index: BTreeMap<(String, u32), usize> = BTreeMap::new();

    for (i, line) in data.lines().enumerate() {
        let ln = i + 1;
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // The L payload is the LAST field: everything after the 4th tab is the raw
        // line (which may contain tabs/pipes). splitn(5) keeps it intact.
        let f: Vec<&str> = line.splitn(5, '\t').collect();
        match f[0] {
            "SCEN" => {
                meta.entry(f[1].to_string()).or_default();
            }
            "TEAM" => {
                let side = if f[2] == "p1" { 0 } else { 1 };
                meta.entry(f[1].to_string()).or_default().teams[side] = Some(f[3].to_string());
            }
            "BATTLE" => { /* seed recorded; the INIT seed is what we replay from */ }
            "INIT" => {
                // INIT <scen> <battleNo> <initSeed>
                let scen = f[1].to_string();
                let battle_no: u32 = f[2].parse().unwrap_or_else(|e| panic!("bad battleNo (line {ln}): {e}"));
                let key = (scen.clone(), battle_no);
                index.insert(key.clone(), cases.len());
                cases.push(BattleCase {
                    scen,
                    battle_no,
                    init_seed: f[3].to_string(),
                    choices: Vec::new(),
                    lines: Vec::new(),
                });
            }
            "DEC" => {
                // DEC <scen> <battleNo> <decNo> <req> <fp1> <fp2> <cp1> <cp2> <seedAfter>
                // (a DEC line has no embedded tabs, so split the WHOLE line cleanly).
                let all: Vec<&str> = line.split('\t').collect();
                let scen = all[1].to_string();
                let battle_no: u32 = all[2].parse().unwrap();
                let (cp1, cp2) = (all[7], all[8]);
                let idx = index[&(scen, battle_no)];
                cases[idx].choices.push(ScriptDecision { p1: parse_choice(cp1), p2: parse_choice(cp2) });
            }
            "L" => {
                // L <scen> <battleNo> <lineNo> <raw...>
                let scen = f[1].to_string();
                let battle_no: u32 = f[2].parse().unwrap();
                let raw = f[4].to_string();
                let idx = index[&(scen, battle_no)];
                cases[idx].lines.push(raw);
            }
            "END" => { /* winner recorded elsewhere; the line stream carries |win|/|tie| */ }
            other => panic!("unknown record {other:?} (line {ln})"),
        }
    }
    (meta, cases)
}

fn opts_for(meta: &ScenMeta, init_seed: &str) -> BattleOptions {
    let t = &meta.teams;
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(init_seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(t[0].clone().expect("p1 team")) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(t[1].clone().expect("p2 team")) },
    }
}

#[test]
fn protocol_golden_matches_showdown() {
    let d = dex();
    let (meta, cases) = parse_golden();
    assert!(meta.len() >= 19, "expected >=19 captured scenarios (Phase 3), got {}", meta.len());
    assert!(cases.len() >= 100, "expected the per-seed capture corpus, got {} battles", cases.len());

    let types = phase2_types();

    let mut asserted_battles = 0usize;
    let mut asserted_lines = 0usize; // Phase-2-filtered lines asserted byte-equal
    let mut deferred_battles = 0usize;
    let mut unreplayable_battles = 0usize; // per-battle unmodeled-mechanic skips (now 0)
    let mut struggle_battles = 0usize; // battles that REPLAY a forced Struggle (>= 3 expected)
    let mut scen_asserted: BTreeSet<String> = BTreeSet::new();
    let mut scen_deferred: BTreeSet<String> = BTreeSet::new();

    for case in &cases {
        if let Some(_why) = is_deferred(&case.scen) {
            deferred_battles += 1;
            scen_deferred.insert(case.scen.clone());
            continue;
        }

        // Per-battle guard: skip an INDIVIDUAL battle that hits an unmodeled engine
        // mechanic so the port can't replay it — documented, honest, NOT a loosened
        // compare. NONE remain (Struggle is now modeled — `gen3_pp_tracking_v1`), so this
        // never fires; it stays as a future-proofing scanner.
        if let Some(_why) = unreplayable_move(&case.lines) {
            unreplayable_battles += 1;
            continue;
        }
        // Count battles that REPLAY a forced Struggle (proving the PP/Struggle layer is
        // exercised end-to-end vs the byte golden — the 3 recover_and_rest CB-Tyranitar
        // battles). A `|move|…|Struggle|…` line in the golden means the byte diff below
        // must reproduce it exactly.
        if case.lines.iter().any(|l| {
            l.strip_prefix("|move|")
                .and_then(|rest| rest.split('|').nth(1))
                .map(|name| name == "Struggle")
                .unwrap_or(false)
        }) {
            struggle_battles += 1;
        }

        let m = meta.get(&case.scen).unwrap_or_else(|| panic!("no meta for {}", case.scen));
        let opts = opts_for(m, &case.init_seed);
        let mut battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("[{}/{}] start failed: {e}", case.scen, case.battle_no));

        // The constructed prng must equal the sim's init seed (switch-ins draw-free) —
        // the same invariant the fullbattle gate pins; a mismatch here means the replay
        // diverged before the first decision.
        assert_eq!(
            battle.state().unwrap().prng_seed(),
            case.init_seed,
            "[{}/{}] init prng seed must equal the sim's (switch-ins draw-free)",
            case.scen, case.battle_no
        );

        let (_outcome, emitted) =
            battle.state_mut().unwrap().run_full_battle_logged(&case.choices, &d);
        let emitted_raw: Vec<String> = emitted.into_iter().map(|l| l.0).collect();

        let golden_filtered = filter_phase2(&case.lines, &types);
        let engine_filtered = filter_phase2(&emitted_raw, &types);

        // A TRUNCATED battle: the golden has NO terminal `|win|`/`|tie|` line — the
        // capture harness hit a decision/turn cap mid-battle (e.g. spikes_and_phaze/2
        // stalls in an infinite Spikes-at-cap ↔ immune-Earthquake loop). The engine plays
        // past the cap, so its output is LONGER. For a truncated battle we assert the
        // golden is a byte-exact PREFIX of the engine output (every recorded golden line
        // matches, in order) rather than full equality — honest, not a loosened compare.
        let truncated = case
            .lines
            .last()
            .map(|l| !(l.starts_with("|win|") || l.starts_with("|tie")))
            .unwrap_or(false);

        // First-divergence diff (the seed-test style): find the first line where the
        // filtered streams differ and panic with both sides + the index. For a truncated
        // battle we only compare up to the golden's length (the prefix).
        let n = if truncated {
            golden_filtered.len()
        } else {
            golden_filtered.len().max(engine_filtered.len())
        };
        for i in 0..n {
            let g = golden_filtered.get(i).map(String::as_str);
            let e = engine_filtered.get(i).map(String::as_str);
            if g != e {
                // Build a small context window for the panic message.
                let ctx = |v: &[String], lo: usize, hi: usize| -> String {
                    (lo..hi.min(v.len())).map(|k| format!("    [{k}] {}", v[k])).collect::<Vec<_>>().join("\n")
                };
                let lo = i.saturating_sub(3);
                panic!(
                    "[{}/{}] byte divergence at filtered line {i} (init_seed {}):\n  \
                     golden: {:?}\n  engine: {:?}\n\
                     --- golden context ---\n{}\n--- engine context ---\n{}\n\
                     FIX THE EMITTED BYTES — do not loosen the assert. If the divergence \
                     is an UNMODELED mechanic, defer the scenario (DEFERRED_SCENARIOS) or \
                     the individual battle (unreplayable_move) with a documented reason.",
                    case.scen, case.battle_no, case.init_seed, g, e,
                    ctx(&golden_filtered, lo, i + 4), ctx(&engine_filtered, lo, i + 4),
                );
            }
        }
        if truncated {
            // The engine must have AT LEAST the golden's prefix length (it plays past the
            // cap), and every golden line matched above.
            assert!(
                engine_filtered.len() >= golden_filtered.len(),
                "[{}/{}] truncated battle: engine ({}) shorter than the golden prefix ({})",
                case.scen, case.battle_no, engine_filtered.len(), golden_filtered.len()
            );
        } else {
            assert_eq!(
                golden_filtered.len(), engine_filtered.len(),
                "[{}/{}] filtered line COUNT mismatch: golden {} vs engine {}",
                case.scen, case.battle_no, golden_filtered.len(), engine_filtered.len()
            );
        }

        asserted_battles += 1;
        asserted_lines += golden_filtered.len();
        scen_asserted.insert(case.scen.clone());
    }

    assert!(asserted_battles >= 100, "expected >=100 asserted battles (Phase 3), got {asserted_battles}");
    assert!(asserted_lines >= 10000, "expected >=10000 lines asserted (Phase 3), got {asserted_lines}");
    // The 4 design-core scenarios MUST be asserted (never accidentally deferred).
    for core in ["both_switch_distinct", "post_faint_sweep_win", "double_faint_replace", "last_mon_double_ko_tie"] {
        assert!(scen_asserted.contains(core), "core scenario {core} must be asserted (Phase-1 target)");
    }
    // The Phase-2 scenarios un-deferred earlier MUST be asserted (the byte-exact
    // status/boost/weather/volatile/side-condition emission is the point).
    for p2 in ["substitute_absorb", "protect_block", "spikes_and_phaze", "recover_and_rest"] {
        assert!(scen_asserted.contains(p2), "Phase-2 scenario {p2} must be asserted");
    }
    // The two all-Seismic-Toss scenarios (formerly deferred for a forced-replacement
    // request-boundary resume phantom) are now ASSERTED byte-exact — the fix was the
    // reject-and-re-request gate in `run_full_battle` (`move_decision_is_legal`) + the
    // standalone-status-move already-statused `|-fail|` emission. Both MUST stay asserted
    // (never silently re-deferred). `DEFERRED_SCENARIOS` is now EMPTY.
    for st in ["status_para_and_boost_drop", "secondary_status_flinch"] {
        assert!(scen_asserted.contains(st), "un-deferred all-Seismic-Toss scenario {st} must be asserted");
    }
    // The Phase-3 scenarios (`gen3_protocol_phase3_v1`) MUST be asserted — they byte-gate
    // the formerly-deferred lines: the taunt/disable residual `-end`s + the Disable
    // retro-edit forms, the Trace `-ability` reveal, the Flash Fire -start/-immune/-end
    // cycle, the STATUS_IMMUNE `-immune [from] ability:` blocks, the Synchronize→Lum
    // `-status`/`-enditem`/`-curestatus` interleave (+ LumRest), the MID-BATTLE switch-in
    // ability lines (weather / Intimidate incl. the Clear-Body fail + Substitute -hint /
    // Pressure), and Leech Seed / Splash / Pay Day.
    for p3 in [
        "taunt_lifecycle", "disable_lifecycle", "trace_switchin", "flashfire_cycle",
        "status_immune_lines", "synchronize_lum_rest", "midswitch_ability_lines",
        "leechseed_splash_payday",
    ] {
        assert!(scen_asserted.contains(p3), "Phase-3 scenario {p3} must be asserted");
    }
    assert_eq!(deferred_battles, 0, "no scenario-level deferrals remain (DEFERRED_SCENARIOS is empty)");
    // The Struggle layer (`gen3_pp_tracking_v1`) un-skips the 3 recover_and_rest CB-Tyranitar
    // battles: they now REPLAY byte-exact. NO battle remains unreplayable.
    assert_eq!(
        unreplayable_battles, 0,
        "no per-battle unmodeled-mechanic skips remain (Struggle is modeled — gen3_pp_tracking_v1)"
    );
    assert!(
        struggle_battles >= 3,
        "expected >=3 forced-Struggle battles to REPLAY byte-exact (the recover_and_rest CB-Tyranitar \
         battles that run out of Rock-Slide PP), got {struggle_battles}"
    );

    eprintln!(
        "protocol golden (Phase 3): {} battles asserted ({} lines byte-equal) across {:?}; \
         {} battles deferred (scenario-level: {:?}); {} battles skipped (per-battle unmodeled mechanic); \
         {} battles REPLAYED a forced Struggle byte-exact (gen3_pp_tracking_v1)",
        asserted_battles, asserted_lines, scen_asserted, deferred_battles, scen_deferred, unreplayable_battles, struggle_battles
    );
}
