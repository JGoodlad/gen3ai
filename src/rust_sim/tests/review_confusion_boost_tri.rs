//! REVIEW differential — the three secondary-completion draw paths this step adds,
//! replayed against a LIVE-SIM-INSTRUMENTED golden so the running PRNG seed matches
//! Showdown EXACTLY for a single isolated turn:
//!
//!   - `waterpulse_confuse_land`  — Water Pulse lands a confusion: the secondary
//!     `random(100)` THEN, inside `addVolatile`'s onStart, `random(2,6)` (the duration);
//!   - `crunch_spd_drop`          — a stat-drop secondary draws its `random(100)` and
//!     applies −1 SpD DRAW-FREE (no extra draw);
//!   - `triattack_status`         — ONE `random(100)` then ONE `random(3)` (`sample`),
//!     NOT three `random(100)`s.
//!
//! Each CMP row was produced by `harness/trace_confusion_boost_tri.js`, which
//! monkey-patched the live sim PRNG, captured `(seedBefore, seedAfter)` for ONE turn
//! (p1 uses the move under test, p2 a passive never-miss Swift), and ALSO printed the
//! ordered draw trace (the human-readable proof in the harness output). This test
//! seeds a `BattleState` at `seedBefore`, runs `run_full_battle` for that one
//! move/move decision, and asserts the post-turn seed == `seedAfter` — an EXACT
//! cross-engine draw-ORDER+COUNT match (a missing `random(2,6)`, an extra
//! `random(100)`, or a 3-draw Tri Attack would diverge the seed here).
//!
//! The ALREADY-CONFUSED gate (Water Pulse on a pre-confused target draws the secondary
//! `random(100)` but NOT `random(2,6)`) needs a pre-injected volatile, so it is NOT
//! replayable through `run_full_battle`; the harness asserts it on the LIVE sim
//! (GATE: PASS) and the Rust unit test
//! `confusion_secondary_already_confused_skips_the_duration_draw` (in `turn.rs`)
//! proves the Rust side. The two together close that lens.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

#[derive(Debug)]
struct Cmp {
    case: String,
    seed_before: String,
    seed_after: String,
    p1_pack: String,
    p2_pack: String,
    /// The sim's post-turn boost stages [atk,def,spa,spd,spe,accuracy,evasion] for the
    /// user (p1) and the foe (p2) — the STATE a draw-free stat-boost apply must match.
    p1_boosts: [i8; 7],
    p2_boosts: [i8; 7],
}

fn parse_b7(s: &str, ln: usize) -> [i8; 7] {
    let v: Vec<i8> = s.split(',').map(|x| x.parse().unwrap_or_else(|e| panic!("bad boost {x:?} (line {ln}): {e}"))).collect();
    assert_eq!(v.len(), 7, "boost csv needs 7 ints (line {ln}): {s:?}");
    [v[0], v[1], v[2], v[3], v[4], v[5], v[6]]
}

fn parse_cmp() -> Vec<Cmp> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/vectors/confusion_boost_tri_cmp.txt");
    let data = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("missing CMP golden ({path}): {e}\nrun: node src/rust_sim/harness/trace_confusion_boost_tri.js")
    });
    let mut out = Vec::new();
    for (i, line) in data.lines().enumerate() {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(f.len(), 8, "CMP row needs 8 fields (line {}): {line:?}", i + 1);
        assert_eq!(f[0], "CMP", "expected CMP record (line {})", i + 1);
        out.push(Cmp {
            case: f[1].to_string(),
            seed_before: f[2].to_string(),
            seed_after: f[3].to_string(),
            p1_pack: f[4].to_string(),
            p2_pack: f[5].to_string(),
            p1_boosts: parse_b7(f[6], i + 1),
            p2_boosts: parse_b7(f[7], i + 1),
        });
    }
    out
}

#[test]
fn confusion_boost_tri_seed_parity() {
    let d = Dex::for_gen(3);
    let rows = parse_cmp();
    assert!(rows.len() >= 9, "expected >=9 CMP rows (3 cases x 3 seeds), got {}", rows.len());

    let mut by_case = std::collections::BTreeMap::new();
    for r in &rows {
        *by_case.entry(r.case.clone()).or_insert(0usize) += 1;
    }
    for c in [
        "waterpulse_confuse_land",
        "crunch_spd_drop",
        "ancientpower_self_all",
        "muddywater_acc_drop",
        "triattack_status",
        "tie_waterpulse_confuse",
    ] {
        assert!(by_case.get(c).copied().unwrap_or(0) >= 3, "case {c} needs >=3 rows");
    }

    for r in &rows {
        let opts = BattleOptions {
            format_id: "gen3customgame".to_string(),
            seed: Some(r.seed_before.clone()),
            p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(r.p1_pack.clone()) },
            p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(r.p2_pack.clone()) },
        };
        // start_with_switchins is draw-free for these NoAbility leads, so the prng
        // sits exactly at seed_before — matching where the live harness instrumented.
        let mut battle = Battle::start_with_switchins(&opts, &d)
            .unwrap_or_else(|e| panic!("[{}] start failed: {e}", r.case));
        assert_eq!(
            battle.state().unwrap().prng_seed(),
            r.seed_before,
            "[{}] init seed must equal seed_before (switch-ins draw-free)",
            r.case
        );

        // One decision: p1 uses move 1 (the move under test), p2 uses move 1 (Swift).
        let script = vec![ScriptDecision::both(Choice::Move(0), Choice::Move(0))];
        let outcome = battle.state_mut().unwrap().run_full_battle(&script, &d);

        assert_eq!(outcome.decisions.len(), 1, "[{}] expected one decision", r.case);
        let dec = &outcome.decisions[0];
        assert_eq!(
            &dec.seed_after, &r.seed_after,
            "[{}] SEED mismatch (before {}): rust {} vs sim {}\n  \
             a confusion random(2,6) / stat-drop apply / Tri Attack sample is \
             mis-ordered/missing/extra. FIX THE DRAW ORDER, do not loosen.",
            r.case, r.seed_before, dec.seed_after, r.seed_after
        );

        // BOOST STATE (draw-free): a multi-stat Ancient Power must raise ALL 5; Muddy
        // Water must drop the foe's ACCURACY (index 5) not a damage stat; a stat-drop
        // must hit the FOE and a self-raise the USER. The seed can't catch a wrong
        // stat/target (boost() draws nothing), so assert the stages directly.
        assert_eq!(
            &dec.active[0].boosts, &r.p1_boosts,
            "[{}] USER boost mismatch (before {}): rust {:?} vs sim {:?}\n  \
             a self stat-raise applied the wrong stat/stage (or to the foe).",
            r.case, r.seed_before, dec.active[0].boosts, r.p1_boosts
        );
        assert_eq!(
            &dec.active[1].boosts, &r.p2_boosts,
            "[{}] FOE boost mismatch (before {}): rust {:?} vs sim {:?}\n  \
             a foe stat-drop applied the wrong stat/stage (or to the user).",
            r.case, r.seed_before, dec.active[1].boosts, r.p2_boosts
        );
    }

    eprintln!(
        "confusion_boost_tri: {} CMP rows seed+boost parity OK across {} cases",
        rows.len(),
        by_case.len()
    );
}
