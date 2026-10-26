//! The RECHARGE family pins (`gen3_recharge_family_v1`, gen3 full parity B2): Blast Burn, Frenzy
//! Plant and Hydro Cannon run Hyper Beam's recharge. Each pin asserts the sim's observable (the
//! protocol lines, probe-settled in `harness/gen_recharge_family_golden.js`'s vector); the seed
//! and byte differential over every board is `recharge_family_golden_test.rs`.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision, RECHARGE_MOVES};

const SEED: &str = "17,29,43,59";

fn set(species: &str, ability: &str, moves: &str, lv: &str) -> String {
    format!("{species}|{species}|Leftovers|{ability}|{moves}|Hardy|85,85,85,85,85,85|M|||{lv}|")
}

fn play(p1: &str, p2: &str, script: &[ScriptDecision]) -> Vec<String> {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(SEED.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let (_o, lines) = b.state_mut().unwrap().run_full_battle_logged(script, &d);
    lines.into_iter().map(|l| l.0).collect()
}

fn both(a: usize, b: usize) -> ScriptDecision {
    ScriptDecision::both(Choice::Move(a), Choice::Move(b))
}

/// RF1 — the family is every gen-3 move whose RESOLVED data carries `self.volatileStatus:
/// 'mustrecharge'`, read off Showdown (fails loudly without node, like the handler audit).
#[test]
fn rf1_the_family_is_every_gen3_mustrecharge_move() {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(
            "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');\
             console.log(d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.self&&m.self.volatileStatus==='mustrecharge')\
             .map(m=>m.id).sort().join(','))",
        )
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let mut ours: Vec<&str> = RECHARGE_MOVES.to_vec();
    ours.sort_unstable();
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), ours.join(","));
}

/// RF2 — each sibling locks its user on a hit: `|-mustrecharge|` after the damage, then a
/// `|cant|…|recharge` turn (before B2 each one panicked at its first use).
#[test]
fn rf2_each_sibling_recharges_after_a_hit() {
    for mv in ["blastburn", "frenzyplant", "hydrocannon"] {
        let lines = play(
            &set("Snorlax", "ThickFat", &format!("{mv},splash"), ""),
            &set("Blissey", "NaturalCure", "splash", ""),
            &[both(0, 0), both(0, 0), both(0, 0), both(0, 0)],
        );
        let lock = lines.iter().position(|l| l == "|-mustrecharge|p1a: Snorlax").unwrap_or_else(|| panic!("{mv}: no lock"));
        let cant = lines.iter().position(|l| l == "|cant|p1a: Snorlax|recharge").unwrap_or_else(|| panic!("{mv}: no recharge turn"));
        assert!(lock < cant, "{mv}: the recharge turn follows the lock");
    }
}

/// RF3 — an ABSORBED hit locks nothing: Flash Fire takes Blast Burn, Water Absorb takes Hydro
/// Cannon (the probe: `-start|…|ability: Flash Fire`, then `-immune|…|[from] ability: Flash Fire`;
/// the user moves again the next turn).
#[test]
fn rf3_an_absorbed_hit_locks_nothing() {
    for (mv, foe, ability, tag) in [
        ("blastburn", "Houndoom", "FlashFire", "Flash Fire"),
        ("hydrocannon", "Lapras", "WaterAbsorb", "Water Absorb"),
    ] {
        let lines = play(
            &set("Snorlax", "ThickFat", &format!("{mv},splash"), ""),
            &set(foe, ability, "splash", ""),
            &[both(0, 0), both(0, 0)],
        );
        assert!(lines.iter().any(|l| l.contains(tag)), "{mv}: the absorb");
        assert!(!lines.iter().any(|l| l.starts_with("|-mustrecharge|")), "{mv}: an absorbed hit locked the user");
        assert_eq!(lines.iter().filter(|l| l.starts_with("|move|p1a: Snorlax|")).count(), 2, "{mv}: the user acts both turns");
    }
}

/// RF4 — a KO keeps the lock across the foe's forced switch (`-mustrecharge` before `faint`).
#[test]
fn rf4_a_ko_keeps_the_lock_across_the_forced_switch() {
    let lines = play(
        &set("Snorlax", "ThickFat", "frenzyplant,splash", ""),
        &[set("Alakazam", "Synchronize", "splash", "30"), set("Jolteon", "VoltAbsorb", "splash", "30")].join("]"),
        &[both(0, 0), ScriptDecision { p1: None, p2: Some(Choice::Switch(1)) }, both(0, 0)],
    );
    let i = |l: &str| lines.iter().position(|x| x == l).unwrap_or_else(|| panic!("no {l:?} in {lines:?}"));
    assert!(i("|-mustrecharge|p1a: Snorlax") < i("|faint|p2a: Alakazam"));
    assert!(i("|faint|p2a: Alakazam") < i("|cant|p1a: Snorlax|recharge"));
}
