//! INDEPENDENT adversarial Lens-1 review of protocol findings F1/F2/F3.
//!
//! Drives the EMITTING engine (`run_full_battle_logged`) on the SAME constructed
//! scenarios captured from the resolved gen-3 sim by `harness/probe_lens1_review.js`
//! (DIFFERENT species + seeds than the builder's probes), and byte-asserts the port's
//! emitted lines carry the exact F1/F2/F3 forms the sim produced:
//!
//!   F1 — `|move|<user>|Leech Seed||[still]` + `|-fail|<user>` (sub-blocked Leech Seed)
//!   F2 — HIT-arm `|-start|<foe>|ability: Flash Fire` (no -immune);
//!        MISS-arm `|move|...|[miss]` + `|-miss|` (NOT -immune), Fire→FF and Water→WA
//!   F3 — `|-immune|<foe>|[from] ability: Water Absorb` / `Volt Absorb` (LANDED absorb)
//!
//! Each `seed` is the sim's POST-SWITCHIN `getSeed()` (== the protocol golden's
//! `initSeed`), captured by the probe — gen3 switch-ins ADVANCE the PRNG, so feeding
//! the port the raw `>start` seed would NOT be draw-aligned. The port asserts its own
//! post-switchin seed equals this (switch-ins draw-free), guaranteeing the turn-1
//! draws land at the identical PRNG position as the sim.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

fn dex() -> Dex {
    Dex::for_gen(3)
}

// The port's `Choice::Move`/`Switch` are 0-based; the sim (`>p1 move N`) is 1-based.
fn mv(n: usize) -> Choice {
    Choice::Move(n - 1)
}
fn sw(n: usize) -> Choice {
    Choice::Switch(n - 1)
}

fn play(p1: &str, p2: &str, seed: &str, choices: Vec<ScriptDecision>) -> Vec<String> {
    let d = dex();
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    };
    let mut battle = Battle::start_with_switchins(&opts, &d).expect("start");
    // The port's post-switchin seed must equal the sim's initSeed (switch-ins
    // draw-free) — the invariant the protocol gate pins; guarantees draw-alignment.
    assert_eq!(
        battle.state().unwrap().prng_seed(),
        seed,
        "port post-switchin seed must equal the sim's initSeed"
    );
    let (_o, emitted) = battle.state_mut().unwrap().run_full_battle_logged(&choices, &d);
    emitted.into_iter().map(|l| l.0).collect()
}

// ── Packed teams (verbatim from probe_lens1_review.js PACK_* lines) ──
const F1_P1: &str = "Meganium|||Overgrow|leechseed,bodyslam|Bold|252,,252,,,|N||||";
const F1_P2: &str = "Milotic||Leftovers|MarvelScale|substitute,surf|Bold|252,,252,,,|N||||";
const F2_P1: &str = "Charizard|||NoAbility|fireblast,aerialace|Modest|,,,252,,252|N||||]Vaporeon|||NoAbility|hydropump,icebeam|Modest|,,,252,,252|N||||";
const F2_P2: &str = "Ninetales|||FlashFire|flamethrower,quickattack|Timid|,,,252,,252|N||||]Lapras||Leftovers|WaterAbsorb|icebeam,bodyslam|Bold|252,,252,,,|N||||";
const F3_P1: &str = "Zapdos|||NoAbility|thunder,drillpeck|Modest|,,,252,,252|N||||";
const F3_P2: &str = "Lanturn||Leftovers|VoltAbsorb|surf,icebeam|Calm|252,,,,252,|N||||";

fn has(lines: &[String], needle: &str) -> bool {
    lines.iter().any(|l| l == needle)
}
fn rel<'a>(lines: &'a [String], subs: &[&str]) -> Vec<&'a String> {
    lines.iter().filter(|l| subs.iter().any(|s| l.contains(s))).collect()
}

/// F1 — a sub-blocked Leech Seed retro-edits to `[still]` + `-fail`, byte-identical
/// to the sim (initSeeds captured by probe_lens1_review.js).
#[test]
fn f1_leechseed_into_substitute_still_fail() {
    for seed in ["44317,42357,9927,48760", "62465,51971,16504,31324"] {
        let ch = vec![
            // Milotic subs (Meganium bodyslam chips), then Meganium Leech Seeds the sub.
            ScriptDecision { p1: Some(mv(2)), p2: Some(mv(1)) },
            ScriptDecision { p1: Some(mv(1)), p2: Some(mv(2)) },
        ];
        let lines = play(F1_P1, F1_P2, seed, ch);
        assert!(
            has(&lines, "|move|p1a: Meganium|Leech Seed||[still]"),
            "F1 {seed}: missing [still] retro-edit. lines: {:?}",
            rel(&lines, &["Leech Seed", "-fail", "-immune", "-miss"])
        );
        assert!(has(&lines, "|-fail|p1a: Meganium"), "F1 {seed}: missing -fail");
        assert!(
            !lines.iter().any(|l| l == "|-immune|p1a: Meganium" || l.starts_with("|-miss|p1a: Meganium")),
            "F1 {seed}: sub-block must not emit -immune/-miss"
        );
    }
}

/// F2 — Fire Blast into Flash Fire: the HIT arm arms FF (no -immune); the MISS arm
/// emits `[miss]`+`-miss` (NOT -immune).
#[test]
fn f2_fireblast_into_flashfire_hit_and_miss() {
    // HIT arm — sim armed FF at this initSeed.
    let hit = play(F2_P1, F2_P2, "4452,50520,38721,4268", vec![ScriptDecision { p1: Some(mv(1)), p2: Some(mv(2)) }]);
    assert!(
        has(&hit, "|-start|p2a: Ninetales|ability: Flash Fire"),
        "F2 HIT: FF should arm. lines: {:?}",
        rel(&hit, &["Fire Blast", "Flash Fire", "-immune", "-miss"])
    );
    assert!(!hit.iter().any(|l| l.contains("-immune")), "F2 HIT: must NOT emit -immune on the FF arm");

    // MISS arm — sim missed at this initSeed (`[miss]` + `-miss`, NOT -immune).
    let miss = play(F2_P1, F2_P2, "50988,44074,60332,47558", vec![ScriptDecision { p1: Some(mv(1)), p2: Some(mv(2)) }]);
    assert!(
        has(&miss, "|move|p1a: Charizard|Fire Blast|p2a: Ninetales|[miss]"),
        "F2 MISS: missing [miss] retro-edit. lines: {:?}",
        rel(&miss, &["Fire Blast", "-miss", "-immune"])
    );
    assert!(has(&miss, "|-miss|p1a: Charizard|p2a: Ninetales"), "F2 MISS: missing -miss");
    assert!(!miss.iter().any(|l| l.contains("-immune")), "F2 MISS: a missed Fire→FF must NOT emit -immune");
}

/// F2/F3 — Hydro Pump into Water Absorb: MISS arm `[miss]`+`-miss`; LANDED arm
/// `-immune|[from] ability: Water Absorb`.
#[test]
fn f2_hydropump_into_waterabsorb_miss_and_land() {
    let ch = || vec![
        ScriptDecision { p1: Some(sw(2)), p2: Some(sw(2)) }, // both bring in Vaporeon / Lapras
        ScriptDecision { p1: Some(mv(1)), p2: Some(mv(2)) }, // Vaporeon Hydro Pump into Lapras-WA
    ];
    // MISS — sim missed at this initSeed.
    let miss = play(F2_P1, F2_P2, "44317,42357,9927,48760", ch());
    assert!(
        has(&miss, "|move|p1a: Vaporeon|Hydro Pump|p2a: Lapras|[miss]"),
        "F2-WA MISS: missing [miss]. lines: {:?}",
        rel(&miss, &["Hydro Pump", "-miss", "-immune"])
    );
    assert!(has(&miss, "|-miss|p1a: Vaporeon|p2a: Lapras"), "F2-WA MISS: missing -miss");
    assert!(!miss.iter().any(|l| l.contains("-immune")), "F2-WA MISS: must NOT emit -immune");

    // LANDED — sim absorbed at this initSeed.
    let land = play(F2_P1, F2_P2, "62465,51971,16504,31324", ch());
    assert!(
        has(&land, "|-immune|p2a: Lapras|[from] ability: Water Absorb"),
        "F3/WA LANDED: missing -immune|[from] ability form. lines: {:?}",
        rel(&land, &["Hydro Pump", "-immune", "-miss"])
    );
    assert!(
        !land.iter().any(|l| l.starts_with("|-immune|p2a: Lapras") && !l.contains("[from] ability: Water Absorb")),
        "F3/WA LANDED: absorb -immune must carry the ability [from] tag"
    );
}

/// F3 — Thunder into Volt Absorb: LANDED arm `-immune|[from] ability: Volt Absorb`;
/// MISS arm `[miss]`+`-miss`.
#[test]
fn f3_thunder_into_voltabsorb_land_and_miss() {
    // LANDED — sim absorbed at this initSeed.
    let land = play(F3_P1, F3_P2, "62465,51971,16504,31324", vec![ScriptDecision { p1: Some(mv(1)), p2: Some(mv(1)) }]);
    assert!(
        has(&land, "|-immune|p2a: Lanturn|[from] ability: Volt Absorb"),
        "F3 LANDED: missing -immune|[from] ability: Volt Absorb. lines: {:?}",
        rel(&land, &["Thunder", "-immune", "-miss"])
    );

    // MISS — sim missed at this initSeed (`[miss]` + `-miss`, NOT immune).
    let miss = play(F3_P1, F3_P2, "4452,50520,38721,4268", vec![ScriptDecision { p1: Some(mv(1)), p2: Some(mv(1)) }]);
    assert!(
        has(&miss, "|move|p1a: Zapdos|Thunder|p2a: Lanturn|[miss]"),
        "F3 MISS: missing [miss]. lines: {:?}",
        rel(&miss, &["Thunder", "-miss", "-immune"])
    );
    assert!(has(&miss, "|-miss|p1a: Zapdos|p2a: Lanturn"), "F3 MISS: missing -miss");
    assert!(!miss.iter().any(|l| l.contains("-immune")), "F3 MISS: a missed Thunder→VA must NOT emit -immune");
}
