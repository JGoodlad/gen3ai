//! Byte-emission pins for the OMNISCIENT BYTE differential fuzzer's first findings
//! (`gen3_omniscient_byte_fuzz_v1`). Each is a CONSTRUCTED gen3customgame battle replayed
//! through the EMITTING engine (`run_full_battle_logged`), asserting the EXACT sim byte
//! form of a `|...|` line the byte fuzzer surfaced as a divergence — and (where relevant)
//! that the WRONG (pre-fix) form is ABSENT, so a revert flips the assertion.
//!
//! The fuzzer (`harness/ab_fuzz.js --protocol` + `src/bin/ab_replay.rs --protocol`) captures
//! the REAL omniscient log per battle and byte-diffs it against the port's logged output; it
//! reported these as `kind=protocol`. Positive line-presence assertions (not full byte-equality)
//! so an UNFIXED emission-form gap elsewhere in the same battle can't perturb the pin.

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

fn dex() -> Dex {
    Dex::for_gen(3)
}

fn opts_cg(p1: &str, p2: &str, seed: &str) -> BattleOptions {
    BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some(seed.to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    }
}

fn run_logged(p1: &str, p2: &str, seed: &str, script: &[ScriptDecision]) -> Vec<String> {
    let d = dex();
    let mut battle = Battle::start_with_switchins(&opts_cg(p1, p2, seed), &d).expect("start");
    let (_out, lines) = battle.state_mut().expect("state").run_full_battle_logged(script, &d);
    lines.into_iter().map(|l| l.0).collect()
}

/// BF1 — typed HIDDEN POWER canonicalizes to the bare `Hidden Power` in the `|move|` line
/// (gen-3 hides the HP TYPE). WRONG (pre-fix, the moment HP became pickable): the port
/// resolved the packed team's TYPED variant (`hiddenpowerice`, num 365) and rendered its
/// dex display name `Hidden Power Ice` — leaking the type.
#[test]
fn typed_hidden_power_renders_bare_name_in_the_move_line() {
    // p1 Zapdos uses Hidden Power [Ice] at Swampert; p2 attacks. HP is modeled at fixed BP
    // 70 (byte-safe for the standard 70-BP IVs); only the DISPLAY collapses here.
    let zapdos = "Zapdos|zapdos|leftovers|pressure|hiddenpowerice,thunderbolt|Timid|,,,,,252|M||||";
    let swampert = "Swampert|swampert|leftovers|torrent|earthquake,surf|Adamant|252,252,,,,|M||||";
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))];
    let raw = run_logged(zapdos, swampert, "40263,34842,41812,24710", &script);

    // The HP `|move|` announce reads the bare name.
    assert!(
        raw.iter().any(|l| l.starts_with("|move|p1a: Zapdos|Hidden Power|")),
        "the typed-HP `|move|` line must read the bare `Hidden Power`; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — the typed name must NEVER appear (reverting the canonicalization emits it).
    for l in &raw {
        assert!(
            !l.contains("Hidden Power Ice"),
            "no line may leak the HP type (`Hidden Power Ice`) — gen-3 hides it. Offending: {l:?}"
        );
    }
}

/// BF2 — the TOXIC residual chip is reported `[from] psn` (NOT `[from] tox`); the HP-field
/// status token stays `tox`. WRONG (pre-fix): the port emitted `[from] tox`. A latent byte
/// gap the constructed protocol golden never realized (no Toxic residual in its scenarios).
#[test]
fn toxic_residual_damage_reports_from_psn() {
    // p1 Gengar Toxics p2 Snorlax (Normal → not immune, not Poison/Steel; Thick Fat, NOT
    // Immunity, so the Toxic lands), then chips it; the end-of-turn Toxic DoT fires.
    let gengar = "Gengar|gengar|leftovers|levitate|toxic,thunderbolt|Timid|,,,,,252|M||||";
    let snorlax = "Snorlax|snorlax|leftovers|thickfat|splash,bodyslam|Careful|252,,,,252,|M||||";
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // Toxic lands; Snorlax Splash
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // chip; tox ramps
    ];
    let raw = run_logged(gengar, snorlax, "40263,34842,41812,24710", &script);

    // The residual chip line carries `tox` in the HP field but `[from] psn` as the cause.
    assert!(
        raw.iter().any(|l| l.starts_with("|-damage|p2a: Snorlax|")
            && l.contains(" tox|[from] psn")),
        "the Toxic residual must be `|-damage|p2a: Snorlax|<hp> tox|[from] psn`; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — the WRONG `tox|[from] tox` cause must NEVER appear.
    for l in &raw {
        assert!(
            !l.contains("|[from] tox"),
            "the Toxic residual `[from]` cause must be `psn`, never `tox`. Offending: {l:?}"
        );
    }
}

/// BF3 — a NON-foe-directed status move (`allySide` Light Screen / `all` weather / `allyTeam`)
/// renders the USER as the `|move|` TARGET field, not the FOE. WRONG (pre-fix): the port
/// rendered the FOE for anything that wasn't literally `target:self` (Light Screen is
/// `allySide`), so `|move|p1a: Starmie|Light Screen|p2a: <foe>`.
#[test]
fn ally_side_move_renders_the_user_as_the_move_target() {
    let starmie = "Starmie|starmie|leftovers|naturalcure|lightscreen,surf|Timid|,,,,,252|M||||";
    let snorlax = "Snorlax|snorlax|leftovers|immunity|splash,bodyslam|Careful|252,,,,252,|M||||";
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))]; // Light Screen up
    let raw = run_logged(starmie, snorlax, "40263,34842,41812,24710", &script);

    assert!(
        raw.iter().any(|l| l == "|move|p1a: Starmie|Light Screen|p1a: Starmie"),
        "an allySide move's `|move|` target field must be the USER; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — the foe must NOT be the target of our own Light Screen.
    for l in &raw {
        assert!(
            !(l.starts_with("|move|p1a: Starmie|Light Screen|") && l.contains("p2a:")),
            "our Light Screen must render the USER, never the foe. Offending: {l:?}"
        );
    }
}

/// BF4 — the PURSUIT INTERRUPT strike's `|move|` announce carries the `|[from] Pursuit`
/// tag (`gen3_move_coverage_batch4_v1` emission form). WRONG (pre-fix): the port omitted it,
/// emitting a bare `|move|<pursuer>|Pursuit|<switcher>`.
#[test]
fn pursuit_interrupt_move_announce_carries_from_pursuit() {
    // p1 Tyranitar (fast) uses Pursuit while p2 voluntarily switches its lead Swampert → the
    // interrupt fires (the pursuer strikes the switching mon), tagged `[from] Pursuit`.
    let ttar = "Tyranitar|tyranitar|leftovers|sandstream|pursuit,rockslide|Adamant|,252,,,,252|M||||";
    let p2 = "Swampert|swampert|leftovers|torrent|surf,earthquake|Bold|252,,252,,,|M||||]\
              Skarmory|skarmory|leftovers|keeneye|drillpeck,spikes|Impish|252,,252,,,|M||||";
    let script = [ScriptDecision { p1: Some(Choice::Move(0)), p2: Some(Choice::Switch(1)) }];
    let raw = run_logged(ttar, p2, "40263,34842,41812,24710", &script);

    assert!(
        raw.iter().any(|l| l.starts_with("|move|p1a: Tyranitar|Pursuit|")
            && l.ends_with("|[from] Pursuit")),
        "the Pursuit interrupt `|move|` line must end with `|[from] Pursuit`; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — a bare Pursuit announce (no `[from] Pursuit`) must NOT be the interrupt form.
    // (Guard against a vacuous pass: the interrupt DID fire — the `-activate move: Pursuit`.)
    assert!(
        raw.iter().any(|l| l.contains("|-activate|p2a: Swampert|move: Pursuit")),
        "the Pursuit interrupt must fire (the `-activate move: Pursuit`); lines:\n{}",
        raw.join("\n")
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// gen3_omniscient_byte_fuzz_v1 — the STATUS-MOVE / did-nothing EMISSION FORMS the
// byte fuzzer surfaced (pool clean rate ~27% → ~95%). Each pin asserts the EXACT
// fixed line is PRESENT and (where the pre-fix form was distinct) that the WRONG
// form is ABSENT — so a revert of the fix flips the assertion. All emission-only:
// the full seed suite stays byte-identical (proven by `cargo test`).
// ─────────────────────────────────────────────────────────────────────────────

const SD: &str = "40263,34842,41812,24710";

fn line_present(raw: &[String], exact: &str) -> bool {
    raw.iter().any(|l| l == exact)
}

/// BF-F5 — a SLEEP move carries `|-status|<t>|slp|[from] move: <Move>` (the sim's
/// `slp.onStart` `sourceEffect.effectType === 'Move'` branch). WRONG (pre-fix): a bare
/// `|-status|<t>|slp`. par/psn/brn stay bare (only sleep carries it).
#[test]
fn sleep_move_status_carries_from_move() {
    let breloom = "Breloom|breloom|leftovers|effectspore|spore,machpunch|Adamant|252,252,,,,|M||||";
    let snorlax = "Snorlax|snorlax|leftovers|immunity|bodyslam,rest|Careful|252,,,,252,|M||||";
    let raw = run_logged(breloom, snorlax, SD, &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))]);
    assert!(
        line_present(&raw, "|-status|p2a: Snorlax|slp|[from] move: Spore"),
        "sleep from a move must carry `[from] move: Spore`; lines:\n{}",
        raw.join("\n")
    );
    assert!(
        !line_present(&raw, "|-status|p2a: Snorlax|slp"),
        "the bare sleep form must NOT appear (a revert emits it)"
    );
}

/// BF-F3 — a status MOVE type-immune to the foe (Toxic → a Steel/Poison mon via the
/// psn `runStatusImmunity`) emits `|-immune|<target>` (only when the source is a status
/// MOVE — `sourceEffect?.status`). WRONG (pre-fix): NOTHING (the gate returned silently).
#[test]
fn toxic_into_steel_reports_immune() {
    let gengar = "Gengar|gengar|leftovers|levitate|toxic,thunderbolt|Timid|,,,,,252|M||||";
    let skarmory = "Skarmory|skarmory|leftovers|keeneye|drillpeck,spikes|Impish|252,,252,,,|M||||";
    // Sweep a few seeds so Toxic's 85 accuracy roll lands (the `-immune` is post-accuracy).
    let seeds = ["40263,34842,41812,24710", "1,2,3,4", "9,9,9,9", "500,500,500,500"];
    let mut hit = false;
    for s in seeds {
        let raw = run_logged(gengar, skarmory, s, &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))]);
        // Only assert on a seed where Toxic did NOT miss (no `[miss]` on the move line).
        if raw.iter().any(|l| l.starts_with("|move|p1a: Gengar|Toxic") && l.ends_with("[miss]")) {
            continue;
        }
        assert!(
            line_present(&raw, "|-immune|p2a: Skarmory"),
            "Toxic into a Steel mon must report `-immune` (seed {s}); lines:\n{}",
            raw.join("\n")
        );
        hit = true;
        break;
    }
    assert!(hit, "no seed landed Toxic on Skarmory to exercise the `-immune` line");
}

/// BF-F4 — a NATURAL CURE mon's status is CURED on switch-out with a
/// `|-curestatus|<mon>|<status>|[from] ability: Natural Cure|[silent]` line, emitted
/// BEFORE the replacement `|switch|`. Uses Rest (never-miss self-sleep) so it's
/// seed-robust. WRONG (pre-fix): the cure was silent (no line).
#[test]
fn natural_cure_emits_curestatus_on_switch_out() {
    let starmie = "Starmie|starmie|leftovers|naturalcure|rest,surf|Timid|,,,,,252|M||||]\
                   Snorlax|snorlax|leftovers|immunity|bodyslam,splash|Careful|252,,,,252,|M||||";
    let blissey = "Blissey|blissey|leftovers|naturalcure|softboiled,seismictoss|Calm|252,,,,252,|F||||";
    let raw = run_logged(
        starmie,
        blissey,
        SD,
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(1)),   // Blissey Seismic Toss → Starmie < full
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // Starmie Rest → slp (heals to full)
            ScriptDecision::both(Choice::Switch(1), Choice::Move(0)), // Starmie pivots → cured
        ],
    );
    assert!(
        line_present(&raw, "|-curestatus|p1a: Starmie|slp|[from] ability: Natural Cure|[silent]"),
        "Natural Cure must emit the `-curestatus … [from] ability: Natural Cure|[silent]`; lines:\n{}",
        raw.join("\n")
    );
    // The cure line precedes the entrant's switch line.
    let cure = raw.iter().position(|l| l.contains("Natural Cure|[silent]"));
    let sw = raw.iter().position(|l| l.starts_with("|switch|p1a: Snorlax|"));
    assert!(
        matches!((cure, sw), (Some(c), Some(s)) if c < s),
        "the `-curestatus` must precede the replacement `|switch|`; lines:\n{}",
        raw.join("\n")
    );
}

/// BF-F1a — RECOVER at full HP emits the did-nothing `[still]` announce form + a
/// `|-fail|<user>|heal` (the `heal` sub-tag). WRONG (pre-fix): the normal target form +
/// a bare `-fail` (no `[still]`, no sub-tag).
#[test]
fn recover_at_full_hp_emits_still_and_fail_heal() {
    let milotic = "Milotic|milotic|leftovers|marvelscale|recover,surf|Bold|252,,252,,,|F||||";
    let snorlax = "Snorlax|snorlax|leftovers|immunity|splash,bodyslam|Careful|252,,,,252,|M||||";
    // p1 Milotic (full HP) Recover; p2 Splash (harmless) — Milotic stays full → the heal fails.
    let raw = run_logged(milotic, snorlax, SD, &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))]);
    assert!(
        line_present(&raw, "|move|p1a: Milotic|Recover||[still]"),
        "Recover@full must render the `[still]` announce; lines:\n{}",
        raw.join("\n")
    );
    assert!(
        line_present(&raw, "|-fail|p1a: Milotic|heal"),
        "Recover@full must emit `|-fail|…|heal`; lines:\n{}",
        raw.join("\n")
    );
    assert!(
        !line_present(&raw, "|move|p1a: Milotic|Recover|p1a: Milotic"),
        "the normal target form must NOT appear (a revert emits it)"
    );
}

/// BF-F8 — gen3customgame Beat Up emits a per-strike `|-activate|<user>|move: Beat Up|
/// [of] <ally>` (the gen3 mod's `condition.onModifySpA`, un-gated by `beatupnicknamesmod`
/// which is ABSENT in customgame). WRONG (pre-fix, over-corrected): NO activate at all.
#[test]
fn beat_up_emits_per_strike_activate_in_customgame() {
    let dugtrio = "Dugtrio|dugtrio|leftovers|arenatrap|beatup,earthquake|Jolly|,252,,,,252|M||||";
    let skarmory = "Skarmory|skarmory|leftovers|keeneye|drillpeck,spikes|Impish|252,,252,,,|M||||";
    let raw = run_logged(dugtrio, skarmory, SD, &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))]);
    assert!(
        raw.iter().any(|l| l.starts_with("|-activate|p1a: Dugtrio|move: Beat Up|[of] ")),
        "gen3customgame Beat Up must emit the per-strike `-activate move: Beat Up`; lines:\n{}",
        raw.join("\n")
    );
}

/// BF-F11 — a Protect BLOCKS a status move at TryHit BEFORE the type-immunity report:
/// Thunder Wave into a Ground-typed PROTECTING foe shows `|-activate|<foe>|Protect`, NOT
/// `|-immune|<foe>`. WRONG (pre-fix): the natural-immunity `-immune` preempted Protect.
#[test]
fn protect_blocks_status_move_before_the_immune_report() {
    // p1 Zapdos Thunder Wave; p2 Flygon (Ground → T-Wave-immune) Protect (priority 3, up first).
    let zapdos = "Zapdos|zapdos|leftovers|pressure|thunderwave,thunderbolt|Timid|,,,,,252|M||||";
    let flygon = "Flygon|flygon|leftovers|levitate|protect,earthquake|Jolly|,252,,,,252|M||||";
    let raw = run_logged(zapdos, flygon, SD, &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))]);
    assert!(
        line_present(&raw, "|-activate|p2a: Flygon|Protect"),
        "a Protecting foe must show `-activate Protect`; lines:\n{}",
        raw.join("\n")
    );
    assert!(
        !line_present(&raw, "|-immune|p2a: Flygon"),
        "Protect wins the TryHit — no `-immune` (a revert emits it)"
    );
}

/// BF-F12 — the `shiny` details flag on a shiny mon's `|switch|` line. WRONG (pre-fix):
/// the flag was dropped.
#[test]
fn shiny_mon_shows_the_shiny_details_flag() {
    // The packed shiny flag is field 9 (`name|species|item|ability|moves|nature|evs|gender|
    // ivs|SHINY|level`) → `…|M||S|` = gender M, ivs empty, shiny S.
    let quagsire = "Quagsire|quagsire|leftovers|waterabsorb|earthquake,surf|Relaxed|252,,252,,,|M||S||";
    let snorlax = "Snorlax|snorlax|leftovers|immunity|splash,bodyslam|Careful|252,,,,252,|M||||";
    let raw = run_logged(quagsire, snorlax, SD, &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))]);
    assert!(
        raw.iter().any(|l| l.starts_with("|switch|p1a: Quagsire|Quagsire, M, shiny|")),
        "a shiny mon's switch details must carry `, shiny`; lines:\n{}",
        raw.join("\n")
    );
}

/// BF-F14 — the Knock Off `-hint` fires ONCE per battle (the sim's `this.hint` dedup),
/// not on every Knock Off. WRONG (pre-fix): the hint re-emitted on each Knock Off.
#[test]
fn knock_off_hint_fires_once_per_battle() {
    let hariyama = "Hariyama|hariyama|leftovers|thickfat|knockoff,brickbreak|Adamant|252,252,,,,|M||||";
    let p2 = "Snorlax|snorlax|leftovers|immunity|splash,bodyslam|Careful|252,,,,252,|M||||]\
              Blissey|blissey|leftovers|naturalcure|softboiled,seismictoss|Calm|252,,,,252,|F||||";
    let raw = run_logged(
        hariyama,
        p2,
        SD,
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // Knock Off Snorlax (hint #1)
            ScriptDecision::both(Choice::Move(0), Choice::Switch(2)), // p2 → Blissey
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),   // Knock Off Blissey (NO hint)
        ],
    );
    let hints = raw
        .iter()
        .filter(|l| l.starts_with("|-hint|In Gens 3-4, Knock Off"))
        .count();
    assert_eq!(hints, 1, "the Knock Off hint must fire exactly ONCE; lines:\n{}", raw.join("\n"));
}

/// BF-F15 — a STAT-DROP status move BLOCKED BY A SUBSTITUTE renders the did-nothing
/// FORM-1: `|move|<user>|<Move>||[still]` (blank target + `[still]`) then a BARE
/// `|-fail|<user>`. WRONG (pre-fix): the stat-drop arm's sub-block returned emitting
/// NOTHING — the FORM-1 framing the build fixed for the OTHER did-nothing paths but
/// missed here. Captured `harness/probe_statdrop_substitute.js` (a Charm into a subbed
/// Snorlax at this seed: `|move|p1a: Gengar|Charm||[still]` + `|-fail|p1a: Gengar`).
#[test]
fn stat_drop_blocked_by_substitute_emits_still_and_fail() {
    // Charm (Atk −2, acc 100 → the roll draws-but-always-passes so it REACHES the sub
    // block, not a miss) into a subbed Snorlax. Substitute is a non-`bypasssub` block.
    let gengar = "Gengar|gengar|leftovers|levitate|charm,splash|Timid|,,,,,252|M||||";
    let snorlax = "Snorlax|snorlax|leftovers|immunity|substitute,splash|Careful|252,,,,252,|M||||";
    let raw = run_logged(
        gengar,
        snorlax,
        SD,
        &[
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)), // p1 Splash; p2 Substitute (up)
            ScriptDecision::both(Choice::Move(0), Choice::Move(1)), // p1 Charm INTO the sub; p2 Splash
        ],
    );
    // Guard against a vacuous pass: the Substitute did go up.
    assert!(
        line_present(&raw, "|-start|p2a: Snorlax|Substitute"),
        "the Substitute must be up before the Charm; lines:\n{}",
        raw.join("\n")
    );
    // THE FIX — the did-nothing `[still]` announce + a BARE `-fail|<user>`.
    assert!(
        line_present(&raw, "|move|p1a: Gengar|Charm||[still]"),
        "a sub-blocked stat-drop must render the `[still]` announce; lines:\n{}",
        raw.join("\n")
    );
    assert!(
        line_present(&raw, "|-fail|p1a: Gengar"),
        "a sub-blocked stat-drop must emit a bare `|-fail|<user>`; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — the normal foe-targeted announce must NOT appear (a revert emits it /
    // emits nothing, either way flipping these assertions).
    assert!(
        !line_present(&raw, "|move|p1a: Gengar|Charm|p2a: Snorlax"),
        "the sub-blocked form must NOT render the foe as the target; lines:\n{}",
        raw.join("\n")
    );
    // And no stat actually dropped (the sub absorbed it).
    for l in &raw {
        assert!(
            !l.starts_with("|-unboost|p2a: Snorlax|atk"),
            "the Charm drop must be absorbed by the sub (no `-unboost`). Offending: {l:?}"
        );
    }
}
