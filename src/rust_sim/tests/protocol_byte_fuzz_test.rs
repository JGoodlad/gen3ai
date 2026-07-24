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

// ─────────────────────────────────────────────────────────────────────────────
// Round-8 STATUS-MOVE / RARE-ABILITY emission-form pins (the wide-net sweep finds)
// ─────────────────────────────────────────────────────────────────────────────

/// BF-F16 — a locked SOLAR BEAM fire-turn announce carries `|[from] lockedmove` WITH a
/// SPACE (the sim's `attrLastMove('[from] lockedmove')`). WRONG (pre-fix): the port emitted
/// the no-space `|[from]lockedmove`, a divergence on any real gen3ou Solar Beam / Shiftry
/// SolarBeam battle (POOL, both formats). (`gen3_omniscient_byte_fuzz_v1` FORM-A.)
#[test]
fn solar_beam_lockedmove_clause_has_a_space() {
    let venu = "Venusaur|venusaur||overgrow|solarbeam,tackle|Serious||N||||";
    let snor = "Snorlax|snorlax||thickfat|splash|Serious||N||||";
    let script = [
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // charge
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)), // fire
    ];
    let raw = run_logged(venu, snor, SD, &script);
    assert!(
        line_present(&raw, "|move|p1a: Venusaur|Solar Beam|p2a: Snorlax|[from] lockedmove"),
        "the fire-turn announce must carry `[from] lockedmove` WITH a space; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — the no-space form must NEVER appear (reverting the space fails this).
    for l in &raw {
        assert!(
            !l.contains("[from]lockedmove"),
            "no line may carry the no-space `[from]lockedmove`. Offending: {l:?}"
        );
    }
}

/// BF-F17 — SPEED BOOST's end-of-turn residual emits the `|-ability|<mon>|Speed Boost|boost`
/// ANNOUNCE before the `|-boost|<mon>|spe|1` (the sim's ability-source `boost()` announce).
/// WRONG (pre-fix): the port emitted only the `-boost`, dropping the announce.
#[test]
fn speed_boost_residual_emits_the_ability_announce() {
    let yanma = "Yanma|yanma||speedboost|tackle|Serious||N||||";
    let snor = "Snorlax|snorlax||thickfat|splash|Serious||N||||";
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))];
    let raw = run_logged(yanma, snor, SD, &script);
    let ab = raw.iter().position(|l| l == "|-ability|p1a: Yanma|Speed Boost|boost");
    let bo = raw.iter().position(|l| l == "|-boost|p1a: Yanma|spe|1");
    assert!(ab.is_some(), "the Speed Boost `-ability` announce is missing; lines:\n{}", raw.join("\n"));
    assert!(bo.is_some(), "the Speed Boost `-boost` is missing; lines:\n{}", raw.join("\n"));
    // THE PIN — the announce PRECEDES the boost (a revert drops the announce → ab is None).
    assert!(ab.unwrap() < bo.unwrap(), "the `-ability` announce must precede the `-boost`");
}

/// BF-F18 — HYPER CUTTER blocking an Atk drop emits the `|-fail|<mon>|unboost|Attack|[from]
/// ability: Hyper Cutter|[of] <mon>` WITH the `Attack` stat token (the sim's single-stat
/// `onTryBoost` form). WRONG (pre-fix): the port dropped `Attack` (the whole-table Clear-Body
/// form). Here an Intimidate switch-in drops the Pinsir's Atk → Hyper Cutter blocks it.
#[test]
fn hyper_cutter_unboost_fail_carries_the_attack_stat_token() {
    let pinsir = "Pinsir|pinsir||hypercutter|tackle|Serious||N||||";
    let gyara = "Gyarados|gyarados||intimidate|tackle|Serious||N||||\
                 ]Snorlax|snorlax||thickfat|splash|Serious||N||||";
    // Turn 1: p2 switches Gyarados out then back in (re-Intimidate) so the block fires
    // mid-battle. Simpler: the LEAD Gyarados Intimidates at switch-in (turn 0) already.
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Switch(1))];
    let raw = run_logged(pinsir, gyara, SD, &script);
    // The lead Intimidate (turn-0 framing) OR the switch-back re-Intimidate blocks the Atk drop.
    assert!(
        line_present(&raw, "|-fail|p1a: Pinsir|unboost|Attack|[from] ability: Hyper Cutter|[of] p1a: Pinsir"),
        "Hyper Cutter's `-fail` must carry the `Attack` stat token; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — the stat-less whole-table form must NOT appear for Hyper Cutter.
    for l in &raw {
        assert!(
            l != "|-fail|p1a: Pinsir|unboost|[from] ability: Hyper Cutter|[of] p1a: Pinsir",
            "Hyper Cutter must NOT emit the stat-less form. Offending: {l:?}"
        );
    }
}

/// BF-F19 — COLOR CHANGE's `|-start|<mon>|typechange|<Type>|…` renders the DISPLAY-cased type
/// name (`Psychic`), NOT the internal UPPERCASE key (`PSYCHIC`). WRONG (pre-fix): the port
/// emitted `PSYCHIC`.
#[test]
fn color_change_typechange_uses_display_cased_type() {
    let kec = "Kecleon|kecleon||colorchange|tackle|Serious||N||||";
    let ala = "Alakazam|alakazam||synchronize|psychic|Serious||N||||";
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))];
    let raw = run_logged(kec, ala, SD, &script);
    assert!(
        line_present(&raw, "|-start|p1a: Kecleon|typechange|Psychic|[from] ability: Color Change"),
        "Color Change must render the display-cased `Psychic`; lines:\n{}",
        raw.join("\n")
    );
    // THE PIN — the uppercase key must NEVER leak into a typechange line.
    for l in &raw {
        assert!(
            !(l.contains("typechange") && l.contains("PSYCHIC")),
            "a typechange line must not carry the uppercase key. Offending: {l:?}"
        );
    }
}

/// BF-F20 — a Figy-family CONFUSION berry emits its `|-start|<mon>|confusion` reveal EXACTLY
/// ONCE. WRONG (pre-fix): the berry site emitted it a SECOND time on top of `add_confusion`'s
/// own reveal (a duplicate `-start confusion`). Snorlax (Modest → −Atk → dislikes the Figy
/// spicy flavor) Belly Drums to exactly maxhp/2 (4 HP EVs → even maxhp) → the berry eats +
/// confuses.
#[test]
fn confusion_berry_emits_a_single_start_confusion() {
    // 4 HP EVs → maxhp even → Belly Drum lands hp == maxhp/2 → `2*hp <= maxhp` → berry eats.
    let snor = "Snorlax|snorlax|figyberry|thickfat|bellydrum,splash|Modest|4|N||||";
    let tauros = "Tauros|tauros||intimidate|splash|Serious||N||||";
    let script = [ScriptDecision::both(Choice::Move(0), Choice::Move(0))];
    let raw = run_logged(snor, tauros, SD, &script);
    let n = raw.iter().filter(|l| *l == "|-start|p1a: Snorlax|confusion").count();
    assert_eq!(
        n, 1,
        "the Figy-berry confusion `-start` must appear EXACTLY once (was double-emitted); lines:\n{}",
        raw.join("\n")
    );
}

/// P4b — PROTECT wins the TryHit precedence over SOUNDPROOF for a sound STATUS move.
/// A sound status move (Screech) into a Protecting + Soundproof foe emits `-activate Protect`,
/// NOT `-immune Soundproof` — gen3 runs Protect's `onTryHit` ahead of Soundproof's within the
/// SAME TryHit event (SIM-PROBED: Loudred Screech into a Protecting Soundproof Electrode →
/// `|-activate|Protect`). WRONG (pre-fix, `gen3_protect_before_soundproof_v1`): the stat-drop
/// (and phaze) arms checked Soundproof FIRST → `-immune Soundproof`. Loudred is fast (base 71 spe
/// + 252 EVs) so Screech resolves after Electrode's priority-3 Protect is up; seed forces the
/// 85%-acc Screech to HIT.
#[test]
fn protect_before_soundproof_for_a_sound_status_move() {
    let loudred = "Loudred|loudred||soundproof|screech,pound|Serious|,,,,,252|N||||";
    let electrode = "Electrode|electrode||soundproof|protect,thunderbolt|Serious||N||||";
    let raw = run_logged(loudred, electrode, "0,1,2,5",
        &[ScriptDecision::both(Choice::Move(0), Choice::Move(0))]);
    assert!(
        line_present(&raw, "|-activate|p2a: Electrode|Protect"),
        "a Protecting Soundproof foe hit by a sound status move must show `-activate Protect`; \
         lines:\n{}",
        raw.join("\n")
    );
    assert!(
        !raw.iter().any(|l| l.contains("|-immune|") && l.contains("Soundproof")),
        "Protect wins the TryHit — no `-immune Soundproof` (a revert emits it); lines:\n{}",
        raw.join("\n")
    );
}

/// Round-7 missing pin #1 — EFFECT SPORE inflicting SLEEP emits a BARE `|-status|<attacker>|slp`
/// (the gen3 `slp.onStart` drops the base `[from] ability` branch — only a MOVE source carries a
/// `[from]` clause). WRONG (pre-fix): the port over-attributed it `[from] ability: Effect Spore|[of]`
/// (par/psn/brn/frz from an ability STILL carry `[from] ability` — that path is unchanged). Muk
/// (Poison → psn-immune) Tackles (contact) an Effect Spore Breloom; the seed forces a proc that
/// samples `slp`.
#[test]
fn effect_spore_sleep_status_is_bare_not_ability_attributed() {
    let muk = "Muk|muk||stench|poisonjab,tackle|Adamant||M||||";
    let brel = "Breloom|breloom||effectspore|spore,machpunch|Adamant|||M||||";
    let raw = run_logged(muk, brel, "40,96,171,230",
        &[ScriptDecision::both(Choice::Move(1), Choice::Move(1))]);
    assert!(
        line_present(&raw, "|-status|p1a: Muk|slp"),
        "the Effect-Spore sleep must appear as a BARE `|-status|p1a: Muk|slp`; lines:\n{}",
        raw.join("\n")
    );
    assert!(
        !raw.iter().any(|l| l.starts_with("|-status|p1a: Muk|slp|")),
        "the sleep `-status` must NOT carry a `[from] ability: Effect Spore` clause (a revert adds \
         it); lines:\n{}",
        raw.join("\n")
    );
}

/// Round-7 missing pin #2 — FUTURE SIGHT resolving against a FAINTED slot occupant emits
/// `|-hint|Future Sight did not hit because the target is fainted.` (the gen3 `futuremove.onEnd`
/// early-return, `data/conditions.ts:399`; `once` falsy → no dedup). WRONG (pre-fix): the port
/// skipped the `-hint` (the fainted-target resolve was uncaptured). Constructed: p1 Tyranitar
/// (Sand Stream) casts Future Sight t1, chips p2 Growlithe (Fire → takes sand, water-weak, frail)
/// with Water Gun t2, then Calm Minds t3 (no p2 damage) so the RESOLVE-turn (t3) SAND CHIP
/// (residual order 8) KOs Growlithe BEFORE the order-11 Future Sight resolve → the slot occupant
/// is fainted (its replacement is deferred to the end of the residual phase) → the `-hint` fires.
#[test]
fn future_sight_resolving_against_a_fainted_slot_emits_the_hint() {
    let ttar = "Tyranitar|tyranitar||sandstream|futuresight,watergun,calmmind|Bold||M||||";
    let p2 = "Growlithe|growlithe||intimidate|splash|Bold||M||||]Sableye|sableye||keeneye|splash|Bold||M||||";
    let raw = run_logged(ttar, p2, "19,47,80,111", &[
        ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ScriptDecision::both(Choice::Move(2), Choice::Move(0)),
    ]);
    assert!(
        line_present(&raw, "|-hint|Future Sight did not hit because the target is fainted."),
        "a Future Sight resolving against a fainted slot occupant must emit the `-hint`; lines:\n{}",
        raw.join("\n")
    );
}

/// A CHARGE-holding mon that SELF-DESTRUCTS (self-KOs) emits NO `|-end|<user>|Charge|[silent]`
/// (`gen3_charge_selfko_no_end_v1`, byte-fuzz find ab_12_17 @ master-seed 200724). The sim's
/// `Pokemon.removeVolatile` returns false for a 0-HP mon, so `charge.onAfterMove`'s
/// `removeVolatile('charge')` is a no-op → NO `-end Charge` line (the faint's later silent
/// clearVolatile drops it). The port used to emit a spurious `|-end|Charge|[silent]` BEFORE the
/// `|faint|`. Ground truth `harness/probe_charge_selfko.js`.
#[test]
fn charge_selfko_emits_no_charge_end_line() {
    // Turn 1: Electrode Charge (sets the volatile). Turn 2: Electrode Self-Destruct (self-KO into a
    // bulky Snorlax that survives). Electrode holds a bench (Voltorb) so the self-KO forces a switch.
    let electrode =
        "Electrode|electrode||soundproof|charge,selfdestruct|Timid||M||||]Voltorb|voltorb||soundproof|spark,splash|Timid||M||||";
    let snorlax = "Snorlax|snorlax||immunity|splash,rest|Careful|252,,,,252,|M||||";
    let raw = run_logged(
        electrode,
        snorlax,
        SD,
        &[
            ScriptDecision::both(Choice::Move(0), Choice::Move(0)),
            ScriptDecision::both(Choice::Move(1), Choice::Move(0)),
        ],
    );
    assert!(
        line_present(&raw, "|faint|p1a: Electrode"),
        "Electrode self-KO'd via Self-Destruct; lines:\n{}",
        raw.join("\n")
    );
    assert!(
        !line_present(&raw, "|-end|p1a: Electrode|Charge|[silent]"),
        "a self-KO'd Charge holder must NOT emit `-end Charge` (removeVolatile no-ops at 0 HP — a \
         revert emits it); lines:\n{}",
        raw.join("\n")
    );
}
