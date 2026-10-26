//! The B7 moves pins (`gen3_b7_moves_v1`, gen3 full parity B7): Stockpile, Spit Up, Swallow, Mud
//! Sport, Water Sport, Mist, Lock-On, Mind Reader. Each pin asserts the sim's observables — the
//! protocol lines and the post-script PRNG seed, recorded from Showdown on the same teams and
//! choices (`new Battle({seed: [17, 29, 43, 59]})`, whose post-start seed — the one the port starts
//! from — is 18944,54804,63962,18698, or the board's own when the lead switch-ins draw; the settled
//! spec is `harness/probe_b7_moves.js`); the seed and
//! byte differential over every board is `b7_moves_golden_test.rs`.
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::turn::{Choice, ScriptDecision};

/// The sim's seed after `setPlayer` (team construction) from `[17, 29, 43, 59]`.
const SEED: &str = "18944,54804,63962,18698";

fn set(species: &str, ability: &str, moves: &str, evs: &str, lv: &str) -> String {
    format!("{species}|{species}||{ability}|{moves}|Hardy|{evs}|M|||{lv}|")
}
fn mon(species: &str, ability: &str, moves: &str) -> String {
    set(species, ability, moves, "85,85,85,85,85,85", "")
}
fn blissey(moves: &str) -> String {
    format!("Blissey|Blissey||NaturalCure|{moves}|Hardy|85,85,85,85,85,85|F||||")
}

/// The post-start seed when the lead switch-ins draw (Trace's target `sample`, a speed-tied lead
/// pair): the sim's `prng.getSeed()` after the second `setPlayer`, the seed the port starts from.
const SEED_PORYGON2: &str = "12055,55990,39841,51637";
const SEED_REGICE_MIRROR: &str = "5086,37025,45627,32254";

/// Play the script in gen3customgame; returns the lines (the `|t:|`, empty, `|` and `|upkeep`
/// separators dropped) from the first `|turn|1` on, and the final PRNG seed.
fn play(p1: &str, p2: &str, script: &[ScriptDecision]) -> (Vec<String>, String) {
    play_from(SEED, p1, p2, script)
}

fn play_from(seed: &str, p1: &str, p2: &str, script: &[ScriptDecision]) -> (Vec<String>, String) {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(seed.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = b.state_mut().expect("state");
    let (_o, lines) = st.run_full_battle_logged(script, &d);
    let all: Vec<String> = lines.into_iter().map(|l| l.0).collect();
    let t1 = all.iter().position(|l| l == "|turn|1").expect("|turn|1");
    let kept = all[t1 + 1..]
        .iter()
        .filter(|l| !(l.is_empty() || *l == "|" || *l == "|upkeep" || l.starts_with("|t:|")))
        .cloned()
        .collect();
    (kept, st.prng_seed())
}

fn mv(a: usize, b: usize) -> ScriptDecision {
    ScriptDecision::both(Choice::Move(a - 1), Choice::Move(b - 1))
}
fn p1_switch(k: usize) -> ScriptDecision {
    ScriptDecision { p1: Some(Choice::Switch(k - 1)), p2: None }
}

/// Assert `want` appears in `lines` as a contiguous run.
fn has_run(lines: &[String], want: &[&str]) {
    assert!(
        lines.windows(want.len()).any(|w| w.iter().zip(want).all(|(a, b)| a == b)),
        "missing the run {want:#?} in {lines:#?}"
    );
}

/// B7-0 — the variable-BP family (`MoveData::is_variable_bp`, which keeps a bp-0 damaging move off
/// the Status path and off Taunt's list) is every gen-3 move whose RESOLVED data has a
/// `basePowerCallback` over a bp-0, non-Status row, read off Showdown (fails loudly without node,
/// like `recharge_family_test.rs` RF1). The bare `hiddenpower` is the one exception: its IV-derived
/// BP and type have their own path (`gen3_iv_derived_hidden_power_bp_v1`).
#[test]
fn b7_0_the_variable_bp_family_is_showdowns() {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(
            "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');\
             const u=d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.id!=='hiddenpower');\
             console.log(u.filter(m=>m.basePower===0&&m.category!=='Status'&&m.basePowerCallback).map(m=>m.id).sort().join(','));\
             console.log(u.map(m=>m.id).join(','))",
        )
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let mut it = text.lines();
    let want = it.next().expect("the family line").trim().to_string();
    let universe = it.next().expect("the universe line");
    let d = Dex::for_gen(3);
    let mut ours: Vec<&str> = universe
        .split(',')
        .filter(|id| d.moves(id).map(|m| m.is_variable_bp()).unwrap_or(false))
        .collect();
    ours.sort_unstable();
    assert_eq!(want, ours.join(","), "the port's variable-BP family");
    assert!(ours.contains(&"spitup"), "Spit Up is in the family");
}

/// B7-1 — STOCKPILE stacks to three (`-start|…|stockpile<n>`), a fourth is the bare announce (its
/// `onTry` false prints nothing), and a 300-BP Spit Up then ends the volatile AFTER the damage
/// (`onAfterMove`: `|-end|<user>|Stockpile`).
#[test]
fn b7_1_stockpile_caps_at_three_and_spit_up_ends_it() {
    let (lines, seed) = play(&mon("Snorlax", "ThickFat", "stockpile,spitup,swallow"), &blissey("splash"), &[mv(1, 1), mv(1, 1), mv(1, 1), mv(1, 1), mv(2, 1)]);
    has_run(&lines, &["|-start|p1a: Snorlax|stockpile3", "|turn|4", "|move|p2a: Blissey|Splash|p2a: Blissey", "|-nothing", "|move|p1a: Snorlax|Stockpile|p1a: Snorlax", "|turn|5"]);
    has_run(&lines, &["|move|p1a: Snorlax|Spit Up|p2a: Blissey", "|-damage|p2a: Blissey|0 fnt", "|-end|p1a: Snorlax|Stockpile", "|faint|p2a: Blissey"]);
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B7-2 — with no stockpile, SPIT UP and SWALLOW are the bare announce: their `onTry` fails at
/// `singleEvent('Try')`, before the accuracy roll — ZERO draws.
#[test]
fn b7_2_spit_up_and_swallow_fail_bare_with_no_stockpile() {
    let (lines, seed) = play(&mon("Snorlax", "ThickFat", "stockpile,spitup,swallow"), &blissey("splash"), &[mv(2, 1), mv(3, 1)]);
    has_run(&lines, &["|move|p1a: Snorlax|Spit Up|p2a: Blissey", "|turn|2"]);
    has_run(&lines, &["|move|p1a: Snorlax|Swallow|p1a: Snorlax", "|turn|3"]);
    assert_eq!(seed, "45997,24898,56288,4140");
}

/// B7-3 — SWALLOW heals `modify(maxhp, 0.25 | 0.5)` for one / two layers (482 → +120 / +241), then
/// ends the stockpile; at full HP `this.heal` fails: `|-fail|<user>|heal`, and the stockpile still ends.
#[test]
fn b7_3_swallow_heals_by_layers_and_fails_at_full_hp() {
    let (lines, seed) = play(&mon("Snorlax", "ThickFat", "stockpile,spitup,swallow"), &blissey("seismictoss"), &[mv(1, 1), mv(3, 1), mv(1, 1), mv(1, 1), mv(3, 1)]);
    has_run(&lines, &["|move|p1a: Snorlax|Swallow|p1a: Snorlax", "|-heal|p1a: Snorlax|402/482", "|-end|p1a: Snorlax|Stockpile"]);
    has_run(&lines, &["|move|p1a: Snorlax|Swallow|p1a: Snorlax", "|-heal|p1a: Snorlax|343/482", "|-end|p1a: Snorlax|Stockpile"]);
    assert_eq!(seed, "41506,59591,10380,6292");
    let (lines, _) = play(&mon("Snorlax", "ThickFat", "stockpile,spitup,swallow"), &blissey("splash"), &[mv(1, 1), mv(3, 1)]);
    has_run(&lines, &["|move|p1a: Snorlax|Swallow|p1a: Snorlax", "|-fail|p1a: Snorlax|heal", "|-end|p1a: Snorlax|Stockpile"]);
}

/// B7-4 — Spit Up's `onAfterMove` runs whatever the move did: into a Ghost it draws its accuracy,
/// reports `-immune`, and the stockpile still ends.
#[test]
fn b7_4_spit_up_into_a_ghost_still_ends_the_stockpile() {
    let (lines, seed) = play(&mon("Snorlax", "ThickFat", "stockpile,spitup"), &mon("Gengar", "Levitate", "splash"), &[mv(1, 1), mv(2, 1)]);
    has_run(&lines, &["|move|p1a: Snorlax|Spit Up|p2a: Gengar", "|-immune|p2a: Gengar", "|-end|p1a: Snorlax|Stockpile"]);
    assert_eq!(seed, "32088,63066,43368,35871");
}

/// B7-5 — a SLEEP-TALK-called Spit Up ends the stockpile too: `runMove`'s AfterMove event takes the
/// activeMove (the called Spit Up), not Sleep Talk.
#[test]
fn b7_5_a_sleep_talk_spit_up_ends_the_stockpile() {
    let (lines, seed) = play(
        &mon("Snorlax", "ThickFat", "stockpile,rest,sleeptalk,spitup"),
        &blissey("seismictoss"),
        &[mv(1, 1), mv(1, 1), mv(2, 1), mv(3, 1), mv(3, 1)],
    );
    has_run(&lines, &["|move|p1a: Snorlax|Spit Up|p2a: Blissey|[from] Sleep Talk", "|-damage|p2a: Blissey|0 fnt", "|-end|p1a: Snorlax|Stockpile"]);
    assert_eq!(seed, "12509,27590,57008,31307");
}

/// B7-6 — MUD SPORT halves an Electric move's base power (Thunderbolt: 60 damage where the
/// no-sport control does 119), draw-free (the control ends at the same seed).
#[test]
fn b7_6_mud_sport_halves_thunderbolt() {
    let (lines, seed) = play(&mon("Snorlax", "ThickFat", "mudsport,splash"), &mon("Jolteon", "VoltAbsorb", "thunderbolt,splash"), &[mv(1, 2), mv(2, 1)]);
    has_run(&lines, &["|move|p1a: Snorlax|Mud Sport|p1a: Snorlax", "|-start|p1a: Snorlax|Mud Sport"]);
    has_run(&lines, &["|move|p2a: Jolteon|Thunderbolt|p1a: Snorlax", "|-damage|p1a: Snorlax|422/482"]);
    assert_eq!(seed, "44586,44725,29843,59136");
}

/// B7-7 — both sports on one mon are two priority-3 `onAnyBasePower` handlers at the same speed:
/// every damaging move's BasePower event shuffles them, one more draw (a Normal Tackle, which
/// neither halves; the one-sport control ends at a different seed with 333 HP left).
#[test]
fn b7_7_both_sports_on_one_mon_draw_the_base_power_tie() {
    let (lines, seed) = play(&mon("Swampert", "Torrent", "watersport,mudsport,splash"), &mon("Houndoom", "EarlyBird", "tackle,splash"), &[mv(1, 2), mv(2, 2), mv(3, 1)]);
    has_run(&lines, &["|-start|p1a: Swampert|move: Water Sport"]);
    has_run(&lines, &["|move|p2a: Houndoom|Tackle|p1a: Swampert", "|-damage|p1a: Swampert|335/362"]);
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B7-8 — Baton Pass passes MUD SPORT (the gen-4 mod's `noCopy: false`): the entrant's Thunderbolt
/// damage is halved (Gyarados takes 250, not ~2×).
#[test]
fn b7_8_baton_pass_passes_mud_sport() {
    let (lines, seed) = play(
        &[mon("Snorlax", "ThickFat", "mudsport,batonpass"), mon("Gyarados", "Intimidate", "splash")].join("]"),
        &mon("Jolteon", "VoltAbsorb", "thunderbolt,splash"),
        &[mv(1, 2), mv(2, 2), p1_switch(2), mv(1, 1)],
    );
    has_run(&lines, &["|move|p2a: Jolteon|Thunderbolt|p1a: Gyarados", "|-supereffective|p1a: Gyarados", "|-damage|p1a: Gyarados|102/352"]);
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B7-9 — MIST blocks a primary stat drop with `|-activate|<mon>|move: Mist` (Screech), drops a
/// secondary one SILENTLY (Psychic's SpD roll still drew), lasts five turns (`|-sideend|…|Mist` at
/// the fifth residual) and then Growl lands again.
#[test]
fn b7_9_mist_blocks_status_drops_loudly_secondaries_silently_for_five_turns() {
    let (lines, seed) = play(
        &mon("Lapras", "ShellArmor", "mist,splash"),
        &mon("Alakazam", "Synchronize", "growl,psychic,screech,splash"),
        &[mv(1, 1), mv(2, 2), mv(2, 3), mv(2, 4), mv(2, 4), mv(2, 1)],
    );
    has_run(&lines, &["|move|p1a: Lapras|Mist|p1a: Lapras", "|-sidestart|p1: P1|Mist"]);
    has_run(&lines, &["|move|p2a: Alakazam|Psychic|p1a: Lapras", "|-damage|p1a: Lapras|277/422", "|move|p1a: Lapras|Splash|p1a: Lapras"]);
    has_run(&lines, &["|move|p2a: Alakazam|Screech|p1a: Lapras", "|-activate|p1a: Lapras|move: Mist", "|move|p1a: Lapras|Splash|p1a: Lapras"]);
    has_run(&lines, &["|-nothing", "|-sideend|p1: P1|Mist", "|turn|6", "|move|p2a: Alakazam|Growl|p1a: Lapras", "|-unboost|p1a: Lapras|atk|1"]);
    assert_eq!(seed, "36368,26096,60906,25025");
}

/// B7-10 — MIST blocks Intimidate (`-ability … boost` then `-activate|<mon>|move: Mist`, and the Attack
/// stays: the next Body Slam lands at full power) and Tickle;
/// at the −6 floor the capped 0 survives it: Tickle shows Mist then `-unboost|…|atk|0`, Intimidate
/// only `-unboost|…|atk|0`.
#[test]
fn b7_10_mist_blocks_intimidate_and_leaves_the_floor_to_its_zero_line() {
    let lapras = [mon("Lapras", "ShellArmor", "mist,splash,bodyslam"), blissey("splash")].join("]");
    let foes = [blissey("charm,tickle"), mon("Gyarados", "Intimidate", "splash")].join("]");
    let to_gyarados = ScriptDecision { p1: Some(Choice::Move(1)), p2: Some(Choice::Switch(1)) };
    let (lines, seed) = play(&lapras, &foes, &[mv(1, 2), to_gyarados.clone(), mv(3, 1)]);
    has_run(&lines, &["|move|p2a: Blissey|Tickle|p1a: Lapras", "|-activate|p1a: Lapras|move: Mist"]);
    has_run(&lines, &["|-ability|p2a: Gyarados|Intimidate|boost", "|-activate|p1a: Lapras|move: Mist"]);
    // The Attack really stayed at 0: the Body Slam that follows lands at full power.
    has_run(&lines, &["|move|p1a: Lapras|Body Slam|p2a: Gyarados", "|-damage|p2a: Gyarados|283/352"]);
    assert_eq!(seed, "45159,18606,10676,27570");
    let (lines, seed) = play(&lapras, &foes, &[mv(2, 1), mv(2, 1), mv(2, 1), mv(1, 2), to_gyarados]);
    has_run(&lines, &["|move|p2a: Blissey|Tickle|p1a: Lapras", "|-activate|p1a: Lapras|move: Mist", "|-unboost|p1a: Lapras|atk|0"]);
    has_run(&lines, &["|-ability|p2a: Gyarados|Intimidate|boost", "|-unboost|p1a: Lapras|atk|0"]);
    assert_eq!(seed, "28376,17824,3029,24061");
}

/// B7-11 — the target's own CLEAR BODY runs before the side's Mist (the mon's handler has its
/// speed, the side condition 0): only Clear Body's `-fail` prints.
#[test]
fn b7_11_clear_body_answers_before_mist() {
    let (lines, seed) = play(&mon("Metagross", "ClearBody", "mist,splash"), &blissey("growl"), &[mv(1, 1), mv(2, 1)]);
    has_run(&lines, &["|move|p2a: Blissey|Growl|p1a: Metagross", "|-fail|p1a: Metagross|unboost|[from] ability: Clear Body|[of] p1a: Metagross", "|turn|2"]);
    assert!(!lines.iter().any(|l| l.contains("move: Mist")), "{lines:#?}");
    assert_eq!(seed, "5086,37025,45627,32254");
}

/// B7-12 — MIST on BOTH sides: the two order-3 side handlers tie at every residual (one draw a
/// turn), and the shuffle decides which `-sideend` prints first.
#[test]
fn b7_12_both_sides_mist_tie_at_the_residual() {
    let (lines, seed) = play(&mon("Lapras", "ShellArmor", "mist,splash"), &mon("Articuno", "Pressure", "mist,splash"), &[mv(1, 1), mv(2, 2), mv(2, 2), mv(2, 2), mv(2, 2)]);
    has_run(&lines, &["|-nothing", "|-sideend|p1: P1|Mist", "|-sideend|p2: P2|Mist", "|turn|6"]);
    assert_eq!(seed, "41506,59591,10380,6292");
}

/// B7-13 — a side's timed conditions are gathered in INSERTION order (`side.sideConditions`), which
/// the selection sort's swaps carry into the tie groups: Reflect / Light Screen then Mist, set on
/// both sides in different orders, end in the sim's order.
#[test]
fn b7_13_side_conditions_gather_in_insertion_order() {
    let (lines, seed) = play(
        &mon("Lapras", "ShellArmor", "reflect,mist,safeguard,splash"),
        &mon("Articuno", "Pressure", "lightscreen,mist,reflect,splash"),
        &[mv(1, 1), mv(2, 2), mv(3, 3), mv(4, 4), mv(4, 4), mv(4, 4)],
    );
    has_run(&lines, &["|-sideend|p1: P1|Reflect", "|-sideend|p2: P2|move: Light Screen", "|turn|6"]);
    has_run(&lines, &["|-sideend|p1: P1|Mist", "|-sideend|p2: P2|Mist", "|turn|7"]);
    assert_eq!(seed, "6367,58923,25654,30952");
}

/// B7-14 — LOCK-ON: `|-activate|<user>|move: Lock-On|[of] <target>`, then the next turn's Zap
/// Cannon draws NO accuracy roll (`onSourceAccuracy` → true); the lock (`duration: 2`) has lapsed
/// by the turn after, whose Zap Cannon rolls (and misses or hits by the seed).
#[test]
fn b7_14_lock_on_skips_the_next_accuracy_roll_then_lapses() {
    let (lines, seed) = play_from(SEED_PORYGON2, &mon("Porygon2", "Trace", "lockon,zapcannon,splash"), &blissey("splash"), &[mv(1, 1), mv(2, 1), mv(2, 1)]);
    has_run(&lines, &["|move|p1a: Porygon2|Lock-On|p2a: Blissey", "|-activate|p1a: Porygon2|move: Lock-On|[of] p2a: Blissey"]);
    has_run(&lines, &["|move|p1a: Porygon2|Zap Cannon|p2a: Blissey", "|-damage|p2a: Blissey|606/672", "|-status|p2a: Blissey|par"]);
    assert_eq!(seed, "6367,58923,25654,30952");
}

/// B7-15 — a held lock FAILS a recast (the move's `onTryHit`: `[still]` + `-fail`, and the lock's
/// own sure-hit means no accuracy draw), and Protect (TryHit priority 3) answers before that check.
#[test]
fn b7_15_a_held_lock_fails_a_recast_after_protect() {
    let (lines, seed) = play_from(SEED_PORYGON2, &mon("Porygon2", "Trace", "lockon,zapcannon,splash"), &blissey("splash,protect"), &[mv(1, 1), mv(1, 1), mv(1, 2)]);
    has_run(&lines, &["|move|p1a: Porygon2|Lock-On||[still]", "|-fail|p1a: Porygon2"]);
    has_run(&lines, &["|move|p1a: Porygon2|Lock-On|p2a: Blissey", "|-activate|p2a: Blissey|Protect", "|turn|4"]);
    assert_eq!(seed, "38721,49529,30239,49603");
}

/// B7-16 — the lock reaches a SEMI-INVULNERABLE target: the source's `onSourceInvulnerability`
/// (priority 1) returns 0 before Dig's handler can miss it, so Ice Beam hits the Golem underground.
#[test]
fn b7_16_lock_on_reaches_a_target_underground() {
    let (lines, seed) = play_from(SEED_PORYGON2, &mon("Porygon2", "Trace", "lockon,icebeam,zapcannon"), &mon("Golem", "Sturdy", "dig,splash"), &[mv(1, 1), mv(2, 1)]);
    has_run(&lines, &["|move|p1a: Porygon2|Ice Beam|p2a: Golem", "|-supereffective|p2a: Golem", "|-damage|p2a: Golem|115/322", "|move|p2a: Golem|Dig|p1a: Porygon2|[from] lockedmove"]);
    assert_eq!(seed, "20579,45021,5279,34087");
}

/// B7-17 — Baton Pass passes the LOCK (the gen-4 mod's `noCopy: false`): on the entrant its
/// NO_ORDER duration handler ties the foe's own lock at equal speed, one residual draw.
#[test]
fn b7_17_baton_pass_passes_the_lock_and_its_residual_tie() {
    let (lines, seed) = play_from(SEED_REGICE_MIRROR, 
        &[mon("Regice", "ClearBody", "lockon,batonpass"), mon("Regice", "ClearBody", "splash")].join("]"),
        &mon("Regice", "ClearBody", "lockon,splash"),
        &[mv(1, 1), mv(2, 2), p1_switch(2), mv(1, 2)],
    );
    has_run(&lines, &["|switch|p1a: Regice|Regice, M|322/322|[from] Baton Pass"]);
    assert_eq!(seed, "20888,5900,19750,40376");
}

/// B7-18 — Perish Song's per-mon Invulnerability check honours the user's lock: a locked Swellow
/// mid-Fly gets its perish count (`-start|…|perish3|[silent]`) instead of a `-miss`.
#[test]
fn b7_18_perish_song_reaches_a_locked_flyer() {
    let (lines, seed) = play_from(SEED_PORYGON2, 
        &set("Porygon2", "Trace", "lockon,perishsong,splash", "0,0,0,0,0,252", ""),
        &set("Swellow", "Guts", "fly,splash", "0,0,0,0,0,0", "50"),
        &[mv(1, 1), mv(2, 1)],
    );
    has_run(&lines, &["|-start|p1a: Porygon2|perish3|[silent]", "|-start|p2a: Swellow|perish3|[silent]", "|-fieldactivate|move: Perish Song"]);
    assert_eq!(seed, "45159,18606,10676,27570");
}

/// B7-19 — a FUTURE SIGHT strike is the caster's move: a lock the caster holds on the target skips
/// the strike's accuracy roll (the Lock-On's own roll makes the seed equal the Splash control's,
/// whose strike rolls).
#[test]
fn b7_19_a_lock_skips_the_future_sight_strikes_accuracy() {
    let (lines, seed) = play_from(SEED_PORYGON2, &mon("Porygon2", "Trace", "futuresight,lockon,splash"), &blissey("splash"), &[mv(1, 1), mv(3, 1), mv(2, 1)]);
    has_run(&lines, &["|-end|p2a: Blissey|move: Future Sight", "|-damage|p2a: Blissey|623/672", "|turn|4"]);
    assert_eq!(seed, "44586,44725,29843,59136");
}
