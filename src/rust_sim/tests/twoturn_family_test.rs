//! The TWO-TURN family pins (`gen3_twoturn_family_v1`, gen3 full parity B4): Razor Wind, Sky
//! Attack, Skull Bash and the semi-invulnerable Fly, Bounce, Dig, Dive. Each pin asserts the sim's
//! observables — a contiguous run of protocol lines and the PRNG seed after the script — recorded
//! from Showdown's synchronous `Battle` at seed `17,29,43,59` (post-start `18944,54804,63962,18698`) (the boards and the settled spec:
//! `harness/probe_twoturn_family.js`). The seed-and-byte differential over every board is
//! `twoturn_family_golden_test.rs`. TF1 / TF2 check the port's hardcoded id sets against the
//! RESOLVED gen-3 dex through node (they fail loudly without it, like the handler audit).
use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::Dex;
use pokesim::state::{SemiInvulnerable, TWO_TURN_MOVES};
use pokesim::turn::{Choice, ScriptDecision};

/// The seed Showdown's `Battle({seed: [17, 29, 43, 59]})` holds after `setPlayer` (the start
/// draws): the port's `start_with_switchins` begins here, as the golden rows do.
const SEED: &str = "18944,54804,63962,18698";
const FAST: &str = "0,0,0,0,0,252";
const EVEN: &str = "85,85,85,85,85,85";

fn set(species: &str, ability: &str, moves: &str, evs: &str, lv: &str) -> String {
    format!("{species}|{species}||{ability}|{moves}|Hardy|{evs}|M|||{lv}|")
}

fn play(p1: &str, p2: &str, script: &[ScriptDecision]) -> (Vec<String>, String) {
    let d = Dex::for_gen(3);
    let opts = BattleOptions {
        format_id: "gen3customgame".into(),
        seed: Some(SEED.into()),
        p1: PlayerOptions { name: "P1".into(), team: PackedTeam(p1.into()) },
        p2: PlayerOptions { name: "P2".into(), team: PackedTeam(p2.into()) },
    };
    let mut b = Battle::start_with_switchins(&opts, &d).expect("start");
    let st = b.state_mut().unwrap();
    let (_o, lines) = st.run_full_battle_logged(script, &d);
    let lines = lines
        .into_iter()
        .map(|l| l.0)
        .filter(|l| !(l.starts_with("|t:|") || l.starts_with("|debug|")))
        .collect();
    (lines, st.prng_seed())
}

fn mv(a: usize, b: usize) -> ScriptDecision {
    ScriptDecision::both(Choice::Move(a - 1), Choice::Move(b - 1))
}

/// Assert `want` occurs as a CONTIGUOUS run of `lines`, and the seed after the script.
fn expect(tag: &str, (lines, seed): (Vec<String>, String), want: &[&str], seed_after: &str) {
    let found = lines.windows(want.len()).any(|w| w.iter().zip(want).all(|(a, b)| a == b));
    assert!(found, "{tag}: the sim's lines {want:#?}\nnot in the port's\n{lines:#?}");
    assert_eq!(seed, seed_after, "{tag}: the seed after the script");
}

fn node(js: &str) -> String {
    let out = std::process::Command::new("node")
        .arg("-e")
        .arg(js)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("node must be runnable (see root CLAUDE.md worktree setup)");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).trim().to_string()
}

/// TF1 — the family is every gen-3 move whose RESOLVED data carries `flags.charge`, and the
/// semi-invulnerable kinds are exactly the ones whose resolved move carries a `condition`.
#[test]
fn tf1_the_family_is_every_gen3_charge_move() {
    let got = node(
        "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');\
         const c=d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3&&m.flags.charge);\
         console.log(c.map(m=>m.id).sort().join(',')+';'+c.filter(m=>m.condition).map(m=>m.id).sort().join(','))",
    );
    let mut ours: Vec<&str> = TWO_TURN_MOVES.to_vec();
    ours.sort_unstable();
    let mut semi: Vec<&str> = TWO_TURN_MOVES.iter().copied().filter(|m| SemiInvulnerable::of(m).is_some()).collect();
    semi.sort_unstable();
    assert_eq!(got, format!("{};{}", ours.join(","), semi.join(",")));
}

/// TF2 — the four conditions' hardcoded sets match the RESOLVED handlers, evaluated over every
/// gen-3 move id: `onInvulnerability` (undefined = bypass), `onSourceModifyDamage` /
/// `onSourceBasePower` (a chainModify(2) call), and `onImmunity('sandstorm' | 'hail')`.
#[test]
fn tf2_the_semi_invulnerable_sets_match_the_resolved_conditions() {
    let got = node(
        "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');\
         const ids=d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3).map(m=>m.id).sort();\
         const ctx={chainModify:(x)=>'x'+x};const out=[];\
         for(const s of ['fly','bounce','dig','dive']){const c=d.moves.get(s).condition;\
           const by=ids.filter(id=>c.onInvulnerability.call(ctx,null,null,{id})!==false);\
           const dmg=c.onSourceModifyDamage?ids.filter(id=>c.onSourceModifyDamage.call(ctx,1,null,null,{id})==='x2'):[];\
           const bp=c.onSourceBasePower?ids.filter(id=>c.onSourceBasePower.call(ctx,1,null,null,{id})==='x2'):[];\
           const imm=c.onImmunity?['sandstorm','hail'].every(w=>c.onImmunity.call(ctx,w,null)===false):false;\
           out.push([s,by.join('+'),dmg.join('+'),bp.join('+'),imm].join(':'));}\
         console.log(out.join(';'))",
    );
    let ids_js = node(
        "const {Dex}=require('../../deps/pokemon-showdown/dist/sim');const d=Dex.forFormat('gen3customgame');\
         console.log(d.moves.all().filter(m=>m.exists&&!m.isNonstandard&&m.gen<=3).map(m=>m.id).sort().join(','))",
    );
    let ids: Vec<&str> = ids_js.split(',').collect();
    let ours: Vec<String> = ["fly", "bounce", "dig", "dive"]
        .iter()
        .map(|s| {
            let k = SemiInvulnerable::of(s).unwrap();
            let pick = |f: &dyn Fn(&str) -> bool| ids.iter().copied().filter(|id| f(id)).collect::<Vec<_>>().join("+");
            format!(
                "{s}:{}:{}:{}:{}",
                pick(&|id| k.bypassed_by(id)),
                pick(&|id| k.doubles_damage(id)),
                pick(&|id| k.doubles_base_power(id)),
                k.weather_immune()
            )
        })
        .collect();
    assert_eq!(got, ours.join(";"));
}

/// TF3 — Razor Wind charges (the `[still]` announce + `-prepare`, zero move draws) and strikes
/// next turn as a `[from] lockedmove` (before B4 it panicked at its first use).
#[test]
fn tf3_razor_wind_charges_then_strikes() {
    let r = play(&set("Aerodactyl", "RockHead", "razorwind,splash", FAST, ""), &set("Snorlax", "Immunity", "splash", EVEN, ""), &[mv(1, 1), mv(1, 1)]);
    expect(
        "TF3",
        r,
        &[
            "|move|p1a: Aerodactyl|Razor Wind||[still]",
            "|-prepare|p1a: Aerodactyl|Razor Wind",
            "|move|p2a: Snorlax|Splash|p2a: Snorlax",
            "|-nothing",
            "|",
            "|upkeep",
            "|turn|2",
            "|",
            "|move|p1a: Aerodactyl|Razor Wind|p2a: Snorlax|[from] lockedmove",
            "|-damage|p2a: Snorlax|393/482",
        ],
        "20996,12330,49662,16889",
    );
}

/// TF4 — a move into a target mid-Fly MISSES before the accuracy roll (`[miss]` + `-miss`).
#[test]
fn tf4_tackle_misses_a_flying_target() {
    let r = play(&set("Snorlax", "Immunity", "fly,splash", EVEN, ""), &set("Jolteon", "VoltAbsorb", "splash,tackle", FAST, ""), &[mv(1, 1), mv(1, 2)]);
    expect(
        "TF4",
        r,
        &[
            "|move|p2a: Jolteon|Tackle|p1a: Snorlax|[miss]",
            "|-miss|p2a: Jolteon|p1a: Snorlax",
            "|move|p1a: Snorlax|Fly|p2a: Jolteon|[from] lockedmove",
            "|-resisted|p2a: Jolteon",
            "|-damage|p2a: Jolteon|221/271",
        ],
        "44586,44725,29843,59136",
    );
}

/// TF5 — Gust hits a flying target for DOUBLE damage (Fly's `onSourceModifyDamage`).
#[test]
fn tf5_gust_doubles_into_fly() {
    let r = play(&set("Snorlax", "Immunity", "fly,splash", EVEN, ""), &set("Jolteon", "VoltAbsorb", "splash,gust", FAST, ""), &[mv(1, 1), mv(1, 2)]);
    expect(
        "TF5",
        r,
        &["|move|p2a: Jolteon|Gust|p1a: Snorlax", "|-damage|p1a: Snorlax|424/482", "|move|p1a: Snorlax|Fly|p2a: Jolteon|[from] lockedmove"],
        "28376,17824,3029,24061",
    );
}

/// TF6 — Gust into Bounce doubles at the BasePower chain (`onSourceBasePower`).
#[test]
fn tf6_gust_doubles_into_bounce() {
    let r = play(&set("Snorlax", "Immunity", "bounce,splash", EVEN, ""), &set("Jolteon", "VoltAbsorb", "splash,gust", FAST, ""), &[mv(1, 1), mv(1, 2)]);
    expect(
        "TF6",
        r,
        &["|move|p2a: Jolteon|Gust|p1a: Snorlax", "|-damage|p1a: Snorlax|425/482", "|move|p1a: Snorlax|Bounce|p2a: Jolteon|[from] lockedmove"],
        "41506,59591,10380,6292",
    );
}

/// TF7 — an underground user takes no sandstorm chip on its charge turn.
#[test]
fn tf7_dig_skips_the_sandstorm_chip() {
    let r = play(&set("Snorlax", "Immunity", "dig,splash", EVEN, ""), &set("Jolteon", "VoltAbsorb", "sandstorm,splash", FAST, ""), &[mv(2, 1), mv(1, 2)]);
    expect(
        "TF7",
        r,
        &["|-prepare|p1a: Snorlax|Dig", "|", "|-weather|Sandstorm|[upkeep]", "|-damage|p2a: Jolteon|239/271|[from] Sandstorm", "|upkeep"],
        "32088,63066,43368,35871",
    );
}

/// TF8 — Perish Song misses the underwater mon per mon (`-miss` without `[miss]`).
#[test]
fn tf8_perish_song_misses_a_diving_mon() {
    let r = play(&set("Aerodactyl", "RockHead", "dive,splash", FAST, ""), &set("Gengar", "Levitate", "perishsong,splash", EVEN, ""), &[mv(1, 1)]);
    expect(
        "TF8",
        r,
        &[
            "|move|p2a: Gengar|Perish Song|p2a: Gengar",
            "|-miss|p2a: Gengar|p1a: Aerodactyl",
            "|-start|p2a: Gengar|perish3|[silent]",
            "|-fieldactivate|move: Perish Song",
        ],
        "45997,24898,56288,4140",
    );
}

/// TF9 — a Future Sight resolving into a Fly charge turn misses with no accuracy draw, its
/// `[miss]` retro-appended to the user's own `[still]` line.
#[test]
fn tf9_future_sight_misses_a_flying_target() {
    let r = play(&set("Snorlax", "Immunity", "fly,splash", EVEN, ""), &set("Alakazam", "Synchronize", "futuresight,splash", EVEN, ""), &[mv(2, 1), mv(2, 2), mv(1, 2)]);
    expect(
        "TF9",
        r,
        &[
            "|move|p1a: Snorlax|Fly||[still]|[miss]",
            "|-prepare|p1a: Snorlax|Fly",
            "|",
            "|-end|p1a: Snorlax|move: Future Sight",
            "|-miss|p2a: Alakazam|p1a: Snorlax",
            "|upkeep",
        ],
        "20996,12330,49662,16889",
    );
}

/// TF10 — Skull Bash's charge-turn boost prints `def|0` at the +6 cap.
#[test]
fn tf10_skull_bash_boost_at_the_cap() {
    let r = play(&set("Blastoise", "Torrent", "skullbash,irondefense", EVEN, ""), &set("Snorlax", "Immunity", "splash", EVEN, ""), &[mv(2, 1), mv(2, 1), mv(2, 1), mv(1, 1)]);
    expect(
        "TF10",
        r,
        &["|move|p1a: Blastoise|Skull Bash||[still]", "|-prepare|p1a: Blastoise|Skull Bash", "|-boost|p1a: Blastoise|def|0"],
        "5086,37025,45627,32254",
    );
}

/// TF11 — an Imprison cant on the strike turn LOSES the charge (MoveAborted): the next turn is
/// a fresh choice (here Struggle, both moves imprisoned), not another locked strike.
#[test]
fn tf11_imprison_loses_the_charge() {
    let r = play(&set("Aerodactyl", "RockHead", "skullbash,splash", FAST, ""), &set("Mew", "Synchronize", "imprison,skullbash,splash", EVEN, ""), &[mv(1, 1), mv(1, 3), mv(2, 3)]);
    expect(
        "TF11",
        r,
        &[
            "|cant|p1a: Aerodactyl|move: Imprison|Skull Bash",
            "|move|p2a: Mew|Splash|p2a: Mew",
            "|-nothing",
            "|",
            "|upkeep",
            "|turn|3",
            "|",
            "|move|p1a: Aerodactyl|Struggle|p2a: Mew",
        ],
        "44586,44725,29843,59136",
    );
}

/// TF11b — the same Imprison abort on a DIG strike turn lands the user at once: the slower foe's
/// Tackle then CONNECTS (a still-underground user would make it miss).
#[test]
fn tf11b_imprison_lands_the_digger() {
    let r = play(&set("Aerodactyl", "RockHead", "dig,splash", FAST, ""), &set("Mew", "Synchronize", "imprison,dig,tackle", EVEN, ""), &[mv(1, 1), mv(1, 3)]);
    expect(
        "TF11b",
        r,
        &[
            "|cant|p1a: Aerodactyl|move: Imprison|Dig",
            "|move|p2a: Mew|Tackle|p1a: Aerodactyl",
            "|-resisted|p1a: Aerodactyl",
            "|-damage|p1a: Aerodactyl|280/301",
        ],
        "44586,44725,29843,59136",
    );
}

/// TF12 — a Fly charge turn's residual gathers the sub-volatile's duration handler beside
/// twoturnmove's (one more tie shuffle than Razor Wind's charge: TF12's seed vs TF3's turn 1).
#[test]
fn tf12_the_fly_sub_volatile_joins_the_residual_tie() {
    let r = play(&set("Aerodactyl", "RockHead", "fly,splash", FAST, ""), &set("Snorlax", "Immunity", "splash", EVEN, ""), &[mv(1, 1)]);
    expect("TF12", r, &["|move|p1a: Aerodactyl|Fly||[still]", "|-prepare|p1a: Aerodactyl|Fly"], "45997,24898,56288,4140");
    let c = play(&set("Aerodactyl", "RockHead", "razorwind,splash", FAST, ""), &set("Snorlax", "Immunity", "splash", EVEN, ""), &[mv(1, 1)]);
    expect("TF12 control", c, &["|-prepare|p1a: Aerodactyl|Razor Wind"], "12055,55990,39841,51637");
}

/// TF13 — a strike CANCELLED by a gen-3 faint (the foe's Explosion) leaves the user underground
/// through that turn's residual (no sandstorm chip) and frees it the turn after.
#[test]
fn tf13_a_faint_cancelled_dig_stays_under_through_the_residual() {
    let p1 = format!("{}]{}", set("Snorlax", "Immunity", "dig,splash", EVEN, "60"), set("Snorlax", "Immunity", "splash", EVEN, ""));
    let p2 = format!("{}]{}", set("Gengar", "Levitate", "sandstorm,explosion", "0,252,0,0,0,252", ""), set("Jolteon", "VoltAbsorb", "splash", EVEN, ""));
    let r = play(&p1, &p2, &[mv(1, 1), mv(1, 2), ScriptDecision { p1: None, p2: Some(Choice::Switch(1)) }, mv(1, 1)]);
    expect(
        "TF13",
        r,
        &[
            "|move|p2a: Gengar|Explosion|p1a: Snorlax|[miss]",
            "|-miss|p2a: Gengar|p1a: Snorlax",
            "|faint|p2a: Gengar",
            "|",
            "|switch|p2a: Jolteon|Jolteon, M|292/292",
            "|",
            "|-weather|Sandstorm|[upkeep]",
            "|-damage|p2a: Jolteon|274/292|[from] Sandstorm",
            "|upkeep",
            "|turn|3",
            "|",
            "|move|p2a: Jolteon|Splash|p2a: Jolteon",
            "|-nothing",
            "|move|p1a: Snorlax|Dig||[still]",
        ],
        "44586,44725,29843,59136",
    );
}

/// TF14 — High Jump Kick into a flying target misses and still CRASHES (`MoveFail`).
#[test]
fn tf14_high_jump_kick_crashes_on_a_flying_target() {
    let r = play(&set("Aerodactyl", "RockHead", "fly,splash", FAST, ""), &set("Hitmonlee", "Limber", "highjumpkick,splash", EVEN, ""), &[mv(1, 1)]);
    expect(
        "TF14",
        r,
        &["|move|p2a: Hitmonlee|High Jump Kick|p1a: Aerodactyl|[miss]", "|-miss|p2a: Hitmonlee|p1a: Aerodactyl", "|-damage|p2a: Hitmonlee|177/262"],
        "5086,37025,45627,32254",
    );
}

/// TF15 — a fixed-damage move (Seismic Toss) into an underground target misses, zero draws.
#[test]
fn tf15_seismic_toss_misses_a_digging_target() {
    let r = play(&set("Aerodactyl", "RockHead", "dig,splash", FAST, ""), &set("Hitmonlee", "Limber", "seismictoss,splash", EVEN, ""), &[mv(1, 1)]);
    expect("TF15", r, &["|move|p2a: Hitmonlee|Seismic Toss|p1a: Aerodactyl|[miss]", "|-miss|p2a: Hitmonlee|p1a: Aerodactyl"], "45997,24898,56288,4140");
}

/// TF16 — a status move (Thunder Wave) into a bouncing target misses: status moves are not exempt.
#[test]
fn tf16_thunder_wave_misses_a_bouncing_target() {
    let r = play(&set("Aerodactyl", "RockHead", "bounce,splash", FAST, ""), &set("Gengar", "Levitate", "thunderwave,splash", EVEN, ""), &[mv(1, 1)]);
    expect("TF16", r, &["|move|p2a: Gengar|Thunder Wave|p1a: Aerodactyl|[miss]", "|-miss|p2a: Gengar|p1a: Aerodactyl"], "45997,24898,56288,4140");
}

/// TF17 — Earthquake hits an underground target for DOUBLE damage.
#[test]
fn tf17_earthquake_doubles_into_dig() {
    let r = play(&set("Snorlax", "Immunity", "dig,splash", EVEN, ""), &set("Jolteon", "VoltAbsorb", "splash,earthquake", FAST, ""), &[mv(1, 1), mv(1, 2)]);
    expect(
        "TF17",
        r,
        &["|move|p2a: Jolteon|Earthquake|p1a: Snorlax", "|-damage|p1a: Snorlax|338/482", "|move|p1a: Snorlax|Dig|p2a: Jolteon|[from] lockedmove"],
        "28376,17824,3029,24061",
    );
}
