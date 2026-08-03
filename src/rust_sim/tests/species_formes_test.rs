//! `gen3_species_formes_v1` — the port must be able to CONSTRUCT a gen-3 ALTERNATE FORME.
//!
//! The data extractor used to drop every non-base forme (`baseSpecies != id`), because
//! poke-env's static pokedex is not gen-filtered by forme and carries 135 post-gen-3 formes
//! riding a gen-3 `num` (Megas / Gmax / regionals / Pikachu cosmetics). That blanket filter
//! also dropped the SIX real gen-3 formes and the 27 Unown letters, so
//! `data/pokemon/gen3_species.json` had no `deoxysspeed` / `unownb` row and
//! `MonState::from_set` fail-louded with `compute_stats: unknown species "Deoxys-Speed"`.
//!
//! Measured over 6000 generated `gen3randombattle` teams that was **6.6% of teams / ~14% of
//! battles** — the single largest team-construction failure cause, and a DATA gap, not an
//! engine gap. This is the durable CONSUMER-side guard: a gen-3-legal species the data layer
//! cannot describe fails HERE, not as a panic in a training run. The PRODUCER-side gate is
//! `node src/rust_sim/harness/dump_gen3_mechanics.js --check` (the committed species file vs
//! the resolved `Dex.mod('gen3')` universe).

use pokesim::battle::{Battle, BattleOptions, PackedTeam, PlayerOptions};
use pokesim::dex::{Dex, Type};

fn dex() -> Dex {
    Dex::for_gen(3)
}

#[test]
fn deoxys_formes_carry_their_own_base_stats() {
    let d = dex();
    let base = d.species("Deoxys").expect("deoxys");
    let attack = d.species("Deoxys-Attack").expect("deoxysattack row missing from gen3_species.json");
    let defense = d.species("Deoxys-Defense").expect("deoxysdefense row missing");
    let speed = d.species("Deoxys-Speed").expect("deoxysspeed row missing");

    assert_eq!((base.base_stats.atk, base.base_stats.spe), (150, 150));
    assert_eq!((attack.base_stats.atk, attack.base_stats.def), (180, 20));
    assert_eq!((defense.base_stats.def, defense.base_stats.spd), (160, 160));
    assert_eq!((speed.base_stats.spe, speed.base_stats.atk), (180, 95));

    // A forme SHARES the base's national-dex num — which is exactly why every num-indexed
    // consumer (the obs species channel, the model's `table[num]` buffers) must key on the
    // BASE forme, and why adding these rows had to be proven value-neutral there.
    for f in [attack, defense, speed] {
        assert_eq!(f.num, base.num, "{} num", f.id);
    }
}

#[test]
fn unown_cosmetic_formes_clone_the_base() {
    let d = dex();
    let unown = d.species("Unown").expect("unown");
    for id in ["Unown-B", "Unown-N", "Unown-Z", "Unown-Question", "Unown-Exclamation"] {
        let f = d.species(id).unwrap_or_else(|| panic!("{id} row missing from gen3_species.json"));
        assert_eq!(f.num, unown.num, "{id} num");
        assert_eq!(f.base_stats.spe, unown.base_stats.spe, "{id} stats");
        assert_eq!(f.base_stats.atk, unown.base_stats.atk, "{id} stats");
        assert_eq!(f.types, unown.types, "{id} types");
        assert_ne!(f.name, unown.name, "{id} keeps its own display name");
    }
}

#[test]
fn castform_weather_formes_are_describable() {
    // battleOnly (Forecast swaps them in-battle) — Forecast itself is deferred/unmodeled and
    // fail-loud, but their TYPES are the whole mechanic, so the data layer must carry them.
    let d = dex();
    assert_eq!(d.species("Castform").expect("castform").types, vec![Type::Normal]);
    assert_eq!(d.species("Castform-Sunny").expect("castformsunny").types, vec![Type::Fire]);
    assert_eq!(d.species("Castform-Rainy").expect("castformrainy").types, vec![Type::Water]);
    assert_eq!(d.species("Castform-Snowy").expect("castformsnowy").types, vec![Type::Ice]);
}

#[test]
fn a_packed_forme_team_constructs() {
    // The end-to-end reproducer: a packed team naming a forme (what `Teams.pack` writes for a
    // gen3 randbats Deoxys / Unown) must construct, with the FORME's stats — not a silent
    // fall-back to the base row, and not an `unknown species` error.
    let d = dex();
    // name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|misc
    let p1 = "|Deoxys-Speed|Leftovers|Pressure|psychic,icebeam,thunderbolt,shadowball|\
              Hardy|85,85,85,85,85,85|N|,,,,,||100|";
    let p2 = "|Unown-B|Leftovers|Levitate|hiddenpowerpsychic,hiddenpowerpsychic,\
              hiddenpowerpsychic,hiddenpowerpsychic|Hardy|85,85,85,85,85,85|N|,,,,,||100|";
    let opts = BattleOptions {
        format_id: "gen3customgame".to_string(),
        seed: Some("1,2,3,4".to_string()),
        p1: PlayerOptions { name: "P1".to_string(), team: PackedTeam(p1.to_string()) },
        p2: PlayerOptions { name: "P2".to_string(), team: PackedTeam(p2.to_string()) },
    };
    let battle = Battle::start(&opts, &d).expect("a gen-3 forme team must construct");
    let st = battle.state().expect("state");

    // L100, IV 31, EV 85 (-> +21), neutral nature: floor(2*base + 31 + 21) + 5.
    // Deoxys-Speed base 180 -> 417 (base Deoxys' 150 would be 357).
    assert_eq!(st.sides[0].pokemon[0].stats[5], 417, "Deoxys-Speed L100 Spe");
    // Unown base 48 -> 153.
    assert_eq!(st.sides[1].pokemon[0].stats[5], 153, "Unown-B L100 Spe");
}
