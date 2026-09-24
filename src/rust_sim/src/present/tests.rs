//! One pin per reading rule (the table in `mod.rs`), each on constructed protocol, each naming
//! the poke-env line it mirrors. The same scenarios are replayed through poke-env ITSELF by
//! `agents/battle/rust_core_present_test.py`, which is what makes each pin a statement about
//! poke-env rather than about this crate's idea of it.

use super::view::MonView;
use super::*;
use crate::core_events::Line;

const OWN_TEAM: &str = "Metagross|||clearbody|meteormash,earthquake,explosion,agility|Adamant|252,252,,,4,|||||]\
Suicune|||pressure|calmmind,surf,rest,icebeam|Bold|252,,252,,4,|||||";

fn request(active: &str, cond_m: &str, cond_s: &str) -> String {
    let (am, as_) = if active == "Metagross" { ("true", "false") } else { ("false", "true") };
    format!(
        "|request|{{\"active\":[{{\"moves\":[{{\"move\":\"Meteor Mash\",\"id\":\"meteormash\",\"pp\":16,\"maxpp\":16,\"target\":\"normal\",\"disabled\":false}}]}}],\
\"side\":{{\"name\":\"me\",\"id\":\"p1\",\"pokemon\":[\
{{\"ident\":\"p1: Metagross\",\"details\":\"Metagross\",\"condition\":\"{cond_m}\",\"active\":{am},\"stats\":{{\"atk\":405,\"def\":296,\"spa\":203,\"spd\":216,\"spe\":214}},\"moves\":[\"meteormash\",\"earthquake\",\"explosion\",\"agility\"],\"baseAbility\":\"clearbody\",\"item\":\"leftovers\",\"pokeball\":\"pokeball\"}},\
{{\"ident\":\"p1: Suicune\",\"details\":\"Suicune\",\"condition\":\"{cond_s}\",\"active\":{as_},\"stats\":{{\"atk\":139,\"def\":266,\"spa\":260,\"spd\":308,\"spe\":213}},\"moves\":[\"calmmind\",\"surf\",\"rest\",\"icebeam\"],\"baseAbility\":\"pressure\",\"item\":\"leftovers\",\"pokeball\":\"pokeball\"}}]}}}}"
    )
}

const PREFIX: &[&str] = &["|player|p1|me||", "|player|p2|foe||", "|teamsize|p1|2", "|teamsize|p2|2", "|gen|3", "|start"];

fn run(lines: &[String]) -> BoardReading {
    let mut t = BoardReading::new(0, "me", Some(OWN_TEAM)).unwrap();
    for l in PREFIX {
        t.feed(&Line::parse(l).unwrap()).unwrap();
    }
    for l in lines {
        t.feed(&Line::parse(l).unwrap()).unwrap_or_else(|e| panic!("{l}: {e}"));
    }
    t
}

fn base() -> Vec<String> {
    vec![
        request("Metagross", "301/301", "341/341"),
        "|switch|p1a: Metagross|Metagross|301/301".into(),
        "|switch|p2a: Zapdos|Zapdos|100/100".into(),
        "|turn|1".into(),
    ]
}

fn with(extra: &[&str]) -> Vec<String> {
    let mut v = base();
    v.extend(extra.iter().map(|s| s.to_string()));
    v
}

fn view(lines: &[String]) -> OneSidedView {
    present(&run(lines)).unwrap()
}

fn opp<'a>(v: &'a OneSidedView, sp: &str) -> &'a MonView {
    v.opp.mons.iter().find(|m| m.species == sp).expect("opp mon")
}

fn ours<'a>(v: &'a OneSidedView, sp: &str) -> &'a MonView {
    v.ours.mons.iter().find(|m| m.species == sp).expect("own mon")
}

/// V1/V2 — `_update_team_from_request` fixes our slot order; the opponent's is first-named.
#[test]
fn v1_v2_slot_order_and_reveal() {
    let v = view(&with(&["|switch|p2a: Snorlax|Snorlax, M|100/100"]));
    assert_eq!(v.ours.mons.iter().map(|m| m.species.as_str()).collect::<Vec<_>>(), ["metagross", "suicune"]);
    assert_eq!(v.opp.mons.iter().map(|m| m.species.as_str()).collect::<Vec<_>>(), ["zapdos", "snorlax"]);
    assert!(ours(&v, "metagross").revealed && !ours(&v, "suicune").revealed);
    assert_eq!(v.opp.active.map(|i| v.opp.mons[i].species.as_str()), Some("snorlax"));
}

/// V3 — `_pressure_on`: a gen-3 Zapdos reads `pressure` by inference, so our move at it costs two.
#[test]
fn v3_pressure_by_inference_costs_two() {
    let v = view(&with(&["|move|p2a: Zapdos|Thunderbolt|p1a: Metagross", "|move|p1a: Metagross|Meteor Mash|p2a: Zapdos"]));
    let z = opp(&v, "zapdos");
    assert_eq!(z.ability.as_deref(), Some("pressure"), "single-possible-ability inference (V8)");
    assert_eq!((z.moves[0].id.as_str(), z.moves[0].current_pp, z.moves[0].max_pp), ("thunderbolt", 23, 24));
    let own = ours(&v, "metagross");
    let mm = own.moves.iter().find(|m| m.id == "meteormash").unwrap();
    assert_eq!(mm.current_pp, 14, "16 - (1 + Pressure)");
}

/// V4 — `Pokemon.end_turn` deletes an `ends_on_turn` effect; `start_effect` re-creates it.
#[test]
fn v4_an_ends_on_turn_effect_is_dropped_at_turn() {
    let v = view(&with(&["|-singleturn|p2a: Zapdos|Protect"]));
    assert!(opp(&v, "zapdos").volatiles.iter().any(|(k, _)| k == "protect"));
    let v = view(&with(&["|-singleturn|p2a: Zapdos|Protect", "|turn|2"]));
    assert!(!opp(&v, "zapdos").volatiles.iter().any(|(k, _)| k == "protect"));
    let v = view(&with(&["|-start|p2a: Zapdos|move: Taunt", "|turn|2", "|turn|3"]));
    assert_eq!(opp(&v, "zapdos").volatiles, vec![("taunt".to_string(), 2)], "turn-countable");
}

/// V4 — `apply_baton_pass`: the entrant takes the stages and the copied effects.
#[test]
fn v4_baton_pass_carries_stages_and_copied_effects() {
    let v = view(&with(&[
        "|-boost|p2a: Zapdos|spa|2",
        "|-start|p2a: Zapdos|Substitute",
        "|-start|p2a: Zapdos|move: Taunt",
        "|switch|p2a: Celebi|Celebi|100/100|[from] Baton Pass",
    ]));
    let c = opp(&v, "celebi");
    assert_eq!(c.boosts, vec![("spa", 2)]);
    assert_eq!(c.volatiles, vec![("substitute".to_string(), 0)], "taunt is not copied");
    assert!(opp(&v, "zapdos").boosts.is_empty(), "switch_out clears the passer");
}

/// V5 — fork R1: a CHANGED status restarts the count; `-cureteam` keeps it.
#[test]
fn v5_a_new_status_restarts_the_count() {
    let v = view(&with(&["|-status|p2a: Zapdos|tox", "|turn|2", "|turn|3", "|-status|p2a: Zapdos|slp", "|cant|p2a: Zapdos|slp"]));
    assert_eq!(opp(&v, "zapdos").status_counter, 1);
}

/// V6 — `moved`: consecutive stall moves count; anything else resets.
#[test]
fn v6_the_protect_streak_counts_consecutive_stall_moves() {
    let v = view(&with(&["|move|p2a: Zapdos|Protect|p2a: Zapdos", "|turn|2", "|move|p2a: Zapdos|Protect|p2a: Zapdos"]));
    assert_eq!(opp(&v, "zapdos").protect_counter, 2);
    let v = view(&with(&["|move|p2a: Zapdos|Protect|p2a: Zapdos", "|move|p2a: Zapdos|Thunderbolt|p1a: Metagross"]));
    assert_eq!(opp(&v, "zapdos").protect_counter, 0);
}

/// V7 — `_check_heal_message_for_item` never assigns a berry; `-enditem` records the consumed.
#[test]
fn v7_item_disclosure_rules() {
    let v = view(&with(&["|-heal|p2a: Zapdos|100/100|[from] item: Leftovers"]));
    assert_eq!(opp(&v, "zapdos").item.as_deref(), Some("leftovers"));
    let v = view(&with(&["|-enditem|p2a: Zapdos|Salac Berry|[eat]", "|-heal|p2a: Zapdos|100/100|[from] item: Salac Berry"]));
    let z = opp(&v, "zapdos");
    assert_eq!((z.item.as_deref(), z.consumed_item.as_deref()), (None, Some("salacberry")));
}

/// V8 — Trace: the `-ability` handler assigns `trace` then the copied ability; `switch_out`
/// drops the temporary slot.
#[test]
fn v8_trace_and_the_temporary_slot() {
    let v = view(&with(&[
        "|switch|p2a: Porygon2|Porygon2|100/100",
        "|-ability|p2a: Porygon2|Clear Body|[from] ability: Trace|[of] p1a: Metagross",
    ]));
    assert_eq!(opp(&v, "porygon2").ability.as_deref(), Some("clearbody"));
    let v = view(&with(&[
        "|switch|p2a: Porygon2|Porygon2|100/100",
        "|-ability|p2a: Porygon2|Clear Body|[from] ability: Trace|[of] p1a: Metagross",
        "|switch|p2a: Zapdos|Zapdos|100/100",
    ]));
    assert_eq!(opp(&v, "porygon2").ability.as_deref(), Some("trace"));
}

/// V9 — an opponent's spread and stats are unknown; its HP is the percent.
#[test]
fn v9_the_opponent_is_hidden() {
    let v = view(&with(&["|-damage|p2a: Zapdos|54/100"]));
    let z = opp(&v, "zapdos");
    assert_eq!((z.ivs.clone(), z.spread_known, z.stats), (None, false, [None; 6]));
    assert_eq!((z.current_hp, z.max_hp, z.hp_fraction), (54, 100, 0.54));
    let m = ours(&v, "metagross");
    assert_eq!(m.stats, [Some(301), Some(405), Some(296), Some(203), Some(216), Some(214)]);
    assert_eq!(m.evs.as_deref(), Some(&[252, 252, 0, 0, 4, 0][..]));
    assert_eq!(m.nature.as_deref(), Some("adamant"));
}

/// PE-V10 (was a poke-env reading FINDING; the fork is fixed, `gen3_pe_reading_fixes_v1`): the sim
/// clears a fainted mon's stages at the faint (`clearVolatile`), so the view holds none. Fails on a
/// revert to upstream poke-env's `faint()`.
#[test]
fn pe_v10_a_fainted_mon_holds_no_stages() {
    let v = view(&with(&["|-boost|p2a: Zapdos|spa|1", "|faint|p2a: Zapdos"]));
    assert!(opp(&v, "zapdos").fainted);
    assert!(opp(&v, "zapdos").boosts.is_empty(), "{:?}", opp(&v, "zapdos").boosts);
    let v = view(&with(&["|-boost|p2a: Zapdos|spa|1"]));
    assert_eq!(opp(&v, "zapdos").boosts, vec![("spa", 1)], "non-vacuity: the boost was read");
}

/// PE-V16: the sim's `flashfire` volatile lasts until its holder leaves the field, so the view
/// keeps Flash Fire through the holder's own Fire move (upstream poke-env's `moved` ended it there).
#[test]
fn pe_v16_flash_fire_survives_its_holders_fire_move() {
    let ff = |v: &OneSidedView| opp(v, "zapdos").volatiles.iter().any(|(k, _)| k == "flashfire");
    let lines = with(&["|-start|p2a: Zapdos|ability: Flash Fire", "|move|p2a: Zapdos|Flamethrower|p1a: Metagross"]);
    assert!(ff(&view(&lines)), "the holder's Fire move does not end it");
    let mut out = lines;
    out.push("|switch|p2a: Snorlax|Snorlax|100/100".into());
    assert!(!ff(&view(&out)), "leaving the field does");
}

/// PE-R1b: a badly-poisoned mon's counter is the sim's STAGE — one per residual chip since its
/// switch-in — so a mon that entered AFTER the residual reads 0 at the next `|turn|`, where
/// upstream poke-env (+1 per `|turn|`) read 1.
#[test]
fn pe_r1b_the_toxic_counter_is_the_stage() {
    let v = view(&with(&["|-status|p2a: Zapdos|tox", "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
                         "|-damage|p2a: Zapdos|82/100 tox|[from] psn", "|turn|3"]));
    assert_eq!(opp(&v, "zapdos").status_counter, 2);
    let v = view(&with(&["|-status|p2a: Zapdos|tox", "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
                         "|switch|p2a: Snorlax|Snorlax|100/100", "|switch|p2a: Zapdos|Zapdos|94/100 tox", "|turn|3"]));
    assert_eq!(opp(&v, "zapdos").status_counter, 0, "no residual since it re-entered");
    let v = view(&with(&["|-status|p2a: Zapdos|tox", "|-damage|p2a: Zapdos|94/100 tox|[from] psn", "|turn|2",
                         "|-damage|p2a: Zapdos|82/100 tox|[from] psn"]));
    assert_eq!(opp(&v, "zapdos").status_counter, 2, "stage 2 already, before the next |turn|");
}

/// V11 — a screen stores its start turn; Spikes counts layers.
#[test]
fn v11_screens_store_their_start_turn() {
    let v = view(&with(&["|turn|4", "|-sidestart|p2: foe|Reflect", "|-sidestart|p1: me|Spikes", "|-sidestart|p1: me|Spikes"]));
    assert_eq!(v.opp.side_conditions, vec![("reflect".to_string(), 4)]);
    assert_eq!(v.ours.side_conditions, vec![("spikes".to_string(), 2)]);
}

/// V12 — an ability-set weather is permanent; `turns_active` counts from its set turn.
#[test]
fn v12_weather_is_folded_from_its_set_line() {
    let v = view(&with(&["|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Zapdos", "|turn|2", "|-weather|Sandstorm|[upkeep]", "|turn|3"]));
    assert_eq!((v.weather.weather.as_deref(), v.weather.is_permanent, v.weather.turns_active), (Some("sandstorm"), true, 2));
}

/// V13 — the request's legality: Struggle is the flag, never a slot.
#[test]
fn v13_legality_comes_from_the_request() {
    let t = run(&base());
    let l = legal_actions(&t).unwrap();
    assert_eq!(l.move_slots.len(), 1);
    assert_eq!(l.switches, vec![legal::LegalSwitch { species: "suicune".into(), slot: 1 }]);
    assert_eq!(mask(&l), [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]);
}

/// A parse error or a line poke-env raises on is refused, never guessed.
#[test]
fn a_line_poke_env_would_raise_on_is_refused() {
    let mut t = run(&base());
    assert!(t.feed(&Line::parse("|-boost|p2a: Zapdos|nonsense|1").unwrap()).is_err());
    assert!(t.feed(&Line::parse("|move|p2a: Zapdos|Thunderbolt|p1a: Metagross|[from] move: Assist").unwrap()).is_err());
}
