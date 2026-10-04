//! F-X5-3 REGRESSION PIN (`gen3_label_lookup_guard_v1`): a label lookup that cannot be made is an
//! `Err` (the core makes it a FAULT), never a skipped species / move / item / stat. Driven through
//! `labels::write`, whose signature the guard did not change, so each case FAILS on a revert of the
//! guard: the old writers returned `Ok` with the slot silently left PAD / mask 0.
//!
//! * `the_clean_board_labels_every_slot` — the happy path, so the guards are not vacuous: every
//!   hidden mon fills a believed slot, every revealed slot gets its full truth moveset, item and
//!   Hidden Power type.
//! * `a_forme_matches_its_base_species_by_num` — a revealed forme id the truth team spells as its
//!   base species still matches (a forme SHARES its base species' num); the id match it replaced
//!   left the hidden set one too long and `known_moves` PAD.
//! * one test per guard, each asserting the `Err` names its cause.

use pokesim::dex::Dex;
use pokesim::encoder::layout::{OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, TEAM_SIZE};
use pokesim::encoder::oracle::{Level, Oracle};
use pokesim::present::board_reading::BoardReading;
use pokesim::present::mon::PMon;
use pokesim::team::{pack, PokemonSet};
use pokesim::trackers::clock::ClockConfig;
use pokesim::trackers::{IntentLabel, SideTrackers, KIND_MOVE, KIND_SWITCH};
use pokesim_env::core::columns::col;
use pokesim_env::core::OwnedCols;
use pokesim_env::labels::{self, EpisodeState};

const TEAM: [(&str, [&str; 4], &str); 6] = [
    ("tyranitar", ["crunch", "rockslide", "earthquake", "hiddenpowerbug"], "leftovers"),
    ("skarmory", ["spikes", "roar", "drillpeck", "toxic"], "leftovers"),
    ("blissey", ["softboiled", "seismictoss", "toxic", "icebeam"], "leftovers"),
    ("metagross", ["meteormash", "earthquake", "explosion", "agility"], "choiceband"),
    ("swampert", ["surf", "earthquake", "icebeam", "protect"], "leftovers"),
    ("gengar", ["thunderbolt", "icepunch", "firepunch", "willowisp"], "leftovers"),
];

fn mon(species: &str, moves: &[&str], item: &str) -> PMon {
    let mut m = PMon::from_species(species, None).expect("species");
    for mv in moves {
        m.learn_move(mv).expect("move");
    }
    m.item = Some(item.to_string());
    m
}

/// The truth team (the OTHER side's own reading) and the trainee's reading with `revealed` opp
/// mons, plus a row whose `species_known` marks the leading `revealed.len()` slots.
fn board(revealed: &[&str]) -> (BoardReading, BoardReading, Vec<f32>) {
    let mut truth = BoardReading::new(1, "truth", None).unwrap();
    for (sp, mv, it) in TEAM {
        truth.team.push((format!("p2: {sp}"), mon(sp, &mv, it)));
    }
    let mut own = BoardReading::new(0, "own", None).unwrap();
    for sp in revealed {
        own.opp.push((format!("p2: {sp}"), PMon::from_species(sp, None).unwrap()));
    }
    let mut row = vec![0f32; OFFSET_OPP_TEAM + TEAM_SIZE * POKEMON_FULL_DIM];
    for i in 0..revealed.len() {
        row[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0;
    }
    (own, truth, row)
}

fn write(fams: &[&'static str], own: &BoardReading, truth: &BoardReading, row: &[f32], trk: Option<&SideTrackers>, cols: &mut OwnedCols) -> Result<(), String> {
    write_with(fams, own, truth, None, row, trk, cols)
}

fn write_with(fams: &[&'static str], own: &BoardReading, truth: &BoardReading, oracle: Option<&Oracle>, row: &[f32], trk: Option<&SideTrackers>, cols: &mut OwnedCols) -> Result<(), String> {
    let a = cols.addrs();
    let mut c = unsafe { a.env(0) };
    labels::write(fams, 0, own, trk, None, 0, truth, oracle, row, &mut EpisodeState::default(), &mut c)
}

/// The ORACLE REVEAL's opponent team: the truth team's six species, packed.
fn oracle_of_team() -> Oracle {
    let dex = Dex::for_gen(3);
    let sets: Vec<PokemonSet> = TEAM
        .iter()
        .map(|(sp, mv, it)| PokemonSet {
            name: sp.to_string(),
            species: sp.to_string(),
            item: it.to_string(),
            ability: "pressure".into(),
            moves: mv.iter().map(|m| m.to_string()).collect(),
            nature: "Hardy".into(),
            ..Default::default()
        })
        .collect();
    Oracle::new(Level::Species, &pack(&sets, &dex), &dex).expect("oracle")
}

fn snum(id: &str) -> i64 {
    pokesim::encoder::data::tables().species[id].num as i64
}

fn mnum(id: &str) -> i64 {
    pokesim::encoder::data::tables().moves[id].num
}

fn err_of(r: Result<(), String>) -> String {
    r.expect_err("a lookup that cannot be made must be an Err, never a silent skip (F-X5-3)")
}

#[test]
fn the_clean_board_labels_every_slot() {
    let (own, truth, row) = board(&["tyranitar", "gengar"]);
    let mut cols = OwnedCols::new(1);
    write(&["belief", "hp_type", "item"], &own, &truth, &row, None, &mut cols).unwrap();
    let bs = &cols.slice::<i64>(col::BELIEF_SPECIES)[..TEAM_SIZE];
    let mut hidden: Vec<i64> = ["skarmory", "blissey", "metagross", "swampert"].map(snum).to_vec();
    hidden.sort();
    assert_eq!(bs, [-1, -1, hidden[0], hidden[1], hidden[2], hidden[3]], "every hidden mon fills a believed slot");
    let bm = &cols.slice::<i64>(col::BELIEF_MOVES)[..TEAM_SIZE * 4];
    assert!(bm[8..].iter().all(|&m| m > 0), "every believed slot carries four move nums: {bm:?}");
    let km = &cols.slice::<i64>(col::KNOWN_MOVES)[..TEAM_SIZE * 4];
    assert_eq!(&km[..4], &TEAM[0].1.map(mnum), "tyranitar's full truth moveset");
    assert_eq!(&km[4..8], &TEAM[5].1.map(mnum), "gengar's full truth moveset");
    assert_eq!(&cols.slice::<i64>(col::HP_TYPE_LABEL)[..2], &[0, -1], "tyranitar runs HP Bug (index 0); gengar none");
    assert_eq!(&cols.slice::<f32>(col::ITEM_MASK)[..3], &[1.0, 1.0, 0.0]);
}

#[test]
fn a_forme_matches_its_base_species_by_num() {
    // The truth side fields Castform; the trainee's reading saw it in its Sunny forme.
    let (mut own, mut truth, row) = board(&["tyranitar", "castformsunny"]);
    truth.team[5] = ("p2: castform".into(), mon("castform", &["weatherball", "sunnyday", "flamethrower", "icebeam"], "leftovers"));
    own.opp[1].1.species = "castformsunny".into();
    let mut cols = OwnedCols::new(1);
    write(&["belief", "item"], &own, &truth, &row, None, &mut cols).unwrap();
    let bs = &cols.slice::<i64>(col::BELIEF_SPECIES)[..TEAM_SIZE];
    assert_eq!(&bs[..2], &[-1, -1]);
    assert!(!bs.contains(&snum("castform")), "a revealed forme is not ALSO a hidden mon: {bs:?}");
    let km = &cols.slice::<i64>(col::KNOWN_MOVES)[4..8];
    assert_eq!(km, &["weatherball", "sunnyday", "flamethrower", "icebeam"].map(mnum), "the forme's slot gets the base mon's moveset");
}

#[test]
fn a_truth_species_with_no_dex_num_is_an_err() {
    let (own, mut truth, row) = board(&["tyranitar"]);
    truth.team[3].1.species = "notaspecies".into();
    let e = err_of(write(&["belief"], &own, &truth, &row, None, &mut OwnedCols::new(1)));
    assert!(e.contains("notaspecies") && e.contains("gen3_species.json"), "{e}");
}

#[test]
fn a_revealed_species_off_the_truth_team_is_an_err() {
    for fam in ["belief", "hp_type", "item", "spread"] {
        let (own, truth, row) = board(&["tyranitar", "zapdos"]);
        let e = err_of(write(&[fam], &own, &truth, &row, None, &mut OwnedCols::new(1)));
        assert!(e.contains("zapdos") && e.contains("not on the truth team"), "{fam}: {e}");
    }
}

#[test]
fn more_hidden_mons_than_believed_slots_is_an_err() {
    // Five slots read revealed but the reading lists only one: five truth mons are hidden and one
    // slot is believed.
    let (own, truth, mut row) = board(&["tyranitar"]);
    for i in 1..5 {
        row[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0;
    }
    let e = err_of(write(&["belief"], &own, &truth, &row, None, &mut OwnedCols::new(1)));
    assert!(e.contains("hidden truth mons"), "{e}");
}

#[test]
fn an_item_with_no_dex_num_is_an_err() {
    let (own, mut truth, row) = board(&["tyranitar"]);
    truth.team[0].1.item = Some("notanitem".into());
    let e = err_of(write(&["item"], &own, &truth, &row, None, &mut OwnedCols::new(1)));
    assert!(e.contains("notanitem"), "{e}");
}

#[test]
fn a_truth_mon_with_an_unknown_stat_is_an_err() {
    let (own, mut truth, row) = board(&["tyranitar"]);
    truth.team[2].1.stats = [None; 6];
    let e = err_of(write(&["spread"], &own, &truth, &row, None, &mut OwnedCols::new(1)));
    assert!(e.contains("no stats"), "{e}");
}

fn tracker(label: IntentLabel) -> SideTrackers {
    let mut t = SideTrackers::new(ClockConfig::default());
    t.label = Some(label);
    t
}

#[test]
fn an_intent_lookup_that_cannot_be_made_is_an_err() {
    let (own, truth, row) = board(&["tyranitar"]);
    let mv = |id: &str, attacker: &str| {
        tracker(IntentLabel { kind: KIND_MOVE, move_id: Some(id.into()), switch_species: None, switch_slot: None, attacker: Some(attacker.into()) })
    };
    let sw = |sp: Option<&str>| {
        tracker(IntentLabel { kind: KIND_SWITCH, move_id: None, switch_species: sp.map(String::from), switch_slot: None, attacker: None })
    };
    // the clean cases: a bare Hidden Power resolves to the attacker's TRUE typed num
    let mut cols = OwnedCols::new(1);
    write(&["intent"], &own, &truth, &row, Some(&mv("hiddenpower", "tyranitar")), &mut cols).unwrap();
    assert_eq!(cols.slice::<i64>(col::OPP_ACTION_NUM)[0], mnum("hiddenpowerbug"));
    write(&["intent"], &own, &truth, &row, Some(&sw(Some("skarmory"))), &mut cols).unwrap();
    assert_eq!(cols.slice::<i64>(col::OPP_SWITCH_SPECIES)[0], snum("skarmory"));
    // each was a silent UNKNOWN label / 237 / species 0
    for (trk, want) in [
        (mv("notamove", "tyranitar"), "notamove"),
        (mv("hiddenpower", "zapdos"), "not on the truth team"),
        (sw(Some("notaspecies")), "notaspecies"),
        (sw(None), "no switch-in species"),
    ] {
        let e = err_of(write(&["intent"], &own, &truth, &row, Some(&trk), &mut OwnedCols::new(1)));
        assert!(e.contains(want), "want {want:?}: {e}");
    }
}

// ------------------------------------------------------------------ the ORACLE REVEAL's labels

#[test]
fn the_oracle_labels_every_stated_slot_in_the_encoders_order() {
    // two seen (tyranitar, gengar), four unseen the row also states: dex-num order after the seen
    let (own, truth, mut row) = board(&["tyranitar", "gengar"]);
    for i in 2..TEAM_SIZE {
        row[OFFSET_OPP_TEAM + i * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0;
    }
    let oracle = oracle_of_team();
    let mut cols = OwnedCols::new(1);
    write_with(&["belief", "hp_type", "item"], &own, &truth, Some(&oracle), &row, None, &mut cols).unwrap();
    let bs = &cols.slice::<i64>(col::BELIEF_SPECIES)[..TEAM_SIZE];
    assert!(bs.iter().all(|&x| x == -1), "every species is stated: nothing is believed, {bs:?}");
    assert!(cols.slice::<i64>(col::BELIEF_MOVES)[..TEAM_SIZE * 4].iter().all(|&x| x == -1));
    let mut unseen: Vec<&(&str, [&str; 4], &str)> = TEAM.iter().filter(|t| t.0 != "tyranitar" && t.0 != "gengar").collect();
    unseen.sort_by_key(|t| snum(t.0));
    let km = &cols.slice::<i64>(col::KNOWN_MOVES)[..TEAM_SIZE * 4];
    assert_eq!(&km[..4], &TEAM[0].1.map(mnum));
    assert_eq!(&km[4..8], &TEAM[5].1.map(mnum));
    for (j, t) in unseen.iter().enumerate() {
        let slot = 2 + j;
        assert_eq!(&km[slot * 4..slot * 4 + 4], &t.1.map(mnum), "slot {slot} is {} (dex-num order)", t.0);
    }
    assert!(cols.slice::<f32>(col::ITEM_MASK)[..TEAM_SIZE].iter().all(|&m| m == 1.0), "every stated slot's item is labelled");
}

#[test]
fn a_row_and_an_oracle_that_disagree_are_refused_not_labelled_a_slot_off() {
    // the oracle names six species but the row states only the two seen: the producer and the
    // consumer disagree about the opp-slot packing
    let (own, truth, row) = board(&["tyranitar", "gengar"]);
    let oracle = oracle_of_team();
    let e = err_of(write_with(&["belief"], &own, &truth, Some(&oracle), &row, None, &mut OwnedCols::new(1)));
    assert!(e.contains("drifted") && e.contains("2 opp species"), "{e}");
    // ... and the reverse: the row states three, the reading + no oracle name two (a three-mon truth team,
    // so the older hidden-count guard does not fire first)
    let (own, mut truth, mut row) = board(&["tyranitar", "gengar"]);
    truth.team.truncate(3);
    truth.team.push(("p2: gengar".into(), mon("gengar", &TEAM[5].1, "leftovers")));
    row[OFFSET_OPP_TEAM + 2 * POKEMON_FULL_DIM + POKEMON_SPECIES_KNOWN_OFFSET] = 1.0;
    let e = err_of(write(&["belief"], &own, &truth, &row, None, &mut OwnedCols::new(1)));
    assert!(e.contains("drifted") && e.contains("3 opp species"), "{e}");
}
