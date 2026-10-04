//! The ORACLE REVEAL's gates (`--oracle-reveal`, `designs/endstate/design_x5_belief_tokens.md` §7.6).
//!
//! * `off_is_inert` — the OFF path's bytes are PINNED to the value recorded on the commit BEFORE the
//!   reveal existed (`e0d56693`): the obs, mask and every label column over a corpus of real battles.
//!   A change to the off path fails here.
//! * `species_bytes_are_pinned` — the `species` level's own bytes, pinned, so a later level cannot move them.
//! * `species_differs_from_off_only_in_the_declared_cells` — the DIFFERENTIAL FUZZ over real bridge
//!   battles: two cores, one staging, one seeded policy (the actions are the `off` core's mask picks,
//!   fed to both), `off` against `species`, at EVERY decision of every episode: the seen opponent mons'
//!   slots and EVERY cell outside the opponent block's unseen tail are bit-identical; the unseen tail is
//!   the true team's unseen species, in dex-num order, each equal to `hypothesis_slot` (the encoder's
//!   own row for a never-seen mon); and the label columns agree on the seen slots and cover the tail.
//! * the edge cases (formes share a dex num; Species Clause; a slot revealed in play after the preview
//!   leaves the tail exactly once; a revealed mon the oracle team lacks is a FAULT).

mod common;

use pokesim::dex::Dex;
use pokesim::encoder::hypothesis::hypothesis_slot;
use pokesim::encoder::layout::{
    OBS_DIM, OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_SPECIES_OFFSET, TEAM_SIZE,
};
use pokesim::encoder::oracle::{species_num, Level, Oracle};
use pokesim::present::dex::to_id;
use pokesim::present::mon::PMon;
use pokesim::team::{pack, PokemonSet};
use pokesim_env::core::columns::{col, counter, SIDES};
use pokesim_env::core::{Core, OwnedCols};

/// FNV-1a-64 over the obs, mask, need and every label column (the columns the model trains on).
fn digest_into(h: &mut u64, cols: &OwnedCols) {
    let mut eat = |b: &[u8]| {
        for &x in b {
            *h ^= x as u64;
            *h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for c in [
        col::OBS, col::MASK, col::NEED, col::DONE, col::EPISODE, col::DEC_N, col::TURN,
        col::BELIEF_SPECIES, col::BELIEF_MOVES, col::KNOWN_MOVES, col::BELIEF_SPREAD, col::BELIEF_SPREAD_MASK,
        col::BELIEF_NATURE, col::BELIEF_NATURE_MASK, col::BELIEF_EV, col::BELIEF_EV_MASK, col::HP_TYPE_LABEL,
        col::HP_TYPE_MASK, col::ITEM_LABEL, col::ITEM_MASK, col::WIN_MARGIN, col::OPP_ACTION_KIND,
        col::OPP_ACTION_NUM, col::OPP_SWITCH_SLOT, col::OPP_SWITCH_SPECIES,
    ] {
        eat(cols.bytes(c));
    }
}

pub const FAMILIES: [&str; 6] = ["belief", "spread", "hp_type", "item", "intent", "margin"];

fn spec_with(n: usize, threads: usize, teams: Vec<String>, level: Level) -> pokesim_env::core::Spec {
    let mut spec = common::spec(n, threads, teams);
    spec.labels = pokesim_env::labels::declare(&FAMILIES.map(String::from)).expect("labels");
    spec.oracle_reveal = level;
    spec
}

/// Play `steps` ops of a seeded random policy over `n` envs and return the digest of everything
/// the model reads.
fn off_corpus_digest(n: usize, threads: usize, seed: u64, steps: usize) -> u64 {
    corpus_digest(Level::Off, n, threads, seed, steps)
}

/// [`off_corpus_digest`] at any level: the digest of everything the model reads over a corpus of real battles.
fn corpus_digest(level: Level, n: usize, threads: usize, seed: u64, steps: usize) -> u64 {
    let teams = common::corpus_teams();
    let nt = teams.len();
    let mut core = Core::new(spec_with(n, threads, teams, level)).expect("core");
    let mut cols = OwnedCols::new(n);
    let mut stage_rng = common::Rng(seed ^ 0x5157_A6E5);
    let mut act_rng = common::Rng(seed);
    for i in 0..n {
        common::stage(&mut cols, i, &mut stage_rng, nt);
    }
    let a = cols.addrs();
    core.freeze(a).expect("freeze");
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    assert_eq!(core.dispatch(b'R', a), 0, "RESET: {:?}", core.last_error());
    digest_into(&mut h, &cols);
    for _ in 0..steps {
        for i in 0..n {
            common::stage(&mut cols, i, &mut stage_rng, nt);
        }
        common::random_actions(&mut cols, &mut act_rng);
        assert_eq!(core.dispatch(b'S', a), 0, "STEP: {:?}", core.last_error().map(|e| e.json()));
        digest_into(&mut h, &cols);
    }
    let k = core.counters();
    assert!(k[counter::EPISODES_ENDED] >= 5 && k[counter::DECISIONS] > 2 * steps as u64, "non-vacuity: {k:?}");
    eprintln!("seed {seed}: decisions {} episodes ended {} refusals {}", k[counter::DECISIONS], k[counter::EPISODES_ENDED], k[counter::REFUSALS]);
    h
}

/// Recorded on `e0d56693` (the commit before the reveal), `n = 8`, `threads = 2`, 600 ops, the three
/// seeds below (27,333 decisions, 168 episodes, 0 refusals).
const PRE_REVEAL_DIGEST: [u64; 3] = [0xc67256dbfedb798f, 0x08b7ef47213baf22, 0x781a48de5326b9b6];

#[test]
fn off_is_inert() {
    let got = [11u64, 12, 13].map(|s| off_corpus_digest(8, 2, s, 600));
    assert_eq!(got, PRE_REVEAL_DIGEST, "the OFF path's obs / mask / label bytes moved");
}

/// The `species` level's bytes, recorded when it was built: the same corpus and seeds as `PRE_REVEAL_DIGEST`. A later
/// level (or any change to the encoder) must leave `species` byte-identical, and this is what says so.
const SPECIES_DIGEST: [u64; 3] = [0x60eeb3f53cc2c6fb, 0x653d80b821978e8b, 0x0f0e923713e27880];

#[test]
fn species_bytes_are_pinned() {
    let got = [11u64, 12, 13].map(|s| corpus_digest(Level::Species, 8, 2, s, 600));
    assert_eq!(got, SPECIES_DIGEST, "the `species` level's obs / mask / label bytes moved");
}

// ------------------------------------------------------------------ the differential fuzz

#[derive(Default, Debug)]
struct Seen {
    /// (env, side) decision rows compared.
    rows: u64,
    /// ... whose unseen tail was non-empty.
    tail_rows: u64,
    /// the first decision of an episode (turn 1: only the lead is seen).
    first_rows: u64,
    /// a decision where a mon was revealed since the previous one (the tail shrank).
    reveals: u64,
    /// ... rows where all of the team was seen (the tail is empty again).
    all_seen_rows: u64,
}

fn slot_of(row: &[f32], k: usize) -> &[f32] {
    &row[OFFSET_OPP_TEAM + k * POKEMON_FULL_DIM..OFFSET_OPP_TEAM + (k + 1) * POKEMON_FULL_DIM]
}

fn bits(x: &[f32]) -> Vec<u32> {
    x.iter().map(|v| v.to_bits()).collect()
}

/// The true team's (species id, dex num), packed order.
fn true_team(core: &Core, env: usize, side: usize, dex: &Dex) -> Vec<(String, i64)> {
    let packed = &core.inline_env(env).expect("inline pool").log.teams[1 - side];
    pokesim::team::unpack(packed, dex)
        .expect("unpack")
        .into_iter()
        .map(|s| {
            let id = to_id(&s.species);
            let n = species_num(&id).expect("num");
            (id, n)
        })
        .collect()
}

struct Prev {
    known: Vec<usize>,
    episode: Vec<u32>,
}

/// Compare one decision row of `off` (A) against `species` (B). Panics, naming the cell.
fn compare_row(env: usize, side: usize, a: &OwnedCols, b: &OwnedCols, core_a: &Core, dex: &Dex, prev: &mut Prev, seen: &mut Seen) {
    let k = env * SIDES + side;
    let ra = &a.slice::<f32>(col::OBS)[k * OBS_DIM..(k + 1) * OBS_DIM];
    let rb = &b.slice::<f32>(col::OBS)[k * OBS_DIM..(k + 1) * OBS_DIM];
    let ctx = format!("env {env} p{} episode {} turn {}", side + 1, a.slice::<u32>(col::EPISODE)[env], a.slice::<u32>(col::TURN)[env]);
    // r = the opponent mons the viewer has SEEN (off: the species_known slots)
    let r = (0..TEAM_SIZE).filter(|&j| slot_of(ra, j)[POKEMON_SPECIES_KNOWN_OFFSET] >= 0.5).count();
    for j in r..TEAM_SIZE {
        assert!(slot_of(ra, j).iter().all(|x| x.to_bits() == 0), "{ctx}: off writes an unseen slot {j} that is not zero");
    }
    // EVERY cell outside the unseen tail is bit-identical
    let tail_lo = OFFSET_OPP_TEAM + r * POKEMON_FULL_DIM;
    let tail_hi = OFFSET_OPP_TEAM + TEAM_SIZE * POKEMON_FULL_DIM;
    for i in (0..tail_lo).chain(tail_hi..OBS_DIM) {
        assert_eq!(ra[i].to_bits(), rb[i].to_bits(), "{ctx}: cell {i} ({}) differs outside the tail ({r} seen)", pokesim::encoder::cell_name(i));
    }
    // the tail is the true team's unseen species, dex-num order, each the encoder's own unseen-mon row
    let team = true_team(core_a, env, side, dex);
    let mut left: Vec<&(String, i64)> = team.iter().collect();
    for j in 0..r {
        let num = slot_of(ra, j)[POKEMON_SPECIES_OFFSET] as i64;
        let at = left.iter().position(|(_, n)| *n == num).unwrap_or_else(|| panic!("{ctx}: seen slot {j} num {num} is not on the true team {team:?}"));
        left.remove(at);
    }
    left.sort_by_key(|(_, n)| *n);
    for (j, (sp, num)) in left.iter().enumerate() {
        let got = slot_of(rb, r + j);
        assert_eq!(got[POKEMON_SPECIES_OFFSET] as i64, *num, "{ctx}: tail slot {j} is not the true unseen species {sp}");
        assert_eq!(bits(got), bits(&hypothesis_slot(sp).unwrap()), "{ctx}: tail slot {j} ({sp}) is not the encoder's unseen-mon row");
    }
    for j in r + left.len()..TEAM_SIZE {
        assert!(slot_of(rb, j).iter().all(|x| x.to_bits() == 0), "{ctx}: slot {j} past the oracle team is not zero");
    }
    // every opponent slot's species is the true team (the multiset: no double count, none missing)
    let mut nums_b: Vec<i64> = (0..TEAM_SIZE)
        .filter(|&j| slot_of(rb, j)[POKEMON_SPECIES_KNOWN_OFFSET] >= 0.5)
        .map(|j| slot_of(rb, j)[POKEMON_SPECIES_OFFSET] as i64)
        .collect();
    let mut nums_t: Vec<i64> = team.iter().map(|(_, n)| *n).collect();
    nums_b.sort_unstable();
    nums_t.sort_unstable();
    assert_eq!(nums_b, nums_t, "{ctx}: the opponent block's species are not the true team");
    // ---- the label columns
    let t = TEAM_SIZE;
    let i64s = |c: &OwnedCols, name: usize, per: usize| c.slice::<i64>(name)[k * t * per..(k + 1) * t * per].to_vec();
    let f32s = |c: &OwnedCols, name: usize, per: usize| -> Vec<u32> {
        c.slice::<f32>(name)[k * t * per..(k + 1) * t * per].iter().map(|x| x.to_bits()).collect()
    };
    let n_b = r + left.len();
    assert!(i64s(b, col::BELIEF_SPECIES, 1).iter().all(|&x| x == -1), "{ctx}: with every species stated there is nothing left to believe");
    assert!(i64s(b, col::BELIEF_MOVES, 4).iter().all(|&x| x == -1), "{ctx}: belief_moves must be PAD");
    // the seen slots' labels are `off`'s
    for (name, per) in [(col::KNOWN_MOVES, 4), (col::HP_TYPE_LABEL, 1), (col::ITEM_LABEL, 1), (col::BELIEF_NATURE, 1)] {
        assert_eq!(i64s(a, name, per)[..r * per], i64s(b, name, per)[..r * per], "{ctx}: label column {name} differs on the seen slots");
    }
    for (name, per) in [(col::ITEM_MASK, 1), (col::HP_TYPE_MASK, 1), (col::BELIEF_SPREAD_MASK, 1), (col::BELIEF_NATURE_MASK, 1), (col::BELIEF_EV_MASK, 1), (col::BELIEF_SPREAD, 5), (col::BELIEF_EV, 5)] {
        assert_eq!(f32s(a, name, per)[..r * per], f32s(b, name, per)[..r * per], "{ctx}: label column {name} differs on the seen slots");
    }
    // the unseen slots carry the true set's labels: every stated species has its moves, item and spread labelled
    let km = i64s(b, col::KNOWN_MOVES, 4);
    let item_mask: Vec<f32> = f32s(b, col::ITEM_MASK, 1).into_iter().map(f32::from_bits).collect();
    let spread_mask: Vec<f32> = f32s(b, col::BELIEF_SPREAD_MASK, 1).into_iter().map(f32::from_bits).collect();
    for j in 0..n_b {
        assert!(km[j * 4] >= 1, "{ctx}: slot {j}'s known_moves are not labelled");
        assert_eq!(item_mask[j], 1.0, "{ctx}: slot {j}'s item label is not masked in");
        assert_eq!(spread_mask[j], 1.0, "{ctx}: slot {j}'s spread label is not masked in");
    }
    for j in n_b..t {
        assert_eq!(km[j * 4], -1, "{ctx}: slot {j} past the oracle team has a known_moves label");
    }
    // the labels `off` and `species` agree on entirely
    for name in [col::OPP_ACTION_KIND, col::OPP_ACTION_NUM, col::OPP_SWITCH_SLOT, col::OPP_SWITCH_SPECIES] {
        assert_eq!(a.slice::<i64>(name)[k], b.slice::<i64>(name)[k], "{ctx}: intent column {name}");
    }
    assert_eq!(a.slice::<f32>(col::WIN_MARGIN)[k].to_bits(), b.slice::<f32>(col::WIN_MARGIN)[k].to_bits(), "{ctx}: win margin");
    // ---- coverage bookkeeping
    seen.rows += 1;
    if r < team.len() {
        seen.tail_rows += 1;
    } else {
        seen.all_seen_rows += 1;
    }
    let ep = a.slice::<u32>(col::EPISODE)[env];
    if a.slice::<u32>(col::DEC_N)[k] == 0 {
        seen.first_rows += 1;
        // turn 1: the lead alone is seen, and the observation already states the whole true team
        assert_eq!(r, 1, "{ctx}: the first decision shows one seen opponent");
        assert_eq!(nums_b.len(), team.len(), "{ctx}: the first decision's row states the whole team");
    }
    if prev.episode[k] == ep && prev.known[k] < r {
        seen.reveals += 1;
    }
    prev.episode[k] = ep;
    prev.known[k] = r;
}

/// Play `steps` ops of ONE seeded policy over `off` and `species` cores built from `teams` and compare
/// every decision. Returns what the corpus covered.
fn differential(n: usize, seed: u64, steps: usize, teams: Vec<String>) -> Seen {
    let nt = teams.len();
    let dex = Dex::for_gen(3);
    let mut ca = Core::new(spec_with(n, 1, teams.clone(), Level::Off)).expect("off core");
    let mut cb = Core::new(spec_with(n, 1, teams, Level::Species)).expect("species core");
    let (mut a, mut b) = (OwnedCols::new(n), OwnedCols::new(n));
    let mut stage_rng = common::Rng(seed ^ 0x5157_A6E5);
    let mut act_rng = common::Rng(seed);
    let restage = |a: &mut OwnedCols, b: &mut OwnedCols, rng: &mut common::Rng| {
        for i in 0..n {
            common::stage(a, i, rng, nt);
        }
        let team = a.slice::<u32>(col::EP_TEAM).to_vec();
        let seed_w = a.slice::<u32>(col::EP_SEED).to_vec();
        b.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&team);
        b.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&seed_w);
    };
    restage(&mut a, &mut b, &mut stage_rng);
    let (aa, ab) = (a.addrs(), b.addrs());
    ca.freeze(aa).expect("freeze a");
    cb.freeze(ab).expect("freeze b");
    let mut prev = Prev { known: vec![0usize; n * SIDES], episode: vec![u32::MAX; n * SIDES] };
    let mut seen = Seen::default();
    let mut step = |op: u8, ca: &mut Core, cb: &mut Core, a: &mut OwnedCols, b: &mut OwnedCols, seen: &mut Seen| {
        assert_eq!(ca.dispatch(op, aa), 0, "off: {:?}", ca.last_error().map(|e| e.json()));
        assert_eq!(cb.dispatch(op, ab), 0, "species: {:?}", cb.last_error().map(|e| e.json()));
        for c in [col::MASK, col::NEED, col::DONE, col::EPISODE, col::DEC_N, col::TURN, col::REWARD, col::TERMINATED, col::TRUNCATED, col::REFUSED] {
            assert_eq!(a.bytes(c), b.bytes(c), "column {c} differs between off and species");
        }
        for env in 0..n {
            for side in 0..SIDES {
                if a.slice::<u8>(col::NEED)[env * SIDES + side] == 1 {
                    compare_row(env, side, a, b, ca, &dex, &mut prev, seen);
                }
            }
        }
    };
    step(b'R', &mut ca, &mut cb, &mut a, &mut b, &mut seen);
    for _ in 0..steps {
        restage(&mut a, &mut b, &mut stage_rng);
        common::random_actions(&mut a, &mut act_rng);
        let actions = a.slice::<i32>(col::ACTION).to_vec();
        b.slice_mut::<i32>(col::ACTION).copy_from_slice(&actions);
        step(b'S', &mut ca, &mut cb, &mut a, &mut b, &mut seen);
    }
    for c in [&ca, &cb] {
        let k = c.counters();
        assert_eq!(k[counter::REFUSALS], 0, "a refusal: {:?}", c.bank().items().iter().map(|x| x.json()).collect::<Vec<_>>());
    }
    assert_eq!(ca.counters()[counter::DECISIONS], cb.counters()[counter::DECISIONS]);
    seen
}

#[test]
fn species_differs_from_off_only_in_the_declared_cells() {
    let mut total = Seen::default();
    for seed in [21u64, 22, 23] {
        let s = differential(8, seed, 500, common::corpus_teams());
        total.rows += s.rows;
        total.tail_rows += s.tail_rows;
        total.first_rows += s.first_rows;
        total.reveals += s.reveals;
        total.all_seen_rows += s.all_seen_rows;
    }
    eprintln!("species differential: {total:?}");
    // NON-VACUITY: the comparison saw turn-1 rows, mid-battle reveals and a fully-seen team
    assert!(total.rows > 20_000, "{total:?}");
    assert!(total.first_rows >= 300, "{total:?}");
    assert!(total.tail_rows > 5_000 && total.reveals > 1_000 && total.all_seen_rows > 1_000, "{total:?}");
}

// ------------------------------------------------------------------ edge cases

fn set(species: &str, moves: &[&str], item: &str, ability: &str) -> PokemonSet {
    PokemonSet {
        name: species.into(),
        species: species.into(),
        item: item.into(),
        ability: ability.into(),
        moves: moves.iter().map(|m| m.to_string()).collect(),
        nature: "Hardy".into(),
        ..Default::default()
    }
}

fn packed(sets: &[PokemonSet]) -> String {
    pack(sets, &Dex::for_gen(3))
}

fn seen_mon(species: &str) -> (String, PMon) {
    (format!("p2: {species}"), PMon::from_species(species, None).unwrap())
}

fn oracle(sets: &[PokemonSet]) -> Oracle {
    Oracle::new(Level::Species, &packed(sets), &Dex::for_gen(3)).expect("oracle")
}

fn tail_species(o: &Oracle, revealed: &[(String, PMon)]) -> Vec<String> {
    o.tail(revealed).unwrap().into_iter().map(|m| m.species.clone()).collect()
}

fn trio() -> Vec<PokemonSet> {
    vec![
        set("tyranitar", &["crunch"], "leftovers", "sandstream"),
        set("castform", &["weatherball"], "leftovers", "forecast"),
        set("skarmory", &["spikes"], "leftovers", "keeneye"),
    ]
}

#[test]
fn the_tail_is_the_unseen_species_in_dex_num_order_and_shrinks_as_play_reveals() {
    let o = oracle(&trio());
    // dex nums: skarmory 227, tyranitar 248, castform 351
    assert_eq!(tail_species(&o, &[]), ["skarmory", "tyranitar", "castform"]);
    assert_eq!(tail_species(&o, &[seen_mon("tyranitar")]), ["skarmory", "castform"]);
    // a slot revealed after the preview leaves the tail ONCE, and the rest keep their order
    assert_eq!(tail_species(&o, &[seen_mon("tyranitar"), seen_mon("skarmory")]), ["castform"]);
    assert!(tail_species(&o, &[seen_mon("tyranitar"), seen_mon("skarmory"), seen_mon("castform")]).is_empty());
}

#[test]
fn a_forme_shares_its_base_species_num_so_a_changed_forme_still_leaves_the_tail() {
    let o = oracle(&trio());
    // the reading's Castform has changed to its rainy forme; the team holds `castform`
    assert_eq!(species_num("castformrainy").unwrap(), species_num("castform").unwrap(), "a forme shares its base species' num");
    assert_eq!(tail_species(&o, &[seen_mon("castformrainy")]), ["skarmory", "tyranitar"], "matched by dex num, never by id");
    // ... and the reverse: the team holds a forme, the reading shows the base
    let d = oracle(&[set("deoxysattack", &["psychoboost"], "leftovers", "pressure"), set("skarmory", &["spikes"], "leftovers", "keeneye")]);
    assert_eq!(tail_species(&d, &[seen_mon("deoxys")]), ["skarmory"]);
}

#[test]
fn species_clause_duplicates_are_consumed_one_per_revealed_mon() {
    // a team that breaks Species Clause (never legal in gen3ou): one oracle entry per revealed mon,
    // so a duplicate is neither double-counted nor dropped
    let o = oracle(&[
        set("skarmory", &["spikes"], "leftovers", "keeneye"),
        set("skarmory", &["roar"], "leftovers", "keeneye"),
        set("tyranitar", &["crunch"], "leftovers", "sandstream"),
    ]);
    assert_eq!(tail_species(&o, &[seen_mon("skarmory")]), ["skarmory", "tyranitar"]);
    assert_eq!(tail_species(&o, &[seen_mon("skarmory"), seen_mon("skarmory")]), ["tyranitar"]);
}

#[test]
fn a_revealed_mon_the_oracle_team_does_not_hold_is_a_fault() {
    let o = oracle(&trio());
    let e = o.tail(&[seen_mon("zapdos")]).unwrap_err().message().to_string();
    assert!(e.contains("zapdos") && e.contains("not on the oracle team"), "{e}");
    // revealing one more mon than the team holds (a double count) is refused too
    let e = o.tail(&[seen_mon("tyranitar"), seen_mon("tyranitar")]).unwrap_err().message().to_string();
    assert!(e.contains("not on the oracle team"), "{e}");
}

#[test]
fn the_oracle_refuses_what_it_cannot_render() {
    let dex = Dex::for_gen(3);
    assert!(Oracle::new(Level::Off, &packed(&trio()), &dex).is_err(), "Off builds no oracle");
    let e = Oracle::new(Level::Species, "", &dex).unwrap_err().message().to_string();
    assert!(e.contains("0 mons"), "{e}");
    let seven: Vec<PokemonSet> = (0..7).map(|_| set("skarmory", &["spikes"], "leftovers", "keeneye")).collect();
    assert!(Oracle::new(Level::Species, &packed(&seven), &dex).is_err(), "more than six");
}

/// REAL battles with a forme change: Castform (Forecast) against a Rain Dance team, so Castform's forme
/// moves off its base id mid-battle while the species stay the true team at every decision.
#[test]
fn a_real_forecast_battle_keeps_the_team_exact_through_a_forme_change() {
    let dex = Dex::for_gen(3);
    let a = packed(&[
        set("castform", &["weatherball", "raindance", "sunnyday", "recover"], "leftovers", "forecast"),
        set("skarmory", &["spikes", "roar", "drillpeck", "toxic"], "leftovers", "keeneye"),
        set("tyranitar", &["crunch", "rockslide", "earthquake", "pursuit"], "leftovers", "sandstream"),
    ]);
    let b = packed(&[
        set("swampert", &["raindance", "surf", "earthquake", "protect"], "leftovers", "torrent"),
        set("blissey", &["softboiled", "seismictoss", "toxic", "icebeam"], "leftovers", "naturalcure"),
        set("metagross", &["meteormash", "earthquake", "explosion", "agility"], "leftovers", "clearbody"),
    ]);
    assert!(pokesim::team::unpack(&a, &dex).is_ok() && pokesim::team::unpack(&b, &dex).is_ok());
    let s = differential(8, 31, 400, vec![a, b]);
    eprintln!("forecast differential: {s:?}");
    assert!(s.rows > 2_000 && s.reveals > 100, "{s:?}");
}
