//! IMPRISON cannot livelock the agent side (`gen3_imprison_no_livelock_v1`, closes the P1 TECH_DEBT row
//! "Possible IMPRISON LIVELOCK on the agent side", found by the belief win-rate A/B, 2026-09-24, `10212cff`).
//!
//! The hypothesis was: the sim REJECTS a move a foe's Imprison seals, and a greedy policy handed the
//! same observation picks it again, forever. On the Rust env core it cannot happen, and these pins hold
//! why (each against the sim's own rule, `sim/side.ts` `chooseMove` → `emitChoiceError(…, {update})`):
//!
//! * an imprisoned pick is REFUSED and RE-REQUESTED with that slot `disabled` (`gen3_imprison_choice_reject_v1`
//!   + `gen3_rereq_accumulate_v1`), so the re-opened decision's MASK forbids it — the same pick is a
//!   `CALLER` error, never a second rejection; the turn does not advance on the refusal;
//! * an ALL-imprisoned mon's pick is STRUGGLE-substituted, never refused (`gen3_imprison_all_struggle_v1`);
//! * the two A/B matchups that ran 4,101 / 4,691 decisions (ladder index 1492: a Mean Look + Imprison
//!   Dusclops vs a Will-O-Wisp Gengar; 2345: an Imprison Dusclops vs a Will-O-Wisp Gengar), replayed on
//!   seeded battles with the WORST-CASE policy — a FIXED preference order (what a greedy argmax over an
//!   unchanging observation is) — all end, with at most one refusal per move slot per turn.
//!
//! The 2026-09-24 battles ran on the poke-env bridge stack BEFORE the port refused an imprisoned pick
//! (`gen3_imprison_choice_reject_v1` landed 2026-09-25, `0896b7d9`); there a sealed pick was a `cant`
//! and a turn passed, so its "4,101 decisions" are TURNS a no-progress policy spent against a Rest /
//! status-only Dusclops with no turn cap in that runner — a game-level stall, which the env's stall
//! forfeit (`turn_limit`) bounds, not a code livelock.

mod common;

use pokesim_env::core::columns::{col, counter, status, ACT, SIDES};
use pokesim_env::core::{Core, OwnedCols};

/// The action layout's move slots (`pokesim::present::legal`: switches 0..6, moves 6..10, Struggle 10).
const MOVE_START: usize = 6;
const N_MOVE_SLOTS: usize = 4;

fn set(species: &str, ability: &str, moves: &str) -> String {
    format!("{species}|||{ability}|{moves}|Hardy|||||100|")
}

/// A one-env core on `p1` vs `p2`, RESET.
fn core(p1: &str, p2: &str, turn_limit: Option<u32>) -> (Core, OwnedCols) {
    let mut spec = common::spec(1, 1, vec![p1.to_string(), p2.to_string()]);
    spec.turn_limit = turn_limit;
    let mut core = Core::new(spec).unwrap();
    let mut cols = OwnedCols::new(1);
    cols.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&[0, 1]);
    cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&[1, 2, 3, 4]);
    let a = cols.addrs();
    core.freeze(a).unwrap();
    assert_eq!(core.dispatch(b'R', a), status::OK, "{:?}", core.last_error());
    (core, cols)
}

fn index_of(core: &Core, side: usize, tok: &str) -> Option<i32> {
    core.inline_env(0).unwrap().open_tokens(side)?.iter().find(|(_, t)| t == tok).map(|(i, _)| *i as i32)
}

/// One STEP: each side that needs a decision plays `acts[side]`.
fn step(core: &mut Core, cols: &mut OwnedCols, acts: [Option<i32>; 2]) -> i32 {
    let need = cols.slice::<u8>(col::NEED).to_vec();
    let a: Vec<i32> = (0..SIDES).map(|s| if need[s] == 1 { acts[s].expect("an action for a side that needs one") } else { -1 }).collect();
    cols.slice_mut::<i32>(col::ACTION).copy_from_slice(&a);
    core.dispatch(b'S', cols.addrs())
}

#[test]
fn an_imprisoned_pick_is_refused_and_the_reopened_mask_forbids_it() {
    let p1 = [set("gengar", "levitate", "willowisp,thunderbolt"), set("snorlax", "immunity", "rest")].join("]");
    let p2 = [set("dusclops", "pressure", "imprison,willowisp"), set("snorlax", "immunity", "rest")].join("]");
    let (mut core, mut cols) = core(&p1, &p2, Some(300));
    // Turn 1: Gengar Thunderbolts (Dusclops lives), Dusclops casts Imprison.
    let tb = index_of(&core, 0, "move thunderbolt").expect("thunderbolt is open");
    let imp = index_of(&core, 1, "move imprison").expect("imprison is open");
    assert_eq!(step(&mut core, &mut cols, [Some(tb), Some(imp)]), status::OK, "{:?}", core.last_error());
    let turn = cols.slice::<u32>(col::TURN)[0];
    // Turn 2: Will-O-Wisp is sealed but HIDDEN — the request still offers it (the sim's `disabled: 'hidden'`).
    let wisp = index_of(&core, 0, "move willowisp").expect("the sealed move is offered: Imprison is hidden");
    let decisions = core.counters()[counter::DECISIONS];
    let dusclops = index_of(&core, 1, "move willowisp").unwrap();
    assert_eq!(step(&mut core, &mut cols, [Some(wisp), Some(dusclops)]), status::OK, "{:?}", core.last_error());
    // Refused and RE-REQUESTED: p1 decides again, the turn did not advance, the slot is now masked.
    assert_eq!(cols.slice::<u8>(col::NEED)[0], 1, "p1 is asked again after the refusal");
    assert_eq!(cols.slice::<u32>(col::TURN)[0], turn, "a refusal is not a turn");
    assert_eq!(core.counters()[counter::DECISIONS], decisions + 1, "the re-request is ONE new decision");
    assert_eq!(cols.slice::<u8>(col::MASK)[wisp as usize], 0, "the re-opened mask forbids the sealed move");
    assert_eq!(index_of(&core, 0, "move willowisp"), None);
    // The other move goes through, and the turn advances.
    let tb = index_of(&core, 0, "move thunderbolt").expect("thunderbolt is still open");
    assert_eq!(step(&mut core, &mut cols, [Some(tb), None]), status::OK, "{:?}", core.last_error());
    assert_eq!(cols.slice::<u32>(col::TURN)[0], turn + 1);
}

#[test]
fn an_all_imprisoned_mon_struggles_instead_of_being_refused() {
    let p1 = [set("gengar", "levitate", "willowisp"), set("snorlax", "immunity", "rest")].join("]");
    let p2 = [set("dusclops", "pressure", "imprison,willowisp"), set("snorlax", "immunity", "rest")].join("]");
    let (mut core, mut cols) = core(&p1, &p2, Some(300));
    let wisp = index_of(&core, 0, "move willowisp").unwrap();
    let imp = index_of(&core, 1, "move imprison").unwrap();
    assert_eq!(step(&mut core, &mut cols, [Some(wisp), Some(imp)]), status::OK);
    let turn = cols.slice::<u32>(col::TURN)[0];
    let wisp = index_of(&core, 0, "move willowisp").expect("offered (hidden)");
    let dw = index_of(&core, 1, "move willowisp").unwrap();
    assert_eq!(step(&mut core, &mut cols, [Some(wisp), Some(dw)]), status::OK, "{:?}", core.last_error());
    assert_eq!(cols.slice::<u32>(col::TURN)[0], turn + 1, "Struggle-substituted: the turn advanced, nothing was refused");
}

/// The two A/B matchups (`~/gen3ai_archive/belief_winrate_ab_2026-09-24/manifest_full.json`, ladder indices
/// 1492 and 2345: the trainee's pool team vs the ladder opponent with the Imprison Dusclops).
const AB_1492: [&str; 2] = [
    "Tyranitar||leftovers|sandstream|taunt,toxic,earthquake,rockslide|Serious|184,104,,,,220|||||]Cloyster||leftovers|shellarmor|spikes,rapidspin,icebeam,explosion|Modest|,,,168,244,96|||||]Gengar||leftovers|levitate|taunt,willowisp,icepunch,thunderbolt|Timid|168,,,,164,176||,0,,,,|||]Swampert||leftovers|torrent|protect,earthquake,hydropump,icebeam|Relaxed|240,,176,40,44,8|||||]Jolteon||leftovers|voltabsorb|thunderbolt,hiddenpowergrass,roar,batonpass|Timid|8,,,248,,252||,2,,30,,|||]Aerodactyl||choiceband|rockhead|earthquake,rockslide,hiddenpowerbug,doubleedge|Jolly|4,252,,,,252||,30,30,,30,|||",
    "Dugtrio||choiceband|arenatrap|earthquake,rockslide,hiddenpower,aerialace|Jolly|124,252,24,,,108|||||,Dark,]Zapdos||leftovers|pressure|substitute,batonpass,thunderbolt,drillpeck|Modest|216,,,116,,176|||||]Dusclops||leftovers|pressure|meanlook,willowisp,rest,imprison|Bold|252,,132,,124,|||||]Suicune||leftovers|pressure|surf,calmmind,substitute,rest|Bold|252,,240,,,16|||||]Blissey||leftovers|naturalcure|thunderwave,aromatherapy,softboiled,thunderbolt|Modest|,,252,252,,4|F||||]Claydol||leftovers|levitate|rapidspin,earthquake,explosion,psychic|Sassy|252,124,96,20,16,|||||",
];
const AB_2345: [&str; 2] = [
    "Swampert||leftovers|torrent|hydropump,earthquake,icebeam,focuspunch|Brave|176,136,,156,4,36|||||]Breloom||leftovers|effectspore|brickbreak,hiddenpowerghost,machpunch,spore|Jolly|,252,,,4,252||,30,,,30,|||]Arcanine||leftovers|intimidate|flamethrower,hiddenpowergrass,extremespeed,toxic|Quiet|252,,,180,36,40||,30,,30,,|||]Snorlax||leftovers|immunity|bodyslam,shadowball,focuspunch,selfdestruct|Adamant|80,124,184,,108,12|||||]Gengar||leftovers|levitate|thunderbolt,icepunch,willowisp,explosion|Timid|112,,,208,,188|||||]Claydol||leftovers|levitate|earthquake,psychic,rapidspin,explosion|Sassy|240,220,,,32,16|||||",
    "Forretress||leftovers|sturdy|spikes,explosion,rapidspin,zapcannon|Careful|252,4,,,252,|||||]Gengar||leftovers|levitate|thunderbolt,firepunch,explosion,willowisp|Timid|168,,,,164,176|||||]Dusclops||leftovers|pressure|imprison,shadowball,willowisp,rest|Careful|252,,4,,252,|||||]Tyranitar||leftovers|sandstream|fireblast,crunch,rockslide,dragondance|Quiet|248,,,212,,48|||||]Metagross||leftovers|clearbody|earthquake,explosion,protect,rockslide|Adamant|168,252,,,,88|||||]Aerodactyl||choiceband|rockhead|rockslide,hiddenpower,earthquake,doubleedge|Jolly|4,252,,,,252|||||,Dark,",
];

/// The A/B's two matchups, both sides on a FIXED preference order (per side per battle a seeded order
/// of the four move slots, then the switches; the first legal action is played) — a greedy policy whose observation
/// never changes, the worst case for a repeated sealed pick. Every battle ENDS within the stall
/// forfeit, nothing is refused or quarantined, and no side is asked more than 4 + 1 times in one turn
/// (each move slot can be refused at most once per decision: the re-request masks it).
#[test]
fn the_ab_imprison_matchups_end_under_a_fixed_preference_policy() {
    const SEEDS: u64 = 24;
    const TURN_LIMIT: u32 = 300;
    let (mut refusals, mut turns_total, mut ended) = (0u64, 0u64, 0u32);
    for (name, teams) in [("ab_1492", AB_1492), ("ab_2345", AB_2345)] {
        for seed in 0..SEEDS {
            let mut spec = common::spec(1, 1, vec![teams[0].to_string(), teams[1].to_string()]);
            spec.turn_limit = Some(TURN_LIMIT);
            let mut core = Core::new(spec).unwrap();
            let mut cols = OwnedCols::new(1);
            cols.slice_mut::<u32>(col::EP_TEAM).copy_from_slice(&[0, 1]);
            let mut rng = common::Rng(0x1A9_0000 + seed);
            let words: Vec<u32> = (0..4).map(|_| rng.below(65536) as u32).collect();
            cols.slice_mut::<u32>(col::EP_SEED).copy_from_slice(&words);
            let a = cols.addrs();
            core.freeze(a).unwrap();
            assert_eq!(core.dispatch(b'R', a), status::OK);
            // MOVES first (a seeded order of the four slots), then the switches, then Struggle: a
            // policy that attacks with its favourite move every turn — the one a sealed pick hits.
            let shuffled = |r: std::ops::Range<usize>, rng: &mut common::Rng| {
                let mut p: Vec<usize> = r.collect();
                for i in (1..p.len()).rev() {
                    p.swap(i, rng.below(i + 1));
                }
                p
            };
            let pref: Vec<Vec<usize>> = (0..SIDES)
                .map(|_| {
                    let mut p = shuffled(MOVE_START..MOVE_START + N_MOVE_SLOTS, &mut rng);
                    p.extend(shuffled(0..MOVE_START, &mut rng));
                    p.extend(MOVE_START + N_MOVE_SLOTS..ACT);
                    p
                })
                .collect();
            let (mut last_turn, mut asked_this_turn) = (0u32, [0u32; SIDES]);
            let mut done = false;
            // Bound: every turn costs at most (4 refusals + 1) decisions per side.
            for _ in 0..(TURN_LIMIT as usize + 2) * 5 * SIDES {
                let need = cols.slice::<u8>(col::NEED).to_vec();
                let mask = cols.slice::<u8>(col::MASK).to_vec();
                let turn = cols.slice::<u32>(col::TURN)[0];
                if turn != last_turn {
                    last_turn = turn;
                    asked_this_turn = [0; SIDES];
                }
                let mut acts = [-1i32; SIDES];
                for s in 0..SIDES {
                    if need[s] == 1 {
                        asked_this_turn[s] += 1;
                        assert!(asked_this_turn[s] <= 5 + 1, "{name} seed {seed}: p{} asked {} times in turn {turn}", s + 1, asked_this_turn[s]);
                        acts[s] = *pref[s].iter().find(|&&k| mask[s * ACT + k] == 1).expect("a legal action") as i32;
                    }
                }
                cols.slice_mut::<i32>(col::ACTION).copy_from_slice(&acts);
                let decisions = core.counters()[counter::DECISIONS];
                assert_eq!(core.dispatch(b'S', a), status::OK, "{name} seed {seed}: {:?}", core.last_error());
                // A refused-and-re-requested MOVE pick: same turn, the side is asked again, the picked
                // move slot is now masked while another move slot is still open (a forced
                // replacement after a faint opens no move slot).
                let (now_mask, now_need) = (cols.slice::<u8>(col::MASK).to_vec(), cols.slice::<u8>(col::NEED).to_vec());
                let moves = MOVE_START..MOVE_START + N_MOVE_SLOTS;
                let reopened = cols.slice::<u32>(col::TURN)[0] == turn && cols.slice::<u8>(col::DONE)[0] == 0
                    && core.counters()[counter::DECISIONS] > decisions
                    && (0..SIDES).filter(|&s| {
                        let k = acts[s];
                        need[s] == 1 && now_need[s] == 1 && moves.contains(&(k as usize))
                            && now_mask[s * ACT + k as usize] == 0
                            && moves.clone().any(|j| now_mask[s * ACT + j] == 1)
                    }).count() > 0;
                refusals += u64::from(reopened);
                if cols.slice::<u8>(col::DONE)[0] == 1 {
                    turns_total += u64::from(turn);
                    done = true;
                    break;
                }
            }
            assert!(done, "{name} seed {seed}: the battle did not end within the stall forfeit's bound");
            assert_eq!(core.bank().len(), 0, "{name} seed {seed}: quarantined");
            ended += 1;
        }
    }
    eprintln!("imprison A/B matchups: {ended} battles ended, {turns_total} turns, {refusals} refused-and-re-requested picks");
    assert_eq!(ended, 2 * SEEDS as u32);
    assert!(refusals > 0, "the fixed-preference policy never picked a sealed move: the pin has no teeth");
}
