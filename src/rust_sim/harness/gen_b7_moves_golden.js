// GOLDEN GENERATOR — the B7 moves (`gen3_b7_moves_v1`, gen3 full parity B7): Stockpile, Spit Up,
// Swallow, Mud Sport, Water Sport, Mist, Lock-On, Mind Reader. The settled spec is
// `harness/probe_b7_moves.js`'s header. Each board runs over N seeds; each row records the format,
// the decisions as played ([p1, p2], null = no choice), the sim SEED_BEFORE / SEED_AFTER, the
// FRAMING window and the post-`|turn|1` omniscient lines (the `|split|` pairs resolved to their
// omniscient half). Every set carries at most 510 EVs.
//
//   node src/rust_sim/harness/gen_b7_moves_golden.js -> tests/vectors/b7_moves_golden.txt
//   cargo test --test b7_moves_golden_test
// env: B7_SEEDS (rows per board, default 8), B7_OUT.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.B7_SEEDS || '8', 10);
const OUT = process.env.B7_OUT || path.resolve(__dirname, '../tests/vectors/b7_moves_golden.txt');

const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash'], { g: 'F' });
const BLISSEY = (moves, o = {}) => set('Blissey', '', 'NaturalCure', moves, { g: 'F', ...o });
// want(t) -> [p1, p2] for the t-th regular turn: 'move K' / 'switch K'. The driver answers a
// forced switch itself, plays a locked / Struggle request's only option, and falls back to the
// first enabled move when the wanted one is disabled (0 PP, Choice lock, Encore, Taunt).
const M = (a, b) => () => [`move ${a}`, `move ${b}`];
const seq = (rows) => (t) => rows[Math.min(t, rows.length - 1)].map((x) => (typeof x === 'number' ? `move ${x}` : x));
const FAST = { evs: '0,0,0,0,0,252' };
const SLOW = { evs: '0,0,0,0,0,0' };

const BOARDS = {
  // ---------------------------------------------------------------- STOCKPILE / SPIT UP / SWALLOW
  st_ladder: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), p2: BLISSEY(['splash']), turns: 6, want: seq([[1, 1], [1, 1], [1, 1], [1, 1], [2, 1]]) },
  st_zero: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), p2: BLISSEY(['splash']), turns: 3, want: seq([[2, 1], [3, 1], [1, 1]]) },
  st_swallow: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), p2: BLISSEY(['seismictoss']), turns: 9, want: seq([[1, 1], [3, 1], [1, 1], [1, 1], [3, 1], [1, 1], [1, 1], [1, 1], [3, 1]]) },
  st_swallow_full: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), p2: BLISSEY(['splash']), turns: 3, want: seq([[1, 1], [3, 1], [3, 1]]) },
  st_spitup: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], { evs: '252,0,252,0,4,0' }), turns: 7, want: seq([[1, 1], [2, 1], [1, 1], [1, 1], [2, 1], [1, 1]]) },
  st_spitup3: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: set('Snorlax', '', 'ThickFat', ['splash'], { evs: '252,0,252,0,4,0' }), turns: 4, want: seq([[1, 1], [1, 1], [1, 1], [2, 1]]) },
  st_ghost: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: set('Gengar', '', 'Levitate', ['splash']), turns: 3, want: seq([[1, 1], [2, 1], [2, 1]]) },
  st_protect: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: set('Gengar', '', 'Levitate', ['protect', 'splash']), turns: 3, want: seq([[1, 2], [2, 1], [2, 2]]) },
  st_evasion: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: BLISSEY(['doubleteam', 'splash']), turns: 9, want: seq([[1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [1, 1], [2, 2], [2, 2], [2, 2]]) },
  st_sub: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: BLISSEY(['substitute', 'splash']), turns: 3, want: seq([[1, 1], [2, 2], [2, 2]]) },
  st_switch: { p1: team(set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), BENCH), p2: BLISSEY(['splash']), turns: 5, want: seq([[1, 1], ['switch 2', 1], ['switch 2', 1], [2, 1]]) },
  st_batonpass: { p1: team(set('Snorlax', '', 'ThickFat', ['stockpile', 'batonpass']), set('Zangoose', '', 'Immunity', ['spitup', 'swallow'])), p2: BLISSEY(['splash']), turns: 4, want: seq([[1, 1], [2, 1], [1, 1], [2, 1]]) },
  st_snatch: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'swallow', 'spitup']), p2: set('Sableye', '', 'KeenEye', ['snatch', 'splash', 'stockpile', 'swallow']), turns: 5, want: seq([[1, 1], [2, 3], [2, 1], [1, 1], [3, 2]]) },
  st_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'rest', 'sleeptalk', 'spitup']), p2: BLISSEY(['seismictoss']), turns: 7, seeds: 16, want: seq([[1, 1], [1, 1], [2, 1], [3, 1]]) },
  st_sleeptalk_swallow: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'rest', 'sleeptalk', 'swallow']), p2: BLISSEY(['seismictoss']), turns: 7, seeds: 16, want: seq([[1, 1], [1, 1], [2, 1], [3, 1]]) },
  st_faint: { p1: team(set('Pichu', '', 'Static', ['stockpile', 'splash'], { lv: 5 }), set('Snorlax', '', 'ThickFat', ['spitup', 'splash'])), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 4, want: seq([[1, 2], [1, 1], [1, 2]]) },
  st_counter: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: set('Wobbuffet', '', 'ShadowTag', ['counter', 'splash'], { evs: '252,0,252,0,4,0' }), turns: 4, want: seq([[1, 2], [1, 2], [2, 1]]) },
  st_taunt: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup', 'swallow']), p2: set('Gengar', '', 'Levitate', ['taunt', 'splash']), turns: 4, want: seq([[1, 2], [1, 1], [2, 2], [3, 2]]) },
  st_wonderguard: { p1: set('Snorlax', '', 'ThickFat', ['stockpile', 'spitup']), p2: set('Shedinja', '', 'WonderGuard', ['splash']), turns: 3, want: seq([[1, 1], [2, 1], [2, 1]]) },
  st_ou: { format: 'gen3ou', p1: set('Snorlax', 'Leftovers', 'ThickFat', ['stockpile', 'spitup', 'swallow', 'bodyslam']), p2: set('Swampert', 'Leftovers', 'Torrent', ['earthquake', 'surf', 'toxic']), turns: 6, want: seq([[1, 3], [1, 1], [3, 2], [1, 1], [2, 1], [4, 2]]) },
  // ---------------------------------------------------------------- MUD SPORT / WATER SPORT
  sp_mud: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'thunderpunch', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'splash']), turns: 4, want: seq([[1, 2], [3, 1], [2, 1], [1, 1]]) },
  sp_water: { p1: set('Snorlax', '', 'ThickFat', ['watersport', 'firepunch', 'splash']), p2: set('Houndoom', '', 'EarlyBird', ['flamethrower', 'splash']), turns: 4, want: seq([[1, 2], [3, 1], [2, 1], [1, 1]]) },
  sp_both_same: { p1: set('Snorlax', '', 'ThickFat', ['watersport', 'mudsport', 'bodyslam']), p2: set('Houndoom', '', 'EarlyBird', ['tackle', 'flamethrower']), turns: 5, want: seq([[1, 1], [2, 2], [3, 1], [3, 2]]) },
  sp_mirror: { p1: set('Swampert', '', 'Torrent', ['mudsport', 'tackle'], SLOW), p2: set('Swampert', '', 'Torrent', ['mudsport', 'tackle'], SLOW), turns: 4, want: seq([[1, 1], [2, 2]]) },
  sp_quarter: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'mudsport']), turns: 4, want: seq([[1, 2], [2, 1]]) },
  sp_batonpass: { p1: team(set('Snorlax', '', 'ThickFat', ['mudsport', 'batonpass']), set('Gyarados', '', 'Intimidate', ['splash'])), p2: set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'splash']), turns: 4, want: seq([[1, 2], [2, 2], [1, 1]]) },
  sp_switch: { p1: team(set('Snorlax', '', 'ThickFat', ['mudsport', 'splash']), set('Gyarados', '', 'Intimidate', ['splash'])), p2: set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'splash']), turns: 4, want: seq([[1, 2], ['switch 2', 2], ['switch 2', 1], [2, 1]]) },
  sp_pressure: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport']), p2: set('Zapdos', '', 'Pressure', ['thunderbolt', 'splash']), turns: 3, want: seq([[1, 2], [2, 1], [2, 1]]) },
  sp_hiddenpower: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['hiddenpowerelectric', 'splash']), turns: 3, want: seq([[1, 2], [2, 1]]) },
  sp_confusion: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'splash']), p2: set('Gengar', '', 'Levitate', ['confuseray', 'splash']), turns: 6, seeds: 16, want: seq([[1, 2], [2, 1], [3, 2]]) },
  sp_beatup: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'splash']), p2: team(set('Sneasel', '', 'InnerFocus', ['beatup', 'splash']), BENCH), turns: 4, want: seq([[1, 2], [2, 2], [3, 1]]) },
  sp_hjk: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'protect']), p2: set('Hitmonlee', '', 'Limber', ['highjumpkick', 'splash']), turns: 4, want: seq([[1, 2], [2, 2], [3, 1], [1, 2]]) },
  sp_explosion: { p1: team(set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'explosion']), BENCH), p2: BLISSEY(['splash']), turns: 4, want: seq([[1, 1], [2, 1], [3, 1]]) },
  sp_weatherball: { p1: set('Snorlax', '', 'ThickFat', ['watersport', 'splash']), p2: set('Castform', '', 'Forecast', ['sunnyday', 'weatherball']), turns: 4, want: seq([[2, 1], [2, 2], [1, 2], [2, 2]]) },
  sp_charge: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'splash']), p2: set('Raichu', '', 'Static', ['charge', 'thunderbolt']), turns: 4, want: seq([[1, 1], [2, 2], [2, 1], [2, 2]]) },
  sp_blaze_thickfat: { p1: set('Snorlax', '', 'ThickFat', ['watersport', 'splash']), p2: set('Charizard', '', 'Blaze', ['bellydrum', 'substitute', 'flamethrower']), turns: 5, want: seq([[1, 1], [2, 2], [2, 3]]) },
  sp_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['rest', 'sleeptalk', 'mudsport', 'watersport']), p2: set('Jolteon', '', 'VoltAbsorb', ['thunderbolt', 'seismictoss']), turns: 5, seeds: 16, want: seq([[1, 2], [2, 2], [2, 1], [2, 1]]) },
  sp_futuresight: { p1: set('Snorlax', '', 'ThickFat', ['mudsport', 'watersport', 'splash']), p2: set('Alakazam', '', 'Synchronize', ['futuresight', 'splash']), turns: 5, want: seq([[1, 2], [2, 2], [3, 1], [3, 2]]) },
  sp_ou: { format: 'gen3ou', p1: set('Swampert', 'Leftovers', 'Torrent', ['mudsport', 'earthquake', 'surf', 'protect']), p2: set('Zapdos', 'Leftovers', 'Pressure', ['thunderbolt', 'hiddenpowerice', 'thunderwave', 'substitute']), turns: 5, want: seq([[1, 3], [2, 1], [3, 2], [4, 4], [2, 1]]) },
  // ---------------------------------------------------------------- MIST
  mi_basic: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'splash']), p2: set('Alakazam', '', 'Synchronize', ['growl', 'psychic', 'screech', 'splash']), turns: 8, want: seq([[1, 1], [2, 2], [2, 3], [2, 4], [2, 4], [2, 4], [2, 1], [2, 1]]) },
  mi_icywind: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'splash']), p2: set('Alakazam', '', 'Synchronize', ['icywind', 'splash']), turns: 3, want: seq([[1, 2], [2, 1]]) },
  mi_recast: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'splash']), p2: set('Alakazam', '', 'Synchronize', ['splash']), turns: 3, want: M(1, 1) },
  mi_intimidate: { p1: team(set('Lapras', '', 'ShellArmor', ['mist', 'splash']), BENCH), p2: team(BLISSEY(['splash']), set('Gyarados', '', 'Intimidate', ['splash'])), turns: 3, want: seq([[1, 1], [2, 'switch 2'], [2, 1]]) },
  mi_selfdrop: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'overheat', 'superpower', 'curse']), p2: BLISSEY(['splash']), turns: 4, want: seq([[1, 1], [2, 1], [3, 1], [4, 1]]) },
  mi_haze: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'haze', 'swordsdance']), p2: BLISSEY(['splash', 'haze', 'growl']), turns: 5, want: seq([[1, 1], [3, 1], [2, 2], [1, 3], [3, 3]]) },
  mi_tickle: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'splash']), p2: team(BLISSEY(['tickle', 'charm', 'memento']), BENCH), turns: 5, want: seq([[1, 1], [2, 1], [2, 2], [2, 3], [2, 1]]) },
  mi_clearbody: { p1: set('Metagross', '', 'ClearBody', ['mist', 'splash']), p2: BLISSEY(['growl', 'tickle']), turns: 3, want: seq([[1, 1], [2, 1], [2, 2]]) },
  mi_hypercutter: { p1: team(set('Kingler', '', 'HyperCutter', ['mist', 'splash']), BENCH), p2: team(BLISSEY(['tickle', 'splash']), set('Gyarados', '', 'Intimidate', ['splash'])), turns: 3, want: seq([[1, 2], [2, 1], [2, 'switch 2']]) },
  mi_keeneye: { p1: set('Pidgeot', '', 'KeenEye', ['mist', 'splash']), p2: BLISSEY(['sandattack', 'growl', 'flash']), turns: 4, want: seq([[1, 1], [2, 1], [2, 2], [2, 3]]) },
  mi_both: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'splash']), p2: set('Articuno', '', 'Pressure', ['mist', 'splash']), turns: 7, seeds: 16, want: seq([[1, 1], [2, 2]]) },
  mi_screens: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'reflect', 'safeguard', 'splash']), p2: set('Articuno', '', 'Pressure', ['mist', 'lightscreen', 'splash']), turns: 8, seeds: 16, want: seq([[1, 1], [2, 2], [3, 3], [4, 3]]) },
  mi_screens_b: { p1: set('Lapras', '', 'ShellArmor', ['reflect', 'mist', 'safeguard', 'splash']), p2: set('Articuno', '', 'Pressure', ['lightscreen', 'mist', 'reflect', 'splash']), turns: 8, seeds: 16, want: seq([[1, 1], [2, 2], [3, 3], [4, 4]]) },
  mi_snatch: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'splash']), p2: set('Sableye', '', 'KeenEye', ['snatch', 'growl']), turns: 3, want: seq([[1, 1], [2, 2], [1, 2]]) },
  mi_sub: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'substitute']), p2: set('Sableye', '', 'KeenEye', ['tickle', 'sandattack', 'splash']), turns: 4, want: seq([[1, 3], [2, 1], [2, 2]]) },
  mi_floor: { p1: team(set('Lapras', '', 'ShellArmor', ['mist', 'splash']), BENCH), p2: team(BLISSEY(['charm', 'tickle']), set('Gyarados', '', 'Intimidate', ['splash'])), turns: 6, want: seq([[2, 1], [2, 1], [2, 1], [1, 2], [2, 'switch 2'], [2, 1]]) },
  mi_switch: { p1: team(set('Lapras', '', 'ShellArmor', ['mist', 'splash']), BENCH), p2: BLISSEY(['growl']), turns: 7, want: seq([[1, 1], ['switch 2', 1], ['switch 2', 1], [2, 1]]) },
  mi_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['rest', 'sleeptalk', 'mist', 'mudsport']), p2: BLISSEY(['seismictoss', 'growl']), turns: 5, seeds: 16, want: seq([[1, 1], [2, 2], [2, 1], [2, 2]]) },
  mi_crunch: { p1: set('Lapras', '', 'ShellArmor', ['mist', 'splash']), p2: set('Tyranitar', '', 'SandStream', ['crunch', 'splash']), turns: 4, want: seq([[1, 2], [2, 1]]) },
  mi_ou: { format: 'gen3ou', p1: set('Lapras', 'Leftovers', 'WaterAbsorb', ['mist', 'icebeam', 'thunderbolt', 'rest']), p2: set('Gyarados', 'Leftovers', 'Intimidate', ['dragondance', 'hiddenpowerflying', 'earthquake', 'taunt']), turns: 5, want: seq([[1, 1], [2, 2], [3, 3], [2, 1], [3, 2]]) },
  // ---------------------------------------------------------------- LOCK-ON / MIND READER
  lo_zap: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'mindreader', 'splash']), p2: BLISSEY(['splash']), turns: 4, want: seq([[1, 1], [2, 1], [2, 1], [3, 1]]) },
  lo_recast: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'mindreader', 'splash']), p2: BLISSEY(['splash']), turns: 5, want: seq([[1, 1], [1, 1], [3, 1], [4, 1], [3, 1]]) },
  lo_lapse: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'mindreader', 'splash']), p2: BLISSEY(['splash']), turns: 3, want: seq([[1, 1], [4, 1], [2, 1]]) },
  lo_protect_sub: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'splash']), p2: BLISSEY(['protect', 'substitute', 'splash']), turns: 5, want: seq([[1, 1], [3, 2], [1, 3], [1, 1], [2, 3]]) },
  lo_locked_protect: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'splash']), p2: BLISSEY(['splash', 'protect']), turns: 3, want: seq([[1, 1], [1, 2], [2, 1]]) },
  lo_dig: { p1: set('Porygon2', '', 'Trace', ['lockon', 'icebeam', 'zapcannon']), p2: set('Golem', '', 'Sturdy', ['dig', 'splash']), turns: 4, want: seq([[1, 1], [2, 1], [1, 1], [3, 1]]) },
  lo_fly: { p1: set('Porygon2', '', 'Trace', ['lockon', 'swift', 'splash'], FAST), p2: set('Skarmory', '', 'KeenEye', ['fly', 'splash'], SLOW), turns: 4, want: seq([[1, 1], [2, 1], [3, 2], [1, 1]]) },
  lo_fly_miss: { p1: set('Porygon2', '', 'Trace', ['lockon', 'thunderbolt', 'splash'], SLOW), p2: set('Swellow', '', 'Guts', ['fly', 'splash'], FAST), turns: 3, want: seq([[1, 1], [1, 1]]) },
  lo_switch: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'splash']), p2: team(BLISSEY(['splash']), set('Snorlax', '', 'ThickFat', ['splash'])), turns: 4, want: seq([[1, 1], [1, 'switch 2'], [2, 1], [1, 'switch 2']]) },
  lo_switch_back: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'splash']), p2: team(BLISSEY(['splash']), set('Snorlax', '', 'ThickFat', ['splash'])), turns: 3, want: seq([[1, 1], [3, 'switch 2'], [2, 'switch 2']]) },
  lo_holder_switch: { p1: team(set('Porygon2', '', 'Trace', ['lockon', 'zapcannon']), set('Regice', '', 'ClearBody', ['zapcannon', 'lockon'])), p2: BLISSEY(['splash']), turns: 4, want: seq([[1, 1], ['switch 2', 1], ['switch 2', 1], [2, 1]]) },
  // Equal-speed Regice (Clear Body has no switch-in line, so the lead framing carries no
  // construction speed-tie attribution): the two locks tie at the residual.
  lo_batonpass: { p1: team(set('Regice', '', 'ClearBody', ['lockon', 'batonpass']), set('Regice', '', 'ClearBody', ['splash', 'zapcannon'])), p2: set('Regice', '', 'ClearBody', ['lockon', 'splash']), turns: 4, want: seq([[1, 1], [2, 2], [2, 2]]) },
  lo_mirror: { p1: set('Regice', '', 'ClearBody', ['lockon', 'splash']), p2: set('Regice', '', 'ClearBody', ['lockon', 'splash']), turns: 4, want: seq([[1, 1], [2, 2]]) },
  lo_evasion: { p1: set('Porygon2', '', 'Trace', ['lockon', 'zapcannon', 'mindreader', 'splash']), p2: BLISSEY(['doubleteam', 'splash']), turns: 10, want: seq([[4, 1], [4, 1], [4, 1], [4, 1], [4, 1], [4, 1], [3, 2], [2, 2], [1, 2], [2, 2]]) },
  lo_hypnosis: { p1: set('Porygon2', '', 'Trace', ['lockon', 'hypnosis', 'splash']), p2: BLISSEY(['splash']), turns: 3, want: seq([[1, 1], [2, 1]]) },
  lo_futuresight: { p1: set('Porygon2', '', 'Trace', ['futuresight', 'lockon', 'splash']), p2: BLISSEY(['splash']), turns: 4, want: seq([[1, 1], [3, 1], [2, 1]]) },
  lo_perish: { p1: set('Porygon2', '', 'Trace', ['lockon', 'perishsong', 'splash'], FAST), p2: set('Swellow', '', 'Guts', ['fly', 'splash'], { evs: '0,0,0,0,0,0', lv: 50 }), turns: 3, want: seq([[1, 1], [2, 1], [3, 2]]) },
  lo_pressure: { p1: set('Porygon2', '', 'Trace', ['lockon', 'mindreader', 'splash']), p2: set('Zapdos', '', 'Pressure', ['splash']), turns: 4, want: seq([[1, 1], [3, 1], [2, 1]]) },
  lo_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['rest', 'sleeptalk', 'lockon', 'mindreader']), p2: BLISSEY(['seismictoss', 'splash']), turns: 5, seeds: 16, want: seq([[1, 1], [2, 2]]) },
  lo_rockblast: { p1: set('Golem', '', 'Sturdy', ['lockon', 'rockblast', 'dynamicpunch']), p2: BLISSEY(['doubleteam', 'splash']), turns: 6, want: seq([[1, 1], [1, 1], [1, 2], [2, 2], [1, 2], [3, 2]]) },
  lo_ou: { format: 'gen3ou', p1: set('Regice', 'Leftovers', 'ClearBody', ['lockon', 'zapcannon', 'icebeam', 'rest']), p2: set('Skarmory', 'Leftovers', 'KeenEye', ['spikes', 'toxic', 'drillpeck', 'protect']), turns: 5, want: seq([[1, 1], [2, 2], [3, 4], [1, 3], [2, 3]]) },
};

function answer(side, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) return `switch ${side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1}`;
  const moves = req.active[0].moves;
  if (want.startsWith('switch') && !req.active[0].trapped) return want;
  if (moves.length === 1 || want.startsWith('switch')) return 'move 1';
  const k = parseInt(want.split(' ')[1], 10);
  if (moves[k - 1] && !moves[k - 1].disabled) return want;
  const first = moves.findIndex((m) => !m.disabled);
  return first < 0 ? 'move 1' : `move ${first + 1}`;
}

function run(board, seed) {
  const battle = new Battle({ formatid: board.format || 'gen3customgame', seed });
  battle.setPlayer('p1', { name: 'P1', team: board.p1 });
  battle.setPlayer('p2', { name: 'P2', team: board.p2 });
  for (const [i, s] of [board.p1, board.p2].entries()) {
    const want = s.split(']').map((x) => x.split('|')[1]);
    const got = battle.sides[i].pokemon.map((p) => p.species.name);
    if (JSON.stringify(want) !== JSON.stringify(got)) throw new Error(`team ${i} unpacked as ${got}`);
  }
  const seedBefore = battle.prng.getSeed();
  const script = [];
  let t = 0;
  while (!battle.ended && t < board.turns) {
    const forced = battle.sides.some((s) => s.activeRequest && s.activeRequest.forceSwitch);
    const [w1, w2] = board.want(t);
    const d = [answer(battle.sides[0], w1), answer(battle.sides[1], w2)];
    if (d[0] === null && d[1] === null) break;
    if (d[0] !== null && !battle.choose('p1', d[0])) throw new Error(`p1 choice ${d[0]} rejected`);
    if (d[1] !== null && !battle.choose('p2', d[1])) throw new Error(`p2 choice ${d[1]} rejected`);
    script.push(d);
    if (!forced) t++;
  }
  const omni = [];
  for (let i = 0; i < battle.log.length; i++) {
    const l = battle.log[i];
    if (l.startsWith('|split|')) { omni.push(battle.log[i + 1]); i += 2; continue; }
    omni.push(l);
  }
  const drop = (l) => l === '' || l.startsWith('|t:|') || l.startsWith('|debug|') || l.startsWith('|error|');
  const g = omni.indexOf('|gametype|singles');
  const t1 = omni.indexOf('|turn|1');
  return {
    seedBefore, seedAfter: battle.prng.getSeed(), script,
    framing: omni.slice(g, t1 + 1).filter((l) => !drop(l)),
    lines: omni.slice(t1 + 1).filter((l) => !drop(l)),
  };
}

const rows = [];
for (const [name, b] of Object.entries(BOARDS)) {
  for (let k = 0; k < (b.seeds || N); k++) {
    const seed = [31 + k * 7, 47 + k * 13, 61 + k * 29, 73 + k * 51];
    rows.push({ name, format: b.format || 'gen3customgame', p1: b.p1, p2: b.p2, ...run(b, seed) });
  }
}
const n = (pred) => rows.filter(pred).length;
const has = (r, s) => r.lines.some((l) => l.includes(s));
const pair = (r, a, b) => r.lines.some((l, i) => a(l) && b(r.lines[i + 1] || ''));
const isMove = (l, name) => l.startsWith('|move|') && l.split('|')[3] === name;
const cov = {
  stockpileStart: n((r) => has(r, '|stockpile1')),
  stockpile3: n((r) => has(r, '|stockpile3')),
  stockpileCapBare: n((r) => pair(r, (l) => isMove(l, 'Stockpile'), (b) => !b.startsWith('|-start|') && !b.startsWith('|-activate|'))),
  spitUpHit: n((r) => pair(r, (l) => isMove(l, 'Spit Up'), (b) => /^\|-(damage|crit|supereffective|resisted)\|/.test(b))),
  spitUpBare: n((r) => pair(r, (l) => isMove(l, 'Spit Up'), (b) => !/^\|-(damage|crit|supereffective|resisted|immune|activate|miss)\|/.test(b))),
  stockpileEnd: n((r) => has(r, '|Stockpile') && r.lines.some((l) => l.startsWith('|-end|') && l.endsWith('|Stockpile'))),
  spitUpMiss: n((r) => r.lines.some((l) => isMove(l, 'Spit Up') && l.endsWith('[miss]'))),
  spitUpGhost: n((r) => r.name === 'st_ghost' && has(r, '|-immune|p2a: Gengar')),
  spitUpProtect: n((r) => r.name === 'st_protect' && has(r, '|-activate|p2a: Gengar|Protect')),
  spitUpWonderGuard: n((r) => r.name === 'st_wonderguard' && has(r, 'ability: Wonder Guard')),
  spitUpCounter: n((r) => r.name === 'st_counter' && r.lines.some((l) => isMove(l, 'Counter')) && has(r, '|-damage|p1a: Snorlax')),
  swallowHeal: n((r) => pair(r, (l) => isMove(l, 'Swallow'), (b) => b.startsWith('|-heal|'))),
  swallowFull: n((r) => has(r, '|-fail|') && r.lines.some((l) => l.startsWith('|-fail|') && l.endsWith('|heal'))),
  snatchStockpile: n((r) => has(r, '|Stockpile|p2a: Sableye|[from] Snatch')),
  snatchSwallow: n((r) => has(r, '|Swallow|p2a: Sableye|[from] Snatch')),
  sleepTalkSpitUp: n((r) => r.lines.some((l) => isMove(l, 'Spit Up') && l.includes('[from] Sleep Talk'))),
  sleepTalkSwallow: n((r) => r.lines.some((l) => isMove(l, 'Swallow') && l.includes('[from] Sleep Talk'))),
  mudStart: n((r) => has(r, '|-start|') && has(r, '|Mud Sport')),
  waterStart: n((r) => has(r, '|move: Water Sport')),
  sportFail: n((r) => r.lines.some((l, i) => /\|(Mud|Water) Sport\|\|\[still\]$/.test(l) && (r.lines[i + 1] || '').startsWith('|-fail|'))),
  sportBatonPass: n((r) => r.name === 'sp_batonpass' && has(r, '[from] Baton Pass')),
  sportConfusion: n((r) => r.name === 'sp_confusion' && has(r, '[from] confusion')),
  sportSleepTalk: n((r) => r.lines.some((l) => /\|(Mud|Water) Sport\|/.test(l) && l.includes('[from] Sleep Talk'))),
  mistStart: n((r) => r.lines.some((l) => l.startsWith('|-sidestart|') && l.endsWith('|Mist'))),
  mistActivate: n((r) => has(r, '|move: Mist')),
  mistEnd: n((r) => r.lines.some((l) => l.startsWith('|-sideend|') && l.endsWith('|Mist'))),
  mistFail: n((r) => r.lines.some((l, i) => /\|Mist\|\|\[still\]$/.test(l) && (r.lines[i + 1] || '').startsWith('|-fail|'))),
  mistIntimidate: n((r) => pair(r, (l) => l.endsWith('|Intimidate|boost'), (b) => b.endsWith('|move: Mist'))),
  mistFloor: n((r) => pair(r, (l) => l.endsWith('|move: Mist'), (b) => b.startsWith('|-unboost|') && b.endsWith('|0'))),
  mistClearBody: n((r) => r.name === 'mi_clearbody' && has(r, 'ability: Clear Body')),
  mistBothEnd: n((r) => pair(r, (l) => l.startsWith('|-sideend|') && l.endsWith('|Mist'), (b) => b.startsWith('|-sideend|') && b.endsWith('|Mist'))),
  mistSnatch: n((r) => has(r, '|Mist|p2a: Sableye|[from] Snatch')),
  lockOn: n((r) => has(r, '|move: Lock-On|[of]')),
  mindReader: n((r) => has(r, '|move: Mind Reader|[of]')),
  lockOnFail: n((r) => r.lines.some((l, i) => /\|(Lock-On|Mind Reader)\|\|\[still\]$/.test(l) && (r.lines[i + 1] || '').startsWith('|-fail|'))),
  lockOnMiss: n((r) => r.lines.some((l) => (isMove(l, 'Lock-On') || isMove(l, 'Mind Reader')) && l.endsWith('[miss]'))),
  lockOnDig: n((r) => r.name === 'lo_dig' && r.lines.some((l, i) => isMove(l, 'Ice Beam') && r.lines.slice(i + 1, i + 4).some((b) => b.startsWith('|-damage|p2a: Golem')) && r.lines.slice(i + 1).some((b) => b.includes('|Dig|') && b.includes('[from] lockedmove')))),
  lockOnFly: n((r) => r.name === 'lo_fly' && r.lines.some((l, i) => isMove(l, 'Swift') && r.lines.slice(i + 1, i + 4).some((b) => b.startsWith('|-damage|p2a: Skarmory')) && r.lines.slice(i + 1).some((b) => b.includes('|Fly|') && b.includes('[from] lockedmove')))),
  lockOnPerish: n((r) => r.name === 'lo_perish' && has(r, '|-start|p2a: Swellow|perish3|[silent]')),
  lockOnBatonPass: n((r) => r.name === 'lo_batonpass' && has(r, '[from] Baton Pass')),
  lockOnSleepTalk: n((r) => r.lines.some((l) => (isMove(l, 'Lock-On') || isMove(l, 'Mind Reader')) && l.includes('[from] Sleep Talk'))),
  ouRows: n((r) => r.format === 'gen3ou'),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0 && !process.env.B7_NOFLOOR) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
for (const r of rows) for (const sset of [r.p1, r.p2]) for (const m of sset.split(']')) {
  const evs = m.split('|')[6].split(',').reduce((a, x) => a + (parseInt(x, 10) || 0), 0);
  if (evs > 510) { console.error(`REFUSING: ${r.name} carries a set over 510 EVs`); process.exit(1); }
}
const header = [
  '# B7 MOVES GOLDEN (gen3_b7_moves_v1): Stockpile, Spit Up, Swallow, Mud Sport, Water Sport, Mist, Lock-On, Mind Reader',
  `# generated by harness/gen_b7_moves_golden.js  rows/board=${N}  boards=${Object.keys(BOARDS).length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the format, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE /',
  '# SEED_AFTER, the FRAMING window and the post-|turn|1 omniscient lines (|t:| dropped).',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
