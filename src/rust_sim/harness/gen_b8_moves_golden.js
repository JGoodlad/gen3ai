// GOLDEN GENERATOR — the B8 moves (`gen3_b8_moves_v1`, gen3 full parity B8): Bide, Magnitude, Psywave,
// Present, Triple Kick and the OHKO moves (Fissure, Horn Drill, Guillotine, Sheer Cold). The settled
// spec is `harness/probe_b8_moves.js`'s header. Each board runs over N seeds; each row records the
// format, the decisions as played ([p1, p2], null = no choice), the sim SEED_BEFORE / SEED_AFTER, the
// FRAMING window and the post-`|turn|1` omniscient lines (the `|split|` pairs resolved to their
// omniscient half). The lines are captured AS SENT: each decision's new lines are frozen when the sim
// streams them, so an `attrLastMove` that retro-edits an already-streamed line (a Bide release that
// misses with no `|move|` line this turn) is recorded the way every client saw it — the port's
// `ProtocolBuilder` send boundary models the same thing.
//
//   node src/rust_sim/harness/gen_b8_moves_golden.js -> tests/vectors/b8_moves_golden.txt
//   cargo test --test b8_moves_golden_test
// env: B8_SEEDS (rows per board, default 8), B8_OUT.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.B8_SEEDS || '8', 10);
const OUT = process.env.B8_OUT || path.resolve(__dirname, '../tests/vectors/b8_moves_golden.txt');

const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash'], { g: 'F' });
const R = (moves, o = {}) => set('Raichu', o.item || '', 'Static', moves, o);
// want(t) -> [p1, p2] for the t-th regular turn: 'move K' / 'switch K'. The driver answers a
// forced switch itself, plays a locked request's only option, and falls back to the first
// enabled move when the wanted one is disabled (0 PP, Choice lock, Encore, Disable).
const M = (a, b) => () => [`move ${a}`, `move ${b}`];
const seq = (rows) => (t) => rows[Math.min(t, rows.length - 1)].map((x) => (typeof x === 'number' ? `move ${x}` : x));
const SNORLAX = (moves, o) => set('Snorlax', (o && o.item) || '', 'ThickFat', moves, o);

const BOARDS = {
  // ---------------------------------------------------------------- BIDE
  bd_basic: { p1: R(['bide', 'splash']), p2: SNORLAX(['bodyslam', 'splash']), turns: 5, want: seq([[1, 1], [1, 1], [1, 2], [2, 2], [1, 1]]) },
  bd_nothing: { p1: R(['bide', 'splash']), p2: SNORLAX(['bodyslam', 'splash']), turns: 3, want: M(1, 2) },
  bd_slow_user: { p1: set('Shuckle', 'Leftovers', 'Sturdy', ['bide', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['tackle', 'splash']), turns: 4, want: M(1, 1) },
  bd_ghost: { p1: R(['bide', 'splash']), p2: set('Gengar', '', 'Levitate', ['thunderbolt', 'splash']), turns: 3, want: seq([[1, 2], [1, 1], [1, 2]]) },
  bd_switched: { p1: R(['bide', 'splash']), p2: team(SNORLAX(['bodyslam', 'splash']), BENCH), turns: 3, want: seq([[1, 1], [1, 'switch 2'], [1, 1]]) },
  bd_multihit: { p1: set('Shuckle', '', 'Sturdy', ['bide', 'splash']), p2: set('Cloyster', '', 'ShellArmor', ['spikecannon', 'splash']), turns: 4, want: seq([[1, 1], [1, 1], [1, 2]]) },
  bd_futuresight: { p1: SNORLAX(['bide', 'splash']), p2: set('Alakazam', '', 'Synchronize', ['futuresight', 'splash']), turns: 5, want: seq([[2, 1], [2, 2], [1, 2]]) },
  bd_ownsub: { p1: R(['bide', 'substitute']), p2: SNORLAX(['bodyslam', 'splash']), turns: 4, want: seq([[2, 2], [1, 1]]) },
  bd_protect: { p1: R(['bide', 'splash']), p2: SNORLAX(['seismictoss', 'protect']), turns: 3, want: seq([[1, 1], [1, 1], [1, 2]]) },
  bd_foesub: { p1: set('Shuckle', '', 'Sturdy', ['bide', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['tackle', 'substitute']), turns: 3, want: seq([[1, 2], [1, 1], [1, 1]]) },
  bd_wonderguard: { p1: R(['bide', 'splash']), p2: team(SNORLAX(['bodyslam', 'splash']), set('Shedinja', '', 'WonderGuard', ['splash'])), turns: 3, want: seq([[1, 1], [1, 'switch 2'], [1, 1]]) },
  bd_roughskin: { p1: R(['bide', 'splash']), p2: set('Sharpedo', '', 'RoughSkin', ['crunch', 'splash']), turns: 3, want: seq([[1, 1], [1, 1], [1, 2]]) },
  bd_fly: { p1: set('Shuckle', '', 'Sturdy', ['bide', 'splash']), p2: set('Pidgeot', '', 'KeenEye', ['fly', 'quickattack']), turns: 4, want: seq([[1, 2], [1, 2], [1, 1]]) },
  bd_sleep: { p1: SNORLAX(['bide', 'splash']), p2: set('Breloom', '', 'EffectSpore', ['spore', 'tackle']), turns: 3, want: seq([[1, 2], [1, 1], [2, 2]]) },
  bd_confusion: { p1: R(['bide', 'splash']), p2: SNORLAX(['confuseray', 'bodyslam']), turns: 3, seeds: 16, want: seq([[1, 1], [1, 2]]) },
  bd_para: { p1: set('Swampert', '', 'Torrent', ['bide', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['thunderwave', 'tackle']), turns: 4, seeds: 16, want: seq([[1, 1], [1, 2]]) },
  bd_disable: { p1: SNORLAX(['bide', 'splash']), p2: set('Gengar', '', 'Levitate', ['disable', 'shadowball']), turns: 3, want: seq([[1, 2], [1, 1], [1, 2]]) },
  bd_imprison: { p1: R(['bide', 'splash']), p2: set('Smeargle', '', 'OwnTempo', ['tackle', 'imprison', 'bide']), turns: 3, want: seq([[1, 1], [1, 2], [1, 1]]) },
  bd_roar: { p1: team(R(['bide', 'splash']), SNORLAX(['splash'])), p2: SNORLAX(['bodyslam', 'roar']), turns: 3, want: seq([[1, 1], [1, 2], [1, 1]]) },
  bd_pressure: { p1: R(['bide', 'splash']), p2: set('Zapdos', '', 'Pressure', ['drillpeck', 'splash']), turns: 3, want: seq([[1, 1], [1, 1], [1, 2]]) },
  bd_shellbell: { p1: R(['bide', 'splash'], { item: 'Shell Bell' }), p2: SNORLAX(['seismictoss', 'splash'], { lv: 30 }), turns: 4, want: seq([[1, 1], [1, 1], [1, 2]]) },
  bd_evasion: { p1: SNORLAX(['bide', 'doubleteam']), p2: set('Jolteon', '', 'VoltAbsorb', ['tackle', 'splash']), turns: 9, want: seq([[2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [2, 2], [1, 1]]) },
  bd_kingsrock: { p1: R(['bide', 'splash'], { item: 'Kings Rock' }), p2: SNORLAX(['bodyslam', 'splash']), turns: 3, seeds: 24, want: seq([[1, 2], [1, 1], [1, 2]]) },
  bd_mirror: { p1: SNORLAX(['bide', 'tackle']), p2: SNORLAX(['bide', 'tackle']), turns: 4, want: seq([[1, 2], [1, 2], [1, 2], [2, 2]]) },
  bd_faint_skip: { p1: set('Shuckle', '', 'Sturdy', ['bide', 'splash']), p2: team(set('Electrode', '', 'Soundproof', ['splash', 'explosion']), BENCH), turns: 4, want: seq([[1, 1], [1, 1], [1, 2], [1, 1]]) },
  bd_encore: { p1: R(['bide', 'splash']), p2: set('Smeargle', '', 'OwnTempo', ['encore', 'tackle']), turns: 6, want: seq([[1, 1], [1, 2], [1, 2], [2, 2]]) },
  bd_choiceband: { p1: R(['bide', 'thunderbolt'], { item: 'Choice Band' }), p2: SNORLAX(['seismictoss', 'splash'], { lv: 50 }), turns: 5, want: seq([[1, 1], [1, 1], [1, 2], [2, 2]]) },
  bd_counter: { p1: R(['bide', 'splash']), p2: SNORLAX(['seismictoss', 'counter'], { lv: 60 }), turns: 3, want: seq([[1, 1], [1, 1], [1, 2]]) },
  bd_endure: { p1: set('Shuckle', '', 'Sturdy', ['bide', 'endure'], { lv: 20 }), p2: set('Machamp', '', 'Guts', ['crosschop', 'splash']), turns: 3, want: seq([[2, 1], [1, 1]]) },
  // ---------------------------------------------------------------- MAGNITUDE
  mg_basic: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: SNORLAX(['splash'], { item: 'Leftovers' }), turns: 4, seeds: 24, want: M(1, 1) },
  mg_flying: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: set('Skarmory', '', 'KeenEye', ['splash']), turns: 2, want: M(1, 1) },
  mg_levitate: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: set('Gengar', '', 'Levitate', ['splash']), turns: 2, want: M(1, 1) },
  mg_dig: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: set('Dugtrio', '', 'ArenaTrap', ['dig', 'splash']), turns: 2, want: M(1, 1) },
  mg_fly: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: set('Pidgeot', '', 'KeenEye', ['fly', 'splash']), turns: 2, want: M(1, 1) },
  mg_protect: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: SNORLAX(['protect', 'splash']), turns: 2, want: seq([[1, 1], [1, 2]]) },
  mg_sub: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), turns: 3, want: M(1, 1) },
  mg_miss: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'splash']), p2: SNORLAX(['doubleteam'], { item: 'Bright Powder' }), turns: 9, want: seq([[2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [2, 1], [1, 1]]) },
  mg_sleeptalk: { p1: set('Golem', '', 'Sturdy', ['magnitude', 'sleeptalk', 'rest']), p2: SNORLAX(['bodyslam']), turns: 4, want: seq([[1, 1], [3, 1], [2, 1]]) },
  mg_kingsrock: { p1: set('Golem', 'Kings Rock', 'Sturdy', ['magnitude', 'splash']), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  // ---------------------------------------------------------------- PSYWAVE
  pw_l100: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash']), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  pw_l50: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash'], { lv: 50 }), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  pw_l37: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash'], { lv: 37 }), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  pw_l5: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash'], { lv: 5 }), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  pw_l1: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash'], { lv: 1 }), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  pw_dark: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash']), p2: set('Umbreon', '', 'Synchronize', ['splash']), turns: 2, want: M(1, 1) },
  pw_wonderguard: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash']), p2: set('Shedinja', '', 'WonderGuard', ['splash']), turns: 2, want: M(1, 1) },
  pw_sub: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), turns: 4, want: seq([[2, 1], [1, 2]]) },
  pw_protect: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash']), p2: SNORLAX(['protect', 'splash']), turns: 2, want: seq([[1, 1], [1, 2]]) },
  pw_kingsrock: { p1: set('Gengar', 'Kings Rock', 'Levitate', ['psywave', 'splash']), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  pw_ko: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash']), p2: team(set('Pichu', '', 'Static', ['splash'], { lv: 5 }), BENCH), turns: 2, want: M(1, 1) },
  pw_focusband: { p1: set('Gengar', '', 'Levitate', ['psywave', 'splash']), p2: team(set('Pichu', 'Focus Band', 'Static', ['splash'], { lv: 5 }), BENCH), turns: 3, seeds: 16, want: M(1, 1) },
  // ---------------------------------------------------------------- PRESENT
  pr_basic: { p1: set('Delibird', '', 'Hustle', ['present', 'splash']), p2: SNORLAX(['splash', 'bellydrum']), turns: 4, seeds: 24, want: seq([[1, 2], [1, 1]]) },
  pr_fullhp: { p1: set('Delibird', '', 'Hustle', ['present', 'splash']), p2: SNORLAX(['splash'], { item: 'Leftovers' }), turns: 3, seeds: 16, want: M(1, 1) },
  pr_sub: { p1: set('Delibird', '', 'Hustle', ['present', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), turns: 4, seeds: 16, want: seq([[2, 1], [1, 2]]) },
  pr_ghost: { p1: set('Delibird', '', 'Hustle', ['present', 'splash']), p2: set('Gengar', '', 'Levitate', ['splash']), turns: 3, want: M(1, 1) },
  pr_wonderguard: { p1: set('Delibird', '', 'Hustle', ['present', 'splash']), p2: set('Shedinja', '', 'WonderGuard', ['splash']), turns: 3, want: M(1, 1) },
  pr_protect: { p1: set('Delibird', '', 'Hustle', ['present', 'splash']), p2: SNORLAX(['protect', 'splash']), turns: 2, want: seq([[1, 1], [1, 2]]) },
  pr_fly: { p1: set('Delibird', '', 'Hustle', ['present', 'splash']), p2: set('Pidgeot', '', 'KeenEye', ['fly', 'splash']), turns: 2, want: M(1, 1) },
  // ---------------------------------------------------------------- TRIPLE KICK
  tk_evasion: { p1: set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), p2: SNORLAX(['doubleteam']), turns: 5, seeds: 16, want: M(1, 1) },
  tk_ghost: { p1: set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), p2: set('Gengar', '', 'Levitate', ['splash']), turns: 2, want: M(1, 1) },
  tk_sub: { p1: set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash']), turns: 3, want: seq([[2, 1], [1, 2]]) },
  tk_ko: { p1: set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), p2: team(set('Pichu', '', 'Static', ['splash'], { lv: 5 }), BENCH), turns: 2, want: M(1, 1) },
  tk_kingsrock: { p1: set('Hitmontop', 'Kings Rock', 'Intimidate', ['triplekick', 'splash']), p2: SNORLAX(['splash']), turns: 3, want: M(1, 1) },
  tk_brightpowder: { p1: set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), p2: SNORLAX(['splash'], { item: 'Bright Powder' }), turns: 4, want: M(1, 1) },
  tk_roughskin: { p1: set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), p2: set('Sharpedo', '', 'RoughSkin', ['splash']), turns: 2, want: M(1, 1) },
  tk_protect: { p1: set('Hitmontop', '', 'Intimidate', ['triplekick', 'splash']), p2: SNORLAX(['protect', 'splash']), turns: 2, want: seq([[1, 1], [1, 2]]) },
  // ---------------------------------------------------------------- OHKO
  oh_equal: { p1: set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash']), p2: team(SNORLAX(['splash']), BENCH), turns: 4, want: M(1, 1) },
  oh_lower: { p1: set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash'], { lv: 50 }), p2: SNORLAX(['splash']), turns: 2, want: M(1, 1) },
  oh_higher: { p1: set('Rhydon', '', 'LightningRod', ['horndrill', 'splash']), p2: team(SNORLAX(['splash'], { lv: 40 }), BENCH), turns: 3, want: M(1, 1) },
  oh_ghost: { p1: set('Rhydon', '', 'LightningRod', ['horndrill', 'splash']), p2: set('Gengar', '', 'Levitate', ['splash']), turns: 2, want: M(1, 1) },
  oh_ghost_lower: { p1: set('Rhydon', '', 'LightningRod', ['horndrill', 'splash'], { lv: 50 }), p2: set('Gengar', '', 'Levitate', ['splash']), turns: 2, want: M(1, 1) },
  oh_flying: { p1: set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash']), p2: set('Skarmory', '', 'KeenEye', ['splash']), turns: 2, want: M(1, 1) },
  oh_flying_lower: { p1: set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash'], { lv: 50 }), p2: set('Skarmory', '', 'KeenEye', ['splash']), turns: 2, want: M(1, 1) },
  oh_sturdy: { p1: set('Lapras', '', 'ShellArmor', ['sheercold', 'splash']), p2: set('Golem', '', 'Sturdy', ['splash'], { lv: 40 }), turns: 3, want: M(1, 1) },
  oh_sturdy_lower: { p1: set('Lapras', '', 'ShellArmor', ['sheercold', 'splash'], { lv: 50 }), p2: set('Golem', '', 'Sturdy', ['splash']), turns: 2, want: M(1, 1) },
  oh_sub: { p1: set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['substitute', 'splash'], { lv: 1 }), turns: 4, want: seq([[2, 1], [1, 2]]) },
  oh_sub_held: { p1: set('Kingler', '', 'HyperCutter', ['guillotine', 'toxic', 'splash']), p2: SNORLAX(['substitute', 'bellydrum', 'splash'], { lv: 1 }), turns: 5, want: seq([[2, 1], [3, 2], [1, 3]]) },
  oh_endure: { p1: set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), p2: SNORLAX(['endure', 'splash'], { lv: 1 }), turns: 3, want: seq([[1, 1], [1, 2]]) },
  oh_focusband: { p1: set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), p2: team(SNORLAX(['splash'], { lv: 1, item: 'Focus Band' }), BENCH), turns: 4, seeds: 24, want: M(1, 1) },
  oh_protect: { p1: set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), p2: SNORLAX(['protect', 'splash'], { lv: 1 }), turns: 2, want: seq([[1, 1], [1, 2]]) },
  oh_wonderguard: { p1: set('Kingler', '', 'HyperCutter', ['guillotine', 'splash']), p2: set('Shedinja', '', 'WonderGuard', ['splash']), turns: 2, want: M(1, 1) },
  oh_dig: { p1: set('Whiscash', '', 'Oblivious', ['fissure', 'splash']), p2: set('Dugtrio', '', 'ArenaTrap', ['dig', 'splash']), turns: 2, want: M(1, 1) },
  oh_brightpowder: { p1: set('Dugtrio', '', 'ArenaTrap', ['fissure', 'splash']), p2: team(SNORLAX(['doubleteam', 'splash'], { lv: 1, item: 'Bright Powder' }), BENCH), turns: 5, want: seq([[2, 1], [2, 1], [1, 1]]) },
  oh_static: { p1: set('Rhydon', '', 'LightningRod', ['horndrill', 'splash']), p2: team(set('Pikachu', '', 'Static', ['splash'], { lv: 5 }), BENCH), turns: 2, seeds: 32, want: M(1, 1) },
  oh_sheercold_l: { p1: set('Lapras', '', 'ShellArmor', ['sheercold', 'splash']), p2: team(SNORLAX(['splash'], { lv: 70 }), BENCH), turns: 3, want: M(1, 1) },
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
  // The lines AS SENT: freeze each decision's new lines (the sim streams them at the request).
  const sent = [];
  const freeze = () => { for (let i = sent.length; i < battle.log.length; i++) sent.push(battle.log[i]); };
  freeze();
  const script = [];
  let t = 0;
  while (!battle.ended && t < board.turns) {
    const forced = battle.sides.some((s) => s.activeRequest && s.activeRequest.forceSwitch);
    const [w1, w2] = board.want(t);
    const d = [answer(battle.sides[0], w1), answer(battle.sides[1], w2)];
    if (d[0] === null && d[1] === null) break;
    if (d[0] !== null && !battle.choose('p1', d[0])) throw new Error(`p1 choice ${d[0]} rejected`);
    if (d[1] !== null && !battle.choose('p2', d[1])) throw new Error(`p2 choice ${d[1]} rejected`);
    freeze();
    script.push(d);
    if (!forced) t++;
  }
  freeze();
  const omni = [];
  for (let i = 0; i < sent.length; i++) {
    const l = sent[i];
    if (l.startsWith('|split|')) { omni.push(sent[i + 1]); i += 2; continue; }
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
const pair = (r, a, b) => r.lines.some((l, i) => a(l) && i + 1 < r.lines.length && b(r.lines[i + 1]));
const isRelease = (l) => /^\|-end\|[^|]+\|move: Bide$/.test(l);
const cov = {
  // BIDE
  bideCast: n((r) => has(r, '|-start|') && has(r, '|move: Bide')),
  bideWait: n((r) => r.lines.some((l) => /^\|-activate\|[^|]+\|move: Bide$/.test(l))),
  bideReleaseHit: n((r) => pair(r, isRelease, (b) => b.startsWith('|-damage|'))),
  bideReleaseFail: n((r) => pair(r, isRelease, (b) => b.startsWith('|-fail|'))),
  bideReleaseImmune: n((r) => pair(r, isRelease, (b) => /^\|-immune\|[^|]+$/.test(b))),
  bideReleaseProtect: n((r) => pair(r, isRelease, (b) => b.endsWith('|Protect'))),
  bideReleaseWonderGuard: n((r) => pair(r, isRelease, (b) => b.endsWith('ability: Wonder Guard'))),
  bideReleaseSub: n((r) => pair(r, isRelease, (b) => b.includes('|Substitute'))),
  bideReleaseMissFly: n((r) => pair(r, isRelease, (b) => b.startsWith('|-miss|')) && has(r, '|Fly||[still]|[miss]')),
  bideCastMiss: n((r) => r.lines.some((l) => /\|Bide\|[^|]+\|\[miss\]$/.test(l))),
  bideCantEnd: n((r) => pair(r, (a) => a.startsWith('|cant|'), (b) => b.endsWith('|move: Bide|[silent]'))),
  bideSelfHitEnd: n((r) => pair(r, (a) => a.endsWith('[from] confusion'), (b) => b.endsWith('|move: Bide|[silent]'))),
  bideResidualEnd: n((r) => r.name === 'bd_faint_skip' && has(r, '|move: Bide|[silent]') && !r.lines.some(isRelease)),
  bideRoughSkin: n((r) => r.name === 'bd_roughskin' && pair(r, isRelease, (b) => b.startsWith('|-damage|')) && has(r, 'ability: Rough Skin')),
  bideFutureSight: n((r) => r.name === 'bd_futuresight' && pair(r, isRelease, (b) => b.startsWith('|-damage|'))),
  bideKingsRockTie: n((r) => r.name === 'bd_kingsrock'),
  bideEncore: n((r) => r.name === 'bd_encore' && has(r, '|-start|p1a: Raichu|Encore')),
  // MAGNITUDE
  ...Object.fromEntries([4, 5, 6, 7, 8, 9, 10].map((m) => [`magnitude${m}`, n((r) => has(r, `|move: Magnitude|${m}`))])),
  magnitudeMissAttr: n((r) => r.lines.some((l, i) => l.includes('|Magnitude|') && l.endsWith('[miss]') && (r.lines[i + 1] || '').includes('move: Magnitude'))),
  magnitudeImmune: n((r) => pair(r, (a) => a.includes('move: Magnitude'), (b) => b.startsWith('|-immune|'))),
  magnitudeProtect: n((r) => pair(r, (a) => a.includes('move: Magnitude'), (b) => b.endsWith('|Protect'))),
  magnitudeDig: n((r) => r.name === 'mg_dig' && pair(r, (a) => a.includes('move: Magnitude'), (b) => b.startsWith('|-damage|'))),
  magnitudeSleepTalk: n((r) => r.lines.some((l) => l.includes('|Magnitude|') && l.includes('[from] Sleep Talk'))),
  // PSYWAVE
  psywaveHit: n((r) => pair(r, (a) => a.includes('|Psywave|'), (b) => b.startsWith('|-damage|'))),
  psywaveImmune: n((r) => r.name === 'pw_dark' && has(r, '|-immune|')),
  psywaveMiss: n((r) => r.lines.some((l) => l.includes('|Psywave|') && l.endsWith('[miss]'))),
  psywaveSub: n((r) => r.name === 'pw_sub' && pair(r, (a) => a.includes('|Psywave|'), (b) => b.includes('|Substitute'))),
  psywaveLowLevel: n((r) => ['pw_l50', 'pw_l37', 'pw_l5', 'pw_l1'].includes(r.name) && has(r, '|-damage|')),
  // PRESENT
  presentHeal: n((r) => pair(r, (a) => a.includes('|Present|'), (b) => /^\|-heal\|[^|]+\|[^|]+$/.test(b))),
  presentHealFullHp: n((r) => pair(r, (a) => a.endsWith('|Present||[still]'), (b) => b.endsWith('|heal'))),
  presentHealSub: n((r) => pair(r, (a) => a.endsWith('|Present||[still]'), (b) => /^\|-fail\|p1a: [^|]+$/.test(b))),
  presentDamage: n((r) => pair(r, (a) => a.includes('|Present|'), (b) => b.startsWith('|-damage|') || b.startsWith('|-crit|') || b.startsWith('|-supereffective|') || b.startsWith('|-resisted|'))),
  presentImmune: n((r) => r.name === 'pr_ghost' && has(r, '|-immune|')),
  presentWonderGuard: n((r) => r.name === 'pr_wonderguard' && has(r, 'Wonder Guard')),
  // TRIPLE KICK
  ...Object.fromEntries([1, 2, 3].map((k) => [`tripleKick${k}`, n((r) => r.lines.some((l) => /^\|-hitcount\|/.test(l) && l.endsWith(`|${k}`)) && has(r, '|Triple Kick|'))])),
  tripleKickFirstMiss: n((r) => r.lines.some((l) => l.includes('|Triple Kick|') && l.endsWith('[miss]'))),
  // OHKO
  ohkoImmuneLevel: n((r) => has(r, '|[ohko]')),
  ohkoHit: n((r) => pair(r, (a) => a.endsWith(' fnt') || a.endsWith('|Substitute') || a.endsWith('[damage]'), (b) => b === '|-ohko|' || b === '|-ohko')),
  ohkoKo: n((r) => r.lines.some((l, i) => (l === '|-ohko|' || l === '|-ohko') && (r.lines[i - 1] || '').endsWith(' fnt'))),
  ohkoMiss: n((r) => r.lines.some((l) => /\|(Fissure|Horn Drill|Guillotine|Sheer Cold)\|[^|]+\|\[miss\]$/.test(l))),
  ohkoSturdy: n((r) => has(r, 'ability: Sturdy')),
  ohkoSubBroke: n((r) => pair(r, (a) => a.endsWith('|Substitute') && a.startsWith('|-end|'), (b) => b.startsWith('|-ohko'))),
  ohkoSubHeld: n((r) => pair(r, (a) => a.endsWith('|Substitute|[damage]'), (b) => b.startsWith('|-ohko'))),
  ohkoEndure: n((r) => r.name === 'oh_endure' && has(r, 'move: Endure')),
  ohkoFocusBand: n((r) => r.name === 'oh_focusband' && has(r, 'item: Focus Band')),
  ohkoNaturalImmune: n((r) => ['oh_ghost', 'oh_flying'].includes(r.name) && r.lines.some((l) => /^\|-immune\|[^|]+$/.test(l))),
  ohkoProtect: n((r) => r.name === 'oh_protect' && has(r, '|Protect')),
  ohkoWonderGuard: n((r) => r.name === 'oh_wonderguard' && has(r, 'Wonder Guard')),
  ohkoInvulnerable: n((r) => r.name === 'oh_dig' && has(r, '|-miss|')),
  ohkoStatic: n((r) => r.name === 'oh_static' && has(r, 'ability: Static')),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0 && !process.env.B8_NOFLOOR) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# B8 MOVES GOLDEN (gen3_b8_moves_v1): Bide, Magnitude, Psywave, Present, Triple Kick, Fissure / Horn Drill / Guillotine / Sheer Cold',
  `# generated by harness/gen_b8_moves_golden.js  rows/board=${N}  boards=${Object.keys(BOARDS).length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the format, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE /',
  '# SEED_AFTER, the FRAMING window and the post-|turn|1 omniscient lines AS SENT (|t:| dropped).',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
