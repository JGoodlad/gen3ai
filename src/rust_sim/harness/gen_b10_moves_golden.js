// GOLDEN GENERATOR — the B10 moves and the Endless Battle Clause (`gen3_b10_moves_v1`,
// `gen3_b10_endless_battle_v1`, gen3 full parity B10): Teleport, Follow Me, Helping Hand, Role Play,
// Camouflage, and the clause's staleness. The settled spec is `harness/probe_b10_moves.js`'s header.
// Each board runs over N seeds; each row records the format, the decisions as played ([p1, p2],
// null = no choice), the sim SEED_BEFORE / SEED_AFTER, the FRAMING window and the post-`|turn|1`
// omniscient lines (the `|split|` pairs resolved to their omniscient half).
//
// The Endless Battle Clause needs a board past turn 100, so its rows are TARGETED long scenarios (the
// ebc_* boards, ~101-105 turns, fewer seeds), not random battles: a Recycle loss, a mutual tie, a Thief
// bare tie, a Trick + Pickup loss, a Shadow-Tag-trapped side, and four no-trigger controls.
//
//   node src/rust_sim/harness/gen_b10_moves_golden.js -> tests/vectors/b10_moves_golden.txt
//   cargo test --test b10_moves_golden_test
// env: B10_SEEDS (rows per move board, default 8), B10_EBC_SEEDS (rows per clause board, default 2), B10_OUT.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.B10_SEEDS || '8', 10);
const NE = parseInt(process.env.B10_EBC_SEEDS || '2', 10);
const OUT = process.env.B10_OUT || path.resolve(__dirname, '../tests/vectors/b10_moves_golden.txt');

const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash', 'harden'], { g: 'F' });
// want(t) -> [p1, p2] for the t-th regular turn; each entry a choice string ('move K' / 'switch K'), a
// number K (= 'move K'), or an array of preferences (the first enabled one is played). The driver
// answers a forced switch itself, plays a locked / Struggle request's only option, and falls back to the
// first enabled move.
const norm = (x) => (Array.isArray(x) ? x : [x]).map((y) => (typeof y === 'number' ? `move ${y}` : y));
const M = (a, b) => () => [a, b];
const seq = (rows) => (t) => rows[Math.min(t, rows.length - 1)];

const NOOP = [3, 4];
const AZ = set('Alakazam', 'Sitrus Berry', 'Synchronize', ['skillswap', 'recycle', 'splash', 'harden']);
const SX = set('Snorlax', 'Leftovers', 'ThickFat', ['seismictoss', 'splash', 'harden']);
// Toss Alakazam twice (Sitrus eaten at 72/272: the starting berry, no staleness), Recycle it (pending),
// eat it again at the residual (INTERNAL staleness), Skill Swap the foe (EXTERNAL), then idle.
const planA = (t) => (t < 2 ? [3, 1] : t === 2 ? [2, 2] : t === 3 ? [1, [2, 3]] : [NOOP, [2, 3]]);
const EBC = { format: 'gen3ou', turns: 104, ebc: true };

const BOARDS = {
  // ---------------------------------------------------------------- TELEPORT
  tp_basic: { p1: set('Abra', '', 'Synchronize', ['teleport', 'psychic']), p2: team(set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), BENCH), turns: 3, want: seq([[1, 2], [1, 1], [2, 1]]) },
  tp_tie: { p1: set('Abra', '', 'Synchronize', ['teleport', 'psychic']), p2: set('Abra', '', 'Synchronize', ['teleport', 'psychic']), turns: 3, want: M(1, 1) },
  tp_encore: { p1: set('Abra', '', 'Synchronize', ['teleport', 'psychic']), p2: set('Jolteon', '', 'VoltAbsorb', ['encore', 'splash']), turns: 4, want: seq([[1, 2], [2, 1], [2, 2]]) },
  tp_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['sleeptalk', 'rest', 'teleport', 'followme']), p2: set('Blissey', '', 'NaturalCure', ['seismictoss'], { g: 'F' }), turns: 4, want: seq([[2, 1], [1, 1]]) },
  tp_ou: { format: 'gen3ou', p1: set('Abra', '', 'Synchronize', ['teleport', 'thunderwave']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'teleport']), turns: 3, want: seq([[2, 2], [1, 1], [1, 2]]) },
  // ---------------------------------------------------------------- FOLLOW ME
  fm_protect: { p1: set('Clefable', '', 'CuteCharm', ['followme', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['protect', 'thunderbolt']), turns: 3, want: seq([[1, 1], [1, 2], [1, 1]]) },
  fm_tie: { p1: set('Clefable', '', 'CuteCharm', ['followme', 'bodyslam']), p2: set('Clefable', '', 'CuteCharm', ['followme', 'bodyslam']), turns: 3, want: seq([[1, 1], [2, 1], [1, 2]]) },
  fm_encore: { p1: set('Clefable', '', 'CuteCharm', ['followme', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['encore', 'thunderbolt']), turns: 4, want: seq([[1, 2], [2, 1], [2, 2]]) },
  fm_hit: { p1: set('Togetic', '', 'Hustle', ['followme', 'seismictoss']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'protect']), turns: 3, want: seq([[1, 1], [2, 2], [1, 1]]) },
  // ---------------------------------------------------------------- HELPING HAND
  hh_protect: { p1: set('Plusle', '', 'Plus', ['helpinghand', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['protect', 'thunderbolt']), turns: 3, want: seq([[1, 1], [1, 2], [1, 1]]) },
  hh_tie: { p1: set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), p2: set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), turns: 3, want: seq([[1, 1], [2, 1], [1, 2]]) },
  hh_pressure: { p1: set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), p2: set('Zapdos', '', 'Pressure', ['splash', 'thunderbolt']), turns: 3, want: M(1, 1) },
  hh_encore: { p1: set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), p2: set('Jolteon', '', 'VoltAbsorb', ['encore', 'splash']), turns: 4, want: seq([[1, 2], [2, 1], [2, 2]]) },
  hh_disable: { p1: set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), p2: set('Jolteon', '', 'VoltAbsorb', ['disable', 'splash']), turns: 4, want: seq([[1, 2], [2, 1], [1, 2]]) },
  hh_spite: { p1: set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), p2: set('Jolteon', '', 'VoltAbsorb', ['spite', 'splash']), turns: 3, want: seq([[1, 2], [2, 1]]) },
  hh_mimic: { p1: set('Plusle', '', 'Plus', ['helpinghand', 'thunderbolt']), p2: set('Jolteon', '', 'VoltAbsorb', ['mimic', 'splash']), turns: 4, want: seq([[1, 2], [2, 1], [2, 1]]) },
  hh_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['sleeptalk', 'rest', 'helpinghand', 'camouflage']), p2: set('Blissey', '', 'NaturalCure', ['seismictoss'], { g: 'F' }), turns: 4, want: seq([[2, 1], [1, 1]]) },
  hh_ou: { format: 'gen3ou', p1: set('Minun', '', 'Minus', ['helpinghand', 'thunderwave']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'helpinghand']), turns: 3, want: seq([[2, 2], [1, 1], [1, 2]]) },
  // ---------------------------------------------------------------- ROLE PLAY
  rp_intimidate: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'psychic']), p2: set('Gyarados', '', 'Intimidate', ['splash', 'waterfall']), turns: 3, want: seq([[1, 1], [2, 2]]) },
  rp_wonderguard: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), p2: set('Shedinja', '', 'WonderGuard', ['splash']), turns: 2, want: M(1, 1) },
  rp_same: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), p2: set('Espeon', '', 'Synchronize', ['splash']), turns: 2, want: M(1, 1) },
  rp_wg_user: { p1: set('Shedinja', '', 'WonderGuard', ['roleplay', 'shadowball']), p2: set('Gyarados', '', 'Intimidate', ['splash', 'waterfall']), turns: 3, want: seq([[1, 1], [2, 2]]) },
  rp_protect: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), p2: set('Gyarados', '', 'Intimidate', ['protect', 'splash']), turns: 2, want: M(1, 1) },
  rp_sub: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), p2: set('Gyarados', '', 'Intimidate', ['substitute', 'splash']), turns: 3, want: seq([[2, 1], [1, 2]]) },
  rp_bounce: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash'], { evs: '0,0,0,0,0,0' }), p2: set('Jolteon', '', 'VoltAbsorb', ['bounce', 'splash']), turns: 3, want: M(1, 1) },
  rp_pressure: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'psychic']), p2: set('Zapdos', '', 'Pressure', ['splash']), turns: 3, want: seq([[1, 1], [2, 1]]) },
  rp_truant: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'psychic']), p2: set('Slaking', '', 'Truant', ['splash', 'bodyslam']), turns: 5, want: seq([[2, 1], [1, 1], [2, 2]]) },
  rp_truant_user: { p1: set('Slaking', '', 'Truant', ['roleplay', 'splash']), p2: set('Gyarados', '', 'Intimidate', ['splash']), turns: 5, want: seq([[2, 1], [1, 1], [1, 1], [2, 1]]) },
  rp_insomnia: { p1: set('Snorlax', '', 'ThickFat', ['roleplay', 'rest', 'sleeptalk']), p2: set('Hypno', '', 'Insomnia', ['seismictoss', 'splash']), turns: 4, want: seq([[2, 1], [3, 2], [1, 2]]) },
  rp_waterveil: { p1: set('Snorlax', '', 'ThickFat', ['roleplay', 'splash']), p2: set('Vaporeon', '', 'WaterVeil', ['willowisp', 'splash']), turns: 3, want: seq([[2, 1], [1, 2]]) },
  rp_flashfire: { p1: set('Houndoom', '', 'FlashFire', ['roleplay', 'flamethrower']), p2: set('Gyarados', '', 'Intimidate', ['splash', 'ember']), turns: 4, want: seq([[2, 2], [1, 1], [2, 1]]) },
  rp_flashfire_same: { p1: set('Houndoom', '', 'FlashFire', ['roleplay', 'flamethrower']), p2: set('Ninetales', '', 'FlashFire', ['flamethrower', 'splash']), turns: 2, want: M(1, 1) },
  rp_cloudnine: { p1: set('Golduck', '', 'CloudNine', ['roleplay', 'raindance', 'surf']), p2: set('Gyarados', '', 'Intimidate', ['splash']), turns: 4, want: seq([[2, 1], [1, 1], [3, 1]]) },
  rp_airlock: { p1: set('Gyarados', '', 'Intimidate', ['roleplay', 'raindance', 'surf']), p2: set('Rayquaza', '', 'AirLock', ['splash']), turns: 4, want: seq([[2, 1], [1, 1], [3, 1]]) },
  rp_castform: { p1: set('Castform', '', 'Forecast', ['roleplay', 'raindance', 'sunnyday']), p2: set('Gyarados', '', 'Intimidate', ['splash']), turns: 4, want: seq([[2, 1], [1, 1], [3, 1]]) },
  rp_speedboost: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), p2: set('Ninjask', '', 'SpeedBoost', ['splash', 'aerialace']), turns: 3, want: seq([[1, 1], [2, 2]]) },
  rp_shadowtag: { p1: team(set('Alakazam', '', 'Synchronize', ['roleplay', 'psychic']), BENCH), p2: team(set('Wobbuffet', '', 'ShadowTag', ['splash']), BENCH), turns: 3, want: seq([[1, 1], ['switch 2', 'switch 2']]) },
  rp_naturalcure: { p1: team(set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), BENCH), p2: set('Blissey', '', 'NaturalCure', ['thunderwave', 'splash'], { g: 'F' }), turns: 4, want: seq([[2, 1], [1, 2], ['switch 2', 2]]) },
  rp_synchronize: { p1: set('Snorlax', '', 'ThickFat', ['roleplay', 'splash']), p2: set('Espeon', '', 'Synchronize', ['toxic', 'splash']), turns: 3, want: seq([[1, 2], [2, 1]]) },
  rp_colorchange: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), p2: set('Kecleon', '', 'ColorChange', ['splash', 'thunderbolt']), turns: 3, want: seq([[1, 1], [2, 2]]) },
  rp_shedskin: { p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), p2: set('Dragonair', '', 'ShedSkin', ['toxic', 'splash']), turns: 5, seeds: 12, want: seq([[2, 1], [1, 2], [2, 2]]) },
  rp_switch: { p1: team(set('Alakazam', '', 'Synchronize', ['roleplay', 'splash']), BENCH), p2: set('Gyarados', '', 'Intimidate', ['splash']), turns: 3, want: seq([[1, 1], ['switch 2', 1], ['switch 2', 1]]) },
  rp_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['sleeptalk', 'rest', 'roleplay']), p2: set('Gyarados', '', 'Intimidate', ['seismictoss', 'splash']), turns: 4, want: seq([[2, 1], [1, 2]]) },
  rp_ou: { format: 'gen3ou', p1: set('Alakazam', '', 'Synchronize', ['roleplay', 'thunderwave']), p2: set('Snorlax', '', 'Immunity', ['bodyslam', 'roleplay']), turns: 4, want: seq([[2, 1], [1, 2], [1, 1]]) },
  // ---------------------------------------------------------------- CAMOUFLAGE
  cf_colorchange: { p1: set('Kecleon', '', 'ColorChange', ['camouflage', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['thunderbolt']), turns: 3, want: M(1, 1) },
  cf_normal: { p1: set('Snorlax', '', 'ThickFat', ['camouflage', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['splash']), turns: 2, want: M(1, 1) },
  cf_normalflying: { p1: set('Pidgeot', '', 'KeenEye', ['camouflage', 'aerialace']), p2: set('Jolteon', '', 'VoltAbsorb', ['splash']), turns: 2, want: M(1, 1) },
  cf_snatch: { p1: set('Kecleon', '', 'ColorChange', ['camouflage', 'splash']), p2: set('Sableye', '', 'KeenEye', ['snatch', 'shadowball']), turns: 3, want: seq([[1, 1], [2, 2]]) },
  cf_snatch_normal: { p1: set('Kecleon', '', 'ColorChange', ['camouflage']), p2: set('Snorlax', '', 'ThickFat', ['snatch']), turns: 2, want: M(1, 1) },
  cf_switch: { p1: team(set('Gengar', '', 'Levitate', ['camouflage', 'splash']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 4, want: seq([[1, 2], [2, 1], ['switch 2', 2], ['switch 2', 1]]) },
  cf_transform: { p1: set('Ditto', '', 'Limber', ['transform']), p2: set('Gengar', '', 'Levitate', ['camouflage', 'splash']), turns: 3, want: seq([[1, 2], [1, 1]]) },
  cf_conversion: { p1: set('Porygon2', '', 'Trace', ['conversion', 'camouflage', 'icebeam']), p2: set('Snorlax', '', 'ThickFat', ['splash', 'shadowball']), turns: 4, want: seq([[1, 1], [2, 1], [3, 2]]) },
  cf_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['sleeptalk', 'rest', 'camouflage', 'teleport']), p2: set('Gengar', '', 'Levitate', ['seismictoss', 'splash']), turns: 4, want: seq([[2, 1], [1, 1]]) },
  cf_ghost_immunity: { p1: set('Gengar', '', 'Levitate', ['camouflage', 'splash']), p2: set('Machamp', '', 'Guts', ['crosschop', 'bodyslam']), turns: 3, want: seq([[2, 1], [1, 1], [2, 2]]) },
  cf_ou: { format: 'gen3ou', p1: set('Kecleon', '', 'ColorChange', ['camouflage', 'thunderwave']), p2: set('Starmie', '', 'NaturalCure', ['surf', 'camouflage']), turns: 3, want: seq([[1, 1], [2, 2], [1, 1]]) },
  // ---------------------------------------------------------------- ENDLESS BATTLE CLAUSE (gen3ou)
  ebc_recycle_loses: { ...EBC, p1: AZ, p2: SX, want: planA },
  ebc_both_cycle_tie: { ...EBC, p1: AZ, p2: set('Snorlax', 'Sitrus Berry', 'ThickFat', ['seismictoss', 'splash', 'harden', 'recycle']), want: planA },
  ebc_thief_bare_tie: {
    ...EBC, p1: set('Alakazam', '', 'Synchronize', ['skillswap', 'thief', 'splash', 'harden']), p2: set('Snorlax', 'Sitrus Berry', 'ThickFat', ['seismictoss', 'splash', 'harden']),
    want: (t) => (t === 0 ? [2, 1] : t === 1 ? [3, 1] : t === 2 ? [1, 2] : [NOOP, [2, 3]]),
  },
  ebc_trick_pickup_loses: {
    ...EBC, p1: set('Linoone', 'Sitrus Berry', 'Pickup', ['trick', 'seismictoss', 'splash', 'harden']), p2: set('Alakazam', 'Leftovers', 'Synchronize', ['skillswap', 'splash', 'harden']),
    want: (t) => (t === 0 ? [1, 2] : t < 3 ? [2, 2] : t === 3 ? [3, 1] : [NOOP, [2, 3]]),
  },
  ebc_shadowtag_trapped: { ...EBC, p1: team(AZ, BENCH), p2: team(set('Wobbuffet', 'Leftovers', 'ShadowTag', ['seismictoss', 'splash', 'harden']), BENCH), want: planA },
  ebc_ctrl_both_can_switch: { ...EBC, p1: team(AZ, BENCH), p2: team(SX, BENCH), want: planA },
  ebc_ctrl_switch_clears: { ...EBC, p1: AZ, p2: team(SX, BENCH), want: (t) => (t === 4 || t === 5 ? [NOOP, 'switch 2'] : planA(t)) },
  ebc_ctrl_starting_berry: { ...EBC, p1: AZ, p2: SX, want: (t) => (t === 2 ? [3, 2] : planA(t)) },
  ebc_ctrl_customgame: { ...EBC, format: 'gen3customgame', p1: AZ, p2: SX, want: planA },
};

function answer(side, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) return `switch ${side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1}`;
  const moves = req.active[0].moves;
  for (const w of norm(want)) {
    if (w.startsWith('switch')) { if (!req.active[0].trapped) return w; continue; }
    if (moves.length === 1) return 'move 1';
    const k = parseInt(w.split(' ')[1], 10);
    if (moves[k - 1] && !moves[k - 1].disabled) return w;
  }
  if (moves.length === 1) return 'move 1';
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
  for (let k = 0; k < (b.seeds || (b.ebc ? NE : N)); k++) {
    const seed = [31 + k * 7, 47 + k * 13, 61 + k * 29, 73 + k * 51];
    rows.push({ name, format: b.format || 'gen3customgame', p1: b.p1, p2: b.p2, ...run(b, seed) });
  }
}
const n = (pred) => rows.filter(pred).length;
const has = (r, s) => r.lines.some((l) => l.includes(s));
const pair = (r, a, b) => r.lines.some((l, i) => a(l) && b(r.lines[i + 1] || ''));
const bare = (mv) => (r) => r.lines.some((l, i) => new RegExp(`^\\|move\\|(p[12]a: [^|]+)\\|${mv}\\|\\1$`).test(l) && !/^\|-(fail|activate|singleturn)/.test(r.lines[i + 1] || ''));
const cov = {
  teleportBare: n(bare('Teleport')),
  teleportSleepTalk: n((r) => has(r, '|Teleport|') && r.lines.some((l) => l.includes('|Teleport|') && l.includes('[from] Sleep Talk'))),
  teleportEncored: n((r) => r.name === 'tp_encore' && has(r, '|-start|p1a: Abra|Encore')),
  teleportOu: n((r) => r.name === 'tp_ou' && bare('Teleport')(r)),
  followMeBare: n(bare('Follow Me')),
  followMeSleepTalk: n((r) => r.lines.some((l) => l.includes('|Follow Me|') && l.includes('[from] Sleep Talk'))),
  followMeEncored: n((r) => r.name === 'fm_encore' && has(r, '|-start|p1a: Clefable|Encore')),
  helpingHand: n((r) => pair(r, (a) => /\|Helping Hand\|null\|\[notarget\]$/.test(a), (b) => b.startsWith('|-notarget|'))),
  helpingHandSleepTalk: n((r) => has(r, '|Helping Hand|null|[from] Sleep Talk|[notarget]')),
  helpingHandSpite: n((r) => has(r, '|move: Spite|helpinghand|')),
  helpingHandMimic: n((r) => has(r, '|move: Mimic|Helping Hand')),
  helpingHandEncored: n((r) => r.name === 'hh_encore' && has(r, '|-start|p1a: Plusle|Encore')),
  helpingHandDisabled: n((r) => has(r, '|Disable|Helping Hand')),
  helpingHandOu: n((r) => r.name === 'hh_ou' && has(r, '|-notarget|')),
  rolePlay: n((r) => has(r, '|[from] move: Role Play|[of] ')),
  rolePlayFail: n((r) => pair(r, (a) => a.endsWith('|Role Play||[still]'), (b) => b.startsWith('|-fail|'))),
  rolePlayWonderGuardFail: n((r) => r.name === 'rp_wonderguard' && has(r, '|Role Play||[still]')),
  rolePlayProtectNoBlock: n((r) => r.name === 'rp_protect' && has(r, '[from] move: Role Play')),
  rolePlaySub: n((r) => r.name === 'rp_sub' && has(r, '[from] move: Role Play')),
  rolePlayMiss: n((r) => r.name === 'rp_bounce' && has(r, '|Role Play|p2a: Jolteon|[miss]')),
  rolePlayTruantLoaf: n((r) => has(r, '|cant|p1a: Alakazam|ability: Truant')),
  rolePlayStatusCure: n((r) => r.lines.some((l, i) => /\|ability: (Insomnia|Water Veil)$/.test(l) && (r.lines[i + 1] || '').startsWith('|-curestatus|p1a'))),
  rolePlayFlashFireEnd: n((r) => pair(r, (a) => a === '|-end|p1a: Houndoom|ability: Flash Fire|[silent]', (b) => b.includes('[from] move: Role Play'))),
  rolePlayAirLock: n((r) => r.name === 'rp_airlock' && has(r, '|Air Lock|Intimidate|[from] move: Role Play')),
  rolePlaySpeedBoost: n((r) => has(r, '|-ability|p1a: Alakazam|Speed Boost|boost')),
  rolePlayShadowTag: n((r) => r.name === 'rp_shadowtag' && has(r, 'Shadow Tag|Synchronize|[from] move: Role Play')),
  rolePlayNaturalCure: n((r) => has(r, '|-curestatus|p1a: Alakazam|par|[from] ability: Natural Cure')),
  rolePlaySleepTalk: n((r) => r.lines.some((l) => l.includes('|Role Play|') && l.includes('[from] Sleep Talk'))),
  rolePlayOu: n((r) => r.name === 'rp_ou' && has(r, '[from] move: Role Play')),
  camouflage: n((r) => has(r, '|typechange|Normal') && r.lines.some((l) => /\|typechange\|Normal$/.test(l))),
  camouflageFail: n((r) => pair(r, (a) => /\|Camouflage\|\|(\[from\] [^|]+\|)?\[still\]$/.test(a), (b) => b.startsWith('|-fail|'))),
  camouflageSnatched: n((r) => has(r, '|Camouflage|p2a: Sableye|[from] Snatch')),
  camouflageSnatchFail: n((r) => has(r, '|Camouflage||[from] Snatch|[still]')),
  camouflageTransformed: n((r) => r.name === 'cf_transform' && has(r, '|-start|p1a: Ditto|typechange|Normal')),
  camouflageSleepTalk: n((r) => r.lines.some((l) => l.includes('|Camouflage|') && l.includes('[from] Sleep Talk'))),
  camouflageOu: n((r) => r.name === 'cf_ou' && has(r, '|typechange|Normal')),
  ebcLoses: n((r) => has(r, 'berry-cycling and thus loses.') && has(r, '|win|')),
  ebcBoth: n((r) => has(r, "|-message|Each side's team") && r.lines[r.lines.length - 1] === '|tie'),
  ebcBareTie: n((r) => r.name === 'ebc_thief_bare_tie' && r.lines[r.lines.length - 1] === '|tie' && !has(r, '|-message|')),
  ebcPickup: n((r) => r.name === 'ebc_trick_pickup_loses' && has(r, "|-message|P1's team")),
  ebcTrapped: n((r) => r.name === 'ebc_shadowtag_trapped' && has(r, 'thus loses.')),
  ebcControls: n((r) => r.name.startsWith('ebc_ctrl_') && has(r, '|turn|101') && has(r, '|turn|104') && !has(r, 'berry-cycling')),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0 && !process.env.B10_NOFLOOR) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# B10 MOVES GOLDEN (gen3_b10_moves_v1, gen3_b10_endless_battle_v1): Teleport, Follow Me, Helping Hand, Role Play,',
  '# Camouflage, and the Endless Battle Clause (targeted ebc_* boards past turn 100)',
  `# generated by harness/gen_b10_moves_golden.js  rows/board=${N} (ebc ${NE})  boards=${Object.keys(BOARDS).length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the format, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE /',
  '# SEED_AFTER, the FRAMING window and the post-|turn|1 omniscient lines (|t:| dropped).',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
