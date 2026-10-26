// GOLDEN GENERATOR — the B5 moves (`gen3_b5_moves_v1`, gen3 full parity B5): Spite, Grudge, Snore,
// Nightmare, Ingrain, and Rest under a live Uproar. The settled spec is `harness/probe_b5_moves.js`'s
// header. Each board runs over N seeds; each row records the format, the decisions as played
// ([p1, p2], null = no choice), the sim SEED_BEFORE / SEED_AFTER, the FRAMING window and the
// post-`|turn|1` omniscient lines (the `|split|` pairs resolved to their omniscient half).
//
//   node src/rust_sim/harness/gen_b5_moves_golden.js -> tests/vectors/b5_moves_golden.txt
//   cargo test --release --test b5_moves_golden_test
// env: B5_SEEDS (rows per board, default 8), B5_OUT.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.B5_SEEDS || '8', 10);
const OUT = process.env.B5_OUT || path.resolve(__dirname, '../tests/vectors/b5_moves_golden.txt');

const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BENCH = set('Blissey', '', 'NaturalCure', ['splash'], { g: 'F' });
// want(t) -> [p1, p2] for the t-th regular turn: 'move K' / 'switch K'. The driver answers a
// forced switch itself, plays a locked / Struggle request's only option, and falls back to the
// first enabled move when the wanted one is disabled (0 PP, Choice lock, Encore).
const M = (a, b) => () => [`move ${a}`, `move ${b}`];
const seq = (rows) => (t) => rows[Math.min(t, rows.length - 1)].map((x) => (typeof x === 'number' ? `move ${x}` : x));

const BOARDS = {
  // ---------------------------------------------------------------- SPITE
  sp_basic: { p1: set('Gengar', '', 'Levitate', ['spite', 'splash']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 4, want: seq([[2, 1], [1, 1]]) },
  sp_no_lastmove: { p1: set('Jolteon', '', 'VoltAbsorb', ['spite', 'splash']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 2, want: M(1, 1) },
  sp_pressure: { p1: set('Gengar', '', 'Levitate', ['spite', 'splash']), p2: set('Zapdos', '', 'Pressure', ['thunderbolt', 'splash'], { evs: '252,0,0,0,252,0' }), turns: 3, want: seq([[2, 1], [1, 1]]) },
  sp_sub: { p1: set('Snorlax', '', 'ThickFat', ['spite', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['substitute', 'thunderbolt']), turns: 3, want: seq([[2, 1], [1, 2]]) },
  sp_protect: { p1: set('Snorlax', '', 'ThickFat', ['spite', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['protect', 'thunderbolt']), turns: 3, want: seq([[2, 2], [1, 1]]) },
  // Spite until the move is empty: the roll above the PP left, then a 0-PP lastMove, then Struggle.
  sp_drain: { p1: set('Gengar', '', 'Levitate', ['spite', 'splash']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['zapcannon', 'splash'], { evs: '252,0,252,0,4,0' }), turns: 7, want: seq([[2, 1], [1, 1]]) },
  sp_drain_struggle: { p1: set('Gengar', '', 'Levitate', ['spite', 'splash']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['zapcannon'], { evs: '252,0,252,0,4,0' }), turns: 7, want: seq([[2, 1], [1, 1]]) },
  sp_sleeptalk: { p1: set('Gengar', '', 'Levitate', ['spite', 'spore']), p2: set('Snorlax', '', 'ThickFat', ['sleeptalk', 'bodyslam']), turns: 4, want: seq([[2, 1], [1, 1]]) },
  sp_choice: { p1: set('Gengar', '', 'Levitate', ['spite', 'splash']), p2: set('Snorlax', 'Choice Band', 'ThickFat', ['zapcannon', 'bodyslam'], { evs: '252,0,252,0,4,0' }), turns: 6, want: seq([[2, 1], [1, 1]]) },
  sp_encore: { p1: set('Gengar', '', 'Levitate', ['spite', 'encore', 'splash']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['zapcannon', 'bodyslam'], { evs: '252,0,252,0,4,0' }), turns: 6, want: seq([[3, 1], [2, 1], [1, 1]]) },
  sp_mimic: { p1: set('Gengar', '', 'Levitate', ['spite', 'shadowball']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['mimic', 'splash'], { evs: '252,0,252,0,4,0' }), turns: 3, want: seq([[2, 1], [1, 2]]) },
  // ---------------------------------------------------------------- GRUDGE
  gr_ko: { p1: team(set('Shedinja', '', 'WonderGuard', ['grudge', 'splash']), BENCH), p2: set('Houndoom', '', 'EarlyBird', ['flamethrower', 'crunch']), turns: 3, want: seq([[1, 2], [2, 1], [1, 1]]) },
  gr_removed: { p1: team(set('Shedinja', '', 'WonderGuard', ['grudge', 'splash']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['shadowball', 'splash']), turns: 3, want: seq([[1, 2], [2, 1], [1, 1]]) },
  gr_ko_same_turn: { p1: team(set('Shedinja', '', 'WonderGuard', ['grudge', 'splash']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['shadowball', 'splash']), turns: 3, want: M(1, 1) },
  gr_slept: { p1: team(set('Shedinja', '', 'WonderGuard', ['grudge', 'splash']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['spore', 'shadowball']), turns: 3, want: seq([[1, 1], [2, 2], [1, 1]]) },
  gr_recoil: { p1: team(set('Pichu', '', 'Static', ['grudge', 'splash'], { lv: 5 }), BENCH), p2: team(set('Shedinja', '', 'WonderGuard', ['doubleedge', 'splash']), BENCH), turns: 3, want: seq([[1, 2], [2, 1], [1, 1]]) },
  // The killer Rests, then Sleep Talk calls Shadow Ball (half the seeds): Grudge zeroes SLEEP TALK.
  gr_sleeptalk_killer: { p1: team(set('Shedinja', '', 'WonderGuard', ['slash', 'grudge']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['rest', 'sleeptalk', 'shadowball']), turns: 4, seeds: 16, want: seq([[1, 1], [2, 2]]) },
  gr_explosion: { p1: team(set('Pichu', '', 'Static', ['grudge', 'splash'], { lv: 5 }), BENCH), p2: team(set('Electrode', '', 'Soundproof', ['explosion', 'splash']), BENCH), turns: 3, want: seq([[1, 2], [2, 1], [1, 1]]) },
  gr_pursuit: { p1: team(set('Pichu', '', 'Static', ['grudge', 'splash'], { lv: 5 }), BENCH), p2: set('Tyranitar', '', 'SandStream', ['pursuit', 'splash']), turns: 3, want: seq([[1, 2], ['switch 2', 1], [1, 1]]) },
  gr_futuresight: { p1: team(set('Pichu', '', 'Static', ['grudge', 'splash'], { lv: 5 }), BENCH), p2: set('Alakazam', '', 'Synchronize', ['futuresight', 'splash']), turns: 4, want: seq([[2, 1], [2, 2], [1, 2], [1, 2]]) },
  gr_struggle_killer: { p1: team(set('Shedinja', '', 'WonderGuard', ['grudge', 'splash']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['shadowball']), turns: 4, want: seq([[1, 1], [1, 1]]) },
  // ---------------------------------------------------------------- SNORE
  sn_awake: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'rest']), p2: set('Blissey', '', 'NaturalCure', ['seismictoss', 'splash'], { g: 'F' }), turns: 2, want: M(1, 2) },
  sn_asleep: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'rest']), p2: set('Blissey', '', 'NaturalCure', ['seismictoss', 'splash'], { g: 'F' }), turns: 5, want: seq([[2, 1], [1, 1]]) },
  sn_flinch: { p1: set('Jolteon', '', 'VoltAbsorb', ['snore', 'rest']), p2: set('Snorlax', '', 'ThickFat', ['seismictoss', 'splash'], { evs: '252,0,252,0,4,0' }), turns: 5, seeds: 16, want: seq([[1, 1], [2, 1], [1, 1]]) },
  sn_sleeptalk: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'rest', 'sleeptalk']), p2: set('Blissey', '', 'NaturalCure', ['seismictoss', 'splash'], { g: 'F' }), turns: 5, want: seq([[2, 1], [3, 1]]) },
  sn_soundproof: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'rest']), p2: set('Electrode', '', 'Soundproof', ['seismictoss', 'splash']), turns: 3, want: seq([[2, 1], [1, 1]]) },
  sn_ghost: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'rest']), p2: set('Gengar', '', 'Levitate', ['seismictoss', 'splash']), turns: 3, want: seq([[2, 1], [1, 1]]) },
  sn_sub: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'rest']), p2: set('Jolteon', '', 'VoltAbsorb', ['substitute', 'seismictoss']), turns: 4, want: seq([[2, 2], [1, 1]]) },
  sn_protect: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'rest']), p2: set('Jolteon', '', 'VoltAbsorb', ['protect', 'seismictoss']), turns: 4, want: seq([[2, 2], [1, 1]]) },
  sn_spore: { p1: set('Snorlax', '', 'ThickFat', ['snore', 'splash']), p2: set('Breloom', '', 'EffectSpore', ['spore', 'splash']), turns: 5, want: seq([[2, 1], [1, 2]]) },
  // ---------------------------------------------------------------- NIGHTMARE
  nm_basic: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['splash']), turns: 6, want: seq([[1, 1], [2, 1]]) },
  nm_awake: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['splash']), turns: 2, want: M(2, 1) },
  nm_leech: { p1: set('Venusaur', '', 'Overgrow', ['spore', 'nightmare', 'leechseed']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['splash']), turns: 6, want: seq([[1, 1], [2, 1], [3, 1]]) },
  nm_toxic_curse: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'curse']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['splash']), turns: 5, want: seq([[1, 1], [2, 1], [3, 1]]) },
  nm_sub: { p1: set('Gengar', '', 'Levitate', ['nightmare', 'splash']), p2: set('Snorlax', '', 'ThickFat', ['substitute', 'rest']), turns: 4, want: seq([[2, 1], [2, 2], [1, 2]]) },
  nm_protect: { p1: set('Snorlax', '', 'ThickFat', ['nightmare', 'splash']), p2: set('Jolteon', '', 'VoltAbsorb', ['protect', 'splash']), turns: 2, want: M(1, 1) },
  nm_shedskin: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: set('Dragonair', '', 'ShedSkin', ['splash']), turns: 6, want: seq([[1, 1], [2, 1], [3, 1]]) },
  nm_naturalcure: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: team(set('Blissey', '', 'NaturalCure', ['splash'], { g: 'F' }), set('Snorlax', '', 'ThickFat', ['splash'])), turns: 4, want: seq([[1, 1], [2, 1], [3, 'switch 2']]) },
  nm_switch: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: team(set('Snorlax', '', 'ThickFat', ['splash']), BENCH), turns: 4, want: seq([[1, 1], [2, 1], [3, 'switch 2']]) },
  nm_healbell: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: set('Blissey', '', 'NaturalCure', ['healbell', 'sleeptalk', 'splash'], { g: 'F' }), turns: 6, want: seq([[1, 3], [2, 2], [3, 2]]) },
  nm_aromatherapy: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: set('Blissey', '', 'NaturalCure', ['aromatherapy', 'sleeptalk', 'splash'], { g: 'F' }), turns: 6, want: seq([[1, 3], [2, 2], [3, 2]]) },
  // (No Uproar-wake board: the port's Uproar wake prints its cure AFTER the damage where the sim
  // prints it at TryHit, BEFORE the damage — a pre-existing `gen3_uproar_v1` order defect outside
  // this batch, reported as a FINDING. The Nightmare end still follows the cure line there.)
  nm_chesto_trick: { p1: set('Gengar', 'Chesto Berry', 'Levitate', ['spore', 'nightmare', 'trick']), p2: set('Snorlax', '', 'ThickFat', ['splash']), turns: 4, want: seq([[1, 1], [2, 1], [3, 1]]) },
  nm_skillswap: { p1: set('Hypno', '', 'Insomnia', ['spore', 'nightmare', 'skillswap']), p2: set('Snorlax', '', 'ThickFat', ['splash']), turns: 4, want: seq([[1, 1], [2, 1], [3, 1]]) },
  nm_ko: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: team(set('Pichu', '', 'Static', ['splash'], { lv: 5 }), BENCH), turns: 5, want: seq([[1, 1], [2, 1], [3, 1]]) },
  nm_focusband: { p1: set('Gengar', '', 'Levitate', ['spore', 'nightmare', 'splash']), p2: team(set('Pichu', 'Focus Band', 'Static', ['splash'], { lv: 5 }), BENCH), turns: 5, want: seq([[1, 1], [2, 1], [3, 1]]) },
  // Both Rest, then Sleep Talk picks Nightmare on both sides (1/9 a turn): two equal-speed chips tie.
  nm_mirror: { p1: set('Snorlax', '', 'ThickFat', ['seismictoss', 'rest', 'sleeptalk', 'nightmare']), p2: set('Snorlax', '', 'ThickFat', ['seismictoss', 'rest', 'sleeptalk', 'nightmare']), turns: 5, seeds: 32, want: seq([[1, 1], [2, 2], [3, 3]]) },
  nm_sleeptalk_called: { p1: set('Gengar', '', 'Levitate', ['nightmare', 'sleeptalk']), p2: set('Snorlax', '', 'ThickFat', ['splash', 'spore']), turns: 5, want: seq([[1, 2], [2, 2], [2, 1]]) },
  // ---------------------------------------------------------------- INGRAIN
  ig_basic: { p1: set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 4, want: seq([[1, 1], [2, 1]]) },
  ig_twice: { p1: set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 3, want: M(1, 1) },
  ig_full_hp: { p1: set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 3, want: seq([[1, 2], [2, 2]]) },
  ig_roar: { p1: team(set('Cradily', '', 'Chlorophyll', ['ingrain', 'splash']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['roar', 'whirlwind']), turns: 3, want: seq([[1, 2], [2, 1], [2, 2]]) },
  ig_roar_nobench: { p1: set('Cradily', '', 'Chlorophyll', ['ingrain', 'splash']), p2: set('Snorlax', '', 'ThickFat', ['roar', 'whirlwind']), turns: 3, want: seq([[1, 2], [2, 1]]) },
  ig_suctioncups: { p1: team(set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['roar', 'whirlwind']), turns: 3, want: seq([[1, 2], [2, 1]]) },
  ig_batonpass: { p1: team(set('Cradily', '', 'Chlorophyll', ['ingrain', 'batonpass']), BENCH), p2: set('Snorlax', '', 'ThickFat', ['bodyslam', 'splash']), turns: 4, want: seq([[1, 1], [2, 1], [1, 1]]) },
  ig_mirror: { p1: set('Snorlax', 'Leftovers', 'ThickFat', ['ingrain', 'bodyslam']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['ingrain', 'bodyslam']), turns: 5, want: seq([[1, 1], [2, 2]]) },
  ig_speedboost: { p1: set('Cradily', 'Leftovers', 'SuctionCups', ['ingrain', 'splash']), p2: set('Ninjask', '', 'SpeedBoost', ['aerialace', 'splash']), turns: 4, want: seq([[1, 1], [2, 1]]) },
  ig_snatch: { p1: set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), p2: set('Sableye', '', 'KeenEye', ['snatch', 'splash']), turns: 3, want: seq([[1, 1], [2, 2]]) },
  ig_sleeptalk: { p1: set('Cradily', '', 'SuctionCups', ['ingrain', 'sleeptalk']), p2: set('Snorlax', '', 'ThickFat', ['spore', 'bodyslam']), turns: 4, want: seq([[2, 1], [2, 2]]) },
  ig_wrap: { p1: set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), p2: set('Dragonite', '', 'InnerFocus', ['wrap', 'splash']), turns: 5, want: seq([[1, 1], [2, 1], [2, 2]]) },
  ig_meanlook: { p1: set('Cradily', '', 'SuctionCups', ['ingrain', 'splash']), p2: set('Umbreon', '', 'Synchronize', ['meanlook', 'splash']), turns: 4, want: seq([[1, 1], [2, 2]]) },
  ig_ko: { p1: team(set('Cradily', '', 'SuctionCups', ['ingrain', 'splash'], { lv: 20 }), BENCH), p2: set('Snorlax', '', 'ThickFat', ['earthquake', 'splash']), turns: 4, want: seq([[1, 2], [2, 1], [2, 1]]) },
  // ---------------------------------------------------------------- REST UNDER UPROAR
  ru_basic: { p1: set('Snorlax', '', 'ThickFat', ['rest', 'splash']), p2: set('Exploud', '', 'Soundproof', ['uproar']), turns: 4, want: seq([[2, 1], [1, 1]]) },
  ru_ou: { format: 'gen3ou', p1: set('Snorlax', '', 'ThickFat', ['rest', 'bodyslam']), p2: set('Exploud', '', 'Soundproof', ['uproar']), turns: 4, want: seq([[2, 1], [1, 1]]) },
  ru_after: { p1: set('Snorlax', '', 'ThickFat', ['rest', 'splash']), p2: set('Exploud', '', 'Soundproof', ['uproar', 'splash']), turns: 7, want: seq([[2, 1], [1, 1], [1, 2]]) },
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
const re = (r, rx) => r.lines.some((l) => rx.test(l));
const cov = {
  spiteActivate: n((r) => has(r, '|move: Spite|')),
  spiteFail: n((r) => r.lines.some((l, i) => /\|Spite\|\|\[still\]$/.test(l) && (r.lines[i + 1] || '').startsWith('|-fail|'))),
  spiteSub: n((r) => r.name === 'sp_sub' && has(r, '|move: Spite|')),
  spiteProtect: n((r) => r.name === 'sp_protect' && has(r, '|-activate|p2a: Jolteon|Protect')),
  spiteSleepTalk: n((r) => has(r, '|move: Spite|sleeptalk|')),
  spiteNopp: n((r) => r.lines.some((l) => /^\|cant\|[^|]+\|nopp\|/.test(l))),
  spiteStruggle: n((r) => r.name === 'sp_drain_struggle' && has(r, '|Struggle|') && has(r, '|Spite||[still]')),
  grudgeStart: n((r) => has(r, '|-singlemove|') && has(r, '|Grudge')),
  grudgeActivate: n((r) => has(r, '|move: Grudge|')),
  grudgeRecoil: n((r) => r.name === 'gr_recoil' && has(r, '|move: Grudge|Double-Edge')),
  grudgeSleepTalk: n((r) => has(r, '|move: Grudge|Sleep Talk')),
  grudgePursuit: n((r) => has(r, '|move: Grudge|Pursuit')),
  grudgeNoActivate: n((r) => ['gr_removed', 'gr_slept', 'gr_explosion', 'gr_futuresight'].includes(r.name) && has(r, '|faint|p1a') && !has(r, 'move: Grudge|')),
  snoreHit: n((r) => r.lines.some((l, i) => l.includes('|Snore|') && i > 0 && r.lines[i - 1].startsWith('|cant|') && (r.lines[i + 1] || '').startsWith('|-damage|'))),
  snoreAwake: n((r) => r.lines.some((l, i) => /\|Snore\|p2a: [^|]+$/.test(l) && !/^\|-(damage|immune|activate|crit|supereffective|resisted|miss)/.test(r.lines[i + 1] || ''))),
  snoreFlinch: n((r) => r.name === 'sn_flinch' && has(r, '|cant|p2a: Snorlax|flinch')),
  snoreViaSleepTalk: n((r) => has(r, '|Snore|') && has(r, '[from] Sleep Talk') && r.lines.some((l) => l.includes('|Snore|') && l.includes('[from] Sleep Talk'))),
  snoreSoundproof: n((r) => r.name === 'sn_soundproof' && has(r, 'ability: Soundproof')),
  nightmareStart: n((r) => has(r, '|Nightmare') && has(r, '|-start|')),
  nightmareChip: n((r) => has(r, '[from] Nightmare')),
  nightmareFail: n((r) => r.lines.some((l, i) => /\|Nightmare\|\|\[still\]$/.test(l) && (r.lines[i + 1] || '').startsWith('|-fail|'))),
  nightmareWakeEnd: n((r) => r.lines.some((l, i) => l.startsWith('|-curestatus|') && (r.lines[i + 1] || '').endsWith('|Nightmare|[silent]'))),
  nightmareNaturalCure: n((r) => r.name === 'nm_naturalcure' && has(r, 'Natural Cure') && has(r, '|Nightmare|[silent]')),
  nightmareShedSkin: n((r) => r.name === 'nm_shedskin' && has(r, 'Shed Skin') && has(r, '|Nightmare|[silent]')),
  nightmareCureSitesOther: n((r) => ['nm_healbell', 'nm_aromatherapy', 'nm_chesto_trick', 'nm_skillswap'].includes(r.name) && has(r, '|Nightmare|[silent]')),
  nightmareTwoChips: n((r) => {
    for (let i = 0; i + 1 < r.lines.length; i++) if (r.lines[i].endsWith('[from] Nightmare') && r.lines[i + 1].endsWith('[from] Nightmare')) return true;
    return false;
  }),
  ingrainStart: n((r) => has(r, '|move: Ingrain') && has(r, '|-start|')),
  ingrainHeal: n((r) => has(r, '[from] Ingrain')),
  ingrainDragBlock: n((r) => has(r, '|-activate|p1a: Cradily|move: Ingrain')),
  ingrainSuctionCups: n((r) => r.name === 'ig_suctioncups' && has(r, 'ability: Suction Cups')),
  ingrainBatonPass: n((r) => r.name === 'ig_batonpass' && has(r, '[from] Baton Pass') && has(r, '|-heal|p1a: Blissey') && has(r, '[from] Ingrain')),
  ingrainSnatch: n((r) => has(r, '|Ingrain|p2a: Sableye|[from] Snatch')),
  ingrainTwoHeals: n((r) => {
    for (let i = 0; i + 1 < r.lines.length; i++) if (r.lines[i].endsWith('[from] Ingrain') && r.lines[i + 1].endsWith('[from] Ingrain')) return true;
    return false;
  }),
  restUproar: n((r) => has(r, '|slp|[from] Uproar')),
  restUproarOu: n((r) => r.name === 'ru_ou' && has(r, '|slp|[from] Uproar')),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0 && !process.env.B5_NOFLOOR) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# B5 MOVES GOLDEN (gen3_b5_moves_v1): Spite, Grudge, Snore, Nightmare, Ingrain, Rest under Uproar',
  `# generated by harness/gen_b5_moves_golden.js  rows/board=${N}  boards=${Object.keys(BOARDS).length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the format, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE /',
  '# SEED_AFTER, the FRAMING window and the post-|turn|1 omniscient lines (|t:| dropped).',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
