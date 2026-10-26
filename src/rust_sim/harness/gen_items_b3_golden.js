// GOLDEN GENERATOR — the B3 items (`gen3_items_b3_v1`, gen3 full parity B3): Shell Bell, Macho
// Brace, Mail, Mental Herb. The settled spec is `harness/probe_items_b3.js`'s header. Each board
// runs over N seeds; each row records the decisions as played, the sim SEED_BEFORE / SEED_AFTER,
// the FRAMING window (the turn-0 lead order is Macho-Brace-sensitive: the slower lead's weather
// wins) and the post-`|turn|1` omniscient lines.
//
//   node src/rust_sim/harness/gen_items_b3_golden.js -> tests/vectors/items_b3_golden.txt
//   cargo test --test items_b3_golden_test
// env: IB_SEEDS (rows per board, default 8), IB_OUT.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.IB_SEEDS || '8', 10);
const OUT = process.env.IB_OUT || path.resolve(__dirname, '../tests/vectors/items_b3_golden.txt');

const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const BLISSEY = set('Blissey', 'Leftovers', 'NaturalCure', ['seismictoss', 'splash'], { g: 'F' });

// want(t) -> [p1 choice, p2 choice] for the t-th regular turn (a locked / forced request is
// answered by the driver).
const M = (a, b) => () => [`move ${a}`, `move ${b}`];
const BOARDS = {
  sb_plain: { p1: set('Snorlax', 'Shell Bell', 'ThickFat', ['bodyslam', 'splash']), p2: BLISSEY, turns: 5, want: M(1, 1) },
  sb_multihit: { p1: set('Cloyster', 'Shell Bell', 'ShellArmor', ['spikecannon', 'splash']), p2: BLISSEY, turns: 4, want: M(1, 1) },
  sb_sub: { p1: set('Snorlax', 'Shell Bell', 'ThickFat', ['bodyslam', 'splash']), p2: set('Jolteon', 'Leftovers', 'VoltAbsorb', ['substitute', 'seismictoss']), turns: 5, want: (t) => ['move 1', t % 2 === 0 ? 'move 1' : 'move 2'] },
  sb_doubleedge: { p1: set('Snorlax', 'Shell Bell', 'ThickFat', ['doubleedge', 'splash']), p2: BLISSEY, turns: 3, want: M(1, 1) },
  sb_fixed: { p1: set('Snorlax', 'Shell Bell', 'ThickFat', ['seismictoss', 'splash']), p2: BLISSEY, turns: 3, want: M(1, 1) },
  sb_ko: { p1: set('Snorlax', 'Shell Bell', 'ThickFat', ['seismictoss', 'splash']), p2: team(set('Pichu', '', 'Static', ['splash'], { lv: 5 }), BLISSEY), turns: 3, want: M(1, 1) },
  sb_beatup: {
    p1: team(set('Sneasel', 'Shell Bell', 'InnerFocus', ['beatup', 'splash']), set('Snorlax', '', 'ThickFat', ['splash']), set('Ursaring', '', 'Guts', ['splash'])),
    p2: BLISSEY, turns: 3, want: M(1, 1),
  },
  sb_pursuit: {
    p1: set('Tyranitar', 'Shell Bell', 'SandStream', ['pursuit', 'splash']),
    p2: team(set('Alakazam', 'Leftovers', 'Synchronize', ['seismictoss', 'splash']), BLISSEY),
    turns: 3, want: (t) => ['move 1', t === 1 ? 'switch 2' : 'move 1'],
  },
  sb_sleeptalk: {
    p1: set('Snorlax', 'Shell Bell', 'ThickFat', ['rest', 'sleeptalk', 'bodyslam']), p2: BLISSEY, turns: 4,
    want: (t) => [t === 0 ? 'move 3' : t === 1 ? 'move 1' : 'move 2', 'move 1'],
  },
  mb_order: { p1: set('Jolteon', 'Macho Brace', 'VoltAbsorb', ['tackle']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['tackle'], { evs: '0,0,0,0,0,252' }), turns: 3, want: M(1, 1) },
  mb_control: { p1: set('Jolteon', 'Leftovers', 'VoltAbsorb', ['tackle']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['tackle'], { evs: '0,0,0,0,0,252' }), turns: 3, want: M(1, 1) },
  // The turn-0 lead order: a raw-faster Drought Ninetales halved below a Drizzle Politoed fires
  // LAST, so its sun wins (the control's rain wins).
  mb_lead_weather: { p1: set('Ninetales', 'Macho Brace', 'Drought', ['splash']), p2: set('Politoed', 'Leftovers', 'Drizzle', ['splash']), turns: 2, want: M(1, 1) },
  mb_lead_weather_control: { p1: set('Ninetales', 'Leftovers', 'Drought', ['splash']), p2: set('Politoed', 'Leftovers', 'Drizzle', ['splash']), turns: 2, want: M(1, 1) },
  // A Macho-Brace speed TIE (Jolteon 300 halved = a 150 mon): the action-order shuffle draws.
  // Jolteon 359 Speed through ×0.5 is 179 (the 4096 fixed point), a Clefable with 92 Speed EVs is
  // 179: a tie, so the action-order shuffle draws every turn.
  mb_tie: { p1: set('Jolteon', 'Macho Brace', 'VoltAbsorb', ['tackle'], { evs: '0,0,0,0,0,252' }), p2: set('Clefable', 'Leftovers', 'ThickFat', ['tackle'], { evs: '0,0,0,0,0,92' }), turns: 3, want: M(1, 1) },
  mail_trick_target: { p1: set('Alakazam', 'Choice Band', 'Synchronize', ['trick', 'splash']), p2: set('Snorlax', 'Mail', 'ThickFat', ['splash']), turns: 2, want: M(1, 1) },
  mail_trick_user: { p1: set('Alakazam', 'Mail', 'Synchronize', ['trick', 'splash']), p2: set('Snorlax', 'Leftovers', 'ThickFat', ['splash']), turns: 2, want: M(1, 1) },
  mail_thief: { p1: set('Sneasel', '', 'InnerFocus', ['thief', 'splash']), p2: set('Snorlax', 'Mail', 'ThickFat', ['splash']), turns: 2, want: M(1, 1) },
  mail_knockoff: { p1: set('Sneasel', '', 'InnerFocus', ['knockoff', 'splash']), p2: set('Snorlax', 'Mail', 'ThickFat', ['splash']), turns: 2, want: M(1, 1) },
  mail_covet: { p1: set('Sneasel', '', 'InnerFocus', ['covet', 'splash']), p2: set('Snorlax', 'Mail', 'ThickFat', ['splash']), turns: 2, want: M(1, 1) },
  mh_cutecharm: { p1: set('Snorlax', 'Mental Herb', 'ThickFat', ['tackle', 'splash']), p2: set('Clefable', 'Leftovers', 'CuteCharm', ['splash'], { g: 'F' }), turns: 5, want: M(1, 1) },
  mh_control: { p1: set('Snorlax', 'Leftovers', 'ThickFat', ['tackle', 'splash']), p2: set('Clefable', 'Leftovers', 'CuteCharm', ['splash'], { g: 'F' }), turns: 5, want: M(1, 1) },
};

function answer(side, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) return `switch ${side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1}`;
  return req.active[0].moves.length === 1 && want.startsWith('move') ? 'move 1' : want;
}

function run(board, seed) {
  const battle = new Battle({ formatid: 'gen3customgame', seed });
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
    if (d[0] !== null) battle.choose('p1', d[0]);
    if (d[1] !== null) battle.choose('p2', d[1]);
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
  for (let k = 0; k < N; k++) {
    const seed = [31 + k * 7, 47 + k * 13, 61 + k * 29, 73 + k * 51];
    rows.push({ name, p1: b.p1, p2: b.p2, ...run(b, seed) });
  }
}
const n = (pred) => rows.filter(pred).length;
const has = (r, s) => r.lines.some((l) => l.includes(s));
const cov = {
  shellBellHeal: n((r) => has(r, '[from] item: Shell Bell')),
  shellBellMultihit: n((r) => r.name === 'sb_multihit' && has(r, '[from] item: Shell Bell')),
  shellBellBeatUp: n((r) => r.name === 'sb_beatup' && has(r, '[from] item: Shell Bell')),
  shellBellPursuit: n((r) => r.name === 'sb_pursuit' && has(r, '[from] item: Shell Bell')),
  // The heal of the move Sleep Talk CALLED (its own useMove), not the turn-1 Body Slam's.
  shellBellSleepTalk: n((r) => r.lines.some((l, i) => l.endsWith('|[from] Sleep Talk') &&
    r.lines.slice(i + 1, i + 4).some((x) => x.includes('[from] item: Shell Bell')))),
  machoBraceTie: n((r) => r.name === 'mb_tie' && r.lines.some((l) => l.startsWith('|move|p1a')) &&
    r.lines.findIndex((l) => l.startsWith('|move|p1a')) < r.lines.findIndex((l) => l.startsWith('|move|p2a'))),
  machoBraceSun: n((r) => r.name === 'mb_lead_weather' && r.framing.some((l) => l.startsWith('|-weather|SunnyDay'))),
  mailTrickFail: n((r) => r.name.startsWith('mail_trick') && has(r, '|-fail|p1a: Alakazam')),
  mailThief: n((r) => r.name === 'mail_thief' && has(r, '|-item|p1a: Sneasel|Mail')),
  mentalHerb: n((r) => has(r, '[from] item: Mental Herb')),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# B3 ITEMS GOLDEN (gen3_items_b3_v1): Shell Bell, Macho Brace, Mail, Mental Herb',
  `# generated by harness/gen_items_b3_golden.js  rows/board=${N}  boards=${Object.keys(BOARDS).length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE / SEED_AFTER,',
  '# the FRAMING window and the post-|turn|1 omniscient lines (|t:| dropped). gen3customgame.',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
