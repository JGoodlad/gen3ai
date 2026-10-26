// GOLDEN GENERATOR — the ATTRACT move (`gen3_attract_move_v1`, gen3 full parity B9). The settled
// spec is `harness/probe_attract_move.js`: the move's `onTryImmunity` (an M<->F pair only) sets the
// natural immunity, the accuracy roll is drawn anyway, Protect shows before the immunity, the volatile
// add fails `[still]` + `-fail` on an already-attracted or Oblivious target, a Substitute and Safeguard
// do not block it, and the success line is a bare `|-start|<t>|Attract`. Every set pins its GENDER.
//
//   node src/rust_sim/harness/gen_attract_move_golden.js -> tests/vectors/attract_move_golden.txt
//   cargo test --test attract_move_golden_test
// env: AM_SEEDS (rows per board, default 8), AM_OUT.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.AM_SEEDS || '8', 10);
const OUT = process.env.AM_OUT || path.resolve(__dirname, '../tests/vectors/attract_move_golden.txt');

const set = (sp, item, ab, moves, o = {}) =>
  `${sp}|${sp}|${item}|${ab}|${moves.join(',')}|${o.nature || 'Hardy'}|${o.evs || '85,85,85,85,85,85'}|${o.g || 'M'}|||${o.lv || ''}|`;
const team = (...s) => s.join(']');
const CASTER = (g = 'M', item = 'Leftovers') => set('Nidoking', item, 'PoisonPoint', ['attract', 'splash', 'tackle'], { g });
const QUEEN = (moves = ['splash', 'tackle'], o = {}) => set('Nidoqueen', o.item || 'Leftovers', 'PoisonPoint', moves, { g: 'F', ...o });
const M = (a, b) => () => [`move ${a}`, `move ${b}`];
const BOARDS = {
  plain: { p1: CASTER(), p2: QUEEN(), turns: 5, want: (t) => [t === 0 ? 'move 1' : 'move 3', 'move 2'] },
  same_gender: { p1: CASTER(), p2: set('Nidoking', 'Leftovers', 'PoisonPoint', ['splash'], { g: 'M' }), turns: 2, want: M(1, 1) },
  genderless: { p1: CASTER(), p2: set('Magneton', 'Leftovers', 'MagnetPull', ['splash'], { g: 'N' }), turns: 2, want: M(1, 1) },
  twice: { p1: CASTER(), p2: QUEEN(), turns: 3, want: M(1, 1) },
  oblivious: { p1: CASTER(), p2: set('Slowbro', 'Leftovers', 'Oblivious', ['splash'], { g: 'F' }), turns: 2, want: M(1, 1) },
  substitute: { p1: CASTER(), p2: QUEEN(['substitute', 'splash']), turns: 3, want: (t) => [t === 0 ? 'move 2' : 'move 1', t === 0 ? 'move 1' : 'move 2'] },
  safeguard: { p1: CASTER(), p2: QUEEN(['safeguard', 'splash']), turns: 3, want: (t) => [t === 0 ? 'move 2' : 'move 1', t === 0 ? 'move 1' : 'move 2'] },
  protect: { p1: CASTER(), p2: QUEEN(['protect', 'splash'], { evs: '85,85,85,85,85,252' }), turns: 3, want: (t) => ['move 1', t % 2 === 0 ? 'move 1' : 'move 2'] },
  protect_immune: { p1: CASTER(), p2: set('Magneton', 'Leftovers', 'MagnetPull', ['protect', 'splash'], { g: 'N', evs: '85,85,85,85,85,252' }), turns: 2, want: M(1, 1) },
  // Evasion so the 100-accuracy move can MISS: a miss on an immune target shows -immune.
  evasive_f: { p1: CASTER(), p2: QUEEN(['doubleteam', 'splash'], { evs: '85,85,85,85,85,252' }), turns: 6, want: (t) => [t < 3 ? 'move 2' : 'move 1', t < 3 ? 'move 1' : 'move 2'] },
  evasive_m: { p1: CASTER(), p2: set('Nidoking', 'Leftovers', 'PoisonPoint', ['doubleteam', 'splash'], { g: 'M', evs: '85,85,85,85,85,252' }), turns: 6, want: (t) => [t < 3 ? 'move 2' : 'move 1', t < 3 ? 'move 1' : 'move 2'] },
  mental_herb: { p1: CASTER(), p2: QUEEN(['splash', 'tackle'], { item: 'Mental Herb' }), turns: 4, want: (t) => ['move 1', 'move 2'] },
  source_leaves: { p1: team(CASTER(), set('Blissey', 'Leftovers', 'NaturalCure', ['splash'], { g: 'F' })), p2: QUEEN(), turns: 4, want: (t) => [t === 1 ? 'switch 2' : 'move 1', 'move 2'] },
  holder_leaves: { p1: CASTER(), p2: team(QUEEN(), set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], { g: 'F' })), turns: 4, want: (t) => ['move 1', t === 1 ? 'switch 2' : t === 2 ? 'switch 2' : 'move 1'] },
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
    const seed = [37 + k * 7, 53 + k * 13, 67 + k * 29, 79 + k * 51];
    rows.push({ name, p1: b.p1, p2: b.p2, ...run(b, seed) });
  }
}
const n = (pred) => rows.filter(pred).length;
const has = (r, s) => r.lines.some((l) => l.includes(s));
const cov = {
  start: n((r) => has(r, '|-start|p2a: Nidoqueen|Attract')),
  immune: n((r) => has(r, '|-immune|p2a:')),
  missImmune: n((r) => (r.name === 'evasive_m') && has(r, '|-immune|p2a:')),
  missPlain: n((r) => r.name === 'evasive_f' && has(r, '|-miss|p1a: Nidoking')),
  fail: n((r) => has(r, '|-fail|p1a: Nidoking')),
  protect: n((r) => r.name.startsWith('protect') && has(r, '|-activate|p2a:') && has(r, '|Protect')),
  sub: n((r) => r.name === 'substitute' && has(r, '|-start|p2a: Nidoqueen|Attract')),
  safeguard: n((r) => r.name === 'safeguard' && has(r, '|-start|p2a: Nidoqueen|Attract')),
  immobilized: n((r) => has(r, '|cant|p2a: Nidoqueen|Attract')),
  mentalHerb: n((r) => has(r, '[from] item: Mental Herb')),
  sourceLeft: n((r) => r.name === 'source_leaves' && has(r, '|Attract|[silent]')),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# ATTRACT MOVE GOLDEN (gen3_attract_move_v1)',
  `# generated by harness/gen_attract_move_golden.js  rows/board=${N}  boards=${Object.keys(BOARDS).length}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the decisions as played ([p1, p2], null = no choice), SEED_BEFORE / SEED_AFTER,',
  '# the FRAMING window and the post-|turn|1 omniscient lines (|t:| dropped). gen3customgame.',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
