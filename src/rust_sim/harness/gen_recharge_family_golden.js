// GOLDEN GENERATOR — the RECHARGE family (`gen3_recharge_family_v1`, gen3 full parity B2).
//
// Blast Burn, Frenzy Plant and Hydro Cannon resolve in gen 3 to Hyper Beam's exact shape: 150 BP,
// 90 accuracy, `self: {volatileStatus: 'mustrecharge'}`, `flags.recharge`, no handler of their
// own (`Dex.mod('gen3').moves.get(id)`; their only difference is the type). The engine modelled
// the recharge for `hyperbeam` alone and fail-louded the three. Every board here runs all four
// (Hyper Beam is the CONTROL: it has a committed golden of its own, so a row where it agrees and a
// sibling does not localizes the defect to the sibling). The boards cover the branches where a
// type matters: the absorbing abilities (Flash Fire takes Blast Burn, Water Absorb takes Hydro
// Cannon: no damage, so no lock) and the Ghost immunity of the Normal control.
//
//   node src/rust_sim/harness/gen_recharge_family_golden.js -> tests/vectors/recharge_family_golden.txt
//   cargo test --test recharge_family_golden_test
// env: RF_SEEDS (rows per board per move, default 6), RF_OUT.
//
// The sim is driven synchronously; each decision boundary answers the side's OWN request (a forced
// switch takes the first live bench slot, a `wait` side passes), and the decisions are recorded
// as played, `null` for a side with no choice, which is the port's `ScriptDecision` shape.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.RF_SEEDS || '6', 10);
const OUT = process.env.RF_OUT || path.resolve(__dirname, '../tests/vectors/recharge_family_golden.txt');
const MOVES = ['hyperbeam', 'blastburn', 'frenzyplant', 'hydrocannon'];

const set = (species, ability, moves, o = {}) =>
  `${species}|${species}|${o.item || 'Leftovers'}|${ability}|${moves.join(',')}|Hardy|${o.evs || '85,85,85,85,85,85'}|M|||${o.lv && o.lv !== 100 ? o.lv : ''}|`;
const team = (...s) => s.join(']');

// Each board: the teams for move `m`, and each side's intended slot per turn index.
const BOARDS = {
  cycle: (m) => ({ p1: team(set('Snorlax', 'ThickFat', [m, 'splash'])), p2: team(set('Blissey', 'NaturalCure', ['splash'], { evs: '252,0,252,0,252,0' })), turns: 5, want: () => [1, 1] }),
  protect: (m) => ({ p1: team(set('Snorlax', 'ThickFat', [m, 'splash'])), p2: team(set('Skarmory', 'KeenEye', ['protect', 'splash'])), turns: 5, want: (t) => [1, t % 2 === 0 ? 1 : 2] }),
  sub: (m) => ({ p1: team(set('Snorlax', 'ThickFat', [m, 'splash'])), p2: team(set('Jolteon', 'VoltAbsorb', ['substitute', 'splash'], { evs: '252,0,0,0,0,252' })), turns: 5, want: (t) => [1, t % 3 === 0 ? 1 : 2] }),
  ko: (m) => ({
    p1: team(set('Snorlax', 'ThickFat', [m, 'splash'], { evs: '0,0,0,252,0,0' })),
    p2: team(set('Alakazam', 'Synchronize', ['splash'], { lv: 30 }), set('Jolteon', 'VoltAbsorb', ['splash'], { lv: 30 }), set('Espeon', 'Synchronize', ['splash'], { lv: 30 })),
    turns: 7, want: () => [1, 1],
  }),
  // The type-specific blockers: Flash Fire (Fire), Water Absorb (Water), a Ghost (Normal).
  flashfire: (m) => ({ p1: team(set('Snorlax', 'ThickFat', [m, 'splash'])), p2: team(set('Houndoom', 'FlashFire', ['splash'])), turns: 3, want: () => [1, 1] }),
  waterabsorb: (m) => ({ p1: team(set('Snorlax', 'ThickFat', [m, 'splash'])), p2: team(set('Lapras', 'WaterAbsorb', ['splash'])), turns: 3, want: () => [1, 1] }),
  ghost: (m) => ({ p1: team(set('Snorlax', 'ThickFat', [m, 'splash'])), p2: team(set('Gengar', 'Levitate', ['splash'])), turns: 3, want: () => [1, 1] }),
  paralyzed: (m) => ({ p1: team(set('Snorlax', 'ThickFat', [m, 'splash'])), p2: team(set('Jolteon', 'VoltAbsorb', ['thunderwave', 'splash'])), turns: 5, want: (t) => [1, t === 0 ? 1 : 2] }),
  truant: (m) => ({ p1: team(set('Slaking', 'Truant', [m, 'splash'])), p2: team(set('Blissey', 'NaturalCure', ['splash'], { evs: '252,0,252,0,252,0' })), turns: 6, want: () => [1, 1] }),
};

function answer(side, want) {
  const req = side.activeRequest;
  if (!req || req.wait) return null;
  if (req.forceSwitch) {
    const k = side.pokemon.findIndex((p, i) => i > 0 && !p.fainted) + 1;
    return `switch ${k}`;
  }
  const moves = req.active[0].moves;
  // A locked turn (the recharge) offers one move: choose it.
  return moves.length === 1 ? 'move 1' : `move ${want}`;
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
  while (!battle.ended && t < board.turns + 4) {
    const forced = battle.sides.some((s) => s.activeRequest && s.activeRequest.forceSwitch);
    const [w1, w2] = board.want(t);
    const d = [answer(battle.sides[0], w1), answer(battle.sides[1], w2)];
    if (d[0] === null && d[1] === null) break;
    if (d[0] !== null) battle.choose('p1', d[0]);
    if (d[1] !== null) battle.choose('p2', d[1]);
    script.push(d);
    if (!forced) t++;
    if (t >= board.turns) break;
  }
  const omni = [];
  for (let i = 0; i < battle.log.length; i++) {
    const l = battle.log[i];
    if (l.startsWith('|split|')) { omni.push(battle.log[i + 1]); i += 2; continue; }
    omni.push(l);
  }
  const t1 = omni.indexOf('|turn|1');
  const lines = omni.slice(t1 + 1).filter((l) => !(l === '' || l.startsWith('|t:|') || l.startsWith('|debug|') || l.startsWith('|error|')));
  return { seedBefore, seedAfter: battle.prng.getSeed(), script, lines };
}

const rows = [];
for (const name of Object.keys(BOARDS)) {
  for (const m of MOVES) {
    for (let k = 0; k < N; k++) {
      const seed = [17 + k * 7, 29 + k * 13, 43 + k * 29, 59 + k * 51];
      const b = BOARDS[name](m);
      rows.push({ name, move: m, p1: b.p1, p2: b.p2, ...run(b, seed) });
    }
  }
}
const n = (pred) => rows.filter(pred).length;
const has = (r, re) => r.lines.some((l) => re.test(l));
const cov = {
  mustrecharge: n((r) => r.move !== 'hyperbeam' && has(r, /^\|-mustrecharge\|/)),
  rechargeCant: n((r) => r.move !== 'hyperbeam' && has(r, /^\|cant\|.*\|recharge$/)),
  miss: n((r) => r.move !== 'hyperbeam' && has(r, /^\|-miss\|/)),
  protectBlock: n((r) => r.move !== 'hyperbeam' && has(r, /^\|-activate\|.*\|Protect$/)),
  subHit: n((r) => r.move !== 'hyperbeam' && has(r, /\|Substitute\|\[damage\]$|^\|-end\|.*\|Substitute$/)),
  koForced: n((r) => r.move !== 'hyperbeam' && r.script.some((d) => d[0] === null)),
  flashFire: n((r) => r.move === 'blastburn' && has(r, /Flash Fire/)),
  waterAbsorb: n((r) => r.move === 'hydrocannon' && has(r, /Water Absorb/)),
  paraRecharge: n((r) => r.move !== 'hyperbeam' && r.name === 'paralyzed' && has(r, /^\|cant\|.*\|recharge$/)),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# RECHARGE FAMILY GOLDEN (gen3_recharge_family_v1)',
  `# generated by harness/gen_recharge_family_golden.js  rows/board/move=${N}  moves=${MOVES.join(',')}`,
  `# rows=${rows.length}  COVERAGE (siblings only): ${JSON.stringify(cov)}`,
  '# Each row: the board, the decisions as played ([p1, p2], null = no choice), the sim SEED_BEFORE /',
  '# SEED_AFTER, and the post-|turn|1 omniscient lines (|t:| dropped). gen3customgame.',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
