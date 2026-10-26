// GOLDEN GENERATOR — the gen-3 FORMAT differential (`gen3_format_rules_v1`, gen3 full parity B1).
//
// The engine used to run every format id except gen3customgame as gen3ou. Every gen-3 singles
// format Showdown defines (bar the refused `gen31v1`) is driven here through the same boards, and
// the port must match each one's draws and bytes:
//   twave    — one status application: OU / UUBL / Draft gather BOTH clause `onSetStatus` handlers
//              and draw the size-2 shuffle; the Sleep-Clause-only tiers and Custom Game draw none.
//   sleep2   — a second foe sleep: blocked by Sleep Clause Mod everywhere but Custom Game.
//   freeze2  — a second freeze: blocked only where Freeze Clause Mod is (OU / UUBL / Draft).
//              Seed-SEARCHED per format (two 10 % freezes), so every format carries the case.
//   rest     — a self-Rest reaches runEvent('SetStatus'): the same shuffle-or-not.
//   beatup   — Beat Up Nicknames Mod: the per-strike `-activate` appears only without it.
//   leak     — the request's `maybeTrapped` against a Sand Veil Dugtrio: the unheld-Arena-Trap
//              leak runs where abilities are obtainable, except where the banlist removes Arena
//              Trap (UU / RU).
// Every row also carries the FRAMING window (`|gametype|` .. `|turn|1`): the `|tier|` and
// `|rule|` lines are the format's.
//
//   node src/rust_sim/harness/gen_format_rules_golden.js -> tests/vectors/format_rules_golden.txt
//   cargo test --test format_rules_golden_test
// env: FR_SEEDS (rows per plain scenario per format, default 4), FR_OUT.
//
// The sim is driven SYNCHRONOUSLY (`Battle.choose`) and its raw log is resolved to the
// omniscient view: a `|split|pN` marker is followed by the owner's exact line, then the
// spectator's percentage line; the omniscient view keeps the first.
'use strict';
const fs = require('fs');
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));

const N = parseInt(process.env.FR_SEEDS || '4', 10);
const OUT = process.env.FR_OUT || path.resolve(__dirname, '../tests/vectors/format_rules_golden.txt');
const FORMATS = JSON.parse(fs.readFileSync(path.resolve(__dirname, '../tests/vectors/gen3_formats.json'), 'utf8'))
  .supported.map((r) => r.id);

// A packed set; `lv` is blank at 100 (Teams.pack's form), the gender pinned so construction draws none.
const set = (species, item, ability, moves, lv) =>
  `${species}|${species}|${item}|${ability}|${moves.join(',')}|Hardy|85,85,85,85,85,85|M|||${lv === 100 ? '' : lv}|`;
const team = (...sets) => sets.join(']');

const BOARDS = {
  twave: (lv) => ({
    p1: team(set('Jolteon', 'Leftovers', 'VoltAbsorb', ['thunderwave', 'splash'], lv)),
    p2: team(set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], lv)),
    script: [['move 1', 'move 1']],
  }),
  sleep2: (lv) => ({
    p1: team(set('Breloom', 'Leftovers', 'EffectSpore', ['spore', 'splash'], lv)),
    p2: team(set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], lv), set('Blissey', 'Leftovers', 'NaturalCure', ['splash'], lv)),
    script: [['move 1', 'move 1'], ['move 2', 'switch 2'], ['move 1', 'move 1']],
  }),
  freeze2: (lv) => ({
    p1: team(set('Jynx', 'Leftovers', 'Oblivious', ['icebeam', 'splash'], lv)),
    p2: team(set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], lv), set('Blissey', 'Leftovers', 'NaturalCure', ['splash'], lv)),
    script: [['move 1', 'move 1'], ['move 2', 'switch 2'], ['move 1', 'move 1']],
  }),
  rest: (lv) => ({
    p1: team(set('Snorlax', 'Leftovers', 'ThickFat', ['rest', 'splash'], lv)),
    p2: team(set('Jolteon', 'Leftovers', 'VoltAbsorb', ['thunderbolt', 'splash'], lv)),
    script: [['move 2', 'move 1'], ['move 1', 'move 2']],
  }),
  beatup: (lv) => ({
    p1: team(set('Sneasel', 'Leftovers', 'InnerFocus', ['beatup'], lv), set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], lv),
      set('Blissey', 'Leftovers', 'NaturalCure', ['splash'], lv)),
    p2: team(set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], lv)),
    script: [['move 1', 'move 1']],
  }),
  leak: (lv) => ({
    p1: team(set('Snorlax', 'Leftovers', 'ThickFat', ['splash'], lv), set('Blissey', 'Leftovers', 'NaturalCure', ['splash'], lv)),
    p2: team(set('Dugtrio', 'Leftovers', 'SandVeil', ['splash'], lv)),
    script: [['move 1', 'move 1']],
  }),
};

function run(format, board, seed) {
  const battle = new Battle({ formatid: format, seed });
  battle.setPlayer('p1', { name: 'P1', team: board.p1 });
  battle.setPlayer('p2', { name: 'P2', team: board.p2 });
  // A packed string Teams.unpack rejects makes Showdown GENERATE a random team in its place;
  // refuse rather than record a battle between the wrong teams.
  for (const [i, side] of [board.p1, board.p2].entries()) {
    const want = side.split(']').map((s) => s.split('|')[1]);
    const got = battle.sides[i].pokemon.map((p) => p.species.name);
    if (JSON.stringify(want) !== JSON.stringify(got)) throw new Error(`${format}: team ${i} unpacked as ${got}`);
  }
  const seedBefore = battle.prng.getSeed();
  for (const [a, b] of board.script) {
    battle.choose('p1', a);
    battle.choose('p2', b);
  }
  const seedAfter = battle.prng.getSeed();
  const omni = [];
  for (let i = 0; i < battle.log.length; i++) {
    const l = battle.log[i];
    if (l.startsWith('|split|')) { omni.push(battle.log[i + 1]); i += 2; continue; }
    omni.push(l);
  }
  const drop = (l) => l === '' || l.startsWith('|t:|') || l.startsWith('|debug|') || l.startsWith('|error|');
  const t1 = omni.indexOf('|turn|1');
  const g = omni.indexOf('|gametype|singles');
  if (t1 < 0 || g < 0) throw new Error(`${format}: no framing window`);
  return {
    seedBefore,
    seedAfter,
    framing: omni.slice(g, t1 + 1).filter((l) => !drop(l)),
    lines: omni.slice(t1 + 1).filter((l) => !drop(l)),
    // The sim's `pokemon.maybeTrapped` as endTurn leaves it (what `is_maybe_trapped` models).
    maybeTrapped: !!battle.sides[0].active[0].maybeTrapped,
    ended: battle.ended,
  };
}

const rows = [];
for (const format of FORMATS) {
  const lv = format === 'gen3lc' ? 5 : 100;
  for (const name of ['twave', 'sleep2', 'rest', 'beatup', 'leak']) {
    for (let k = 0; k < N; k++) {
      const seed = [11 + k * 7, 23 + k * 13, 37 + k * 29, 41 + k * 51];
      const b = BOARDS[name](lv);
      rows.push({ name, format, p1: b.p1, p2: b.p2, script: b.script, seed, ...run(format, b, seed) });
    }
  }
  // freeze2: search for seeds whose turn 1 freezes the first foe and whose turn 3 lands a second
  // freeze roll (blocked by the clause, or applied without it).
  let found = 0;
  for (let k = 0; k < 20000 && found < N; k++) {
    const seed = [101 + k, 202 + 3 * k, 303 + 7 * k, 404 + 11 * k];
    const b = BOARDS.freeze2(lv);
    const r = run(format, b, seed);
    const t2 = r.lines.indexOf('|turn|2');
    const firstFreeze = r.lines.slice(0, t2).includes('|-status|p2a: Snorlax|frz');
    const second = r.lines.includes('|-message|Freeze Clause activated.') || r.lines.includes('|-status|p2a: Blissey|frz');
    if (firstFreeze && second) {
      rows.push({ name: 'freeze2', format, p1: b.p1, p2: b.p2, script: b.script, seed, ...r });
      found++;
    }
  }
  if (found < N) throw new Error(`${format}: only ${found} freeze2 seeds in the search window`);
}

const count = (pred) => rows.filter(pred).length;
const cov = {
  freezeClauseBlocks: count((r) => r.lines.includes('|-message|Freeze Clause activated.')),
  secondFreezeLands: count((r) => r.name === 'freeze2' && r.lines.includes('|-status|p2a: Blissey|frz')),
  sleepClauseBlocks: count((r) => r.lines.includes('|-message|Sleep Clause Mod activated.')),
  beatUpActivate: count((r) => r.lines.some((l) => l.startsWith('|-activate|') && l.includes('Beat Up'))),
  leakTrue: count((r) => r.name === 'leak' && r.maybeTrapped),
  leakFalse: count((r) => r.name === 'leak' && !r.maybeTrapped),
};
for (const [k, v] of Object.entries(cov)) {
  if (v === 0) { console.error(`REFUSING: coverage ${k} = 0`); process.exit(1); }
}
const header = [
  '# FORMAT RULES GOLDEN (gen3_format_rules_v1)',
  `# generated by harness/gen_format_rules_golden.js  rows/scenario/format=${N}  formats=${FORMATS.join(',')}`,
  `# rows=${rows.length}  COVERAGE: ${JSON.stringify(cov)}`,
  '# Each row: the board, the script ([p1, p2] choices per decision), the sim SEED_BEFORE / SEED_AFTER, the',
  '# FRAMING window and the post-|turn|1 lines (omniscient, |t:| dropped), and p1\'s request maybeTrapped.',
  '',
].join('\n');
fs.writeFileSync(OUT, header + rows.map((r) => JSON.stringify(r)).join('\n') + '\n');
console.log(`wrote ${OUT}: ${rows.length} rows; ${JSON.stringify(cov)}`);
