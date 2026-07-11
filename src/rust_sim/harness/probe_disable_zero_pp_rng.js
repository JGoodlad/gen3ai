// probe_disable_zero_pp_rng.js — nail the sim's behavior when DISABLE lands on a target whose
// LAST-USED move has 0 PP remaining (the gen4-inherited onStart 0-PP guard, `!moveSlot.pp →
// return false`, which gen3's condition inherits — gen3 only overrides durationCallback + the
// residual orders). The reviewer-probed claim to verify:
//
//   1. The accuracy roll (acc 55) IS drawn.
//   2. `addVolatile` fires the durationCallback → `random(2,6)` IS drawn (BEFORE onStart).
//   3. onStart's 0-PP guard then returns false → the volatile is REMOVED: the target's
//      volatiles stay EMPTY, protocol shows `|move|<user>|Disable||[still]` (attrLastMove
//      empties the target field) + `|-fail|<user>`, NO `-start`, NO residual duration handler.
//
// Scenario (organically reachable — a mon spends its mono-move's last PP, then Struggles):
//   P1 Suicune [disable, calmmind]  (faster)   — calm-minds through the PP clock.
//   P2 Blissey [detect]             (slower)   — Detect (5 PP → 8 with PP Ups) FAILS every
//      turn draw-free (Blissey acts LAST → willAct() false; PP still deducts) → after 8
//      turns the slot is at 0 PP with lastMove = detect, and dec8's request offers ONLY
//      Struggle. Suicune then Disables INTO the 0-PP lastMove while Blissey's Struggle is
//      still pending (willMove(target) TRUE → no onStart duration bump either way).
//
// Ground truth for the regression pin `disable_into_a_zero_pp_lastmove_fails_draws_but_no_volatile`
// in tests/regression_test.rs is COPIED VERBATIM from this probe's output (seed sweep + the
// per-decision post-SEEDs + the dec8 draw bracket + protocol lines + empty volatiles). NOTE the
// pin seeds the port's draw-free `start_with_switchins` with the printed POST-INIT seed (the
// sim's init consumes ONE draw) — the same convention as every TD pin.
//
// Run:  node src/rust_sim/harness/probe_disable_zero_pp_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

const SUICUNE = [mon('Suicune', ['disable', 'calmmind'], { evs: { hp: 252, def: 252 } })];
const BLISSEY = [mon('Blissey', ['detect'], { evs: { hp: 252, def: 252 } })];
// dec0..dec7: calmmind / detect (detect fails draw-free, PP 8 → 0). dec8: DISABLE into the
// 0-PP lastMove / forced Struggle. dec9: calmmind / Struggle again (post-fail stability).
const PLAN = [
  ...Array(8).fill({ p1: 'move 2', p2: 'move 1' }),
  { p1: 'move 1', p2: 'move 1' },
  { p1: 'move 2', p2: 'move 1' },
];

async function run(seed, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(SUICUNE) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(BLISSEY) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const initSeed = battle.prng.getSeed(); // post-init: the sim's init consumed ONE draw

  // Instrument EVERY PRNG advance (prng.random is the single funnel: randomChance /
  // shuffle / sample all delegate to it).
  let draws = [];
  const realRandom = battle.prng.random.bind(battle.prng);
  battle.prng.random = function (from, to) {
    const r = realRandom(from, to);
    draws.push(`random(${from === undefined ? '' : from}${to === undefined ? '' : ',' + to})=${r}`);
    return r;
  };

  const rows = [];
  let i = 0, safety = 0;
  while (!battle.ended && i < PLAN.length && safety < 80) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const logLen0 = log.length;
    draws = [];
    const entry = PLAN[i]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    rows.push({
      dec: i - 1, choice: entry, after: battle.prng.getSeed(), draws: draws.slice(),
      lines: log.slice(logLen0).filter((l) => /\|(move|cant|-start|-end|-damage|-heal|-boost|-fail|-miss|-activate|faint)\|/.test(l)),
      p1: a0 ? `${a0.species.name} ${a0.hp}/${a0.maxhp} volatiles=[${Object.keys(a0.volatiles).join(',')}]` : '-',
      p2: a1 ? `${a1.species.name} ${a1.hp}/${a1.maxhp} volatiles=[${Object.keys(a1.volatiles).join(',')}] detectPP=${a1.moveSlots[0].pp} lastMove=${a1.lastMove ? a1.lastMove.id : null}` : '-',
      p2req: (() => {
        const req = battle.sides[1].activeRequest;
        return req && req.active && req.active[0] && req.active[0].moves
          ? req.active[0].moves.map((mv) => `${mv.id}${mv.disabled ? '(DISABLED)' : ''}`).join(' ')
          : '-';
      })(),
    });
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return { battle, rows, log, initSeed };
}

function printRun(label, seed, r) {
  console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)}  initSeed(post-init, the pin's start seed)=${r.initSeed}`);
  for (const row of r.rows) {
    console.log(`  [dec ${row.dec}] ${JSON.stringify(row.choice)} after=${row.after}  draws=${row.draws.length}: ${row.draws.join(' ')}`);
    for (const l of row.lines) console.log(`        ${l}`);
    console.log(`        p1: ${row.p1}`);
    console.log(`        p2: ${row.p2}  next-request: ${row.p2req}`);
  }
}

async function main() {
  // Seed sweep: dec8's acc-55 Disable must HIT (so the 0-PP guard, not a miss, decides).
  let hitSeed = null, hitRun = null;
  let missSeed = null, missRun = null;
  for (let s = 1; s <= 60 && (!hitSeed || !missSeed); s++) {
    const seed = [s, s + 7, s + 13, s + 21];
    const r = await run(seed);
    const d8 = r.rows[8];
    if (!d8) continue;
    const failed = d8.lines.some((l) => l === '|move|p1a: Suicune|Disable||[still]');
    const missed = d8.lines.some((l) => /\|-miss\|p1a: Suicune/.test(l));
    if (failed && !hitSeed) { hitSeed = seed; hitRun = r; }
    if (missed && !missSeed) { missSeed = seed; missRun = r; }
  }

  if (!hitRun) { console.log('NO seed found where the disable HITS at dec8'); process.exit(1); }
  printRun('DISABLE into a 0-PP lastMove — accuracy HITS, 0-PP guard REJECTS', hitSeed, hitRun);
  const d8 = hitRun.rows[8];
  console.log('\n  --- dec8 verdicts ---');
  console.log(`  draws: ${d8.draws.join(' | ')}`);
  console.log(`  duration draw random(2,6) present: ${d8.draws.some((x) => x.startsWith('random(2,6)'))}`);
  console.log(`  |-start| Disable emitted: ${d8.lines.some((l) => /\|-start\|.*Disable/.test(l))}`);
  console.log(`  |move|p1a: Suicune|Disable||[still] emitted: ${d8.lines.some((l) => l === '|move|p1a: Suicune|Disable||[still]')}`);
  console.log(`  |-fail|p1a: Suicune emitted: ${d8.lines.some((l) => l === '|-fail|p1a: Suicune')}`);
  console.log(`  target volatiles after dec8: ${d8.p2}`);
  console.log(`  per-decision post-seeds (for the pin):`);
  for (const row of hitRun.rows) console.log(`    dec${row.dec}: "${row.after}"`);

  if (missRun) {
    printRun('CONTROL — same scenario, dec8 disable MISSES (accuracy-only, no random(2,6))', missSeed, missRun);
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
