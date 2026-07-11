// probe_protect_rng.js — INSTRUMENTED Gen-3 PROTECT/DETECT PRNG draw probe.
//
// Purpose (the CRUX investigation, NOT a golden generator): nail the EXACT
// PRNG-draw model of Protect/Detect's gen3 consecutive-use stall mechanic + the
// blocked-foe-move draw, by monkey-patching the sim PRNG to record EVERY draw with
// its sim/* call site, over CONSTRUCTED scenarios:
//   (A) a SINGLE Protect vs a foe attack — what does the FIRST protect draw, and
//       what does the BLOCKED foe move draw (accuracy? nothing?).
//   (B) CONSECUTIVE Protects (turn1/2/3) — the stall onStallMove randomChance
//       denominator sequence + whether the FIRST use draws.
//   (C) a Protect then a NON-protect move then a Protect — the counter reset.
//   (D) a Protect vs a STATUS move (Thunder Wave) — block of a status move.
//
// Run:  node src/rust_sim/harness/probe_protect_rng.js
// (Needs the submodule dist/ + node_modules symlinks; see root CLAUDE.md.)

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
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

function isWrapperFrame(ln) {
  if (ln.includes('/sim/prng.js')) return true;
  if (ln.includes('probe_protect_rng.js')) return true;
  if (/at Battle\.(random|randomChance|sample) /.test(ln)) return true;
  return false;
}
function frameStr(ln) {
  const fn = (ln.match(/at ([\w.<>]+) /) || [])[1] || '?';
  const loc = (ln.match(/\/(sim\/[^\s):]+:\d+):\d+/) || ln.match(/\/(data\/[^\s):]+:\d+):\d+/) || [])[1] || '?';
  return `${fn}@${loc}`;
}
function siteOf(depth = 4) {
  const e = {};
  Error.captureStackTrace(e, siteOf);
  const lines = (e.stack || '').split('\n').slice(1).filter((ln) => !isWrapperFrame(ln));
  const sim = lines.filter((ln) => /\/(sim|data)\//.test(ln)).slice(0, depth);
  return sim.map(frameStr).join(' <- ') || '?';
}

function instrument(prng, sink) {
  const rng = prng.rng;
  let nextCount = 0;
  const origNext = rng.next.bind(rng);
  rng.next = function () { nextCount++; return origNext(); };
  let reentrant = false;
  const wrap = (name) => {
    const orig = prng[name].bind(prng);
    prng[name] = function (...args) {
      if (reentrant && name === 'random') return orig(...args);
      const wasRe = reentrant;
      reentrant = true;
      const site = siteOf();
      const before = nextCount;
      const result = orig(...args);
      const consumed = nextCount - before;
      reentrant = wasRe;
      sink.push({ method: name, args, result, nexts: consumed, site });
      return result;
    };
  };
  for (const m of ['randomChance', 'sample', 'shuffle']) wrap(m);
  wrap('random');
}

async function runScenario(name, p1team, p2team, seed, choices) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const draws = [];
  const markers = [];

  // read omniscient output, queueing turn markers interleaved with draws
  (async () => {
    for await (const chunk of streams.omniscient) {
      for (const line of chunk.split('\n')) {
        if (line.startsWith('|turn|')) markers.push({ at: draws.length, turn: line.slice(6).trim() });
        if (line.startsWith('|-singleturn|') && line.includes('Protect')) markers.push({ at: draws.length, note: 'protect-up' });
        if (line.startsWith('|-activate|') && line.includes('Protect')) markers.push({ at: draws.length, note: 'PROTECT-BLOCK' });
        if (line.startsWith('|move|')) markers.push({ at: draws.length, note: 'move:' + line.split('|')[3] });
      }
    }
  })();

  const spec = { formatid: FORMAT, seed };
  void streams.omniscient.write(`>start ${JSON.stringify(spec)}\n`);
  void streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}\n`);
  void streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}\n`);
  await tick();

  // instrument AFTER start so construction draws (gender) are excluded
  instrument(stream.battle.prng, draws);

  for (const [c1, c2] of choices) {
    void streams.omniscient.write(`>p1 ${c1}\n`);
    void streams.omniscient.write(`>p2 ${c2}\n`);
    await tick();
    if (stream.battle.ended) break;
  }
  await tick();

  // print
  console.log(`\n===== ${name}  seed=${JSON.stringify(seed)} =====`);
  let mi = 0;
  for (let i = 0; i <= draws.length; i++) {
    while (mi < markers.length && markers[mi].at === i) {
      const m = markers[mi];
      if (m.turn) console.log(`  --- TURN ${m.turn} ---`);
      else console.log(`  [${m.note}]`);
      mi++;
    }
    if (i < draws.length) {
      const d = draws[i];
      console.log(`    ${String(i).padStart(2)}  ${d.method}(${JSON.stringify(d.args)}) = ${JSON.stringify(d.result)}  nexts=${d.nexts}  @ ${d.site}`);
    }
  }
  console.log(`  TOTAL DRAWS = ${draws.length}`);
  return draws;
}

(async () => {
  // Slow Snorlax with Protect; fast attacker with Earthquake (always hits) so the
  // foe move resolves AFTER protect is up. Snorlax max HP so it never faints.
  const protector = (moves) => [mon('Snorlax', moves, { evs: { hp: 252 } })];
  const attacker = [mon('Dugtrio', ['Earthquake', 'Tackle', 'Substitute'], { evs: { spe: 252 } })];
  const twaver = [mon('Dugtrio', ['Thunder Wave', 'Earthquake'], { evs: { spe: 252 } })];

  // (A) single Protect vs Earthquake
  await runScenario('A: single Protect vs EQ',
    protector(['Protect', 'Tackle']), attacker, [1, 2, 3, 4],
    [['move 1', 'move 1']]);

  // (B) consecutive Protects — turn1,2,3,4. Sweep a few seeds to see success/fail.
  for (const s of [[1, 2, 3, 4], [9, 9, 9, 9], [42, 7, 13, 99], [5, 5, 5, 5]]) {
    await runScenario('B: consecutive Protect x4',
      protector(['Protect', 'Tackle']), attacker, s,
      [['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1'], ['move 1', 'move 1']]);
  }

  // (C) Protect, then Tackle (non-protect → reset), then Protect again
  await runScenario('C: Protect, Tackle (reset), Protect',
    protector(['Protect', 'Tackle']), attacker, [1, 2, 3, 4],
    [['move 1', 'move 1'], ['move 2', 'move 1'], ['move 1', 'move 1']]);

  // (D) Protect vs a STATUS move (Thunder Wave)
  await runScenario('D: Protect vs Thunder Wave',
    protector(['Protect', 'Tackle']), twaver, [1, 2, 3, 4],
    [['move 1', 'move 1']]);
})();
