// probe_maybetrapped_probe_accepted.js — is a `maybeTrapped` switch probe always REJECTED?
//
// `gen_sim_bridge_diff.js` issues a REJECTED-switch probe whenever a request carries
// `trapped` OR `maybeTrapped`, then drains with `requireRequest:false` — i.e. it assumes
// SOMETHING (an `|error|`, plus a re-request for a hidden trap) always comes back.
//
// But the sim sets `maybeTrapped` in the request for TWO different situations
// (`sim/pokemon.ts::getRequestData` — `data.maybeTrapped` only when `trapped !== true`):
//   (a) a HIDDEN trap — `tryTrap(true)` set `trapped = 'hidden'` (truthy, not `true`).
//       `Side.chooseSwitch` REJECTS it (`sim/side.ts:968`) → `|error|` + a re-request.
//   (b) a SPECULATIVE maybe — `battle.ts:1730` runs `FoeMaybeTrapPokemon` for every ability
//       the foe's SPECIES could have (cancel-leak protection). The mon is NOT trapped, so
//       `chooseSwitch` ACCEPTS the switch (`side.ts:981` only sets `cantUndo`) and the sim
//       emits NOTHING until the boundary completes.
// In case (b) a drain that waits for output blocks for its full cap, and the harness then
// sends a SECOND choice for a side that has already committed one.
//
// This probe prints, for a constructed matchup, the request flags and what the sim does
// with the probe — so the two cases are distinguished by evidence, not by source reading.
//
// USAGE: node src/rust_sim/harness/probe_maybetrapped_probe_accepted.js [--format gen3ou]
'use strict';

const path = require('path');
const ROOT = path.resolve(__dirname, '../../..');
const { BattleStream, getPlayerStreams } = require(path.join(ROOT, 'deps/pokemon-showdown/dist/sim/battle-stream'));
const { Teams } = require(path.join(ROOT, 'deps/pokemon-showdown/dist/sim'));

const fmtIdx = process.argv.indexOf('--format');
const FORMAT = fmtIdx === -1 ? 'gen3customgame' : process.argv[fmtIdx + 1];

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const set = (species, moves, ability) => ({
  name: species, species, item: 'Leftovers', ability, moves,
  evs: EV0, ivs: IV31, nature: 'Serious', level: 100, gender: 'M',
});

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// p2's Dugtrio holds SAND VEIL — the species CAN have Arena Trap, so the sim flags p1
// `maybeTrapped` defensively, but p1 is NOT actually trapped.
async function run(label, p2Ability) {
  const p1 = Teams.pack([set('Snorlax', ['bodyslam', 'rest'], 'Immunity'), set('Blissey', ['softboiled', 'seismictoss'], 'Natural Cure')]);
  const p2 = Teams.pack([set('Dugtrio', ['earthquake', 'rockslide'], p2Ability)]);

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const seen = { p1: [], p2: [] };
  for (const side of ['p1', 'p2']) {
    (async () => { for await (const c of streams[side]) seen[side].push(c); })();
  }
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  await sleep(400);

  // The p1 request at the first move boundary.
  let req = null;
  for (const c of seen.p1) {
    for (const l of c.split('\n')) if (l.startsWith('|request|') && l.length > 14) {
      try { const j = JSON.parse(l.slice(9)); if (j.active) req = j; } catch (e) {}
    }
  }
  const a = req && req.active && req.active[0];
  const flags = a ? { trapped: !!a.trapped, maybeTrapped: !!a.maybeTrapped } : null;
  console.log(`\n[${label}] format=${FORMAT} p2 ability=${p2Ability}`);
  console.log(`  p1 request flags: ${JSON.stringify(flags)}`);
  if (!flags || (!flags.trapped && !flags.maybeTrapped)) {
    console.log('  → the differ would NOT probe this side (no trapped/maybeTrapped). n/a');
    return null;
  }

  // The differ's probe: a switch, then "drain until something arrives".
  const mark = seen.p1.length;
  streams.p1.write('switch 2');
  await sleep(600);
  const after = seen.p1.slice(mark);
  const errs = [], reqs = [];
  for (const c of after) for (const l of c.split('\n')) {
    if (l.startsWith('|error|')) errs.push(l);
    if (l.startsWith('|request|')) reqs.push('|request|');
  }
  const accepted = after.length === 0;
  console.log(`  probe "switch 2" → chunks=${after.length} errors=${JSON.stringify(errs)} requests=${reqs.length}`);
  console.log(accepted
    ? '  ⇒ ACCEPTED SILENTLY: a drain waiting for output blocks for its FULL cap, and the'
      + '\n    switch is now the side\'s COMMITTED choice — a second choice for this side is bogus.'
    : '  ⇒ REJECTED (the differ\'s assumption holds here).');
  return accepted;
}

(async () => {
  const speculative = await run('SPECULATIVE maybe (Sand Veil Dugtrio)', 'Sand Veil');
  const hidden = await run('REAL trap (Arena Trap Dugtrio)', 'Arena Trap');
  console.log('\n[probe] verdict: speculative-probe-accepted='
    + `${speculative} real-trap-probe-accepted=${hidden}`);
  process.exit(0);
})();
