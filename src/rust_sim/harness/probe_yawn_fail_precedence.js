// probe_yawn_fail_precedence.js — settle the gen3 YAWN fail-branch PRECEDENCE against the
// OMNISCIENT in-process BattleStream (no server). The sim is the ONLY oracle.
//
// WHY: the fuzz_r25 randbats byte-fuzz repro `rms9nh02e_ab_707_4` (line 431) shows a target
// carrying BOTH a SUBSTITUTE and a sleep-immune ability (Vital Spirit), and the two engines
// pick DIFFERENT fail branches:
//     sim   |move|p2a: Swalot|Yawn||[still]        + |-fail|p2a: Swalot
//     port  |move|p2a: Swalot|Yawn|p1a: Primeape   + |-immune|p1a: Primeape|[from] ability: …
// i.e. the sim reports the SUBSTITUTE block, the port reports the ABILITY immunity.
//
// The port's documented order (CLAUDE.md "## YAWN") is
//     Protect > already-statused > sleep-immune(-immune) > Substitute([still]+-fail) > ADD
// A source read supports that order (yawn's own `onTryHit` — `target.status ||
// !target.runStatusImmunity('slp')` — runs in `runEvent('TryHit')`, while the Substitute
// block is the LATER `onTryPrimaryHit`). The golden says otherwise, so the source read is a
// hypothesis and this probe decides.
//
// MATRIX (each cell prints the emitted lines + a one-word BRANCH):
//   1 clean                  → expect ADD (`-start … move: Yawn`)
//   2 substitute only        → expect SUB   ([still] + -fail on the USER)
//   3 sleep-immune only      → expect IMMUNE(-immune on the TARGET, normal announce)
//   4 substitute + immune    → THE CASE UNDER TEST
//   5 statused only          → expect STATUSED ([still] + -fail)
//   6 statused + substitute  → precedence between those two
//   7 protect + substitute   → expect PROTECT (-activate Protect)
//   8 already-yawned re-cast → expect YAWNED ([still] + -fail, duration NOT reset — YW1)
//
// Run:  node src/rust_sim/harness/probe_yawn_fail_precedence.js
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

// p1 = the YAWNER (Swalot: Yawn + Splash + Thunder Wave to self-inflict a status on the foe).
// p2 = the TARGET, ability/moves varied per case.
async function run(label, targetSpecies, targetAbility, setup) {
  const p1 = [mon('Swalot', ['yawn', 'splash', 'thunderwave'], { level: 100 }),
              mon('Sudowoodo', ['splash'], { level: 100 })];
  const p2 = [mon(targetSpecies, ['substitute', 'splash', 'protect'], { ability: targetAbility, level: 100 }),
              mon('Snorlax', ['splash'], { level: 100 })];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  for (const step of setup) {
    if (step.p1) streams.omniscient.write(`>p1 ${step.p1}`);
    if (step.p2) streams.omniscient.write(`>p2 ${step.p2}`);
    for (let k = 0; k < 18; k++) await tick();
  }
  const mark = log.length;
  streams.omniscient.write('>p1 move yawn');
  streams.omniscient.write('>p2 move splash');
  for (let k = 0; k < 18; k++) await tick();

  const out = log.slice(mark).filter((l) => /\|(move\|p1a|-fail|-immune|-activate|-start|-status)/.test(l));
  const joined = out.join(' ');
  let branch = 'ADD?';
  if (joined.includes('-activate') && joined.includes('Protect')) branch = 'PROTECT';
  else if (joined.includes('-immune')) branch = 'IMMUNE (ability)';
  else if (joined.includes('[still]')) branch = 'STILL+FAIL (sub / statused / yawned)';
  else if (joined.includes('move: Yawn')) branch = 'ADD (yawn applied)';

  const t = battle.sides[1].active[0];
  console.log(`\n=== ${label} ===`);
  console.log(`    target: ${targetSpecies} (${targetAbility}) status=${t && t.status || '-'} sub=${!!(t && t.volatiles && t.volatiles['substitute'])} yawn=${!!(t && t.volatiles && t.volatiles['yawn'])}`);
  for (const l of out) console.log('   ', l);
  console.log(`    BRANCH: ${branch}`);
}

(async () => {
  // 1 clean
  await run('1 clean (no sub, no immunity)', 'Snorlax', 'Immunity', []);
  // 2 substitute only
  await run('2 SUBSTITUTE only', 'Snorlax', 'Immunity', [{ p1: 'move splash', p2: 'move substitute' }]);
  // 3 sleep-immune only (Vital Spirit)
  await run('3 SLEEP-IMMUNE only (Vital Spirit)', 'Primeape', 'Vital Spirit', []);
  // 4 THE CASE: substitute + sleep-immune
  await run('4 SUBSTITUTE + SLEEP-IMMUNE  <-- the ab_707_4 case', 'Primeape', 'Vital Spirit',
    [{ p1: 'move splash', p2: 'move substitute' }]);
  // 5 statused only (Thunder Wave first)
  await run('5 STATUSED only', 'Snorlax', 'Immunity', [{ p1: 'move thunderwave', p2: 'move splash' }]);
  // 6 statused + substitute
  await run('6 STATUSED + SUBSTITUTE', 'Snorlax', 'Immunity',
    [{ p1: 'move thunderwave', p2: 'move splash' }, { p1: 'move splash', p2: 'move substitute' }]);
  // 7 protect + substitute
  await run('7 PROTECT + SUBSTITUTE', 'Snorlax', 'Immunity',
    [{ p1: 'move splash', p2: 'move substitute' }, { p1: 'move splash', p2: 'move protect' }]);
})();
