// probe_batch4b_beatup.js — ground-truth BEAT UP (id `beatup`) bit-for-bit vs the
// OMNISCIENT in-process BattleStream (no server). Beat Up is the MULTI-STRIKE move:
//   - ONE hit PER healthy (non-statused, non-fainted) team member of the USER's side.
//   - each strike a fixed base-Atk-derived damage (a basePowerCallback).
//   - a whole new multi-strike draw model — each strike its own crit/damage draw?
//
// The mod chain is the ONLY oracle. Probe the exact:
//   1. the multi-strike LOOP: how many strikes (does the user itself count? does a
//      fainted/statused teammate skip?), and the EXACT per-strike BP formula.
//   2. the per-strike DRAW MODEL: accuracy? crit? damage roll? in what order? typeless/Dark?
//   3. edge cases: sub, protect, ghost/immune, KO mid-sequence, teammate-status gate.
//
// Run:  node src/rust_sim/harness/probe_batch4b_beatup.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

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

function dumpResolved() {
  const d = Dex.forFormat(FORMAT);
  const m = d.moves.get('beatup');
  console.log('=== resolved gen3 beatup ===');
  console.log(`  cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} target=${m.target} ` +
    `priority=${m.priority} flags=${JSON.stringify(m.flags)} multihit=${JSON.stringify(m.multihit)}`);
  console.log(`  critRatio=${m.critRatio} willCrit=${m.willCrit} ignoreImmunity=${JSON.stringify(m.ignoreImmunity)}`);
  if (m.basePowerCallback) console.log(`  basePowerCallback src: ${m.basePowerCallback.toString().replace(/\s+/g, ' ')}`);
  if (m.onModifyMove) console.log(`  onModifyMove src: ${m.onModifyMove.toString().replace(/\s+/g, ' ')}`);
  if (m.onModifyHit) console.log(`  onModifyHit src: ${m.onModifyHit.toString().replace(/\s+/g, ' ')}`);
  if (m.onHit) console.log(`  onHit src: ${m.onHit.toString().replace(/\s+/g, ' ')}`);
  if (m.beforeTurnCallback) console.log(`  beforeTurnCallback src: ${m.beforeTurnCallback.toString().replace(/\s+/g, ' ')}`);
  // Compare across gens to see what gen3 inherits.
  for (const g of ['gen3', 'gen4', 'gen5']) {
    const dm = Dex.mod(g).moves.get('beatup');
    console.log(`  [${g}] bp=${dm.basePower} bpCb=${!!dm.basePowerCallback} multihit=${JSON.stringify(dm.multihit)} onModifyMove=${!!dm.onModifyMove}`);
  }
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 4; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of ((inject && inject.acts) || [])) {
    const side = battle.sides[inj.side];
    const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
    if (inj.faint) { m.hp = 0; m.fainted = true; }
    if (inj.item !== undefined) m.item = inj.item;
  }

  // Print USER-side party base-Atk stats so we can decode the per-strike BP formula.
  const p1 = battle.sides[0];
  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  p1.pokemon.forEach((mn, k) => {
    console.log(`   p1.party[${k}] ${mn.species.name} baseAtk=${mn.species.baseStats.atk} storedAtk=${mn.storedStats.atk} status=${mn.status || '-'} hp=${mn.hp}/${mn.maxhp} fnt=${mn.fainted}`);
  });

  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  let i = 0, safety = 0;
  while (!battle.ended && safety < 8) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}`);
    console.log(`        p2=${fmt(a1)}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||-damage|-heal|-boost|-unboost|-fail|-immune|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end|switch|drag|faint|-start/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // 1) BASELINE — full healthy 6-mon user side. Count strikes + decode the per-strike
  //    BP formula (5 + floor(teammate.baseStats.atk/10)?) + the per-strike draw model.
  await run('BEATUP baseline: 6 healthy user-side mons (count strikes + per-strike BP + draws)',
    [mon('Slaking', ['beatup'], { ability: 'No Ability' }),  // huge base atk
     mon('Machamp', ['bulkup']),
     mon('Alakazam', ['calmmind']),   // low base atk
     mon('Snorlax', ['bodyslam']),
     mon('Blissey', ['softboiled']),  // very low base atk
     mon('Gengar', ['shadowball'])],
    [mon('Skarmory', ['spikes'], { evs: { hp: 252 } }),
     mon('Forretress', ['spikes'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 2) A STATUSED teammate — does a burned/paralyzed teammate SKIP its strike?
  await run('BEATUP with a STATUSED teammate (does it skip that strike?)',
    [mon('Slaking', ['beatup']),
     mon('Machamp', ['bulkup']),
     mon('Snorlax', ['bodyslam']),
     mon('Blissey', ['softboiled']),
     mon('Gengar', ['shadowball']),
     mon('Alakazam', ['calmmind'])],
    [mon('Skarmory', ['spikes'], { evs: { hp: 252 } }),
     mon('Forretress', ['spikes'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 1, status: 'brn' }, { side: 0, slot: 3, status: 'par' }] });

  // 3) A FAINTED teammate — does it skip? (inject a fainted bench mon)
  await run('BEATUP with a FAINTED teammate (skip?)',
    [mon('Slaking', ['beatup']),
     mon('Machamp', ['bulkup']),
     mon('Snorlax', ['bodyslam']),
     mon('Blissey', ['softboiled'])],
    [mon('Skarmory', ['spikes'], { evs: { hp: 252 } }),
     mon('Forretress', ['spikes'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 2, faint: true }] });

  // 4) GHOST target — Beat Up is DARK; does Dark hit Ghost per strike? type effectiveness?
  await run('BEATUP into a GHOST (Dark→Ghost per strike?)',
    [mon('Slaking', ['beatup']),
     mon('Machamp', ['bulkup']),
     mon('Snorlax', ['bodyslam'])],
    [mon('Gengar', ['shadowball'], { ability: 'Levitate', evs: { hp: 252 } }),
     mon('Misdreavus', ['shadowball'], { ability: 'Levitate', evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 5) SUBSTITUTE — a strike into a sub. Does the sub absorb ONE strike then later strikes hit the mon?
  await run('BEATUP into a SUBSTITUTE (sub break + later strikes hit the mon?)',
    [mon('Slaking', ['beatup']),
     mon('Machamp', ['bulkup']),
     mon('Snorlax', ['bodyslam']),
     mon('Blissey', ['softboiled'])],
    [mon('Snorlax', ['substitute', 'bodyslam'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2', stop: true }]);

  // 6) PROTECT — the whole multi-strike blocked by one Protect?
  await run('BEATUP into a PROTECT (whole multi-strike blocked?)',
    [mon('Slaking', ['beatup']),
     mon('Machamp', ['bulkup']),
     mon('Snorlax', ['bodyslam'])],
    [mon('Snorlax', ['protect', 'bodyslam'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 7) KO MID-SEQUENCE — low-HP target: does the sequence stop at the KO, or keep striking?
  await run('BEATUP KOs the target mid-sequence (later strikes fire?)',
    [mon('Slaking', ['beatup']),
     mon('Machamp', ['bulkup']),
     mon('Snorlax', ['bodyslam']),
     mon('Blissey', ['softboiled'])],
    [mon('Gengar', ['shadowball'], { ability: 'Levitate' }),  // frail
     mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, hp: 30 }] });

  // 8) The USER ITSELF — confirm the ACTIVE user (Slaking) DOES contribute a strike (its own base Atk).
  //    Single-mon user side: exactly ONE strike, from the active mon.
  await run('BEATUP single-mon user side (does the user itself strike?)',
    [mon('Slaking', ['beatup'])],
    [mon('Snorlax', ['bodyslam'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 9) The ACTIVE user STATUSED — does the active's OWN strike skip when the active is statused?
  await run('BEATUP active-user STATUSED (does the active own-strike skip?)',
    [mon('Slaking', ['beatup']),
     mon('Machamp', ['bulkup']),
     mon('Snorlax', ['bodyslam'])],
    [mon('Skarmory', ['spikes'], { evs: { hp: 252 } }),
     mon('Forretress', ['spikes'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 0, slot: 0, status: 'brn' }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
