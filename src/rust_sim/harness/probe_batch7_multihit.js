// probe_batch7_multihit.js — ground-truth the GENERIC MULTIHIT family (Twineedle /
// Double Kick / Bonemerang FIXED-2, Pin Missile / Bullet Seed / … VARIABLE [2,5],
// Triple Kick = 3 + multiaccuracy) bit-for-bit vs the OMNISCIENT in-process
// BattleStream (no server). Unlike Beat Up (a stat-swap basePowerCallback), these
// run the NORMAL damage path N times with the move's real type/BP/category.
//
// The mod chain is the ONLY oracle. Probe the EXACT:
//   1. the COUNT: fixed (no draw) vs variable ([2,5]) — the exact `sample(array)`
//      distribution + WHICH draw (before the per-hit loop). Sweep many seeds.
//   2. the PER-HIT DRAW MODEL: crit? damage roll? eachEvent('Update') on a tie?
//      in what ORDER, per hit?
//   3. the SECONDARY placement (Twineedle 20% psn): per-hit or ONCE after all hits?
//   4. KO-mid-sequence: does the loop STOP at the target's faint?
//   5. SUBSTITUTE: sub absorbs, later hits hit the mon?
//   6. Triple Kick (multiaccuracy) — its per-hit accuracy re-roll (why we fail-loud).
//   7. VOLT TACKLE (recoil + para) + PSYCHO BOOST (selfDrops) — confirm engine-ready.
//
// Run:  node src/rust_sim/harness/probe_batch7_multihit.js
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
  console.log('=== resolved gen3 multihit family ===');
  const d = Dex.forFormat(FORMAT);
  for (const id of ['twineedle', 'doublekick', 'bonemerang', 'pinmissile', 'bulletseed',
                    'iciclespear', 'rockblast', 'barrage', 'cometpunch', 'doubleslap',
                    'spikecannon', 'armthrust', 'furyattack', 'furyswipes', 'triplekick',
                    'bonerush', 'voltackle', 'psychoboost']) {
    const m = d.moves.get(id);
    if (!m || !m.exists) { console.log(`  ${id}: (missing)`); continue; }
    console.log(`  ${id}: cat=${m.category} bp=${m.basePower} acc=${m.accuracy} type=${m.type} ` +
      `multihit=${JSON.stringify(m.multihit)} multiaccuracy=${m.multiaccuracy} ` +
      `sec=${JSON.stringify(m.secondary || m.secondaries)} self=${JSON.stringify(m.self)} recoil=${JSON.stringify(m.recoil)}`);
  }
  // The gen3 multihit COUNT distribution — read the resolved sampleRandom / the loop source.
  const bs = Dex.mod('gen3').data.Scripts;
  console.log('  gen3 Scripts.hitStepMoveHitLoop present:', !!(bs && bs.hitStepMoveHitLoop));
}

function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 5; i++) {
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
    if (inj.item !== undefined) m.item = inj.item;
  }

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

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
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${draws.length}  seed ${before}->${after}`);
    console.log(`        p1=${fmt(a0)}  p2=${fmt(a1)}`);
    draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||-damage|-heal|-boost|-unboost|-fail|-immune|-crit|-supereffective|-resisted|cant|-activate|-hitcount|-end|-status|switch|drag|faint|-start/.test(l));
    for (const l of newLines) console.log(`        LINE ${l}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

// Sweep the VARIABLE-count distribution: run N seeds, count hits per use.
async function sweepCount(label, moveId, seeds) {
  const dist = {};
  for (const s of seeds) {
    const stream = new BattleStream();
    const streams = getPlayerStreams(stream);
    let hitcount = null;
    (async () => { for await (const ch of streams.omniscient) {
      for (const l of ch.split('\n')) { const m = l.match(/^\|-hitcount\|[^|]+\|(\d+)/); if (m) hitcount = +m[1]; }
    } })();
    streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(s)}}`);
    streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack([mon('Marowak', [moveId], { evs: { atk: 252 } })]) })}`);
    streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack([mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } })]) })}`);
    for (let k = 0; k < 12; k++) await tick();
    streams.omniscient.write('>p1 move 1'); streams.omniscient.write('>p2 move 1');
    for (let k = 0; k < 20; k++) await tick();
    if (hitcount != null) dist[hitcount] = (dist[hitcount] || 0) + 1;
    try { streams.omniscient.destroy(); } catch (e) {}
  }
  console.log(`\n=== COUNT DISTRIBUTION ${label} (${moveId}, ${seeds.length} seeds) ===`);
  console.log('  ', JSON.stringify(dist));
}

async function main() {
  dumpResolved();

  // 1) TWINEEDLE (fixed 2, 20% psn secondary) into a bulky NON-poison foe — the CANONICAL
  //    draw model: no count draw, per-hit crit+damage, then WHERE the 20% poison fires.
  await run('TWINEEDLE (fixed 2 + 20% psn) into bulky Snorlax — per-hit draws + secondary placement',
    [mon('Beedrill', ['twineedle'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 2) DOUBLE KICK (fixed 2, NO secondary) — the plainest fixed-2 draw model.
  await run('DOUBLE KICK (fixed 2, no secondary) into Snorlax',
    [mon('Hitmonlee', ['doublekick'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 3) PIN MISSILE (variable [2,5]) — the COUNT draw (the sample) BEFORE the loop.
  await run('PIN MISSILE (variable [2,5]) into Snorlax — the count `sample` draw',
    [mon('Beedrill', ['pinmissile'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 4) KO-mid-sequence: a fixed-2 into a mon that dies on hit 1 — does hit 2 fire? Quick Claw?
  await run('DOUBLE KICK KO on hit 1 (does the loop STOP? Quick Claw skipped?)',
    [mon('Hitmonlee', ['doublekick'], { evs: { atk: 252 } })],
    [mon('Diglett', ['splash'])],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 5) SUBSTITUTE: fixed-2 into a sub — sub break then later hit hits the mon?
  await run('PIN MISSILE into a SUBSTITUTE (absorb + break + later hits hit the mon?)',
    [mon('Beedrill', ['pinmissile'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['substitute', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 2', p2: 'move 1' }, { p1: 'move 1', p2: 'move 2', stop: true }]);

  // 6) TWINEEDLE speed-TIE (Beedrill mirror) — does each hit draw the eachEvent('Update') shuffle?
  await run('TWINEEDLE speed-TIE mirror (per-strike eachEvent Update on a tie?)',
    [mon('Beedrill', ['twineedle'], { evs: { atk: 252 } })],
    [mon('Beedrill', ['splash'], { evs: { atk: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 7) TRIPLE KICK (3 + multiaccuracy) — the per-hit accuracy re-roll (why we FAIL-LOUD).
  await run('TRIPLE KICK (3 + multiaccuracy) — per-hit accuracy re-roll draw model',
    [mon('Hitmontop', ['triplekick'], { evs: { atk: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 8) VOLT TACKLE (recoil + para) — confirm the recoil + para secondary draw model (engine-ready).
  await run('VOLT TACKLE (recoil 1/3 + 10% para) into Snorlax',
    [mon('Pikachu', ['voltackle'], { evs: { atk: 252 }, item: '' })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252, def: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // 9) PSYCHO BOOST (selfDrops spa -2) — confirm the selfDrops random(100) (engine-ready).
  await run('PSYCHO BOOST (self spa -2 selfDrops random(100)) into Snorlax',
    [mon('Deoxys', ['psychoboost'], { evs: { spa: 252 } })],
    [mon('Snorlax', ['splash'], { evs: { hp: 252, spd: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }]);

  // COUNT distributions over many seeds — nail the exact [2,5] distribution.
  const seeds = [];
  for (let a = 0; a < 12; a++) for (let b = 0; b < 12; b++) seeds.push([a, b, a * 7 + 1, b * 13 + 3]);
  await sweepCount('variable [2,5]', 'pinmissile', seeds);
  await sweepCount('fixed 2 (control)', 'doublekick', seeds.slice(0, 40));
  await sweepCount('triple kick (3, multiaccuracy)', 'triplekick', seeds);
}

main().catch((e) => { console.error(e); process.exit(1); });
