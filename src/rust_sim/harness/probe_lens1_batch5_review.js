// probe_lens1_batch5_review.js — LENS 1 (draw-order) INDEPENDENT review probes for Batch 5.
// Edges deliberately NOT in the build's probes:
//   R1: foe recoil-self-KOs AFTER arming Counter — what does the -5 Counter do into the
//       fainted-occupied slot (draws? lines?).
//   R2: multihit BREAKS the counter user's sub mid-sequence — do post-break strikes hit the
//       mon and arm Counter (2x last strike)?
//   R3: the `damage || 1` clamp — Focus Band proc leaves the counter user at 1 HP taking a
//       0-damage recorded hit (dealt = hp-1 = 0): does the armed Counter deal 1 (sim `|| 1`)?
//   R4: foe physical move MISSES — counter fail draw count (same as no-damage?).
// Run:  node src/rust_sim/harness/probe_lens1_batch5_review.js
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
function drawLabel() {
  const st = new Error().stack.split('\n');
  const frames = [];
  for (let i = 3; i < st.length && frames.length < 4; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

async function run(label, p1team, p2team, plan, inject, quiet) {
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
  const applyActs = (acts) => {
    for (const inj of acts || []) {
      const side = battle.sides[inj.side];
      const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
      if (!m) continue;
      if (inj.status) m.setStatus(inj.status, m, null, true);
      if (inj.hp !== undefined) m.hp = inj.hp;
      if (inj.item !== undefined) m.item = inj.item;
      if (inj.boosts) Object.assign(m.boosts, inj.boosts);
    }
  };
  applyActs(inject && inject.acts);
  if (!quiet) console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };
  const out = { turns: [], log };
  let i = 0, safety = 0;
  while (!battle.ended && safety < 16) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    draws = [];
    const logLen0 = log.length;
    const before = battle.prng.getSeed();
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';
    const newLines = log.slice(logLen0).filter((l) =>
      /\|move\||\|turn\||-damage|-heal|-fail|-immune|-miss|-crit|cant|-activate|-hitcount|-end\b|-start|switch|drag|faint/.test(l));
    out.turns.push({ rs, draws: draws.slice(), before, after, lines: newLines, p1: fmt(a0), p2: fmt(a1) });
    if (!quiet) {
      console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
      console.log(`        p1=${fmt(a0)}  p2=${fmt(a1)}`);
      draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
      for (const l of newLines) console.log(`        LINE ${l}`);
    }
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}

async function main() {
  // R1: foe Double-Edge recoil self-KO after arming counter. Skarmory at 5 hp uses
  // Double-Edge into Snorlax (arms counter 2x), recoil (1/3 of dealt, gen3) kills the
  // Skarmory. Counter (-5) then executes into the fainted-occupied slot.
  await run('R1 counter into a recoil-self-KOd foe slot',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['doubleedge', 'splash'], { evs: { hp: 252 } }), mon('Forretress', ['splash'])],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, hp: 5 }] });

  // R2: multihit breaks the counter user's SUB mid-sequence. Alakazam (frail, sub =
  // floor(251/4)=62 hp) subs turn 1; Marowak Bonemerang (2 hits, ~130 each with Thick Club?
  // no item -> ~90) breaks the sub on hit 1, hit 2 lands on the mon -> recorded?
  await run('R2 multihit sub-break mid-sequence: post-break strike arms counter?',
    [mon('Alakazam', ['counter', 'substitute', 'splash'], { evs: { hp: 252 } })],
    [mon('Marowak', ['bonemerang', 'splash'], { evs: { atk: 252 } })],
    [{ p1: 'move 2', p2: 'move 2' },               // sub up
     { p1: 'move 1', p2: 'move 1', stop: true }]); // bonemerang: h1 breaks sub, h2 -> mon?

  // R3: the `damage || 1` clamp. Snorlax at 1 hp + Focus Band selected counter; Skarmory
  // Drill Peck is lethal -> FB proc (random(10)<1) leaves it at 1: dealt = 0 -> recorded
  // damage = 2*0 = 0, slot SET -> onTry passes -> damageCallback `0 || 1` = 1?
  // Seed-scan for the FB proc.
  for (let s = 1; s <= 80; s++) {
    const r = await run(`R3 seed ${s}`,
      [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 }, item: 'focusband' })],
      [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
      [{ p1: 'move 1', p2: 'move 1', stop: true }],
      { seed: [s, s, s, s], acts: [{ side: 0, hp: 1 }] }, true);
    const lines = r.turns[0] ? r.turns[0].lines : [];
    if (lines.some((l) => l.includes('Focus Band'))) {
      console.log(`\n=== R3 FB proc at seed [${s},${s},${s},${s}] ===`);
      console.log(`  draws=${r.turns[0].draws.length} seed ${r.turns[0].before} -> ${r.turns[0].after}`);
      r.turns[0].draws.forEach((dl, k) => console.log(`  DRAW[${k}] ${dl}`));
      for (const l of lines) console.log(`  LINE ${l}`);
      console.log(`  post: p1=${r.turns[0].p1}  p2=${r.turns[0].p2}`);
      break;
    }
  }

  // R4: the foe's physical move MISSES (acc-fail) — counter fail draw count. Skarmory
  // Drill Peck at -6 accuracy (inject boosts) vs Snorlax counter.
  await run('R4 foe physical MISSES: counter fail draws',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['drillpeck', 'splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1', stop: true }],
    { acts: [{ side: 1, boosts: { accuracy: -6 } }] });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
