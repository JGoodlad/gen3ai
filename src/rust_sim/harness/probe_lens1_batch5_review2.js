'use strict';
// LENS1 follow-up: R1b counter into the mid-turn REPLACEMENT; R2b multihit sub-break arming.
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: opts.gender || 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
function drawLabel() {
  const st = new Error().stack.split('\n'); const f = [];
  for (let i = 3; i < st.length && f.length < 3; i++) { const m = st[i].match(/at ([\w.<>]+) /); if (m) f.push(m[1]); }
  return f.join('<');
}
async function run(label, p1team, p2team, plan, inject, quiet) {
  const stream = new BattleStream(); const streams = getPlayerStreams(stream); const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = (inject && inject.seed) || [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const applyActs = (acts) => { for (const inj of acts || []) {
    const side = battle.sides[inj.side]; const m = inj.slot === undefined ? side.active[0] : side.pokemon[inj.slot];
    if (!m) continue; if (inj.hp !== undefined) m.hp = inj.hp; if (inj.item !== undefined) m.item = inj.item;
  } };
  applyActs(inject && inject.acts);
  if (!quiet) console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let draws = [];
  const rng = battle.prng.rng; const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };
  const out = { turns: [], log };
  let i = 0, safety = 0;
  while (!battle.ended && safety < 16) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[Math.min(i, plan.length - 1)]; i++;
    applyActs(entry.pre);
    draws = []; const l0 = log.length; const before = battle.prng.getSeed();
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''} vols=[${Object.keys(m.volatiles).join(',')}]` : '-';
    const lines = log.slice(l0).filter((l) => /\|move\||\|turn\||-damage|-fail|-immune|-miss|-crit|cant|-activate|-hitcount|-end\b|-start|switch|drag|faint/.test(l));
    out.turns.push({ rs, draws: draws.slice(), lines, p1: fmt(a0), p2: fmt(a1), before, after });
    if (!quiet) {
      console.log(`  [${rs}] ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} draws=${draws.length}  seed ${before}->${after}`);
      console.log(`        p1=${fmt(a0)}  p2=${fmt(a1)}`);
      draws.forEach((dl, k) => console.log(`        DRAW[${k}] ${dl}`));
      for (const l of lines) console.log(`        LINE ${l}`);
    }
    if (entry.stop) break;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return out;
}
async function main() {
  // R1b: after the recoil self-KO, p2 replaces mid-turn; does the queued Counter execute
  // into the REPLACEMENT with the recorded 2x?
  await run('R1b counter tail into the replacement',
    [mon('Snorlax', ['counter', 'splash'], { evs: { hp: 252 } })],
    [mon('Skarmory', ['doubleedge', 'splash'], { evs: { hp: 252 } }), mon('Forretress', ['splash'], { evs: { hp: 252 } })],
    [{ p1: 'move 1', p2: 'move 1' },
     { p2: 'switch 2', stop: true }],
    { acts: [{ side: 1, hp: 5 }] });

  // R2b: multihit sub-break mid-sequence. Scan seeds for bonemerang HIT that breaks the sub
  // on strike 1 and lands strike 2 on the mon; then does counter fire with 2x strike 2?
  for (let s = 1; s <= 40; s++) {
    const r = await run(`R2b seed ${s}`,
      [mon('Alakazam', ['counter', 'substitute', 'splash'], { evs: { hp: 252 } })],
      [mon('Marowak', ['bonemerang', 'splash'], { evs: { atk: 252 } })],
      [{ p1: 'move 2', p2: 'move 2' },
       { p1: 'move 1', p2: 'move 1', stop: true }],
      { seed: [s, s, s, s] }, true);
    const t = r.turns[1];
    if (!t) continue;
    const subEnd = t.lines.some((l) => l.includes('-end') && l.includes('Substitute'));
    const hitcount = t.lines.find((l) => l.includes('-hitcount'));
    if (subEnd && hitcount) {
      console.log(`\n=== R2b sub broke mid-multihit at seed [${s},${s},${s},${s}] ===`);
      console.log(`  draws=${t.draws.length} seed ${t.before} -> ${t.after}`);
      t.draws.forEach((dl, k) => console.log(`  DRAW[${k}] ${dl}`));
      for (const l of t.lines) console.log(`  LINE ${l}`);
      console.log(`  post: p1=${t.p1}  p2=${t.p2}`);
      break;
    }
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
