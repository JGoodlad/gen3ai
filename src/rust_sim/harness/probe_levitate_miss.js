// probe_levitate_miss.js — does a MISSABLE Ground move into a Levitate/Flying mon
// show `-immune` on EVERY seed (pre-accuracy runImmunity) or `[miss]` on a miss?
// gen3: type-chart 0x AND Levitate resolve at runImmunity (BEFORE the accuracy roll),
// so they ALWAYS report `-immune` regardless of the accuracy outcome. Contrast the
// onTryHit ability immunities (Flash Fire / Water/Volt Absorb) which are POST-accuracy.
// Magnitude/Bonemerang aren't in gen3 OU; use a lower-accuracy Ground: Bone Rush? Not
// gen3. Use Fissure? OHKO special. Simplest missable Ground: Sand Tomb (70)/Bulldoze n/a.
// Gen3 missable Ground: Earthquake=100. Use Dig? 100. Use Mud Shot? n/a gen3. Use
// Magnitude (100)? Use BONE CLUB? Ground 85 acc EXISTS in gen3 (Bone Club, Marowak).
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function run(p1, p2, plan, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) { if (l && !l.startsWith('|t:|') && !l.startsWith('|split') && l !== '|') lines.push(l); } } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();
  let d = 0, safety = 0;
  while (!stream.battle.ended && safety < 40 && d < plan.length) {
    safety++;
    const rs = stream.battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const step = plan[d];
    if (step.p1) { try { streams.omniscient.write(`>p1 ${step.p1}`); } catch (e) {} }
    if (step.p2) { try { streams.omniscient.write(`>p2 ${step.p2}`); } catch (e) {} }
    for (let i = 0; i < 16; i++) await tick();
    d++;
  }
  try { streams.omniscient.destroy(); } catch (e) {}
  return lines;
}
async function main() {
  const d3 = Dex.forFormat(FORMAT);
  const bc = d3.moves.get('boneclub');
  console.log(`boneclub: type=${bc.type} acc=${bc.accuracy}`);
  // p1 Marowak Bone Club (Ground, 85 acc) into p2 Salamence (Levitate, Flying → Ground 0x anyway).
  // Use a Flying-immune non-Levitate to isolate type-chart, and Gengar (Levitate) for ability.
  const p1 = [mon('Marowak', ['boneclub'], { nature: 'Adamant', evs: { atk: 252, spe: 252 } })];
  const p2lev = [mon('Gengar', ['shadowball'], { ability: 'Levitate', nature: 'Timid', evs: { spa: 252, spe: 252 } })];
  const p2fly = [mon('Skarmory', ['drillpeck'], { ability: 'Keen Eye', nature: 'Impish', evs: { hp: 252, def: 252 } })];
  const seeds = [];
  for (let a = 1; a <= 40; a++) seeds.push([a, a * 7 + 3, a * 13 + 5, a * 29 + 7]);
  for (const [tag, p2] of [['Levitate', p2lev], ['Flying-typechart', p2fly]]) {
    let imm = 0, miss = 0;
    let sample = null;
    for (const s of seeds) {
      const ls = await run(p1, p2, [{ p1: 'move 1', p2: 'move 1' }], s);
      const rel = ls.filter((l) => l.includes('Bone Club') || l.includes('-immune') || l.includes('-miss'));
      if (rel.some((l) => l.includes('[miss]') || l.includes('-miss'))) { miss++; if (!sample) sample = rel; }
      else if (rel.some((l) => l.includes('-immune'))) { imm++; if (!sample) sample = rel; }
    }
    console.log(`${tag}: immune=${imm} miss=${miss} of ${seeds.length}  sample=${JSON.stringify(sample)}`);
  }
}
main();
