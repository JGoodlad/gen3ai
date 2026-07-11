// probe_f2_ff_armed_miss.js — the F2 Flash Fire branch: a MISSED Fire move into an
// ALREADY-ARMED Flash Fire holder. Turn 1 arms FF (Flamethrower 100-acc); turn 2 a
// missable Fire Blast (85 acc) into the armed holder. Expect on a MISS `[miss]`+`-miss`,
// on a HIT `-immune|[from] ability: Flash Fire`.
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return { species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31, nature: opts.nature || 'Serious', level: 100, gender: 'N' };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }
async function run(p1team, p2team, plan, seed) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) { if (l && !l.startsWith('|t:|') && !l.startsWith('|split') && l !== '|') lines.push(l); } } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  let d = 0, safety = 0;
  while (!stream.battle.ended && safety < 60 && d < plan.length) {
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
  // p1 Moltres: Flamethrower (100 acc, arms) then Fire Blast (85 acc, missable) into armed FF.
  const p1 = [mon('Moltres', ['flamethrower', 'fireblast'], { nature: 'Modest', evs: { spa: 252, spe: 252 } })];
  const p2 = [mon('Houndoom', ['crunch'], { ability: 'Flash Fire', nature: 'Timid', evs: { spa: 252, spe: 252 } })];
  const seeds = [];
  for (let a = 1; a <= 30; a++) seeds.push([a, a * 2 + 1, a * 3 + 1, a * 5 + 1]);
  let miss = 0, hit = 0;
  for (const s of seeds) {
    const ls = await run(p1, p2, [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }], s);
    // Only look at turn-2 Fire Blast lines (skip the turn-1 Flamethrower arm).
    const fb = ls.filter((l) => l.includes('Fire Blast') || (l.includes('-immune') || l.includes('-miss')));
    const rel = fb.filter((l) => l.includes('Fire Blast') || l.includes('-immune') || l.includes('-miss'));
    const isMiss = rel.some((l) => l.includes('[miss]'));
    const isImm = rel.some((l) => l.includes('-immune'));
    if (isMiss) { if (miss++ < 2) console.log(`MISS seed ${s.join(',')}: ` + JSON.stringify(rel)); }
    else if (isImm) { if (hit++ < 2) console.log(`HIT  seed ${s.join(',')}: ` + JSON.stringify(rel)); }
  }
  console.log(`\ntotals: miss=${miss} hit=${hit} of ${seeds.length}`);
}
main();
