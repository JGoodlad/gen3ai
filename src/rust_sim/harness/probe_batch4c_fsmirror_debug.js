// probe_batch4c_fsmirror_debug.js — one-off diagnostic for the fs_mirror_tie golden case
// (raw seed [59902,13663,56236,8309]): per-decision PRNG draw labels + seeds + lines.
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
  for (let i = 3; i < st.length && frames.length < 5; i++) {
    const mm = st[i].match(/at ([\w.<>]+) /);
    if (mm) frames.push(mm[1]);
  }
  return frames.join('<');
}

async function main() {
  const p1 = [
    mon('Jirachi', ['futuresight', 'seismictoss'], { evs: { spa: 252, hp: 252, spe: 252 } }),
    mon('Blissey', ['seismictoss'], { evs: { hp: 252 } }),
  ];
  const p2 = [mon('Jirachi', ['futuresight', 'seismictoss'], { evs: { spa: 252, hp: 252, spe: 252 } })];
  const seed = [59902, 13663, 56236, 8309];

  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  let draws = [];
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { const v = realNext(...a); draws.push(drawLabel()); return v; };

  let n = 0, safety = 0;
  while (!battle.ended && safety < 30 && n < 12) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    draws = [];
    const l0 = log.length;
    let c1, c2;
    if (rs === 'switch') {
      c1 = battle.sides[0].activeRequest && battle.sides[0].activeRequest.forceSwitch ? 'switch 2' : null;
      c2 = null;
    } else {
      const want = (n % 2 === 0) ? 1 : 2;
      c1 = `move ${want}`; c2 = `move ${want}`;
    }
    if (c1) streams.omniscient.write(`>p1 ${c1}`);
    if (c2) streams.omniscient.write(`>p2 ${c2}`);
    for (let k = 0; k < 20; k++) await tick();
    console.log(`dec ${n} [${rs}] ${c1}/${c2} draws=${draws.length} seed ${before} -> ${battle.prng.getSeed()}`);
    draws.forEach((dl, k) => console.log(`   DRAW[${k}] ${dl}`));
    log.slice(l0).forEach((l) => { if (/move|cant|-start|-end|-damage|-miss|faint|turn|residual|upkeep/.test(l)) console.log(`   LINE ${l}`); });
    n++;
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
