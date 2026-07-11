// probe_forced_replacement_resume.js — instrument the SIM's decision-boundary
// structure across a MID-TURN FORCED REPLACEMENT, to characterize the "phantom
// zero-draw move-decision right after a replacement" the port collapses.
//
// Reproduces the DEFERRED protocol scenario `status_para_and_boost_drop` battle 1
// (seed 3152,13225,27580,52191, all-Seismic-Toss). A p2 mon faints mid-turn (a MOVE
// KO), p2 replaces, then the turn resumes. The omniscient capture recorded an EXTRA
// `move`-request boundary right after the replacement whose seedAfter == the prior
// switch decision's (a zero-draw phantom). We instrument makeRequest + the PRNG to
// print, PER request boundary: the requestState, forceSwitch table, the running seed,
// and the draws consumed since the previous boundary. This is the ground-truth oracle
// for the fix: is the phantom truly zero-draw (observation-only) and WHY does it exist?
//
// Run:  node src/rust_sim/harness/probe_forced_replacement_resume.js
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

// The exact status_para_and_boost_drop teams (from the protocol golden TEAM lines).
const P1 = [
  mon('Tyranitar', ['thunderwave', 'crunch', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
  mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
];
const P2 = [
  mon('Starmie', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
  mon('Blissey', ['seismictoss', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
];

// battle 1's recorded choices (the DEC lines), in order. A `switch` request supplies
// only the forced side.
const CHOICES = [
  { p1: 'move 1', p2: 'move 1' },   // dec0
  { p1: 'move 2', p2: 'move 1' },   // dec1
  { p1: 'switch 2', p2: null },     // dec2 (switch, force p1) — p1 replaces
  { p1: 'move 2', p2: 'move 1' },   // dec3
  { p2: 'switch 2', p1: null },     // dec4 (switch, force p2) — p2 replaces
  { p1: 'move 3', p2: 'move 1' },   // dec5 (the PHANTOM? move m2/m0)
  { p1: 'move 1', p2: 'move 1' },   // dec6
  { p1: 'move 1', p2: 'move 1' },   // dec7
];
const SEED = [3152, 13225, 27580, 52191];

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const proto = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) proto.push(l); } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(P1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(P2) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  // Instrument makeRequest to log WHEN the sim opens a request boundary + the seed.
  const realMakeRequest = battle.makeRequest.bind(battle);
  battle.makeRequest = function (type) {
    const t = type || battle.requestState;
    console.log(`      -- makeRequest(${JSON.stringify(type)}) -> requestState will be ${JSON.stringify(t)}` +
      `  seed=${battle.prng.getSeed()}  drawsSoFar=${drawCount}`);
    return realMakeRequest(type);
  };

  console.log(`initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0, prevDraw = 0;
  while (!battle.ended && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= CHOICES.length) break;
    const force = [false, false];
    for (let s = 0; s < 2; s++) {
      const req = battle.sides[s].activeRequest;
      if (req && req.forceSwitch && req.forceSwitch[0]) force[s] = true;
    }
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = CHOICES[i]; i++;
    console.log(`\n[DEC ${i - 1}] requestState=${rs} force=[${force}] choices=${JSON.stringify(entry)} seedBefore=${before}`);
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 24; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''} ${m.status || ''}` : '-';
    console.log(`  -> draws=${drawCount - dc0}  seedAfter=${after}  requestStateNow=${battle.requestState}`);
    console.log(`     p1=${fmt(a0)} | p2=${fmt(a1)}  left=[${battle.sides[0].pokemonLeft},${battle.sides[1].pokemonLeft}]`);
    prevDraw = drawCount;
  }
  console.log(`\nended=${battle.ended} winner=${JSON.stringify(battle.winner)} totalDraws=${drawCount}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
