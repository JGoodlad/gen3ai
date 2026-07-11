// probe_forced_replacement_queue.js — deep-instrument the SIM's turnLoop / runAction /
// queue across the mid-turn forced replacement of status_para_and_boost_drop battle 1,
// to see EXACTLY what the resumed turn tail runs after the p2 replacement and WHY the
// sim then opens a zero-draw phantom `move` request boundary (DEC 5).
//
// We instrument battle.runAction (log each action.choice + the queue state before/after),
// battle.makeRequest, and the PRNG. Focus: DEC 3 (the KO turn), DEC 4 (the switch), and
// the phantom DEC 5.
//
// Run:  node src/rust_sim/harness/probe_forced_replacement_queue.js
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

const P1 = [
  mon('Tyranitar', ['thunderwave', 'crunch', 'rockslide'], { item: 'Leftovers', ability: 'Sand Stream', nature: 'Careful', evs: { hp: 252, spd: 252 } }),
  mon('Snorlax', ['bodyslam', 'earthquake'], { item: 'Leftovers', ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } }),
];
const P2 = [
  mon('Starmie', ['surf', 'icebeam'], { item: 'Leftovers', nature: 'Timid', evs: { spa: 252, spe: 252 } }),
  mon('Blissey', ['seismictoss', 'icebeam'], { item: 'Leftovers', nature: 'Bold', evs: { hp: 252, def: 252 } }),
];
const CHOICES = [
  { p1: 'move 1', p2: 'move 1' },   // dec0
  { p1: 'move 2', p2: 'move 1' },   // dec1
  { p1: 'switch 2', p2: null },     // dec2 (force p1)
  { p1: 'move 2', p2: 'move 1' },   // dec3 (the KO turn — p2 Starmie faints)
  { p2: 'switch 2', p1: null },     // dec4 (force p2 → Blissey)
  { p1: 'move 3', p2: 'move 1' },   // dec5 (PHANTOM)
  { p1: 'move 1', p2: 'move 1' },   // dec6
];
const SEED = [3152, 13225, 27580, 52191];

function qstr(battle) {
  return battle.queue.list.map((a) => {
    const who = a.pokemon ? `${a.pokemon.side.id}:${a.pokemon.species.name}` : '';
    return `${a.choice}${who ? `(${who}${a.move ? ',' + a.move.name : ''})` : ''}@o${a.order}`;
  }).join(', ') || '<empty>';
}

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();

  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(P1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(P2) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  let trace = false;

  const realRunAction = battle.runAction.bind(battle);
  battle.runAction = function (action) {
    if (trace) {
      console.log(`    runAction: ${action.choice}` +
        `${action.pokemon ? `(${action.pokemon.side.id}:${action.pokemon.species.name}${action.move ? ',' + action.move.name : ''})` : ''}` +
        ` seed=${battle.prng.getSeed()} draws=${drawCount}  midTurn=${battle.midTurn} req=${JSON.stringify(battle.requestState)}`);
    }
    return realRunAction(action);
  };
  const realMakeRequest = battle.makeRequest.bind(battle);
  battle.makeRequest = function (type) {
    if (trace) {
      console.log(`      makeRequest(${JSON.stringify(type)})  seed=${battle.prng.getSeed()} draws=${drawCount}` +
        ` queue=[${qstr(battle)}]`);
    }
    return realMakeRequest(type);
  };
  const realTurnLoop = battle.turnLoop.bind(battle);
  battle.turnLoop = function () {
    if (trace) console.log(`    >> turnLoop()  req=${JSON.stringify(battle.requestState)} midTurn=${battle.midTurn} queue=[${qstr(battle)}]`);
    const r = realTurnLoop();
    if (trace) console.log(`    << turnLoop() -> ${JSON.stringify(r)}  req=${JSON.stringify(battle.requestState)} midTurn=${battle.midTurn} queue=[${qstr(battle)}]`);
    return r;
  };
  const realCommit = battle.commitChoices ? battle.commitChoices.bind(battle) : null;
  if (realCommit) {
    battle.commitChoices = function () {
      if (trace) console.log(`    commitChoices()  seed=${battle.prng.getSeed()} draws=${drawCount}` +
        ` p1.choice.actions=${battle.sides[0].choice.actions.length} p2.choice.actions=${battle.sides[1].choice.actions.length}`);
      return realCommit();
    };
  }
  // Instrument each side's `choose` to see whether a submitted choice is accepted.
  for (let s = 0; s < 2; s++) {
    const side = battle.sides[s];
    const realChoose = side.choose.bind(side);
    side.choose = function (input) {
      if (trace) console.log(`      ${side.id}.choose(${JSON.stringify(input)})  req=${JSON.stringify(battle.requestState)}` +
        ` activeReq=${side.activeRequest ? JSON.stringify(side.activeRequest.forceSwitch || 'move') : 'null'}`);
      const r = realChoose(input);
      if (trace) console.log(`      ${side.id}.choose -> ${r}  isDone=${side.isChoiceDone()} err=${JSON.stringify(side.choice.error)}`);
      return r;
    };
  }

  let i = 0, safety = 0;
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
    trace = i - 1 >= 3 && i - 1 <= 6; // trace DEC 3..6
    if (trace) console.log(`\n[DEC ${i - 1}] req=${rs} force=[${force}] ${JSON.stringify(entry)} seed=${before} midTurn=${battle.midTurn} queue=[${qstr(battle)}]`);
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 24; k++) await tick();
    if (trace) {
      const after = battle.prng.getSeed();
      console.log(`  => draws=${drawCount - dc0} seedAfter=${after} reqNow=${JSON.stringify(battle.requestState)} midTurnNow=${battle.midTurn} queueNow=[${qstr(battle)}]`);
    }
  }
  console.log(`\nended=${battle.ended} totalDraws=${drawCount}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
