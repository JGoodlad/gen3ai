'use strict';
// Ground truth for the R21 pin (`gen3_batonpass_stall_pursuit_copy_v1`): a Pursuit-into-a
// Baton-Passing (Protect-having) foe. The passer carries `stall` (turn-1 Protect) + `pursuit`
// (the beforeTurnMove-laid volatile); Baton Pass copies BOTH to the entrant, whose two
// NO_ORDER/subOrder-2 residual duration handlers TIE → the turn-2 residual draws one
// Fisher-Yates shuffle. Prints the per-decision post-boundary seed.
const path = require('path');
const PS = '/home/goodlad/dev/gen3ai/deps/pokemon-showdown';
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
// p1 Tyranitar (Pursuit, Sand Stream); p2 Zapdos (Protect, Baton Pass) → Metagross entrant.
const p1 = 'Tyranitar||Leftovers|SandStream|Crunch,Pursuit,Flamethrower,Roar|Modest|240,,,176,76,16|M|,0,,,,|||';
const p2 = 'Zapdos||Leftovers|Pressure|Thunderbolt,HiddenPowerGrass,Protect,BatonPass|Modest|,,,192,116,200|N|,2,,30,,|||,Grass,,,,]Metagross||ChoiceBand|ClearBody|MeteorMash,Earthquake,HiddenPowerRock,Explosion|Adamant|196,252,,,,60|N|,,30,,30,30|||,Rock,,,,';
const SEED = [11, 22, 33, 44];
const cmds = [
  ['p1', 'move 4'], ['p2', 'move 3'],   // T1: Roar (blocked by Protect) + Protect → Zapdos gets stall
  ['p1', 'move 2'], ['p2', 'move 4'],   // T2: Pursuit (lays pursuit on Zapdos) + Baton Pass
  ['p2', 'switch 2'],                   // T2 forced switch: Metagross in (inherits stall + pursuit)
  ['p1', 'move 1'], ['p2', 'move 4'],   // T3: Crunch + Explosion
];
function tick() { return new Promise((r) => setTimeout(r, 0)); }
(async () => {
  const stream = new BattleStream(); const streams = getPlayerStreams(stream);
  (async () => { for await (const _ of streams.omniscient) {} })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[${SEED}]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 16; i++) await tick();
  const b = stream.battle;
  // Reseed to the RAW seed before the first decision (the port's start_with_switchins is
  // draw-free), matching the regression-pin convention (probe_batch4_pursuit_bench_regression_rng.js).
  b.prng = new PRNG(SEED.slice());
  console.log('RAW seed (port seeds here)=' + b.prng.getSeed());
  const queue = cmds.map((c) => ({ side: c[0], choice: c[1] }));
  let d = 0;
  while (!b.ended && queue.length) {
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const need = [false, false];
    if (rs === 'move') { need[0] = need[1] = true; }
    else { const f = b.sides.map((x) => x.activeRequest && x.activeRequest.forceSwitch && x.activeRequest.forceSwitch[0]); need[0] = !!f[0]; need[1] = !!f[1]; }
    for (let i = 0; i < 2; i++) {
      if (!need[i]) continue;
      const qi = queue.findIndex((c) => c.side === `p${i + 1}`);
      if (qi < 0) { console.log('OUT OF CMDS'); return; }
      streams[`p${i + 1}`].write(queue.splice(qi, 1)[0].choice);
    }
    for (let i = 0; i < 16; i++) await tick();
    console.log(`decision ${d++} seedAfter=` + b.prng.getSeed() + ' | p2active=' + b.sides[1].active[0].species.name + ' hp=' + b.sides[1].active[0].hp + '/' + b.sides[1].active[0].maxhp + ' | p1active=' + b.sides[0].active[0].species.name + ' hp=' + b.sides[0].active[0].hp);
  }
})();
