// probe_tox_stage_switch.js — does the gen3 TOXIC stage RESET when the poisoned
// mon switches out and back in?
//
// The resolved gen3 `tox` condition CARRIES an `onSwitchIn(){ this.effectState.
// stage = 0 }` — but gen3's runSwitch (mods/gen4/scripts inherited) never
// dispatches the SwitchIn event, so the question is whether that reset is DEAD
// CODE in gen3. Scenario: Zangoose (Immunity-free foe) toxics Milotic; Milotic
// takes residual #1 (stage 1 = maxhp/16), switches out, switches back in, takes
// its next residual. If the stage RESET: that residual is maxhp/16 again (stage
// ramps 0→1). If it PERSISTED: it is 2×maxhp/16 (stage ramps 1→2).
//
// Run from src/rust_sim: node harness/probe_tox_stage_switch.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));

const tick = () => new Promise(r => setImmediate(r));

const p1 = 'Smeargle||leftovers|owntempo|toxic,splash,tackle,spore|Serious|85,85,85,85,85,85|M|||100|';
// two bulky mons so nothing faints; NO leftovers on p2 (clean damage reads), no Natural Cure/Immunity/Synchronize
const p2 = 'Milotic||choiceband|marvelscale|splash,tackle,surf,recover|Serious|85,85,85,85,85,85|M|||100|]Swampert||choiceband|torrent|splash,tackle,surf,earthquake|Serious|85,85,85,85,85,85|M|||100|';

(async () => {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[7,7,7,7]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();

  const b = stream.battle;
  const milo = () => b.sides[1].pokemon.find(p => p.name === 'Milotic');
  const maxhp = milo().maxhp;
  const step = async (c1, c2) => {
    streams.omniscient.write(`>p1 ${c1}`);
    streams.omniscient.write(`>p2 ${c2}`);
    for (let i = 0; i < 16; i++) await tick();
  };

  // T1: toxic Milotic (Smeargle Toxic never misses? acc 85/90 — retry loop not needed at this seed; check)
  let hpBefore = milo().hp;
  await step('move 1', 'move 1'); // toxic vs splash
  if (milo().status !== 'tox') { console.log('TOXIC MISSED at this seed — pick another seed'); process.exit(1); }
  const dmg1 = hpBefore - milo().hp;
  console.log(`residual #1 (fresh tox): ${dmg1} (maxhp=${maxhp}, maxhp/16=${Math.floor(maxhp / 16)}) stage=${b.sides[1].pokemon[0].statusState.stage}`);

  // T2: Milotic switches out (Swampert in)
  await step('move 2', 'switch 2');
  // T3: Milotic back in
  await step('move 2', 'switch 2');
  const hpAtReentry = milo().hp; // after entry (no hazards) — residual for THIS turn already applied at end of T3
  const stageAtReentry = milo().statusState.stage; // THE DISCRIMINATOR: 1 = RESET fired at runSwitch, >=2 = persisted
  console.log(`re-entry turn residual: ${hpAtReentry}, stage=${stageAtReentry}`);
  // The T3 residual is the first one after re-entry: delta vs hp before T3's residual
  // Simpler: read statusState.stage directly (the oracle's own counter) + one more residual:
  const before = milo().hp;
  await step('move 2', 'move 1'); // splash vs splash
  const dmgNext = before - milo().hp;
  console.log(`next residual after re-entry: ${dmgNext} = ${dmgNext / Math.floor(maxhp / 16)}x maxhp/16, stage=${milo().statusState.stage}`);
  // VERDICT on the DISCRIMINATING signal (the re-entry stage, the oracle's own counter):
  // stage 1 at re-entry = the tox onSwitchIn reset FIRED at runSwitch-time (gen3_tox_stage_persists_v1:
  // it resets when the runSwitch RUNS; it persists only when the queued runSwitch is CANCELLED by a
  // faint — see pins TX1/TX2). The old verdict compared dmgNext (2x vs 3x maxhp/16), which exceeds
  // maxhp/16 under EITHER hypothesis — a non-discriminating comparison that printed a FALSE conclusion.
  console.log(stageAtReentry <= 1
    ? 'VERDICT: stage RESET at the runSwitch (re-entry stage 1) — the onSwitchIn reset IS live in gen3'
    : 'VERDICT: stage PERSISTED across the switch (re-entry stage >= 2)');
  process.exit(0);
})().catch(e => { console.error(e); process.exit(1); });
