// probe_confusion_choiceband_regression_rng.js — GROUND TRUTH for the
// `confusion_self_hit_applies_choice_band` regression pin.
//
// THE BUG (e2e_194 dec15): the CONFUSION self-hit damage. gen-4 confusion (which gen-3
// inherits, data/mods/gen4/conditions.ts:74-83) runs `this.actions.getDamage(pokemon,
// pokemon, 40)` — the FULL getDamage, so the attacker's `onModifyAtk` item **Choice Band
// ×1.5 (physical)** folds into the self-hit. The port's `apply_confusion_self_hit` was
// passing NO atk stat mods → it used the stored Atk, not the CB-boosted Atk → the self-hit
// under-dealt (Aerodactyl: ~71 base vs the CB-boosted 90-106), diverging the HP.
//
// This probe INJECTS a `confusion` volatile on a Choice-Band Aerodactyl (right after
// `>start`, DRAW-FREE, mirroring the Rust pin's `mon.confusion = Some(n)` inject), reseeds
// to the RAW seed (the identical-speed leads make `>start` tie-shuffle draw), then runs ONE
// move turn where Aerodactyl is confused. With the chosen seed the confusion self-hit FIRES
// (the 50% `random(1,2)` lands on self-hit) and the CB-boosted self-hit damage is applied.
// Prints Aerodactyl's post-turn HP + the post-turn seed the pin asserts.
//
// Run:  node src/rust_sim/harness/probe_confusion_choiceband_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));

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

// Try a handful of raw seeds; pick one where the confusion self-hit FIRES (Aerodactyl's HP
// drops on its own turn) so the pin exercises the CB-boosted self-hit deterministically.
async function trySeed(rawSeed, confTime) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(rawSeed)}}`);
  // Aerodactyl (Choice Band) vs a passive Regirock (Splash) — Regirock is bulky + Rock so it
  // won't KO / interfere; Aerodactyl is confused and must risk a self-hit. Both leads are
  // DISTINCT-speed (Aerodactyl 130-base fast, Regirock 50-base slow) so `>start` is DRAW-FREE
  // — but we still reseed to raw for exact alignment.
  const p1 = [mon('Aerodactyl', ['rockslide'], { item: 'choiceband', ability: 'Pressure', evs: { atk: 252, spe: 252 }, nature: 'Adamant' })];
  const p2 = [mon('Regirock', ['splash'], { ability: 'Clear Body', evs: { hp: 252, def: 252 } })];
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  const aero = battle.sides[0].active[0];
  // INJECT confusion (draw-free here — we set the volatile + its time directly).
  aero.addVolatile('confusion');
  aero.volatiles['confusion'].time = confTime; // survive this turn (decrements to confTime-1)
  const hp0 = aero.hp, maxhp = aero.maxhp;
  const atkStored = aero.storedStats.atk, atkCB = aero.getStat('atk');

  // Reseed to the RAW seed for exact Rust alignment (the inject above is draw-free).
  battle.prng = new PRNG(rawSeed.slice());

  streams.omniscient.write(`>p1 move 1`);
  streams.omniscient.write(`>p2 move 1`);
  for (let k = 0; k < 20; k++) await tick();
  const after = battle.prng.getSeed();
  const aeroHp = battle.sides[0].active[0].hp;
  const selfHit = hp0 - aeroHp; // damage Aerodactyl took (Regirock Splashes → only self-hit)
  const confAfter = battle.sides[0].active[0].volatiles['confusion'] ? battle.sides[0].active[0].volatiles['confusion'].time : 0;
  try { streams.omniscient.destroy(); } catch (e) {}
  return { rawSeed, seedAfter: after, hp0, maxhp, aeroHp, selfHit, atkStored, atkCB, confAfter };
}

async function main() {
  const seeds = [
    [1, 2, 3, 4], [5, 7, 11, 13], [7, 11, 13, 17], [11, 22, 33, 44],
    [100, 200, 300, 400], [12345, 23456, 34567, 45678], [999, 888, 777, 666],
  ];
  console.log('Looking for a seed where the confusion self-hit FIRES (Aerodactyl HP drops, Regirock Splashes)...\n');
  for (const s of seeds) {
    const r = await trySeed(s, 3);
    const fired = r.selfHit > 0;
    console.log(`rawSeed=${JSON.stringify(s)}  selfHit=${r.selfHit}  aeroHp=${r.aeroHp}/${r.maxhp}  seedAfter=${r.seedAfter}  atk(stored=${r.atkStored}, CB=${r.atkCB})  confAfter=${r.confAfter}  ${fired ? '<<< FIRES' : ''}`);
  }
  console.log('\nPick a FIRES seed for the pin: assert Aerodactyl.hp == aeroHp (CB-boosted self-hit) + seedAfter.');
  console.log('(A no-CB port would leave MORE HP — the self-hit under-deals — so the pin FAILS if CB is reverted.)');
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
