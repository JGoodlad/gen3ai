// probe_beatup_focus_band_rng.js — REAL-Showdown ground truth for the BEAT UP strike ×
// FOCUS BAND draw (`gen3_beatup_focus_band_v1`, random-mode byte-fuzz find ab_7_7 @
// master-seed 200724).
//
// gen-4's `focusband.onDamage` puts `this.randomChance(1,10)` FIRST in its `&&`, so a Focus
// Band holder draws the `randomChance(1,10)` roll on EVERY move-damage hit (JS short-circuit),
// not only lethal ones. A Beat Up STRIKE runs the full `spreadMoveHit` → `spreadDamage` →
// `runEvent('Damage')` → the Focus Band handler, so a Beat Up strike into a FB holder draws it
// (the sim's draw#5 at ab_7_7 dec 19). `run_beat_up` used to apply the strike damage without
// `focus_band_damage` → a one-fewer-draw desync.
//
// Prints the post-turn seed for a FB target (draws the roll) and a no-item control (does not),
// so the Rust pin asserts the seed matches AND differs from the control.
//
// Run from src/rust_sim:  node harness/probe_beatup_focus_band_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream.js'));
const { Teams } = require(path.join(PS, 'dist/sim/teams.js'));

const FORMAT = 'gen3customgame';
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: opts.ivs || IV31,
    nature: opts.nature || 'Serious', level: opts.level || 100, gender: 'N',
  };
}
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  const before = battle.prng.getSeed();
  if (plan.p1) streams.omniscient.write(`>p1 ${plan.p1}`);
  if (plan.p2) streams.omniscient.write(`>p2 ${plan.p2}`);
  for (let k = 0; k < 18; k++) await tick();
  const after = battle.prng.getSeed();
  const t = battle.sides[1].active[0];
  console.log(`\n=== ${label} ===  seed=${seed.join(',')}`);
  console.log(`  seedBefore=${before}`);
  console.log(`  seedAfter =${after}`);
  console.log(`  target=${t.species.name} ${t.hp}/${t.maxhp} item=${t.item || '-'} fnt=${t.fainted}`);
  try { streams.omniscient.destroy(); } catch (e) { void e; }
}

async function main() {
  // p1 Houndour is a 1-MON team → Beat Up strikes exactly ONCE (only the active is healthy).
  // p2 Snorlax is bulky → the typeless small-BP strike is non-lethal → the FB roll draws but
  // the survive branch is not taken.
  const houndour = [mon('Houndour', ['beatup', 'splash'], { ability: 'Flash Fire', nature: 'Adamant', evs: { atk: 252 } })];
  const SEED = [3, 5, 7, 9];

  await run('FB: Beat Up (1 strike) into a FOCUS BAND Snorlax (the FB roll DRAWS)',
    SEED, houndour,
    [mon('Snorlax', ['splash'], { item: 'focusband', ability: 'Immunity', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    { p1: 'move 1', p2: 'move 1' }); // p1 Beat Up ; p2 Splash (draw-free)

  await run('CTRL: Beat Up (1 strike) into a NO-ITEM Snorlax (NO FB roll → one fewer draw)',
    SEED, houndour,
    [mon('Snorlax', ['splash'], { item: '', ability: 'Immunity', nature: 'Careful', evs: { hp: 252, spd: 252 } })],
    { p1: 'move 1', p2: 'move 1' });
}
main().catch((e) => { console.error(e); process.exit(1); });
