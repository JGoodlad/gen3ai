// probe_cascade_hazard_order.js — reproduce the e2e_9 DOUBLE-FAINT → CASCADE bug
// shape in the OMNISCIENT sim and nail: when p1's fresh entrant faints on its OWN
// side's Spikes (chaining a THIRD replacement), does p2's fresh entrant take its
// side's Spikes ONCE, and WHEN (before/after the cascade pause)?
//
// Setup mirrors e2e_9 dec41-43: p1 SIDE has 3 Spikes, p2 SIDE has 1 Spike, Sandstorm
// active; a double faint forces BOTH to replace; p1's replacement is pre-damaged so
// its OWN 3-layer Spikes KO it on entry (the cascade); p2's replacement is a fresh
// grounded Steel (Jirachi) on the 1-Spike side. We trace EACH runSwitch (which side/
// mon, hp before/after) + the request pauses, printing the ground-truth per-entrant HP.
//
// Run:  node src/rust_sim/harness/probe_cascade_hazard_order.js
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

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify([7, 11, 13, 17])}}`);
  // p1: Tyranitar (Sand Stream, sets Sandstorm) leads. Forretress lays Spikes on p2.
  //   A Weezing-esque grounded mon that Explodes for the double faint. A LOW-HP
  //   grounded replacement that dies to 3-layer Spikes (the cascade). Skarmory lays
  //   Spikes on p1 side (via p2's spiker). We need p1 side=3, p2 side=1.
  // Simpler: p1 has a spiker for p2 side (1 layer), p2 has a spiker for p1 side (3
  //   layers). p1 has Sand-Stream Tyranitar for weather. Then a mutual Explosion.
  const p1 = [
    mon('Tyranitar', ['spikes', 'crunch', 'splash'], { ability: 'Sand Stream', evs: { hp: 252, def: 252 } }), // weather + lays p2side spikes
    mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }), // the double-faint trigger
    // A FAST grounded mon so its runSwitch sorts BEFORE p2's Jirachi (the order that
    // mirrors e2e_9: p1's fainting entrant runSwitch runs FIRST). Pre-damaged so its
    // OWN 3-layer spikes KO it on entry (the cascade).
    mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', evs: { spe: 252 } }), // fast grounded → runSwitch first
    mon('Sandshrew', ['splash'], { level: 5, ability: 'Sand Veil' }),
  ];
  const p2 = [
    mon('Forretress', ['spikes', 'splash'], { ability: 'No Ability', evs: { hp: 252, def: 252 } }), // lays p1side spikes (3)
    mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
    mon('Jirachi', ['bodyslam', 'splash'], { ability: 'Serene Grace', evs: { hp: 248 } }), // fresh grounded Steel on p2 1-spike side
  ];
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  const actions = battle.actions;
  const realRunSwitch = actions.runSwitch.bind(actions);
  actions.runSwitch = function (pokemon) {
    const layersFor = (s) => (s.sideConditions.spikes ? s.sideConditions.spikes.layers : 0);
    const hpBefore = pokemon.hp;
    console.log(`    >>runSwitch(${pokemon.side.id}:${pokemon.species.name}) hpBefore=${hpBefore}/${pokemon.maxhp} mySpikes=${layersFor(pokemon.side)}`);
    const r = realRunSwitch(pokemon);
    console.log(`    <<runSwitch(${pokemon.side.id}:${pokemon.species.name}) hpAfter=${pokemon.hp}${pokemon.fainted ? ' FNT' : ''} chip=${hpBefore - pokemon.hp}`);
    return r;
  };

  const spk = () => [battle.sides[0].sideConditions.spikes, battle.sides[1].sideConditions.spikes].map((s) => (s ? s.layers : 0));
  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''}` : '-';

  // Plan: lay spikes to p1side=3, p2side=1; bring Electrodes in; mutual Explosion →
  // double faint; then the forced replacements. Pre-damage p1's Diglett is unneeded —
  // lvl-3 Diglett dies to 3-layer spikes automatically.
  const plan = [
    { p1: 'move 1', p2: 'move 1' }, // p1 Tyranitar Spikes→p2(1); p2 Forretress Spikes→p1(1)
    { p1: 'move 3', p2: 'move 1' }, // p1 splash; p2 Spikes→p1(2)
    { p1: 'move 3', p2: 'move 1' }, // p1 splash; p2 Spikes→p1(3)   (p1 side=3, p2 side=1)
    { p1: 'switch 2', p2: 'switch 2' }, // Electrodes in
    { p1: 'move 1', p2: 'move 1', tag: 'DOUBLE-FAINT' }, // mutual Explosion
    { p1: 'switch 3', p2: 'switch 3', tag: 'FORCED-DOUBLE-REPLACE' }, // p1 Diglett (dies on 3-spikes), p2 Jirachi
    { p1: 'switch 4', tag: 'CASCADE-P1' }, // p1 cascade Sandshrew
    { tag: 'END' },
  ];
  let i = 0, safety = 0;
  while (!battle.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[i]; i++;
    // Before the FORCED double-replace, pre-damage p1's Diglett (slot 2) so its OWN
    // 3-layer Spikes KO it on entry → the CASCADE (mirrors e2e_9's pre-damaged
    // Tyranitar re-entering at 37 HP and dying to 3-layer spikes).
    if (entry.tag === 'FORCED-DOUBLE-REPLACE') {
      const jol = battle.sides[0].pokemon.find((m) => m.species.id === 'jolteon');
      if (jol) { jol.hp = 10; console.log(`    (pre-damaged Jolteon to hp=10 → 3-layer spikes = floor(maxhp/4) KO on entry)`); }
    }
    const before = battle.prng.getSeed();
    console.log(`\n[${rs}] ${entry.tag || ''} ${JSON.stringify({ p1: entry.p1, p2: entry.p2 })} seedBefore=${before}`);
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 24; k++) await tick();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(`  => seedAfter=${battle.prng.getSeed()} spikes=${JSON.stringify(spk())} p1=${fmt(a0)} | p2=${fmt(a1)} left=[${battle.sides[0].pokemonLeft},${battle.sides[1].pokemonLeft}]`);
  }
  console.log(`\nended=${battle.ended} winner=${JSON.stringify(battle.winner)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
