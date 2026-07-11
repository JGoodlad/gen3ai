// probe_double_replacement_cascade_regression_rng.js — GROUND TRUTH for the
// `double_replacement_cascade_does_not_rechip_the_other_sides_entrant` regression pin.
//
// Constructs (in the OMNISCIENT sim) the exact e2e_9-class scenario the Rust pin replays:
// a mutual-Explosion DOUBLE FAINT forces BOTH sides to replace; p1's fresh entrant is
// FAST + pre-damaged so its OWN 3-layer Spikes KO it on entry (its runSwitch runs FIRST,
// then faints → the cascade). Because the FAINTING side's runSwitch is first, the sim's
// gen-3-singles `faintMessages` → `cancelAction(getAllActive())` DROPS p2's still-pending
// runSwitch → p2's fresh entrant is NEVER chipped by p2's 1-layer Spikes (stays FULL HP).
// The p1 cascade replacement then takes p1's 3-layer Spikes.
//
// To make the pin CONSTRUCTED + INJECTED (no long spike-laying script), we set the two
// sides' Spikes + pre-damage p1's cascade entrant DIRECTLY on the sim battle object right
// after `>start` (before the first decision), matching the Rust pin's injection. We then
// run ONE mutual-Explosion turn + the forced replacements, and print the ground-truth
// per-entrant HP + the post-decision SEEDS the pin asserts.
//
// Run:  node src/rust_sim/harness/probe_double_replacement_cascade_regression_rng.js
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

// The Rust pin seeds `start_with_switchins` at this RAW seed. The sim's `>start` switch-in
// events (both leads, no weather/Intimidate here → DRAW-FREE) leave the seed unchanged, so
// the sim's first-decision seed == this raw seed (matching the port's start_with_switchins).
const RAW_SEED = [11, 22, 33, 44];

async function main() {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(RAW_SEED)}}`);
  // p1: Electrode lead (fast, mutual Explosion); Jolteon = the FAST pre-damaged cascade
  //   entrant (grounded → 3-layer Spikes KO on entry, runSwitch FIRST since it's faster
  //   than p2's Snorlax); Sandshrew bench = the cascade replacement.
  // p2: Electrode lead (mutual Explosion); Snorlax = the SLOW grounded entrant on the
  //   1-Spike side (its runSwitch would chip it — but must be CANCELLED by the cascade).
  const p1 = [
    mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
    mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', evs: { spe: 252 } }),
    mon('Sandshrew', ['scratch'], { level: 20, ability: 'Sand Veil' }),
  ];
  const p2 = [
    mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
    mon('Snorlax', ['bodyslam'], { ability: 'Immunity', evs: { hp: 252 } }),
  ];
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  // INJECT the board the pin injects: 3 Spikes on p1 side, 1 Spike on p2 side; pre-damage
  // p1's Jolteon so its 3-layer Spikes (floor(maxhp/4)) KO it on entry. Use the lead as the
  // source so addSideCondition is happy (source only affects [of]-attribution, not the chip).
  const src0 = battle.sides[0].active[0], src1 = battle.sides[1].active[0];
  const spikesEff = battle.dex.conditions.get('spikes');
  battle.sides[0].addSideCondition('spikes', src1, spikesEff);
  battle.sides[0].addSideCondition('spikes', src1, spikesEff);
  battle.sides[0].addSideCondition('spikes', src1, spikesEff);
  battle.sides[1].addSideCondition('spikes', src0, spikesEff);
  const jol = battle.sides[0].pokemon.find((m) => m.species.id === 'jolteon');
  jol.hp = 1; // 3-layer spikes on entry (floor(maxhp/4) >= 1) KO
  console.log(`spikes p1=${battle.sides[0].sideConditions.spikes.layers} p2=${battle.sides[1].sideConditions.spikes.layers}; Jolteon pre-damaged to hp=1 (maxhp=${jol.maxhp})`);

  // RESEED to the RAW seed right before the first decision (the two identical-speed Electrode
  // leads make the sim's `>start` switch-in speed-tie shuffle DRAW, advancing the seed past
  // RAW_SEED — which the bounded Rust `start_with_switchins` does NOT model; it leaves
  // `prng = new Prng(RAW_SEED)`). Reset the sim to the SAME raw seed so the DECISION draws line
  // up bit-for-bit with the Rust pin (the documented CONSTRUCTED-pin reseed, like
  // probe_switch_tie_weather_regression_rng.js). The board injection above is DRAW-FREE.
  battle.prng = new PRNG(RAW_SEED.slice());
  console.log(`reseeded to RAW_SEED = ${battle.prng.getSeed()} (== Rust start_with_switchins prng)`);

  const actions = battle.actions;
  const realRunSwitch = actions.runSwitch.bind(actions);
  actions.runSwitch = function (pokemon) {
    const lf = (s) => (s.sideConditions.spikes ? s.sideConditions.spikes.layers : 0);
    const hb = pokemon.hp;
    console.log(`    >>runSwitch(${pokemon.side.id}:${pokemon.species.name}) hpBefore=${hb}/${pokemon.maxhp} mySpikes=${lf(pokemon.side)}`);
    const r = realRunSwitch(pokemon);
    console.log(`    <<runSwitch(${pokemon.side.id}:${pokemon.species.name}) hpAfter=${pokemon.hp}${pokemon.fainted ? ' FNT' : ''} chip=${hb - pokemon.hp}`);
    return r;
  };

  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''}` : '-';
  const plan = [
    { p1: 'move 1', p2: 'move 1', tag: 'DOUBLE-FAINT (mutual Explosion)' },
    { p1: 'switch 2', p2: 'switch 2', tag: 'FORCED-DOUBLE-REPLACE (p1 Jolteon KO on 3-spikes → cascade; p2 Snorlax UNCHIPPED)' },
    { p1: 'switch 3', tag: 'CASCADE-P1 (Sandshrew, takes p1 3-spikes)' },
  ];
  let i = 0, safety = 0;
  const seeds = [];
  while (!battle.ended && safety < 40 && i < plan.length) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[i]; i++;
    const before = battle.prng.getSeed();
    console.log(`\n[${rs}] ${entry.tag} seedBefore=${before}`);
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 24; k++) await tick();
    const after = battle.prng.getSeed();
    seeds.push(after);
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    console.log(`  => seedAfter=${after}  p1=${fmt(a0)} | p2=${fmt(a1)} left=[${battle.sides[0].pokemonLeft},${battle.sides[1].pokemonLeft}]`);
  }
  console.log(`\n=== GROUND TRUTH for the pin ===`);
  console.log(`decision seeds (in order): ${JSON.stringify(seeds)}`);
  console.log(`(the pin asserts p2 Snorlax stays at FULL HP [unchipped] after the forced replacement, and these seeds)`);
  try { streams.omniscient.destroy(); } catch (e) {}
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
