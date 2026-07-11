// probe_sun_freeze_regression_rng.js — GROUND-TRUTH seed for the FZ1 sun-freeze pin.
//
// FZ1 pins `gen3_sun_freeze_immunity_v1`: under Sun (Drought), a mon CANNOT be frozen —
// the base `sunnyday` weather's `onImmunity` returns false for `frz` at
// `runStatusImmunity`, BEFORE `runEvent('SetStatus')` — so an Ice Beam whose freeze
// secondary WOULD land leaves the target UN-frozen, DRAW-FREE (the secondary random(100)
// still fires; only the application is suppressed). WRONG (pre-fix): the port froze it.
//
// This probe (direct omniscient Battle, race-free) constructs the exact pin scenario —
// p1 Regice Ice-Beams p2 Groudon(Drought, sun up) — and SWEEPS an init seed to find one
// where the freeze secondary WOULD land WITHOUT sun (a control) but the mon stays
// un-frozen WITH sun. For that seed it prints the SEED-BEFORE the turn, the SEED-AFTER
// (the exact ground truth the Rust FZ1 pin hardcodes), the target's post-turn status, and
// the raw draw count — so the port's post-turn seed for the SAME scenario must match.
//
// It also captures a WITHOUT-SUN control at the SAME seed (a Groudon with a non-weather
// ability) that DOES freeze — proving the fix's STATE difference is the sun immunity and
// the DRAW COUNT is identical (draw-free gate).
//
// Run:  node src/rust_sim/harness/probe_sun_freeze_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams } = require(path.join(PS, 'dist/sim'));

const FORMAT = 'gen3customgame';
const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: 'N',
  };
}

// Drive ONE Ice-Beam turn on a direct omniscient Battle and return the ground truth.
function iceBeamTurn(seed, p2Ability) {
  const p1 = [mon('Regice', ['icebeam', 'thunderbolt', 'psychic', 'explosion'], { evs: { spa: 252 } })];
  const p2 = [mon('Groudon', ['swordsdance', 'earthquake', 'rockslide', 'thunderwave'], { ability: p2Ability })];
  const battle = new Battle({ formatid: FORMAT, seed });
  battle.setPlayer('p1', { name: 'P1', team: Teams.pack(p1) });
  battle.setPlayer('p2', { name: 'P2', team: Teams.pack(p2) });
  const seedBefore = battle.prng.getSeed();
  let draws = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = (...a) => { draws++; return realNext(...a); };

  battle.choose('p1', 'move icebeam');   // Regice Ice Beams the Groudon
  battle.choose('p2', 'move swordsdance'); // Groudon just boosts (a no-draw self-move)

  const target = battle.sides[1].active[0];
  return {
    seedBefore, seedAfter: battle.prng.getSeed(), draws,
    status: target.status || 'none', targetHp: target.hp, targetMax: target.maxhp,
    weather: battle.field.effectiveWeather(),
  };
}

function main() {
  // Sweep for a seed where NO-sun freezes (Pressure control) — that's a seed whose
  // freeze secondary lands — then confirm SUN (Drought) leaves it un-frozen at the same
  // seed, with an IDENTICAL draw count.
  for (let s = 0; s < 5000; s++) {
    const seed = [s & 0xffff, (s * 7 + 1) & 0xffff, (s * 13 + 3) & 0xffff, (s * 101 + 7) & 0xffff];
    const ctrl = iceBeamTurn(seed, 'Pressure');
    if (ctrl.status !== 'frz') continue;
    const sun = iceBeamTurn(seed, 'Drought');
    if (sun.status !== 'none' || sun.weather !== 'sunnyday') continue;
    if (sun.draws !== ctrl.draws) continue; // require the draw-free equality for a clean pin
    console.log('=== FZ1 ground truth (sun-freeze immunity) ===');
    console.log(`  init seed (Rust seeds this): ${seed.join(',')}`);
    console.log(`  SEED-BEFORE turn           : ${sun.seedBefore}`);
    console.log('');
    console.log('  --- SUN (Drought) : the PIN scenario ---');
    console.log(`    weather=${sun.weather} target_status=${sun.status} target_hp=${sun.targetHp}/${sun.targetMax}`);
    console.log(`    SEED-AFTER = ${sun.seedAfter}   draws=${sun.draws}`);
    console.log('');
    console.log('  --- NO-SUN (Pressure) : the control (freeze LANDS) ---');
    console.log(`    weather=${ctrl.weather} target_status=${ctrl.status} target_hp=${ctrl.targetHp}/${ctrl.targetMax}`);
    console.log(`    SEED-AFTER = ${ctrl.seedAfter}   draws=${ctrl.draws}`);
    console.log('');
    console.log(`  => sun blocks freeze (state), draw count IDENTICAL (${sun.draws}==${ctrl.draws}): the pin`);
    return;
  }
  console.log('NO qualifying seed found in 5000 tries (unexpected).');
}

main();
