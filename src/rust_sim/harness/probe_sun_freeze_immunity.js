// probe_sun_freeze_immunity.js — verify the gen3 SUN → freeze-immunity rule bit-for-bit.
//
// The A/B fuzzer's "ice-freeze cluster" (196 repros, fingerprint
// expected=None got=Some(Freeze), seed matching) is the port FREEZING a mon the sim
// leaves un-frozen. Root cause (source read): the base `sunnyday` weather registers
//   onImmunity(type, pokemon) { if (effectiveWeather() !== 'sunnyday') return;
//                               if (type === 'frz') return false; }
// so `setStatus` → `runStatusImmunity('frz')` → `runEvent('Immunity', ..., 'frz')`
// returns FALSE while the field weather is Sun (Drought / Sunny Day) → the freeze is
// NOT applied. The immunity is checked BEFORE `runEvent('SetStatus')` (the gen3ou clause
// shuffle), and the sun `onImmunity` draws NOTHING — so the freeze SECONDARY's
// random(100) still draws (seed matches) but the freeze must simply not land.
//
// This probe drives the OMNISCIENT in-process Battle (no server, no streams — a direct
// `new Battle()` so prng hooking is race-free) and verifies:
//   A. Under DROUGHT sun, an Ice Beam whose freeze-secondary WOULD land (roll < 10)
//      leaves the target UN-frozen (the sim), while the SAME seed with NO sun freezes it.
//   B. The DRAW COUNT of the Ice-Beam turn is IDENTICAL sun vs no-sun (the secondary
//      random(100) still fires under sun; the immunity gate is draw-free) — proving the
//      port only needs a draw-free STATE gate. [gen3customgame — no clause shuffle either]
//   C. gen3ou: the SetStatus clause shuffle is SKIPPED under sun (the immunity returns
//      before runEvent('SetStatus')) — so a sun freeze-secondary draws ONE FEWER than an
//      ou no-sun landed freeze (which DOES draw the size-2 clause shuffle).
//   D. frz has no weather-cure handler → an already-frozen mon persists under sun (the
//      port must gate only APPLICATION, never thaw an already-frozen mon).
//
// Run:  node src/rust_sim/harness/probe_sun_freeze_immunity.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { Battle } = require(path.join(PS, 'dist/sim/battle'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

const IV31 = { hp: 31, atk: 31, def: 31, spa: 31, spd: 31, spe: 31 };
const EV0 = { hp: 0, atk: 0, def: 0, spa: 0, spd: 0, spe: 0 };
function mon(species, moves, opts = {}) {
  return {
    species, item: opts.item || '', ability: opts.ability || 'No Ability', moves,
    evs: { ...EV0, ...(opts.evs || {}) }, ivs: IV31,
    nature: opts.nature || 'Hardy', level: opts.level || 100, gender: opts.gender || 'N',
  };
}

// Drive ONE Ice-Beam turn on a direct Battle. p1 Ice-Beams p2's active; both submit a
// move so the turn commits. Count raw prng.next() draws for the turn. Returns the
// target's post-turn status + the draw count + the effective weather.
function runIceBeamTurn({ format, p1Team, p2Team, seed }) {
  const battle = new Battle({ formatid: format, seed });
  battle.setPlayer('p1', { name: 'A', team: Teams.pack(p1Team) });
  battle.setPlayer('p2', { name: 'B', team: Teams.pack(p2Team) });

  let draws = 0;
  // The raw draw primitive: PRNG.random() → this.rng.next(). Everything
  // (randomChance / sample / shuffle) funnels through rng.next(), so hooking the
  // BACKEND next() counts every raw PRNG consumption exactly.
  const backend = battle.prng.rng;
  const origNext = backend.next.bind(backend);
  let counting = false;
  backend.next = (...a) => { if (counting) draws++; return origNext(...a); };

  counting = true;
  battle.choose('p1', 'move icebeam');
  battle.choose('p2', 'move earthquake');
  counting = false;

  const target = battle.sides[1].active[0];
  return { status: target.status || 'none', draws, weather: battle.field.effectiveWeather() };
}

function main() {
  const dex3 = Dex.forFormat('gen3customgame');
  console.log('=== resolved gen3 sunnyday.onImmunity present? ===');
  const sunny = dex3.conditions.get('sunnyday');
  console.log('  sunnyday.onImmunity =', typeof sunny.onImmunity === 'function' ? 'FUNCTION (frz immunity)' : sunny.onImmunity);
  const frz = dex3.conditions.get('frz');
  console.log('  frz.onImmunity =', frz.onImmunity, ' (status has none — immunity is on the weather)');
  console.log('  frz.onWeather/onUpdate cure? =',
    (typeof frz.onWeather === 'function' || typeof frz.onUpdate === 'function') ? 'YES (would thaw!)' : 'NO (already-frozen persists under sun)');

  // Freeze target: Groudon (Drought for sun; Pressure for the no-sun control — customgame
  // accepts any ability). Ground-type → Ice Beam neutral, no frz type-immunity, high HP.
  const p1 = [mon('Regice', ['icebeam', 'thunderbolt', 'psychic', 'explosion'], { evs: { spa: 252 } })];
  const p2sun = [mon('Groudon', ['swordsdance', 'earthquake', 'rockslide', 'thunderwave'], { ability: 'Drought' })];
  const p2nosun = [mon('Groudon', ['swordsdance', 'earthquake', 'rockslide', 'thunderwave'], { ability: 'Pressure' })];

  // A + B (gen3customgame): find a seed whose freeze-secondary lands in the NO-sun control,
  // then run the SAME seed under sun.
  let found = null;
  for (let s = 0; s < 800 && !found; s++) {
    const seed = [s & 0xffff, (s * 7 + 1) & 0xffff, (s * 13 + 3) & 0xffff, (s * 101 + 7) & 0xffff];
    const noSun = runIceBeamTurn({ format: 'gen3customgame', p1Team: p1, p2Team: p2nosun, seed });
    if (noSun.status === 'frz') {
      const sun = runIceBeamTurn({ format: 'gen3customgame', p1Team: p1, p2Team: p2sun, seed });
      found = { seed, noSun, sun };
    }
  }
  console.log('\n=== A/B: landed-freeze seed, no-sun vs sun (gen3customgame) ===');
  if (!found) {
    console.log('  NO landed-freeze seed found in 800 tries (unexpected).');
  } else {
    console.log('  seed:', found.seed.join(','));
    console.log(`  NO-SUN control : weather=${found.noSun.weather} status=${found.noSun.status} draws=${found.noSun.draws}`);
    console.log(`  SUN (Drought)  : weather=${found.sun.weather} status=${found.sun.status} draws=${found.sun.draws}`);
    console.log(`  => sun BLOCKS freeze? ${found.sun.status === 'none' ? 'YES' : 'NO'}`);
    console.log(`  => draw COUNT identical (immunity draw-free)? ${found.noSun.draws === found.sun.draws ? 'YES' : `NO (delta=${found.noSun.draws - found.sun.draws})`}`);
  }

  // C (gen3ou): the clause shuffle is skipped under sun → sun draws ONE FEWER.
  let foundOu = null;
  for (let s = 0; s < 800 && !foundOu; s++) {
    const seed = [(s * 3 + 5) & 0xffff, (s * 11 + 2) & 0xffff, (s * 17 + 9) & 0xffff, (s * 29 + 4) & 0xffff];
    const noSun = runIceBeamTurn({ format: 'gen3ou', p1Team: p1, p2Team: p2nosun, seed });
    if (noSun.status === 'frz') {
      const sun = runIceBeamTurn({ format: 'gen3ou', p1Team: p1, p2Team: p2sun, seed });
      foundOu = { seed, noSun, sun };
    }
  }
  console.log('\n=== C: gen3ou clause-shuffle skipped under sun ===');
  if (!foundOu) {
    console.log('  NO landed-freeze seed found in gen3ou (ou changes the seed cadence — non-fatal).');
  } else {
    console.log('  seed:', foundOu.seed.join(','));
    console.log(`  gen3ou NO-SUN : status=${foundOu.noSun.status} draws=${foundOu.noSun.draws}  (drew the SetStatus size-2 shuffle)`);
    console.log(`  gen3ou SUN    : status=${foundOu.sun.status} draws=${foundOu.sun.draws}`);
    console.log(`  => sun draws ONE FEWER (no clause shuffle)? ${foundOu.sun.draws === foundOu.noSun.draws - 1 ? 'YES' : `NO (delta=${foundOu.noSun.draws - foundOu.sun.draws})`}`);
  }
}

main();
