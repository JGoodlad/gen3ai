// probe_leechseed_regression_rng.js — GROUND-TRUTH seeds for the DETERMINISTIC Leech Seed
// regression tests (tests/regression_test.rs). Each scenario INJECTS the exact post-move
// board into the omniscient sim, runs ONE residual turn (both Splash / filler), and reads
// the post-turn per-mon HP + the post-turn PRNG seed. The Rust regression test reproduces
// the identical injected board + the same scripted move and asserts the same HP + seed.
//
//   L1 — THE 4-WAY RESIDUAL ORDER (the risk case): leech + Leftovers + SANDSTORM chip +
//        BURN DoT on the SAME seeded mon. The verified residual order is sandstorm(o=8) →
//        Leftovers(o=10,s=4) → LEECH(o=10,s=5) → burn(o=10,s=6). A wrong leech subOrder
//        re-orders the heal/drain → a divergent post-turn HP (the leech is sub 5 — BETWEEN
//        Leftovers and burn). The seeder Meganium heals the leech amount. Distinct speeds
//        (no residual handler tie → no shuffle draw), so the SEED isolates the move accuracy
//        draws + Quick Claw; the HP isolates the residual ORDER.
//
//   L2 — THE LEECH RESIDUAL HANDLER TIE (both actives seeded at EQUAL speed): two seeded
//        mons at IDENTICAL cached speed → their two leech handlers (order 10, sub 5) TIE →
//        the residual handler-sort's Fisher-Yates tie-shuffle draws ONE random(0,2). A wrong
//        leech subOrder / a missing leech handler in the sort changes the tie-group count →
//        a divergent SEED. We mirror-seed two Snorlax from two Meganium... no, single active;
//        instead: each side's active is seeded BY THE OTHER (p1 active seeds p2, p2 seeds p1)
//        at equal speed, so BOTH leech handlers tie. The drain/heal cross — each seeder heals.
//
//   L3 — THE SEEDER-FAINTED GATE: a seeded mon whose SEEDER's active is fainted takes NO
//        leech drain (the whole onResidual returns: `if (!target || target.fainted) return`).
//        We seed a mon, faint the seeder's active (inject hp 0 + fainted), and confirm the
//        seeded mon's HP is UNCHANGED by the leech that turn (only Leftovers/etc. apply).
//
// Run:  node src/rust_sim/harness/probe_leechseed_regression_rng.js
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

// Run a single injected residual turn. `inject(battle)` sets the post-move board; `plan` is
// the per-side `>pN ...` move strings for the ONE turn we record.
async function run(label, seed, p1team, p2team, inject, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const lines = [];
  (async () => { for await (const ch of streams.omniscient) for (const l of ch.split('\n')) if (l) lines.push(l); })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  inject(battle);

  const before = battle.prng.getSeed();
  if (plan.p1) streams.omniscient.write(`>p1 ${plan.p1}`);
  if (plan.p2) streams.omniscient.write(`>p2 ${plan.p2}`);
  for (let k = 0; k < 18; k++) await tick();
  const after = battle.prng.getSeed();
  const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
  const seededOf = (m) => (m && m.volatiles && m.volatiles['leechseed']) ? 'SEEDED' : '';
  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} ${seededOf(m)}` : '-';
  console.log(`\n=== ${label} ===  seed=${seed.join(',')}`);
  console.log(`  seedBefore=${before}`);
  console.log(`  seedAfter =${after}`);
  console.log(`  p1=${fmt(a0)}`);
  console.log(`  p2=${fmt(a1)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // L1 — the 4-way residual order (leech + Leftovers + sand + burn on the seeded mon).
  // p1 Meganium (Leftovers) is the SEEDER; p2 Gengar (Leftovers, burned, under sand) is the
  // seeded mon. We inject the seed + sand + burn + chip Gengar, then both Splash so the
  // residual is the SOLE HP change. Distinct speeds (Gengar faster) so no residual tie.
  await run('L1: leech + leftovers + sand + burn order',
    [1, 2, 3, 4],
    [mon('Meganium', ['synthesis', 'leechseed'], { item: 'Leftovers', evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { item: 'Leftovers', ability: 'Levitate', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    (battle) => {
      const meg = battle.sides[0].active[0], gen = battle.sides[1].active[0];
      battle.field.setWeather('sandstorm', meg); battle.field.weatherState.duration = 0;
      gen.setStatus('brn', gen, null, true);
      gen.addVolatile('leechseed', meg);
      gen.volatiles['leechseed'].sourceSlot = meg.getSlot();
      gen.hp = 200; meg.hp = 100;
    },
    { p1: 'move 1', p2: 'move 1' }); // Meganium Synthesis (so we ALSO see the leech heal stack), Gengar Splash

  // L2 — BOTH actives seeded at EQUAL speed (the leech handler TIE → one shuffle draw).
  // Two Snorlax mirror (equal speed). Each is seeded BY THE OTHER's slot, so both leech
  // handlers tie at order 10 sub 5 → the residual tie-shuffle draws one random(0,2). Both
  // Splash so the residual is the sole change. A wrong leech subOrder / missing handler in
  // the sort changes the tie group → a divergent seed.
  await run('L2: both seeded at equal speed (leech handler tie)',
    [5, 6, 7, 8],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', nature: 'Serious', evs: { hp: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', nature: 'Serious', evs: { hp: 252 } })],
    (battle) => {
      const a = battle.sides[0].active[0], b = battle.sides[1].active[0];
      // Cross-seed: a seeded by b, b seeded by a (so the heals cross). Both leech handlers tie.
      a.addVolatile('leechseed', b); a.volatiles['leechseed'].sourceSlot = b.getSlot();
      b.addVolatile('leechseed', a); b.volatiles['leechseed'].sourceSlot = a.getSlot();
      a.hp = 300; b.hp = 300;
    },
    { p1: 'move 1', p2: 'move 1' });

  // L3 — the SEEDER-FAINTED gate (no drain when the seeder's active is fainted). p1 Meganium
  // seeds p2 Gengar, then p1's active FAINTS (inject hp 0 + fainted). The leech residual sees
  // the seeder's active fainted → returns early → Gengar takes NO leech drain this turn. We
  // record Gengar's HP UNCHANGED by leech. (Only p2 acts; p1 is fainted → forced switch, so
  // we drive p2 Splash and read the residual; the leech is skipped.)
  await run('L3: seeder active fainted → no leech drain',
    [9, 10, 11, 12],
    [mon('Meganium', ['leechseed', 'splash'], { evs: { hp: 252 } }),
     mon('Blissey', ['softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [mon('Gengar', ['splash'], { ability: 'Levitate', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    (battle) => {
      const meg = battle.sides[0].active[0], gen = battle.sides[1].active[0];
      gen.addVolatile('leechseed', meg); gen.volatiles['leechseed'].sourceSlot = meg.getSlot();
      gen.hp = 200;
      // FAINT the seeder's active (Meganium): zero HP + fainted flag + leave it active so the
      // residual reads a fainted seeder (a forced switch happens AFTER the residual).
      meg.hp = 0; meg.faint();
      battle.faintMessages();
    },
    { p1: '', p2: 'move 1' }); // p1 is fainted (no move); p2 Splash → residual runs the leech with a fainted seeder
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
