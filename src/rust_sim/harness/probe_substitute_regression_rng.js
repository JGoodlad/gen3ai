// probe_substitute_regression_rng.js — GROUND-TRUTH seeds for the DETERMINISTIC Substitute
// regression tests (tests/regression_test.rs). Each scenario INJECTS an exact pre-move board
// (a sub up, optional confusion) into the omniscient sim, runs ONE move turn, and reads the
// post-turn per-mon HP + sub HP + status + the post-turn PRNG seed. The Rust regression test
// reproduces the identical injected board + the same scripted move and asserts the same
// HP/sub-HP/status + seed.
//
//   S1 — THE SECONDARY draw-COUNT (the CRUX). A Body Slam (par 30 secondary) into a SUBBED
//        mon draws acc+crit+dmg+SECONDARY(100)+QuickClaw — the SAME count as a bare hit (the
//        gen-3 quirk: the secondary random(100) is STILL drawn against a sub). A model that
//        SKIPPED the secondary random(100) behind a sub draws ONE FEWER → a divergent seed.
//        The sub absorbs the damage (mon HP unchanged) and NO paralysis applies (status `-`).
//
//   S2 — THE BREAK (excess does NOT carry). A big Body Slam into a SMALL sub breaks it
//        (sub HP → 0 → removed); the mon's HP is UNCHANGED (no carry-over). Draws = the same
//        acc+crit+dmg+secondary(100)+QC. Pins the break STATE + the no-carry HP.
//
//   S3 — THE CONFUSION self-hit hits the MON (not the sub). A subbed + confused mon that
//        self-hits damages its OWN HP while the sub HP stays put. The confusion draws
//        (randomChance(1,2) then random(16)) are unchanged by the sub. Pins the self-hit
//        target + the unchanged sub HP + the seed.
//
//   S4 — TRI ATTACK secondary suppression draw-COUNT. Tri Attack into a SUBBED mon draws
//        acc+crit+dmg+random(100) but NOT the random(3) `sample` (the secondary's onHit runs
//        on the null target) — so it draws ONE FEWER than a Tri Attack that LANDS on a bare
//        mon. A model that drew the random(3) behind a sub → a divergent seed.
//
// Run:  node src/rust_sim/harness/probe_substitute_regression_rng.js
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

// Run a single injected move turn. `inject(battle)` sets the pre-move board (a sub up, etc.);
// `plan` is the per-side `>pN ...` move strings. Reads the post-turn HP/subHP/status + seed.
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
  const subOf = (m) => (m && m.volatiles && m.volatiles['substitute']) ? `SUB(${m.volatiles['substitute'].hp})` : 'noSub';
  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} ${subOf(m)} b=[${m.boosts.atk},${m.boosts.spd}]` : '-';
  console.log(`\n=== ${label} ===  seed=${seed.join(',')}`);
  console.log(`  seedBefore=${before}`);
  console.log(`  seedAfter =${after}`);
  console.log(`  p1=${fmt(a0)}`);
  console.log(`  p2=${fmt(a1)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // S1 — secondary draw-COUNT: p1 Snorlax Body Slams (par 30) the SUBBED p2 Blissey. The sub
  //   absorbs the hit (Blissey HP unchanged), NO par applies, and the secondary random(100) IS
  //   drawn (the seed proof). p1 faster (Adamant Snorlax > Bold Blissey). Inject a sub on Blissey.
  await run('S1: Body Slam into a sub — secondary random(100) STILL drawn',
    [1, 2, 3, 4],
    [mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Blissey', ['softboiled', 'splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    (battle) => {
      const bliss = battle.sides[1].active[0];
      bliss.addVolatile('substitute'); // creates a sub at floor(maxhp/4) (draw-free onStart)
      bliss.volatiles['substitute'].hp = Math.floor(bliss.maxhp / 4);
    },
    { p1: 'move 1', p2: 'move 2' }); // p1 Body Slam INTO the sub ; p2 Splash (draw-free)

  // S2 — BREAK (no carry): p1 Snorlax Body Slams the SUBBED p2 Snorlax with a SMALL sub (we
  //   inject the sub HP to 1 so ANY hit breaks it). The mon HP is UNCHANGED (no carry), the sub
  //   is gone. Same draws (acc+crit+dmg+secondary(100)+QC). (p2 Snorlax is Normal — NOT immune
  //   to Body Slam, unlike a Ghost; the 1-HP sub is broken by any hit.)
  await run('S2: Body Slam BREAKS a 1-HP sub — no carry to the mon',
    [1, 2, 3, 4],
    [mon('Snorlax', ['bodyslam', 'splash'], { ability: 'Immunity', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['splash'], { ability: 'Immunity', nature: 'Careful', evs: { hp: 252 } })],
    (battle) => {
      const lax = battle.sides[1].active[0];
      lax.addVolatile('substitute');
      lax.volatiles['substitute'].hp = 1; // any hit breaks it; the mon takes NO carry-over
    },
    { p1: 'move 1', p2: 'move 1' });

  // S3 — CONFUSION self-hit hits the MON: p1 Snorlax has a sub up AND is confused; it Splashes
  //   but (on a failed confusion check) self-hits — the MON's HP drops while the sub HP stays.
  //   We sweep seeds; pick one where the confusion check FAILS (a self-hit happens — the mon's
  //   HP DROPS while SUB(131) is unchanged). p2 Blissey Splashes (draw-free). The confusion
  //   counter is pinned to 4 (so the decrement → 3 is deterministic; the Rust test injects 4).
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [5, 4, 3, 2], [9, 9, 9, 9], [42, 17, 8, 3]]) {
    await run(`S3: confusion self-hit behind a sub hits the MON (seed ${seed})`,
      seed,
      [mon('Snorlax', ['splash', 'bodyslam'], { ability: 'Immunity', nature: 'Adamant', evs: { hp: 252, atk: 252 } })],
      [mon('Blissey', ['splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252 } })],
      (battle) => {
        const lax = battle.sides[0].active[0];
        lax.addVolatile('substitute');
        lax.volatiles['substitute'].hp = Math.floor(lax.maxhp / 4);
        // Confuse with a FIXED counter so the test is deterministic (we set time directly,
        // matching a known confusion; the Rust test injects the same fixed counter).
        lax.addVolatile('confusion');
        lax.volatiles['confusion'].time = 4; // a long confusion so it persists; counter pinned
      },
      { p1: 'move 1', p2: 'move 1' });
  }

  // S4 — TRI ATTACK secondary suppression: p1 Porygon2 Tri Attacks the SUBBED p2 Blissey. The
  //   random(100) draws (the 20% gate) but NOT the random(3) sample — so behind a sub it draws
  //   ONE FEWER than a Tri Attack that lands on a bare mon. We pick a seed where the random(100)
  //   PASSES (so a bare hit WOULD draw the random(3)) to isolate the suppression.
  for (const seed of [[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]]) {
    await run(`S4: Tri Attack into a sub — random(3) NOT drawn (seed ${seed})`,
      seed,
      [mon('Porygon2', ['triattack', 'splash'], { ability: 'Trace', nature: 'Modest', evs: { spa: 252, spe: 252 } })],
      [mon('Blissey', ['softboiled', 'splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
      (battle) => {
        const bliss = battle.sides[1].active[0];
        bliss.addVolatile('substitute');
        bliss.volatiles['substitute'].hp = Math.floor(bliss.maxhp / 4);
      },
      { p1: 'move 1', p2: 'move 2' });
  }
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
