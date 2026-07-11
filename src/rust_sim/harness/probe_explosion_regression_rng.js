// probe_explosion_regression_rng.js — GROUND-TRUTH seeds for the DETERMINISTIC Explosion /
// Self-Destruct regression tests (tests/regression_test.rs). Each scenario runs ONE constructed
// move turn in the omniscient sim and reads the post-turn per-mon HP + FAINTED + sub HP + the
// post-turn PRNG seed. The Rust regression test reproduces the identical board + scripted move
// and asserts the same FAINTED/HP/sub-HP + seed.
//
// THE CRUX: gen-3 `useMoveInner` (battle-actions.ts:501-503) faints the USER BEFORE the hit
// resolves — so the user faints UNCONDITIONALLY (through Protect / immunity / a sub / a miss).
// These pins prove that AND that the self-KO is DRAW-FREE (the acc/crit/dmg draws are the same
// count as a normal damaging move; only the resulting faint changes pokemon_left / who acts).
//
//   E1 — Explosion into a PROTECT: the foe Protects; the Explosion is BLOCKED (`-activate
//        Protect`, no foe damage) — but the USER STILL FAINTS. Draws = the foe's first-Protect
//        (draw-free) + the Explosion accuracy (randomChance(100,100)) — then the user faint pauses
//        for a replacement (no Quick Claw). A model that DIDN'T faint the user through a block
//        → a FAINTED-state divergence; a wrong draw count → a seed divergence.
//
//   E2 — Explosion into a GHOST (Normal-immune): no foe damage (`-immune`) — the USER STILL
//        FAINTS. Draws = the Explosion accuracy only (immune short-circuits before crit/dmg),
//        then the user faint (no Quick Claw). Pins the faint-through-immunity STATE + seed.
//
//   E3 — Explosion BREAKS a SUBSTITUTE + the user STILL FAINTS: the foe has a small sub; the
//        Explosion damage hits the sub (breaks it, no carry to the mon) AND the user faints.
//        Draws = acc+crit+dmg (the same as a bare hit; Explosion has no secondary), then the
//        user faint (no Quick Claw). Pins the sub-break + the subber's unchanged HP + the
//        user faint + the seed.
//
//   E4 — MUTUAL Explosion (both last mons) → a gen-3 double-faint TIE. Both Explode the SAME
//        turn (equal speed → an action-order tie-shuffle). Both faint; win(null) TIE. Pins the
//        double-faint STATE + the post-turn seed (the tie-shuffle + acc/crit/dmg draws).
//
// Run:  node src/rust_sim/harness/probe_explosion_regression_rng.js
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
  if (inject) inject(battle);

  const before = battle.prng.getSeed();
  if (plan.p1) streams.omniscient.write(`>p1 ${plan.p1}`);
  if (plan.p2) streams.omniscient.write(`>p2 ${plan.p2}`);
  for (let k = 0; k < 20; k++) await tick();
  const after = battle.prng.getSeed();
  const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
  const subOf = (m) => (m && m.volatiles && m.volatiles['substitute']) ? `SUB(${m.volatiles['substitute'].hp})` : 'noSub';
  const protOf = (m) => (m && m.volatiles && m.volatiles['protect']) ? ' PROT' : '';
  const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} ${subOf(m)}${protOf(m)}` : '-';
  console.log(`\n=== ${label} ===  seed=${seed.join(',')}`);
  console.log(`  seedBefore=${before}`);
  console.log(`  seedAfter =${after}`);
  console.log(`  p1=${fmt(a0)}  left=${battle.sides[0].pokemonLeft}`);
  console.log(`  p2=${fmt(a1)}  left=${battle.sides[1].pokemonLeft}`);
  console.log(`  ended=${battle.ended} winner=${JSON.stringify(battle.winner)}`);
  console.log('  proto: ' + lines.filter((l) => /move|faint|-immune|-activate|-damage|-end|win/.test(l)).join(' ; '));
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // E1 — Explosion into a PROTECT. p1 Electrode Explodes; p2 Blissey Protects (first Protect
  //   always succeeds, draw-free). The move is `-activate Protect` (blocked) — but the USER
  //   faints. Electrode faster (Hasty spe > Bold Blissey). 2-mon p1 team so it can replace.
  await run('E1: Explosion into a Protect — user faints through the block',
    [1, 2, 3, 4],
    [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
     mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    [mon('Blissey', ['protect', 'softboiled'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    null,
    { p1: 'move 1', p2: 'move 1' });

  // E2 — Explosion into a GHOST (Normal-immune). p1 Electrode Explodes into a Gengar; `-immune`,
  //   no damage — the USER faints. Electrode faster. 2-mon p1 team so it can replace.
  await run('E2: Explosion into a Ghost — user faints through immunity',
    [1, 2, 3, 4],
    [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
     mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    [mon('Gengar', ['splash', 'shadowball'], { ability: 'Levitate', nature: 'Timid', evs: { hp: 252, spe: 252 } })],
    null,
    { p1: 'move 1', p2: 'move 1' });

  // E3 — Explosion BREAKS a SUBSTITUTE + user still faints. p2 Blissey has a small sub injected
  //   (hp 1 so ANY Explosion breaks it); p1 Electrode Explodes → the sub breaks (no carry), the
  //   USER faints. Draws = acc+crit+dmg (Explosion has no secondary). Electrode faster.
  await run('E3: Explosion breaks a sub — sub gone, mon HP unchanged, user faints',
    [1, 2, 3, 4],
    [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Hasty', evs: { atk: 252, spe: 252 } }),
     mon('Jolteon', ['thunderbolt'], { ability: 'Volt Absorb', nature: 'Timid', evs: { spe: 252 } })],
    [mon('Blissey', ['softboiled', 'splash'], { ability: 'Natural Cure', nature: 'Bold', evs: { hp: 252, def: 252 } })],
    (battle) => {
      const bliss = battle.sides[1].active[0];
      bliss.addVolatile('substitute');
      bliss.volatiles['substitute'].hp = 1; // ANY hit breaks it; the mon takes NO carry-over
    },
    { p1: 'move 1', p2: 'move 2' }); // p1 Explode INTO the sub ; p2 Splash (draw-free)

  // E4 — MUTUAL Explosion (both last mons) → a gen-3 double-faint TIE. Both Electrodes Explode
  //   the SAME turn; equal speed → an action-order tie-shuffle. Both faint; win(null) TIE.
  await run('E4: mutual Explosion — double faint TIE',
    [1, 2, 3, 4],
    [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Timid', evs: { spe: 252 } })],
    [mon('Electrode', ['explosion', 'thunderbolt'], { ability: 'Soundproof', nature: 'Timid', evs: { spe: 252 } })],
    null,
    { p1: 'move 1', p2: 'move 1' });
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
