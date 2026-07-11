// probe_explosion_rng.js — instrument the gen3 EXPLOSION / SELF-DESTRUCT draw model
// bit-for-bit against the OMNISCIENT in-process BattleStream (no server).
//
// THE CRUX: gen-3 `useMoveInner` (battle-actions.ts:501-503) faints the USER
// (`this.battle.faint(pokemon)`) BEFORE the hit resolves (`trySpreadMoveHit`), for
// `gen !== 4 && move.selfdestruct === 'always'`. So the self-KO is UNCONDITIONAL —
// the user faints regardless of whether the hit lands, misses, is immune, is
// blocked by Protect, or hits a Substitute. This probe VERIFIES:
//
//   (A) PLAIN hit: Explosion is a Normal damaging move (BP 250 gen3), halves the
//       target's Def; the foe takes big damage, the USER faints. Draws = acc(-skip:
//       explosion accuracy is 100 → randomChance(100,100)) + crit + damage.
//   (B) vs SUBSTITUTE: the damage hits the SUB (breaks it, no carry); the USER still
//       faints. Draw count = same as a bare hit (acc+crit+dmg; the secondary is N/A —
//       Explosion has no secondary).
//   (C) vs PROTECT: the move is BLOCKED (no damage to the foe) — but does the USER
//       still faint? (useMoveInner faints the user at :501 BEFORE trySpreadMoveHit,
//       so YES.) Draw count: the protect resolved (priority 3) with its own stall
//       roll; the Explosion draws its accuracy then is blocked (no crit/dmg).
//   (D) vs a GHOST (immune to Normal): no damage — does the USER still faint? (YES.)
//       Draw count: acc only (immune short-circuits before crit/damage).
//   (E) MISS: gen3 Explosion accuracy is 100 → it never misses on its own accuracy,
//       BUT we can force the picture by confirming the resolved accuracy value and
//       that the user faints on a landed hit. (If accuracy could fail, the user would
//       STILL faint — the faint precedes the hit.)
//   (F) MUTUAL Explosion (both last mons): a true double-faint TIE.
//
// We wrap battle.prng.next to count raw draws per decision window and snapshot the
// user's fainted state + the foe's HP + any substitute each turn.
//
// Run:  node src/rust_sim/harness/probe_explosion_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams, Dex } = require(path.join(PS, 'dist/sim'));

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

function dumpResolved() {
  const dex3 = Dex.forFormat(FORMAT);
  for (const id of ['explosion', 'selfdestruct']) {
    const m = dex3.moves.get(id);
    console.log(`=== resolved gen3 ${id} ===`);
    console.log(`  accuracy=${m.accuracy} basePower=${m.basePower} type=${m.type} ` +
      `category=${m.category} selfdestruct=${m.selfdestruct} target=${m.target} ` +
      `priority=${m.priority}`);
  }
}

async function run(label, p1team, p2team, plan, inject) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  for (const inj of (inject || [])) {
    const m = inj.side === undefined ? null : battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 50) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const entry = plan[i]; i++;
    if (entry.injectBefore) {
      for (const inj of entry.injectBefore) {
        const m = battle.sides[inj.side].active[0];
        if (inj.hp !== undefined) m.hp = inj.hp;
        if (inj.status) m.setStatus(inj.status, m, null, true);
      }
    }
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 20; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const subOf = (m) => (m && m.volatiles && m.volatiles['substitute'])
      ? `SUB(${m.volatiles['substitute'].hp})` : '';
    const protOf = (m) => (m && m.volatiles && m.volatiles['protect']) ? 'PROT' : '';
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp} ${m.status || '-'}${m.fainted ? ' FNT' : ''} ${subOf(m)}${protOf(m)}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedBefore=${before} seedAfter=${after}`);
    console.log(`        p1=${fmt(a0)} | p2=${fmt(a1)}  left=[${battle.sides[0].pokemonLeft},${battle.sides[1].pokemonLeft}]`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${JSON.stringify(battle.winner)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  dumpResolved();

  // (A) PLAIN Explosion: Electrode Explodes into Snorlax. The USER faints; the foe
  //     takes big def-halved damage. Baseline: a Splash/Splash draws 1 (Quick Claw);
  //     an Explosion turn that KOs the user faints → no Quick Claw (faint pause).
  await run('PLAIN Explosion (user faints, foe takes def-halved damage)',
    [mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Gengar', ['splash'])],
    [mon('Snorlax', ['splash', 'softboiled'], { evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // Splash/Splash baseline → 1 (Quick Claw)
      { p1: 'move 1', p2: 'move 1' }, // Electrode Explosion → user faints, Snorlax hit
    ]);

  // (B) Explosion into a SUBSTITUTE: the damage hits the sub (breaks it, no carry);
  //     the USER STILL faints. Snorlax subs, Electrode Explodes into the sub.
  await run('Explosion into a SUBSTITUTE (breaks sub, user still faints)',
    [mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Gengar', ['splash'])],
    [mon('Blissey', ['substitute', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'move 1' }, // p1 Splash, p2 Blissey Substitute → sub up
      { p1: 'move 1', p2: 'move 2' }, // p1 Explosion INTO the sub ; user faints
    ]);

  // (C) Explosion into a PROTECT: the move is BLOCKED (no damage to the foe). Does
  //     the USER still faint? (YES — faint precedes the hit.) Blissey Protects,
  //     Electrode Explodes.
  await run('Explosion into a PROTECT (blocked — does the user still faint?)',
    [mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Gengar', ['splash'])],
    [mon('Blissey', ['protect', 'softboiled'], { ability: 'Natural Cure', evs: { hp: 252, def: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 Explosion, p2 Protect → blocked, user faints
    ]);

  // (D) Explosion into a GHOST (immune to Normal): no damage — does the USER still
  //     faint? (YES.) Electrode Explodes into Gengar (Ghost).
  await run('Explosion into a GHOST (Normal-immune — user still faints?)',
    [mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Jolteon', ['splash'])],
    [mon('Gengar', ['splash', 'shadowball'], { evs: { hp: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 Explosion into Ghost → immune, user faints
    ]);

  // (E) MUTUAL Explosion (both last mons): a true double-faint TIE. Single-mon teams.
  await run('MUTUAL Explosion (both last mons) → gen-3 TIE',
    [mon('Electrode', ['explosion', 'thunderbolt'], { evs: { spe: 252 } })],
    [mon('Electrode', ['explosion', 'thunderbolt'], { evs: { spe: 252 } })],
    [
      { p1: 'move 1', p2: 'move 1' }, // both Explode → double faint TIE
    ]);

  // (F) Explosion-KO-forces-a-replacement: Electrode Explodes into a foe it KOs, and
  //     the USER faints — BOTH sides need a replacement. (2-mon teams both sides.)
  await run('Explosion KO forces a DOUBLE replacement (user + foe both faint)',
    [mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Jolteon', ['splash'])],
    [mon('Gengar', ['splash']), // Gengar is Ghost — immune! use a frail non-ghost instead:
     mon('Jolteon', ['splash'])],
    [
      { p1: 'move 1', p2: 'move 1' }, // Explosion into Gengar (ghost) → user faints, foe immune (single replace p1)
    ]);

  // (F2) Explosion that KOs a frail non-ghost foe → DOUBLE replacement (both faint).
  await run('Explosion KOs a frail foe → DOUBLE replacement',
    [mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 }, nature: 'Adamant' }),
     mon('Jolteon', ['splash'])],
    [mon('Pichu', ['splash'], { level: 5 }),
     mon('Jolteon', ['splash'])],
    [
      { p1: 'move 1', p2: 'move 1' }, // Explosion KOs Pichu; user faints → double replace
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
