// probe_double_replacement_spikes_rng.js — instrument the gen3 DOUBLE-FAINT →
// DOUBLE-REPLACEMENT → SPIKES-CASCADE entry-hazard attribution bit-for-bit against
// the OMNISCIENT in-process BattleStream (no server). This is the ground-truth
// oracle for the OPEN e2e bug (EDGE_CASES.md): when a mutual double-faint forces
// BOTH sides to replace, each fresh entrant must take ITS OWN side's Spikes chip
// (grounded, per-layer) — and if one entrant faints on Spikes chaining a THIRD
// replacement, the OTHER side's entrant must NOT be re-chipped.
//
// The scenarios:
//   (A) MUTUAL Explosion double-faint, DISTINCT Spikes layers per side (1 on p1, 3
//       on p2). Both entrants switch in — each takes its OWN side's Spikes:
//       p1 entrant = maxhp/8, p2 entrant = maxhp/4. Assert the exact post-replacement
//       HP of BOTH new actives + the running seed.
//   (B) MUTUAL Explosion double-faint where p1's low-HP entrant is KO'd by its OWN
//       (3-layer) Spikes on entry → chains a THIRD replacement. Verify the p2 entrant
//       is chipped EXACTLY ONCE (its own side's layers) and NOT re-chipped by the
//       cascade; verify the p1 cascade entrant takes p1's Spikes.
//   (C) MUTUAL Explosion double-faint where ONE entrant is Spikes-IMMUNE
//       (Flying / Levitate) — pins the grounded gate under the cascade (the immune
//       entrant takes ZERO; the grounded one takes its own side's chip).
//
// We wrap battle.prng.next to count draws per decision window, snapshot the two
// actives' HP each step, AND instrument the queue + runSwitch to print, PER runSwitch,
// which pokemon/side is being processed and the resulting HP (the direct attribution
// trace the port must match).
//
// Run:  node src/rust_sim/harness/probe_double_replacement_spikes_rng.js
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

const SEED = [7, 11, 13, 17];

async function run(label, p1team, p2team, plan, opts = {}) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  (async () => { for await (const ch of streams.omniscient) { void ch; } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(opts.seed || SEED)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;

  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  // Instrument the gen3/gen4 runSwitch to trace PER-ENTRANT hazard attribution.
  const actions = battle.actions;
  const realRunSwitch = actions.runSwitch.bind(actions);
  actions.runSwitch = function (pokemon) {
    const sideId = pokemon.side.id;
    const hpBefore = pokemon.hp;
    const layersFor = (s) => (s.sideConditions.spikes ? s.sideConditions.spikes.layers : 0);
    console.log(
      `      >> runSwitch(${sideId}: ${pokemon.species.name}) hpBefore=${hpBefore}/${pokemon.maxhp}` +
      ` mySideSpikes=${layersFor(pokemon.side)} foeSideSpikes=${layersFor(battle.sides[1 - pokemon.side.n])}`
    );
    const r = realRunSwitch(pokemon);
    console.log(
      `      << runSwitch(${sideId}: ${pokemon.species.name}) hpAfter=${pokemon.hp}/${pokemon.maxhp}` +
      `${pokemon.fainted ? ' FAINTED' : ''}  (chip=${hpBefore - pokemon.hp})`
    );
    return r;
  };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);
  let i = 0, safety = 0;
  while (!battle.ended && safety < 60) {
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
    for (let k = 0; k < 24; k++) await tick();
    const after = battle.prng.getSeed();
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const spk = [battle.sides[0].sideConditions.spikes, battle.sides[1].sideConditions.spikes]
      .map((s) => (s ? s.layers : 0));
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0}  seedBefore=${before} seedAfter=${after}`);
    console.log(`        spikes[p1,p2]=${JSON.stringify(spk)}  p1=${fmt(a0)} | p2=${fmt(a1)}  left=[${battle.sides[0].pokemonLeft},${battle.sides[1].pokemonLeft}]`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${JSON.stringify(battle.winner)}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // Both leads = Electrode (fast, so it acts; Explosion halves foe Def but the target
  // here is the OTHER Electrode which also explodes → mutual double-faint).
  // We lay Spikes on both sides FIRST via a Skarmory-esque helper, but simpler: use
  // dedicated Spikes-layers via injection is not possible; instead script Spikes moves.
  // We give each side a spiker + an Electrode + grounded/immune bench.

  // (A) MUTUAL Explosion double-faint, DISTINCT spikes (p1 side=1, p2 side=3).
  //     After both explode, each side's fresh grounded entrant takes ITS OWN spikes.
  //     p1 lays 3 spikes on p2 (via 3 Spikes moves); p2 lays 1 spike on p1.
  //     Then both bring in Electrode and mutually Explode; the double replacement
  //     brings in each side's grounded Snorlax → p1 entrant maxhp/8, p2 entrant maxhp/4.
  await run('(A) MUTUAL Explosion double-faint, spikes p1=1 p2=3 → each entrant its OWN chip',
    [mon('Skarmory', ['spikes', 'roost', 'splash'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 } }),
     mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [mon('Forretress', ['spikes', 'rapidspin', 'splash'], { ability: 'No Ability', evs: { hp: 252, def: 252 } }),
     mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      // Lay spikes: p1 Skarmory lays on p2 side; p2 Forretress lays on p1 side.
      { p1: 'move 1', p2: 'move 1' }, // p1 spikes→p2(1); p2 spikes→p1(1)
      { p1: 'move 1', p2: 'move 3' }, // p1 spikes→p2(2); p2 splash  (p1 side stays 1)
      { p1: 'move 1', p2: 'move 3' }, // p1 spikes→p2(3); p2 splash  (p2 side now 3, p1 side 1)
      // Bring both Electrodes in.
      { p1: 'switch 2', p2: 'switch 2' }, // Electrode in both (grounded → each takes its side's spikes)
      // Mutual Explosion → double faint → double replacement (Snorlax each side).
      { p1: 'move 1', p2: 'move 1' },
      // Forced replacements: p1 switch Snorlax (slot 3), p2 switch Snorlax (slot 3).
      { p1: 'switch 3', p2: 'switch 3', force: true },
      { stop: true },
    ]);

  // (B) MUTUAL Explosion double-faint where p1's entrant is KO'd by its OWN spikes on
  //     entry → chains a THIRD replacement. p1 has 3 spikes on it; its Snorlax
  //     replacement is a low-HP lvl-1 mon → spikes-KO on entry → p1 replaces AGAIN.
  //     The p2 entrant (grounded, on a 1-spike side) must be chipped EXACTLY ONCE.
  await run('(B) double-faint → p1 entrant spikes-KO → THIRD replacement (p2 entrant chipped once)',
    [mon('Forretress', ['spikes', 'splash'], { ability: 'No Ability', evs: { hp: 252, def: 252 } }),
     mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Diglett', ['splash'], { level: 1, ability: 'Sand Veil' }),   // lvl1 grounded → spikes-KO on entry
     mon('Sandshrew', ['splash'], { level: 1, ability: 'Sand Veil' })], // lvl1 grounded → spikes-KO too
    [mon('Skarmory', ['spikes', 'splash'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 } }),
     mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Snorlax', ['splash'], { ability: 'Immunity', evs: { hp: 252 } })],
    [
      // p1 side must reach 3 spikes; p2 side 1 spike.
      { p1: 'move 2', p2: 'move 1' }, // p1 splash; p2 spikes→p1(1)
      { p1: 'move 2', p2: 'move 1' }, // p1 splash; p2 spikes→p1(2)
      { p1: 'move 2', p2: 'move 1' }, // p1 splash; p2 spikes→p1(3)
      // p1 lay 1 spike on p2.
      { p1: 'move 1', p2: 'move 2' }, // p1 spikes→p2(1); p2 splash
      // Bring Electrodes in (grounded → p1 Electrode takes maxhp/4, p2 Electrode maxhp/8).
      { p1: 'switch 2', p2: 'switch 2' },
      // Mutual Explosion → double faint → double replacement.
      { p1: 'move 1', p2: 'move 1' },
      // p1 forced: Diglett (lvl1) → spikes-KO on entry → p1 replaces again (Sandshrew).
      // p2 forced: Snorlax → chipped ONCE (maxhp/8).
      { p1: 'switch 3', p2: 'switch 3', force: true },
      // p1 cascade replacement (Sandshrew) — also a lvl1 grounded → may spikes-KO again.
      { p1: 'switch 4', force: true },
      { stop: true },
    ]);

  // (C) MUTUAL Explosion double-faint with ONE immune entrant (Flying/Levitate).
  //     p1 spikes=2, p2 spikes=2. p1 entrant = Salamence (Flying → ZERO); p2 entrant =
  //     Claydol (Levitate → ZERO). Pins the grounded gate under the cascade — NEITHER
  //     entrant is chipped.
  await run('(C) double-faint, both entrants Spikes-IMMUNE (Flying/Levitate) → ZERO chip',
    [mon('Skarmory', ['spikes', 'splash'], { ability: 'Keen Eye', evs: { hp: 252, def: 252 } }),
     mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Salamence', ['dragonclaw'], { ability: 'Intimidate', evs: { hp: 252 } })], // Flying
    [mon('Forretress', ['spikes', 'splash'], { ability: 'No Ability', evs: { hp: 252, def: 252 } }),
     mon('Electrode', ['explosion', 'splash'], { evs: { atk: 252, spe: 252 } }),
     mon('Claydol', ['psychic'], { ability: 'Levitate', evs: { hp: 252 } })], // Levitate
    [
      { p1: 'move 1', p2: 'move 1' }, // p1 spikes→p2(1); p2 spikes→p1(1)
      { p1: 'move 1', p2: 'move 1' }, // p1 spikes→p2(2); p2 spikes→p1(2)
      { p1: 'switch 2', p2: 'switch 2' }, // Electrodes in (grounded → both chipped maxhp/6)
      { p1: 'move 1', p2: 'move 1' }, // mutual Explosion → double faint
      { p1: 'switch 3', p2: 'switch 3', force: true }, // Salamence(Flying)/Claydol(Levitate) → ZERO
      { stop: true },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
