// probe_trapping_regression_rng.js — GROUND-TRUTH seeds + exact per-decision state for
// the deterministic TRAPPING regression pins in tests/regression_test.rs. Constructs the
// SAME gen3customgame scenarios the pins use (fixed seed + scripted choices — including
// the deliberately-ILLEGAL trapped `switch 2` submissions the sim REJECTS draw-free) and
// prints the post-decision SEED + the per-side `pokemon.trapped` + HP/species the pins
// assert. Ground truth is COPIED VERBATIM from here into the pins.
//
// Pins covered:
//   - arena_trap_rejects_a_grounded_foes_switch_draw_free :
//       a grounded foe (Snorlax) vs Arena Trap: its voluntary `switch 2` is REJECTED
//       ("Can't switch: The active Pokémon is trapped") with the seed UNCHANGED and the
//       request boundary left OPEN (the reject-and-re-request pattern — the port SKIPS
//       the scripted Switch decision draw-free); the next splash/splash decision runs
//       normally. Arena Trap itself adds ZERO draws (onFoe → 1 handler per trap event).
//   - arena_trap_does_not_trap_flying_or_levitate :
//       Zapdos (Flying) and Gengar (Levitate) switch out FREELY vs Arena Trap — both
//       voluntary switches ACCEPTED (is_trapped false), gen-3 grounded == not-Flying &&
//       not-Levitate.
//   - magnet_pull_traps_steel_only :
//       (a) the MAGNETON MIRROR: mutual trap (Steel↔Steel) — a switch REJECTED draw-free
//       — and the endTurn TrapPokemon+MaybeTrapPokemon tie-shuffles: the speed-tied
//       mirror draws 4 PER endTurn (2 events × 2 mons; gen3 magnetpull is onAny), pinned
//       by the splash/splash boundary seeds (dropping the trap-event shuffles desyncs
//       them); (b) the non-Steel control: Snorlax switches out of a Magnet Pull FREELY.
//   - roar_drags_a_trapped_mon_out :
//       PHAZE BYPASSES TRAPPING — the same trapped Snorlax whose `switch 2` was just
//       rejected is Roar-DRAGGED out (accuracy + the n=1 `sample` draws), and the dragged
//       Regice (grounded) is trapped in turn.
//   - grounded_ghost_is_trapped_by_arena_trap_in_showdown_gen3 :
//       the probe-settled SURPRISE: Showdown-gen3 resolves NO `trapped` type-immunity
//       (Ghost damageTaken.trapped = undefined in the gen3 dex) → a grounded GHOST
//       (Sableye) IS trapped — its switch is rejected. (The cartridge gen6+ Ghost escape
//       does not exist here; the port must model the SIM.)
//
// Run:  node src/rust_sim/harness/probe_trapping_regression_rng.js
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

async function run(label, seed, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;

  console.log(`\n=== ${label} ===  seed=${JSON.stringify(seed)}`);
  console.log(`  initSeed=${battle.prng.getSeed()}  (p1 team: ${Teams.pack(p1team)})`);
  console.log(`                                     (p2 team: ${Teams.pack(p2team)})`);
  let i = 0, safety = 0;
  while (!battle.ended && i < plan.length && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const before = battle.prng.getSeed();
    const logLen0 = log.length;
    const entry = plan[i]; i++;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) {}
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) {}
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const errs = log.slice(logLen0).filter((l) => l.startsWith('|error|'));
    const dragged = log.slice(logLen0).some((l) => l.split('|')[1] === 'drag');
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''} trapped=${JSON.stringify(m.trapped)}` : '-';
    console.log(`  [entry ${i - 1}] ${JSON.stringify(entry)}  ${errs.length ? 'REJECTED (draw-free)' : 'ran'}${dragged ? ' +DRAG' : ''}`);
    console.log(`        seedBefore=${before}`);
    console.log(`        seedAfter =${after}  (unchanged=${String(before) === String(after)})`);
    console.log(`        p1: ${fmt(a0)} | p2: ${fmt(a1)}`);
    if (errs.length) console.log(`        ${errs[0]}`);
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  // ---------------------------------------------------------------------------
  // PIN arena_trap_rejects_a_grounded_foes_switch_draw_free
  //   entry0: p2 `switch 2` — REJECTED (trapped), seed UNCHANGED, boundary OPEN.
  //   entry1: splash/splash — the SAME boundary commits (the port skips the Switch
  //           decision and consumes the next ScriptDecision).
  //   entry2: splash/splash — one more boundary (seed rhythm; AT adds 0 draws).
  // ---------------------------------------------------------------------------
  await run('PIN T1: Arena Trap rejects a grounded foe\'s switch (draw-free)', [11, 22, 33, 44],
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252 } }),
     mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252 } })],
    [
      { p2: 'switch 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // ---------------------------------------------------------------------------
  // PIN arena_trap_does_not_trap_flying_or_levitate
  //   entry0: Zapdos (FLYING) switches out — ACCEPTED (not grounded).
  //   entry1: Gengar (LEVITATE) switches out — ACCEPTED.
  //   entry2: splash/splash.
  // ---------------------------------------------------------------------------
  await run('PIN T2: Flying (Zapdos) + Levitate (Gengar) switch freely vs Arena Trap', [5, 6, 7, 8],
    [mon('Dugtrio', ['rockslide', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Zapdos', ['drillpeck', 'splash'], { ability: 'Pressure', nature: 'Modest', evs: { hp: 252 } }),
     mon('Gengar', ['sludgebomb', 'splash'], { ability: 'Levitate', nature: 'Modest', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'switch 2' },
      { p1: 'move 2', p2: 'switch 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // ---------------------------------------------------------------------------
  // PIN magnet_pull_traps_steel_only — (a) the MAGNETON MIRROR (mutual trap + the
  // +4-per-endTurn tie-shuffle draws in the splash/splash boundary seeds).
  //   entry0: p2 `switch 2` — REJECTED (Steel trapped by Magnet Pull).
  //   entry1: splash/splash — commits (endTurn draws 4 trap-event shuffles + QC).
  //   entry2: splash/splash — again.
  // ---------------------------------------------------------------------------
  await run('PIN T3a: MAGNETON MIRROR — mutual Steel trap + the 4/endTurn tie-shuffle draws', [9, 8, 7, 6],
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull', nature: 'Modest', evs: { spa: 252 } }),
     mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Adamant', evs: { hp: 252 } })],
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull', nature: 'Modest', evs: { spa: 252 } }),
     mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252 } })],
    [
      { p2: 'switch 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // (b) the non-Steel CONTROL: Snorlax switches out of a Magnet Pull FREELY (accepted).
  await run('PIN T3b: Magnet Pull does NOT trap a non-Steel foe (free switch accepted)', [2, 4, 6, 8],
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull', nature: 'Modest', evs: { spa: 252 } })],
    [mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252 } }),
     mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252 } })],
    [
      { p1: 'move 2', p2: 'switch 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // ---------------------------------------------------------------------------
  // PIN roar_drags_a_trapped_mon_out — phaze BYPASSES trapping.
  //   entry0: p2 `switch 2` — REJECTED (trapped by Arena Trap).
  //   entry1: Roar / splash — the SAME Snorlax that could not leave is DRAGGED out
  //           (Roar acc(100) + the n=1 `sample`); Regice (grounded) enters TRAPPED.
  //   entry2: splash/splash.
  // ---------------------------------------------------------------------------
  await run('PIN T4: Roar drags the TRAPPED Snorlax out (phaze bypasses trapping)', [13, 14, 15, 16],
    [mon('Dugtrio', ['roar', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252 } }),
     mon('Regice', ['icebeam', 'splash'], { nature: 'Modest', evs: { hp: 252 } })],
    [
      { p2: 'switch 2' },
      { p1: 'move 1', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // ---------------------------------------------------------------------------
  // PIN grounded_ghost_is_trapped_by_arena_trap_in_showdown_gen3 — the Showdown-gen3
  // surprise (NO trapped type-immunity in the gen3 dex).
  //   entry0: Sableye (GHOST/Dark, grounded) `switch 2` — REJECTED (trapped!).
  //   entry1: splash/splash.
  // ---------------------------------------------------------------------------
  await run('PIN T5: a grounded GHOST (Sableye) IS trapped in Showdown-gen3', [21, 22, 23, 24],
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap', nature: 'Adamant', evs: { atk: 252, spe: 252 } })],
    [mon('Sableye', ['shadowball', 'splash'], { ability: 'Keen Eye', nature: 'Bold', evs: { hp: 252 } }),
     mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 }, evs: { hp: 252 } })],
    [
      { p2: 'switch 2' },
      { p1: 'move 2', p2: 'move 2' },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
