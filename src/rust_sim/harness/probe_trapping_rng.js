// probe_trapping_rng.js — instrument the gen3 TRAPPING (Arena Trap / Magnet Pull)
// semantics + draw model bit-for-bit, against the OMNISCIENT in-process BattleStream
// (no server). The sim is the ONLY oracle — source reads are hypotheses.
//
// SETTLES:
//   1. ARENA TRAP: which foes are trapped? (hypothesis: grounded — Flying-type / Levitate
//      escape). Is a grounded GHOST trapped? (the BASE typechart has Ghost `trapped: 3`
//      [immune] and NO gen3/4/5 mod removes it — so SHOWDOWN-gen3 Ghosts may be trap-immune
//      even though the cartridge immunity is gen6+; the probe decides). The DUGTRIO MIRROR:
//      do both actives trap each other (each side's request `trapped: true`)?
//   2. MAGNET PULL: Steel-type foes only? Skarmory (Steel/Flying — NOT grounded) still
//      trapped? The MAGNETON MIRROR (Electric/Steel both sides): mutual trap?
//   3. THE DRAW MODEL (the bit-for-bit crux): the trapped flags are computed at endTurn
//      (battle.ts ~1723: per active mon, `runEvent('TrapPokemon')` then, gated on
//      `!knownType || getImmunity('trapped')`, `runEvent('MaybeTrapPokemon')`) — BEFORE the
//      gen3 quickClawRoll draw (battle.ts:1795). gen3 magnetpull is overridden to
//      `onAnyTrapPokemon` (data/mods/gen3/abilities.ts) so BOTH actives' magnetpulls
//      register on EVERY TrapPokemon event → in a speed-TIED Magneton mirror the 2 handlers
//      tie on (order, priority, speed, subOrder, effectOrder) → `speedSort` Fisher-Yates
//      draws?! Arena Trap stays `onFoe` (1 handler per event → no tie possible). Count raw
//      draws per decision for the magnetpull mirror vs a Sturdy-mirror CONTROL (same species
//      → identical eachEvent shuffle baseline) to isolate the trap-event draws.
//   4. THE REJECTION FLOW: a trapped mon's request carries `trapped: true`; a voluntary
//      `switch N` is rejected by `side.choose` ("Can't switch: The active Pokémon is
//      trapped") DRAW-FREE (seed unchanged, request stays open). Moves still work.
//   5. PHAZE BYPASS: Roar still DRAGS a trapped mon out (forceSwitch/dragIn never consults
//      `trapped` — trapping blocks VOLUNTARY switching only).
//   6. FORCED REPLACEMENT: a fainted mon's replacement `switch N` is accepted even while
//      the foe traps (side.ts: the trapped check is `requestState === 'move'`-only). And
//      the TRAPPING mon itself can switch freely.
//
// ===================================================================================
// CONFIRMED FINDINGS (vs the omniscient sim — run this file to regenerate):
//
//   (filled in after running — see the run log below in the repo docs; the findings are
//    transcribed into CLAUDE.md "## Trapping".)
// ===================================================================================
//
// Run:  node src/rust_sim/harness/probe_trapping_rng.js
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

function trapInfo(battle, side) {
  const a = battle.sides[side].active[0];
  const req = battle.sides[side].activeRequest;
  const reqTrap = req && req.active && req.active[0] ? `${req.active[0].trapped ? 'trapped' : ''}${req.active[0].maybeTrapped ? ' maybe' : ''}` : '?';
  return `trapped=${JSON.stringify(a ? a.trapped : '-')} maybeTrapped=${a ? !!a.maybeTrapped : '-'} req=[${reqTrap.trim() || 'free'}]`;
}

async function run(label, p1team, p2team, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  const seed = [7, 11, 13, 17];
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: Teams.pack(p1team) })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: Teams.pack(p2team) })}`);
  for (let i = 0; i < 12; i++) await tick();

  const battle = stream.battle;
  let drawCount = 0;
  const rng = battle.prng.rng;
  const realNext = rng.next.bind(rng);
  rng.next = function (...a) { drawCount++; return realNext(...a); };

  console.log(`\n=== ${label} ===  initSeed=${battle.prng.getSeed()}`);

  let i = 0, safety = 0;
  let logIdx = log.length;
  while (!battle.ended && safety < 60) {
    safety++;
    const rs = battle.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    if (i >= plan.length) break;
    const before = battle.prng.getSeed();
    const dc0 = drawCount;
    const t0 = `p1 ${trapInfo(battle, 0)} | p2 ${trapInfo(battle, 1)}`;
    const entry = plan[i]; i++;
    logIdx = log.length;
    try { if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`); } catch (e) { console.log('  p1 err', e.message); }
    try { if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`); } catch (e) { console.log('  p2 err', e.message); }
    for (let k = 0; k < 18; k++) await tick();
    const after = battle.prng.getSeed();
    const errs = log.slice(logIdx).filter((l) => l.startsWith('|error|'));
    const a0 = battle.sides[0].active[0], a1 = battle.sides[1].active[0];
    const fmt = (m) => m ? `${m.species.name} ${m.hp}/${m.maxhp}${m.fainted ? ' FNT' : ''}` : '-';
    console.log(`  [${rs}] ${JSON.stringify(entry)} draws=${drawCount - dc0} seedEq=${String(before) === String(after)}`);
    console.log(`        pre : ${t0}`);
    console.log(`        post: p1=${fmt(a0)} p2=${fmt(a1)} | p1 ${trapInfo(battle, 0)} | p2 ${trapInfo(battle, 1)}`);
    console.log(`        before=${before}`);
    console.log(`        after =${after}${errs.length ? `  ERRORS=${JSON.stringify(errs)}` : ''}`);
    if (entry.stop) break;
  }
  console.log(`  ended=${battle.ended} winner=${battle.winner} finalSeed=${battle.prng.getSeed()}`);
  try { streams.omniscient.destroy(); } catch (e) {}
}

async function main() {
  const d = Dex.forFormat(FORMAT);
  console.log('=== resolved gen3 ability facts ===');
  for (const id of ['arenatrap', 'magnetpull']) {
    const ab = d.abilities.get(id);
    console.log(`  ${id}: onFoeTrapPokemon=${!!ab.onFoeTrapPokemon} onAnyTrapPokemon=${!!ab.onAnyTrapPokemon} ` +
      `onFoeMaybeTrapPokemon=${!!ab.onFoeMaybeTrapPokemon} onAnyMaybeTrapPokemon=${!!ab.onAnyMaybeTrapPokemon}`);
  }
  console.log(`  typechart Ghost.damageTaken.trapped=${d.types.get('Ghost').damageTaken['trapped']}` +
    ` (3 = immune to trapping in THIS dex)`);

  // (1) ARENA TRAP vs a GROUNDED foe: p2 Snorlax trapped from turn 1 (leads' trapped
  //     computed at the pre-turn-1 endTurn). p2 tries `switch 2` → expect |error| +
  //     seed UNCHANGED + 0 draws (rejection is draw-free, request stays open); then p2
  //     must fight. p1's own switch is FREE (the trapper is not trapped).
  await run('AT-1 Arena Trap traps a grounded foe; rejection is draw-free; trapper switches freely',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [mon('Snorlax', ['bodyslam', 'splash'], { evs: { hp: 252 } }),
     mon('Regice', ['icebeam', 'splash'])],
    [
      { p2: 'switch 2' },                      // REJECTED (trapped) — draw-free, boundary open
      { p1: 'move 2', p2: 'move 2' },          // both splash — battle advances normally
      { p1: 'switch 2', p2: 'move 2' },        // the TRAPPER switches out freely; p2 now free?
      { p1: 'move 2', p2: 'switch 2' },        // p2 switch should now be ACCEPTED (no trapper)
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // (2) ARENA TRAP vs FLYING + vs LEVITATE: both switch FREELY.
  await run('AT-2 Flying (Zapdos) + Levitate (Gengar) escape Arena Trap',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
    [mon('Zapdos', ['thunderbolt', 'splash'], { ability: 'Pressure' }),
     mon('Gengar', ['thunderbolt', 'splash'], { ability: 'Levitate' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [
      { p1: 'move 2', p2: 'switch 2' },        // Zapdos free → Gengar in
      { p1: 'move 2', p2: 'switch 3' },        // Gengar (Levitate) free → Snorlax in
      { p2: 'switch 2' },                      // Snorlax grounded → REJECTED
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // (3) ARENA TRAP vs a GROUNDED GHOST (Sableye, Ghost/Dark, Keen Eye): the BASE typechart
  //     marks Ghost trap-immune (`trapped: 3`) with no gen3 mod — does SHOWDOWN-gen3 trap it?
  await run('AT-3 grounded GHOST (Sableye) vs Arena Trap — Showdown-gen3 trap immunity?',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' })],
    [mon('Sableye', ['shadowball', 'splash'], { ability: 'Keen Eye' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [
      { p1: 'move 2', p2: 'switch 2' },        // ACCEPTED (Ghost trap-immune) or REJECTED?
      { p1: 'move 2', p2: 'move 2' },
    ]);

  // (4) THE DUGTRIO MIRROR: both actives Arena Trap → MUTUAL trap? Both requests trapped?
  //     Draw counts on splash turns (arenatrap is onFoe → 1 handler/event → expect NO extra
  //     draws vs the Sand Veil control below, even at a full speed tie).
  await run('AT-4 DUGTRIO MIRROR — mutual trap? extra endTurn draws?',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap' }),
     mon('Regice', ['icebeam', 'splash'])],
    [
      { p1: 'move 2', p2: 'move 2' },
      { p2: 'switch 2' },                      // REJECTED (mutually trapped)
      { p1: 'switch 2' },                      // REJECTED too (mutual)
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);
  await run('AT-4c CONTROL: Sand Veil Dugtrio mirror (same speeds, no trap) — baseline draws',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Sand Veil' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Sand Veil' }),
     mon('Regice', ['icebeam', 'splash'])],
    [
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);

  // (5) MAGNET PULL vs STEEL (Skarmory — Steel/FLYING: groundedness must be IRRELEVANT)
  //     and vs non-Steel (Snorlax free).
  await run('MP-1 Magnet Pull traps Steel (even Flying Skarmory); non-Steel free',
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' })],
    [mon('Skarmory', ['drillpeck', 'splash'], { ability: 'Keen Eye' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [
      { p2: 'switch 2' },                      // Skarmory Steel → REJECTED (despite Flying)
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 1', p2: 'move 2' },          // chip so the battle can end later if needed
      { stop: true },
    ]);
  await run('MP-1b Magnet Pull does NOT trap a non-Steel foe',
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' })],
    [mon('Snorlax', ['bodyslam', 'splash']),
     mon('Regice', ['icebeam', 'splash'])],
    [
      { p1: 'move 2', p2: 'switch 2' },        // ACCEPTED (Snorlax not Steel)
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);

  // (6) THE MAGNETON MIRROR (both Magnet Pull, IDENTICAL sets → speed TIE): mutual trap?
  //     AND the endTurn draw question — gen3 magnetpull is onANY, so each TrapPokemon event
  //     carries 2 handlers; a full tie → Fisher-Yates draw(s)?! Compare draws/turn vs the
  //     Sturdy-mirror CONTROL (identical species/sets → identical eachEvent baseline).
  await run('MP-2 MAGNETON MIRROR (Magnet Pull both) — mutual trap? EXTRA endTurn draws?',
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' }),
     mon('Regice', ['icebeam', 'splash'])],
    [
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p2: 'switch 2' },                      // REJECTED (mutual Steel trap)
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);
  await run('MP-2c CONTROL: Sturdy Magneton mirror — baseline draws/turn',
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Sturdy' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Sturdy' }),
     mon('Regice', ['icebeam', 'splash'])],
    [
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);
  // (6b) ONE-SIDED magnetpull vs a Steel foe with a DIFFERENT speed (Metagross): 1 handler
  //     per event → expect NO extra draws vs control; Metagross trapped.
  await run('MP-2d one-sided Magnet Pull vs Metagross (Clear Body, slower) — no extra draws',
    [mon('Magneton', ['thunderbolt', 'splash'], { ability: 'Magnet Pull' })],
    [mon('Metagross', ['meteormash', 'splash'], { ability: 'Clear Body' }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [
      { p1: 'move 2', p2: 'move 2' },
      { p2: 'switch 2' },                      // REJECTED (Metagross is Steel)
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);

  // (7) PHAZE BYPASSES TRAPPING: p1 Dugtrio (Arena Trap) traps Snorlax, then ROARS it out —
  //     the drag must fire (forceSwitch ignores `trapped`), with the usual acc + sample draws.
  await run('PH-1 Roar drags a TRAPPED mon out (phaze bypasses trapping)',
    [mon('Dugtrio', ['roar', 'splash'], { ability: 'Arena Trap' })],
    [mon('Snorlax', ['bodyslam', 'splash'], { nature: 'Brave', ivs: { ...IV31, spe: 0 } }),
     mon('Regice', ['icebeam', 'splash'])],
    [
      { p2: 'switch 2' },                      // REJECTED (trapped)
      { p1: 'move 1', p2: 'move 2' },          // Roar → Snorlax DRAGGED (Regice in)
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);

  // (8) FORCED REPLACEMENT is unaffected: Dugtrio KOs the trapped foe; the replacement
  //     `switch 2` at the forced-switch request is ACCEPTED (trapped gates 'move' only).
  await run('FR-1 a fainted trapped mon\'s forced replacement is ACCEPTED',
    [mon('Dugtrio', ['earthquake', 'splash'], { ability: 'Arena Trap', evs: { atk: 252 } })],
    [mon('Electrode', ['thunderbolt', 'splash'], { evs: EV0, ivs: { ...IV31, hp: 0, def: 0 }, level: 55 }),
     mon('Snorlax', ['bodyslam', 'splash'])],
    [
      { p1: 'move 1', p2: 'move 2' },          // EQ KOs Electrode (grounded, frail)
      { p2: 'switch 2' },                      // forced replacement — ACCEPTED
      { p1: 'move 2', p2: 'move 2' },          // Snorlax now trapped at its move request
      { p2: 'switch 2' },                      // and a VOLUNTARY retreat is REJECTED
      { p1: 'move 2', p2: 'move 2' },
      { stop: true },
    ]);
}
main().catch((e) => { console.error(e.stack || String(e)); process.exit(1); });
