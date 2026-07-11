// probe_statusimmune_regression_rng.js — GROUND-TRUTH seeds + post-turn STATUS for the
// `tests/regression_test.rs` STATUS_IMMUNE pins (`gen3_status_immune_v1`). Constructed
// `gen3customgame` scenarios with EXPLICIT seeds; prints the ALIGNED init seed + each
// decision's `seedAfter` + the target's status so the Rust pins copy them verbatim.
// THE PROBE IS THE ONLY ORACLE.
//
//   SI1 limber_blocks_par_draw_free          — Thunder Wave into a Limber Snorlax: STAYS
//        unparalyzed (onSetStatus block, DRAW-FREE in customgame — the ability is the only
//        SetStatus handler → no clause shuffle). SEED == a status-lands-elsewhere baseline; the
//        real teeth: reverting the block PARALYZES it → its speed drops → later turns' order +
//        para roll diverge → the multi-turn seed diverges.
//   SI2 insomnia_blocks_slp_draw_free        — Spore into an Insomnia Snorlax: STAYS awake. The
//        block is DRAW-FREE, but a LANDED sleep draws the `random(2,6)` duration → reverting the
//        block desyncs the SAME turn's seed (a draw-count pin). STATE + SEED.
//   SI3 magma_armor_blocks_frz_state         — Ice Beam (frz secondary FIRES on the chosen seed)
//        into a Magma Armor Snorlax: NEVER frozen (immunity-phase block BEFORE the SetStatus
//        event). The freeze secondary's `random(100)` draws either way (draw-free block), so this
//        is a STATE pin (reverting the block freezes it; the seed is unchanged).
//   SI4 immunity_blocks_tox_not_brn          — Toxic into an Immunity Snorlax is BLOCKED (stays
//        clean); Will-O-Wisp into the SAME Immunity Snorlax BURNS (the block is status-SPECIFIC).
//        STATE (two arms) + SEED.
//
// Run:  node src/rust_sim/harness/probe_statusimmune_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { Teams } = require(path.join(PS, 'dist/sim'));
const FORMAT = 'gen3customgame';

function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(p1packed, p2packed, seed, plan, inject = []) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"${FORMAT}","seed":${JSON.stringify(seed)}}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1packed })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2packed })}`);
  for (let i = 0; i < 12; i++) await tick();
  const battle = stream.battle;
  // The ALIGNED init seed = the sim's PRNG state right before the FIRST decision (post-switch-in).
  // The Rust `start_with_switchins` is DRAW-FREE at start, so the pins seed with THIS value.
  const initSeed = battle.prng.getSeed();
  for (const inj of inject) {
    const m = battle.sides[inj.side].active[0];
    if (!m) continue;
    if (inj.status) m.setStatus(inj.status, m, null, true);
    if (inj.hp !== undefined) m.hp = inj.hp;
  }
  const seeds = [];
  seeds.initSeed = initSeed;
  for (const step of plan) {
    if (step.p1) { try { streams.omniscient.write(`>p1 ${step.p1}`); } catch (e) {} }
    if (step.p2) { try { streams.omniscient.write(`>p2 ${step.p2}`); } catch (e) {} }
    for (let k = 0; k < 14; k++) await tick();
    seeds.push({ note: step.note || '', seed: battle.prng.getSeed(), status: battle.sides[0].active[0] ? (battle.sides[0].active[0].status || 'ok') : '-' });
  }
  return { battle, seeds, log };
}

// Snorlax holder (Normal — no status type-immunity) with the ability under test. A frail-ish foe
// that RE-FIRES a status move. The holder Body Slams (a Normal move; the foe is Normal so it
// lands). Body Slam's OWN par secondary could paralyze the foe — irrelevant to the holder's block.
function holder(ability) {
  return `Snorlax|||${abilityId(ability)}|bodyslam,seismictoss|Careful|252,,128,,128,|||||`;
}
function abilityId(a) { return a.toLowerCase().replace(/[^a-z0-9]/g, ''); }
function foe(statusMove) {
  return `Blissey|||serenegrace|${statusMove},seismictoss|Bold|4,,252,,252,|||||`;
}

(async () => {
  // ── SI1 — Limber blocks par. Multi-turn so a REVERTED block (paralysis) desyncs the seed via
  //   the changed action order / para roll on later turns. Thunder Wave (slot 0) each turn.
  const si1 = await run(holder('Limber'), foe('thunderwave'), [7, 11, 13, 17], [
    { p1: 'move 1', p2: 'move 1', note: 'TWave into Limber (blocked)' },
    { p1: 'move 1', p2: 'move 1', note: 'holder still full-speed (unparalyzed)' },
    { p1: 'move 1', p2: 'move 1', note: 'TWave again (still blocked)' },
  ]);
  console.log('=== SI1 limber_blocks_par_draw_free ===');
  console.log('  initSeed:', si1.seeds.initSeed);
  console.log('  seeds:   ', si1.seeds.map((s) => `${s.seed} [p1status=${s.status}]`).join('  |  '));

  // ── SI2 — Insomnia blocks slp. A LANDED sleep would draw the random(2,6) duration → reverting
  //   the block desyncs THIS turn's seed. Spore (slot 0).
  const si2 = await run(holder('Insomnia'), foe('spore'), [3, 5, 7, 11], [
    { p1: 'move 1', p2: 'move 1', note: 'Spore into Insomnia (blocked, NO random(2,6))' },
    { p1: 'move 1', p2: 'move 1', note: 'holder acts normally (awake)' },
  ]);
  console.log('=== SI2 insomnia_blocks_slp_draw_free ===');
  console.log('  initSeed:', si2.seeds.initSeed);
  console.log('  seeds:   ', si2.seeds.map((s) => `${s.seed} [p1status=${s.status}]`).join('  |  '));

  // ── SI3 — Magma Armor blocks frz. Sweep an Ice-Beam foe until the FREEZE SECONDARY FIRES on a
  //   Snorlax control (so we know the seed CAN freeze); then use that seed with Magma Armor (which
  //   blocks it). The block is draw-free (the secondary random(100) drew either way). STATE pin.
  const iceFoe = `Regice|||clearbody|icebeam,seismictoss|Modest|4,252,,252,,|||||`;
  let si3 = null, si3seed = null, ctrlFroze = false;
  for (let s = 0; s < 400; s++) {
    const seed = [s + 1, 2, 3, 4];
    const ctrl = await run(holder('No Ability'), iceFoe, seed, [{ p1: 'move 2', p2: 'move 1', note: 'IceBeam into a NO-ability Snorlax' }]);
    if (ctrl.seeds[0].status === 'frz') {
      const ma = await run(holder('Magma Armor'), iceFoe, seed, [{ p1: 'move 2', p2: 'move 1', note: 'IceBeam into a Magma Armor Snorlax (frz blocked)' }]);
      si3 = { ctrl, ma, seed };
      si3seed = seed;
      ctrlFroze = true;
      break;
    }
  }
  console.log('=== SI3 magma_armor_blocks_frz_state ===');
  if (ctrlFroze) {
    console.log('  seed (raw >start):', JSON.stringify(si3seed), '| aligned initSeed:', si3.ma.seeds.initSeed);
    console.log('  control(No Ability) status:', si3.ctrl.seeds[0].status, '(FROZE — the seed can freeze)');
    console.log('  MagmaArmor status:        ', si3.ma.seeds[0].status, '(BLOCKED — never frozen)');
    console.log('  MA seedAfter:  ', si3.ma.seeds[0].seed, '| control seedAfter:', si3.ctrl.seeds[0].seed);
    console.log('  draw-free (MA seed == control seed):', si3.ma.seeds[0].seed === si3.ctrl.seeds[0].seed);
  } else {
    console.log('  NO freezing seed found in 400 tries — widen.');
  }

  // ── SI4 — Immunity blocks tox but NOT brn (status-specific). Two runs, SAME Immunity Snorlax.
  const si4tox = await run(holder('Immunity'), foe('toxic'), [21, 22, 23, 24], [
    { p1: 'move 1', p2: 'move 1', note: 'Toxic into Immunity (BLOCKED)' },
  ]);
  const si4brn = await run(holder('Immunity'), foe('willowisp'), [21, 22, 23, 24], [
    { p1: 'move 1', p2: 'move 1', note: 'Will-O-Wisp into Immunity (BURNS — not blocked)' },
  ]);
  console.log('=== SI4 immunity_blocks_tox_not_brn ===');
  console.log('  TOX initSeed:', si4tox.seeds.initSeed, '| BRN initSeed:', si4brn.seeds.initSeed);
  console.log('  TOX arm: status=', si4tox.seeds[0].status, 'seedAfter=', si4tox.seeds[0].seed);
  console.log('  BRN arm: status=', si4brn.seeds[0].status, 'seedAfter=', si4brn.seeds[0].seed);
})();
