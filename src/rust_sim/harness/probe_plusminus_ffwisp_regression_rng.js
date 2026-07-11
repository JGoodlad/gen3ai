// probe_plusminus_ffwisp_regression_rng.js — GROUND TRUTH for the `gen3_plus_minus_v1` +
// `gen3_ff_wisp_absorb_v1` regression pins (tests/regression_test.rs::
// minus_boosts_spa_when_the_foe_active_has_plus / will_o_wisp_into_flash_fire_is_absorbed).
//
// THE BUGS (the A/B fuzzer's state@move cluster, auto_0709_0805 re-triage 2026-07-10):
//  1. PLUS/MINUS: the gen3 RESOLVED `onModifySpA` scans `getAllActive()` — FOES INCLUDED
//     (gen5+ narrowed it to allies) — so a Minus attacker facing a Plus active gets SpA
//     ×1.5. The old NOOP classification ("partner-less in singles → no-op") never tested
//     the OPPOSING active carrying the paired ability; the port priced Minun-vs-Plusle
//     thunderbolt flat (18 recurring repros, Δ ≈ damage/3). Settled by
//     harness/probe_plus_minus_gen3.js (90 vs 60 = ×1.5 both directions; same-ability
//     pairs NO boost; SpA-only; draw-free; live while the partner is active).
//  2. FF-WISP: Will-O-Wisp into a NON-Fire, status-free, un-subbed Flash Fire holder is
//     ABSORBED (the resolved `flashfire.onTryHit` — the volatile ARMS, NO burn); the port
//     burned it → a maxhp/8 DoT desync per residual (the willowisp state cluster, incl. a
//     TRACED Flash Fire on Porygon2). Settled by harness/probe_flashfire_rng.js A3.
//
// Each scenario drives the OMNISCIENT BattleStream over a CONSTRUCTED gen3customgame
// board whose EXACT packed teams + raw seed the Rust pin replays.
// Run:  node src/rust_sim/harness/probe_plusminus_ffwisp_regression_rng.js
'use strict';
const path = require('path');
const PS = path.resolve(__dirname, '../../../deps/pokemon-showdown');
const { BattleStream, getPlayerStreams } = require(path.join(PS, 'dist/sim/battle-stream'));
const { PRNG } = require(path.join(PS, 'dist/sim/prng'));
function tick() { return new Promise((r) => setTimeout(r, 0)); }

async function run(label, p1, p2, rawSeed, plan) {
  const stream = new BattleStream();
  const streams = getPlayerStreams(stream);
  const log = [];
  (async () => { for await (const ch of streams.omniscient) { for (const l of ch.split('\n')) if (l) log.push(l); } })();
  streams.omniscient.write(`>start {"formatid":"gen3customgame","seed":[1,2,3,4]}`);
  streams.omniscient.write(`>player p1 ${JSON.stringify({ name: 'P1', team: p1 })}`);
  streams.omniscient.write(`>player p2 ${JSON.stringify({ name: 'P2', team: p2 })}`);
  for (let i = 0; i < 12; i++) await tick();
  const b = stream.battle;
  b.prng = new PRNG(rawSeed.slice());
  console.log(`\n=== ${label} (raw seed ${rawSeed.join(',')}) ===`);
  let i = 0, safety = 0;
  while (!b.ended && safety < 60 && i < plan.length) {
    safety++;
    const rs = b.requestState;
    if (rs !== 'move' && rs !== 'switch') { await tick(); continue; }
    const entry = plan[i]; i++;
    if (entry.p1) streams.omniscient.write(`>p1 ${entry.p1}`);
    if (entry.p2) streams.omniscient.write(`>p2 ${entry.p2}`);
    for (let k = 0; k < 18; k++) await tick();
    const u = b.sides[0].active[0];
    const t = b.sides[1].active[0];
    console.log(`  dec ${i - 1} ${JSON.stringify(entry)} seedAfter=${b.prng.getSeed()}`);
    console.log(`    p1=${u.species.name} ${u.hp}/${u.maxhp} st=${u.status || '-'} ff=${!!u.volatiles['flashfire']}  p2=${t.species.name} ${t.hp}/${t.maxhp} st=${t.status || '-'} ff=${!!t.volatiles['flashfire']}`);
  }
  const wisp = log.filter((l) => /-immune|-start|curestatus|-status|Flash Fire/.test(l));
  if (wisp.length) console.log('  ff/status lines:', wisp.join(' | '));
}

// Packed sets (level 100, Hardy, 85-EV flat, explicit genders — no construction draws
// beyond the standard ones absorbed by the raw-seed reseed).
const MINUN = 'Minun||Leftovers|Minus|thunderbolt,splash|Hardy|85,85,85,85,85,85|M|||100|';
const PLUSLE = 'Plusle||Leftovers|Plus|splash,thunderbolt|Hardy|85,85,85,85,85,85|M|||100|';
const PLUSLE_STURDY = 'Plusle||Leftovers|Sturdy|splash,thunderbolt|Hardy|85,85,85,85,85,85|M|||100|';
const GENGAR = 'Gengar||Leftovers|Levitate|willowisp,splash|Hardy|85,85,85,85,85,85|M|||100|';
const SNORLAX_FF = 'Snorlax||Leftovers|FlashFire|splash,flamethrower|Hardy|85,85,85,85,85,85|M|||100|';

async function main() {
  // PM: Minun tbolt into Plusle (Plus) — boosted; then the SAME board with a Sturdy
  // Plusle — unboosted control (same raw seed → same rolls, the hp delta isolates ×1.5).
  await run('PM_boosted (Minus vs Plus foe)', MINUN, PLUSLE, [0, 0, 0, 21],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);
  await run('PM_control (Minus vs Sturdy foe)', MINUN, PLUSLE_STURDY, [0, 0, 0, 21],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 1' }]);

  // FFW: Gengar Will-O-Wisp into a NON-Fire Flash Fire Snorlax — absorbed (no burn, the
  // volatile arms), then Snorlax's own flamethrower next turn is ×1.5 (the armed boost).
  await run('FFW_absorb (WoW into non-Fire FF)', GENGAR, SNORLAX_FF, [0, 0, 0, 33],
    [{ p1: 'move 1', p2: 'move 1' }, { p1: 'move 2', p2: 'move 2' }]);
}
main();
